#!/usr/bin/env python3
"""Batch driver for extracting Wan2.2 VAE latents from multiple Cobot_Magic datasets."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import tempfile
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from queue import Empty
from types import SimpleNamespace

from tool.extract_cobot_wan_vae_latents import (
    load_jsonl,
    normalize_metadata_video_rel,
    resolve_under_root,
    run_extraction,
    write_jsonl_atomic,
)


# DEFAULT_DATASETS = [
#     "Cobot_Magic_classification_of_fruits_and_vegetables",
#     "Cobot_Magic_classification_of_fruits_and_vegetables_a",
#     "Cobot_Magic_clean_blackboard",
#     "Cobot_Magic_cut_banana",
#     "Cobot_Magic_desktop_organization",
#     "Cobot_Magic_food_packaging",
#     "Cobot_Magic_make_fruit_salad",
#     "Cobot_Magic_plate_storaje_baozi",
#     "Cobot_Magic_prepare_breakfast",
#     "Cobot_Magic_the_box_stores_table_tennis_balls",
#     "Cobot_Magic_the_plate_holds_the_fruit",
#     "Cobot_Magic_the_plate_holds_the_vegetables",
#     "Cobot_Magic_turn_off_the_desk_lamp",
#     "Cobot_Magic_vase_storage_flower",
# ]


DEFAULT_DATASETS = [
    "Cobot_Magic_clean_blackboard",
]


@dataclass(frozen=True)
class ExtractionTask:
    task_id: str
    dataset_name: str
    dataset_root: str
    chunk_name: str
    metadata_input: str
    metadata_output: str
    final_metadata_output: str
    row_indices: tuple[int, ...]
    episode_count: int


@dataclass(frozen=True)
class DatasetPlan:
    dataset_name: str
    dataset_root: str
    final_metadata_output: str
    total_rows: int
    task_ids: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-extract Wan2.2 VAE latents for multiple Cobot_Magic datasets."
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("/data/linzengrong/Datasets/Cobot_Magic_all"),
        help="Root directory that contains all Cobot_Magic dataset folders.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Dataset folder names to process.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("meta/episodes_clipped.jsonl"),
        help="Per-dataset input metadata path, relative to dataset root unless absolute.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("meta/episodes_latents.jsonl"),
        help="Per-dataset output metadata path, relative to dataset root unless absolute.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("videos_clipped"),
        help="Per-dataset source video root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("latents"),
        help="Per-dataset latent output root.",
    )
    parser.add_argument(
        "--chunk-name",
        type=str,
        default="all",
        help="Chunk filter to pass into the single-dataset extractor. Use `all` to process every chunk.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/linzengrong/Models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
        help="Local Wan2.2 VAE checkpoint path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Single-process computation device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="",
        help="Comma-separated device list for parallel execution, e.g. cuda:0,cuda:1,...",
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="VAE compute dtype.",
    )
    parser.add_argument(
        "--save-dtype",
        type=str,
        default="float16",
        choices=("float32", "float16", "bfloat16"),
        help="Latent save dtype.",
    )
    parser.add_argument(
        "--spatial-division-factor",
        type=int,
        default=32,
        help="Height/width divisibility used during preprocessing.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=288,
        help="Target fit-box height used during preprocessing.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Target fit-box width used during preprocessing.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1920 * 1080,
        help="Max pixels for ImageCropAndResize.",
    )
    parser.add_argument(
        "--resize-mode",
        type=str,
        default="fit",
        choices=("fit", "crop"),
        help="Resize mode for ImageCropAndResize.",
    )
    parser.add_argument(
        "--save-suffix",
        type=str,
        default=".pth",
        help="Latent output file suffix.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing latent files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N rows per dataset after chunk filtering; 0 means all.",
    )
    return parser.parse_args()


def parse_devices(devices_arg: str) -> list[str]:
    devices = [item.strip() for item in str(devices_arg).split(",") if item.strip()]
    if not devices:
        return []
    if len(set(devices)) != len(devices):
        raise ValueError(f"Duplicate devices are not allowed: {devices}")
    return devices


def extract_chunk_name(record: dict, *, input_root: PurePosixPath) -> str:
    video_rel = normalize_metadata_video_rel(record["video"], input_root=input_root)
    parts = video_rel.parts
    if len(parts) < 4:
        raise ValueError(f"Unexpected video path in metadata: {video_rel}")
    return parts[1]


def filter_rows_for_dataset(rows: list[dict], *, input_root: PurePosixPath, chunk_name: str, limit: int) -> list[dict]:
    normalized_chunk = str(chunk_name).lower()
    if normalized_chunk not in {"all", "*"}:
        rows = [row for row in rows if extract_chunk_name(row, input_root=input_root) == chunk_name]
    if limit > 0:
        rows = rows[:limit]
    return rows


def build_tasks_for_dataset(
    dataset_name: str,
    dataset_root: Path,
    args: argparse.Namespace,
    temp_root: Path,
) -> tuple[DatasetPlan, list[ExtractionTask]]:
    metadata_path = resolve_under_root(dataset_root, args.metadata_path).resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    input_root = PurePosixPath(args.input_root.as_posix())
    final_metadata_output = resolve_under_root(dataset_root, args.metadata_output).resolve()
    rows = load_jsonl(metadata_path)
    rows = filter_rows_for_dataset(rows, input_root=input_root, chunk_name=args.chunk_name, limit=args.limit)
    if not rows:
        raise ValueError(f"No metadata rows found for dataset {dataset_name} with chunk {args.chunk_name}.")

    grouped_rows: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, row in enumerate(rows):
        chunk = extract_chunk_name(row, input_root=input_root)
        grouped_rows[chunk].append((index, row))

    dataset_temp_root = temp_root / dataset_name
    dataset_temp_root.mkdir(parents=True, exist_ok=True)

    tasks: list[ExtractionTask] = []
    for chunk_name, indexed_rows in sorted(grouped_rows.items()):
        task_id = f"{dataset_name}:{chunk_name}"
        task_input = dataset_temp_root / f"{chunk_name}.input.jsonl"
        task_output = dataset_temp_root / f"{chunk_name}.output.jsonl"
        write_jsonl_atomic(task_input, (row for _, row in indexed_rows))
        tasks.append(
            ExtractionTask(
                task_id=task_id,
                dataset_name=dataset_name,
                dataset_root=str(dataset_root),
                chunk_name=chunk_name,
                metadata_input=str(task_input),
                metadata_output=str(task_output),
                final_metadata_output=str(final_metadata_output),
                row_indices=tuple(index for index, _ in indexed_rows),
                episode_count=len(indexed_rows),
            )
        )

    plan = DatasetPlan(
        dataset_name=dataset_name,
        dataset_root=str(dataset_root),
        final_metadata_output=str(final_metadata_output),
        total_rows=len(rows),
        task_ids=tuple(task.task_id for task in tasks),
    )
    return plan, tasks


def make_task_args(base_args: argparse.Namespace, task: ExtractionTask, *, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_root=Path(task.dataset_root),
        metadata_path=Path(task.metadata_input),
        metadata_output=Path(task.metadata_output),
        input_root=base_args.input_root,
        output_root=base_args.output_root,
        chunk_name="all",
        model_path=base_args.model_path,
        device=device,
        compute_dtype=base_args.compute_dtype,
        save_dtype=base_args.save_dtype,
        spatial_division_factor=base_args.spatial_division_factor,
        height=base_args.height,
        width=base_args.width,
        max_pixels=base_args.max_pixels,
        resize_mode=base_args.resize_mode,
        save_suffix=base_args.save_suffix,
        overwrite=base_args.overwrite,
        limit=0,
    )


def assign_tasks_to_devices(tasks: list[ExtractionTask], devices: list[str]) -> dict[str, list[ExtractionTask]]:
    assignments: dict[str, list[ExtractionTask]] = {device: [] for device in devices}
    device_loads = {device: 0 for device in devices}
    for task in sorted(tasks, key=lambda item: (-item.episode_count, item.task_id)):
        device = min(devices, key=lambda item: (device_loads[item], item))
        assignments[device].append(task)
        device_loads[device] += task.episode_count
    return assignments


def worker_run(
    device: str,
    tasks: list[ExtractionTask],
    base_args: argparse.Namespace,
    result_queue,
) -> None:
    try:
        for task in tasks:
            print(
                f"[worker {device}] start dataset={task.dataset_name} chunk={task.chunk_name} "
                f"episodes={task.episode_count}",
                flush=True,
            )
            summary = run_extraction(make_task_args(base_args, task, device=device))
            result_queue.put(
                {
                    "ok": True,
                    "task_id": task.task_id,
                    "device": device,
                    "summary": summary,
                }
            )
            print(
                f"[worker {device}] done dataset={task.dataset_name} chunk={task.chunk_name} "
                f"episodes={task.episode_count}",
                flush=True,
            )
    except Exception as exc:
        result_queue.put(
            {
                "ok": False,
                "task_id": task.task_id if "task" in locals() else "<startup>",
                "device": device,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def drain_result_queue(result_queue) -> list[dict]:
    items: list[dict] = []
    while True:
        try:
            items.append(result_queue.get_nowait())
        except Empty:
            break
    return items


def run_parallel(args: argparse.Namespace, devices: list[str]) -> None:
    datasets_root = args.datasets_root.resolve()
    print(f"Datasets root: {datasets_root}")
    print(f"Datasets to process: {len(args.datasets)}")
    print(f"Chunk selection: {args.chunk_name}")
    print(f"Resize target: {args.height}x{args.width} ({args.resize_mode})")
    print(f"Parallel devices: {', '.join(devices)}")

    all_tasks: list[ExtractionTask] = []
    dataset_plans: list[DatasetPlan] = []
    with tempfile.TemporaryDirectory(prefix="cobot_wan_vae_batch_") as tmp_dir_name:
        temp_root = Path(tmp_dir_name)
        for dataset_name in args.datasets:
            dataset_root = (datasets_root / dataset_name).resolve()
            if not dataset_root.is_dir():
                raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
            plan, tasks = build_tasks_for_dataset(dataset_name, dataset_root, args, temp_root)
            dataset_plans.append(plan)
            all_tasks.extend(tasks)

        assignments = assign_tasks_to_devices(all_tasks, devices)
        print()
        print("=== Task Assignment ===")
        for device in devices:
            device_tasks = assignments[device]
            task_desc = ", ".join(
                f"{task.dataset_name}/{task.chunk_name}({task.episode_count})" for task in device_tasks
            )
            total_episodes = sum(task.episode_count for task in device_tasks)
            print(f"{device}: tasks={len(device_tasks)} episodes={total_episodes} :: {task_desc}")

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        processes = []
        for device in devices:
            device_tasks = assignments[device]
            if not device_tasks:
                continue
            process = ctx.Process(
                target=worker_run,
                args=(device, device_tasks, args, result_queue),
                daemon=False,
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        results = drain_result_queue(result_queue)
        result_by_task = {item["task_id"]: item for item in results if item["ok"]}
        errors = [item for item in results if not item["ok"]]

        abnormal_exitcodes = [process.exitcode for process in processes if process.exitcode not in (0, None)]
        if abnormal_exitcodes:
            raise RuntimeError(f"Parallel workers exited abnormally: {abnormal_exitcodes}")
        if errors:
            error = errors[0]
            raise RuntimeError(
                f"Worker failed on {error['task_id']} @ {error['device']}: {error['error']}\n{error['traceback']}"
            )

        expected_task_ids = {task.task_id for task in all_tasks}
        missing_task_ids = expected_task_ids.difference(result_by_task)
        if missing_task_ids:
            raise RuntimeError(f"Missing worker results for tasks: {sorted(missing_task_ids)}")

        print()
        print("=== Batch Summary ===")
        total_episodes = 0
        total_view_files = 0
        for plan in dataset_plans:
            merged_rows: list[dict | None] = [None] * plan.total_rows
            dataset_view_files = 0
            for task_id in plan.task_ids:
                task = next(task for task in all_tasks if task.task_id == task_id)
                output_rows = load_jsonl(Path(task.metadata_output))
                if len(output_rows) != len(task.row_indices):
                    raise RuntimeError(
                        f"Metadata row count mismatch for {task.task_id}: "
                        f"expected {len(task.row_indices)}, got {len(output_rows)}"
                    )
                for row_index, row in zip(task.row_indices, output_rows):
                    merged_rows[row_index] = row
                dataset_view_files += int(result_by_task[task_id]["summary"]["processed_view_files"])

            if any(row is None for row in merged_rows):
                raise RuntimeError(f"Missing merged metadata rows for dataset {plan.dataset_name}")

            final_rows = [row for row in merged_rows if row is not None]
            final_metadata_output = Path(plan.final_metadata_output)
            write_jsonl_atomic(final_metadata_output, final_rows)
            total_episodes += len(final_rows)
            total_view_files += dataset_view_files
            print(
                f"{plan.dataset_name}: episodes={len(final_rows)} "
                f"view_files={dataset_view_files} "
                f"metadata={final_metadata_output}"
            )

        print(f"Total episodes: {total_episodes}")
        print(f"Total view files: {total_view_files}")


def run_serial(args: argparse.Namespace) -> None:
    datasets_root = args.datasets_root.resolve()

    print(f"Datasets root: {datasets_root}")
    print(f"Datasets to process: {len(args.datasets)}")
    print(f"Chunk selection: {args.chunk_name}")
    print(f"Resize target: {args.height}x{args.width} ({args.resize_mode})")

    summaries = []
    for dataset_name in args.datasets:
        dataset_root = (datasets_root / dataset_name).resolve()
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

        print()
        print(f"=== Dataset: {dataset_name} ===")
        dataset_args = SimpleNamespace(
            dataset_root=dataset_root,
            metadata_path=args.metadata_path,
            metadata_output=args.metadata_output,
            input_root=args.input_root,
            output_root=args.output_root,
            chunk_name=args.chunk_name,
            model_path=args.model_path,
            device=args.device,
            compute_dtype=args.compute_dtype,
            save_dtype=args.save_dtype,
            spatial_division_factor=args.spatial_division_factor,
            height=args.height,
            width=args.width,
            max_pixels=args.max_pixels,
            resize_mode=args.resize_mode,
            save_suffix=args.save_suffix,
            overwrite=args.overwrite,
            limit=args.limit,
        )
        summary = run_extraction(dataset_args)
        summaries.append((dataset_name, summary))

    print()
    print("=== Batch Summary ===")
    total_episodes = 0
    total_view_files = 0
    for dataset_name, summary in summaries:
        total_episodes += int(summary["processed_episodes"])
        total_view_files += int(summary["processed_view_files"])
        print(
            f"{dataset_name}: episodes={summary['processed_episodes']} "
            f"view_files={summary['processed_view_files']} "
            f"metadata={summary['metadata_output']}"
        )
    print(f"Total episodes: {total_episodes}")
    print(f"Total view files: {total_view_files}")


def main() -> None:
    args = parse_args()
    devices = parse_devices(args.devices)
    if devices:
        run_parallel(args, devices)
    else:
        run_serial(args)


if __name__ == "__main__":
    main()
