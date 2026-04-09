#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from queue import Empty
from pathlib import Path
from typing import Dict, List

import cv2
import torch
from tqdm import tqdm

from diffsynth.core import ModelConfig, UnifiedDataset
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.pipelines.wan_video_spec import WanModuleSpec


DEFAULT_VIEWS = [
    "observation.images.cam_high_rgb",
    "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_right_wrist_rgb",
]
DEFAULT_DATASETS_BASE = "/data/linzengrong/Datasets/Cobot_Magic_all"
DEFAULT_DATASET_NAMES = [
    "Cobot_Magic_classification_of_fruits_and_vegetables",
    "Cobot_Magic_classification_of_fruits_and_vegetables_a",
    "Cobot_Magic_clean_blackboard",
    "Cobot_Magic_cut_banana",
    "Cobot_Magic_desktop_organization",
    "Cobot_Magic_food_packaging",
    "Cobot_Magic_make_fruit_salad",
    "Cobot_Magic_make_hamburger",
    "Cobot_Magic_plate_storaje_baozi",
    "Cobot_Magic_prepare_breakfast",
    "Cobot_Magic_the_box_stores_table_tennis_balls",
    "Cobot_Magic_the_plate_holds_the_fruit",
    "Cobot_Magic_the_plate_holds_the_vegetables",
    "Cobot_Magic_turn_off_the_desk_lamp",
    "Cobot_Magic_vase_storage_flower",
]
DEFAULT_EXCLUDE_DATASETS = ["Cobot_Magic_cut_banana_lingbot"]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".wmv", ".mkv", ".flv", ".webm"}


def parse_csv_list(raw: str) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_episode_filter(raw: str) -> set[str] | None:
    names = parse_csv_list(raw)
    if not names:
        return None
    return {Path(name).stem for name in names}


def parse_gpu_list(raw: str) -> List[int]:
    gpu_tokens = parse_csv_list(raw)
    if not gpu_tokens:
        return []
    gpu_ids = []
    for token in gpu_tokens:
        try:
            gpu_ids.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid GPU id: {token!r}") from exc
    # Keep order and drop duplicates.
    seen = set()
    unique_ids = []
    for gpu_id in gpu_ids:
        if gpu_id in seen:
            continue
        unique_ids.append(gpu_id)
        seen.add(gpu_id)
    return unique_ids


def resolve_dataset_roots(args: argparse.Namespace) -> List[Path]:
    if args.dataset_root:
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Missing dataset root: {dataset_root}")
        return [dataset_root]

    datasets_base = Path(args.datasets_base).expanduser().resolve()
    if not datasets_base.is_dir():
        raise FileNotFoundError(f"Missing datasets base directory: {datasets_base}")

    dataset_names = parse_csv_list(args.dataset_names)
    if not dataset_names:
        dataset_names = list(DEFAULT_DATASET_NAMES)
    excluded = set(parse_csv_list(args.exclude_datasets))

    dataset_roots = []
    for dataset_name in dataset_names:
        if dataset_name in excluded:
            continue
        candidate = Path(dataset_name)
        if candidate.is_absolute() or "/" in dataset_name:
            dataset_root = candidate.expanduser().resolve()
        else:
            dataset_root = (datasets_base / dataset_name).resolve()
        if not dataset_root.is_dir():
            print(f"[warn] Skip missing dataset: {dataset_root}", file=sys.stderr, flush=True)
            continue
        dataset_roots.append(dataset_root)
    return dataset_roots


def resolve_torch_dtype(name: str) -> torch.dtype:
    if not hasattr(torch, name):
        raise ValueError(f"Unsupported torch dtype: {name}")
    value = getattr(torch, name)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return value


def list_videos_by_stem(view_dir: Path) -> Dict[str, Path]:
    videos = {}
    if not view_dir.is_dir():
        return videos
    for file_path in sorted(view_dir.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        videos[file_path.stem] = file_path
    return videos


def discover_tasks(
    dataset_root: Path,
    dataset_name: str,
    videos_subdir: str,
    output_subdir: str,
    chunk_glob: str,
    chunk_names: List[str],
    views: List[str],
    episode_filter: set[str] | None,
) -> List[dict]:
    videos_root = dataset_root / videos_subdir
    if not videos_root.is_dir():
        raise FileNotFoundError(f"Missing videos directory: {videos_root}")
    output_root = dataset_root / output_subdir

    if chunk_names:
        chunk_dirs = [videos_root / chunk_name for chunk_name in chunk_names]
    else:
        chunk_dirs = sorted(path for path in videos_root.glob(chunk_glob) if path.is_dir())

    tasks = []
    for chunk_dir in chunk_dirs:
        if not chunk_dir.is_dir():
            print(f"[warn] Skip missing chunk directory: {chunk_dir}", file=sys.stderr, flush=True)
            continue

        videos_by_view = {}
        missing_view_dir = False
        for view in views:
            view_dir = chunk_dir / view
            if not view_dir.is_dir():
                print(f"[warn] Skip chunk {chunk_dir.name}, missing view directory: {view_dir}", file=sys.stderr, flush=True)
                missing_view_dir = True
                break
            videos_by_view[view] = list_videos_by_stem(view_dir)

        if missing_view_dir:
            continue

        common_stems = None
        for view in views:
            stems = set(videos_by_view[view].keys())
            common_stems = stems if common_stems is None else (common_stems & stems)
        common_stems = set() if common_stems is None else common_stems
        if episode_filter is not None:
            common_stems &= episode_filter

        if len(common_stems) == 0:
            print(f"[warn] No common episodes in {chunk_dir}", file=sys.stderr, flush=True)
            continue

        for episode_stem in sorted(common_stems):
            input_paths = {view: str(videos_by_view[view][episode_stem]) for view in views}
            output_paths = {
                view: str(output_root / chunk_dir.name / view / f"{episode_stem}.pt")
                for view in views
            }
            tasks.append(
                {
                    "dataset_name": dataset_name,
                    "chunk": chunk_dir.name,
                    "episode_stem": episode_stem,
                    "input_paths": input_paths,
                    "output_paths": output_paths,
                }
            )
    return tasks


def count_video_frames(path: str) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for frame counting: {path}")
    try:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames > 0:
            return num_frames

        # Fallback for broken container metadata.
        frame_count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            frame_count += 1
        if frame_count <= 0:
            raise ValueError(f"Invalid frame count for video: {path}")
        return frame_count
    finally:
        cap.release()


def atomic_torch_save(tensor: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(tensor, temp_path)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def build_vae_pipeline(model_dir: str, device: str, torch_dtype: torch.dtype) -> WanVideoPipeline:
    runtime = WanModuleSpec.parse("vae").build_runtime(model_dir, ["video"])
    if len(runtime.model_paths) == 0:
        raise FileNotFoundError(f"No VAE checkpoint found in model directory: {model_dir}")
    vae_config = ModelConfig(path=runtime.model_paths[0])
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[vae_config],
        tokenizer_config=None,
        modules=["vae"],
    )
    pipe.load_models_to_device(["vae"])
    pipe.vae.eval()
    return pipe


def build_video_operator(args: argparse.Namespace):
    return UnifiedDataset.default_video_operator(
        base_path="",
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        height_division_factor=args.spatial_division_factor,
        width_division_factor=args.spatial_division_factor,
        num_frames=args.dataset_num_frames,
        time_division_factor=4,
        time_division_remainder=1,
        resize_mode=args.resize_mode,
    )


def encode_one_episode(
    task: dict,
    views: List[str],
    video_operator,
    pipe: WanVideoPipeline,
    device: str,
    model_dtype: torch.dtype,
    frame_stride: int,
) -> tuple[torch.Tensor, torch.dtype]:
    video_paths = [task["input_paths"][view] for view in views]
    frame_counts = [count_video_frames(path) for path in video_paths]
    common_frame_count = min(frame_counts)
    if common_frame_count <= 0:
        raise ValueError(f"Invalid common frame count: {common_frame_count}")

    frame_indices = list(range(0, common_frame_count, frame_stride))
    if len(frame_indices) == 0:
        raise ValueError(f"Empty sampled frame indices for task: {task}")

    payload = [
        {
            "data": path,
            "start_frame": 0,
            "end_frame": common_frame_count - 1,
            "frame_indices": frame_indices,
        }
        for path in video_paths
    ]
    video_tensor = video_operator(payload)
    if not isinstance(video_tensor, torch.Tensor) or video_tensor.ndim != 5:
        raise TypeError(f"Expected video tensor with shape (V,C,T,H,W), got {type(video_tensor)}")
    # Keep input dtype aligned with model weights (e.g. bfloat16 on CUDA) to avoid conv dtype mismatch.
    video_tensor = video_tensor.to(dtype=model_dtype).contiguous()
    input_dtype = video_tensor.dtype

    with torch.inference_mode():
        latents = pipe.vae.encode(
            video_tensor,
            device=device,
            tiled=False,
        )
    if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
        raise TypeError(f"Expected latent tensor with shape (V,C,T,H,W), got {type(latents)}")
    return latents, input_dtype


def worker_main(
    worker_id: int,
    device: str,
    tasks: List[dict],
    args_dict: dict,
    views: List[str],
    progress_queue,
    result_queue,
) -> None:
    args = argparse.Namespace(**args_dict)
    stats = {
        "worker_id": worker_id,
        "device": device,
        "total": len(tasks),
        "processed": 0,
        "saved": 0,
        "skipped_existing": 0,
        "failed": 0,
        "status": "ok",
        "error": "",
    }

    try:
        cpu_threads = max(1, int(args.cpu_threads_per_worker))
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(cpu_threads, 4)))
        except RuntimeError:
            # set_num_interop_threads may only be called once per process.
            pass

        torch_dtype = resolve_torch_dtype(args.torch_dtype)
        save_dtype = resolve_torch_dtype(args.save_dtype)
        if device.startswith("cuda"):
            gpu_id = int(device.split(":")[1])
            torch.cuda.set_device(gpu_id)

        video_operator = build_video_operator(args)
        pipe = build_vae_pipeline(model_dir=args.model_dir, device=device, torch_dtype=torch_dtype)
        vae_param_dtype = next(pipe.vae.parameters()).dtype
        dtype_logged = False

        for task_id, task in enumerate(tasks, start=1):
            output_paths = [Path(task["output_paths"][view]) for view in views]
            if not args.overwrite and all(path.exists() for path in output_paths):
                stats["skipped_existing"] += 1
                progress_queue.put(
                    {
                        "status": "skipped",
                        "worker_id": worker_id,
                        "dataset_name": task.get("dataset_name", ""),
                    }
                )
                continue

            try:
                latents, input_dtype = encode_one_episode(
                    task=task,
                    views=views,
                    video_operator=video_operator,
                    pipe=pipe,
                    device=device,
                    model_dtype=torch_dtype,
                    frame_stride=args.frame_stride,
                )
                if not dtype_logged:
                    print(
                        f"[worker-{worker_id} {device}] dtype check: "
                        f"input={input_dtype}, vae={vae_param_dtype}, save={save_dtype}",
                        flush=True,
                    )
                    dtype_logged = True
                latents = latents.to(dtype=save_dtype)

                for view_id, view_name in enumerate(views):
                    output_path = Path(task["output_paths"][view_name])
                    if not args.overwrite and output_path.exists():
                        continue
                    atomic_torch_save(latents[view_id].contiguous().cpu(), output_path)
                    stats["saved"] += 1

                stats["processed"] += 1
                progress_queue.put(
                    {
                        "status": "processed",
                        "worker_id": worker_id,
                        "dataset_name": task.get("dataset_name", ""),
                    }
                )

            except Exception as exc:  # pragma: no cover - defensive logging
                stats["failed"] += 1
                progress_queue.put(
                    {
                        "status": "failed",
                        "worker_id": worker_id,
                        "dataset_name": task.get("dataset_name", ""),
                    }
                )
                print(
                    f"[worker-{worker_id} {device}] failed "
                    f"{task.get('dataset_name', 'dataset')}/{task['chunk']}/{task['episode_stem']}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.print_traceback:
                    print(traceback.format_exc(), file=sys.stderr, flush=True)

            if device.startswith("cuda") and args.empty_cache_every > 0 and (task_id % args.empty_cache_every == 0):
                torch.cuda.empty_cache()

    except Exception as exc:  # pragma: no cover - worker hard-fail
        stats["status"] = "crashed"
        stats["error"] = f"{type(exc).__name__}: {exc}"
        if args.print_traceback:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
    finally:
        result_queue.put(stats)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract multi-view latents with Wan2.2 VAE.")
    parser.add_argument(
        "--datasets_base",
        type=str,
        default=DEFAULT_DATASETS_BASE,
        help="Base directory containing multiple Cobot_Magic_* datasets.",
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        default=",".join(DEFAULT_DATASET_NAMES),
        help="Comma-separated dataset names under datasets_base.",
    )
    parser.add_argument(
        "--exclude_datasets",
        type=str,
        default=",".join(DEFAULT_EXCLUDE_DATASETS),
        help="Comma-separated dataset names to exclude.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="",
        help="Single dataset root override. If set, datasets_base/dataset_names are ignored.",
    )
    parser.add_argument(
        "--videos_subdir",
        type=str,
        default="videos_clipped",
        help="Subdirectory under dataset_root containing chunk-* video folders.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="latents",
        help="Subdirectory under dataset_root for latent outputs.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/data/linzengrong/Models/Wan2.2-TI2V-5B",
        help="WAN model root directory containing Wan2.2_VAE weights.",
    )
    parser.add_argument(
        "--views",
        type=str,
        default=",".join(DEFAULT_VIEWS),
        help="Comma-separated camera view keys.",
    )
    parser.add_argument("--height", type=int, default=192, help="Target frame height.")
    parser.add_argument("--width", type=int, default=320, help="Target frame width.")
    parser.add_argument(
        "--spatial_division_factor",
        type=int,
        choices=[16, 32],
        default=32,
        help="Height/width division factor for fit-resize operator.",
    )
    parser.add_argument(
        "--resize_mode",
        type=str,
        choices=["crop", "fit"],
        default="fit",
        help="Resize behavior in ImageCropAndResize.",
    )
    parser.add_argument("--max_pixels", type=int, default=4096 * 4096, help="Max pixels for dynamic resize.")
    parser.add_argument("--frame_stride", type=int, default=2, help="Sample every N frames (0, N, 2N...).")
    parser.add_argument("--dataset_num_frames", type=int, default=100001, help="Upper bound for dataset loader.")
    parser.add_argument("--chunk_glob", type=str, default="chunk-00*", help="Chunk glob under videos_subdir.")
    parser.add_argument("--chunk_names", type=str, default="", help="Optional chunk list: chunk-000,chunk-001")
    parser.add_argument(
        "--episodes",
        type=str,
        default="",
        help="Optional episode list (stem or filename): episode_000001,episode_000002",
    )
    parser.add_argument("--max_tasks", type=int, default=0, help="For debug: process only first N tasks.")

    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma-separated GPU ids. Empty means CPU.")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        help="Model compute dtype (e.g., bfloat16, float16, float32).",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="bfloat16",
        help="Saved latent tensor dtype (e.g., bfloat16, float16, float32).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing latent files.")
    parser.add_argument("--empty_cache_every", type=int, default=10, help="Call torch.cuda.empty_cache every N tasks.")
    parser.add_argument("--print_traceback", action="store_true", help="Print traceback on worker failures.")
    parser.add_argument("--dry_run", action="store_true", help="Only scan and print stats; do not run encoding.")
    parser.add_argument("--procs_per_gpu", type=int, default=1, help="Number of worker processes per GPU.")
    parser.add_argument(
        "--cpu_threads_per_worker",
        type=int,
        default=1,
        help="CPU threads per worker process (torch/OMP/MKL).",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    if args.frame_stride <= 0:
        raise ValueError(f"--frame_stride must be > 0, got {args.frame_stride}")
    if args.max_tasks < 0:
        raise ValueError(f"--max_tasks must be >= 0, got {args.max_tasks}")
    if args.procs_per_gpu <= 0:
        raise ValueError(f"--procs_per_gpu must be > 0, got {args.procs_per_gpu}")
    if args.cpu_threads_per_worker <= 0:
        raise ValueError(f"--cpu_threads_per_worker must be > 0, got {args.cpu_threads_per_worker}")

    views = parse_csv_list(args.views)
    if not views:
        raise ValueError("No valid views found. Please provide --views.")

    chunk_names = parse_csv_list(args.chunk_names)
    episode_filter = parse_episode_filter(args.episodes)
    dataset_roots = resolve_dataset_roots(args)
    if not dataset_roots:
        raise RuntimeError("No valid datasets to process after filtering.")

    tasks = []
    discovered_counts_by_dataset = {}
    dataset_order = []
    for dataset_root in dataset_roots:
        dataset_name = dataset_root.name
        dataset_order.append(dataset_name)
        try:
            dataset_tasks = discover_tasks(
                dataset_root=dataset_root,
                dataset_name=dataset_name,
                videos_subdir=args.videos_subdir,
                output_subdir=args.output_subdir,
                chunk_glob=args.chunk_glob,
                chunk_names=chunk_names,
                views=views,
                episode_filter=episode_filter,
            )
        except Exception as exc:
            print(f"[warn] Skip dataset {dataset_name}: {exc}", file=sys.stderr, flush=True)
            discovered_counts_by_dataset[dataset_name] = 0
            continue
        discovered_counts_by_dataset[dataset_name] = len(dataset_tasks)
        tasks.extend(dataset_tasks)

    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    effective_counts_by_dataset = {}
    for task in tasks:
        name = task.get("dataset_name", "dataset")
        effective_counts_by_dataset[name] = effective_counts_by_dataset.get(name, 0) + 1

    print(f"datasets_base={Path(args.datasets_base).resolve()}")
    if args.dataset_root:
        print(f"dataset_root_override={Path(args.dataset_root).resolve()}")
    print(f"exclude_datasets={parse_csv_list(args.exclude_datasets)}")
    print("datasets_selected:")
    for name in dataset_order:
        discovered_count = discovered_counts_by_dataset.get(name, 0)
        effective_count = effective_counts_by_dataset.get(name, 0)
        print(f"  {name}: discovered={discovered_count}, scheduled={effective_count}")
    print(f"model_dir={args.model_dir}")
    print(f"views={views}")
    print(f"discovered_tasks={len(tasks)}")
    if args.dry_run:
        return

    if len(tasks) == 0:
        print("No tasks to process. Exit.")
        return

    gpu_ids = parse_gpu_list(args.gpus)
    if gpu_ids:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but --gpus is set.")
        devices = [
            f"cuda:{gpu_id}"
            for gpu_id in gpu_ids
            for _ in range(int(args.procs_per_gpu))
        ]
    else:
        devices = ["cpu"]
        if args.torch_dtype != "float32":
            print("[warn] CPU mode usually needs --torch_dtype float32.", file=sys.stderr, flush=True)

    partitions: List[List[dict]] = [[] for _ in devices]
    for index, task in enumerate(tasks):
        partitions[index % len(devices)].append(task)

    run_plan = [(device, subset) for device, subset in zip(devices, partitions) if subset]
    print(f"workers={len(run_plan)}")
    for worker_id, (device, subset) in enumerate(run_plan):
        print(f"  worker-{worker_id}: device={device}, tasks={len(subset)}")

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    result_queue = ctx.Queue()
    processes = []
    args_dict = vars(args).copy()

    start_time = time.time()
    for worker_id, (device, subset) in enumerate(run_plan):
        process = ctx.Process(
            target=worker_main,
            args=(worker_id, device, subset, args_dict, views, progress_queue, result_queue),
        )
        process.start()
        processes.append(process)

    progress = {"processed": 0, "skipped": 0, "failed": 0}
    progress_by_dataset = {
        name: {"processed": 0, "skipped": 0, "failed": 0}
        for name in effective_counts_by_dataset
    }
    events_total = len(tasks)
    events_done = 0
    pbar = tqdm(total=events_total, desc="Extracting latents", dynamic_ncols=True)
    while events_done < events_total:
        try:
            event = progress_queue.get(timeout=0.5)
            status = event.get("status", "processed")
            if status not in progress:
                status = "processed"
            dataset_name = event.get("dataset_name", "")
            progress[status] += 1
            if dataset_name:
                if dataset_name not in progress_by_dataset:
                    progress_by_dataset[dataset_name] = {"processed": 0, "skipped": 0, "failed": 0}
                progress_by_dataset[dataset_name][status] += 1
            events_done += 1
            pbar.update(1)
            pbar.set_postfix(
                processed=progress["processed"],
                skipped=progress["skipped"],
                failed=progress["failed"],
                refresh=False,
            )
        except Empty:
            if not any(process.is_alive() for process in processes):
                # Drain remaining events once workers have exited.
                while True:
                    try:
                        event = progress_queue.get_nowait()
                        status = event.get("status", "processed")
                        if status not in progress:
                            status = "processed"
                        dataset_name = event.get("dataset_name", "")
                        progress[status] += 1
                        if dataset_name:
                            if dataset_name not in progress_by_dataset:
                                progress_by_dataset[dataset_name] = {"processed": 0, "skipped": 0, "failed": 0}
                            progress_by_dataset[dataset_name][status] += 1
                        events_done += 1
                        pbar.update(1)
                    except Empty:
                        break
                break
    pbar.close()

    results = []
    for process in processes:
        process.join()
    for _ in processes:
        results.append(result_queue.get())

    total_processed = sum(int(result["processed"]) for result in results)
    total_saved = sum(int(result["saved"]) for result in results)
    total_skipped = sum(int(result["skipped_existing"]) for result in results)
    total_failed = sum(int(result["failed"]) for result in results)
    crashed = [result for result in results if result.get("status") != "ok"]
    elapsed = max(1e-6, time.time() - start_time)
    throughput = events_done / elapsed

    print("\nSummary:")
    print(f"  tasks_total={len(tasks)}")
    print(f"  progress_events={events_done}")
    print(f"  episodes_processed={total_processed}")
    print(f"  files_saved={total_saved}")
    print(f"  skipped_existing={total_skipped}")
    print(f"  failed={total_failed}")
    print(f"  elapsed_sec={elapsed:.2f}")
    print(f"  episodes_per_sec={throughput:.4f}")
    print("  per_dataset:")
    for name in dataset_order:
        if name not in effective_counts_by_dataset and name not in progress_by_dataset:
            continue
        scheduled = effective_counts_by_dataset.get(name, 0)
        dataset_stats = progress_by_dataset.get(name, {"processed": 0, "skipped": 0, "failed": 0})
        print(
            "    "
            + f"{name}: scheduled={scheduled} "
            + f"processed={dataset_stats['processed']} "
            + f"skipped={dataset_stats['skipped']} "
            + f"failed={dataset_stats['failed']}"
        )
    for result in sorted(results, key=lambda item: item["worker_id"]):
        print(
            "  "
            + f"worker-{result['worker_id']} {result['device']} "
            + f"processed={result['processed']} "
            + f"saved={result['saved']} "
            + f"skipped={result['skipped_existing']} "
            + f"failed={result['failed']} "
            + f"status={result['status']}"
        )
        if result.get("error"):
            print(f"    error={result['error']}")

    if crashed:
        raise RuntimeError(f"{len(crashed)} workers crashed.")
    if total_failed > 0:
        raise RuntimeError(f"Encoding failed for {total_failed} episodes.")


if __name__ == "__main__":
    main()
