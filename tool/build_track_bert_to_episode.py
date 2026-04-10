#!/usr/bin/env python3
import argparse
import json
from pathlib import Path, PurePosixPath


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in {path}:{line_no}, got {type(row).__name__}")
            yield line_no, row


def derive_dataset_dir(episodes_jsonl: Path) -> Path:
    if episodes_jsonl.parent.name == "meta":
        return episodes_jsonl.parent.parent
    return episodes_jsonl.parent


def default_output_path(episodes_jsonl: Path) -> Path:
    name = episodes_jsonl.name
    if name.endswith(".jsonl"):
        stem = name[:-len(".jsonl")]
        return episodes_jsonl.with_name(f"{stem}.track_bert.jsonl")
    return episodes_jsonl.with_name(f"{episodes_jsonl.stem}.track_bert.jsonl")


def normalize_relative_path(path_value: str, dataset_dir: Path) -> PurePosixPath:
    candidate = Path(path_value)
    if candidate.is_absolute():
        try:
            rel = candidate.resolve().relative_to(dataset_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Absolute path {candidate} is not inside dataset_dir {dataset_dir}"
            ) from exc
        return PurePosixPath(rel.as_posix())

    rel = PurePosixPath(path_value)
    if rel.is_absolute():
        raise ValueError(f"Expected relative path, got {path_value}")
    if any(part == ".." for part in rel.parts):
        raise ValueError(f"Relative path must not contain '..': {path_value}")
    if rel.parts and rel.parts[0] == ".":
        rel = PurePosixPath(*rel.parts[1:])
    return rel


def require_file_exists(path: Path, context: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{context}: missing file {path}")


def load_prompt_to_index(tasks_jsonl: Path) -> dict[str, int]:
    prompt_to_index: dict[str, int] = {}
    for line_no, row in iter_jsonl(tasks_jsonl):
        prompt = row.get("task")
        if prompt is None:
            prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing valid task/prompt in {tasks_jsonl}:{line_no}")

        task_index = row.get("task_index")
        try:
            task_index = int(task_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Missing valid task_index in {tasks_jsonl}:{line_no}") from exc

        existing = prompt_to_index.get(prompt)
        if existing is not None and existing != task_index:
            raise ValueError(
                f"Conflicting task_index for prompt {prompt!r} in {tasks_jsonl}:{line_no}: "
                f"{existing} vs {task_index}"
            )
        prompt_to_index[prompt] = task_index

    if not prompt_to_index:
        raise ValueError(f"No task entries found in {tasks_jsonl}")
    return prompt_to_index


def build_prompt_embed_map(
    prompt_to_index: dict[str, int],
    dataset_dir: Path,
    prompt_embed_subdir: PurePosixPath,
) -> dict[str, str]:
    prompt_embed_map: dict[str, str] = {}
    for prompt, task_index in prompt_to_index.items():
        rel_path = prompt_embed_subdir / f"pos_{task_index}.pt"
        require_file_exists(
            dataset_dir / Path(rel_path.as_posix()),
            context=f"prompt_embed_bert for prompt {prompt!r}",
        )
        prompt_embed_map[prompt] = rel_path.as_posix()
    return prompt_embed_map


def map_video_to_track_path(
    video_path: str,
    dataset_dir: Path,
    cache: dict[str, str],
) -> str:
    rel_video = normalize_relative_path(video_path, dataset_dir)
    rel_video_key = rel_video.as_posix()
    cached = cache.get(rel_video_key)
    if cached is not None:
        return cached

    parts = rel_video.parts
    if len(parts) != 4 or parts[0] != "videos":
        raise ValueError(
            "Video path must match videos/chunk-XXX/observation.images.<camera>/episode_xxxxxx.mp4, "
            f"got {video_path}"
        )

    _, chunk_dir, camera_dir, filename = parts
    if not camera_dir.startswith("observation.images."):
        raise ValueError(f"Unsupported video camera directory in {video_path}")
    if not filename.startswith("episode_") or not filename.endswith(".mp4"):
        raise ValueError(f"Unsupported video filename in {video_path}")

    track_camera_dir = camera_dir.replace("observation.images.", "observation.tracks.", 1)
    track_rel = PurePosixPath("tracks", chunk_dir, track_camera_dir, f"{Path(filename).stem}.npz")
    require_file_exists(
        dataset_dir / Path(track_rel.as_posix()),
        context=f"track for video {video_path}",
    )
    track_rel_key = track_rel.as_posix()
    cache[rel_video_key] = track_rel_key
    return track_rel_key


def build_output_row(
    row: dict,
    *,
    dataset_dir: Path,
    prompt_embed_map: dict[str, str],
    track_cache: dict[str, str],
    source_name: str,
    line_no: int,
) -> dict:
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Missing valid prompt in {source_name}:{line_no}")
    if prompt not in prompt_embed_map:
        raise KeyError(f"Prompt not found in tasks.jsonl for {source_name}:{line_no}: {prompt!r}")

    video_list = row.get("video")
    if not isinstance(video_list, list) or not video_list:
        raise ValueError(f"`video` must be a non-empty list in {source_name}:{line_no}")

    track_list: list[str] = []
    for item in video_list:
        if not isinstance(item, str) or not item:
            raise ValueError(f"Invalid video entry in {source_name}:{line_no}: {item!r}")
        track_list.append(map_video_to_track_path(item, dataset_dir, track_cache))

    output_row = dict(row)
    output_row["track"] = track_list
    output_row["prompt_embed_bert"] = prompt_embed_map[prompt]
    return output_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a new episodes jsonl by adding `track` and `prompt_embed_bert` "
            "fields with on-disk validation."
        )
    )
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        default="/data_jbx/Datasets/Realbot/4_4_four_tasks_wan/meta/episodes_train.jsonl",
        help="Input episodes jsonl, e.g. dataset_dir/meta/episodes_train.jsonl",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default="/data_jbx/Datasets/Realbot/4_4_four_tasks_wan",
        help="Dataset root. Defaults to the parent of `meta/` when omitted.",
    )
    parser.add_argument(
        "--tasks-jsonl",
        type=Path,
        default="/data_jbx/Datasets/Realbot/4_4_four_tasks_wan/meta/tasks.jsonl",
        help="Task metadata jsonl. Defaults to dataset_dir/meta/tasks.jsonl.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Output jsonl. Defaults to <episodes>.track_bert.jsonl in the source directory.",
    )
    parser.add_argument(
        "--prompt-embed-subdir",
        type=str,
        default="prompt_emb/bert",
        help="Relative prompt embedding directory under dataset_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    episodes_jsonl = args.episodes_jsonl.expanduser().resolve()
    dataset_dir = (
        args.dataset_dir.expanduser().resolve()
        if args.dataset_dir is not None
        else derive_dataset_dir(episodes_jsonl).resolve()
    )
    tasks_jsonl = (
        args.tasks_jsonl.expanduser().resolve()
        if args.tasks_jsonl is not None
        else (dataset_dir / "meta" / "tasks.jsonl").resolve()
    )
    output_jsonl = (
        args.output_jsonl.expanduser().resolve()
        if args.output_jsonl is not None
        else default_output_path(episodes_jsonl).resolve()
    )

    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"dataset_dir does not exist: {dataset_dir}")
    require_file_exists(episodes_jsonl, context="episodes_jsonl")
    require_file_exists(tasks_jsonl, context="tasks_jsonl")
    if output_jsonl.parent and not output_jsonl.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_jsonl.parent}")

    prompt_embed_subdir = normalize_relative_path(args.prompt_embed_subdir, dataset_dir)
    prompt_to_index = load_prompt_to_index(tasks_jsonl)
    prompt_embed_map = build_prompt_embed_map(prompt_to_index, dataset_dir, prompt_embed_subdir)

    track_cache: dict[str, str] = {}
    record_count = 0
    tmp_output = output_jsonl.with_name(f"{output_jsonl.name}.tmp")
    with tmp_output.open("w", encoding="utf-8") as handle:
        for line_no, row in iter_jsonl(episodes_jsonl):
            output_row = build_output_row(
                row,
                dataset_dir=dataset_dir,
                prompt_embed_map=prompt_embed_map,
                track_cache=track_cache,
                source_name=str(episodes_jsonl),
                line_no=line_no,
            )
            handle.write(json.dumps(output_row, ensure_ascii=True))
            handle.write("\n")
            record_count += 1

    tmp_output.replace(output_jsonl)

    print(f"dataset_dir: {dataset_dir}")
    print(f"episodes_jsonl: {episodes_jsonl}")
    print(f"tasks_jsonl: {tasks_jsonl}")
    print(f"output_jsonl: {output_jsonl}")
    print(f"records_written: {record_count}")


if __name__ == "__main__":
    main()
