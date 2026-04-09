#!/usr/bin/env python3
"""Convert robot_data/0325_total to a wan-style dataset (0325_wan)."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


TARGET_NAMES = [
    "left_arm_joint_1_rad",
    "left_arm_joint_2_rad",
    "left_arm_joint_3_rad",
    "left_arm_joint_4_rad",
    "left_arm_joint_5_rad",
    "left_arm_joint_6_rad",
    "left_gripper_open",
    "left_eef_pos_x_m",
    "left_eef_pos_y_m",
    "left_eef_pos_z_m",
    "left_eef_rot_euler_x_rad",
    "left_eef_rot_euler_y_rad",
    "left_eef_rot_euler_z_rad",
    "right_arm_joint_1_rad",
    "right_arm_joint_2_rad",
    "right_arm_joint_3_rad",
    "right_arm_joint_4_rad",
    "right_arm_joint_5_rad",
    "right_arm_joint_6_rad",
    "right_gripper_open",
    "right_eef_pos_x_m",
    "right_eef_pos_y_m",
    "right_eef_pos_z_m",
    "right_eef_rot_euler_x_rad",
    "right_eef_rot_euler_y_rad",
    "right_eef_rot_euler_z_rad",
]

SOURCE_EEF_7_NAMES = [
    "X_axis.pos",
    "Y_axis.pos",
    "Z_axis.pos",
    "RX_axis.pos",
    "RY_axis.pos",
    "RZ_axis.pos",
    "gripper.pos",
]

JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19]
POSE_INDICES = [7, 8, 9, 10, 11, 12, 6, 20, 21, 22, 23, 24, 25, 19]

DEFAULT_PROMPT = "Pick up the cube and place on the plate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build robot_data/0325_wan from robot_data/0325_total"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("robot_data/0325_total"),
        help="Source dataset root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("robot_data/0325_wan"),
        help="Output dataset root",
    )
    parser.add_argument(
        "--prompt-emb-src",
        type=Path,
        default=Path("robot_data/piper/prompt_emb/pos_0.pt"),
        help="Source prompt embedding file",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=17,
        help="Window length for meta/train.jsonl",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Window stride for meta/train.jsonl",
    )
    parser.add_argument(
        "--min-tail",
        type=int,
        default=5,
        help="Keep tail window if remaining frames >= min-tail",
    )
    parser.add_argument(
        "--range-start",
        type=int,
        default=0,
        help="Clip range start frame for each episode",
    )
    parser.add_argument(
        "--range-end",
        type=int,
        default=-1,
        help="Clip range end frame for each episode (-1 means full episode end)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output root if exists",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def dump_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def ensure_2d_float32(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def column_to_2d_float32(column: pa.ChunkedArray, name: str) -> np.ndarray:
    return ensure_2d_float32(np.asarray(column.to_pylist(), dtype=np.float32), name)


def map_7d_to_26d(arr7: np.ndarray) -> np.ndarray:
    """Map 7D single-arm [xyzrpy + gripper] to duplicated 26D."""
    arr7 = ensure_2d_float32(arr7, "input_7d")
    if arr7.shape[1] != 7:
        raise ValueError(f"Expected 7 dims, got {arr7.shape[1]}")

    out = np.zeros((arr7.shape[0], 26), dtype=np.float32)
    xyzrpy = arr7[:, 0:6]
    gripper = arr7[:, 6]

    out[:, 6] = gripper
    out[:, 19] = gripper
    out[:, 7:13] = xyzrpy
    out[:, 20:26] = xyzrpy
    return out


def normalize_to_26d(arr: np.ndarray, col_name: str) -> np.ndarray:
    arr = ensure_2d_float32(arr, col_name)
    if arr.shape[1] == 26:
        return arr
    if arr.shape[1] == 7:
        return map_7d_to_26d(arr)
    raise ValueError(f"Unsupported {col_name} dimension {arr.shape[1]}, expected 7/26")


def validate_source_info_7d(src_info: dict) -> None:
    features = src_info.get("features", {})
    for key in ("action", "observation.state"):
        feat = features.get(key)
        if not isinstance(feat, dict):
            raise ValueError(f"info.json missing features.{key}")
        shape = feat.get("shape")
        names = feat.get("names")
        if shape != [7]:
            raise ValueError(f"features.{key}.shape must be [7], got {shape}")
        if names != SOURCE_EEF_7_NAMES:
            raise ValueError(
                f"features.{key}.names mismatch; expected {SOURCE_EEF_7_NAMES}, got {names}"
            )


def ndarray_to_fixed_size_list(arr: np.ndarray) -> pa.Array:
    flat = pa.array(arr.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, arr.shape[1])


def convert_parquet(src_path: Path, dst_path: Path) -> int:
    table = pq.read_table(src_path)
    names = table.schema.names

    if "action" not in names or "observation.state" not in names:
        raise KeyError(f"Missing action or observation.state in {src_path}")

    action_np = normalize_to_26d(column_to_2d_float32(table["action"], "action"), "action")
    state_np = normalize_to_26d(
        column_to_2d_float32(table["observation.state"], "observation.state"),
        "observation.state",
    )

    if action_np.shape[0] != state_np.shape[0]:
        raise ValueError(
            f"Row mismatch in {src_path}: action={action_np.shape[0]}, state={state_np.shape[0]}"
        )

    new_cols = []
    for name in names:
        if name == "action":
            new_cols.append(ndarray_to_fixed_size_list(action_np))
        elif name == "observation.state":
            new_cols.append(ndarray_to_fixed_size_list(state_np))
        else:
            new_cols.append(table[name])

    new_table = pa.table(new_cols, names=names)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst_path, compression="snappy")
    return int(new_table.num_rows)


def summarize(arr: np.ndarray) -> dict:
    return {
        "shape": [arr.shape[1]],
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "p01": np.percentile(arr, 1, axis=0).tolist(),
        "p99": np.percentile(arr, 99, axis=0).tolist(),
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
    }


def compute_stat_from_parquets(parquet_files: List[Path], out_path: Path) -> None:
    state_joint_list = []
    action_joint_list = []
    state_pose_list = []
    action_pose_list = []

    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path, columns=["action", "observation.state"])
        action = column_to_2d_float32(table["action"], "action")
        state = column_to_2d_float32(table["observation.state"], "observation.state")

        if action.shape[1] != 26 or state.shape[1] != 26:
            raise ValueError(
                f"Expected 26 dims in {parquet_path}, got action={action.shape[1]}, state={state.shape[1]}"
            )

        action_joint_list.append(action[:, JOINT_INDICES])
        state_joint_list.append(state[:, JOINT_INDICES])
        action_pose_list.append(action[:, POSE_INDICES])
        state_pose_list.append(state[:, POSE_INDICES])

    action_joint = np.concatenate(action_joint_list, axis=0)
    state_joint = np.concatenate(state_joint_list, axis=0)
    action_pose = np.concatenate(action_pose_list, axis=0)
    state_pose = np.concatenate(state_pose_list, axis=0)

    payload = {
        "state_joint": summarize(state_joint),
        "action_joint": summarize(action_joint),
        "state_pose": summarize(state_pose),
        "action_pose": summarize(action_pose),
    }
    dump_json(out_path, payload)


def build_episode_row(episode_index: int, length: int, prompt: str) -> dict:
    return {
        "episode_index": int(episode_index),
        "length": int(length),
        "start_frame": 0,
        "end_frame": max(int(length) - 1, 0),
        "video": [
            f"videos/chunk-000/observation.images.image/episode_{episode_index:06d}.mp4",
            f"videos/chunk-000/observation.images.wrist_image/episode_{episode_index:06d}.mp4",
        ],
        "action": f"data/chunk-000/episode_{episode_index:06d}.parquet",
        "prompt": prompt,
        "prompt_emb": "prompt_emb/pos_0.pt",
    }


def build_train_rows(
    val_rows: List[dict],
    clip_length: int,
    stride: int,
    min_tail: int,
    range_start: int,
    range_end: int,
) -> List[dict]:
    if clip_length <= 0:
        raise ValueError("clip_length must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if min_tail <= 0:
        raise ValueError("min_tail must be > 0")
    if range_start < 0:
        raise ValueError("range_start must be >= 0")

    out = []
    for row in val_rows:
        ep = int(row["episode_index"])
        full_start = int(row["start_frame"])
        full_end = int(row["end_frame"])
        length = int(row["length"])

        if full_end < full_start:
            continue
        if length != full_end - full_start + 1:
            raise ValueError(
                f"length mismatch in episode {ep}: length={length}, range={full_start}-{full_end}"
            )

        start = full_start + range_start
        end = full_end if range_end < 0 else min(full_start + range_end, full_end)
        if start > end:
            continue

        cur = start
        while cur + clip_length - 1 <= end:
            train_row = dict(row)
            train_row["start_frame"] = int(cur)
            train_row["end_frame"] = int(cur + clip_length - 1)
            train_row["length"] = int(clip_length)
            out.append(train_row)
            cur += stride

        rem = end - cur + 1
        if rem >= min_tail:
            train_row = dict(row)
            train_row["start_frame"] = int(cur)
            train_row["end_frame"] = int(end)
            train_row["length"] = int(rem)
            out.append(train_row)

    return out


def update_info_json(src_info: dict, total_episodes: int, total_frames: int, total_videos: int) -> dict:
    info = json.loads(json.dumps(src_info))
    features = dict(info.get("features", {}))

    features["action"] = {
        "dtype": "float32",
        "shape": [26],
        "names": TARGET_NAMES,
    }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [26],
        "names": TARGET_NAMES,
    }

    info["features"] = features
    info["total_episodes"] = int(total_episodes)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = 1
    info["total_videos"] = int(total_videos)
    chunks_size = int(info.get("chunks_size", 1000))
    info["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes else 0
    info["splits"] = {"train": f"0:{total_episodes}"}
    return info


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    prompt_emb_src = args.prompt_emb_src.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Missing input root: {input_root}")
    if not prompt_emb_src.exists():
        raise FileNotFoundError(f"Missing prompt embedding source: {prompt_emb_src}")

    src_meta = input_root / "meta"
    src_data = input_root / "data"
    src_videos = input_root / "videos"

    src_info = load_json(src_meta / "info.json")
    validate_source_info_7d(src_info)
    src_episodes = load_jsonl(src_meta / "episodes.jsonl")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_root}. Use --overwrite.")
        shutil.rmtree(output_root)

    data_out = output_root / "data"
    meta_out = output_root / "meta"
    videos_out = output_root / "videos"
    prompt_emb_out = output_root / "prompt_emb"

    data_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)
    prompt_emb_out.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Copy videos: {src_videos} -> {videos_out}")
    shutil.copytree(src_videos, videos_out, dirs_exist_ok=True)

    parquet_files = sorted(src_data.glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {src_data}")

    print(f"[2/6] Convert parquet to 26D: {len(parquet_files)} files")
    total_rows = 0
    out_parquet_files = []
    for src_parquet in parquet_files:
        rel = src_parquet.relative_to(src_data)
        dst_parquet = data_out / rel
        rows = convert_parquet(src_parquet, dst_parquet)
        total_rows += rows
        out_parquet_files.append(dst_parquet)

    print("[3/6] Build episodes/tasks metadata")
    episodes_rows = []
    for row in src_episodes:
        ep = int(row["episode_index"])
        length = int(row["length"])
        episodes_rows.append(build_episode_row(ep, length, DEFAULT_PROMPT))

    episodes_rows.sort(key=lambda x: x["episode_index"])
    episodes_val_rows = list(episodes_rows)
    tasks_rows = [{"task_index": 0, "task": DEFAULT_PROMPT}]

    print("[4/6] Build train.jsonl clips")
    train_rows = build_train_rows(
        val_rows=episodes_val_rows,
        clip_length=args.clip_length,
        stride=args.stride,
        min_tail=args.min_tail,
        range_start=args.range_start,
        range_end=args.range_end,
    )

    print("[5/6] Write meta files and prompt_emb")
    dump_jsonl(meta_out / "episodes.jsonl", episodes_rows)
    dump_jsonl(meta_out / "episodes_val.jsonl", episodes_val_rows)
    dump_jsonl(meta_out / "train.jsonl", train_rows)
    dump_jsonl(meta_out / "episodes_train.jsonl", train_rows)
    dump_jsonl(meta_out / "tasks.jsonl", tasks_rows)

    videos_count = len(list(videos_out.glob("chunk-*/*/episode_*.mp4")))
    info_out = update_info_json(
        src_info=src_info,
        total_episodes=len(episodes_rows),
        total_frames=total_rows,
        total_videos=videos_count,
    )
    dump_json(meta_out / "info.json", info_out)

    shutil.copy2(prompt_emb_src, prompt_emb_out / "pos_0.pt")

    print("[6/6] Compute stat.json")
    compute_stat_from_parquets(out_parquet_files, meta_out / "stat.json")

    print("Done.")
    print(f"Output root: {output_root}")
    print(
        f"Episodes={len(episodes_rows)}, Frames={total_rows}, Videos={videos_count}, "
        f"Train clips={len(train_rows)}"
    )


if __name__ == "__main__":
    main()
