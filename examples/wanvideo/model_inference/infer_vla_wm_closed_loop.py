#!/usr/bin/env python
"""
Closed-loop inference:
1) use VLA with latest observation frame + 7D state to predict future states
2) convert VLA outputs to action and feed Wan world model
3) export rollout as LeRobot v2.1 dataset format
"""
import argparse
import fcntl
import json
import logging
import os
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from diffsynth.pipelines.wan_video_data import (
    WAN_INFERENCE_DATASET_NUM_FRAMES,
    build_wan_video_dataset,
)
from diffsynth.utils.data import save_video

from inference_support import (
    CheckpointPipelineManager,
    VideoSaver,
    WanInferenceConfig,
    build_wan_inference_config,
    flatten_grouped_config,
    resolve_optional_path,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
VLA_INTERFACE_DIR = REPO_ROOT / "VLA" / "interface"
import sys

sys.path.append(str(VLA_INTERFACE_DIR))
from websocket_policy_server import ExternalRobotInferenceClient


LEFT_POSE7_INDICES = [7, 8, 9, 10, 11, 12, 6]  # xyz rpy + gripper
POSE7_FEATURE_NAMES = [
    "left_eef_pos_x_m",
    "left_eef_pos_y_m",
    "left_eef_pos_z_m",
    "left_eef_rot_euler_x_rad",
    "left_eef_rot_euler_y_rad",
    "left_eef_rot_euler_z_rad",
    "left_gripper_open",
]


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("infer_vla_wm_closed_loop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLA + Wan WM closed-loop inference")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--config_json", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)

    parser.add_argument("--model_paths", type=str, default=None)
    parser.add_argument("--load_modules", type=str, default=None)

    parser.add_argument("--dataset_base_path", type=str, required=True)
    parser.add_argument("--dataset_metadata_path", type=str, required=True)
    parser.add_argument("--action_stat_path", type=str, default=None)
    parser.add_argument("--action_type", type=str, default=None)

    parser.add_argument("--vla_host", type=str, default="100.64.147.46")
    parser.add_argument("--vla_port", type=int, default=6667)
    parser.add_argument("--robot_uid", type=str, default="piper_real")

    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--metrics", type=int, choices=[0, 1], default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--process_all_episodes", type=int, choices=[0, 1], default=int(os.getenv("PROCESS_ALL_EPISODES", "1")))
    parser.add_argument("--repeat_per_episode", type=int, default=int(os.getenv("REPEAT_PER_EPISODE", "5")))
    parser.add_argument("--start_jitter", type=int, default=int(os.getenv("START_JITTER", "5")))
    parser.add_argument("--max_length_multiplier", type=float, default=float(os.getenv("MAX_LENGTH_MULTIPLIER", "1.3")))
    parser.add_argument(
        "--init_frame_sampling_mode",
        type=str,
        choices=["jitter", "random_within_source"],
        default=os.getenv("INIT_FRAME_SAMPLING_MODE", "jitter"),
    )
    parser.add_argument(
        "--length_sampling_mode",
        type=str,
        choices=["source_multiplier", "reference_prompt"],
        default=os.getenv("LENGTH_SAMPLING_MODE", "source_multiplier"),
    )
    parser.add_argument(
        "--length_reference_metadata_path",
        type=str,
        default=os.getenv("LENGTH_REFERENCE_METADATA_PATH", ""),
    )
    parser.add_argument("--chunk_size", type=int, default=int(os.getenv("CHUNK_SIZE", "1000")))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--follow_metadata", type=int, choices=[0, 1], default=1)
    parser.add_argument("--poll_interval_sec", type=float, default=2.0)
    parser.add_argument("--idle_timeout_sec", type=float, default=300.0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--strict_lerobot_v21", type=int, choices=[0, 1], default=1)

    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "The video is not of a high quality, it has a low resolution. "
            "Watermark present in each frame. The background is solid. "
            "Strange body and strange trajectory. Distortion"
        ),
    )
    parser.add_argument("--negative_prompt_emb", type=str, default="prompt_emb/neg_prompt.pt")
    args = parser.parse_args()
    if args.repeat_per_episode < 1:
        raise ValueError(f"repeat_per_episode must be >= 1, got {args.repeat_per_episode}")
    if args.start_jitter < 0:
        raise ValueError(f"start_jitter must be >= 0, got {args.start_jitter}")
    if args.max_length_multiplier < 1.0:
        raise ValueError(f"max_length_multiplier must be >= 1.0, got {args.max_length_multiplier}")
    if args.chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {args.chunk_size}")
    if args.worker_id < 0:
        raise ValueError(f"worker_id must be >= 0, got {args.worker_id}")
    if args.num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {args.num_workers}")
    if args.worker_id >= args.num_workers:
        raise ValueError(f"worker_id must be < num_workers, got worker_id={args.worker_id}, num_workers={args.num_workers}")
    if args.poll_interval_sec <= 0:
        raise ValueError(f"poll_interval_sec must be > 0, got {args.poll_interval_sec}")
    if args.idle_timeout_sec <= 0:
        raise ValueError(f"idle_timeout_sec must be > 0, got {args.idle_timeout_sec}")
    return args


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_paths_from_tag_epoch(args: argparse.Namespace) -> Tuple[str, str]:
    if args.ckpt_path and args.config_json:
        return args.ckpt_path, args.config_json
    if args.tag is None or args.epoch is None:
        raise ValueError("Either provide both --tag/--epoch or explicitly set --ckpt_path and --config_json.")
    ckpt_path = args.ckpt_path or f"Ckpt/{args.tag}/epoch-{args.epoch}/epoch-{args.epoch}.safetensors"
    config_json = args.config_json or f"Ckpt/{args.tag}/epoch-{args.epoch}/config.json"
    return ckpt_path, config_json


def to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    frame = frame_chw.detach().to(torch.float32).cpu().numpy().transpose(1, 2, 0)
    min_v, max_v = float(frame.min()), float(frame.max())
    if min_v >= -1.0 and max_v <= 1.0:
        if min_v < 0.0:
            frame = (frame + 1.0) * 127.5
        else:
            frame = frame * 255.0
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], repeats=3, axis=2)
    return frame


def concat_views_frame(video_vchw: torch.Tensor) -> np.ndarray:
    frames = [to_uint8_hwc(video_vchw[v]) for v in range(video_vchw.shape[0])]
    return np.hstack(frames)


def parse_vla_actions(response: Dict[str, Any]) -> np.ndarray:
    if "action.position" in response and "action.rotation" in response and "action.gripper" in response:
        pos = np.asarray(response["action.position"], dtype=np.float32)
        rot = np.asarray(response["action.rotation"], dtype=np.float32)
        grp = np.asarray(response["action.gripper"], dtype=np.float32)
        if grp.ndim == 1:
            grp = grp[:, None]
        return np.concatenate([pos, rot, grp], axis=1)
    if "actions" in response:
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Unexpected actions shape from VLA: {actions.shape}")
        return actions
    raise KeyError(f"Unknown VLA response format, keys={sorted(response.keys())}")


def resize_uint8_image(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    pil = Image.fromarray(image)
    pil = pil.resize((target_w, target_h), Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def get_vla_action_with_fallback(
    client: ExternalRobotInferenceClient,
    base_img: np.ndarray,
    wrist_img: np.ndarray,
    state7: np.ndarray,
    prompt: str,
) -> Dict[str, Any]:
    payload_variants = [
        {
            "video.image": base_img[None, :],
            "video.wrist_image": wrist_img[None, :],
            "state": state7[None, :],
            "annotation.human.task_description": [prompt],
        },
        {
            "video.image": resize_uint8_image(base_img, 480, 640)[None, :],
            "video.wrist_image": resize_uint8_image(wrist_img, 480, 640)[None, :],
            "state": state7[None, :],
            "annotation.human.task_description": [prompt],
        },
    ]
    last_err: Exception | None = None
    for i, payload in enumerate(payload_variants, start=1):
        try:
            return client.get_action(payload)
        except Exception as exc:
            last_err = exc
            bshape = payload["video.image"].shape
            wshape = payload["video.wrist_image"].shape
            raise_msg = (
                f"VLA get_action failed on variant#{i}: base={bshape}, wrist={wshape}, "
                f"state={payload['state'].shape}, prompt_len={len(prompt)}; err={exc}"
            )
            if i < len(payload_variants):
                continue
            raise RuntimeError(raise_msg) from last_err
    raise RuntimeError(f"VLA get_action failed unexpectedly: {last_err}")


def state7_to_state14_copy(state7: np.ndarray) -> np.ndarray:
    if state7.ndim == 1:
        state7 = state7[None, :]
    if state7.shape[1] != 7:
        raise ValueError(f"Expected (*, 7), got {state7.shape}")
    return np.concatenate([state7, state7], axis=1)


def normalize_bound(
    data: np.ndarray,
    data_min: np.ndarray,
    data_max: np.ndarray,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    ndata = 2.0 * (data - data_min) / (data_max - data_min + eps) - 1.0
    return np.clip(ndata, clip_min, clip_max)


def load_raw_pose7_sequence(parquet_path: str, start_frame: int, end_frame: int) -> np.ndarray:
    table = pq.read_table(parquet_path, columns=["observation.state"])
    rows = table.to_pydict()["observation.state"]
    arr = np.asarray(rows[start_frame : end_frame + 1], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 13:
        raise ValueError(f"Unexpected observation.state shape in {parquet_path}: {arr.shape}")
    return arr[:, LEFT_POSE7_INDICES]


def prepare_runtime_config(args: argparse.Namespace, logger: logging.Logger) -> Tuple[WanInferenceConfig, Dict[str, Any]]:
    ckpt_path, config_json = resolve_paths_from_tag_epoch(args)
    grouped_cfg = load_json(config_json)
    base_cfg = flatten_grouped_config(grouped_cfg)

    model_paths = args.model_paths or base_cfg.get("model_paths")
    load_modules = args.load_modules or base_cfg.get("load_modules")
    if not model_paths:
        raise ValueError("model_paths is required (pass --model_paths or provide it in config.json).")

    config = build_wan_inference_config(
        {
            "checkpoint_path": ckpt_path,
            "model_paths": model_paths,
            "load_modules": load_modules,
        },
        grouped_config=grouped_cfg,
        data_file_keys=("video", "action"),
    )
    logger.info("Resolved checkpoint: %s", ckpt_path)
    logger.info("Resolved config: %s", config_json)
    return config, base_cfg


def build_rollout_dataset(
    runtime,
    *,
    dataset_base_path: str,
    dataset_metadata_path: str,
    action_stat_path: str,
    action_type: str,
    height: int,
    width: int,
    sample_index: int,
    num_frames: int,
    num_history_frames: int,
    history_template_sampling: int = 0,
    history_anchor_stride: int = 8,
):
    return build_wan_video_dataset(
        runtime,
        base_path=dataset_base_path,
        metadata_path=dataset_metadata_path,
        height=height,
        width=width,
        num_frames=num_frames,
        num_history_frames=num_history_frames,
        repeat=1,
        resize_mode="fit",
        data_file_keys=("video", "action"),
        dataset_num_frames=WAN_INFERENCE_DATASET_NUM_FRAMES,
        sample_indices=[sample_index],
        action_stat_path=action_stat_path,
        action_type=action_type,
        history_template_sampling=history_template_sampling,
        history_anchor_stride=history_anchor_stride,
    )


def fixed_size_list_array(values: np.ndarray, width: int) -> pa.Array:
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"Expected shape (*, {width}), got {values.shape}")
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, width)


def write_episode_parquet(
    output_path: Path,
    action_raw7: np.ndarray,
    state_raw7: np.ndarray,
    fps: int,
    batch_episode_index: int,
    global_index_start: int,
    task_index: int,
) -> int:
    if action_raw7.shape != state_raw7.shape:
        raise ValueError(f"Action/state shape mismatch: {action_raw7.shape} vs {state_raw7.shape}")
    length = action_raw7.shape[0]
    frame_index = np.arange(length, dtype=np.int64)
    timestamp = (frame_index.astype(np.float32) / float(fps)).astype(np.float32)
    episode_index = np.full(length, batch_episode_index, dtype=np.int64)
    index = global_index_start + frame_index
    task_index_col = np.full(length, task_index, dtype=np.int64)

    table = pa.Table.from_arrays(
        [
            fixed_size_list_array(action_raw7.astype(np.float32), 7),
            fixed_size_list_array(state_raw7.astype(np.float32), 7),
            pa.array(timestamp, type=pa.float32()),
            pa.array(frame_index, type=pa.int64()),
            pa.array(episode_index, type=pa.int64()),
            pa.array(index, type=pa.int64()),
            pa.array(task_index_col, type=pa.int64()),
        ],
        names=["action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"],
    )
    pq.write_table(table, output_path)
    return int(global_index_start + length)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def read_metadata_rows(path: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                logger.warning("Skip invalid metadata line %s in %s: %s", line_no, path, exc)
                continue
            if not isinstance(item, dict):
                logger.warning("Skip non-dict metadata line %s in %s", line_no, path)
                continue
            rows.append(item)
    return rows


def build_prompt_length_map(
    rows: List[Dict[str, Any]],
    logger: logging.Logger,
    source_path: str,
) -> Dict[str, List[int]]:
    prompt_to_lengths: Dict[str, List[int]] = {}
    invalid_rows = 0
    for idx, row in enumerate(rows):
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            invalid_rows += 1
            continue
        length_raw = row.get("length")
        try:
            length = int(length_raw)
        except Exception:
            invalid_rows += 1
            continue
        if length <= 0:
            invalid_rows += 1
            continue
        prompt_to_lengths.setdefault(prompt, []).append(length)
    if invalid_rows > 0:
        logger.warning(
            "Skipped %s invalid prompt-length rows while building reference map from %s",
            invalid_rows,
            source_path,
        )
    return prompt_to_lengths


def load_prompt_length_map(path: str, logger: logging.Logger) -> Dict[str, List[int]]:
    rows = read_metadata_rows(path, logger)
    return build_prompt_length_map(rows, logger, path)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    os.replace(tmp_path, path)


def load_state(path: Path) -> Dict[str, Any]:
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {}
    if not isinstance(state, dict):
        state = {}
    state.setdefault("next_task_id", 0)
    state.setdefault("next_episode_index", 0)
    state.setdefault("next_frame_index", 0)
    state.setdefault("num_success", 0)
    state.setdefault("num_failed", 0)
    prompt_map = state.get("prompt_to_task_index")
    if not isinstance(prompt_map, dict):
        prompt_map = {}
    normalized_prompt_map: Dict[str, int] = {}
    for k, v in prompt_map.items():
        try:
            normalized_prompt_map[str(k)] = int(v)
        except Exception:
            continue
    state["prompt_to_task_index"] = normalized_prompt_map
    return state


def with_locked_state(meta_dir: Path, fn) -> Any:
    lock_path = meta_dir / "state.lock"
    state_path = meta_dir / "state.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            state = load_state(state_path)
            result = fn(state)
            atomic_write_json(state_path, state)
            return result
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def append_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")
        f.flush()


def _as_numpy_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    return arr.reshape(-1)


def compute_numeric_feature_stats(values: np.ndarray) -> Dict[str, List[float]]:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array for stats, got shape={arr.shape}")
    arr_f64 = arr.astype(np.float64)
    minimum = arr.min(axis=0)
    maximum = arr.max(axis=0)
    mean = arr_f64.mean(axis=0)
    std = arr_f64.std(axis=0)
    count = np.asarray([arr.shape[0]], dtype=np.int64)
    return {
        "min": _as_numpy_1d(minimum).tolist(),
        "max": _as_numpy_1d(maximum).tolist(),
        "mean": _as_numpy_1d(mean).tolist(),
        "std": _as_numpy_1d(std).tolist(),
        "count": _as_numpy_1d(count).tolist(),
    }


def build_episode_stats_row(
    action_raw7: np.ndarray,
    state_raw7: np.ndarray,
    fps: int,
    batch_episode_index: int,
    global_index_start: int,
    task_index: int,
) -> Dict[str, Any]:
    if action_raw7.shape != state_raw7.shape:
        raise ValueError(f"Action/state shape mismatch for stats: {action_raw7.shape} vs {state_raw7.shape}")
    length = int(action_raw7.shape[0])
    frame_index = np.arange(length, dtype=np.int64)
    timestamp = frame_index.astype(np.float32) / float(fps)
    episode_index_col = np.full(length, batch_episode_index, dtype=np.int64)
    index_col = global_index_start + frame_index
    task_index_col = np.full(length, task_index, dtype=np.int64)
    stats = {
        "action": compute_numeric_feature_stats(action_raw7.astype(np.float32)),
        "observation.state": compute_numeric_feature_stats(state_raw7.astype(np.float32)),
        "timestamp": compute_numeric_feature_stats(timestamp.astype(np.float32)),
        "frame_index": compute_numeric_feature_stats(frame_index),
        "episode_index": compute_numeric_feature_stats(episode_index_col),
        "index": compute_numeric_feature_stats(index_col),
        "task_index": compute_numeric_feature_stats(task_index_col),
    }
    return {"episode_index": int(batch_episode_index), "stats": stats}


def aggregate_episode_stats(stats_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    stats_list = [row.get("stats") for row in stats_rows if isinstance(row.get("stats"), dict)]
    if not stats_list:
        return {}

    aggregated: Dict[str, Dict[str, List[float]]] = {}
    feature_keys = sorted({k for stats in stats_list for k in stats.keys()})
    for key in feature_keys:
        items = [stats[key] for stats in stats_list if key in stats and isinstance(stats[key], dict)]
        if not items:
            continue
        mins = np.stack([_as_numpy_1d(np.asarray(item["min"], dtype=np.float64)) for item in items], axis=0)
        maxs = np.stack([_as_numpy_1d(np.asarray(item["max"], dtype=np.float64)) for item in items], axis=0)
        means = np.stack([_as_numpy_1d(np.asarray(item["mean"], dtype=np.float64)) for item in items], axis=0)
        stds = np.stack([_as_numpy_1d(np.asarray(item["std"], dtype=np.float64)) for item in items], axis=0)
        counts = np.stack([_as_numpy_1d(np.asarray(item["count"], dtype=np.float64)) for item in items], axis=0)
        weights = counts[:, :1]
        safe_total_weight = np.maximum(weights.sum(axis=0), 1.0)

        total_mean = (means * weights).sum(axis=0) / safe_total_weight
        total_var = ((stds**2 + (means - total_mean) ** 2) * weights).sum(axis=0) / safe_total_weight

        aggregated[key] = {
            "min": np.min(mins, axis=0).tolist(),
            "max": np.max(maxs, axis=0).tolist(),
            "mean": _as_numpy_1d(total_mean).tolist(),
            "std": _as_numpy_1d(np.sqrt(np.maximum(total_var, 0.0))).tolist(),
            "count": [int(round(float(weights.sum())))],
        }

    return aggregated


def build_tasks_rows(prompt_to_task_index: Dict[str, int]) -> List[Dict[str, Any]]:
    rows: List[Tuple[int, str]] = []
    for prompt, idx in prompt_to_task_index.items():
        try:
            rows.append((int(idx), str(prompt)))
        except Exception:
            continue
    rows.sort(key=lambda item: (item[0], item[1]))
    return [{"task_index": idx, "task": prompt} for idx, prompt in rows]


def build_lerobot_v21_info(
    state: Dict[str, Any],
    args: argparse.Namespace,
    height: int,
    width: int,
) -> Dict[str, Any]:
    num_episodes = int(state.get("num_success", 0))
    total_frames = int(state.get("next_frame_index", 0))
    prompt_to_task_index = state.get("prompt_to_task_index", {})
    total_tasks = len(prompt_to_task_index) if isinstance(prompt_to_task_index, dict) else 0
    total_chunks = (num_episodes + args.chunk_size - 1) // args.chunk_size if num_episodes > 0 else 0

    video_info = {
        "video.height": int(height),
        "video.width": int(width),
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": float(args.fps),
        "video.channels": 3,
        "has_audio": False,
    }
    features = {
        "action": {"dtype": "float32", "shape": [7], "names": POSE7_FEATURE_NAMES},
        "observation.state": {"dtype": "float32", "shape": [7], "names": POSE7_FEATURE_NAMES},
        "observation.images.image": {
            "dtype": "video",
            "shape": [int(height), int(width), 3],
            "names": ["height", "width", "channels"],
            "info": dict(video_info),
        },
        "observation.images.wrist_image": {
            "dtype": "video",
            "shape": [int(height), int(width), 3],
            "names": ["height", "width", "channels"],
            "info": dict(video_info),
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    return {
        "codebase_version": "v2.1",
        "robot_type": args.robot_uid,
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": num_episodes,
        "total_chunks": total_chunks,
        "chunks_size": int(args.chunk_size),
        "fps": int(args.fps),
        "splits": {"train": f"0:{num_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def refresh_lerobot_v21_meta(
    meta_dir: Path,
    state: Dict[str, Any],
    args: argparse.Namespace,
    height: int,
    width: int,
    stats_payload: Dict[str, Any] | None = None,
) -> None:
    if args.strict_lerobot_v21 != 1:
        return
    tasks_rows = build_tasks_rows(state.get("prompt_to_task_index", {}))
    write_jsonl(meta_dir / "tasks.jsonl", tasks_rows)
    atomic_write_json(meta_dir / "info.json", build_lerobot_v21_info(state=state, args=args, height=height, width=width))
    if stats_payload is not None:
        atomic_write_json(meta_dir / "stats.json", stats_payload)
        atomic_write_json(meta_dir / "stat.json", stats_payload)
    else:
        for stats_path in [meta_dir / "stats.json", meta_dir / "stat.json"]:
            if not stats_path.is_file():
                atomic_write_json(stats_path, {})


def update_summary_file(
    summary_path: Path,
    args: argparse.Namespace,
    state: Dict[str, Any],
    metadata_rows: int,
    total_tasks_observed: int,
    batch_root: Path,
) -> None:
    payload = {
        "tag": args.tag,
        "epoch": args.epoch,
        "seed": args.seed,
        "run_id": args.run_id,
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "num_source_episodes_observed": metadata_rows,
        "repeat_per_episode": args.repeat_per_episode,
        "total_tasks_observed": total_tasks_observed,
        "next_task_id": int(state.get("next_task_id", 0)),
        "num_episodes": int(state.get("num_success", 0)),
        "num_failed": int(state.get("num_failed", 0)),
        "next_episode_index": int(state.get("next_episode_index", 0)),
        "next_frame_index": int(state.get("next_frame_index", 0)),
        "prompt_count": len(state.get("prompt_to_task_index", {})),
        "episodes_jsonl": str((batch_root / "meta" / "episodes.jsonl").relative_to(batch_root)),
        "rollout_rows_jsonl": str((batch_root / "meta" / "rollout_rows.jsonl").relative_to(batch_root)),
        "failures_jsonl": str((batch_root / "meta" / "failures.jsonl").relative_to(batch_root)),
        "tasks_jsonl": str((batch_root / "meta" / "tasks.jsonl").relative_to(batch_root)),
        "episodes_stats_jsonl": str((batch_root / "meta" / "episodes_stats.jsonl").relative_to(batch_root)),
        "info_json": str((batch_root / "meta" / "info.json").relative_to(batch_root)),
        "stats_json": str((batch_root / "meta" / "stats.json").relative_to(batch_root)),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(summary_path, payload)


def run_single_episode(
    args: argparse.Namespace,
    logger: logging.Logger,
    pipeline: Any,
    runtime,
    client: ExternalRobotInferenceClient,
    row: Dict[str, Any],
    chosen_index: int,
    dataset_base_path: str,
    dataset_metadata_path: str,
    action_stat_path: str,
    action_type: str,
    height: int,
    width: int,
    num_frames: int,
    num_history_frames: int,
    future_frames: int,
    stat_min: np.ndarray,
    stat_max: np.ndarray,
    negative_prompt_emb_path: str,
    attempt_id: int,
    attempt_seed: int,
    start_jitter: int,
    init_frame_sampling_mode: str,
    length_sampling_mode: str,
    reference_lengths_by_prompt: Dict[str, List[int]],
    max_length_multiplier: float,
    history_template_sampling: int = 0,
    history_anchor_stride: int = 8,
) -> Dict[str, Any]:
    episode_index = int(row["episode_index"])
    prompt = str(row.get("prompt", ""))
    prompt_emb_path = resolve_optional_path(row.get("prompt_emb"), dataset_base_path)
    if not prompt_emb_path or not os.path.isfile(prompt_emb_path):
        raise FileNotFoundError(f"Missing prompt_emb for episode {episode_index}: {prompt_emb_path}")
    if not negative_prompt_emb_path or not os.path.isfile(negative_prompt_emb_path):
        raise FileNotFoundError(f"Missing negative_prompt_emb: {negative_prompt_emb_path}")

    source_start_frame = int(row.get("start_frame", 0))
    source_end_frame = int(row.get("end_frame", source_start_frame + int(row.get("length", 1)) - 1))
    source_total_frames = source_end_frame - source_start_frame + 1
    if source_total_frames <= 0:
        raise ValueError(
            f"Invalid source frame range for metadata row {chosen_index}: "
            f"start={source_start_frame}, end={source_end_frame}"
        )

    max_start_for_history = source_end_frame - (num_history_frames - 1)
    if max_start_for_history < source_start_frame:
        raise ValueError(
            f"Invalid start range for history: start={source_start_frame}, end={source_end_frame}, "
            f"num_history_frames={num_history_frames}"
        )

    rng = random.Random(attempt_seed)
    if init_frame_sampling_mode == "random_within_source":
        jittered_start_frame = rng.randint(source_start_frame, max_start_for_history)
    else:
        jitter = rng.randint(-start_jitter, start_jitter) if start_jitter > 0 else 0
        # Dataset sample from metadata starts at source_start_frame, so we clamp lower bound for alignment.
        jittered_start_frame = max(
            source_start_frame,
            min(max_start_for_history, source_start_frame + jitter),
        )

    length_multiplier = 1.0
    sampled_reference_length: int | None = None
    length_fallback_used = 0
    target_length_mode = length_sampling_mode
    if length_sampling_mode == "reference_prompt":
        prompt_lengths = reference_lengths_by_prompt.get(prompt, [])
        if len(prompt_lengths) > 0:
            sampled_reference_length = int(prompt_lengths[rng.randrange(len(prompt_lengths))])
            target_frames = max(num_history_frames, sampled_reference_length)
            length_multiplier = float(target_frames) / float(max(source_total_frames, 1))
        else:
            length_fallback_used = 1
            target_length_mode = "source_multiplier_fallback"
            length_multiplier = rng.uniform(1.0, max_length_multiplier)
            target_frames = max(num_history_frames, int(np.ceil(source_total_frames * length_multiplier)))
    else:
        length_multiplier = rng.uniform(1.0, max_length_multiplier)
        target_frames = max(num_history_frames, int(np.ceil(source_total_frames * length_multiplier)))

    dataset = build_rollout_dataset(
        runtime,
        dataset_base_path=dataset_base_path,
        dataset_metadata_path=dataset_metadata_path,
        action_stat_path=action_stat_path,
        action_type=action_type,
        height=height,
        width=width,
        sample_index=chosen_index,
        num_frames=num_frames,
        num_history_frames=num_history_frames,
        history_template_sampling=history_template_sampling,
        history_anchor_stride=history_anchor_stride,
    )
    sample = dataset[0]
    gt_video_full = sample["video"].detach().cpu()  # (V, C, T, H, W), normalized
    gt_action_full = np.asarray(sample["action"], dtype=np.float32)[0]  # (T, 14), normalized

    if gt_video_full.shape[2] < num_history_frames:
        raise ValueError(f"Sample has {gt_video_full.shape[2]} frames < num_history_frames={num_history_frames}")

    local_start = jittered_start_frame - source_start_frame
    if local_start < 0 or local_start >= gt_video_full.shape[2]:
        raise ValueError(
            f"Jittered start out of sample range: local_start={local_start}, "
            f"sample_frames={gt_video_full.shape[2]}"
        )
    gt_video = gt_video_full[:, :, local_start:]
    gt_action = gt_action_full[local_start:]
    if gt_video.shape[2] < num_history_frames or gt_action.shape[0] < num_history_frames:
        raise ValueError(
            f"Insufficient history after jitter for row {chosen_index}, attempt={attempt_id}: "
            f"video_frames={gt_video.shape[2]}, action_frames={gt_action.shape[0]}, "
            f"required={num_history_frames}"
        )

    parquet_rel = row.get("action")
    parquet_path = parquet_rel if os.path.isabs(parquet_rel) else os.path.join(dataset_base_path, parquet_rel)
    history_end_frame = jittered_start_frame + num_history_frames - 1
    raw_pose7 = load_raw_pose7_sequence(parquet_path, jittered_start_frame, history_end_frame)
    if raw_pose7.shape[0] < num_history_frames:
        raise ValueError(
            f"Insufficient raw pose history for row {chosen_index}, attempt={attempt_id}: "
            f"pose_frames={raw_pose7.shape[0]}, required={num_history_frames}"
        )

    total_frames = int(target_frames)
    if total_frames < num_history_frames:
        raise ValueError(
            f"total_frames={total_frames} < num_history_frames={num_history_frames} "
            f"for row {chosen_index}, attempt={attempt_id}"
        )

    gt_video = gt_video[:, :, : min(gt_video.shape[2], total_frames)]
    gt_action = gt_action[:total_frames]
    init_history_actions = gt_action[:num_history_frames].copy()

    logger.info(
        "Episode src_idx=%s (metadata row=%s, attempt=%s), source_frames=%s, target_frames=%s, actual_frames=%s, "
        "start=%s->%s, init_frame_mode=%s, length_mode=%s, sampled_ref_len=%s, multiplier=%.4f, "
        "num_frames=%s, num_history=%s, future=%s",
        episode_index,
        chosen_index,
        attempt_id,
        source_total_frames,
        target_frames,
        total_frames,
        source_start_frame,
        jittered_start_frame,
        init_frame_sampling_mode,
        target_length_mode,
        sampled_reference_length,
        length_multiplier,
        num_frames,
        num_history_frames,
        future_frames,
    )

    history_frames: deque[torch.Tensor] = deque(maxlen=num_history_frames)
    history_states_raw7: deque[np.ndarray] = deque(maxlen=num_history_frames)
    history_actions_norm14: deque[np.ndarray] = deque(maxlen=num_history_frames)

    generated_frames: List[torch.Tensor] = []
    generated_states_raw7: List[np.ndarray] = []
    generated_actions_raw7: List[np.ndarray] = []
    generated_actions_norm14: List[np.ndarray] = []

    for t in range(num_history_frames):
        frame_t = gt_video[:, :, t]
        state_t = raw_pose7[t].astype(np.float32)
        action_t = init_history_actions[t].astype(np.float32)

        history_frames.append(frame_t)
        history_states_raw7.append(state_t)
        history_actions_norm14.append(action_t)

        generated_frames.append(frame_t)
        generated_states_raw7.append(state_t)
        generated_actions_raw7.append(state_t.copy())
        generated_actions_norm14.append(action_t)

    del gt_action

    vla_calls = 0
    while len(generated_frames) < total_frames:
        latest_frame = history_frames[-1]
        latest_state = history_states_raw7[-1]

        base_img = to_uint8_hwc(latest_frame[0])
        wrist_img = to_uint8_hwc(latest_frame[1] if latest_frame.shape[0] > 1 else latest_frame[0])
        action_chunk = get_vla_action_with_fallback(
            client=client,
            base_img=base_img,
            wrist_img=wrist_img,
            state7=latest_state,
            prompt=prompt,
        )
        future_raw7 = parse_vla_actions(action_chunk)
        vla_calls += 1

        if future_raw7.shape[0] != future_frames:
            raise ValueError(
                f"VLA future length mismatch: expected {future_frames}, got {future_raw7.shape[0]}"
            )
        if future_raw7.shape[1] != 7:
            raise ValueError(f"Expected VLA output width 7, got {future_raw7.shape[1]}")

        future_raw14 = state7_to_state14_copy(future_raw7)
        future_norm14 = normalize_bound(future_raw14, stat_min[None, :], stat_max[None, :]).astype(np.float32)

        history_action_arr = np.stack(list(history_actions_norm14), axis=0).astype(np.float32)
        action_cond = np.concatenate([history_action_arr, future_norm14], axis=0)[None, :, :]
        if action_cond.shape[1] != num_frames:
            raise ValueError(f"Action cond length mismatch: {action_cond.shape} vs expected (1,{num_frames},14)")

        input_video = torch.stack(list(history_frames), dim=2).to(dtype=torch.float32)
        pred_video = pipeline(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            prompt_emb=prompt_emb_path,
            negative_prompt_emb=negative_prompt_emb_path,
            input_video=input_video,
            action=action_cond,
            seed=attempt_seed,
            tiled=False,
            height=height,
            width=width,
            num_frames=num_frames,
            num_history_frames=num_history_frames,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
        )
        pred_video = pred_video.detach().cpu()
        pred_video[:, :, :num_history_frames] = input_video
        future_video = pred_video[:, :, num_history_frames:]

        remain = total_frames - len(generated_frames)
        append_count = min(remain, future_frames)
        for i in range(append_count):
            f = future_video[:, :, i]
            s_raw7 = future_raw7[i].astype(np.float32)
            a_norm14 = future_norm14[i].astype(np.float32)

            generated_frames.append(f)
            generated_states_raw7.append(s_raw7)
            generated_actions_raw7.append(s_raw7.copy())
            generated_actions_norm14.append(a_norm14)

            history_frames.append(f)
            history_states_raw7.append(s_raw7)
            history_actions_norm14.append(a_norm14)

        logger.info("Rollout progress: %s/%s frames", len(generated_frames), total_frames)

    pred_video_full = torch.stack(generated_frames, dim=2)
    pred_video_full = pred_video_full[:, :, :total_frames]
    gt_video = gt_video[:, :, :total_frames]

    return {
        "source_episode_index": episode_index,
        "metadata_row_index": chosen_index,
        "attempt_id": attempt_id,
        "source_start_frame": source_start_frame,
        "source_end_frame": source_end_frame,
        "jittered_start_frame": jittered_start_frame,
        "init_frame_mode": init_frame_sampling_mode,
        "length_multiplier": float(length_multiplier),
        "target_length_mode": target_length_mode,
        "sampled_reference_length": sampled_reference_length,
        "length_fallback_used": int(length_fallback_used),
        "target_frames": int(target_frames),
        "prompt": prompt,
        "prompt_emb": row.get("prompt_emb"),
        "total_frames": int(total_frames),
        "vla_calls": vla_calls,
        "pred_video_full": pred_video_full,
        "gt_video": gt_video,
        "state_raw7": np.stack(generated_states_raw7, axis=0).astype(np.float32),
        "action_raw7": np.stack(generated_actions_raw7, axis=0).astype(np.float32),
        "action_norm14": np.stack(generated_actions_norm14, axis=0).astype(np.float32),
    }


def claim_task_id(meta_dir: Path, total_tasks: int) -> int | None:
    def _mutate(state: Dict[str, Any]) -> int | None:
        next_task = int(state.get("next_task_id", 0))
        if next_task >= total_tasks:
            return None
        state["next_task_id"] = next_task + 1
        return next_task

    return with_locked_state(meta_dir, _mutate)


def reserve_output_ids(meta_dir: Path, prompt: str, frame_length: int) -> Tuple[int, int, int]:
    def _mutate(state: Dict[str, Any]) -> Tuple[int, int, int]:
        episode_index = int(state.get("next_episode_index", 0))
        state["next_episode_index"] = episode_index + 1

        prompt_map = state.setdefault("prompt_to_task_index", {})
        if prompt not in prompt_map:
            prompt_map[prompt] = len(prompt_map)
        task_index = int(prompt_map[prompt])

        frame_start = int(state.get("next_frame_index", 0))
        state["next_frame_index"] = frame_start + int(frame_length)
        return episode_index, task_index, frame_start

    return with_locked_state(meta_dir, _mutate)


def record_success(
    meta_dir: Path,
    episodes_path: Path,
    episodes_stats_path: Path,
    rollout_rows_path: Path,
    summary_path: Path,
    args: argparse.Namespace,
    batch_root: Path,
    episode_row: Dict[str, Any],
    episode_stats_row: Dict[str, Any] | None,
    rollout_row: Dict[str, Any],
    metadata_rows: int,
    total_tasks_observed: int,
    height: int,
    width: int,
) -> None:
    def _mutate(state: Dict[str, Any]) -> None:
        append_jsonl_row(episodes_path, episode_row)
        if args.strict_lerobot_v21 == 1 and episode_stats_row is not None:
            append_jsonl_row(episodes_stats_path, episode_stats_row)
        append_jsonl_row(rollout_rows_path, rollout_row)
        state["num_success"] = int(state.get("num_success", 0)) + 1
        refresh_lerobot_v21_meta(meta_dir=meta_dir, state=state, args=args, height=height, width=width)
        update_summary_file(
            summary_path=summary_path,
            args=args,
            state=state,
            metadata_rows=metadata_rows,
            total_tasks_observed=total_tasks_observed,
            batch_root=batch_root,
        )

    with_locked_state(meta_dir, _mutate)


def record_failure(
    meta_dir: Path,
    failures_path: Path,
    summary_path: Path,
    args: argparse.Namespace,
    batch_root: Path,
    failure_row: Dict[str, Any],
    metadata_rows: int,
    total_tasks_observed: int,
) -> None:
    def _mutate(state: Dict[str, Any]) -> None:
        append_jsonl_row(failures_path, failure_row)
        state["num_failed"] = int(state.get("num_failed", 0)) + 1
        update_summary_file(
            summary_path=summary_path,
            args=args,
            state=state,
            metadata_rows=metadata_rows,
            total_tasks_observed=total_tasks_observed,
            batch_root=batch_root,
        )

    with_locked_state(meta_dir, _mutate)


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    worker_seed = args.seed + args.worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    runtime_cfg, base_cfg = prepare_runtime_config(args, logger)
    manager = CheckpointPipelineManager(runtime_cfg, logger)
    checkpoints = manager.discover_checkpoints()
    pipeline = manager.initialize_pipeline(checkpoints)
    manager.update_checkpoint(checkpoints[0])

    dataset_base_path = args.dataset_base_path
    dataset_metadata_path = args.dataset_metadata_path
    action_stat_path = args.action_stat_path or base_cfg.get("action_stat_path")
    action_type = args.action_type or base_cfg.get("action_type", "state_pose")
    if not action_stat_path:
        raise ValueError("action_stat_path is required (pass --action_stat_path or provide in config.json).")

    grouped_cfg = load_json(resolve_paths_from_tag_epoch(args)[1])
    video_cfg = grouped_cfg.get("video", {})
    height = int(video_cfg.get("height", 240))
    width = int(video_cfg.get("width", 320))
    num_frames = int(video_cfg.get("num_frames", 17))
    num_history_frames = int(video_cfg.get("num_history_frames", 1))
    history_template_sampling = int(video_cfg.get("history_template_sampling", 0))
    history_anchor_stride = int(video_cfg.get("history_anchor_stride", 8))
    future_frames = num_frames - num_history_frames
    if future_frames <= 0:
        raise ValueError(f"Invalid frame setup: num_frames={num_frames}, num_history_frames={num_history_frames}")

    stat = load_json(action_stat_path)
    if action_type not in stat:
        raise KeyError(f"Missing action stats for type '{action_type}' in {action_stat_path}")
    stat_entry = stat[action_type]
    stat_min = np.asarray(stat_entry.get("p01", stat_entry.get("min")), dtype=np.float32)
    stat_max = np.asarray(stat_entry.get("p99", stat_entry.get("max")), dtype=np.float32)
    if stat_min.shape[0] != 14 or stat_max.shape[0] != 14:
        raise ValueError(f"Expected 14D stats for {action_type}, got {stat_min.shape} / {stat_max.shape}")

    negative_prompt_emb_path = resolve_optional_path(args.negative_prompt_emb, dataset_base_path)
    if not negative_prompt_emb_path or not os.path.isfile(negative_prompt_emb_path):
        raise FileNotFoundError(f"Missing negative_prompt_emb: {negative_prompt_emb_path}")

    length_reference_metadata_path = resolve_optional_path(args.length_reference_metadata_path, dataset_base_path)
    reference_lengths_by_prompt: Dict[str, List[int]] = {}
    if args.length_sampling_mode == "reference_prompt":
        if not length_reference_metadata_path or not os.path.isfile(length_reference_metadata_path):
            logger.warning(
                "length_sampling_mode=reference_prompt but reference metadata is unavailable: %s; "
                "falling back to source_multiplier when prompt has no reference lengths.",
                length_reference_metadata_path,
            )
        else:
            reference_lengths_by_prompt = load_prompt_length_map(length_reference_metadata_path, logger)
            logger.info(
                "Loaded reference prompt lengths from %s: prompts=%s",
                length_reference_metadata_path,
                len(reference_lengths_by_prompt),
            )

    client = ExternalRobotInferenceClient(host=args.vla_host, port=args.vla_port)
    if not client.ping():
        raise RuntimeError(f"Cannot ping VLA server at {args.vla_host}:{args.vla_port}")
    client.set_robot_uid(args.robot_uid)
    logger.info("Connected VLA server: %s:%s, robot_uid=%s", args.vla_host, args.vla_port, args.robot_uid)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_id = run_id
    base_output = (
        Path(args.output_dir)
        if args.output_dir
        else (REPO_ROOT / "Ckpt" / (args.tag or "manual") / f"epoch-{args.epoch or 'manual'}" / "vla_closed_loop_lerobot")
    )
    batch_root = base_output / f"batch_{run_id}"

    meta_dir = batch_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = meta_dir / "episodes.jsonl"
    episodes_stats_path = meta_dir / "episodes_stats.jsonl"
    rollout_rows_path = meta_dir / "rollout_rows.jsonl"
    failures_path = meta_dir / "failures.jsonl"
    summary_path = meta_dir / "rollout_summary.json"
    for p in [episodes_path, episodes_stats_path, rollout_rows_path, failures_path]:
        p.touch(exist_ok=True)

    saver = VideoSaver(fps=args.fps, quality=5)
    initial_metadata = read_metadata_rows(dataset_metadata_path, logger)
    metadata_rows_observed = len(initial_metadata)
    effective_rows = metadata_rows_observed if args.process_all_episodes == 1 else min(1, metadata_rows_observed)
    total_tasks_observed = effective_rows * args.repeat_per_episode

    def _init_state(state: Dict[str, Any]) -> None:
        refresh_lerobot_v21_meta(meta_dir=meta_dir, state=state, args=args, height=height, width=width, stats_payload={})
        update_summary_file(
            summary_path=summary_path,
            args=args,
            state=state,
            metadata_rows=metadata_rows_observed,
            total_tasks_observed=total_tasks_observed,
            batch_root=batch_root,
        )

    with_locked_state(meta_dir, _init_state)

    logger.info(
        "Worker %s/%s started: run_id=%s, process_all=%s, repeat=%s, start_jitter=%s, "
        "max_length_multiplier=%.3f, init_frame_sampling_mode=%s, length_sampling_mode=%s, "
        "length_reference_metadata_path=%s, chunk_size=%s, follow_metadata=%s, idle_timeout_sec=%.1f, strict_lerobot_v21=%s",
        args.worker_id,
        args.num_workers,
        run_id,
        args.process_all_episodes,
        args.repeat_per_episode,
        args.start_jitter,
        args.max_length_multiplier,
        args.init_frame_sampling_mode,
        args.length_sampling_mode,
        length_reference_metadata_path,
        args.chunk_size,
        args.follow_metadata,
        args.idle_timeout_sec,
        args.strict_lerobot_v21,
    )

    idle_start = time.monotonic()
    while True:
        metadata = read_metadata_rows(dataset_metadata_path, logger)
        metadata_rows_observed = len(metadata)
        effective_rows = metadata_rows_observed if args.process_all_episodes == 1 else min(1, metadata_rows_observed)
        total_tasks = effective_rows * args.repeat_per_episode
        total_tasks_observed = max(total_tasks_observed, total_tasks)

        task_id = claim_task_id(meta_dir, total_tasks=total_tasks) if total_tasks > 0 else None
        if task_id is None:
            if args.follow_metadata == 1:
                idle_elapsed = time.monotonic() - idle_start
                if idle_elapsed >= args.idle_timeout_sec:
                    logger.info(
                        "Worker %s idle timeout reached (%.1fs). Stop polling metadata.",
                        args.worker_id,
                        idle_elapsed,
                    )
                    break
                time.sleep(args.poll_interval_sec)
                continue
            logger.info("Worker %s no remaining tasks, exiting.", args.worker_id)
            break

        idle_start = time.monotonic()
        chosen_index = task_id // args.repeat_per_episode
        attempt_id = task_id % args.repeat_per_episode
        if chosen_index >= effective_rows:
            failure_row = {
                "worker_id": args.worker_id,
                "task_id": task_id,
                "metadata_row_index": chosen_index,
                "attempt_id": attempt_id,
                "stage": "scheduler",
                "error": f"Task index out of range: chosen_index={chosen_index}, effective_rows={effective_rows}",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            record_failure(
                meta_dir=meta_dir,
                failures_path=failures_path,
                summary_path=summary_path,
                args=args,
                batch_root=batch_root,
                failure_row=failure_row,
                metadata_rows=metadata_rows_observed,
                total_tasks_observed=total_tasks_observed,
            )
            continue

        row = metadata[chosen_index]
        attempt_seed = args.seed + task_id
        try:
            result = run_single_episode(
                args=args,
                logger=logger,
                pipeline=pipeline,
                runtime=runtime_cfg.runtime,
                client=client,
                row=row,
                chosen_index=chosen_index,
                dataset_base_path=dataset_base_path,
                dataset_metadata_path=dataset_metadata_path,
                action_stat_path=action_stat_path,
                action_type=action_type,
                height=height,
                width=width,
                num_frames=num_frames,
                num_history_frames=num_history_frames,
                future_frames=future_frames,
                stat_min=stat_min,
                stat_max=stat_max,
                negative_prompt_emb_path=negative_prompt_emb_path,
                attempt_id=attempt_id,
                attempt_seed=attempt_seed,
                start_jitter=args.start_jitter,
                init_frame_sampling_mode=args.init_frame_sampling_mode,
                length_sampling_mode=args.length_sampling_mode,
                reference_lengths_by_prompt=reference_lengths_by_prompt,
                max_length_multiplier=args.max_length_multiplier,
                history_template_sampling=history_template_sampling,
                history_anchor_stride=history_anchor_stride,
            )
        except Exception as exc:
            logger.warning(
                "Worker %s failed task=%s row=%s attempt=%s at inference stage: %s",
                args.worker_id,
                task_id,
                chosen_index,
                attempt_id,
                exc,
            )
            failure_row = {
                "worker_id": args.worker_id,
                "task_id": task_id,
                "metadata_row_index": chosen_index,
                "attempt_id": attempt_id,
                "stage": "inference",
                "error": str(exc),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            record_failure(
                meta_dir=meta_dir,
                failures_path=failures_path,
                summary_path=summary_path,
                args=args,
                batch_root=batch_root,
                failure_row=failure_row,
                metadata_rows=metadata_rows_observed,
                total_tasks_observed=total_tasks_observed,
            )
            continue

        prompt = result["prompt"]
        frame_length = int(result["total_frames"])
        batch_episode_index, task_index, global_index_start = reserve_output_ids(
            meta_dir=meta_dir,
            prompt=prompt,
            frame_length=frame_length,
        )
        chunk_id = batch_episode_index // args.chunk_size
        chunk_name = f"chunk-{chunk_id:03d}"

        data_chunk_dir = batch_root / "data" / chunk_name
        image_video_dir = batch_root / "videos" / chunk_name / "observation.images.image"
        wrist_video_dir = batch_root / "videos" / chunk_name / "observation.images.wrist_image"
        comparison_video_dir = batch_root / "videos" / chunk_name / "comparison"
        for p in [data_chunk_dir, image_video_dir, wrist_video_dir, comparison_video_dir]:
            p.mkdir(parents=True, exist_ok=True)

        ep_name = f"episode_{batch_episode_index:06d}"
        pred_video_full = result["pred_video_full"]
        gt_video = result["gt_video"]

        image_frames = [to_uint8_hwc(pred_video_full[0, :, t]) for t in range(pred_video_full.shape[2])]
        wrist_view_idx = 1 if pred_video_full.shape[0] > 1 else 0
        wrist_frames = [to_uint8_hwc(pred_video_full[wrist_view_idx, :, t]) for t in range(pred_video_full.shape[2])]

        image_video_rel = f"videos/{chunk_name}/observation.images.image/{ep_name}.mp4"
        wrist_video_rel = f"videos/{chunk_name}/observation.images.wrist_image/{ep_name}.mp4"
        comparison_video_rel = f"videos/{chunk_name}/comparison/{ep_name}.mp4"
        action_rel = f"data/{chunk_name}/{ep_name}.parquet"

        try:
            save_video(image_frames, str(batch_root / image_video_rel), fps=args.fps, quality=5)
            save_video(wrist_frames, str(batch_root / wrist_video_rel), fps=args.fps, quality=5)
            saver.save_comparison(
                original_video=gt_video,
                predicted_video=pred_video_full,
                output_dir=comparison_video_dir,
                video_name=f"{ep_name}.mp4",
                length_mode="pad_to_pred_black_gt",
            )

            write_episode_parquet(
                output_path=batch_root / action_rel,
                action_raw7=result["action_raw7"],
                state_raw7=result["state_raw7"],
                fps=args.fps,
                batch_episode_index=batch_episode_index,
                global_index_start=global_index_start,
                task_index=task_index,
            )
        except Exception as exc:
            logger.warning(
                "Worker %s failed task=%s row=%s attempt=%s at output stage: %s",
                args.worker_id,
                task_id,
                chosen_index,
                attempt_id,
                exc,
            )
            failure_row = {
                "worker_id": args.worker_id,
                "task_id": task_id,
                "metadata_row_index": chosen_index,
                "attempt_id": attempt_id,
                "stage": "output",
                "episode_index": batch_episode_index,
                "error": str(exc),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            record_failure(
                meta_dir=meta_dir,
                failures_path=failures_path,
                summary_path=summary_path,
                args=args,
                batch_root=batch_root,
                failure_row=failure_row,
                metadata_rows=metadata_rows_observed,
                total_tasks_observed=total_tasks_observed,
            )
            continue

        episode_row = {
            "episode_index": batch_episode_index,
            "source_episode_index": result["source_episode_index"],
            "attempt_id": result["attempt_id"],
            "source_start_frame": result["source_start_frame"],
            "source_end_frame": result["source_end_frame"],
            "jittered_start_frame": result["jittered_start_frame"],
            "init_frame_mode": result["init_frame_mode"],
            "length_multiplier": result["length_multiplier"],
            "target_length_mode": result["target_length_mode"],
            "sampled_reference_length": result["sampled_reference_length"],
            "length_fallback_used": result["length_fallback_used"],
            "target_frames": result["target_frames"],
            "length": int(result["total_frames"]),
            "start_frame": 0,
            "end_frame": int(result["total_frames"] - 1),
            "video": [image_video_rel, wrist_video_rel],
            "action": action_rel,
            "prompt": prompt,
            "prompt_emb": result.get("prompt_emb"),
            "comparison_video": comparison_video_rel,
        }

        rollout_row = {
            "episode_index": batch_episode_index,
            "source_episode_index": result["source_episode_index"],
            "attempt_id": result["attempt_id"],
            "chunk_id": chunk_id,
            "worker_id": args.worker_id,
            "task_id": task_id,
            "task_index": task_index,
            "metadata_row_index": result["metadata_row_index"],
            "jittered_start_frame": result["jittered_start_frame"],
            "init_frame_mode": result["init_frame_mode"],
            "length_multiplier": result["length_multiplier"],
            "target_length_mode": result["target_length_mode"],
            "sampled_reference_length": result["sampled_reference_length"],
            "length_fallback_used": result["length_fallback_used"],
            "target_frames": result["target_frames"],
            "total_frames": result["total_frames"],
            "vla_calls": result["vla_calls"],
        }
        episode_stats_row = (
            build_episode_stats_row(
                action_raw7=result["action_raw7"],
                state_raw7=result["state_raw7"],
                fps=args.fps,
                batch_episode_index=batch_episode_index,
                global_index_start=global_index_start,
                task_index=task_index,
            )
            if args.strict_lerobot_v21 == 1
            else None
        )

        record_success(
            meta_dir=meta_dir,
            episodes_path=episodes_path,
            episodes_stats_path=episodes_stats_path,
            rollout_rows_path=rollout_rows_path,
            summary_path=summary_path,
            args=args,
            batch_root=batch_root,
            episode_row=episode_row,
            episode_stats_row=episode_stats_row,
            rollout_row=rollout_row,
            metadata_rows=metadata_rows_observed,
            total_tasks_observed=total_tasks_observed,
            height=height,
            width=width,
        )
        logger.info(
            "Worker %s saved %s (chunk=%s, task=%s, row=%s, attempt=%s)",
            args.worker_id,
            ep_name,
            chunk_name,
            task_id,
            chosen_index,
            attempt_id,
        )

    def _finalize(state: Dict[str, Any]) -> None:
        latest_metadata_rows = len(read_metadata_rows(dataset_metadata_path, logger))
        latest_total_tasks = (
            latest_metadata_rows if args.process_all_episodes == 1 else min(1, latest_metadata_rows)
        ) * args.repeat_per_episode
        if args.strict_lerobot_v21 == 1:
            stats_rows = read_metadata_rows(str(episodes_stats_path), logger)
            aggregate_stats_payload = aggregate_episode_stats(stats_rows)
            refresh_lerobot_v21_meta(
                meta_dir=meta_dir,
                state=state,
                args=args,
                height=height,
                width=width,
                stats_payload=aggregate_stats_payload,
            )
        update_summary_file(
            summary_path=summary_path,
            args=args,
            state=state,
            metadata_rows=latest_metadata_rows,
            total_tasks_observed=max(total_tasks_observed, latest_total_tasks),
            batch_root=batch_root,
        )

    with_locked_state(meta_dir, _finalize)
    logger.info("Worker %s finished. Shared output: %s", args.worker_id, batch_root.resolve())


if __name__ == "__main__":
    main()
