#!/usr/bin/env python
import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from atm.atm_inference import ATMInference
from atm.dataloader import RoboCoinATMActionDataset
from diffsynth.core.data.operators import LoadCobotAction
from diffsynth.core.loader import load_wan_checkpoint_into_pipeline
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.pipelines.wan_video_data import build_wan_video_dataset
from diffsynth.utils.data import save_video

from inference_support import (
    FrameConverter,
    VideoSaver,
    build_wan_inference_config,
    load_flat_config_defaults,
    resolve_optional_path,
)


DEFAULT_NEGATIVE_PROMPT = (
    "The video is not of a high quality, it has a low resolution. "
    "Watermark present in each frame. The background is solid. "
    "Strange body and strange trajectory. Distortion"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ATM-conditioned WAN video inference.")
    parser.add_argument(
        "--atm_ckpt_path",
        type=str,
        default="/data_jbx/Codes/ATM/results/track_transformer/0409_realbot_track_transformer_001B_action_bs_16_grad_acc_4_numtrack_256_ep1001_0047/model_best.ckpt",
    )
    parser.add_argument(
        "--wan_ckpt_path",
        type=str,
        default=str(PROJECT_ROOT / "Ckpt/wan_atm001B_480_640_202604101603/epoch-25/epoch-25.safetensors"),
    )
    parser.add_argument("--model_paths", type=str, default="/data1/modelscope/models/Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--load_modules", type=str, default="dit,text:emb,vae,image:off,action:noise")
    parser.add_argument("--dataset_base_path", type=str, default="/data_jbx/Datasets/Realbot/4_4_four_tasks_wan")
    parser.add_argument(
        "--dataset_metadata_path",
        type=str,
        default="/data_jbx/Datasets/Realbot/4_4_four_tasks_wan/meta/episodes_train.track_bert.jsonl",
    )
    parser.add_argument("--action_stat_path", type=str, default="/data_jbx/Datasets/Realbot/4_4_four_tasks_wan/meta/stat.json")
    parser.add_argument("--action_type", type=str, default="action_pose")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--num_history_frames", type=int, default=1)
    parser.add_argument("--spatial_division_factor", type=int, default=32)
    parser.add_argument("--resize_mode", type=str, default="fit")
    parser.add_argument("--max_pixels", type=int, default=4096 * 4096)
    parser.add_argument("--history_template_sampling", type=int, default=0)
    parser.add_argument("--history_anchor_stride", type=int, default=8)
    parser.add_argument("--data_file_keys", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--negative_prompt_emb", type=str, default="prompt_emb/neg_prompt.pt")
    parser.add_argument("--sample_indices", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--output_dir", type=str, default="results/inference_wan_atm")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=5)
    return parser.parse_args()


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("inference_wan_atm")


def set_global_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata_records(metadata_path: str) -> List[Dict[str, Any]]:
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    if path.suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise TypeError(f"Expected metadata JSON list, got {type(data).__name__}")
        return data
    raise ValueError(f"Unsupported metadata format: {path}")


def merge_config_values(args: argparse.Namespace) -> Dict[str, Any]:
    merged = dict(load_flat_config_defaults(args.wan_ckpt_path))
    for key, value in vars(args).items():
        if value is not None:
            merged[key] = value
    if not merged.get("load_modules"):
        merged["load_modules"] = "dit,text:emb,vae,image:off,action:noise"
    if not merged.get("action_type"):
        merged["action_type"] = "action_pose"
    if not merged.get("num_frames"):
        merged["num_frames"] = 17
    if not merged.get("num_history_frames"):
        merged["num_history_frames"] = 1
    if not merged.get("resize_mode"):
        merged["resize_mode"] = "fit"
    if not merged.get("max_pixels"):
        merged["max_pixels"] = 4096 * 4096
    if not merged.get("spatial_division_factor"):
        merged["spatial_division_factor"] = 32
    if not merged.get("history_anchor_stride"):
        merged["history_anchor_stride"] = 8
    if "history_template_sampling" not in merged:
        merged["history_template_sampling"] = 0
    return merged


def resolve_required_paths(values: Dict[str, Any]) -> None:
    required_keys = ("model_paths", "dataset_base_path", "dataset_metadata_path", "action_stat_path")
    missing = [key for key in required_keys if not values.get(key)]
    if missing:
        raise ValueError(f"Missing required arguments/config values: {missing}")


def normalize_view_paths(video_field: Any) -> List[str]:
    if isinstance(video_field, (list, tuple)):
        return [str(item) for item in video_field]
    if isinstance(video_field, str):
        return [video_field]
    raise TypeError(f"Unsupported video field type: {type(video_field).__name__}")


def resolve_data_path(base_path: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.join(base_path, value)


def fit_sequence_length(sequence, target_length: int):
    if sequence is None:
        return None
    current_length = int(sequence.shape[1])
    if current_length == target_length:
        return sequence
    if current_length > target_length:
        return sequence[:, :target_length]
    if current_length <= 0:
        raise ValueError("Cannot pad an empty sequence.")
    pad_frames = target_length - current_length
    if isinstance(sequence, torch.Tensor):
        pad = sequence[:, -1:, ...].repeat(1, pad_frames, *([1] * (sequence.ndim - 2)))
        return torch.cat([sequence, pad], dim=1)
    pad = np.repeat(sequence[:, -1:, ...], repeats=pad_frames, axis=1)
    return np.concatenate([sequence, pad], axis=1)


def resolve_prompt_inputs(
    sample: Dict[str, Any],
    *,
    config,
    negative_prompt_emb: str | None,
) -> tuple[str, str | None, str | None]:
    prompt = str(sample.get("prompt") or "")
    prompt_emb = sample.get("prompt_emb")
    if prompt_emb:
        prompt_emb = resolve_optional_path(prompt_emb, config.dataset_base_path)
    if config.text_mode == "emb":
        if not prompt_emb or not os.path.isfile(prompt_emb):
            raise FileNotFoundError(
                f"Missing `prompt_emb` for sample when WAN text mode is `emb`: {prompt_emb}"
            )
        if not negative_prompt_emb or not os.path.isfile(negative_prompt_emb):
            raise FileNotFoundError(
                f"Missing `negative_prompt_emb` for WAN text mode `emb`: {negative_prompt_emb}"
            )
    else:
        if prompt_emb and not os.path.isfile(prompt_emb):
            prompt_emb = None
        if negative_prompt_emb and not os.path.isfile(negative_prompt_emb):
            negative_prompt_emb = None
    return prompt, prompt_emb, negative_prompt_emb


def build_wan_dataset(config):
    keys = [key for key in config.runtime.data_file_keys if key != "track"]
    return build_wan_video_dataset(
        config.runtime,
        base_path=config.dataset_base_path,
        metadata_path=config.metadata_path,
        height=config.values.get("height"),
        width=config.values.get("width"),
        num_frames=int(config.num_frames),
        num_history_frames=int(config.num_history_frames),
        repeat=1,
        resize_mode=config.resize_mode,
        max_pixels=int(config.max_pixels),
        data_file_keys=keys,
        action_stat_path=config.action_stat_path,
        action_type=config.action_type,
        track_num_points_per_view=1,
        history_template_sampling=config.history_template_sampling,
        history_anchor_stride=int(config.history_anchor_stride),
        height_division_factor=int(config.spatial_division_factor),
        width_division_factor=int(config.spatial_division_factor),
        time_division_factor=4,
        time_division_remainder=1,
    )


def build_state_pose_loader(dataset_base_path: str, stat_path: str, num_frames: int) -> LoadCobotAction:
    with open(stat_path, "r", encoding="utf-8") as handle:
        stats = json.load(handle)
    return LoadCobotAction(
        base_path=dataset_base_path,
        action_type="state_pose",
        stat=stats,
        num_frames=num_frames,
        time_division_factor=4,
        time_division_remainder=1,
    )


def load_task_embedding(entry: Dict[str, Any], dataset_base_path: str) -> torch.Tensor:
    task_path = entry.get("prompt_embed_bert")
    if not task_path:
        raise KeyError("Metadata entry is missing `prompt_embed_bert`, which ATM inference requires.")
    resolved_path = resolve_data_path(dataset_base_path, str(task_path))
    task_emb = torch.load(resolved_path, map_location="cpu")
    task_emb = torch.as_tensor(task_emb, dtype=torch.float32)
    if task_emb.ndim == 1:
        return task_emb.unsqueeze(0)
    if task_emb.ndim == 2 and task_emb.shape[0] == 1:
        return task_emb
    raise ValueError(f"Unexpected task embedding shape: {tuple(task_emb.shape)}")


def load_state_pose_condition(
    entry: Dict[str, Any],
    *,
    dataset_base_path: str,
    state_loader: LoadCobotAction,
    target_steps: int,
) -> torch.Tensor:
    action_rel = entry.get("action")
    if not action_rel:
        raise KeyError("Metadata entry is missing `action`.")
    parquet_path = resolve_data_path(dataset_base_path, str(action_rel))
    total_rows = int(pq.read_metadata(parquet_path).num_rows)
    start_frame = int(entry["start_frame"])
    end_frame = min(start_frame + target_steps, total_rows)
    frame_indices = list(range(start_frame, end_frame))
    if not frame_indices:
        raise ValueError(
            f"No ATM state frames available for sample with start_frame={start_frame} in {parquet_path}"
        )

    # ATM consumes normalized absolute state_pose over 81 steps.
    state = state_loader(parquet_path, frame_indices=frame_indices)
    state = torch.as_tensor(state, dtype=torch.float32)
    if state.ndim != 3 or state.shape[0] != 1:
        raise ValueError(f"Unexpected ATM state shape: {tuple(state.shape)}")
    if int(state.shape[1]) < target_steps:
        pad_steps = target_steps - int(state.shape[1])
        pad = torch.zeros((1, pad_steps, int(state.shape[2])), dtype=state.dtype)
        state = torch.cat([state, pad], dim=1)
    return state


def load_atm_view_frames(
    video_path: str,
    *,
    start_frame: int,
    frame_stack: int,
    img_size: Sequence[int],
) -> torch.Tensor:
    img_start_idx = max(start_frame + 1 - frame_stack, 0)
    img_end_idx = start_frame + 1
    frame_indices = np.arange(img_start_idx, img_end_idx, dtype=np.int64)
    frame_array = RoboCoinATMActionDataset._load_video_frames(video_path, frame_indices=frame_indices)
    frames = torch.from_numpy(frame_array).float().permute(0, 3, 1, 2).contiguous()
    if int(frames.shape[0]) < frame_stack:
        pad = torch.zeros((frame_stack - int(frames.shape[0]), *frames.shape[1:]), dtype=frames.dtype)
        frames = torch.cat([pad, frames], dim=0)
    target_h, target_w = int(img_size[0]), int(img_size[1])
    if tuple(frames.shape[-2:]) != (target_h, target_w):
        frames = F.interpolate(frames, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return frames


def run_multiview_atm(
    entry: Dict[str, Any],
    *,
    dataset_base_path: str,
    atm_engine: ATMInference,
    state_loader: LoadCobotAction,
) -> torch.Tensor:
    task_emb = load_task_embedding(entry, dataset_base_path)
    state = load_state_pose_condition(
        entry,
        dataset_base_path=dataset_base_path,
        state_loader=state_loader,
        target_steps=int(atm_engine.num_track_ts),
    )
    start_frame = int(entry["start_frame"])
    track_by_view: List[torch.Tensor] = []
    for view_idx, video_rel in enumerate(normalize_view_paths(entry["video"])):
        video_path = resolve_data_path(dataset_base_path, video_rel)
        history_frames = load_atm_view_frames(
            video_path,
            start_frame=start_frame,
            frame_stack=int(atm_engine.frame_stack),
            img_size=atm_engine.img_size,
        )
        pred = atm_engine.infer(
            history_frames.unsqueeze(0),
            task_emb,
            state,
            track=None,
        )
        pred = pred.detach().to(dtype=torch.float32).cpu()
        if pred.ndim != 4 or pred.shape[0] != 1:
            raise ValueError(f"Unexpected ATM prediction shape for view {view_idx}: {tuple(pred.shape)}")
        track_by_view.append(pred)

    if not track_by_view:
        raise ValueError("No ATM tracks were produced because the sample has no video views.")
    return torch.cat(track_by_view, dim=2)


def compose_multiview_panel(video: torch.Tensor, converter: FrameConverter) -> np.ndarray:
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise TypeError("Expected `video` tensor with shape (V,C,T,H,W).")
    video = video.detach().to(dtype=torch.float32).cpu()
    num_views = int(video.shape[0])
    num_frames = int(video.shape[2])
    frames: List[np.ndarray] = []
    for frame_idx in range(num_frames):
        panel_rows: List[np.ndarray] = []
        for view_idx in range(num_views):
            frame = converter.ensure_rgb(converter.to_uint8(video[view_idx, :, frame_idx]))
            panel_rows.append(frame)
        frames.append(np.vstack(panel_rows))
    return np.asarray(frames)


def save_generated_panel_video(video: torch.Tensor, output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_frames = compose_multiview_panel(video, FrameConverter())
    save_video(panel_frames, str(output_path), fps=fps, quality=5, show_progress=False)


def build_wan_pipeline(config, *, device: str, num_views: int, track_points_per_view: int, ckpt_path: str | None):
    if not config.action_enabled:
        raise ValueError("The selected WAN modules do not enable the action/track branch.")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=config.build_model_configs(offload_device="cpu"),
        tokenizer_config=config.build_tokenizer_config(),
        modules=list(config.modules),
        track_num_points_per_view=track_points_per_view,
        num_track_views=num_views,
        track_noise_std=0.0,
    )
    if ckpt_path:
        load_wan_checkpoint_into_pipeline(
            pipe,
            ckpt_path,
            torch_dtype=torch.bfloat16,
            device="cpu",
            message_prefix="Loading WAN checkpoint",
        )
    return pipe


def validate_sample_indices(sample_indices: Iterable[int], dataset_size: int) -> List[int]:
    normalized = [int(index) for index in sample_indices]
    invalid = [index for index in normalized if index < 0 or index >= dataset_size]
    if invalid:
        raise IndexError(f"Sample indices out of range: {invalid} (dataset size: {dataset_size})")
    return normalized


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    set_global_seed(int(args.seed))

    merged_values = merge_config_values(args)
    resolve_required_paths(merged_values)
    config = build_wan_inference_config(merged_values)
    metadata_records = load_metadata_records(config.metadata_path)
    sample_indices = validate_sample_indices(args.sample_indices, len(metadata_records))

    dataset = build_wan_dataset(config)
    if len(dataset) != len(metadata_records):
        raise ValueError(
            f"WAN dataset size mismatch: dataset={len(dataset)}, metadata={len(metadata_records)}"
        )

    first_sample = dataset[sample_indices[0]]
    first_video = first_sample["video"]
    if not isinstance(first_video, torch.Tensor) or first_video.ndim != 5:
        raise TypeError(f"Unexpected WAN video sample shape: {type(first_video)}")
    num_views = int(first_video.shape[0])

    atm_engine = ATMInference(args.atm_ckpt_path, device=args.device)
    if int(atm_engine.num_track_ids) <= 0:
        raise ValueError(f"ATM checkpoint has invalid num_track_ids={atm_engine.num_track_ids}")
    if int(atm_engine.num_track_ts) < int(config.num_frames):
        raise ValueError(
            f"ATM checkpoint predicts only {atm_engine.num_track_ts} steps, but WAN needs {config.num_frames} frames."
        )
    state_loader = build_state_pose_loader(
        dataset_base_path=config.dataset_base_path,
        stat_path=config.action_stat_path,
        num_frames=int(atm_engine.num_track_ts),
    )
    pipe = build_wan_pipeline(
        config,
        device=args.device,
        num_views=num_views,
        track_points_per_view=int(atm_engine.num_track_ids),
        ckpt_path=args.wan_ckpt_path,
    )

    negative_prompt_emb = resolve_optional_path(args.negative_prompt_emb, config.dataset_base_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_saver = VideoSaver(fps=int(args.fps), quality=5, show_progress=False)

    logger.info(
        "Running %s samples with %s WAN views, ATM steps=%s, WAN frames=%s",
        len(sample_indices),
        num_views,
        atm_engine.num_track_ts,
        config.num_frames,
    )

    for sample_idx in sample_indices:
        entry = metadata_records[sample_idx]
        sample = dataset[sample_idx]
        video = sample["video"]
        if int(video.shape[0]) != num_views:
            raise ValueError(
                f"Sample {sample_idx} has {int(video.shape[0])} views, expected {num_views}."
            )

        prompt, prompt_emb, negative_prompt_emb_path = resolve_prompt_inputs(
            sample,
            config=config,
            negative_prompt_emb=negative_prompt_emb,
        )
        track_81 = run_multiview_atm(
            entry,
            dataset_base_path=config.dataset_base_path,
            atm_engine=atm_engine,
            state_loader=state_loader,
        )
        expected_track_points = num_views * int(atm_engine.num_track_ids)
        if int(track_81.shape[2]) != expected_track_points:
            raise ValueError(
                f"ATM merged track points mismatch for sample {sample_idx}: "
                f"expected {expected_track_points}, got {int(track_81.shape[2])}."
            )
        track_17 = track_81[:, : int(config.num_frames)].contiguous()
        action = fit_sequence_length(sample.get("action"), int(config.num_frames))
        input_video = video[:, :, : int(config.num_history_frames)]

        logger.info(
            "Sample %s | episode=%s | input_video=%s | action=%s | track81=%s | track17=%s",
            sample_idx,
            entry.get("episode_index"),
            tuple(input_video.shape),
            None if action is None else tuple(action.shape),
            tuple(track_81.shape),
            tuple(track_17.shape),
        )

        predicted_video = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            prompt_emb=prompt_emb,
            negative_prompt_emb=negative_prompt_emb_path,
            input_video=input_video,
            action=action,
            track=track_17,
            seed=int(args.seed),
            tiled=False,
            height=int(video.shape[-2]),
            width=int(video.shape[-1]),
            num_frames=int(config.num_frames),
            num_history_frames=int(config.num_history_frames),
            cfg_scale=float(args.cfg_scale),
            num_inference_steps=int(args.num_inference_steps),
            sigma_shift=float(args.sigma_shift),
            progress_bar_cmd=tqdm,
        )
        if not isinstance(predicted_video, torch.Tensor) or predicted_video.ndim != 5:
            raise TypeError(f"Pipeline output must be (V,C,T,H,W), got {type(predicted_video)}")
        predicted_video = predicted_video.detach().cpu()[:, :, : int(config.num_frames)]
        history_frames = min(
            int(config.num_history_frames),
            int(predicted_video.shape[2]),
            int(video.shape[2]),
        )
        if history_frames > 0:
            predicted_video[:, :, :history_frames] = video[:, :, :history_frames]

        episode_index = int(entry.get("episode_index", sample_idx))
        generated_name = f"pred_s{sample_idx:06d}_ep{episode_index}.mp4"
        comparison_name = f"compare_s{sample_idx:06d}_ep{episode_index}.mp4"
        generated_path = output_dir / generated_name
        comparison_path = video_saver.save_comparison(
            video[:, :, : int(predicted_video.shape[2])],
            predicted_video,
            output_dir,
            comparison_name,
        )
        save_generated_panel_video(predicted_video, generated_path, fps=int(args.fps))
        logger.info("Saved generated video: %s", generated_path.resolve())
        logger.info("Saved comparison video: %s", comparison_path.resolve())


if __name__ == "__main__":
    main()
