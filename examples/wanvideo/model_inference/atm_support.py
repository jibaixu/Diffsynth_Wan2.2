from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


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


def resolve_data_path(base_path: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.join(base_path, value)


def _derive_prompt_embed_bert_path(prompt_emb_path: str) -> str:
    path = Path(str(prompt_emb_path))
    if "bert" in path.parts:
        return str(path)
    return str(path.parent / "bert" / path.name)


def resolve_task_embedding_path(entry: Dict[str, Any], dataset_base_path: str) -> str:
    candidates: List[str] = []
    prompt_embed_bert = entry.get("prompt_embed_bert")
    if prompt_embed_bert:
        candidates.append(resolve_data_path(dataset_base_path, str(prompt_embed_bert)))

    prompt_emb = entry.get("prompt_emb")
    if prompt_emb:
        candidates.append(resolve_data_path(dataset_base_path, _derive_prompt_embed_bert_path(str(prompt_emb))))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "Missing ATM task embedding. Checked candidates: "
        + ", ".join(candidates if candidates else ["<none>"])
    )


def load_task_embedding(
    entry: Dict[str, Any],
    dataset_base_path: str,
    *,
    cache: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    resolved_path = resolve_task_embedding_path(entry, dataset_base_path)
    if cache is not None and resolved_path in cache:
        return cache[resolved_path]

    task_emb = torch.load(resolved_path, map_location="cpu")
    task_emb = torch.as_tensor(task_emb, dtype=torch.float32)
    if task_emb.ndim == 1:
        task_emb = task_emb.unsqueeze(0)
    elif task_emb.ndim != 2 or task_emb.shape[0] != 1:
        raise ValueError(f"Unexpected task embedding shape: {tuple(task_emb.shape)}")

    if cache is not None:
        cache[resolved_path] = task_emb
    return task_emb


def build_state_pose_loader(dataset_base_path: str, stat_path: str, num_frames: int):
    from diffsynth.core.data.operators import LoadCobotAction

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


def load_state_pose_condition(
    entry: Dict[str, Any],
    *,
    dataset_base_path: str,
    state_loader,
    target_steps: int,
    start_frame: int,
    row_count_cache: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    import pyarrow.parquet as pq

    action_rel = entry.get("action")
    if not action_rel:
        raise KeyError("Metadata entry is missing `action`.")
    parquet_path = resolve_data_path(dataset_base_path, str(action_rel))
    if row_count_cache is not None and parquet_path in row_count_cache:
        total_rows = row_count_cache[parquet_path]
    else:
        total_rows = int(pq.read_metadata(parquet_path).num_rows)
        if row_count_cache is not None:
            row_count_cache[parquet_path] = total_rows

    start_frame = max(0, int(start_frame))
    end_frame = min(start_frame + int(target_steps), total_rows)
    frame_indices = list(range(start_frame, end_frame))
    if not frame_indices:
        raise ValueError(
            f"No ATM state frames available for sample with start_frame={start_frame} in {parquet_path}"
        )

    state = state_loader(parquet_path, frame_indices=frame_indices)
    state = torch.as_tensor(state, dtype=torch.float32)
    if state.ndim != 3 or state.shape[0] != 1:
        raise ValueError(f"Unexpected ATM state shape: {tuple(state.shape)}")
    if int(state.shape[1]) < int(target_steps):
        pad_steps = int(target_steps) - int(state.shape[1])
        pad = torch.zeros((1, pad_steps, int(state.shape[2])), dtype=state.dtype)
        state = torch.cat([state, pad], dim=1)
    return state


def prepare_atm_history_frames(
    input_video: torch.Tensor,
    *,
    frame_stack: int,
    img_size: Sequence[int],
) -> torch.Tensor:
    if not isinstance(input_video, torch.Tensor) or input_video.ndim != 5:
        raise TypeError("Expected `input_video` tensor with shape (V,C,T,H,W).")
    if int(input_video.shape[2]) <= 0:
        raise ValueError("ATM history input must contain at least one frame.")

    history = input_video.detach().to(dtype=torch.float32).cpu()
    min_value = float(history.min())
    max_value = float(history.max())
    if min_value >= -1.05 and max_value <= 1.05:
        if min_value < 0.0:
            history = (history + 1.0) * 127.5
        else:
            history = history * 255.0
    history = history.clamp(0.0, 255.0)
    history = history.permute(0, 2, 1, 3, 4).contiguous()

    current_frames = int(history.shape[1])
    if current_frames > int(frame_stack):
        history = history[:, -int(frame_stack) :]
    elif current_frames < int(frame_stack):
        pad = torch.zeros(
            (
                int(history.shape[0]),
                int(frame_stack) - current_frames,
                int(history.shape[2]),
                int(history.shape[3]),
                int(history.shape[4]),
            ),
            dtype=history.dtype,
        )
        history = torch.cat([pad, history], dim=1)

    target_h, target_w = int(img_size[0]), int(img_size[1])
    if tuple(history.shape[-2:]) != (target_h, target_w):
        num_views, num_frames, channels, _, _ = history.shape
        history = history.view(num_views * num_frames, channels, history.shape[-2], history.shape[-1])
        history = F.interpolate(history, size=(target_h, target_w), mode="bilinear", align_corners=False)
        history = history.view(num_views, num_frames, channels, target_h, target_w)
    return history


def run_multiview_atm_window(
    entry: Dict[str, Any],
    *,
    dataset_base_path: str,
    atm_engine,
    state_loader,
    input_video: torch.Tensor,
    start_frame_offset: int,
    task_emb_cache: Optional[Dict[str, torch.Tensor]] = None,
    row_count_cache: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    task_emb = load_task_embedding(
        entry,
        dataset_base_path,
        cache=task_emb_cache,
    )
    state = load_state_pose_condition(
        entry,
        dataset_base_path=dataset_base_path,
        state_loader=state_loader,
        target_steps=int(atm_engine.num_track_ts),
        start_frame=int(entry["start_frame"]) + int(start_frame_offset),
        row_count_cache=row_count_cache,
    )
    history_by_view = prepare_atm_history_frames(
        input_video,
        frame_stack=int(atm_engine.frame_stack),
        img_size=atm_engine.img_size,
    )

    track_by_view: List[torch.Tensor] = []
    for view_idx in range(int(history_by_view.shape[0])):
        pred = atm_engine.infer(
            history_by_view[view_idx : view_idx + 1],
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


@dataclass
class ATMConditionProvider:
    dataset_base_path: str
    metadata_records: List[Dict[str, Any]]
    atm_engine: Any
    state_loader: Any
    task_emb_cache: Dict[str, torch.Tensor] = field(default_factory=dict)
    row_count_cache: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config, *, device: str) -> "ATMConditionProvider":
        from atm.atm_inference import ATMInference

        metadata_records = load_metadata_records(config.metadata_path)
        atm_engine = ATMInference(config.atm_ckpt_path, device=device)
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
        return cls(
            dataset_base_path=str(config.dataset_base_path),
            metadata_records=metadata_records,
            atm_engine=atm_engine,
            state_loader=state_loader,
        )

    @property
    def num_track_ids(self) -> int:
        return int(self.atm_engine.num_track_ids)

    @property
    def num_track_ts(self) -> int:
        return int(self.atm_engine.num_track_ts)

    @property
    def frame_stack(self) -> int:
        return int(self.atm_engine.frame_stack)

    def validate_dataset_size(self, dataset_size: int) -> None:
        if int(dataset_size) != len(self.metadata_records):
            raise ValueError(
                f"WAN dataset size mismatch: dataset={dataset_size}, metadata={len(self.metadata_records)}"
            )

    def metadata_for_index(self, sample_index: int) -> Dict[str, Any]:
        normalized_index = int(sample_index)
        if normalized_index < 0 or normalized_index >= len(self.metadata_records):
            raise IndexError(
                f"Sample index out of range for ATM metadata: {normalized_index} "
                f"(metadata size: {len(self.metadata_records)})"
            )
        return self.metadata_records[normalized_index]

    def track_for_window(
        self,
        sample_index: int,
        *,
        input_video: torch.Tensor,
        start_frame_offset: int,
    ) -> torch.Tensor:
        entry = self.metadata_for_index(sample_index)
        return run_multiview_atm_window(
            entry,
            dataset_base_path=self.dataset_base_path,
            atm_engine=self.atm_engine,
            state_loader=self.state_loader,
            input_video=input_video,
            start_frame_offset=start_frame_offset,
            task_emb_cache=self.task_emb_cache,
            row_count_cache=self.row_count_cache,
        )
