from typing import Iterable, Optional

from ..core import UnifiedDataset
from ..core.data.operators import LoadCobotAction, LoadCotrackerTrack, ResolvePromptEmbPath
from .wan_video_spec import WanRuntimeConfig


WAN_INFERENCE_DATASET_NUM_FRAMES = 100001


def _normalize_data_file_keys(
    data_file_keys: Optional[Iterable[str]],
    fallback_keys: Iterable[str],
) -> list[str]:
    source = fallback_keys if data_file_keys is None else data_file_keys
    return [str(key).strip() for key in source if str(key).strip()]


def build_wan_special_operator_map(
    runtime: WanRuntimeConfig,
    *,
    base_path: str,
    data_file_keys: Iterable[str],
    num_frames: int,
    time_division_factor: int,
    time_division_remainder: int,
    track_num_points_per_view: int,
) -> dict:
    operator_map = {}
    if runtime.enable_text:
        for key in ("prompt_emb", "negative_prompt_emb"):
            if key in data_file_keys:
                operator_map[key] = ResolvePromptEmbPath(base_path=base_path)
    if "track" in data_file_keys:
        operator_map["track"] = LoadCotrackerTrack(
            base_path=base_path,
            num_frames=num_frames,
            time_division_factor=time_division_factor,
            time_division_remainder=time_division_remainder,
            num_points_per_view=track_num_points_per_view,
        )
    return operator_map


def build_wan_video_dataset(
    runtime: WanRuntimeConfig,
    *,
    base_path: str,
    metadata_path: str,
    height: Optional[int],
    width: Optional[int],
    num_frames: int,
    num_history_frames: int = 1,
    repeat: int = 1,
    resize_mode: str = "fit",
    max_pixels: int = 4096 * 4096,
    data_file_keys: Optional[Iterable[str]] = None,
    dataset_num_frames: Optional[int] = None,
    sample_indices: Optional[Iterable[int]] = None,
    action_stat_path: Optional[str] = None,
    action_type: Optional[str] = None,
    track_num_points_per_view: int = 256,
    history_template_sampling: bool | int = False,
    history_anchor_stride: int = 8,
    height_division_factor: int = 16,
    width_division_factor: int = 16,
    time_division_factor: int = 4,
    time_division_remainder: int = 1,
) -> UnifiedDataset:
    keys = _normalize_data_file_keys(data_file_keys, runtime.data_file_keys)
    operator_num_frames = int(dataset_num_frames) if dataset_num_frames is not None else int(num_frames)

    dataset = UnifiedDataset(
        base_path=base_path,
        metadata_path=metadata_path,
        repeat=repeat,
        data_file_keys=tuple(keys),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=base_path,
            max_pixels=max_pixels,
            height=height,
            width=width,
            height_division_factor=height_division_factor,
            width_division_factor=width_division_factor,
            num_frames=operator_num_frames,
            time_division_factor=time_division_factor,
            time_division_remainder=time_division_remainder,
            resize_mode=resize_mode,
        ),
        special_operator_map=build_wan_special_operator_map(
            runtime,
            base_path=base_path,
            data_file_keys=keys,
            num_frames=operator_num_frames,
            time_division_factor=time_division_factor,
            time_division_remainder=time_division_remainder,
            track_num_points_per_view=track_num_points_per_view,
        ),
        stat_path=action_stat_path,
        action_type=action_type,
        sample_indices=sample_indices,
        temporal_template_sampling=bool(history_template_sampling),
        temporal_num_frames=int(num_frames),
        temporal_num_history_frames=int(num_history_frames),
        temporal_anchor_stride=int(history_anchor_stride),
    )

    if "action" not in keys:
        return dataset
    if not action_type:
        raise ValueError("`action_type` is required when WAN dataset loads `action`.")

    dataset.special_operator_map["action"] = LoadCobotAction(
        base_path=base_path,
        action_type=action_type,
        stat=dataset.stat,
        num_frames=operator_num_frames,
        time_division_factor=time_division_factor,
        time_division_remainder=time_division_remainder,
    )
    return dataset
