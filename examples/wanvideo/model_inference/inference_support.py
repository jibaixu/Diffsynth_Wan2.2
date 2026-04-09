import json
import logging
import os
from datetime import timedelta
from contextlib import contextmanager, nullcontext, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from diffsynth.core import load_wan_checkpoint_into_pipeline
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.pipelines.wan_video_spec import WanModuleSpec, WanRuntimeConfig
from diffsynth.utils.data import save_video


_TORCHRUN_ENV_KEYS = ("RANK", "WORLD_SIZE", "LOCAL_RANK")


def resolve_optional_path(path_value, base_dir: str):
    if path_value in (None, ""):
        return None
    if isinstance(path_value, os.PathLike):
        path_value = os.fspath(path_value)
    if isinstance(path_value, str) and os.path.isabs(path_value):
        return path_value
    if isinstance(path_value, str):
        return os.path.join(base_dir, path_value)
    return path_value


def flatten_grouped_config(grouped_config: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for value in grouped_config.values():
        if isinstance(value, dict):
            merged.update(value)
    return merged


def load_checkpoint_grouped_config(ckpt_path: Optional[str]) -> Dict[str, Any]:
    if not ckpt_path:
        return {}
    path = Path(ckpt_path)
    config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def load_flat_config_defaults(ckpt_path: Optional[str]) -> Dict[str, Any]:
    grouped_config = load_checkpoint_grouped_config(ckpt_path)
    if not grouped_config:
        return {}
    if any(isinstance(value, dict) for value in grouped_config.values()):
        return flatten_grouped_config(grouped_config)
    return grouped_config


@dataclass(frozen=True)
class DistributedInferenceContext:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: str = "cuda"

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def initialize_distributed_inference(logger: Optional[logging.Logger] = None) -> DistributedInferenceContext:
    env_present = [key for key in _TORCHRUN_ENV_KEYS if key in os.environ]
    if env_present and len(env_present) != len(_TORCHRUN_ENV_KEYS):
        missing = [key for key in _TORCHRUN_ENV_KEYS if key not in os.environ]
        raise RuntimeError(f"Incomplete torchrun environment: missing {missing}")

    if not env_present:
        if logger is not None:
            logger.info("torchrun environment not detected; running single-process inference.")
        return DistributedInferenceContext()

    if not torch.cuda.is_available():
        raise RuntimeError("torchrun inference requires CUDA, but CUDA is not available.")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in the current PyTorch build.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=30),
        )

    context = DistributedInferenceContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=f"cuda:{local_rank}",
    )
    if logger is not None and rank == 0:
        logger.info(
            "Initialized torchrun inference: rank %s/%s on %s",
            context.rank,
            context.world_size,
            context.device,
        )
    return context


def destroy_distributed_inference(context: Optional[DistributedInferenceContext]) -> None:
    if context is None or not context.enabled:
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: Optional[DistributedInferenceContext]) -> None:
    if context is None or not context.enabled:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[context.local_rank])


def broadcast_object(context: Optional[DistributedInferenceContext], value: Any, src: int = 0) -> Any:
    if context is None or not context.enabled:
        return value
    object_list = [value if context.rank == src else None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


@contextmanager
def suppress_stdout_if(enabled: bool):
    if not enabled:
        with nullcontext():
            yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
        yield


def rank_sample_records_path(output_dir: Path, rank: int) -> Path:
    return Path(output_dir) / f"sample_records.rank_{int(rank):03d}.jsonl"


def merged_sample_records_path(output_dir: Path) -> Path:
    return Path(output_dir) / "sample_records.jsonl"


def write_jsonl_records(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        for record in records:
            json.dump(record, file_handle, ensure_ascii=True)
            file_handle.write("\n")


def read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    records: List[Dict[str, Any]] = []
    if not path.is_file():
        raise FileNotFoundError(f"Missing sample-record shard: {path}")
    with path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def merge_rank_sample_records(
    output_dir: Path,
    *,
    world_size: int,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    merged_records: List[Dict[str, Any]] = []
    for rank in range(int(world_size)):
        merged_records.extend(read_jsonl_records(rank_sample_records_path(output_dir, rank)))

    merged_records.sort(
        key=lambda record: (
            int(record.get("sample_index", -1)),
            int(record.get("episode_index", -1)),
            str(record.get("video_path", "")),
        )
    )
    merged_path = merged_sample_records_path(output_dir)
    write_jsonl_records(merged_path, merged_records)
    if logger is not None:
        logger.info("Merged %s sample records -> %s", len(merged_records), merged_path.resolve())
    return merged_records


def _normalize_data_file_keys(data_file_keys) -> list[str]:
    if data_file_keys is None:
        return []
    if isinstance(data_file_keys, str):
        return [item.strip() for item in data_file_keys.split(",") if item.strip()]
    return [str(item).strip() for item in data_file_keys if str(item).strip()]


def build_wan_inference_config(
    values: Dict[str, Any],
    *,
    grouped_config: Optional[Dict[str, Any]] = None,
    data_file_keys=None,
) -> "WanInferenceConfig":
    merged_values = dict(values)
    runtime = WanModuleSpec.parse(merged_values.get("load_modules")).build_runtime(
        merged_values.get("model_paths"),
        _normalize_data_file_keys(
            merged_values.get("data_file_keys") if data_file_keys is None else data_file_keys
        ),
    )
    merged_values["load_modules"] = ",".join(runtime.modules)
    merged_values["data_file_keys"] = runtime.data_file_keys
    merged_values["modules"] = runtime.modules
    merged_values["module_bases"] = runtime.module_bases
    merged_values["text_mode"] = runtime.text_mode
    merged_values["action_mode"] = runtime.action_mode
    merged_values["image_mode"] = runtime.image_mode
    merged_values["enable_text"] = runtime.enable_text
    merged_values["enable_text_encoder"] = runtime.enable_text_encoder
    merged_values["action_enabled"] = runtime.action_enabled
    merged_values["has_text_input_for_dit"] = runtime.has_text_input_for_dit
    merged_values["model_paths"] = json.dumps(runtime.model_paths)
    merged_values["tokenizer_path"] = runtime.tokenizer_path
    return WanInferenceConfig(
        values=merged_values,
        grouped_config=grouped_config or {},
        runtime=runtime,
    )


class FrameConverter:
    @staticmethod
    def to_uint8(frame) -> np.ndarray:
        if isinstance(frame, Image.Image):
            return np.array(frame)
        if isinstance(frame, torch.Tensor):
            if frame.dim() == 3 and frame.shape[0] in [1, 3, 4]:
                frame = frame.permute(1, 2, 0)
            frame = frame.detach().to(dtype=torch.float32)
            return FrameConverter._normalize_to_uint8(frame.cpu().numpy())
        if isinstance(frame, np.ndarray):
            return FrameConverter._normalize_to_uint8(frame)
        return frame

    @staticmethod
    def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
        if array.dtype == np.uint8:
            return array
        max_value = float(array.max())
        min_value = float(array.min())
        if min_value >= -1.0 and max_value <= 1.0:
            if min_value < 0.0:
                array = (array + 1.0) * 127.5
            else:
                array = array * 255.0
        return np.clip(array, 0, 255).astype(np.uint8)

    @staticmethod
    def ensure_rgb(frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 2:
            return np.stack([frame] * 3, axis=-1)
        return frame


class VideoSaver:
    def __init__(self, fps: int = 5, quality: int = 5, show_progress: bool = True) -> None:
        self.fps = fps
        self.quality = quality
        self.show_progress = show_progress
        self.converter = FrameConverter()

    def save_comparison(
        self,
        original_video,
        predicted_video,
        output_dir: Path,
        video_name: str,
        length_mode: str = "truncate_min",
    ) -> Path:
        if not isinstance(original_video, torch.Tensor) or original_video.ndim != 5:
            raise TypeError("`original_video` must be torch.Tensor with shape (V,C,T,H,W).")
        if not isinstance(predicted_video, torch.Tensor) or predicted_video.ndim != 5:
            raise TypeError("`predicted_video` must be torch.Tensor with shape (V,C,T,H,W).")

        original_video = original_video.detach().to(dtype=torch.float32).cpu()
        predicted_video = predicted_video.detach().to(dtype=torch.float32).cpu()
        num_views = min(int(original_video.shape[0]), int(predicted_video.shape[0]))
        gt_frames = int(original_video.shape[2])
        pred_frames = int(predicted_video.shape[2])
        if length_mode == "truncate_min":
            num_frames = min(gt_frames, pred_frames)
        elif length_mode == "pad_to_pred_black_gt":
            num_frames = max(gt_frames, pred_frames)
        else:
            raise ValueError(f"Unknown length_mode={length_mode!r}")

        comparison_frames: List[np.ndarray] = []
        for frame_idx in range(num_frames):
            rows: List[np.ndarray] = []
            for view_idx in range(num_views):
                gt_frame = self._resolve_frame(
                    video=original_video,
                    frame_idx=frame_idx,
                    view_idx=view_idx,
                    fallback_shape=(int(predicted_video.shape[3]), int(predicted_video.shape[4])),
                )
                pred_frame = self._resolve_frame(
                    video=predicted_video,
                    frame_idx=frame_idx,
                    view_idx=view_idx,
                    fallback_shape=(int(original_video.shape[3]), int(original_video.shape[4])),
                )
                rows.append(np.hstack([gt_frame, pred_frame]))
            comparison_frames.append(np.vstack(rows))

        output_path = output_dir / video_name
        save_video(
            np.asarray(comparison_frames),
            str(output_path),
            fps=self.fps,
            quality=self.quality,
            show_progress=self.show_progress,
        )
        return output_path

    def _resolve_frame(
        self,
        *,
        video: torch.Tensor,
        frame_idx: int,
        view_idx: int,
        fallback_shape: tuple[int, int],
    ) -> np.ndarray:
        if frame_idx < int(video.shape[2]):
            frame = self.converter.to_uint8(video[view_idx, :, frame_idx])
            return self.converter.ensure_rgb(frame)
        height, width = fallback_shape
        return np.zeros((height, width, 3), dtype=np.uint8)


@dataclass
class WanInferenceConfig:
    values: Dict[str, Any]
    runtime: WanRuntimeConfig
    grouped_config: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name in self.values:
            return self.values[name]
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    @property
    def metadata_path(self) -> str:
        return str(self._resolve_metadata_path())

    @property
    def resolved_data_base(self) -> str:
        return str(self._resolve_data_base())

    @property
    def videos_path(self) -> str:
        return str(self._resolve_data_base() / "videos")

    @property
    def model_paths_list(self) -> list[Union[str, list[str]]]:
        model_paths = self.values.get("model_paths")
        if not model_paths:
            raise ValueError("Model paths are empty; check `--model_paths` and `--load_modules`.")
        if isinstance(model_paths, str):
            model_paths = json.loads(model_paths)
        if not isinstance(model_paths, (list, tuple)):
            raise ValueError("Model paths must be a list/tuple after parsing.")

        normalized_paths: list[Union[str, list[str]]] = []
        for path in model_paths:
            if isinstance(path, (list, tuple)):
                normalized_paths.append([str(shard_path) for shard_path in path])
            else:
                normalized_paths.append(str(path))
        return normalized_paths

    def build_model_configs(self, offload_device: str = "cpu") -> list[ModelConfig]:
        return [ModelConfig(path=path, offload_device=offload_device) for path in self.model_paths_list]

    def build_tokenizer_config(self) -> Optional[ModelConfig]:
        if not self.enable_text_encoder or not self.tokenizer_path:
            return None
        return ModelConfig(path=self.tokenizer_path)

    def _resolve_metadata_path(self) -> Path:
        metadata_value = self.values.get("dataset_metadata_path")
        if not metadata_value:
            raise ValueError("`dataset_metadata_path` is required for inference.")
        metadata_path = Path(metadata_value)
        if metadata_path.is_absolute():
            return metadata_path
        has_sep = os.sep in metadata_value or (os.altsep and os.altsep in metadata_value)
        if has_sep:
            return metadata_path
        return Path(self.dataset_base_path) / "meta" / metadata_value

    def _resolve_data_base(self) -> Path:
        metadata_path = self._resolve_metadata_path()
        if metadata_path.parent.name == "meta":
            return metadata_path.parent.parent
        return Path(self.dataset_base_path)


class CheckpointPipelineManager:
    def __init__(
        self,
        config: WanInferenceConfig,
        logger: logging.Logger,
        device: str = "cuda",
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.logger = logger
        self.device = device
        self.verbose = verbose
        self.pipeline: Optional[WanVideoPipeline] = None

    def _info(self, message: str, *args) -> None:
        if self.verbose:
            self.logger.info(message, *args)

    def discover_checkpoints(self) -> List[Path]:
        checkpoint_path = self.config.values.get("checkpoint_path")
        if not checkpoint_path:
            self._info("Mode: PRETRAINED ONLY (no --ckpt_path)")
            return []
        ckpt_path = Path(checkpoint_path)
        self._info("Mode: SINGLE CHECKPOINT")
        self._info("  - %s", ckpt_path.name)
        return [ckpt_path]

    def initialize_pipeline(self, checkpoints: List[Path]) -> WanVideoPipeline:
        if checkpoints:
            first_ckpt = checkpoints[0]
            self._info("Using pretrained WAN weights; will apply checkpoint: %s", first_ckpt)
        else:
            self._info("Using pretrained WAN weights only (no checkpoint overlay).")
        self.pipeline = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=self.device,
            model_configs=self.config.build_model_configs(offload_device="cpu"),
            tokenizer_config=self.config.build_tokenizer_config(),
            modules=list(self.config.modules),
        )
        self._info("Pipeline initialized successfully!\n")
        return self.pipeline

    def update_checkpoint(self, checkpoint: Optional[Path]) -> None:
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not initialized")
        if checkpoint is None:
            self._info("No checkpoint provided; skipping overlay update.")
            return
        load_wan_checkpoint_into_pipeline(
            self.pipeline,
            checkpoint,
            torch_dtype=torch.bfloat16,
            device="cpu",
            logger=self.logger if self.verbose else None,
            message_prefix="Updating checkpoint weights",
        )
        self._info("")
