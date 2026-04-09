"""
推理引擎：负责所有推理逻辑
包括：检查点管理、数据加载、视频生成、指标评估
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.pipelines.wan_video_data import (
    WAN_INFERENCE_DATASET_NUM_FRAMES,
    build_wan_video_dataset,
)

from inference_support import (
    CheckpointPipelineManager,
    DistributedInferenceContext,
    VideoSaver,
    WanInferenceConfig,
    barrier,
    broadcast_object,
    merge_rank_sample_records,
    rank_sample_records_path,
    resolve_optional_path,
    suppress_stdout_if,
    write_jsonl_records,
)


@dataclass(frozen=True)
class InferenceSample:
    sample_index: int
    original_video: torch.Tensor
    action: Optional[object]
    prompt: Optional[str]
    prompt_emb: Optional[object]
    negative_prompt_emb: Optional[object]
    sample_type: str
    episode_index: int
    num_views: int
    total_frames: int
    input_height: int
    input_width: int


class InferenceEngine:
    """
    推理引擎：负责整个推理流程

    核心流程：
    1. 加载数据集并选择样本
    2. 发现所有 checkpoint
    3. 初始化 pipeline（只加载一次 VAE/CLIP/T5）
    4. 遍历每个 checkpoint：
       - 更新 DiT/Action Encoder 权重
       - 生成所有样本的视频
       - 计算评估指标
    5. 输出总结报告
    """

    def __init__(
        self,
        config: WanInferenceConfig,
        logger,
        dist_context: Optional[DistributedInferenceContext] = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.dist_context = dist_context or DistributedInferenceContext()
        self.video_saver = VideoSaver(
            fps=config.fps,
            quality=config.quality,
            show_progress=self._is_main_process(),
        )
        self.pipeline_manager = CheckpointPipelineManager(
            config,
            logger,
            device=self.dist_context.device,
            verbose=self._is_main_process(),
        )
        self.dataset: Optional[UnifiedDataset] = None
        self.pipeline: Optional[WanVideoPipeline] = None
        self.sample_indices: List[int] = []
        self.checkpoints: List[Path] = []
        self.total_checkpoint_runs: int = 0
        self.num_views: Optional[int] = None
        self.generated_sample_records: List[Dict] = []

    def run(self) -> None:
        try:
            self._prepare()
            with suppress_stdout_if(not self._is_main_process()):
                self.pipeline = self.pipeline_manager.initialize_pipeline(self.checkpoints)

            checkpoints_to_run: List[Optional[Path]] = self.checkpoints if self.checkpoints else [None]
            self.total_checkpoint_runs = len(checkpoints_to_run)
            summaries: List[Dict] = []
            for ckpt_idx, checkpoint in enumerate(checkpoints_to_run, 1):
                summaries.append(
                    self._process_checkpoint(
                        checkpoint,
                        ckpt_idx=ckpt_idx,
                        total_checkpoints=self.total_checkpoint_runs,
                    )
                )
        except Exception as exc:  # pragma: no cover - runtime protection
            self.logger.error(f"Inference failed: {exc}", exc_info=True)
            raise

    def _prepare(self) -> None:
        self._load_dataset()
        all_sample_indices = list(range(len(self.dataset)))
        self.sample_indices = all_sample_indices[self.dist_context.rank :: self.dist_context.world_size]
        self.checkpoints = self.pipeline_manager.discover_checkpoints()

        if self._is_main_process():
            self.logger.info("Total samples: %s", len(self.dataset))
            if self.checkpoints:
                self.logger.info("Checkpoints to process: %s", len(self.checkpoints))
            else:
                self.logger.info("Checkpoints to process: pretrained-only (1 run)")
            self.logger.info("")

    def _load_dataset(self) -> None:
        if self._is_main_process():
            self.logger.info("Loading dataset...")
            if "action" in self.config.data_file_keys:
                self.logger.info("Action stat path: %s", self.config.action_stat_path)

        with suppress_stdout_if(not self._is_main_process()):
            spatial_division_factor = int(getattr(self.config, "spatial_division_factor", 16))
            self.dataset = build_wan_video_dataset(
                self.config.runtime,
                base_path=self.config.dataset_base_path,
                metadata_path=self.config.dataset_metadata_path,
                height=self.config.height,
                width=self.config.width,
                num_frames=int(getattr(self.config, "num_frames", WAN_INFERENCE_DATASET_NUM_FRAMES)),
                num_history_frames=int(getattr(self.config, "num_history_frames", 1)),
                repeat=1,
                resize_mode=self.config.resize_mode,
                max_pixels=self.config.max_pixels,
                data_file_keys=self.config.data_file_keys,
                dataset_num_frames=WAN_INFERENCE_DATASET_NUM_FRAMES,
                action_stat_path=self.config.action_stat_path,
                action_type=self.config.action_type,
                history_template_sampling=int(getattr(self.config, "history_template_sampling", 0)),
                history_anchor_stride=int(getattr(self.config, "history_anchor_stride", 8)),
                height_division_factor=spatial_division_factor,
                width_division_factor=spatial_division_factor,
            )

    def _process_checkpoint(
        self,
        checkpoint: Optional[Path],
        ckpt_idx: int,
        total_checkpoints: int,
    ) -> Dict:
        checkpoint_name = checkpoint.name if checkpoint is not None else "pretrained"
        if self._is_main_process():
            self.logger.info("\n%s", "#" * 80)
            self.logger.info(
                "Processing checkpoint %s/%s: %s",
                ckpt_idx,
                total_checkpoints,
                checkpoint_name,
            )
            self.logger.info("%s\n", "#" * 80)

        with suppress_stdout_if(not self._is_main_process()):
            self.pipeline_manager.update_checkpoint(checkpoint)
        output_dirs = self._create_output_dirs(checkpoint)
        self.generated_sample_records = []
        self._generate_all_videos(output_dirs, ckpt_idx)
        self._write_rank_sample_records(output_dirs["root"])
        barrier(self.dist_context)
        merged_sample_records = self._merge_sample_records(output_dirs["root"])

        if self.dist_context.is_main_process:
            self.logger.info("\n%s", "=" * 60)
            self.logger.info("VIDEO GENERATION COMPLETED")
            self.logger.info("%s", "=" * 60)
            self.logger.info("Total videos: %s", len(merged_sample_records))
            self.logger.info("%s\n", "=" * 60)

        metric_results = self._evaluate_metrics(output_dirs, checkpoint_name, merged_sample_records)
        return {
            "checkpoint": checkpoint_name,
            "output_dir": output_dirs["root"].resolve(),
            "num_videos": len(merged_sample_records) if self.dist_context.is_main_process else len(self.sample_indices),
            "metrics": metric_results,
        }

    def _create_output_dirs(self, checkpoint: Optional[Path]) -> Dict[str, Path]:
        from datetime import datetime

        output_dir_value = None
        if self.dist_context.is_main_process:
            timestamp = datetime.now().strftime("%mM%dD_%HH%MMin")
            if checkpoint is None:
                output_dir_value = str(Path("Ckpt") / "pretrained" / timestamp)
            else:
                output_dir_value = str(checkpoint.parent / checkpoint.stem / timestamp)
        output_dir = Path(broadcast_object(self.dist_context, output_dir_value))
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.dist_context.is_main_process:
            self._save_config(output_dir)
        barrier(self.dist_context)
        if self._is_main_process():
            self.logger.info("Output directory: %s\n", output_dir.resolve())
        return {"root": output_dir}

    def _save_config(self, output_dir: Path) -> None:
        config_path = output_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            payload = self.config.grouped_config or self.config.values
            json.dump(payload, f, indent=2, ensure_ascii=True)

    def _evaluate_metrics(
        self,
        output_dirs: Dict[str, Path],
        checkpoint_name: str,
        sample_records: List[Dict],
    ) -> Optional[Dict]:
        if not self.dist_context.is_main_process:
            return None
        if not self.config.enable_metrics:
            self.logger.info("Metrics disabled; skipping evaluation.\n")
            return None

        from diffsynth.core.metric.metric import evaluate_and_write_report

        metrics_path = output_dirs["root"] / "metrics.json"
        self.logger.info("Metrics are evaluated on rank 0 only.")
        self.logger.info("Evaluating metrics for checkpoint: %s", checkpoint_name)
        metrics = evaluate_and_write_report(
            str(output_dirs["root"]),
            checkpoint_name=checkpoint_name,
            metrics_output_path=str(metrics_path),
            sample_records=sample_records,
            num_views=self._resolve_num_views(),
            frame_chunk_size=int(self.config.num_frames),
        )
        self.logger.info("Saved metrics to: %s", metrics_path.resolve())
        self.logger.info("%s\n", metrics["overall"])
        return metrics

    def _generate_all_videos(self, output_dirs: Dict[str, Path], ckpt_idx: int) -> None:
        for index, sample_idx in enumerate(self.sample_indices, 1):
            if self._is_main_process():
                self.logger.info("\n%s", "#" * 60)
                if self.total_checkpoint_runs > 1:
                    self.logger.info(
                        "[Checkpoint %s/%s] Video %s/%s (Sample %s)",
                        ckpt_idx,
                        self.total_checkpoint_runs,
                        index,
                        len(self.sample_indices),
                        sample_idx,
                    )
                else:
                    self.logger.info("Video %s/%s (Sample %s)", index, len(self.sample_indices), sample_idx)
                self.logger.info("%s", "#" * 60)

            raw_sample = self.dataset[sample_idx]
            sample = self._prepare_sample(sample_idx, raw_sample)
            self._generate_single_video(sample, output_dirs)

    def _prepare_sample(self, sample_idx: int, sample: Dict) -> InferenceSample:
        original_video = sample["video"]
        if not isinstance(original_video, torch.Tensor) or original_video.ndim != 5:
            raise TypeError(
                f"`sample['video']` must be torch.Tensor with shape (V,C,T,H,W), got {type(original_video)}"
            )
        original_video = original_video.detach().cpu()
        action = sample.get("action")
        if getattr(self.pipeline, "action_injection_mode", "off") == "off":
            action = None
        episode_index = int(sample["episode_index"])
        prompt = sample.get("prompt")
        prompt_emb = resolve_optional_path(sample.get("prompt_emb"), self.config.dataset_base_path)
        negative_prompt_emb = sample.get("negative_prompt_emb", self.config.negative_prompt_emb)
        negative_prompt_emb = resolve_optional_path(negative_prompt_emb, self.config.dataset_base_path)
        sample_type = str(sample.get("type") or "unknown")

        num_views = int(original_video.shape[0])
        total_frames = int(original_video.shape[2])
        input_height = int(original_video.shape[-2])
        input_width = int(original_video.shape[-1])
        if self.num_views is None:
            self.num_views = num_views
        elif self.num_views != num_views:
            self.logger.warning("Mixed num_views detected: %s -> %s", self.num_views, num_views)

        return InferenceSample(
            sample_index=int(sample_idx),
            original_video=original_video,
            action=action,
            prompt=prompt,
            prompt_emb=prompt_emb,
            negative_prompt_emb=negative_prompt_emb,
            sample_type=sample_type,
            episode_index=episode_index,
            num_views=num_views,
            total_frames=total_frames,
            input_height=input_height,
            input_width=input_width,
        )

    def _generate_single_video(
        self,
        sample: InferenceSample,
        output_dirs: Dict[str, Path],
    ) -> None:
        self._log_sample(sample)
        predicted_video = self._predict_video(sample)
        predicted_video = self._finalize_predicted_video(predicted_video, sample.original_video)

        type_dir = output_dirs["root"] / sample.sample_type
        type_dir.mkdir(parents=True, exist_ok=True)
        video_name = f"video_s{sample.sample_index:06d}_ep{sample.episode_index}.mp4"
        saved_path = self.video_saver.save_comparison(
            sample.original_video[:, :, : int(predicted_video.shape[2])],
            predicted_video,
            type_dir,
            video_name,
        )
        self._record_generated_sample(
            video_path=saved_path,
            prompt=sample.prompt,
            sample_type=sample.sample_type,
            episode_index=sample.episode_index,
            sample_index=sample.sample_index,
        )
        if self._is_main_process():
            self.logger.info("Saved: %s\n", saved_path.resolve())

    def _log_sample(self, sample: InferenceSample) -> None:
        if not self._is_main_process():
            return
        self.logger.info("Episode: %s", sample.episode_index)
        self.logger.info("Loaded shape: %s", tuple(sample.original_video.shape))
        self.logger.info("Type: %s", sample.sample_type)
        self.logger.info("Generating with diffusion (frames: %s)", sample.total_frames)
        self.logger.info("Using input size: %sx%s", sample.input_width, sample.input_height)

    def _predict_video(self, sample: InferenceSample) -> torch.Tensor:
        if self.config.chunk_infer and sample.total_frames > int(self.config.num_frames):
            return self._run_chunked_inference(sample)
        return self._run_single_pass_inference(sample)

    def _run_single_pass_inference(self, sample: InferenceSample) -> torch.Tensor:
        history_frames = int(self.config.num_history_frames)
        action = self._trim_action(sample.action, sample.total_frames)
        infer_frames = self._align_num_frames(sample.total_frames)
        infer_action = self._slice_and_pad_action(
            action=action,
            start=0,
            target_frames=sample.total_frames,
            infer_frames=infer_frames,
        )
        predicted_video = self._call_pipeline(
            sample,
            input_video=sample.original_video[:, :, :history_frames],
            action=infer_action,
            infer_frames=infer_frames,
        )
        predicted_video = predicted_video.detach().cpu()[:, :, :sample.total_frames]
        return self._restore_history_frames(predicted_video, sample.original_video, history_frames)

    def _run_chunked_inference(self, sample: InferenceSample) -> torch.Tensor:
        history_frames = int(self.config.num_history_frames)
        chunk_size = int(self.config.num_frames)
        chunk_stride = chunk_size - history_frames
        if chunk_stride <= 0:
            raise ValueError(
                f"Chunked inference requires num_frames > num_history_frames, got {chunk_size} and {history_frames}"
            )

        action = self._trim_action(sample.action, sample.total_frames)
        chunk_starts = [0]
        while chunk_starts[-1] + chunk_stride + chunk_size <= sample.total_frames:
            chunk_starts.append(chunk_starts[-1] + chunk_stride)

        predicted_chunks: List[torch.Tensor] = []
        last_input_video = sample.original_video[:, :, :history_frames]
        for chunk_idx, start in enumerate(chunk_starts):
            chunk_frames = min(chunk_size, sample.total_frames - start)
            infer_frames = self._align_num_frames(chunk_frames)
            chunk_action = self._slice_and_pad_action(
                action=action,
                start=start,
                target_frames=chunk_frames,
                infer_frames=infer_frames,
            )
            chunk_input_video = last_input_video
            chunk_video = self._call_pipeline(
                sample,
                input_video=chunk_input_video,
                action=chunk_action,
                infer_frames=infer_frames,
            )
            chunk_video = chunk_video.detach().cpu()[:, :, :chunk_frames]
            chunk_video = self._restore_history_frames(chunk_video, chunk_input_video, history_frames)
            if int(chunk_video.shape[2]) > 0:
                last_input_video = chunk_video[:, :, -history_frames:].contiguous()
            if chunk_idx > 0:
                chunk_video = chunk_video[:, :, history_frames:]
            if int(chunk_video.shape[2]) > 0:
                predicted_chunks.append(chunk_video)

        if predicted_chunks:
            return torch.cat(predicted_chunks, dim=2)
        return sample.original_video[:, :, :0]

    def _call_pipeline(
        self,
        sample: InferenceSample,
        *,
        input_video: torch.Tensor,
        action,
        infer_frames: int,
    ) -> torch.Tensor:
        predicted_video = self.pipeline(
            prompt=sample.prompt,
            negative_prompt=self.config.negative_prompt,
            prompt_emb=sample.prompt_emb,
            negative_prompt_emb=sample.negative_prompt_emb,
            input_video=input_video,
            action=action,
            seed=self.config.seed,
            tiled=False,
            height=sample.input_height,
            width=sample.input_width,
            num_frames=infer_frames,
            num_history_frames=int(self.config.num_history_frames),
            cfg_scale=self.config.cfg_scale,
            num_inference_steps=self.config.num_inference_steps,
            progress_bar_cmd=self._progress_bar_cmd(),
        )
        if not isinstance(predicted_video, torch.Tensor) or predicted_video.ndim != 5:
            raise TypeError(f"Pipeline output must be (V,C,T,H,W), got {type(predicted_video)}")
        return predicted_video

    def _finalize_predicted_video(
        self,
        predicted_video: torch.Tensor,
        original_video: torch.Tensor,
    ) -> torch.Tensor:
        expected_frames = int(original_video.shape[2])
        pred_frames = int(predicted_video.shape[2])
        if pred_frames > expected_frames:
            return predicted_video[:, :, :expected_frames]
        if pred_frames < expected_frames:
            self.logger.warning("Generated %s frames, expected %s", pred_frames, expected_frames)
        return predicted_video

    def _record_generated_sample(
        self,
        video_path: Path,
        prompt,
        sample_type: str,
        episode_index: int,
        sample_index: int,
    ) -> None:
        if prompt is None:
            raise ValueError("`prompt` must be a non-empty string for metric evaluation.")
        self.generated_sample_records.append(
            {
                "video_path": str(video_path.resolve()),
                "prompt": str(prompt),
                "sample_type": str(sample_type),
                "episode_index": int(episode_index),
                "sample_index": int(sample_index),
                "rank": int(self.dist_context.rank),
            }
        )

    def _write_rank_sample_records(self, output_dir: Path) -> None:
        shard_path = rank_sample_records_path(output_dir, self.dist_context.rank)
        write_jsonl_records(shard_path, self.generated_sample_records)
        if self._is_main_process():
            self.logger.info("Saved sample-record shard: %s", shard_path.resolve())

    def _merge_sample_records(self, output_dir: Path) -> List[Dict]:
        if not self.dist_context.is_main_process:
            return []
        return merge_rank_sample_records(
            output_dir,
            world_size=self.dist_context.world_size,
            logger=self.logger,
        )

    def _resolve_num_views(self) -> int:
        if self.num_views is not None:
            return int(self.num_views)
        if self.dataset is None or len(self.dataset) == 0:
            raise ValueError("Cannot resolve `num_views` from an empty dataset.")
        sample = self.dataset[0]
        video = sample["video"]
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise TypeError("`sample['video']` must be torch.Tensor with shape (V,C,T,H,W).")
        return int(video.shape[0])

    def _is_main_process(self) -> bool:
        return self.dist_context.is_main_process

    def _progress_bar_cmd(self):
        return tqdm if self._is_main_process() else self._noop_progress_bar

    @staticmethod
    def _noop_progress_bar(iterable, *args, **kwargs):
        return iterable

    @staticmethod
    def _align_num_frames(num_frames: int) -> int:
        if (num_frames - 1) % 4 == 0:
            return num_frames
        return num_frames + (4 - ((num_frames - 1) % 4))

    @staticmethod
    def _slice_and_pad_action(action, start: int, target_frames: int, infer_frames: int):
        if action is None:
            return None
        chunk_action = action[:, start : start + target_frames]
        current_frames = int(chunk_action.shape[1])
        if current_frames >= infer_frames:
            return chunk_action[:, :infer_frames]
        if current_frames <= 0:
            return chunk_action
        pad_frames = infer_frames - current_frames
        if isinstance(chunk_action, torch.Tensor):
            pad = chunk_action[:, -1:, :].repeat(1, pad_frames, 1)
            return torch.cat([chunk_action, pad], dim=1)
        pad = np.repeat(chunk_action[:, -1:, :], repeats=pad_frames, axis=1)
        return np.concatenate([chunk_action, pad], axis=1)

    @staticmethod
    def _restore_history_frames(
        predicted_video: torch.Tensor,
        history_source: torch.Tensor,
        history_frames: int,
    ) -> torch.Tensor:
        history_to_copy = min(
            int(history_frames),
            int(predicted_video.shape[2]),
            int(history_source.shape[2]),
        )
        if history_to_copy > 0:
            predicted_video[:, :, :history_to_copy] = history_source[:, :, :history_to_copy]
        return predicted_video

    @staticmethod
    def _trim_action(action, total_frames: int):
        if action is None:
            return None
        return action[:, :total_frames]
