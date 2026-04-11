import os
import sys
from contextlib import nullcontext

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from einops import repeat
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import atm.model as model_zoo
from atm.dataloader import RoboCoinATMActionDataset
from atm.utils.flow_utils import draw_tracks_on_single_image


def _checkpoint_config_path(checkpoint_path):
    return os.path.join(os.path.dirname(checkpoint_path), "config.yaml")


def _load_runtime_cfg(checkpoint_path):
    saved_cfg_path = _checkpoint_config_path(checkpoint_path)
    if not os.path.isfile(saved_cfg_path):
        raise FileNotFoundError(
            f"Training config not found next to checkpoint: {saved_cfg_path}"
        )
    return OmegaConf.load(saved_cfg_path)


def _resolve_cfg_node(cfg, key):
    if key not in cfg:
        raise KeyError(f"Missing '{key}' in saved config.")
    return OmegaConf.to_container(cfg[key], resolve=True)


def _get_model_cfg(cfg):
    model_cfg = dict(_resolve_cfg_node(cfg, "model_cfg"))
    model_cfg.pop("load_path", None)
    if "action_dim" not in model_cfg:
        model_cfg["action_dim"] = 14
    return OmegaConf.create(model_cfg)


def _get_dataset_cfg(cfg):
    return dict(_resolve_cfg_node(cfg, "dataset_cfg"))


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")

    if any(key.startswith(("module.", "model.")) for key in checkpoint.keys()):
        normalized = {}
        for key, value in checkpoint.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            if new_key.startswith("model."):
                new_key = new_key[len("model.") :]
            normalized[new_key] = value
        return normalized

    return checkpoint


def _get_model_input_dtype(model):
    img_proj = getattr(model, "img_proj_encoder", None)
    if img_proj is not None and hasattr(img_proj, "proj") and hasattr(img_proj.proj, "weight"):
        return img_proj.proj.weight.dtype

    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype

    return torch.float32


def _move_tensor_to_model_device(tensor, device, dtype):
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.is_floating_point():
        return tensor.to(device=device, dtype=dtype, non_blocking=True)
    return tensor.to(device=device, non_blocking=True)


def _cast_batch_for_model(model, device, vid, track, vis, task_emb, action):
    target_dtype = _get_model_input_dtype(model)
    return (
        _move_tensor_to_model_device(vid, device, target_dtype),
        _move_tensor_to_model_device(track, device, target_dtype),
        _move_tensor_to_model_device(vis, device, target_dtype),
        _move_tensor_to_model_device(task_emb, device, target_dtype),
        _move_tensor_to_model_device(action, device, target_dtype),
    )


def _get_autocast_context(cfg, device):
    if device.type == "cuda" and cfg.get("mix_precision", False):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _build_query_points(num_track_ids, device, dtype, margin=0.02):
    """
    构建均匀网格采样点
    Args:
        margin: 网格点与图片边缘的最小距离，避免采样点过于靠近边界导致不稳定
    """
    side = int(np.sqrt(num_track_ids))
    if side * side != num_track_ids:
        raise ValueError(
            f"Square uniform grid requires num_track_ids to be a perfect square, got {num_track_ids}."
        )

    # 均匀网格点与图片边缘保持 margin 的距离，避免采样点过于靠近边界导致不稳定
    y = torch.linspace(margin, 1.0 - margin, side, device=device, dtype=dtype)
    x = torch.linspace(margin, 1.0 - margin, side, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)


def _prepare_rgb_image(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"image must have shape (H, W, C) or (C, H, W), got {image.shape}.")

    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"image must have 3 channels after conversion, got {image.shape}.")

    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def _resolve_data_path(dataset_dir, path):
    resolved_path = os.path.expanduser(str(path))
    if os.path.isabs(resolved_path):
        return resolved_path
    return os.path.join(dataset_dir, resolved_path)


def _prepare_video_frames(video_frames, img_size, expected_steps=None):
    if isinstance(video_frames, torch.Tensor):
        video_np = video_frames.detach().cpu().numpy()
    else:
        video_np = np.asarray(video_frames)

    if video_np.ndim == 5:
        if video_np.shape[0] != 1:
            raise ValueError("video_frames only supports batch size 1 when passing 5D input.")
        video_np = video_np[0]

    if video_np.ndim != 4:
        raise ValueError(
            f"video_frames must have shape (T, H, W, C) or (T, C, H, W), got {video_np.shape}."
        )

    if video_np.shape[-1] != 3:
        if video_np.shape[1] in (1, 3):
            video_np = np.transpose(video_np, (0, 2, 3, 1))
        else:
            raise ValueError(
                f"video_frames must have 3 channels, got {video_np.shape}."
            )

    height, width = img_size
    prepared_frames = []
    for frame in video_np:
        frame = _prepare_rgb_image(frame)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        prepared_frames.append(frame)

    if not prepared_frames:
        raise ValueError("video_frames is empty.")

    prepared_frames = np.stack(prepared_frames, axis=0)

    if expected_steps is not None:
        if prepared_frames.shape[0] < expected_steps:
            pad = np.repeat(prepared_frames[-1:], expected_steps - prepared_frames.shape[0], axis=0)
            prepared_frames = np.concatenate([prepared_frames, pad], axis=0)
        elif prepared_frames.shape[0] > expected_steps:
            prepared_frames = prepared_frames[:expected_steps]

    return prepared_frames


def _prepare_track_tensor(tracks, name):
    if isinstance(tracks, torch.Tensor):
        track_tensor = tracks.detach().cpu()
    else:
        track_tensor = torch.as_tensor(tracks)

    if track_tensor.ndim == 3:
        track_tensor = track_tensor.unsqueeze(0)

    if track_tensor.ndim != 4 or track_tensor.shape[0] != 1 or track_tensor.shape[-1] != 2:
        raise ValueError(
            f"{name} must have shape (T, N, 2) or (1, T, N, 2), got {tuple(track_tensor.shape)}."
        )

    return track_tensor.float()


def _annotate_panel(frame, label):
    annotated = frame.copy()
    origin = (12, 30)
    cv2.putText(
        annotated,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (20, 20, 20),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return annotated


class ATMInference:
    def __init__(self, checkpoint_path, device="cuda"):
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available.")

        self.checkpoint_path = checkpoint_path
        self.cfg = _load_runtime_cfg(checkpoint_path)
        self.model_name = str(self.cfg.model_name)
        self.model_cfg = _get_model_cfg(self.cfg)
        self.dataset_cfg = _get_dataset_cfg(self.cfg)

        model_cls = getattr(model_zoo, self.model_name)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)

        self.model = model_cls(**self.model_cfg).to(device=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self.mix_precision = bool(self.cfg.get("mix_precision", False))
        self.num_track_ids = int(self.model_cfg["track_cfg"]["num_track_ids"])
        self.num_track_ts = int(self.model_cfg["track_cfg"]["num_track_ts"])
        self.frame_stack = int(self.model_cfg["vid_cfg"]["frame_stack"])
        self.action_dim = int(self.model_cfg["action_dim"])
        self.task_emb_dim = int(self.model_cfg["language_encoder_cfg"]["input_size"])
        img_size_cfg = self.model_cfg["vid_cfg"]["img_size"]
        self.img_size = (int(img_size_cfg[0]), int(img_size_cfg[1]))

    def build_demo_dataset(
        self,
        jsonl_path=None,
        dataset_dir=None,
        stat_path=None,
        aug_prob=0.0,
        cache_all=None,
    ):
        dataset_cfg = dict(self.dataset_cfg)
        dataset_cfg["aug_prob"] = aug_prob
        if stat_path is not None:
            dataset_cfg["stat_path"] = stat_path
        if cache_all is not None:
            dataset_cfg["cache_all"] = cache_all

        resolved_jsonl = jsonl_path or self.cfg.get("val_jsonl") or self.cfg.get("train_jsonl")
        resolved_dataset_dir = dataset_dir or self.cfg.get("val_dataset_dir") or self.cfg.get("train_dataset_dir")
        if resolved_jsonl is None or resolved_dataset_dir is None:
            raise ValueError(
                "Dataset paths are missing. Provide jsonl_path/dataset_dir or ensure they exist in config.yaml."
            )

        return RoboCoinATMActionDataset(
            jsonl_path=resolved_jsonl,
            dataset_dir=resolved_dataset_dir,
            **dataset_cfg,
        )

    def load_entry_video_frames(self, entry, dataset_dir, expected_steps=None):
        required_keys = ("video", "track", "start_frame", "end_frame")
        missing_keys = [key for key in required_keys if key not in entry]
        if missing_keys:
            raise KeyError(f"Missing required entry fields for visualization: {missing_keys}")

        video_path = _resolve_data_path(dataset_dir, entry["video"])
        track_path = _resolve_data_path(dataset_dir, entry["track"])
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.isfile(track_path):
            raise FileNotFoundError(f"Track file not found: {track_path}")

        start_frame = int(entry["start_frame"])
        end_frame = int(entry["end_frame"])
        if end_frame < start_frame:
            raise ValueError(
                f"Invalid frame range in entry: start_frame={start_frame}, end_frame={end_frame}."
            )

        frame_indices = np.arange(start_frame, end_frame + 1, dtype=np.int64)
        raw_length = entry.get("raw_length")

        if raw_length is not None:
            raw_length = int(raw_length)
            if raw_length <= 0:
                raise ValueError(f"raw_length must be positive, got {raw_length}.")
            frame_indices = np.clip(frame_indices, 0, raw_length - 1)
            frames = RoboCoinATMActionDataset._load_video_frames(video_path, frame_indices=frame_indices)
        else:
            all_frames = RoboCoinATMActionDataset._load_video_frames(video_path)
            if all_frames.shape[0] == 0:
                raise RuntimeError(f"Decoded 0 frames from video: {video_path}")
            frame_indices = np.clip(frame_indices, 0, all_frames.shape[0] - 1)
            frames = all_frames[frame_indices]

        return _prepare_video_frames(
            frames,
            img_size=self.img_size,
            expected_steps=expected_steps,
        )

    def _validate_inputs(self, video, task_emb, action, track):
        if not all(isinstance(tensor, torch.Tensor) for tensor in (video, task_emb, action)):
            raise TypeError("video, task_emb, and action must all be torch.Tensor instances.")
        if video.ndim != 5:
            raise ValueError(f"video must have shape (B, T, C, H, W), got {tuple(video.shape)}.")
        if task_emb.ndim != 2:
            raise ValueError(f"task_emb must have shape (B, E), got {tuple(task_emb.shape)}.")
        if action.ndim != 3:
            raise ValueError(f"action must have shape (B, T, A), got {tuple(action.shape)}.")
        if video.shape[0] != task_emb.shape[0] or video.shape[0] != action.shape[0]:
            raise ValueError(
                "Batch size mismatch among video, task_emb, and action: "
                f"{video.shape[0]}, {task_emb.shape[0]}, {action.shape[0]}."
            )
        if video.shape[1] < self.frame_stack:
            raise ValueError(
                f"video provides {video.shape[1]} frames, but model requires at least {self.frame_stack}."
            )
        if tuple(video.shape[-2:]) != self.img_size:
            raise ValueError(
                f"video spatial size must match checkpoint config {self.img_size}, "
                f"got {tuple(video.shape[-2:])}."
            )
        if task_emb.shape[1] != self.task_emb_dim:
            raise ValueError(
                f"task_emb dim must match checkpoint config {self.task_emb_dim}, "
                f"got {task_emb.shape[1]}."
            )
        if action.shape[1] != self.num_track_ts or action.shape[2] != self.action_dim:
            raise ValueError(
                f"action must have shape (B, {self.num_track_ts}, {self.action_dim}), "
                f"got {tuple(action.shape)}."
            )
        if not task_emb.is_floating_point():
            raise TypeError("task_emb must be a floating-point tensor.")
        if not action.is_floating_point():
            raise TypeError("action must be a floating-point tensor.")

        if track is not None:
            if not isinstance(track, torch.Tensor):
                raise TypeError("track must be a torch.Tensor when provided.")
            if track.ndim != 4:
                raise ValueError(
                    f"track must have shape (B, {self.num_track_ts}, {self.num_track_ids}, 2), "
                    f"got {tuple(track.shape)}."
                )
            expected_shape = (video.shape[0], self.num_track_ts, self.num_track_ids, 2)
            if tuple(track.shape) != expected_shape:
                raise ValueError(
                    f"track must have shape {expected_shape}, got {tuple(track.shape)}."
                )
            if not track.is_floating_point():
                raise TypeError("track must be a floating-point tensor.")

    def _generate_query_tracks(self, batch_size):
        dtype = _get_model_input_dtype(self.model)
        points = _build_query_points(self.num_track_ids, self.device, dtype)
        return repeat(points, "n c -> b t n c", b=batch_size, t=self.num_track_ts)

    @torch.no_grad()
    def infer(self, video, task_emb, action, track=None):
        """
        Args:
            video: (B, T, C, H, W), raw uint8-like image values in [0, 255].
            task_emb: (B, E), language embedding tensor.
            action: (B, num_track_ts, action_dim), reordered and normalized actions.
            track: optional (B, num_track_ts, num_track_ids, 2), sampled query tracks.
        Returns:
            rec_track: (B, num_track_ts, num_track_ids, 2), normalized coordinates in [0, 1].
        """
        self._validate_inputs(video, task_emb, action, track)

        if not video.is_floating_point():
            video = video.float()

        if track is None:
            track = self._generate_query_tracks(video.shape[0])

        video, track, _, task_emb, action = _cast_batch_for_model(
            self.model,
            self.device,
            video,
            track,
            None,
            task_emb,
            action,
        )

        with torch.inference_mode():
            with _get_autocast_context(self.cfg, self.device):
                rec_track, _ = self.model.reconstruct(
                    vid=video,
                    track=track,
                    task_emb=task_emb,
                    p_img=0.0,
                    action=action,
                )
        return rec_track

    def save_predictions_video(
        self,
        video_frames,
        dataset_tracks,
        predictions,
        output_path="output_tracks.mp4",
        track_idx_to_show=None,
        fps=10,
    ):
        """
        Save a side-by-side comparison video for qualitative inspection.

        Args:
            video_frames: (T, H, W, C) or (T, C, H, W), RGB frames for playback.
            dataset_tracks: (T, N, 2) or (1, T, N, 2), GT tracks from dataset sampling.
            predictions: (T, N, 2) or (1, T, N, 2), predicted normalized coordinates.
        """
        gt_tensor = _prepare_track_tensor(dataset_tracks, "dataset_tracks")
        pred_tensor = _prepare_track_tensor(predictions, "predictions")
        if gt_tensor.shape[1] != pred_tensor.shape[1]:
            raise ValueError(
                f"dataset_tracks and predictions must have the same time length, got "
                f"{gt_tensor.shape[1]} and {pred_tensor.shape[1]}."
            )
        if gt_tensor.shape[2] != pred_tensor.shape[2]:
            raise ValueError(
                f"dataset_tracks and predictions must have the same number of tracks, got "
                f"{gt_tensor.shape[2]} and {pred_tensor.shape[2]}."
            )

        video_frames = _prepare_video_frames(
            video_frames,
            img_size=self.img_size,
            expected_steps=gt_tensor.shape[1],
        )

        num_tracks = pred_tensor.shape[2]
        if track_idx_to_show is None:
            num_show = num_tracks
            track_idx_to_show = np.linspace(0, num_tracks - 1, num_show, dtype=int)
        else:
            track_idx_to_show = np.asarray(track_idx_to_show, dtype=int)

        gt_tensor = gt_tensor[:, :, track_idx_to_show]
        pred_tensor = pred_tensor[:, :, track_idx_to_show]

        output_root, _ = os.path.splitext(os.path.abspath(output_path))
        output_path = output_root + ".mp4"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        try:
            writer = imageio.get_writer(
                output_path,
                format="FFMPEG",
                fps=fps,
                codec="libx264",
                ffmpeg_log_level="error",
                ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize ffmpeg-backed MP4 writer via imageio. "
                "Install imageio-ffmpeg and ensure ffmpeg is available."
            ) from exc

        try:
            for t in range(pred_tensor.shape[1]):
                frame = video_frames[t]
                gt_frame = draw_tracks_on_single_image(
                    gt_tensor[:, : t + 1],
                    frame,
                    img_size=self.img_size,
                    tracks_leave_trace=min(15, t),
                )
                pred_frame = draw_tracks_on_single_image(
                    pred_tensor[:, : t + 1],
                    frame,
                    img_size=self.img_size,
                    tracks_leave_trace=min(15, t),
                )
                combined_frame = np.concatenate(
                    [
                        _annotate_panel(gt_frame, "Dataset Track"),
                        _annotate_panel(pred_frame, "Pred Track"),
                    ],
                    axis=1,
                )
                writer.append_data(_prepare_rgb_image(combined_frame))
        finally:
            writer.close()

        print(f"Video saved to {output_path}")


if __name__ == "__main__":
    checkpoint_path = "/data_jbx/Codes/ATM/results/track_transformer/0409_realbot_track_transformer_001B_action_bs_16_grad_acc_4_numtrack_256_ep1001_0047/model_best.ckpt"

    jsonl_path = "/data_jbx/Datasets/Realbot/episodes_train_realbot.jsonl"
    dataset_root = "/data_jbx/Datasets/Realbot"
    stat_path = "/data_jbx/Datasets/Realbot/4_4_four_tasks_wan/meta/stat.json"
    sample_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    output_dir = "results/inference_track"
    os.makedirs(output_dir, exist_ok=True)

    infer_engine = ATMInference(checkpoint_path)
    dataset = infer_engine.build_demo_dataset(
        jsonl_path=jsonl_path,
        dataset_dir=dataset_root,
        stat_path=stat_path,
        aug_prob=0.0,
        cache_all=False,
    )

    for sample_idx in sample_idxs:
        entry = dataset.data_entries[sample_idx]
        video_frames = infer_engine.load_entry_video_frames(
            entry,
            dataset_dir=dataset.dataset_dir,
            expected_steps=infer_engine.num_track_ts,
        )

        real_video, real_tracks, _, real_task_emb, real_action = dataset[sample_idx]

        input_vid = real_video.unsqueeze(0)
        input_track = real_tracks.unsqueeze(0)
        input_task_emb = real_task_emb.unsqueeze(0)
        input_action = real_action.unsqueeze(0)

        print(
            f"Processing sample {sample_idx}, "
            f"input_video={tuple(input_vid.shape)}, vis_video={tuple(video_frames.shape)}, "
            f"track={tuple(input_track.shape)}, task_emb={tuple(input_task_emb.shape)}, "
            f"action={tuple(input_action.shape)}, start_frame={entry['start_frame']}, "
            f"end_frame={entry['end_frame']}"
        )

        preds = infer_engine.infer(
            input_vid,
            input_task_emb,
            input_action,
            track=input_track,
        )

        infer_engine.save_predictions_video(
            video_frames,
            input_track,
            preds,
            output_path=f"{output_dir}/real_sample_inference_{sample_idx}.mp4",
        )
