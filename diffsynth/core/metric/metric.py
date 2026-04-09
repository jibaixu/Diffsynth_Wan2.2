import os
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import lpips
import imageio.v2 as imageio
from tqdm import tqdm
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pytorch_fid.inception import InceptionV3
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.nn.functional import adaptive_avg_pool2d, interpolate
from decord import VideoReader, cpu


FID_DIMS = 2048
BATCH_SIZE = 512
FVD_INPUT_RES = 224
DEFAULT_VIEW_NAMES = [
    "cam_high_rgb",
    "cam_left_wrist_rgb",
    "cam_right_wrist_rgb",
]
DEFAULT_METRIC_PRECISION = 4
BASIC_METRICS = ("psnr", "ssim", "mse", "lpips")
DISTRIBUTION_METRICS = ("fid", "fvd")
PBENCH_METRICS = (
    "aesthetic_quality",
    "imaging_quality",
    "motion_smoothness",
    "background_consistency",
    "subject_consistency",
    "overall_consistency",
    "i2v_background",
    "i2v_subject",
)
VIEW_METRICS = BASIC_METRICS + PBENCH_METRICS
OVERALL_METRICS = BASIC_METRICS + DISTRIBUTION_METRICS + PBENCH_METRICS


@dataclass(frozen=True)
class EvalContext:
    comparison_dir: str
    video_files: List[str]
    view_names: List[str]
    num_workers: int
    num_views: int
    frame_chunk_size: int
    device: torch.device
    sample_records: Optional[List[dict]] = None
    prompt_map_jsonl: Optional[str] = None


@dataclass(frozen=True)
class PreparedViewSample:
    video_path: str
    video_stem: str
    prompt: Optional[str]
    view_name: str
    view_index: int
    gt_video: np.ndarray
    pred_video: np.ndarray
    frames: int


def read_video_decord(path):
    vr = VideoReader(path, ctx=cpu(0))
    frames = vr.get_batch(range(len(vr)))  # (N, H, W, C)
    frames = frames.asnumpy().astype(np.float32) / 255.  # -> [0, 1]

    return frames


def split_comparison_grid(frames, rows=3, cols=2):
    height = frames.shape[1]
    width = frames.shape[2]
    if height == 0 or width == 0:
        raise ValueError(f"Invalid comparison video size: {height}x{width}")
    row_splits = np.array_split(frames, rows, axis=1)
    grid = []
    for row in row_splits:
        col_splits = np.array_split(row, cols, axis=2)
        if len(col_splits) != cols:
            raise ValueError(f"Failed to split comparison video into {cols} columns")
        grid.append(col_splits)
    if len(grid) != rows:
        raise ValueError(f"Failed to split comparison video into {rows} rows")
    return grid


def _crop_to_common_size(gt_frame, pred_frame):
    min_h = min(gt_frame.shape[0], pred_frame.shape[0])
    min_w = min(gt_frame.shape[1], pred_frame.shape[1])
    if min_h == 0 or min_w == 0:
        raise ValueError("Invalid frame size after cropping")
    return gt_frame[:min_h, :min_w], pred_frame[:min_h, :min_w]


def _compute_usable_frames(gt_video, pred_video, frame_chunk_size):
    min_frames = min(gt_video.shape[0], pred_video.shape[0])
    if min_frames <= 0:
        return 0
    return (min_frames // frame_chunk_size) * frame_chunk_size


def _mean(values):
    valid_values = [float(value) for value in values if value is not None and float(value) >= 0.0]
    if not valid_values:
        return -1.0
    return float(sum(valid_values) / len(valid_values))


def _safe_float(value, default=-1.0):
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def _empty_metric_values(metric_names, default=-1.0):
    return {metric_name: float(default) for metric_name in metric_names}


def _prepare_samples(ctx: EvalContext):
    prepared_samples = []
    prompt_map = {}

    for record in ctx.sample_records or []:
        prompt_map[os.path.realpath(str(record["video_path"]))] = str(record["prompt"])

    if not prompt_map and ctx.prompt_map_jsonl and os.path.isfile(ctx.prompt_map_jsonl):
        with open(ctx.prompt_map_jsonl, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = record.get("prompt_en") or record.get("prompt") or ""
                if not prompt:
                    continue
                video_path = record.get("video_path")
                if video_path:
                    prompt_map[os.path.realpath(video_path)] = prompt
                video_relpath = record.get("video_relpath")
                if video_relpath:
                    normalized = os.path.normpath(video_relpath)
                    prompt_map[normalized] = prompt
                    prompt_map[os.path.basename(normalized)] = prompt

    for video_path in tqdm(ctx.video_files, desc="Preparing comparison videos ...", leave=False):
        frames = read_video_decord(video_path)
        grid = split_comparison_grid(frames, rows=ctx.num_views, cols=2)
        prompt = None
        if prompt_map:
            abs_key = os.path.realpath(video_path)
            rel_key = os.path.normpath(os.path.relpath(abs_key, os.path.realpath(ctx.comparison_dir)))
            prompt = prompt_map.get(abs_key) or prompt_map.get(rel_key) or prompt_map.get(os.path.basename(abs_key))
        video_stem = os.path.basename(video_path).removesuffix(".mp4")

        for view_idx, row in enumerate(grid[:ctx.num_views]):
            gt_video = row[0]
            pred_video = row[1]
            usable_frames = _compute_usable_frames(gt_video, pred_video, ctx.frame_chunk_size)
            if usable_frames <= 0:
                continue

            gt_video = gt_video[:usable_frames]
            pred_video = pred_video[:usable_frames]
            min_h = min(gt_video.shape[1], pred_video.shape[1])
            min_w = min(gt_video.shape[2], pred_video.shape[2])
            if min_h <= 0 or min_w <= 0:
                continue

            prepared_samples.append(
                PreparedViewSample(
                    video_path=video_path,
                    video_stem=video_stem,
                    prompt=prompt,
                    view_name=ctx.view_names[view_idx] if view_idx < len(ctx.view_names) else f"view_{view_idx + 1}",
                    view_index=view_idx,
                    gt_video=gt_video[:, :min_h, :min_w],
                    pred_video=pred_video[:, :min_h, :min_w],
                    frames=int(usable_frames),
                )
            )

    if not prepared_samples:
        raise RuntimeError("No valid view pairs found in comparison videos.")
    return prepared_samples


def _to_uint8_frames(frames):
    array = np.asarray(frames)
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    else:
        array = np.clip(array, 0, 255)
    return array.astype(np.uint8)


def _write_mp4(video_path, frames, fps=8):
    frames = _to_uint8_frames(frames)
    with imageio.get_writer(video_path, fps=fps, quality=5) as writer:
        for frame in frames:
            writer.append_data(frame)


def _write_image(image_path, frame):
    frame = _to_uint8_frames(frame)
    imageio.imwrite(image_path, frame)


def _compute_pbench_metrics(prepared_samples, ctx: EvalContext):
    with tempfile.TemporaryDirectory(prefix="pbench_chunk_eval_") as tmp_dir:
        eval_root = os.path.join(tmp_dir, "video_quality")
        videos_dir = os.path.join(eval_root, "videos")
        images_dir = os.path.join(eval_root, "condition_images")
        output_dir = os.path.join(eval_root, "evaluation_results")
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        prompt_entries = []
        chunk_metadata = {}
        chunk_idx = 0

        for sample in tqdm(prepared_samples, desc="Preparing PBench chunks ...", leave=False):
            if not sample.prompt:
                raise RuntimeError(
                    f"PBench metrics require prompt mapping for every sample, missing prompt for {sample.video_path}"
                )
            for start in range(0, sample.frames, ctx.frame_chunk_size):
                pred_chunk = sample.pred_video[start:start + ctx.frame_chunk_size]
                if pred_chunk.shape[0] != ctx.frame_chunk_size:
                    continue
                cond_frame = sample.gt_video[start]
                sample_id = f"{sample.video_stem}_v{sample.view_index:02d}_c{chunk_idx:06d}"
                video_out = os.path.join(videos_dir, f"{sample_id}.mp4")
                image_out = os.path.join(images_dir, f"{sample_id}.jpg")
                _write_mp4(video_out, pred_chunk)
                _write_image(image_out, cond_frame)
                chunk_metadata[sample_id] = {"view_name": sample.view_name}
                prompt_entries.append(
                    {
                        "video_id": sample_id,
                        "prompt": sample.prompt,
                        "prompt_en": sample.prompt,
                        "custom_image_path": image_out,
                    }
                )
                chunk_idx += 1

        if not prompt_entries:
            raise RuntimeError("No valid chunks prepared for PBench metrics.")

        prompt_file = os.path.join(eval_root, "prompts.json")
        with open(prompt_file, "w", encoding="utf-8") as file_handle:
            json.dump(prompt_entries, file_handle, indent=2, ensure_ascii=False)

        try:
            from .pbench import PBench
        except ImportError:
            from pbench import PBench

        full_json_dir = os.path.join(os.path.dirname(__file__), "pbench", "VBench_full_info.json")
        evaluator = PBench(ctx.device, full_json_dir, output_dir)

        video_path_to_prompt = {f"{item['video_id']}.mp4": item for item in prompt_entries}
        prev_force_single = os.environ.get("PBENCH_FORCE_SINGLE_PROCESS")
        os.environ["PBENCH_FORCE_SINGLE_PROCESS"] = "1"
        try:
            evaluator.evaluate(
                videos_path=videos_dir,
                name="results_chunked",
                prompt_list=video_path_to_prompt,
                dimension_list=PBENCH_METRICS,
                local=True,
                read_frame=False,
                mode="custom_input",
                custom_image_folder=images_dir,
                enable_missing_videos=True,
            )
        finally:
            if prev_force_single is None:
                os.environ.pop("PBENCH_FORCE_SINGLE_PROCESS", None)
            else:
                os.environ["PBENCH_FORCE_SINGLE_PROCESS"] = prev_force_single
        result_files = [
            os.path.join(output_dir, filename)
            for filename in os.listdir(output_dir)
            if filename.startswith("results_") and filename.endswith("_eval_results.json")
        ]
        raw_results = {}
        if result_files:
            with open(max(result_files, key=os.path.getmtime), "r", encoding="utf-8") as file_handle:
                raw_results = json.load(file_handle)

    parsed_results = {
        "overall": _empty_metric_values(PBENCH_METRICS),
        "views": {
            view_name: _empty_metric_values(PBENCH_METRICS)
            for view_name in ctx.view_names
        },
    }

    for metric_name in PBENCH_METRICS:
        metric_result = raw_results.get(metric_name)
        overall_value = -1.0
        per_view_scores = {view_name: [] for view_name in ctx.view_names}
        if isinstance(metric_result, list):
            if metric_result:
                overall_value = _safe_float(metric_result[0])
            if len(metric_result) > 1 and isinstance(metric_result[1], list):
                for entry in metric_result[1]:
                    chunk_id = None
                    if isinstance(entry, dict):
                        for key in ("video_path", "image_path"):
                            path = entry.get(key)
                            if path:
                                chunk_id = os.path.splitext(os.path.basename(str(path)))[0]
                                break
                    metadata = chunk_metadata.get(chunk_id or "")
                    if not metadata:
                        continue
                    view_name = metadata["view_name"]
                    score = _safe_float(entry.get("video_results"))
                    if metric_name == "imaging_quality" and score >= 0.0:
                        score /= 100.0
                    if score >= 0.0:
                        per_view_scores[view_name].append(score)
        elif isinstance(metric_result, (int, float)):
            overall_value = float(metric_result)

        for view_name in ctx.view_names:
            view_value = _mean(per_view_scores[view_name])
            parsed_results["views"][view_name][metric_name] = view_value
        parsed_results["overall"][metric_name] = _mean(
            [parsed_results["views"][view_name][metric_name] for view_name in ctx.view_names]
        )
        if parsed_results["overall"][metric_name] < 0.0:
            parsed_results["overall"][metric_name] = overall_value

    return parsed_results


def _compute_stats(features):
    if features.shape[0] <= 0:
        raise RuntimeError("No features to compute statistics")
    mu = np.mean(features, axis=0)
    if features.shape[0] == 1:
        sigma = np.eye(features.shape[1], dtype=np.float64) * 1e-6
    else:
        sigma = np.cov(features, rowvar=False)
    return mu, sigma


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def _extract_inception_features(frames, model, device, batch_size=BATCH_SIZE):
    if frames.shape[0] <= 0:
        return np.empty((0, FID_DIMS), dtype=np.float32)
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float()
    activations = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[start:start + batch_size].to(device)
            pred = model(batch)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            activations.append(pred)
    if not activations:
        return np.empty((0, FID_DIMS), dtype=np.float32)
    return np.concatenate(activations, axis=0)


def _compute_fid_overall(prepared_samples, device):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FID_DIMS]
    model = InceptionV3([block_idx]).to(device).eval()
    gt_acts = []
    pred_acts = []
    for sample in tqdm(prepared_samples, desc="Preparing FID features ...", leave=False):
        gt_feat = _extract_inception_features(sample.gt_video, model, device)
        pred_feat = _extract_inception_features(sample.pred_video, model, device)
        if gt_feat.shape[0] > 0 and pred_feat.shape[0] > 0:
            gt_acts.append(gt_feat)
            pred_acts.append(pred_feat)
    if not gt_acts or not pred_acts:
        return -1.0
    gt_acts = np.concatenate(gt_acts, axis=0)
    pred_acts = np.concatenate(pred_acts, axis=0)
    mu1, sigma1 = _compute_stats(gt_acts)
    mu2, sigma2 = _compute_stats(pred_acts)
    return _frechet_distance(mu1, sigma1, mu2, sigma2)


def _extract_i3d_features(video, i3d_model, device, chunk_size):
    features = []
    if video.shape[0] < chunk_size:
        return features
    for start in range(0, video.shape[0], chunk_size):
        chunk = video[start:start + chunk_size]
        if chunk.shape[0] != chunk_size:
            continue
        tensor = torch.from_numpy(chunk).permute(0, 3, 1, 2).contiguous().float()
        if tensor.shape[-2] != FVD_INPUT_RES or tensor.shape[-1] != FVD_INPUT_RES:
            tensor = interpolate(tensor, size=(FVD_INPUT_RES, FVD_INPUT_RES), mode='bilinear', align_corners=False)
        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)
        tensor = 2.0 * tensor - 1.0
        with torch.no_grad():
            feat = i3d_model(tensor, rescale=False, resize=False, return_features=True)
        features.append(feat.squeeze(0).cpu().numpy())
    return features


def _compute_fvd_overall(prepared_samples, frame_chunk_size, device):
    i3d_path = os.path.join(os.path.dirname(__file__), "i3d_torchscript.pt")
    if not os.path.isfile(i3d_path):
        raise FileNotFoundError(f"FVD model not found: {i3d_path}")
    i3d_model = torch.jit.load(i3d_path, map_location=device).eval()

    gt_feats = []
    pred_feats = []
    for sample in tqdm(prepared_samples, desc="Preparing FVD features ...", leave=False):
        gt_feats.extend(_extract_i3d_features(sample.gt_video, i3d_model, device, chunk_size=frame_chunk_size))
        pred_feats.extend(_extract_i3d_features(sample.pred_video, i3d_model, device, chunk_size=frame_chunk_size))

    if not gt_feats or not pred_feats:
        return -1.0
    gt_feats = np.asarray(gt_feats, dtype=np.float64)
    pred_feats = np.asarray(pred_feats, dtype=np.float64)
    mu1, sigma1 = _compute_stats(gt_feats)
    mu2, sigma2 = _compute_stats(pred_feats)
    return _frechet_distance(mu1, sigma1, mu2, sigma2)


def _compute_metrics_for_pair(gt_video, pred_video, disable_tqdm=False):
    if gt_video.shape[0] <= 0 or pred_video.shape[0] <= 0:
        raise RuntimeError("No frames found in comparison video")
    if gt_video.shape[0] != pred_video.shape[0]:
        raise RuntimeError(f"Frame count mismatch: {gt_video.shape[0]} vs {pred_video.shape[0]}")

    usable_frames = int(gt_video.shape[0])
    psnr_sum = 0.0
    ssim_sum = 0.0
    mse_sum = 0.0
    for i in tqdm(range(usable_frames), desc='traverse frames ...', disable=disable_tqdm, leave=False):
        pred_ = pred_video[i]
        gt_ = gt_video[i]
        gt_, pred_ = _crop_to_common_size(gt_, pred_)
        psnr = peak_signal_noise_ratio(pred_, gt_, data_range=1.)
        ssim = structural_similarity(pred_, gt_, channel_axis=-1, data_range=1.)
        mse = float(np.mean((pred_ - gt_) ** 2))
        psnr_sum += psnr
        ssim_sum += ssim
        mse_sum += mse

    return {
        "frames": usable_frames,
        "psnr": psnr_sum / usable_frames,
        "ssim": ssim_sum / usable_frames,
        "mse": mse_sum / usable_frames,
        "psnr_sum": psnr_sum,
        "ssim_sum": ssim_sum,
        "mse_sum": mse_sum,
    }


def _compute_basic_metrics(prepared_samples, ctx: EvalContext):
    metrics = {
        "overall": _empty_metric_values(BASIC_METRICS),
        "views": {
            view_name: {
                **_empty_metric_values(BASIC_METRICS),
                "frames": 0,
            }
            for view_name in ctx.view_names
        },
    }

    basic_totals = {
        view_name: {"psnr_sum": 0.0, "ssim_sum": 0.0, "mse_sum": 0.0, "frames": 0}
        for view_name in ctx.view_names
    }
    max_workers = max(1, int(ctx.num_workers)) if ctx.num_workers is not None else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {
            executor.submit(_compute_metrics_for_pair, sample.gt_video, sample.pred_video, True): sample
            for sample in prepared_samples
        }
        for future in tqdm(as_completed(future_to_sample), total=len(future_to_sample), desc="Computing PSNR/SSIM/MSE ..."):
            sample = future_to_sample[future]
            sample_metrics = future.result()
            totals = basic_totals[sample.view_name]
            totals["psnr_sum"] += sample_metrics["psnr_sum"]
            totals["ssim_sum"] += sample_metrics["ssim_sum"]
            totals["mse_sum"] += sample_metrics["mse_sum"]
            totals["frames"] += sample_metrics["frames"]

    for view_name, totals in basic_totals.items():
        if totals["frames"] <= 0:
            continue
        metrics["views"][view_name]["psnr"] = totals["psnr_sum"] / totals["frames"]
        metrics["views"][view_name]["ssim"] = totals["ssim_sum"] / totals["frames"]
        metrics["views"][view_name]["mse"] = totals["mse_sum"] / totals["frames"]
        metrics["views"][view_name]["frames"] = int(totals["frames"])

    lpips_model = lpips.LPIPS(net='alex').to(ctx.device).eval()
    lpips_totals = {view_name: {"lpips_sum": 0.0, "frames": 0} for view_name in ctx.view_names}
    for sample in tqdm(prepared_samples, desc="Computing LPIPS ...", leave=False):
        gt_tensor = torch.from_numpy(sample.gt_video).permute(0, 3, 1, 2).contiguous().float()
        pred_tensor = torch.from_numpy(sample.pred_video).permute(0, 3, 1, 2).contiguous().float()
        if gt_tensor.shape[1] != 3 or pred_tensor.shape[1] != 3:
            raise RuntimeError(
                f"LPIPS expects RGB video, got {gt_tensor.shape[1]} and {pred_tensor.shape[1]} channels"
            )
        with torch.no_grad():
            for start in range(0, gt_tensor.shape[0], BATCH_SIZE):
                gt_batch = gt_tensor[start:start + BATCH_SIZE].to(ctx.device)
                pred_batch = pred_tensor[start:start + BATCH_SIZE].to(ctx.device)
                lpips_batch = lpips_model(pred_batch, gt_batch, normalize=True)
                lpips_totals[sample.view_name]["lpips_sum"] += float(lpips_batch.sum().item())
                lpips_totals[sample.view_name]["frames"] += int(lpips_batch.shape[0])
    for view_name, totals in lpips_totals.items():
        if totals["frames"] > 0:
            metrics["views"][view_name]["lpips"] = totals["lpips_sum"] / totals["frames"]

    for metric_name in BASIC_METRICS:
        metrics["overall"][metric_name] = _mean([metrics["views"][view_name][metric_name] for view_name in ctx.view_names])
    return metrics


def _compute_distribution_metrics(prepared_samples, ctx: EvalContext):
    return {
        "overall": {
            "fid": float(_compute_fid_overall(prepared_samples, device=ctx.device)),
            "fvd": float(_compute_fvd_overall(prepared_samples, frame_chunk_size=ctx.frame_chunk_size, device=ctx.device)),
        },
        "views": {},
    }


def _print_stage_overall_metrics(stage_name, metric_names, overall_metrics):
    tqdm.write("\n" + "-" * 60)
    tqdm.write(stage_name)
    tqdm.write("-" * 60)
    for metric_name in metric_names:
        tqdm.write(f"{metric_name}: {float(overall_metrics[metric_name]):.{DEFAULT_METRIC_PRECISION}f}")


def _evaluate_video_group(ctx: EvalContext, prepared_samples):
    metrics = {
        "overall": _empty_metric_values(OVERALL_METRICS),
        "views": {
            view_name: {
                **_empty_metric_values(VIEW_METRICS),
                "frames": 0,
            }
            for view_name in ctx.view_names
        },
    }

    for compute_fn in (_compute_basic_metrics, _compute_distribution_metrics, _compute_pbench_metrics):
        group_metrics = compute_fn(prepared_samples, ctx)
        for metric_name, value in group_metrics.get("overall", {}).items():
            metrics["overall"][metric_name] = float(value)
        for view_name, view_metrics in group_metrics.get("views", {}).items():
            for metric_name, value in view_metrics.items():
                metrics["views"][view_name][metric_name] = int(value) if metric_name == "frames" else float(value)
        if compute_fn is _compute_basic_metrics:
            _print_stage_overall_metrics("BASIC METRICS READY", BASIC_METRICS, metrics["overall"])
        elif compute_fn is _compute_distribution_metrics:
            _print_stage_overall_metrics("DISTRIBUTION METRICS READY", DISTRIBUTION_METRICS, metrics["overall"])

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for view_name in metrics["views"]:
        summary = ", ".join(
            (
                f"{metric_name}={int(metrics['views'][view_name][metric_name])}"
                if metric_name == "frames"
                else f"{metric_name}={float(metrics['views'][view_name][metric_name]):.{DEFAULT_METRIC_PRECISION}f}"
            )
            for metric_name in VIEW_METRICS + ("frames",)
        )
        print(f"{view_name}: {summary}")
    print("-" * 60)
    for metric_name in OVERALL_METRICS:
        print(f"{metric_name}: {float(metrics['overall'][metric_name]):.{DEFAULT_METRIC_PRECISION}f}")
    print("\n" + "=" * 60 + "\n")
    return metrics


def evaluate(
    comparison_dir,
    num_workers=64,
    num_views=3,
    frame_chunk_size=81,
    sample_records=None,
    prompt_map_jsonl=None,
):
    """
    Unified evaluation function that computes all configured metrics and returns
    only all-view summary metrics plus per-view averages.

    Args:
        comparison_dir (str): Directory containing comparison videos (num_views rows x 2 columns)
        num_workers (int): Number of worker threads for PSNR/SSIM.
        num_views (int): Number of camera views in each comparison video.
        frame_chunk_size (int): Frames per chunk; tail frames that do not make a full chunk are dropped.
        sample_records (list[dict]|None): In-memory prompt and output-video records from inference.
        prompt_map_jsonl (str|None): Optional prompt mapping jsonl for standalone metric runs.

    Returns:
        dict: Dictionary containing:
            - 'overall': all-view summary metrics
            - 'views': per-view summary metrics
    """
    sample_records = list(sample_records or [])
    num_views = int(num_views)
    if num_views <= 0:
        raise ValueError(f"`num_views` must be positive, got {num_views}")
    frame_chunk_size = int(frame_chunk_size)
    if frame_chunk_size <= 0:
        raise ValueError(f"`frame_chunk_size` must be positive, got {frame_chunk_size}")

    comparison_root = os.path.realpath(str(comparison_dir))
    if sample_records:
        video_files = []
        for record in sample_records:
            resolved_path = os.path.realpath(str(record["video_path"]))
            if os.path.isfile(resolved_path) and (
                resolved_path == comparison_root or resolved_path.startswith(comparison_root + os.sep)
            ):
                video_files.append(resolved_path)
        video_files = sorted(set(video_files))
    else:
        video_files = sorted(
            os.path.join(root, filename)
            for root, _, files in os.walk(comparison_dir)
            for filename in files
            if filename.lower().endswith(".mp4")
        )

    print(f'Found {len(video_files)} comparison videos.')
    if not video_files:
        raise RuntimeError(f'No comparison videos found in {comparison_dir}')

    ctx = EvalContext(
        comparison_dir=comparison_dir,
        video_files=video_files,
        num_workers=num_workers,
        num_views=num_views,
        frame_chunk_size=frame_chunk_size,
        view_names=list(DEFAULT_VIEW_NAMES) if num_views == len(DEFAULT_VIEW_NAMES) else [f"view_{idx + 1}" for idx in range(num_views)],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        sample_records=sample_records,
        prompt_map_jsonl=prompt_map_jsonl,
    )
    prepared_samples = _prepare_samples(ctx)
    print(f"Prepared {len(prepared_samples)} view samples.")
    return _evaluate_video_group(ctx, prepared_samples)


def evaluate_and_write_report(
    comparison_dir,
    checkpoint_name,
    metrics_output_path,
    sample_records=None,
    num_workers=64,
    num_views=3,
    frame_chunk_size=81,
    prompt_map_jsonl=None,
):
    sample_records = list(sample_records or [])
    import re

    match = re.search(r"epoch[-_]?(\d+)", str(checkpoint_name))
    report = {
        "checkpoint": str(checkpoint_name),
        "epoch": f"epoch-{match.group(1)}" if match else str(checkpoint_name),
        "comparison_dir": str(Path(comparison_dir).resolve()),
        "overall": {},
        "views": {},
        "types": {},
    }

    root_metrics = evaluate(
        comparison_dir=comparison_dir,
        num_workers=num_workers,
        num_views=num_views,
        frame_chunk_size=frame_chunk_size,
        sample_records=sample_records,
        prompt_map_jsonl=prompt_map_jsonl,
    )
    report["overall"] = root_metrics["overall"]
    report["views"] = root_metrics["views"]

    type_groups = {}
    for record in sample_records:
        type_name = str(record.get("sample_type") or "unknown")
        type_groups.setdefault(type_name, []).append(record)

    if type_groups:
        for type_name in sorted(type_groups):
            type_dir = Path(comparison_dir) / type_name
            type_metrics = evaluate(
                comparison_dir=str(type_dir if type_dir.is_dir() else comparison_dir),
                num_workers=num_workers,
                num_views=num_views,
                frame_chunk_size=frame_chunk_size,
                sample_records=type_groups[type_name],
                prompt_map_jsonl=prompt_map_jsonl,
            )
            report["types"][type_name] = {
                "overall": type_metrics["overall"],
                "views": type_metrics["views"],
            }
    else:
        for type_dir in sorted(Path(comparison_dir).iterdir()):
            if not type_dir.is_dir():
                continue
            type_metrics = evaluate(
                comparison_dir=str(type_dir),
                num_workers=num_workers,
                num_views=num_views,
                frame_chunk_size=frame_chunk_size,
                sample_records=None,
                prompt_map_jsonl=prompt_map_jsonl,
            )
            report["types"][type_dir.name] = {
                "overall": type_metrics["overall"],
                "views": type_metrics["views"],
            }

    metrics_output_path = Path(metrics_output_path)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2, ensure_ascii=True)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison_dir', type=str, default=None)
    parser.add_argument('--workers', type=int, default=64, help='Number of worker threads for PSNR/SSIM')
    parser.add_argument('--num_views', type=int, default=3, help='Number of views in each comparison video')
    parser.add_argument('--frame_chunk_size', type=int, default=81, help='Frames per metric chunk; drop final incomplete chunk')
    parser.add_argument('--prompt_map_jsonl', type=str, default=None, help='Video-to-prompt mapping jsonl for PBench')

    args = parser.parse_args()

    results = evaluate(
        args.comparison_dir,
        num_workers=args.workers,
        num_views=args.num_views,
        frame_chunk_size=args.frame_chunk_size,
        prompt_map_jsonl=args.prompt_map_jsonl,
    )
