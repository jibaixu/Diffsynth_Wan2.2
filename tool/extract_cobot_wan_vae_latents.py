#!/usr/bin/env python3
"""Extract Wan2.2 VAE latents for Cobot clipped single-view videos."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Iterable

import torch
import imageio.v2 as imageio
from PIL import Image
from tqdm import tqdm

from diffsynth.configs import MODEL_CONFIGS
from diffsynth.core.data.operators import ImageCropAndResize, LoadVideo, ToVideoTensor
from diffsynth.core.loader.file import hash_model_file
from diffsynth.core.loader.model import load_model


VIEW_NAMES = (
    "observation.images.cam_high_rgb",
    "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_right_wrist_rgb",
)
ALL_VIEWS_NAME = "observation.images.cam_all_views_rgb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Wan2.2 VAE latents for Cobot clipped single-view videos."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/linzengrong/Datasets/Cobot_Magic_all/Cobot_Magic_cut_banana"),
        help="Dataset root containing videos_clipped and meta/",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("meta/episodes_clipped.jsonl"),
        help="Input metadata path, relative to dataset root unless absolute.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("meta/episodes_latents.jsonl"),
        help="Output metadata path, relative to dataset root unless absolute.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("videos_clipped"),
        help="Input video root relative to dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("latents"),
        help="Output latent root relative to dataset root.",
    )
    parser.add_argument(
        "--chunk-name",
        type=str,
        default="chunk-000",
        help="Only process metadata rows from this chunk.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/linzengrong/Models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
        help="Local Wan2.2 VAE checkpoint path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="VAE compute dtype.",
    )
    parser.add_argument(
        "--save-dtype",
        type=str,
        default="float16",
        choices=("float32", "float16", "bfloat16"),
        help="Latent save dtype.",
    )
    parser.add_argument(
        "--spatial-division-factor",
        type=int,
        default=32,
        help="Height/width divisibility used during preprocessing.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=288,
        help="Target fit-box height used during preprocessing.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Target fit-box width used during preprocessing.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1920 * 1080,
        help="Max pixels for ImageCropAndResize.",
    )
    parser.add_argument(
        "--resize-mode",
        type=str,
        default="fit",
        choices=("fit", "crop"),
        help="Resize mode for ImageCropAndResize.",
    )
    parser.add_argument(
        "--save-suffix",
        type=str,
        default=".pth",
        help="Latent output file suffix.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing latent files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N matching episodes; 0 means all.",
    )
    return parser.parse_args()


def resolve_under_root(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def parse_torch_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return table[name]


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def validate_device_choice(requested_device: str, resolved_device: str) -> None:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {requested_device!r}, but torch.cuda.is_available() is False in this environment."
        )
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Resolved CUDA device {resolved_device!r}, but torch.cuda.is_available() is False in this environment."
        )


def resolve_compute_dtype(device: str, dtype_arg: str) -> torch.dtype:
    if dtype_arg != "auto":
        dtype = parse_torch_dtype(dtype_arg)
        if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return dtype
    if device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def import_from_string(path: str):
    module_name, _, attr_name = path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def find_model_config(model_path: Path, model_name: str) -> dict:
    model_hash = hash_model_file(str(model_path))
    matches = [
        config for config in MODEL_CONFIGS
        if config["model_hash"] == model_hash and config["model_name"] == model_name
    ]
    if not matches:
        raise ValueError(
            f"Could not find MODEL_CONFIGS entry for {model_path} (hash={model_hash}, model_name={model_name})."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous MODEL_CONFIGS matches for {model_path} (hash={model_hash}, model_name={model_name})."
        )
    return matches[0]


def load_wan_vae(model_path: Path, device: str, compute_dtype: torch.dtype):
    config = find_model_config(model_path, model_name="wan_video_vae")
    model_class = import_from_string(config["model_class"])
    state_dict_converter = None
    if "state_dict_converter" in config:
        state_dict_converter = import_from_string(config["state_dict_converter"])
    vae = load_model(
        model_class,
        str(model_path),
        config=dict(config.get("extra_kwargs", {})),
        torch_dtype=compute_dtype,
        device=device,
        state_dict_converter=state_dict_converter,
        use_disk_map=True,
    )
    if not hasattr(vae, "encode"):
        raise TypeError(f"Loaded model does not expose encode(): {type(vae).__name__}")
    return vae


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def save_tensor_atomic(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(tensor, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def to_posix_relative(path: PurePosixPath) -> str:
    return path.as_posix()


def validate_saved_latent(path: Path) -> tuple[int, ...]:
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    if tensor.ndim != 5:
        raise ValueError(f"Expected latent tensor with shape (1,C,T,H,W), got {tuple(tensor.shape)} at {path}")
    if int(tensor.shape[0]) != 1:
        raise ValueError(f"Expected leading view dimension 1, got {tuple(tensor.shape)} at {path}")
    return tuple(int(x) for x in tensor.shape)


def record_matches_chunk(record: dict, input_root: PurePosixPath, chunk_name: str) -> bool:
    video_rel = normalize_metadata_video_rel(record["video"], input_root=input_root)
    parts = video_rel.parts
    if len(parts) < 4:
        return False
    if str(chunk_name).lower() in {"all", "*"}:
        return True
    return parts[0] == input_root.as_posix() and parts[1] == chunk_name


def normalize_metadata_video_rel(video_value, *, input_root: PurePosixPath) -> PurePosixPath:
    video_rel = PurePosixPath(str(video_value))
    parts = list(video_rel.parts)
    if len(parts) < 4:
        raise ValueError(f"Unexpected video path in metadata: {video_rel}")

    root_name = parts[0]
    accepted_roots = {input_root.as_posix(), "videos", "videos_clipped"}
    if root_name not in accepted_roots:
        raise ValueError(
            f"Unsupported video root {root_name!r} in metadata path {video_rel}. "
            f"Expected one of {sorted(accepted_roots)}."
        )
    parts[0] = input_root.as_posix()
    return PurePosixPath(*parts)


def build_view_video_rel(record: dict, view_name: str, *, input_root: PurePosixPath) -> PurePosixPath:
    src_video = normalize_metadata_video_rel(record["video"], input_root=input_root)
    parts = list(src_video.parts)
    if parts[2] != ALL_VIEWS_NAME:
        raise ValueError(
            f"Expected metadata to point at {ALL_VIEWS_NAME}, got {parts[2]} for {src_video}"
        )
    parts[2] = view_name
    return PurePosixPath(*parts)


def build_latent_rel(
    output_root: PurePosixPath,
    input_root: PurePosixPath,
    src_video_rel: PurePosixPath,
    suffix: str,
) -> PurePosixPath:
    relative_tail = src_video_rel.relative_to(input_root)
    stem = Path(relative_tail.name).stem
    return output_root / relative_tail.parent / f"{stem}{suffix}"


def selected_frame_count(record: dict) -> int:
    if "frame_indices" in record and record["frame_indices"] is not None:
        return len(record["frame_indices"])
    if "start_frame" in record and "end_frame" in record:
        return int(record["end_frame"]) - int(record["start_frame"]) + 1
    if "length" in record:
        return int(record["length"])
    raise KeyError("Cannot infer frame count from record.")


def build_video_loader(args: argparse.Namespace) -> LoadVideo:
    frame_processor = ImageCropAndResize(
        height=args.height,
        width=args.width,
        max_pixels=args.max_pixels,
        height_division_factor=args.spatial_division_factor,
        width_division_factor=args.spatial_division_factor,
        resize_mode=args.resize_mode,
    )
    return LoadVideo(
        num_frames=100001,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=frame_processor,
    )


def probe_video_shape(video_path: Path) -> tuple[int, int]:
    reader = imageio.get_reader(str(video_path))
    try:
        frame = reader.get_data(0)
    finally:
        reader.close()
    return int(frame.shape[0]), int(frame.shape[1])


def probe_preprocessed_shape(
    video_path: Path,
    *,
    frame_processor,
) -> tuple[int, int]:
    reader = imageio.get_reader(str(video_path))
    try:
        frame = Image.fromarray(reader.get_data(0))
    finally:
        reader.close()
    processed = frame_processor(frame)
    width, height = processed.size
    return int(height), int(width)


def prepare_video_tensor(
    loader: LoadVideo,
    to_video_tensor: ToVideoTensor,
    video_path: Path,
    record: dict,
) -> torch.Tensor:
    payload = {"data": str(video_path)}
    if "start_frame" in record:
        payload["start_frame"] = record["start_frame"]
    if "end_frame" in record:
        payload["end_frame"] = record["end_frame"]
    if "frame_indices" in record:
        payload["frame_indices"] = record["frame_indices"]
    frames = loader(payload)
    return to_video_tensor(frames)


def maybe_encode_latent(
    vae,
    loader: LoadVideo,
    to_video_tensor: ToVideoTensor,
    record: dict,
    src_video_path: Path,
    dst_latent_path: Path,
    device: str,
    compute_dtype: torch.dtype,
    save_dtype: torch.dtype,
    overwrite: bool,
) -> dict:
    if not src_video_path.is_file():
        raise FileNotFoundError(f"Missing source video: {src_video_path}")

    source_shape_hw = probe_video_shape(src_video_path)
    preprocessed_shape_hw = probe_preprocessed_shape(
        src_video_path,
        frame_processor=loader.frame_processor,
    )

    if dst_latent_path.exists() and not overwrite:
        existing_shape = validate_saved_latent(dst_latent_path)
        return {
            "source_shape_hw": source_shape_hw,
            "video_shape": (
                1,
                3,
                selected_frame_count(record),
                preprocessed_shape_hw[0],
                preprocessed_shape_hw[1],
            ),
            "latent_shape": existing_shape,
            "skipped": True,
        }

    video_tensor = prepare_video_tensor(loader, to_video_tensor, src_video_path, record)
    video_shape = tuple(int(x) for x in video_tensor.shape)
    video_tensor = video_tensor.to(device=device, dtype=compute_dtype)
    latent = vae.encode(video_tensor, device=device)
    latent = latent.to(device="cpu", dtype=save_dtype).contiguous()
    save_tensor_atomic(dst_latent_path, latent)

    del video_tensor
    del latent
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "source_shape_hw": source_shape_hw,
        "video_shape": video_shape,
        "latent_shape": validate_saved_latent(dst_latent_path),
        "skipped": False,
    }


def run_extraction(args: argparse.Namespace) -> dict:
    dataset_root = args.dataset_root.resolve()
    metadata_path = resolve_under_root(dataset_root, args.metadata_path).resolve()
    metadata_output = resolve_under_root(dataset_root, args.metadata_output).resolve()
    model_path = args.model_path.resolve()
    input_root = PurePosixPath(args.input_root.as_posix())
    output_root = PurePosixPath(args.output_root.as_posix())

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = resolve_device(args.device)
    validate_device_choice(args.device, device)
    compute_dtype = resolve_compute_dtype(device, args.compute_dtype)
    save_dtype = parse_torch_dtype(args.save_dtype)

    print(f"Dataset root: {dataset_root}")
    print(f"Input metadata: {metadata_path}")
    print(f"Output metadata: {metadata_output}")
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    print(f"Compute dtype: {compute_dtype}")
    print(f"Save dtype: {save_dtype}")
    print(f"Chunk: {args.chunk_name}")
    print(f"Output root: {output_root.as_posix()}")
    print(f"Resize target: {args.height}x{args.width} ({args.resize_mode})")
    print("Latent spatial size is derived as: preprocessed_height/16 x preprocessed_width/16")

    rows = load_jsonl(metadata_path)
    rows = [row for row in rows if record_matches_chunk(row, input_root=input_root, chunk_name=args.chunk_name)]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No metadata rows found for chunk {args.chunk_name} in {metadata_path}")

    loader = build_video_loader(args)
    to_video_tensor = ToVideoTensor()
    sample_video_rel = build_view_video_rel(rows[0], VIEW_NAMES[0], input_root=input_root)
    sample_video_path = (dataset_root / sample_video_rel).resolve()
    if sample_video_path.is_file():
        sample_source_hw = probe_video_shape(sample_video_path)
        sample_pre_hw = probe_preprocessed_shape(sample_video_path, frame_processor=loader.frame_processor)
        print(
            "Sample shape preview: "
            f"source_hw={sample_source_hw} "
            f"preprocessed_hw={sample_pre_hw} "
            f"expected_latent_hw=({sample_pre_hw[0] // 16}, {sample_pre_hw[1] // 16})"
        )
    vae = load_wan_vae(model_path, device=device, compute_dtype=compute_dtype)

    metadata_rows_out: list[dict] = []
    total_view_files = 0
    for row in tqdm(rows, desc="Episodes"):
        src_all_views = normalize_metadata_video_rel(row["video"], input_root=input_root)
        episode_file = src_all_views.name
        per_view_shapes: list[tuple[int, ...]] = []
        latent_relpaths: list[str] = []

        for view_name in VIEW_NAMES:
            src_video_rel = build_view_video_rel(row, view_name, input_root=input_root)
            dst_latent_rel = build_latent_rel(
                output_root=output_root,
                input_root=input_root,
                src_video_rel=src_video_rel,
                suffix=args.save_suffix,
            )
            src_video_path = (dataset_root / src_video_rel).resolve()
            dst_latent_path = (dataset_root / dst_latent_rel).resolve()

            stats = maybe_encode_latent(
                vae=vae,
                loader=loader,
                to_video_tensor=to_video_tensor,
                record=row,
                src_video_path=src_video_path,
                dst_latent_path=dst_latent_path,
                device=device,
                compute_dtype=compute_dtype,
                save_dtype=save_dtype,
                overwrite=args.overwrite,
            )
            per_view_shapes.append(stats["latent_shape"])
            latent_relpaths.append(to_posix_relative(dst_latent_rel))
            total_view_files += 1
            status = "reused" if stats["skipped"] else "saved"
            print(
                f"[{episode_file}] {view_name}: "
                f"source_hw={stats['source_shape_hw']} "
                f"preprocessed_vcthw={stats['video_shape']} "
                f"latent={stats['latent_shape']} "
                f"status={status}"
            )

        if len(set(per_view_shapes)) != 1:
            shape_str = ", ".join(f"{view}:{shape}" for view, shape in zip(VIEW_NAMES, per_view_shapes))
            raise ValueError(f"Latent shape mismatch across views for {episode_file}: {shape_str}")

        row_out = dict(row)
        row_out["video"] = latent_relpaths
        metadata_rows_out.append(row_out)

    write_jsonl_atomic(metadata_output, metadata_rows_out)

    print(f"Processed episodes: {len(metadata_rows_out)}")
    print(f"Processed view files: {total_view_files}")
    print(f"Wrote metadata: {metadata_output}")
    return {
        "dataset_root": dataset_root,
        "metadata_output": metadata_output,
        "processed_episodes": len(metadata_rows_out),
        "processed_view_files": total_view_files,
    }


def main() -> None:
    args = parse_args()
    run_extraction(args)


if __name__ == "__main__":
    main()
