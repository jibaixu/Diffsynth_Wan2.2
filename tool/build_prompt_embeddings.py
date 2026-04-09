#!/usr/bin/env python3
import argparse
import json
import os
import random
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from diffsynth.core.loader import ModelConfig
from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer
from diffsynth.pipelines.wan_video import WanVideoPipeline, WanVideoUnit_PromptEmbedder


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_dtype(device: str) -> str:
    return "bfloat16" if device.startswith("cuda") else "float32"


def _resolve_torch_dtype(name: str) -> torch.dtype:
    if not hasattr(torch, name):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return getattr(torch, name)


def set_global_seed(seed: int = 42) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_pipeline(args: argparse.Namespace) -> WanVideoPipeline:
    device = args.device
    torch_dtype = _resolve_torch_dtype(args.torch_dtype)

    text_encoder_path = os.path.join(
        args.model_root, "models_t5_umt5-xxl-enc-bf16.pth"
    )
    tokenizer_path = os.path.join(args.model_root, "google", "umt5-xxl")
    if not os.path.exists(text_encoder_path):
        raise FileNotFoundError(f"Text encoder not found: {text_encoder_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")

    text_encoder_config = ModelConfig(path=text_encoder_path)

    pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)

    model_pool = pipe.download_and_load_models([text_encoder_config])
    pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
    if pipe.text_encoder is None:
        raise RuntimeError("Failed to load wan_video_text_encoder.")
    pipe.text_encoder.eval()
    pipe.text_encoder.requires_grad_(False)

    pipe.tokenizer = HuggingfaceTokenizer(
        name=tokenizer_path, seq_len=512, clean="whitespace"
    )
    return pipe


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _count_jsonl(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode task prompts into T5 embeddings and update task.jsonl."
    )
    parser.add_argument(
        "--pos-jsonl",
        default="/data1/linzengrong/Code/DiffSynth-Studio/Data/Val_new_81/tasks.jsonl",
        help="Path to positive task jsonl.",
    )
    parser.add_argument(
        "--pos-output",
        default="/data1/linzengrong/Code/DiffSynth-Studio/Data/Val_new_81/prompt_emb",
        help="Directory to store positive prompt embeddings.",
    )
    parser.add_argument(
        "--neg-prompt",
        default=(
            "The video is not of a high quality, it has a low resolution. "
            "Watermark present in each frame. The background is solid. "
            "Strange body and strange trajectory."
        ),
        help="Negative prompt to encode (single embedding mode).",
    )
    parser.add_argument(
        "--neg-output",
        default=(
            "/data1/linzengrong/Code/DiffSynth-Studio/Data/Val_new_81/prompt_emb/neg_prompt.pt"
        ),
        help="Output path for the negative embedding (single embedding mode).",
    )
    parser.add_argument(
        "--mode",
        choices=("pos", "neg"),
        default="pos",
        help="Select encoding mode: pos (jsonl batch) or neg (single prompt).",
    )
    parser.add_argument(
        "--model-root",
        default="/data1/linzengrong/Models/wan2.1/Wan2.1-Fun-V1.1-1.3B-InP",
        help="Root directory for the text encoder and tokenizer assets.",
    )
    parser.add_argument(
        "--device",
        default=_default_device(),
        help="Torch device for inference (e.g. cuda or cpu).",
    )
    parser.add_argument(
        "--torch-dtype",
        default=_default_dtype(_default_device()),
        help="Torch dtype for model weights (e.g. bfloat16, float16, float32).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip encoding if the embedding file already exists.",
    )
    args = parser.parse_args()

    if args.mode == "pos":
        if args.pos_jsonl is None:
            raise ValueError("--pos-jsonl is required when --mode=pos.")
    if args.mode == "neg":
        if args.neg_prompt is None:
            raise ValueError("--neg-prompt is required when --mode=neg.")

    set_global_seed(42)

    output_dir = os.path.abspath(args.pos_output)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output embeddings directory: {output_dir}")

    pipe = _build_pipeline(args)
    embedder = WanVideoUnit_PromptEmbedder()

    if args.mode == "neg":
        neg_output = os.path.abspath(args.neg_output)
        os.makedirs(os.path.dirname(neg_output), exist_ok=True)
        with torch.inference_mode():
            result = embedder.process(pipe, prompt=args.neg_prompt)
            neg_emb = result["context"].detach().cpu()
            torch.save(neg_emb, neg_output)
        print(f"Saved neg embedding to: {neg_output}")
        return

    pos_jsonl = os.path.abspath(args.pos_jsonl)
    tmp_path = f"{pos_jsonl}.tmp"
    total = _count_jsonl(pos_jsonl)
    with open(tmp_path, "w", encoding="utf-8") as out_handle:
        with torch.inference_mode():
            iterator = enumerate(_iter_jsonl(pos_jsonl))
            for new_index, item in tqdm(iterator, total=total, desc="Encoding prompts"):
                prompt = item.get("prompt")
                if prompt is None:
                    raise ValueError(f"Missing prompt in task index {new_index}.")

                embed_path = os.path.join(output_dir, f"pos_{new_index}.pt")
                if not (args.skip_existing and os.path.exists(embed_path)):
                    result = embedder.process(pipe, prompt=prompt)
                    prompt_emb = result["context"].detach().cpu()
                    torch.save(prompt_emb, embed_path)
                    tqdm.write(
                        f"[{new_index}] prompt_emb shape={tuple(prompt_emb.shape)} "
                        f"dtype={prompt_emb.dtype}"
                    )
                else:
                    tqdm.write(f"[{new_index}] prompt_emb exists, skipping encode.")

                item["task_index"] = new_index
                rel_base = os.path.dirname(output_dir)
                item["prompt_emb"] = os.path.relpath(embed_path, rel_base)
                out_handle.write(json.dumps(item, ensure_ascii=True) + "\n")

    os.replace(tmp_path, pos_jsonl)


if __name__ == "__main__":
    main()
