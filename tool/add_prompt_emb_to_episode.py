#!/usr/bin/env python3
# 该脚本用 task_jsonl 的 prompt 匹配 prompt_emb，并写入到 val_jsonl。
import argparse
import json
import os
import sys


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalize_prompt_emb(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return os.path.relpath(path, base_dir)
    return path


def _load_prompt_map(task_jsonl: str, base_dir: str) -> dict:
    prompt_map = {}
    duplicate_count = 0
    for item in _iter_jsonl(task_jsonl):
        prompt = item.get("prompt")
        prompt_emb = item.get("prompt_emb")
        if prompt is None or prompt_emb is None:
            continue
        prompt_emb = _normalize_prompt_emb(prompt_emb, base_dir)
        if prompt not in prompt_map:
            prompt_map[prompt] = prompt_emb
        else:
            if prompt_map[prompt] != prompt_emb:
                duplicate_count += 1
    if duplicate_count:
        print(
            f"Warning: {duplicate_count} duplicate prompts found in task.jsonl; "
            "using the first prompt_emb value.",
            file=sys.stderr,
        )
    return prompt_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add prompt_emb field to val.jsonl by matching prompts from task.jsonl."
    )
    parser.add_argument(
        "--val-jsonl",
        default="/data1/Projects/DiffSynth-Studio/Data/Datasets/meta/val_new_81.jsonl",
        help="Path to val.jsonl.",
    )
    parser.add_argument(
        "--task-jsonl",
        default="/data1/Projects/DiffSynth-Studio/Data/Datasets/meta/tasks.jsonl",
        help="Path to task.jsonl containing prompt_emb.",
    )
    args = parser.parse_args()

    val_jsonl = os.path.abspath(args.val_jsonl)
    task_jsonl = os.path.abspath(args.task_jsonl)
    base_dir = os.path.dirname(val_jsonl)

    prompt_map = _load_prompt_map(task_jsonl, base_dir)
    if not prompt_map:
        raise ValueError("No prompt_emb entries found in task.jsonl.")

    tmp_path = f"{val_jsonl}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as out_handle:
        for item in _iter_jsonl(val_jsonl):
            prompt = item.get("prompt")
            if prompt is None:
                raise ValueError("Missing prompt in val.jsonl entry.")
            if prompt not in prompt_map:
                raise KeyError(f"Prompt not found in task.jsonl: {prompt}")
            item["prompt_emb"] = prompt_map[prompt]
            out_handle.write(json.dumps(item, ensure_ascii=True) + "\n")

    os.replace(tmp_path, val_jsonl)


if __name__ == "__main__":
    main()
