#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import requests

DEFAULT_HOST = "http://127.0.0.1:5454/v1/chat/completions"
DEFAULT_VLA_FILTER_EPISODES_JSONL = (
    "/data1/linzengrong/Code/DiffSynth-Studio/Ckpt/3_14_real_robot_history1/epoch-149/output_vla_lerobot/batch_20260314_230724/meta/episodes.jsonl"
)
DEFAULT_VLA_FILTER_COMPARISON_ROOT = (
    "/data1/linzengrong/Code/DiffSynth-Studio/Ckpt/3_14_real_robot_history1/epoch-149/output_vla_lerobot/batch_20260314_230724/videos/chunk-000/comparison"
)
DEFAULT_VLA_FILTER_OUT_DIR = (
    "/data1/linzengrong/Code/DiffSynth-Studio/Ckpt/3_14_real_robot_history1/epoch-149/output_vla_lerobot/batch_20260314_230724/meta/vla_filter_qwen35"
)
DEFAULT_VLA_FILTER_PROMPT_TEMPLATE = (
    "You are given sampled frames from a robot manipulation comparison video. "
    "IMPORTANT: frames come from the RIGHT HALF only. In this right-half video, TOP is third-person view and "
    "BOTTOM is first-person view. Evaluate against instruction: \"{instruction}\". "
    "Return STRICT JSON only (no markdown): "
    "{\"instruction_score\":1,\"multi_view_score\":1,\"artifact_score\":1,"
    "\"issues\":[\"issue1\"],\"evidence_pairs\":[\"t=01 top:... | bottom:...\"],"
    "\"hard_fail_triggers\":[\"trigger1\"],\"reason\":\"short reason\"}. "
    "Scoring protocol (must follow exactly): "
    "A) First detect problems, then score. Never default to 5. "
    "B) Hard gates (cap score <= 2 for related dimension): "
    "B1) If same-timestamp TOP/BOTTOM event-state mismatch exists (grasp/place/contact/object-state conflict), "
    "then multi_view_score must be <= 2 and add issue view_mismatch. "
    "B2) If temporal anomalies exist (object duplication, sudden disappearance, premature adhesion before gripper "
    "closure/contact), then artifact_score must be <= 2 and add corresponding issue(s). "
    "B3) Focus on the gripper and the main task object; if object appears attached to gripper before valid "
    "closure/contact, mark premature_adhesion and cap artifact_score <= 2. "
    "B4) If final frames do not show task goal completion, cap instruction_score <= 2. "
    "C) Perspective/scale/appearance differences between TOP and BOTTOM are normal; do NOT treat them as errors. "
    "D) Do NOT output spatial_distortion. "
    "E) 1-5 anchors: "
    "instruction_score: 5 clear completion; 4 mostly complete with minor uncertainty; 3 partial completion; "
    "2 mostly incomplete; 1 clearly not done. "
    "multi_view_score: 5 consistent at key timestamps; 4 minor non-critical mismatch; 3 some mismatch but "
    "partially explainable; 2 clear repeated conflict; 1 severe contradiction. "
    "artifact_score: 5 no anomaly; 4 weak suspicion only; 3 minor anomaly; 2 one clear anomaly; "
    "1 multiple/severe anomalies. "
    "F) issues must be lowercase tags from: view_mismatch, object_duplication, object_disappearance, "
    "premature_adhesion. Return [] if none. "
    "G) If issues is not empty, do not output all three scores as 5. "
    "H) evidence_pairs must include at least 2 timestamped top-vs-bottom observations; if unavailable, scores "
    "must not exceed 3. "
    "I) hard_fail_triggers must include any of: instruction_not_completed, cross_view_state_conflict, "
    "object_duplication, object_disappearance, premature_adhesion. "
    "J) reason is one short English sentence."
)

DEFAULT_CHAT_MAX_TOKENS = 8196
DEFAULT_CHAT_TEMPERATURE = 0.7
DEFAULT_CHAT_TOP_P = 0.8
DEFAULT_CHAT_TOP_K = 20
DEFAULT_CHAT_MIN_P = 0.0
DEFAULT_CHAT_PRESENCE_PENALTY = 1.5
DEFAULT_CHAT_REPETITION_PENALTY = 1.0
DEFAULT_VLA_EVAL_TEMPERATURE = 0.0
DEFAULT_VLA_EVAL_TOP_P = 1.0
DEFAULT_VLA_EVAL_TOP_K = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3.5 VLA filter: keep only instruction-complete and artifact-free videos."
    )
    parser.add_argument("task", nargs="?", default="vla_filter", choices=["vla_filter"])
    parser.add_argument("--episodes-jsonl", default=DEFAULT_VLA_FILTER_EPISODES_JSONL)
    parser.add_argument("--comparison-root", default=DEFAULT_VLA_FILTER_COMPARISON_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_VLA_FILTER_OUT_DIR)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--model", default="auto", help="Model name or 'auto'.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3, help="-1 means retry forever.")
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--line-start", type=int, default=0, help="0-based start line index.")
    parser.add_argument("--max-videos", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--rank", default="0/1", help="Rank shard, format i/n.")
    parser.add_argument(
        "--episode-ids",
        default="",
        help="Optional comma-separated episode ids (e.g. 64,88,96) to run targeted regression only.",
    )
    parser.add_argument(
        "--expectation-jsonl",
        default="",
        help="Optional jsonl of per-episode expected constraints; adds meets_expectation to output.",
    )
    parser.add_argument("--sample-frames", type=int, default=0, help="Frames per video; 0 means use all frames.")
    parser.add_argument(
        "--max-frames-cap",
        type=int,
        default=32,
        help="Hard cap on frames sent to model; 0 means no cap.",
    )
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        default=False,
        help="Remove existing rank output files before writing current run.",
    )
    parser.add_argument("--prompt-template", default=DEFAULT_VLA_FILTER_PROMPT_TEMPLATE)
    return parser.parse_args()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Invalid record type at {path}:{line_no}; expect object")
            records.append(row)
    return records


def collect_jsonl_records(path: Path, line_start: int = 0) -> list[tuple[int, dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"jsonl not found: {path}")
    records: list[tuple[int, dict[str, Any]]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line_index < max(0, line_start):
                continue
            text = line.strip()
            if not text:
                continue
            try:
                entry = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            records.append((line_index, entry))
    return records


def parse_rank_spec(rank_spec: str) -> tuple[int, int]:
    text = (rank_spec or "").strip()
    if not text:
        return 0, 1
    if "/" not in text:
        raise ValueError(f"Invalid --rank format: {rank_spec}. Expected i/n.")
    left, right = text.split("/", 1)
    rank = int(left)
    all_ranks = int(right)
    if rank < 0 or all_ranks <= 0 or rank >= all_ranks:
        raise ValueError(f"Invalid --rank values: {rank_spec}")
    return rank, all_ranks


def parse_episode_ids_filter(text: str) -> set[str]:
    selected: set[str] = set()
    for token in str(text or "").split(","):
        raw = token.strip()
        if not raw:
            continue
        if raw.isdigit():
            selected.add(f"{int(raw):05d}")
        else:
            selected.add(raw)
    return selected


def load_expectation_map(path_text: str) -> dict[str, dict[str, Any]]:
    if not str(path_text or "").strip():
        return {}
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"expectation file not found: {path}")
    expectations: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                continue
            episode_raw = str(row.get("episode_id") or "").strip()
            if not episode_raw:
                continue
            episode_id = f"{int(episode_raw):05d}" if episode_raw.isdigit() else episode_raw
            expectations[episode_id] = row
    return expectations


def parse_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = re.split(r"[,;\n]", str(value))
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_values:
        text = str(item or "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def evaluate_expectation(row: dict[str, Any], expectation: dict[str, Any]) -> bool:
    def _to_int(name: str) -> int | None:
        value = expectation.get(name)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    checks: list[bool] = []
    max_instruction = _to_int("max_instruction_score")
    if max_instruction is not None:
        checks.append(int(row.get("instruction_score", 99)) <= max_instruction)
    max_multi = _to_int("max_multi_view_score")
    if max_multi is not None:
        checks.append(int(row.get("multi_view_score", 99)) <= max_multi)
    max_artifact = _to_int("max_artifact_score")
    if max_artifact is not None:
        checks.append(int(row.get("artifact_score", 99)) <= max_artifact)
    min_total = _to_int("min_score")
    if min_total is not None:
        checks.append(int(row.get("score", -1)) >= min_total)
    max_total = _to_int("max_score")
    if max_total is not None:
        checks.append(int(row.get("score", 99)) <= max_total)

    issues = set(parse_string_list(row.get("issues")))
    require_any = parse_string_list(expectation.get("require_issues_any"))
    if require_any:
        checks.append(any(tag in issues for tag in require_any))
    require_all = parse_string_list(expectation.get("require_issues_all"))
    if require_all:
        checks.append(all(tag in issues for tag in require_all))

    expected_passed = expectation.get("passed")
    if isinstance(expected_passed, bool):
        checks.append(bool(row.get("passed")) == expected_passed)
    return all(checks) if checks else True


def detect_textual_contradictions(
    instruction_score: int,
    multi_view_score: int,
    artifact_score: int,
    issues: list[str],
    hard_fail_triggers: list[str],
    evidence_pairs: list[str],
    reason: str,
) -> list[str]:
    reasons: list[str] = []
    joined_text = " ".join([reason, *evidence_pairs]).lower()
    completion_claim = any(
        key in joined_text
        for key in (
            "completed",
            "successfully",
            "placed into",
            "stacked",
            "task is completed",
            "goal achieved",
        )
    )
    non_completion_claim = any(
        key in joined_text
        for key in (
            "not completed",
            "was not completed",
            "fails to",
            "failed to",
            "did not",
            "never",
            "not executed",
            "not achieved",
        )
    )

    if instruction_score <= 2 and completion_claim:
        reasons.append("contradiction_instruction_low_but_completion_claim")
    if instruction_score >= 4 and non_completion_claim:
        reasons.append("contradiction_instruction_high_but_noncompletion_claim")
    if len(evidence_pairs) < 2:
        reasons.append("insufficient_evidence_pairs")

    issue_set = set(issues)
    trigger_set = set(hard_fail_triggers)
    if (instruction_score <= 2 or multi_view_score <= 2 or artifact_score <= 2) and not (issue_set or trigger_set):
        reasons.append("low_score_without_issues_or_triggers")
    if issue_set and instruction_score == 5 and multi_view_score == 5 and artifact_score == 5:
        reasons.append("issues_present_but_all_scores_full")
    if "cross_view_state_conflict" in trigger_set and "view_mismatch" not in issue_set:
        reasons.append("trigger_view_conflict_without_view_issue")
    if any(tag in trigger_set for tag in ("object_duplication", "object_disappearance", "premature_adhesion")) and not (
        {"object_duplication", "object_disappearance", "premature_adhesion"} & issue_set
    ):
        reasons.append("artifact_trigger_without_artifact_issue")
    return reasons


def host_base(host: str) -> str:
    url = host.rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


def detect_model(session: requests.Session, host: str, timeout: int) -> str:
    base = host_base(host)
    if not base.endswith("/v1"):
        base += "/v1"
    response = session.get(base + "/models", timeout=timeout)
    response.raise_for_status()
    models = response.json().get("data", [])
    ids = [str(item.get("id", "")).strip() for item in models if item.get("id")]
    if not ids:
        raise RuntimeError("No model ids found from /v1/models")
    for model_id in ids:
        if "qwen" in model_id.lower():
            return model_id
    return ids[0]


def resolve_model_name(session: requests.Session, model: str, host: str, timeout: int) -> str:
    if model and model.lower() != "auto":
        return model
    return detect_model(session, host, timeout)


def image_to_base64(frame: Any) -> str:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Failed to encode frame")
    return base64.b64encode(encoded).decode("ascii")


def parse_message_content(data: dict[str, Any]) -> str:
    message = data["choices"][0]["message"]["content"]
    if isinstance(message, str):
        return message.strip()
    if isinstance(message, list):
        parts: list[str] = []
        for part in message:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join(parts).strip()
    return str(message).strip()


def strip_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    return text.replace("```", "").strip()


def strip_think_blocks(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()


def sanitize_model_text(text: str) -> str:
    cleaned = strip_think_blocks(text)
    return cleaned if cleaned else text.strip()


def find_balanced_json(text: str, start: int) -> str | None:
    if start < 0 or start >= len(text):
        return None
    opening = text[start]
    if opening not in "[{":
        return None
    closing = "]" if opening == "[" else "}"
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == opening:
            depth += 1
            continue
        if char == closing:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def close_truncated_json_object(text: str, start: int) -> str | None:
    if start < 0 or start >= len(text):
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            continue
    if depth <= 0:
        return text[start:]
    return text[start:] + ("}" * depth)


def parse_json_payload_with_meta(text: str) -> tuple[object, bool]:
    cleaned_text = sanitize_model_text(text)
    for candidate in (strip_fences(cleaned_text), strip_fences(text), cleaned_text.strip(), text.strip()):
        if not candidate:
            continue
        try:
            return json.loads(candidate), False
        except json.JSONDecodeError:
            pass

        # Prefer object extraction/repair to avoid accidentally parsing an inner list.
        object_start = candidate.find("{")
        if object_start >= 0:
            snippet = find_balanced_json(candidate, object_start)
            if snippet:
                try:
                    payload = json.loads(snippet)
                    if isinstance(payload, dict):
                        return payload, False
                except json.JSONDecodeError:
                    pass
            repaired = close_truncated_json_object(candidate, object_start)
            if repaired:
                try:
                    payload = json.loads(repaired)
                    if isinstance(payload, dict):
                        return payload, True
                except json.JSONDecodeError:
                    pass

        starts = [index for index, char in enumerate(candidate) if char in "[{"]
        for start in starts:
            snippet = find_balanced_json(candidate, start)
            if not snippet:
                continue
            try:
                return json.loads(snippet), False
            except json.JSONDecodeError:
                continue
    raise ValueError("No valid JSON found in model response")


def parse_json_payload(text: str) -> object:
    payload, _ = parse_json_payload_with_meta(text)
    return payload


def coerce_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y", "pass", "passed", "ok"}:
        return True
    if text in {"false", "0", "no", "n", "fail", "failed"}:
        return False
    raise ValueError(f"Invalid boolean value for {field_name}: {value!r}")


def coerce_score_1_to_5(value: Any, field_name: str) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid score for {field_name}: {value!r}") from exc
    if number < 1 or number > 5:
        raise ValueError(f"Score out of range for {field_name}: {number}; expect 1..5")
    return number


def parse_issue_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = re.split(r"[,;\n]", value)
    else:
        raw_items = [value]
    out: list[str] = []
    seen: set[str] = set()
    allowed_tags = {"view_mismatch", "object_duplication", "object_disappearance", "premature_adhesion"}
    alias_to_tag = {
        "view mismatch": "view_mismatch",
        "cross view inconsistency": "view_mismatch",
        "cross-view inconsistency": "view_mismatch",
        "cross_view_inconsistency": "view_mismatch",
        "multi view inconsistent": "view_mismatch",
        "multiview inconsistency": "view_mismatch",
        "temporal inconsistency": "view_mismatch",
        "temporal_inconsistency": "view_mismatch",
        "spatial inconsistency": "view_mismatch",
        "spatial_inconsistency": "view_mismatch",
        "object duplication": "object_duplication",
        "sudden object duplication": "object_duplication",
        "object_disappear": "object_disappearance",
        "object disappearance": "object_disappearance",
        "sudden object disappearance": "object_disappearance",
        "premature adhesion": "premature_adhesion",
        "abnormal adhesion": "premature_adhesion",
        "abnormal suction": "premature_adhesion",
        "adhesion before closure": "premature_adhesion",
        "object sticks before closure": "premature_adhesion",
    }
    for item in raw_items:
        text = str(item or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        normalized = text.replace("-", " ").replace("_", " ")
        if "spatial distortion" in normalized:
            continue
        mapped = alias_to_tag.get(normalized, normalized.replace(" ", "_"))
        if mapped not in allowed_tags:
            continue
        if mapped in seen:
            continue
        seen.add(mapped)
        out.append(mapped)
    return out


def parse_evidence_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = re.split(r"[,;\n]", value)
    else:
        raw_items = [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = str(item or "").strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(text)
    return out


def parse_hard_fail_triggers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = re.split(r"[,;\n]", value)
    else:
        raw_items = [value]
    allowed = {
        "instruction_not_completed",
        "cross_view_state_conflict",
        "object_duplication",
        "object_disappearance",
        "premature_adhesion",
    }
    alias = {
        "incomplete_task": "instruction_not_completed",
        "instruction not completed": "instruction_not_completed",
        "view_mismatch": "cross_view_state_conflict",
        "cross_view_inconsistency": "cross_view_state_conflict",
        "cross-view inconsistency": "cross_view_state_conflict",
    }
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = str(item or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        normalized = text.replace("-", "_").replace(" ", "_")
        mapped = alias.get(text, alias.get(normalized, normalized))
        if mapped not in allowed or mapped in seen:
            continue
        seen.add(mapped)
        out.append(mapped)
    return out


def parse_vla_filter_result(payload: object, raw_text: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = parse_json_payload(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("vla_filter expects a JSON object response")

    instruction_score = coerce_score_1_to_5(payload.get("instruction_score"), "instruction_score")
    multi_view_score = coerce_score_1_to_5(payload.get("multi_view_score"), "multi_view_score")
    artifact_score = coerce_score_1_to_5(payload.get("artifact_score"), "artifact_score")
    issues = parse_issue_list(payload.get("issues"))
    evidence_pairs = parse_evidence_list(payload.get("evidence_pairs"))
    if not evidence_pairs:
        evidence_pairs = parse_evidence_list(payload.get("evidence"))
    hard_fail_triggers = parse_hard_fail_triggers(payload.get("hard_fail_triggers"))
    reason = str(payload.get("reason") or "").strip()
    if not reason:
        reason = "no reason provided"

    return {
        "instruction_score": instruction_score,
        "multi_view_score": multi_view_score,
        "artifact_score": artifact_score,
        "issues": issues,
        "evidence_pairs": evidence_pairs,
        "hard_fail_triggers": hard_fail_triggers,
        "reason": reason,
    }


def merge_payload_options(base_payload: dict[str, Any], options: dict[str, Any] | None) -> dict[str, Any]:
    if not options:
        return base_payload
    payload = dict(base_payload)
    for key, value in options.items():
        if key == "extra_body":
            existing = payload.get("extra_body")
            merged = dict(existing) if isinstance(existing, dict) else {}
            if isinstance(value, dict):
                merged.update(value)
            payload["extra_body"] = merged
            continue
        payload[key] = value
    return payload


def call_vllm_caption_once(
    session: requests.Session,
    host: str,
    model: str,
    prompt: str,
    frames: list[Any],
    timeout: int,
    request_options: dict[str, Any] | None = None,
) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(frame)}"},
        }
        for frame in frames
    )
    base_payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": DEFAULT_CHAT_MAX_TOKENS,
        "temperature": DEFAULT_CHAT_TEMPERATURE,
        "top_p": DEFAULT_CHAT_TOP_P,
        "min_p": DEFAULT_CHAT_MIN_P,
        "presence_penalty": DEFAULT_CHAT_PRESENCE_PENALTY,
        "repetition_penalty": DEFAULT_CHAT_REPETITION_PENALTY,
        "extra_body": {
            "top_k": DEFAULT_CHAT_TOP_K,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    payload = merge_payload_options(base_payload, request_options)
    response = session.post(host, json=payload, timeout=timeout)
    if not response.ok:
        detail = response.text.strip().replace("\n", " ")
        if len(detail) > 300:
            detail = detail[:300] + "..."
        raise RuntimeError(f"HTTP {response.status_code}: {detail}")
    chat_response = parse_message_content(response.json())
    print("Chat response:", chat_response, flush=True)
    return chat_response


def call_vllm_with_retry(
    session: requests.Session,
    host: str,
    model: str,
    prompt: str,
    frames: list[Any],
    timeout: int,
    max_retries: int,
    retry_sleep: float,
    episode_tag: str,
    request_options: dict[str, Any] | None = None,
) -> str:
    attempts = 0
    while True:
        attempts += 1
        try:
            return call_vllm_caption_once(
                session=session,
                host=host,
                model=model,
                prompt=prompt,
                frames=frames,
                timeout=timeout,
                request_options=request_options,
            )
        except Exception as exc:
            if max_retries >= 0 and attempts > max_retries:
                raise RuntimeError(
                    f"Retry exhausted: episode={episode_tag}, retries={attempts - 1}, error={exc}"
                ) from exc
            if attempts == 1 or attempts % 10 == 0:
                print(
                    f"Retrying episode={episode_tag} attempt={attempts} err={exc}",
                    file=sys.stderr,
                    flush=True,
                )
            if retry_sleep > 0:
                time.sleep(retry_sleep)


def resolve_comparison_video_path(
    raw_video: str,
    episodes_jsonl: Path,
    comparison_root: Path,
) -> tuple[Path, str]:
    text = str(raw_video or "").strip()
    if not text:
        raise RuntimeError("empty comparison_video")
    raw_path = Path(text)
    if raw_path.is_absolute():
        return raw_path, str(raw_path)

    candidates: list[Path] = []
    if comparison_root:
        candidates.append(comparison_root / raw_path.name)
        candidates.append(comparison_root / raw_path)
    episodes_path = episodes_jsonl.expanduser().resolve()
    candidates.append(episodes_path.parent / raw_path)
    candidates.append(episodes_path.parent.parent / raw_path)

    seen: set[Path] = set()
    deduped: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    for candidate in deduped:
        if candidate.exists():
            return candidate, text
    return deduped[0], text


def extract_episode_id(entry: dict[str, Any], rel_video: Path) -> str:
    for key in ("episode_id", "episode_index"):
        value = entry.get(key)
        if isinstance(value, int):
            return f"{value:05d}"
        if isinstance(value, str) and value.strip():
            text = value.strip()
            if text.isdigit():
                return f"{int(text):05d}"
            return text
    match = re.search(r"episode_(\d+)", rel_video.name)
    if match:
        return f"{int(match.group(1)):05d}"
    return rel_video.stem


def build_vla_filter_key(entry: dict[str, Any], line_index: int, video_rel: str) -> str:
    episode_id = extract_episode_id(entry, Path(video_rel))
    return f"{episode_id}|{video_rel}|{line_index}"


def evenly_sample_indices(total: int, count: int) -> list[int]:
    if total <= 0:
        return []
    if count <= 0 or total <= count:
        return list(range(total))
    if count == 1:
        return [0]
    out: list[int] = []
    for idx in range(count):
        raw = round(idx * (total - 1) / (count - 1))
        value = min(total - 1, max(0, int(raw)))
        if not out or value != out[-1]:
            out.append(value)
    if out[-1] != total - 1:
        out[-1] = total - 1
    while len(out) < count:
        candidate = out[-1] - 1
        while candidate >= 0 and candidate in out:
            candidate -= 1
        if candidate < 0:
            break
        out.append(candidate)
        out.sort()
    return out[:count]


def crop_right_half(frame: Any) -> Any:
    width = int(frame.shape[1]) if frame is not None and hasattr(frame, "shape") else 0
    if width <= 1:
        return frame
    start = width // 2
    return frame[:, start:, :]


def sample_video_frames_right_half(
    video_path: Path,
    sample_frames: int,
    frame_size: tuple[int, int],
    max_frames_cap: int = 128,
) -> list[Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_indices = evenly_sample_indices(total, sample_frames) if total > 0 else []
    frames: list[Any] = []
    frame_index = -1
    target_pos = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if target_indices:
            if target_pos >= len(target_indices):
                break
            if frame_index != target_indices[target_pos]:
                continue
            target_pos += 1
        right_half = crop_right_half(frame)
        frames.append(cv2.resize(right_half, frame_size, interpolation=cv2.INTER_AREA))
    cap.release()
    if sample_frames > 0 and len(frames) > sample_frames:
        keep = evenly_sample_indices(len(frames), sample_frames)
        frames = [frames[i] for i in keep]
    if max_frames_cap > 0 and len(frames) > max_frames_cap:
        keep = evenly_sample_indices(len(frames), max_frames_cap)
        frames = [frames[i] for i in keep]
    return frames


def run_vla_filter(args: argparse.Namespace) -> int:
    rank, all_ranks = parse_rank_spec(args.rank)
    episodes_jsonl = Path(args.episodes_jsonl)
    comparison_root = Path(args.comparison_root)
    records = collect_jsonl_records(episodes_jsonl, args.line_start)
    if not records:
        print("No records to process.", flush=True)
        return 0

    rank_size = len(records) // all_ranks
    start_idx = rank * rank_size
    end_idx = (rank + 1) * rank_size if rank + 1 < all_ranks else len(records)
    records = records[start_idx:end_idx]
    selected_episode_ids = parse_episode_ids_filter(args.episode_ids)
    if selected_episode_ids:
        filtered_records: list[tuple[int, dict[str, Any]]] = []
        for line_index, entry in records:
            raw_video = str(entry.get("comparison_video") or "").strip()
            rel_video = Path(raw_video) if raw_video else Path(f"episode_{line_index:06d}.mp4")
            episode_id = extract_episode_id(entry, rel_video)
            if episode_id in selected_episode_ids:
                filtered_records.append((line_index, entry))
        records = filtered_records
    if args.max_videos > 0:
        records = records[: args.max_videos]
    print(f"rank={rank}/{all_ranks}, selected={len(records)}", flush=True)
    if selected_episode_ids:
        print(f"episode filter active: {sorted(selected_episode_ids)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"rank_{rank}.jsonl"
    error_path = out_dir / f"rank_{rank}.errors.jsonl"
    if args.overwrite_output:
        if save_path.exists():
            save_path.unlink()
        if error_path.exists():
            error_path.unlink()

    pass_threshold = 9
    expectation_map = load_expectation_map(args.expectation_jsonl)
    if expectation_map:
        print(f"Loaded expectations: {len(expectation_map)}", flush=True)
    processed_episode_ids: set[str] = set()
    if args.skip_existing and save_path.exists():
        for row in load_jsonl(save_path):
            episode_id = str(row.get("episode_id") or "").strip()
            if episode_id:
                processed_episode_ids.add(episode_id)
        print(f"Loaded {len(processed_episode_ids)} processed records from {save_path}", flush=True)

    session = requests.Session()
    model_name = resolve_model_name(session, args.model, args.host, args.timeout)
    print(f"Using model: {model_name}", flush=True)

    frame_size = (max(1, args.frame_width), max(1, args.frame_height))
    success = skipped = failed = passed = rejected = 0
    started = time.time()
    for idx, (line_index, entry) in enumerate(records, start=1):
        raw_comparison_video = str(entry.get("comparison_video") or "").strip()
        if not raw_comparison_video:
            append_jsonl(
                error_path,
                {"line_index": line_index, "status": "error", "error": "missing comparison_video field"},
            )
            failed += 1
            continue

        abs_video, rel_video_text = resolve_comparison_video_path(
            raw_comparison_video,
            episodes_jsonl=episodes_jsonl,
            comparison_root=comparison_root,
        )
        rel_video = Path(rel_video_text)
        episode_id = extract_episode_id(entry, rel_video)
        print(
            f"[episode {idx}/{len(records)}] episode_id={episode_id} video={rel_video_text}",
            flush=True,
        )
        if episode_id in processed_episode_ids:
            skipped += 1
            continue
        if not abs_video.exists():
            append_jsonl(
                error_path,
                {
                    "line_index": line_index,
                    "episode_id": episode_id,
                    "comparison_video": rel_video_text,
                    "status": "error",
                    "error": f"video not found: {abs_video}",
                },
            )
            failed += 1
            continue

        instruction = str(entry.get("prompt") or "").strip()
        if not instruction:
            append_jsonl(
                error_path,
                {
                    "line_index": line_index,
                    "episode_id": episode_id,
                    "comparison_video": rel_video_text,
                    "status": "error",
                    "error": "missing prompt field",
                },
            )
            failed += 1
            continue

        prompt = args.prompt_template.replace("{instruction}", instruction)
        try:
            frames = sample_video_frames_right_half(
                abs_video,
                args.sample_frames,
                frame_size,
                max_frames_cap=max(0, int(args.max_frames_cap)),
            )
            if not frames:
                raise RuntimeError("No frames sampled from right-half video")
            response_text = call_vllm_with_retry(
                session=session,
                host=args.host,
                model=model_name,
                prompt=prompt,
                frames=frames,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                episode_tag=f"{episode_id}:{rel_video}",
                request_options={
                    "temperature": DEFAULT_VLA_EVAL_TEMPERATURE,
                    "top_p": DEFAULT_VLA_EVAL_TOP_P,
                    "extra_body": {
                        "top_k": DEFAULT_VLA_EVAL_TOP_K,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                },
            )
            parse_retry_used = False
            json_repaired = False
            try:
                payload, json_repaired = parse_json_payload_with_meta(response_text)
                result = parse_vla_filter_result(payload, response_text)
            except Exception as parse_exc:
                parse_retry_used = True
                print(
                    f"[episode {idx}/{len(records)}] parse_error={parse_exc}; retrying_once=true",
                    flush=True,
                )
                retry_response_text = call_vllm_with_retry(
                    session=session,
                    host=args.host,
                    model=model_name,
                    prompt=prompt,
                    frames=frames,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                    episode_tag=f"{episode_id}:{rel_video}:parse_retry",
                    request_options={
                        "temperature": DEFAULT_VLA_EVAL_TEMPERATURE,
                        "top_p": DEFAULT_VLA_EVAL_TOP_P,
                        "extra_body": {
                            "top_k": DEFAULT_VLA_EVAL_TOP_K,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    },
                )
                payload, json_repaired_retry = parse_json_payload_with_meta(retry_response_text)
                json_repaired = bool(json_repaired or json_repaired_retry)
                response_text = retry_response_text
                result = parse_vla_filter_result(payload, response_text)
            instruction_score = int(result["instruction_score"])
            multi_view_score = int(result["multi_view_score"])
            artifact_score = int(result["artifact_score"])
            score = int(instruction_score + multi_view_score + artifact_score)
            issues = [str(x).strip().lower() for x in (result.get("issues") or []) if str(x).strip()]
            issue_set = set(issues)
            hard_fail_triggers = [str(x).strip().lower() for x in (result.get("hard_fail_triggers") or []) if str(x).strip()]
            evidence_pairs = [str(x).strip() for x in (result.get("evidence_pairs") or []) if str(x).strip()]
            fail_reasons: list[str] = []
            if instruction_score <= 2:
                fail_reasons.append("low_instruction_score")
            if multi_view_score <= 2:
                fail_reasons.append("low_multi_view_score")
            if artifact_score <= 2:
                fail_reasons.append("low_artifact_score")
            hard_issue_tags = {"view_mismatch", "object_duplication", "object_disappearance", "premature_adhesion"}
            triggered_issue_tags = sorted(tag for tag in hard_issue_tags if tag in issue_set)
            for tag in triggered_issue_tags:
                fail_reasons.append(f"issue:{tag}")
            for trigger in hard_fail_triggers:
                fail_reasons.append(f"trigger:{trigger}")
            if len(evidence_pairs) < 2 and score > 9:
                fail_reasons.append("insufficient_evidence_pairs_for_high_score")
            consistency_reasons = detect_textual_contradictions(
                instruction_score=instruction_score,
                multi_view_score=multi_view_score,
                artifact_score=artifact_score,
                issues=issues,
                hard_fail_triggers=hard_fail_triggers,
                evidence_pairs=evidence_pairs,
                reason=str(result.get("reason") or ""),
            )
            fail_reasons.extend(consistency_reasons)
            if score < pass_threshold:
                fail_reasons.append(f"score_below_threshold:{pass_threshold}")
            # Deduplicate while preserving order.
            dedup_fail_reasons: list[str] = []
            seen_fail_reasons: set[str] = set()
            for item in fail_reasons:
                if item in seen_fail_reasons:
                    continue
                seen_fail_reasons.add(item)
                dedup_fail_reasons.append(item)
            fail_reasons = dedup_fail_reasons
            is_pass = len(fail_reasons) == 0
            out_row = {
                "episode_id": episode_id,
                "instruction": instruction,
                "instruction_score": instruction_score,
                "multi_view_score": multi_view_score,
                "artifact_score": artifact_score,
                "issues": issues,
                "evidence_pairs": evidence_pairs,
                "hard_fail_triggers": hard_fail_triggers,
                "reason": result["reason"],
                "score": score,
                "passed": is_pass,
                "fail_reasons": fail_reasons,
                "json_repaired": json_repaired,
                "parse_retry_used": parse_retry_used,
            }
            print(
                f"[episode {idx}/{len(records)}] json_repaired={json_repaired} parse_retry_used={parse_retry_used}",
                flush=True,
            )
            if expectation_map:
                episode_expectation = expectation_map.get(episode_id)
                if episode_expectation is not None:
                    meets_expectation = evaluate_expectation(out_row, episode_expectation)
                    out_row["meets_expectation"] = meets_expectation
                    if not meets_expectation:
                        out_row["passed"] = False
                        out_row["fail_reasons"] = [*out_row["fail_reasons"], "expectation_mismatch"]
            append_jsonl(save_path, out_row)
            processed_episode_ids.add(episode_id)
            success += 1
            if is_pass:
                passed += 1
            else:
                rejected += 1
        except Exception as exc:
            append_jsonl(
                error_path,
                {
                    "line_index": line_index,
                    "episode_id": episode_id,
                    "comparison_video": rel_video_text,
                    "status": "error",
                    "error": str(exc),
                },
            )
            failed += 1

        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(
                f"progress={idx}/{len(records)}, success={success}, passed={passed}, "
                f"rejected={rejected}, skipped={skipped}, failed={failed}",
                flush=True,
            )

    elapsed = round(time.time() - started, 3)
    print(
        f"Done. success={success}, passed={passed}, rejected={rejected}, skipped={skipped}, "
        f"failed={failed}, elapsed_sec={elapsed}",
        flush=True,
    )
    print(f"VLA Filter: {save_path}", flush=True)
    print(f"Errors: {error_path}", flush=True)
    return 0 if failed == 0 else 5


def main() -> int:
    args = parse_args()
    return run_vla_filter(args)


if __name__ == "__main__":
    raise SystemExit(main())
