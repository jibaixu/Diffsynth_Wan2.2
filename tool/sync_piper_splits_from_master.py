#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def build_master_index(master_rows: list[dict]) -> dict[int, dict]:
    idx: dict[int, dict] = {}
    for row in master_rows:
        ep = int(row["episode_index"])
        idx[ep] = row
    return idx


def sync_val_rows(val_rows: list[dict], master_idx: dict[int, dict]) -> tuple[list[dict], int]:
    out: list[dict] = []
    changed = 0
    for row in val_rows:
        ep = int(row["episode_index"])
        if ep not in master_idx:
            raise KeyError(f"episode_index {ep} in val not found in master")
        m = master_idx[ep]
        old = (int(row.get("start_frame", 0)), int(row.get("end_frame", -1)), int(row.get("length", 0)))
        new = (int(m["start_frame"]), int(m["end_frame"]), int(m["length"]))
        row2 = dict(row)
        row2["start_frame"], row2["end_frame"], row2["length"] = new
        if old != new:
            changed += 1
        out.append(row2)
    return out, changed


def rebuild_train_rows(
    train_rows: list[dict],
    master_idx: dict[int, dict],
    window: int,
    overlap: int,
) -> list[dict]:
    if overlap >= window:
        raise ValueError("overlap must be smaller than window")
    stride = window - overlap

    # Preserve original episode order as first appearance in train file.
    ordered_eps: list[int] = []
    seen: set[int] = set()
    for row in train_rows:
        ep = int(row["episode_index"])
        if ep not in seen:
            seen.add(ep)
            ordered_eps.append(ep)

    out: list[dict] = []
    for ep in ordered_eps:
        if ep not in master_idx:
            raise KeyError(f"episode_index {ep} in train not found in master")
        m = master_idx[ep]
        s = int(m["start_frame"])
        e = int(m["end_frame"])
        length = int(m["length"])
        if length != e - s + 1:
            raise ValueError(f"Master length mismatch for episode_index={ep}: length={length}, range={s}-{e}")

        # Keep fields aligned with master for each generated clip.
        base = dict(m)
        start = s
        while start + window - 1 <= e:
            clip = dict(base)
            clip["start_frame"] = int(start)
            clip["end_frame"] = int(start + window - 1)
            clip["length"] = int(window)
            out.append(clip)
            start += stride
        # Tail shorter than `window` is intentionally dropped.
    return out


def summarize_train(rows: list[dict]) -> tuple[int, int, int, float]:
    if not rows:
        return 0, 0, 0, 0.0
    counts: dict[int, int] = {}
    for r in rows:
        ep = int(r["episode_index"])
        counts[ep] = counts.get(ep, 0) + 1
    vals = sorted(counts.values())
    n = len(vals)
    median = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    return min(vals), max(vals), len(counts), float(median)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync piper val/train metadata from master episodes.jsonl"
    )
    parser.add_argument(
        "--master",
        type=Path,
        default=Path("robot_data/piper/meta/episodes.jsonl"),
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=Path("robot_data/piper/meta/episodes_val.jsonl"),
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("robot_data/piper/meta/episodes_train.jsonl"),
    )
    parser.add_argument("--window", type=int, default=17)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    master_rows = load_jsonl(args.master)
    val_rows = load_jsonl(args.val)
    train_rows = load_jsonl(args.train)
    master_idx = build_master_index(master_rows)

    new_val_rows, val_changed = sync_val_rows(val_rows, master_idx)
    new_train_rows = rebuild_train_rows(train_rows, master_idx, args.window, args.overlap)
    tr_min, tr_max, tr_unique, tr_median = summarize_train(new_train_rows)

    if not args.dry_run:
        dump_jsonl(args.val, new_val_rows)
        dump_jsonl(args.train, new_train_rows)

    print(f"master rows: {len(master_rows)}")
    print(f"val rows old/new: {len(val_rows)} -> {len(new_val_rows)}")
    print(f"val rows changed: {val_changed}")
    print(f"train rows old/new: {len(train_rows)} -> {len(new_train_rows)}")
    print(f"train unique episodes: {tr_unique}")
    print(f"train clips per episode min/median/max: {tr_min}/{tr_median}/{tr_max}")
    print("train preview (first 10):")
    for r in new_train_rows[:10]:
        print(
            f"  ep={r['episode_index']}, start={r['start_frame']}, "
            f"end={r['end_frame']}, length={r['length']}"
        )


if __name__ == "__main__":
    main()
