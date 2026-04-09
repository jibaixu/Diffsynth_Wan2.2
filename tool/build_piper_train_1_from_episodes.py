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


def build_clips(
    rows: list[dict],
    clip: int,
    overlap: int,
    min_tail: int,
) -> tuple[list[dict], int, int]:
    if clip <= 0:
        raise ValueError("clip must be > 0")
    if overlap < 0 or overlap >= clip:
        raise ValueError("overlap must satisfy 0 <= overlap < clip")
    if min_tail < 1:
        raise ValueError("min_tail must be >= 1")

    stride = clip - overlap
    out: list[dict] = []
    short_tail_kept = 0
    short_tail_dropped = 0

    for row in rows:
        s = int(row["start_frame"])
        e = int(row["end_frame"])
        length = int(row["length"])
        if length != e - s + 1:
            raise ValueError(
                f"length mismatch in episode_index={row.get('episode_index')}: "
                f"length={length}, range={s}-{e}"
            )

        start = s
        while start + clip - 1 <= e:
            clip_row = dict(row)
            clip_row["start_frame"] = int(start)
            clip_row["end_frame"] = int(start + clip - 1)
            clip_row["length"] = int(clip)
            out.append(clip_row)
            start += stride

        rem = e - start + 1
        if rem <= 0:
            continue
        if rem >= min_tail:
            clip_row = dict(row)
            clip_row["start_frame"] = int(start)
            clip_row["end_frame"] = int(e)
            clip_row["length"] = int(rem)
            out.append(clip_row)
            if rem < clip:
                short_tail_kept += 1
        else:
            short_tail_dropped += 1

    return out, short_tail_kept, short_tail_dropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build piper episodes_train_1.jsonl from episodes.jsonl "
            "with sliding windows and tail policy"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("robot_data/piper/meta/episodes.jsonl"),
        help="Input master episodes jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("robot_data/piper/meta/episodes_train_1.jsonl"),
        help="Output train jsonl",
    )
    parser.add_argument("--clip", type=int, default=17)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument(
        "--min-tail",
        type=int,
        default=5,
        help="Keep tail clip only when remaining frames >= min-tail",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    out, short_tail_kept, short_tail_dropped = build_clips(
        rows, args.clip, args.overlap, args.min_tail
    )

    if not args.dry_run:
        dump_jsonl(args.output, out)

    print(f"input: {args.input.resolve()}")
    print(f"output: {args.output.resolve()}")
    print(f"episodes: {len(rows)}")
    print(f"clip: {args.clip}, overlap: {args.overlap}, stride: {args.clip - args.overlap}")
    print(f"output_clips: {len(out)}")
    print(f"short_tail_kept({args.min_tail}..{args.clip - 1}): {short_tail_kept}")
    print(f"short_tail_dropped(<{args.min_tail}): {short_tail_dropped}")
    print("preview (first 10):")
    for row in out[:10]:
        print(
            f"  ep={row['episode_index']}, start={row['start_frame']}, "
            f"end={row['end_frame']}, length={row['length']}"
        )


if __name__ == "__main__":
    main()
