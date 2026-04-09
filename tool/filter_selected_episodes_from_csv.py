#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


DEFAULT_META_DIR = Path(
    "/data1/linzengrong/Code/DiffSynth-Studio/Ckpt/3_26_real_robot_PnP/"
    "epoch-149/output_vla_lerobot/batch_20260330_161320/meta"
)


def parse_selected_episode_indices(
    csv_path: Path,
    id_col: str,
    selected_col: str,
    selected_value: str,
) -> tuple[set[int], dict[str, int]]:
    stats = {
        "rows_total": 0,
        "rows_selected": 0,
        "rows_selected_missing_id": 0,
        "rows_selected_bad_id": 0,
        "duplicate_selected_ids": 0,
    }
    selected_ids: set[int] = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        if id_col not in reader.fieldnames:
            raise KeyError(f"Missing id column '{id_col}' in CSV: {csv_path}")
        if selected_col not in reader.fieldnames:
            raise KeyError(f"Missing selected column '{selected_col}' in CSV: {csv_path}")

        for row in reader:
            stats["rows_total"] += 1
            selected_raw = str(row.get(selected_col, "")).strip()
            if selected_raw != selected_value:
                continue

            stats["rows_selected"] += 1
            id_raw = str(row.get(id_col, "")).strip()
            if not id_raw:
                stats["rows_selected_missing_id"] += 1
                continue

            try:
                episode_id = int(id_raw)
            except ValueError:
                stats["rows_selected_bad_id"] += 1
                continue

            if episode_id in selected_ids:
                stats["duplicate_selected_ids"] += 1
            selected_ids.add(episode_id)

    return selected_ids, stats


def filter_jsonl_by_episode_index(
    input_path: Path,
    output_path: Path | None,
    keep_episode_indices: set[int],
    dry_run: bool,
) -> tuple[int, int, set[int]]:
    total_rows = 0
    kept_rows = 0
    found_episode_indices: set[int] = set()

    writer = None
    if not dry_run:
        if output_path is None:
            raise ValueError("output_path must be provided when dry_run is False")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = output_path.open("w", encoding="utf-8")

    try:
        with input_path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                total_rows += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {input_path} at line {line_no}: {exc}"
                    ) from exc

                if "episode_index" not in obj:
                    raise KeyError(
                        f"Missing 'episode_index' in {input_path} at line {line_no}"
                    )
                try:
                    episode_index = int(obj["episode_index"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid episode_index in {input_path} at line {line_no}: "
                        f"{obj['episode_index']}"
                    ) from exc

                if episode_index not in keep_episode_indices:
                    continue

                kept_rows += 1
                found_episode_indices.add(episode_index)
                if writer is not None:
                    writer.write(raw_line if raw_line.endswith("\n") else f"{raw_line}\n")
    finally:
        if writer is not None:
            writer.close()

    return total_rows, kept_rows, found_episode_indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter episodes.jsonl and episodes_stats.jsonl by selected ids "
            "in a CSV table."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_META_DIR / "数据集-最终选择.csv",
        help="CSV path containing selected ids",
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        default=DEFAULT_META_DIR / "episodes.jsonl",
        help="Input episodes.jsonl",
    )
    parser.add_argument(
        "--episodes-stats",
        type=Path,
        default=DEFAULT_META_DIR / "episodes_stats.jsonl",
        help="Input episodes_stats.jsonl",
    )
    parser.add_argument(
        "--out-episodes",
        type=Path,
        default=DEFAULT_META_DIR / "episodes.selected.jsonl",
        help="Output filtered episodes jsonl",
    )
    parser.add_argument(
        "--out-episodes-stats",
        type=Path,
        default=DEFAULT_META_DIR / "episodes_stats.selected.jsonl",
        help="Output filtered episodes_stats jsonl",
    )
    parser.add_argument(
        "--id-col",
        default="数据编号",
        help="CSV column name mapped to JSONL episode_index",
    )
    parser.add_argument(
        "--selected-col",
        default="是否入选",
        help="CSV column name indicating selected rows",
    )
    parser.add_argument(
        "--selected-value",
        default="1",
        help="Only rows with selected_col == selected_value are kept",
    )
    parser.add_argument(
        "--strict-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if any selected episode_index is missing in JSONL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print summary, do not write outputs",
    )
    args = parser.parse_args()

    selected_ids, csv_stats = parse_selected_episode_indices(
        csv_path=args.csv,
        id_col=args.id_col,
        selected_col=args.selected_col,
        selected_value=args.selected_value,
    )
    if not selected_ids:
        raise ValueError(
            "No selected ids parsed from CSV. "
            "Please check selected column/value and id column."
        )
    if csv_stats["rows_selected_missing_id"] > 0 or csv_stats["rows_selected_bad_id"] > 0:
        raise ValueError(
            "CSV has selected rows with missing/invalid ids: "
            f"missing={csv_stats['rows_selected_missing_id']}, "
            f"invalid={csv_stats['rows_selected_bad_id']}"
        )

    ep_total, ep_kept, ep_found = filter_jsonl_by_episode_index(
        input_path=args.episodes,
        output_path=args.out_episodes,
        keep_episode_indices=selected_ids,
        dry_run=args.dry_run,
    )
    st_total, st_kept, st_found = filter_jsonl_by_episode_index(
        input_path=args.episodes_stats,
        output_path=args.out_episodes_stats,
        keep_episode_indices=selected_ids,
        dry_run=args.dry_run,
    )

    missing_in_episodes = sorted(selected_ids - ep_found)
    missing_in_stats = sorted(selected_ids - st_found)
    mismatch_between_outputs = sorted(ep_found ^ st_found)

    if args.strict_missing and missing_in_episodes:
        raise KeyError(
            f"Selected episode_index missing in episodes.jsonl: {missing_in_episodes}"
        )
    if args.strict_missing and missing_in_stats:
        raise KeyError(
            "Selected episode_index missing in episodes_stats.jsonl: "
            f"{missing_in_stats}"
        )
    if args.strict_missing and mismatch_between_outputs:
        raise ValueError(
            "Mismatch between filtered episode_index sets of outputs: "
            f"{mismatch_between_outputs}"
        )

    print(f"csv: {args.csv.resolve()}")
    print(f"episodes input: {args.episodes.resolve()}")
    print(f"episodes_stats input: {args.episodes_stats.resolve()}")
    print(f"dry_run: {args.dry_run}")
    if not args.dry_run:
        print(f"episodes output: {args.out_episodes.resolve()}")
        print(f"episodes_stats output: {args.out_episodes_stats.resolve()}")
    print(
        "csv rows: total={rows_total}, selected={rows_selected}, "
        "selected_unique_ids={selected_unique}, duplicate_selected_ids={duplicates}".format(
            rows_total=csv_stats["rows_total"],
            rows_selected=csv_stats["rows_selected"],
            selected_unique=len(selected_ids),
            duplicates=csv_stats["duplicate_selected_ids"],
        )
    )
    print(f"episodes rows: total={ep_total}, kept={ep_kept}")
    print(f"episodes_stats rows: total={st_total}, kept={st_kept}")
    print(f"missing in episodes: {len(missing_in_episodes)}")
    if missing_in_episodes:
        print(f"missing episode indices sample: {missing_in_episodes[:20]}")
    print(f"missing in episodes_stats: {len(missing_in_stats)}")
    if missing_in_stats:
        print(f"missing stats indices sample: {missing_in_stats[:20]}")
    print(f"output set mismatch count: {len(mismatch_between_outputs)}")
    if mismatch_between_outputs:
        print(f"set mismatch sample: {mismatch_between_outputs[:20]}")


if __name__ == "__main__":
    main()
