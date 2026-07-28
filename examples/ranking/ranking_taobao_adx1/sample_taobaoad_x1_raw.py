from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_DIR / "data_preprocess" / "taobao_adx1" / "processed"
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "data_preprocess" / "taobao_adx1" / "processed_sample"
)
REQUIRED_SEQ_COLUMNS = ["uid", "iid", "timestamp", "action"]
OPTIONAL_FILES = ["user_info.csv", "item_fea.csv", "data_format.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a smaller RecKit raw subset from converted TaobaoAd_x1 data."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing converted raw seq.csv/user_info.csv/item_fea.csv/data_format.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write sampled raw subset",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="Keep the most recent N days by timestamp. Set 0 to disable day filtering.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="After day filtering, keep at most the most recent N seq rows. Set 0 to keep all remaining rows.",
    )
    parser.add_argument(
        "--min-user-events",
        type=int,
        default=1,
        help="Drop users with fewer than this many remaining events after sampling.",
    )
    parser.add_argument(
        "--keep-positive-only-users",
        action="store_true",
        help="If set, keep only users that have at least one positive action in the sampled subset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files in output-dir.",
    )
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[sample_taobaoad_x1_raw] {message}", flush=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _ensure_required_seq_columns(seq_df: pd.DataFrame) -> None:
    missing = [
        column for column in REQUIRED_SEQ_COLUMNS if column not in seq_df.columns
    ]
    if missing:
        raise ValueError(f"seq.csv missing required columns: {missing}")


def _latest_day_threshold(seq_df: pd.DataFrame, days: int) -> int:
    timestamps = pd.to_numeric(seq_df["timestamp"], errors="coerce")
    if timestamps.isna().all():
        raise ValueError("seq.csv timestamp column is entirely missing or invalid")
    max_ts = int(timestamps.max())
    seconds = max(int(days), 1) * 24 * 60 * 60
    return max_ts - seconds + 1


def _filter_seq(seq_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = seq_df.copy()
    filtered["timestamp"] = pd.to_numeric(filtered["timestamp"], errors="coerce")
    filtered = filtered.dropna(subset=["timestamp"])
    filtered["timestamp"] = filtered["timestamp"].astype("int64")

    if args.days > 0:
        threshold = _latest_day_threshold(filtered, args.days)
        before = len(filtered)
        filtered = filtered[filtered["timestamp"] >= threshold].copy()
        _log(
            f"apply days filter: days={args.days}, threshold={threshold}, rows {before} -> {len(filtered)}"
        )

    filtered = filtered.sort_values(
        ["timestamp", "uid", "iid"], ascending=[True, True, True]
    )

    if args.max_rows > 0 and len(filtered) > args.max_rows:
        before = len(filtered)
        filtered = filtered.tail(int(args.max_rows)).copy()
        _log(
            f"apply max_rows filter: max_rows={args.max_rows}, rows {before} -> {len(filtered)}"
        )

    if args.min_user_events > 1:
        before = len(filtered)
        user_counts = filtered.groupby("uid").size()
        keep_users = set(
            user_counts[user_counts >= int(args.min_user_events)].index.astype(str)
        )
        filtered = filtered[filtered["uid"].astype(str).isin(keep_users)].copy()
        _log(
            f"apply min_user_events filter: min_user_events={args.min_user_events}, rows {before} -> {len(filtered)}"
        )

    if args.keep_positive_only_users:
        before = len(filtered)
        user_positive = (
            filtered.groupby("uid")["action"].sum().reset_index().query("action > 0")
        )
        keep_users = set(user_positive["uid"].astype(str))
        filtered = filtered[filtered["uid"].astype(str).isin(keep_users)].copy()
        _log(f"apply keep_positive_only_users filter: rows {before} -> {len(filtered)}")

    filtered = filtered.reset_index(drop=True)
    if filtered.empty:
        raise ValueError(
            "Sampled seq.csv is empty after filtering. Relax days/max_rows/min_user_events."
        )
    return filtered


def _filter_side_table(
    df: pd.DataFrame, key_col: str, keep_ids: set[str]
) -> pd.DataFrame:
    if key_col not in df.columns:
        raise ValueError(f"Side table missing key column: {key_col}")
    out = df[df[key_col].astype(str).isin(keep_ids)].copy()
    return out.reset_index(drop=True)


def _copy_data_format(input_dir: Path, output_dir: Path) -> None:
    src = input_dir / "data_format.csv"
    dst = output_dir / "data_format.csv"
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    path = output_dir / "sample_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"write sample summary: {path}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    _log(
        f"start sampling: input_dir={input_dir}, output_dir={output_dir}, days={args.days}, max_rows={args.max_rows}"
    )

    if not input_dir.exists():
        raise FileNotFoundError(f"input-dir does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for file_name in [
            "seq.csv",
            "user_info.csv",
            "item_fea.csv",
            "data_format.csv",
            "sample_summary.json",
        ]:
            path = output_dir / file_name
            if path.exists():
                path.unlink()

    seq_path = input_dir / "seq.csv"
    user_path = input_dir / "user_info.csv"
    item_path = input_dir / "item_fea.csv"

    _log(f"read seq.csv: {seq_path}")
    seq_df = _read_csv(seq_path)
    _ensure_required_seq_columns(seq_df)
    _log(f"loaded seq.csv rows={len(seq_df)}")

    sampled_seq = _filter_seq(seq_df, args)
    keep_users = set(sampled_seq["uid"].astype(str))
    keep_items = set(sampled_seq["iid"].astype(str))

    _log(f"write sampled seq.csv rows={len(sampled_seq)}")
    sampled_seq.to_csv(output_dir / "seq.csv", index=False)

    user_rows = 0
    if user_path.exists():
        _log(f"read user_info.csv: {user_path}")
        user_df = _read_csv(user_path)
        sampled_user = _filter_side_table(user_df, "uid", keep_users)
        user_rows = len(sampled_user)
        _log(f"write sampled user_info.csv rows={user_rows}")
        sampled_user.to_csv(output_dir / "user_info.csv", index=False)

    item_rows = 0
    if item_path.exists():
        _log(f"read item_fea.csv: {item_path}")
        item_df = _read_csv(item_path)
        sampled_item = _filter_side_table(item_df, "iid", keep_items)
        item_rows = len(sampled_item)
        _log(f"write sampled item_fea.csv rows={item_rows}")
        sampled_item.to_csv(output_dir / "item_fea.csv", index=False)

    _copy_data_format(input_dir, output_dir)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "days": int(args.days),
        "max_rows": int(args.max_rows),
        "min_user_events": int(args.min_user_events),
        "keep_positive_only_users": bool(args.keep_positive_only_users),
        "seq_rows": int(len(sampled_seq)),
        "user_rows": int(user_rows),
        "item_rows": int(item_rows),
        "unique_users": int(sampled_seq["uid"].astype(str).nunique()),
        "unique_items": int(sampled_seq["iid"].astype(str).nunique()),
        "timestamp_min": int(sampled_seq["timestamp"].min()),
        "timestamp_max": int(sampled_seq["timestamp"].max()),
        "positive_rows": int(
            pd.to_numeric(sampled_seq["action"], errors="coerce")
            .fillna(0)
            .astype(int)
            .sum()
        ),
    }
    summary["negative_rows"] = int(summary["seq_rows"] - summary["positive_rows"])
    _write_summary(output_dir, summary)
    _log(
        f"finished sampling: seq_rows={summary['seq_rows']}, unique_users={summary['unique_users']}, unique_items={summary['unique_items']}"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
