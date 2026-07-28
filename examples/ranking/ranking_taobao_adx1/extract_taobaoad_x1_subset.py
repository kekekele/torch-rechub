from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_DIR / "data_preprocess" / "taobao_adx1" / "processed"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data_preprocess" / "taobao_adx1" / "processed_2days"
REQUIRED_FILES = ["seq.csv", "user_info.csv", "item_fea.csv", "data_format.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a smaller TaobaoAd_x1 raw subset from already converted RecKit raw files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing converted RecKit raw files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the extracted subset",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="Number of unique calendar days to keep from seq.csv",
    )
    parser.add_argument(
        "--day-mode",
        choices=["first", "last"],
        default="first",
        help="Keep the first N or last N unique days in timestamp order",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional explicit start day in YYYY-MM-DD. When set, keeps this day and the next N-1 days.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files in output-dir",
    )
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[extract_taobaoad_x1_subset] {message}", flush=True)


def _validate_input_dir(input_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"input-dir is missing required files: {missing}. Expected under {input_dir}"
        )


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for name in REQUIRED_FILES + ["subset_summary.json"]:
            path = output_dir / name
            if path.exists():
                path.unlink()
    else:
        existing = [name for name in REQUIRED_FILES if (output_dir / name).exists()]
        if existing:
            raise FileExistsError(
                f"output-dir already contains files {existing}. Use --force to overwrite."
            )


def _load_seq(seq_path: Path) -> pd.DataFrame:
    _log(f"read seq.csv: {seq_path}")
    seq_df = pd.read_csv(seq_path)
    if "timestamp" not in seq_df.columns:
        raise ValueError("seq.csv must contain a timestamp column")
    seq_df["timestamp"] = pd.to_numeric(seq_df["timestamp"], errors="coerce")
    if seq_df["timestamp"].isna().any():
        raise ValueError(
            "seq.csv contains non-numeric timestamp values; can not extract by day"
        )
    seq_df["event_day"] = pd.to_datetime(
        seq_df["timestamp"], unit="s", errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    if seq_df["event_day"].isna().any():
        raise ValueError(
            "timestamp values could not be converted to calendar day. This subset script requires real second-level timestamps."
        )
    return seq_df


def _select_days(
    seq_df: pd.DataFrame, days: int, day_mode: str, start_date: str | None
) -> list[str]:
    unique_days = sorted(seq_df["event_day"].dropna().unique().tolist())
    if not unique_days:
        raise ValueError("No valid event_day values found in seq.csv")
    if days <= 0:
        raise ValueError("--days must be positive")

    if start_date is not None:
        if start_date not in unique_days:
            raise ValueError(
                f"start-date {start_date} not found in seq.csv days. Available days: {unique_days[:10]}"
            )
        start_idx = unique_days.index(start_date)
        selected = unique_days[start_idx : start_idx + days]
        if len(selected) < days:
            raise ValueError(
                f"Not enough days after start-date={start_date}; requested {days}, available {len(selected)}"
            )
        return selected

    if days > len(unique_days):
        raise ValueError(
            f"Requested {days} days but seq.csv only contains {len(unique_days)} unique days"
        )

    if day_mode == "first":
        return unique_days[:days]
    return unique_days[-days:]


def _filter_side_table(path: Path, key_col: str, keep_ids: set[str]) -> pd.DataFrame:
    _log(f"read side table: {path}")
    frame = pd.read_csv(path)
    if key_col not in frame.columns:
        raise ValueError(f"{path.name} missing key column: {key_col}")
    key_series = frame[key_col].astype(str)
    return frame.loc[key_series.isin(keep_ids)].copy()


def main() -> None:
    args = parse_args()
    _log(
        f"start subset extraction: input_dir={args.input_dir}, output_dir={args.output_dir}, "
        f"days={args.days}, day_mode={args.day_mode}, start_date={args.start_date}"
    )

    _validate_input_dir(args.input_dir)
    _prepare_output_dir(args.output_dir, args.force)

    seq_df = _load_seq(args.input_dir / "seq.csv")
    selected_days = _select_days(
        seq_df, int(args.days), str(args.day_mode), args.start_date
    )
    _log(f"selected days: {selected_days}")

    subset_seq = seq_df.loc[seq_df["event_day"].isin(selected_days)].copy()
    subset_seq = subset_seq.drop(columns=["event_day"])
    subset_seq = subset_seq.sort_values(
        ["uid", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    _log(f"subset seq rows={len(subset_seq)}")

    keep_users = set(subset_seq["uid"].astype(str).unique().tolist())
    keep_items = set(subset_seq["iid"].astype(str).unique().tolist())

    subset_user = _filter_side_table(
        args.input_dir / "user_info.csv", "uid", keep_users
    )
    subset_item = _filter_side_table(args.input_dir / "item_fea.csv", "iid", keep_items)

    _log(f"subset user_info rows={len(subset_user)}")
    _log(f"subset item_fea rows={len(subset_item)}")

    subset_seq.to_csv(args.output_dir / "seq.csv", index=False)
    subset_user.to_csv(args.output_dir / "user_info.csv", index=False)
    subset_item.to_csv(args.output_dir / "item_fea.csv", index=False)
    data_format_df = pd.read_csv(args.input_dir / "data_format.csv")
    data_format_df.to_csv(args.output_dir / "data_format.csv", index=False)

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "selected_days": selected_days,
        "seq_rows": int(len(subset_seq)),
        "user_rows": int(len(subset_user)),
        "item_rows": int(len(subset_item)),
        "unique_users": int(len(keep_users)),
        "unique_items": int(len(keep_items)),
        "positive_rows": (
            int(pd.to_numeric(subset_seq["action"], errors="coerce").fillna(0).sum())
            if "action" in subset_seq.columns
            else None
        ),
    }
    (args.output_dir / "subset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log(
        f"finished subset extraction: summary={args.output_dir / 'subset_summary.json'}"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
