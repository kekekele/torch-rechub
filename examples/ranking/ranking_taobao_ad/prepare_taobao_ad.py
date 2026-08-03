"""Convert 飞桨 Taobao Ad dataset (raw_sample.csv + user_profile.csv + ad_feature.csv)
to RecKit ranking raw format (seq.csv / user_info.csv / item_fea.csv / data_format.csv).

Source: https://tianchi.aliyun.com/dataset/dataDetail?dataId=56

Usage (from repo root):
    PYTHONPATH=. python projects/ranking_taobao_ad/prepare_taobao_ad.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DATA_DIR = Path("data_preprocess/飞桨_taobao_ad")
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "raw" / "taobao_ad"

USER_COLUMNS = [
    "cms_segid",
    "cms_group_id",
    "final_gender_code",
    "age_level",
    "pvalue_level",
    "shopping_level",
    "occupation",
    "new_user_class_level",
]

ITEM_COLUMNS = [
    "cate_id",
    "campaign_id",
    "customer",
    "brand",
    "price",
]

MISSING_TOKENS = {"", "nan", "null", "none", "na", "n/a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 飞桨 Taobao Ad CSVs to RecKit raw ranking format."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help="Directory containing raw_sample.csv, user_profile.csv, ad_feature.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for RecKit raw format files.",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output."
    )
    return parser.parse_args()


def _log(msg: str) -> None:
    print(f"[prepare_taobao_ad] {msg}", flush=True)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if text.lower() in MISSING_TOKENS:
            return default
        return int(float(text))
    except (ValueError, TypeError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if text.lower() in MISSING_TOKENS:
            return default
        return float(text)
    except (ValueError, TypeError):
        return default


def _load_user_lookup(path: Path) -> dict[str, dict[str, Any]]:
    """Load user_profile.csv into a dict keyed by userid (str)."""
    _log(f"load user_profile from {path}")
    df = pd.read_csv(path, dtype=object)
    df.columns = [c.strip() for c in df.columns]
    lookup: dict[str, dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        uid = str(getattr(row, "userid", "")).strip()
        if not uid:
            continue
        lookup[uid] = {col: _as_int(getattr(row, col, 0)) for col in USER_COLUMNS}
    _log(f"loaded {len(lookup)} unique users from user_profile")
    return lookup


def _load_item_lookup(path: Path) -> dict[str, dict[str, Any]]:
    """Load ad_feature.csv into a dict keyed by adgroup_id (str)."""
    _log(f"load ad_feature from {path}")
    df = pd.read_csv(path, dtype=object)
    df.columns = [c.strip() for c in df.columns]
    lookup: dict[str, dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        iid = str(getattr(row, "adgroup_id", "")).strip()
        if not iid:
            continue
        entry: dict[str, Any] = {}
        for col in ITEM_COLUMNS:
            raw = getattr(row, col, None)
            entry[col] = _as_float(raw) if col == "price" else _as_int(raw)
        lookup[iid] = entry
    _log(f"loaded {len(lookup)} unique items from ad_feature")
    return lookup


def _write_data_format(output_dir: Path) -> None:
    rows = [
        ("seq.csv", "uid", "str", "false"),
        ("seq.csv", "iid", "str", "false"),
        ("seq.csv", "timestamp", "int", "false"),
        ("seq.csv", "action", "int", "false"),
        ("seq.csv", "pid", "str", "false"),
        ("seq.csv", "split_source", "str", "false"),
        ("user_info.csv", "uid", "str", "false"),
    ]
    for col in USER_COLUMNS:
        rows.append(("user_info.csv", col, "int", "false"))
    rows.append(("item_fea.csv", "iid", "str", "false"))
    for col in ITEM_COLUMNS:
        dtype = "float" if col == "price" else "int"
        rows.append(("item_fea.csv", col, dtype, "false"))

    df = pd.DataFrame(
        rows, columns=["file_name", "column_name", "data_type", "is_list"]
    )
    path = output_dir / "data_format.csv"
    df.to_csv(path, index=False)
    _log(f"wrote data_format.csv ({len(df)} rows) -> {path}")


def _write_side_tables(
    output_dir: Path,
    users: dict[str, dict[str, Any]],
    items: dict[str, dict[str, Any]],
) -> None:
    user_rows = [{"uid": uid, **feats} for uid, feats in sorted(users.items())]
    item_rows = [{"iid": iid, **feats} for iid, feats in sorted(items.items())]

    user_df = pd.DataFrame(user_rows)
    item_df = pd.DataFrame(item_rows)

    user_path = output_dir / "user_info.csv"
    item_path = output_dir / "item_fea.csv"
    user_df.to_csv(user_path, index=False)
    item_df.to_csv(item_path, index=False)
    _log(f"wrote user_info.csv ({len(user_df)} rows) -> {user_path}")
    _log(f"wrote item_fea.csv ({len(item_df)} rows) -> {item_path}")


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_data_dir
    output_dir = args.output_dir

    raw_sample_path = raw_dir / "raw_sample.csv"
    user_profile_path = raw_dir / "user_profile.csv"
    ad_feature_path = raw_dir / "ad_feature.csv"

    for p in [raw_sample_path, user_profile_path, ad_feature_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input file not found: {p}")

    if output_dir.exists() and not args.force:
        existing = [f.name for f in output_dir.iterdir() if f.is_file()]
        if any(n in existing for n in ("seq.csv", "user_info.csv", "item_fea.csv")):
            _log(f"output already exists at {output_dir}. Use --force to overwrite.")
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    user_lookup = _load_user_lookup(user_profile_path)
    item_lookup = _load_item_lookup(ad_feature_path)

    # Collect user/item feature dicts keyed by their id
    users_seen: dict[str, dict[str, Any]] = {}
    items_seen: dict[str, dict[str, Any]] = {}

    seq_path = output_dir / "seq.csv"
    seq_cols = ["uid", "iid", "timestamp", "action", "pid", "split_source"]
    total_rows = 0
    total_positives = 0
    write_header = True

    _log(f"streaming raw_sample.csv in chunks of {args.chunksize}")
    reader = pd.read_csv(
        raw_sample_path,
        chunksize=args.chunksize,
        dtype=object,
        encoding=args.encoding,
    )
    for chunk_idx, chunk in enumerate(reader):
        chunk.columns = [c.strip() for c in chunk.columns]

        uid_series = chunk["user"].astype(str).str.strip()
        iid_series = chunk["adgroup_id"].astype(str).str.strip()

        ts_series = (
            pd.to_numeric(chunk["time_stamp"], errors="coerce")
            .fillna(0)
            .astype("int64")
        )
        action_series = (
            pd.to_numeric(chunk["clk"], errors="coerce").fillna(0).astype("int64")
        )
        pid_series = chunk["pid"].astype(str).str.strip()

        seq_df = pd.DataFrame(
            {
                "uid": uid_series.values,
                "iid": iid_series.values,
                "timestamp": ts_series.values,
                "action": action_series.values,
                "pid": pid_series.values,
                "split_source": "train",
            }
        )
        seq_df.to_csv(
            seq_path,
            index=False,
            mode="w" if write_header else "a",
            header=write_header,
        )
        write_header = False

        for uid, feats in zip(
            uid_series,
            [user_lookup.get(u, {col: 0 for col in USER_COLUMNS}) for u in uid_series],
        ):
            if uid not in users_seen:
                users_seen[uid] = feats

        for iid, feats in zip(
            iid_series,
            [
                item_lookup.get(
                    i, {col: (0.0 if col == "price" else 0) for col in ITEM_COLUMNS}
                )
                for i in iid_series
            ],
        ):
            if iid not in items_seen:
                items_seen[iid] = feats

        total_rows += len(seq_df)
        total_positives += int(action_series.sum())

        if (chunk_idx + 1) % 5 == 0:
            _log(
                f"progress: chunks={chunk_idx + 1}, rows={total_rows}, "
                f"positives={total_positives}, ctr={total_positives/max(total_rows,1):.4f}"
            )

    _log(
        f"finished: total_rows={total_rows}, positives={total_positives}, "
        f"negatives={total_rows - total_positives}, "
        f"ctr={total_positives/max(total_rows,1):.4f}, "
        f"unique_users={len(users_seen)}, unique_items={len(items_seen)}"
    )

    _write_side_tables(output_dir, users_seen, items_seen)
    _write_data_format(output_dir)

    # Write summary json
    summary = {
        "total_rows": total_rows,
        "positives": total_positives,
        "negatives": total_rows - total_positives,
        "ctr": round(total_positives / max(total_rows, 1), 6),
        "unique_users": len(users_seen),
        "unique_items": len(items_seen),
    }
    summary_path = output_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _log(f"wrote summary: {summary}")
    _log(f"done. output dir: {output_dir}")


if __name__ == "__main__":
    main()
