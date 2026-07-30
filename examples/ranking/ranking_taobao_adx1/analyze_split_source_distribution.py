from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

"""
PYTHONPATH=. python projects/ranking_taobao_adx1/analyze_split_source_distribution.py \
  --model rankmixer \
  --config projects/ranking_taobao_adx1/rankmixer/configs/data.json
"""

MODEL_PACKAGES = {
    "dcn_v2": "reckit.ranking.dcn_v2.process",
    "rankmixer": "reckit.ranking.rankmixer.process",
    "onetrans": "reckit.ranking.onetrans.process",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze train/valid/test split_source distribution for TaobaoAd_x1 ranking data."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PACKAGES),
        required=True,
        help="Model whose sample-building logic should be reused.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the model data.json used for process.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def build_samples_dcn_or_rankmixer(
    seq_df: pd.DataFrame, config: dict[str, Any]
) -> list[dict[str, Any]]:
    user_col = config.get("user_col", "uid")
    item_col = config.get("item_col", "iid")
    timestamp_col = config.get("timestamp_col", "timestamp")
    label_col = config.get("label_col", "label")
    sample_cfg = config.get("sample_builder", {})
    min_history_len = int(sample_cfg.get("min_history_len", 1))
    positive_only = bool(sample_cfg.get("history_positive_only", True))
    samples: list[dict[str, Any]] = []
    ordered = seq_df.sort_values([user_col, timestamp_col]).reset_index(drop=True)
    for raw_uid, user_df in ordered.groupby(user_col, sort=False):
        history: list[str] = []
        for row in user_df.itertuples(index=False):
            row_dict = row._asdict()
            label = int(row_dict.get(label_col, 1))
            if len(history) >= min_history_len:
                samples.append(
                    {
                        "uid": str(raw_uid),
                        "timestamp": int(row_dict[timestamp_col]),
                        "label": label,
                        "split_source": str(row_dict.get("split_source", "unknown")),
                    }
                )
            if not positive_only or label == 1:
                history.append(str(row_dict[item_col]))
    return sorted(samples, key=lambda item: item["timestamp"])


def build_samples_onetrans(
    seq_df: pd.DataFrame, config: dict[str, Any], process_module: Any
) -> list[dict[str, Any]]:
    user_col = config.get("user_col", "uid")
    item_col = config.get("item_col", "iid")
    timestamp_col = config.get("timestamp_col", "timestamp")
    label_col = config.get("label_col", "label")
    sample_cfg = config.get("sample_builder", {})
    min_history_len = int(sample_cfg.get("min_history_len", 1))
    sequences = process_module._sequence_field_config(config)
    samples: list[dict[str, Any]] = []
    ordered = seq_df.sort_values([user_col, timestamp_col]).reset_index(drop=True)
    for raw_uid, user_df in ordered.groupby(user_col, sort=False):
        histories = {
            sequence["name"]: {"items": [], "timestamps": []} for sequence in sequences
        }
        for row in user_df.itertuples(index=False):
            row_dict = row._asdict()
            label = int(row_dict.get(label_col, 1))
            history_len = sum(len(history["items"]) for history in histories.values())
            if history_len >= min_history_len:
                samples.append(
                    {
                        "uid": str(raw_uid),
                        "timestamp": int(row_dict[timestamp_col]),
                        "label": label,
                        "split_source": str(row_dict.get("split_source", "unknown")),
                    }
                )
            for sequence in sequences:
                if process_module._event_matches_sequence(
                    row_dict, label_col, sequence
                ):
                    raw_iid = str(row_dict[item_col])
                    histories[sequence["name"]]["items"].append(raw_iid)
                    histories[sequence["name"]]["timestamps"].append(
                        int(row_dict[timestamp_col])
                    )
    return sorted(samples, key=lambda item: item["timestamp"])


def summarize_split(name: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    counter = Counter(str(sample.get("split_source", "unknown")) for sample in samples)
    total = len(samples)
    timestamps = [int(sample["timestamp"]) for sample in samples]
    return {
        "split": name,
        "total": total,
        "timestamp_min": min(timestamps) if timestamps else None,
        "timestamp_max": max(timestamps) if timestamps else None,
        "split_source_counts": dict(counter),
        "split_source_ratio": {
            key: (value / total if total else 0.0)
            for key, value in sorted(counter.items())
        },
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    data_root = Path(str(config["data_root"])).resolve()
    seq_path = data_root / "seq.csv"
    if not seq_path.exists():
        raise FileNotFoundError(f"seq.csv not found: {seq_path}")

    process_module = importlib.import_module(MODEL_PACKAGES[args.model])

    user_col = str(config.get("user_col", "uid"))
    item_col = str(config.get("item_col", "iid"))
    timestamp_col = str(config.get("timestamp_col", "timestamp"))
    label_col = str(config.get("label_col", "label"))
    use_columns = [user_col, item_col, timestamp_col, label_col]
    if "split_source" not in use_columns:
        use_columns.append("split_source")

    seq_df = pd.read_csv(
        seq_path, usecols=lambda col: col in set(use_columns), dtype=object
    )
    if "split_source" not in seq_df.columns:
        raise ValueError(
            f"split_source column not found in {seq_path}. This script expects raw data prepared by prepare_taobaoad_x1.py"
        )
    seq_df[timestamp_col] = pd.to_numeric(seq_df[timestamp_col], errors="raise").astype(
        "int64"
    )
    seq_df[label_col] = (
        pd.to_numeric(seq_df[label_col], errors="coerce").fillna(0).astype("int64")
    )

    if args.model in {"dcn_v2", "rankmixer"}:
        samples = build_samples_dcn_or_rankmixer(seq_df, config)
    else:
        samples = build_samples_onetrans(seq_df, config, process_module)

    train, valid, test = process_module._split_samples(samples, config)

    summary = {
        "model": args.model,
        "config": str(config_path),
        "data_root": str(data_root),
        "sample_builder_split": config.get("sample_builder", {}).get("split", {}),
        "sample_count": len(samples),
        "splits": {
            "train": summarize_split("train", train),
            "valid": summarize_split("valid", valid),
            "test": summarize_split("test", test),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
