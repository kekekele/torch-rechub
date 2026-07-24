import argparse
import csv
from pathlib import Path

import pandas as pd


def parse_list_text(text):
    if pd.isna(text):
        return []
    text = str(text).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def normalize_source_frame(frame):
    required_columns = {"user_id", "key", "value", "label"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    normalized_rows = []
    observed_feature_ids = set()

    for row_number, row in enumerate(frame.itertuples(index=False), start=1):
        key_list = parse_list_text(row.key)
        value_list = parse_list_text(row.value)
        if len(key_list) != len(value_list):
            raise ValueError(
                f"Row {row_number} has mismatched key/value length: "
                f"len(key)={len(key_list)} len(value)={len(value_list)}"
            )

        feature_map = {}
        for feature_id, feature_value in zip(key_list, value_list):
            feature_name = f"f_{feature_id}"
            feature_map[feature_name] = feature_value
            observed_feature_ids.add(feature_name)

        normalized_rows.append(
            {
                "row_number": row_number,
                "uid": str(row.user_id),
                "label": int(row.label),
                "feature_map": feature_map,
            }
        )

    feature_columns = sorted(observed_feature_ids, key=lambda name: int(name[2:]))
    return normalized_rows, feature_columns


def build_user_table(normalized_rows, feature_columns):
    user_records = {}
    for row in normalized_rows:
        uid = row["uid"]
        feature_map = row["feature_map"]
        if uid not in user_records:
            user_records[uid] = {
                "uid": uid,
                **{column: "0" for column in feature_columns},
            }
        for column, value in feature_map.items():
            current_value = user_records[uid][column]
            if current_value in {"", "0"}:
                user_records[uid][column] = value
            elif current_value != value:
                # Multiple rows for the same user are not expected for this dataset.
                # Keep the first observed non-zero value so the output stays deterministic.
                continue

    user_table = pd.DataFrame(user_records.values())
    ordered_columns = ["uid"] + feature_columns
    for column in ordered_columns:
        if column not in user_table.columns:
            user_table[column] = "0"
    return user_table[ordered_columns].sort_values("uid").reset_index(drop=True)


def build_sequence_table(normalized_rows):
    seq_rows = []
    fixed_iid = 1
    fixed_timestamp = 0
    for row in normalized_rows:
        seq_rows.append(
            {
                "uid": row["uid"],
                "iid": fixed_iid,
                "timestamp": fixed_timestamp,
                "label": row["label"],
            }
        )

    seq_table = pd.DataFrame(seq_rows)
    item_table = pd.DataFrame([{"iid": fixed_iid}])
    return seq_table, item_table


def build_format_rows(user_table, seq_table, item_table):
    rows = []
    for column in user_table.columns:
        rows.append(("user_info.csv", column, "int", "false"))
    for column in item_table.columns:
        rows.append(("item_fea.csv", column, "int", "false"))
    for column in seq_table.columns:
        rows.append(("seq.csv", column, "int", "false"))
    return rows


def write_csv(frame, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")


def write_format_file(format_rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_name", "column_name", "data_type", "is_list"])
        writer.writerows(format_rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a flat user-level CSV into generative_ranking-style seq/user/item/format files."
    )
    parser.add_argument("input_csv", type=Path, help="Path to the source CSV file.")
    parser.add_argument(
        "output_dir", type=Path, help="Directory for generated CSV files."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = pd.read_csv(args.input_csv)
    normalized_rows, feature_columns = normalize_source_frame(source)
    user_table = build_user_table(normalized_rows, feature_columns)
    seq_table, item_table = build_sequence_table(normalized_rows)
    format_rows = build_format_rows(user_table, seq_table, item_table)

    output_dir = args.output_dir.resolve()
    write_csv(user_table, output_dir / "user_info.csv")
    write_csv(item_table, output_dir / "item_fea.csv")
    write_csv(seq_table, output_dir / "seq.csv")
    write_format_file(format_rows, output_dir / "data_format.csv")

    print(f"Wrote {len(user_table)} users to {output_dir / 'user_info.csv'}")
    print(f"Wrote {len(item_table)} items to {output_dir / 'item_fea.csv'}")
    print(f"Wrote {len(seq_table)} interactions to {output_dir / 'seq.csv'}")
    print(f"Wrote format metadata to {output_dir / 'data_format.csv'}")
    print(
        "Note: iid=1 and timestamp=0 are fixed placeholders for format conversion only."
    )


if __name__ == "__main__":
    main()
