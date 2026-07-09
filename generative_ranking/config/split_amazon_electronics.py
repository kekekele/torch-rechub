#!/usr/bin/env python3
import argparse
import csv
import sqlite3
from pathlib import Path

CSV_FIELD_LIMIT = 1024 * 1024 * 128
DEFAULT_BATCH_SIZE = 100_000
DB_FILE_NAME = "split_work.sqlite"


def count_history_length(history: str) -> int:
    value = history.strip()
    if not value:
        return 0
    return value.count("^") + 1


def first_entry(history: str) -> str:
    value = history.strip()
    if not value:
        return ""
    return value.split("^", 1)[0]


class SplitDatabase:
    def __init__(self, db_path: Path) -> None:
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA temp_store = MEMORY")
        self.conn.execute("PRAGMA cache_size = -200000")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS staged (
                row_num INTEGER PRIMARY KEY AUTOINCREMENT,
                label INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                cate_id TEXT NOT NULL,
                hist_len INTEGER NOT NULL,
                seed_item_id TEXT,
                seed_cate_id TEXT
            )
            """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                uid TEXT PRIMARY KEY
            )
            """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS items (
                iid TEXT PRIMARY KEY,
                cate_id TEXT NOT NULL
            )
            """)
        self.conn.commit()

    def insert_batches(self, staged_rows, users, items) -> None:
        self.conn.executemany(
            """
            INSERT INTO staged (
                label, user_id, item_id, cate_id, hist_len, seed_item_id, seed_cate_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            staged_rows,
        )
        self.conn.executemany(
            "INSERT OR IGNORE INTO users (uid) VALUES (?)",
            users,
        )
        self.conn.executemany(
            "INSERT OR IGNORE INTO items (iid, cate_id) VALUES (?, ?)",
            items,
        )
        self.conn.commit()

    def prepare_indexes(self) -> None:
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_staged_user_hist_row ON staged (user_id, hist_len, row_num)"
        )
        self.conn.commit()

    def iter_sorted_rows(self):
        cursor = self.conn.execute("""
            SELECT user_id, item_id, cate_id, label, hist_len, seed_item_id, seed_cate_id
            FROM staged
            ORDER BY user_id, hist_len, row_num
            """)
        yield from cursor

    def iter_users(self):
        cursor = self.conn.execute("SELECT uid FROM users ORDER BY uid")
        yield from cursor

    def iter_items(self):
        cursor = self.conn.execute("SELECT iid, cate_id FROM items ORDER BY iid")
        yield from cursor

    def close(self) -> None:
        self.conn.close()


def flush_user_rows(seq_writer, user_rows) -> int:
    if not user_rows:
        return 0

    output_count = 0
    seed_item_id = ""
    seed_cate_id = ""
    for row in user_rows:
        if row[4] == 1 and row[5] and row[6]:
            seed_item_id = row[5]
            seed_cate_id = row[6]
            break

    uid = user_rows[0][0]
    if seed_item_id and seed_cate_id:
        seq_writer.writerow(
            {
                "uid": uid,
                "iid": seed_item_id,
                "timestamp": 1,
                "cate_id": seed_cate_id,
                "label": 1,
            }
        )
        output_count += 1

    timestamp = 2
    for (
        uid,
        item_id,
        cate_id,
        label,
        _hist_len,
        _seed_item_id,
        _seed_cate_id,
    ) in user_rows:
        seq_writer.writerow(
            {
                "uid": uid,
                "iid": item_id,
                "timestamp": timestamp,
                "cate_id": cate_id,
                "label": label,
            }
        )
        timestamp += 1
        output_count += 1

    return output_count


def build_database(
    input_csv: Path, db: SplitDatabase, batch_size: int
) -> tuple[int, int]:
    staged_rows = []
    users = []
    items = []
    input_count = 0

    csv.field_size_limit(CSV_FIELD_LIMIT)
    with input_csv.open("r", encoding="utf-8", newline="") as source_file:
        reader = csv.DictReader(source_file)
        required_columns = {
            "label",
            "user_id",
            "item_id",
            "cate_id",
            "item_history",
            "cate_history",
        }
        if reader.fieldnames is None or not required_columns.issubset(
            set(reader.fieldnames)
        ):
            missing = sorted(required_columns - set(reader.fieldnames or []))
            raise ValueError(f"Input CSV is missing columns: {', '.join(missing)}")

        for row in reader:
            label = int(str(row["label"]).strip())
            user_id = str(row["user_id"]).strip()
            item_id = str(row["item_id"]).strip()
            cate_id = str(row["cate_id"]).strip()
            item_history = str(row["item_history"]).strip()
            cate_history = str(row["cate_history"]).strip()

            hist_len = count_history_length(item_history)
            seed_item_id = first_entry(item_history) if hist_len > 0 else ""
            seed_cate_id = first_entry(cate_history) if hist_len > 0 else ""

            staged_rows.append(
                (label, user_id, item_id, cate_id, hist_len, seed_item_id, seed_cate_id)
            )
            users.append((user_id,))
            items.append((item_id, cate_id))
            if hist_len == 1 and seed_item_id and seed_cate_id:
                items.append((seed_item_id, seed_cate_id))
            input_count += 1

            if len(staged_rows) >= batch_size:
                db.insert_batches(staged_rows, users, items)
                staged_rows.clear()
                users.clear()
                items.clear()

    if staged_rows:
        db.insert_batches(staged_rows, users, items)

    db.prepare_indexes()
    return input_count, len(staged_rows)


def write_seq_file(output_dir: Path, db: SplitDatabase) -> int:
    seq_path = output_dir / "seq.csv"
    written = 0

    with seq_path.open("w", encoding="utf-8", newline="") as seq_file:
        writer = csv.DictWriter(
            seq_file,
            fieldnames=["uid", "iid", "timestamp", "cate_id", "label"],
        )
        writer.writeheader()

        current_uid = None
        user_rows = []
        for row in db.iter_sorted_rows():
            uid = row[0]
            if current_uid is None:
                current_uid = uid
            if uid != current_uid:
                written += flush_user_rows(writer, user_rows)
                user_rows = []
                current_uid = uid
            user_rows.append(row)

        if user_rows:
            written += flush_user_rows(writer, user_rows)

    return written


def write_user_file(output_dir: Path, db: SplitDatabase) -> None:
    user_path = output_dir / "user_info.csv"
    with user_path.open("w", encoding="utf-8", newline="") as user_file:
        writer = csv.writer(user_file)
        writer.writerow(["uid"])
        for (uid,) in db.iter_users():
            writer.writerow([uid])


def write_item_file(output_dir: Path, db: SplitDatabase) -> None:
    item_path = output_dir / "item_fea.csv"
    with item_path.open("w", encoding="utf-8", newline="") as item_file:
        writer = csv.writer(item_file)
        writer.writerow(["iid", "cate_id"])
        for iid, cate_id in db.iter_items():
            writer.writerow([iid, cate_id])


def write_format_file(output_dir: Path) -> None:
    format_path = output_dir / "data_format.csv"
    rows = [
        ("user_info.csv", "uid", "int64", "false"),
        ("item_fea.csv", "iid", "int64", "false"),
        ("item_fea.csv", "cate_id", "int64", "false"),
        ("seq.csv", "uid", "int64", "false"),
        ("seq.csv", "iid", "int64", "false"),
        ("seq.csv", "timestamp", "int64", "false"),
        ("seq.csv", "cate_id", "int64", "false"),
        ("seq.csv", "label", "int64", "false"),
    ]
    with format_path.open("w", encoding="utf-8", newline="") as format_file:
        writer = csv.writer(format_file)
        writer.writerow(["file_name", "column_name", "data_type", "is_list"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split AmazonElectronics-style CSV into user, item, sequence, and format files."
    )
    parser.add_argument("input_csv", type=Path, help="Path to the source CSV file.")
    parser.add_argument(
        "output_dir", type=Path, help="Directory for generated CSV files."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Optional path for the temporary SQLite database.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of source rows to buffer before flushing into SQLite.",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Keep the temporary SQLite database after completion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be a positive integer")

    db_path = args.db_path.resolve() if args.db_path else output_dir / DB_FILE_NAME
    if db_path.exists():
        db_path.unlink()

    db = SplitDatabase(db_path)
    try:
        input_rows, _ = build_database(input_csv, db, args.batch_size)
        seq_rows = write_seq_file(output_dir, db)
        write_user_file(output_dir, db)
        write_item_file(output_dir, db)
        write_format_file(output_dir)
    finally:
        db.close()
        if not args.keep_db and db_path.exists():
            db_path.unlink()

    print(f"Processed {input_rows} source rows.")
    print(f"Wrote {seq_rows} sequence rows.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
