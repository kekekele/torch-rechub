from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve
import zipfile

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = PROJECT_DIR / "data_preprocess" / "taobao_adx1" / "processed"
DEFAULT_DOWNLOAD_DIR = PROJECT_DIR / "outputs" / "downloads" / "taobaoad_x1"

DEFAULT_GITHUB_ARCHIVE_URL = (
    "https://github.com/reczoo/Datasets/archive/refs/heads/main.zip"
)
DEFAULT_HF_TRAIN_URL = "https://huggingface.co/datasets/reczoo/TaobaoAd_x1/resolve/main/train.csv?download=true"
DEFAULT_HF_TEST_URL = "https://huggingface.co/datasets/reczoo/TaobaoAd_x1/resolve/main/test.csv?download=true"

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

SEQ_LIST_COLUMNS = ["btag_his", "cate_his", "brand_his"]

MISSING_TOKENS = {"", "nan", "null", "none", "na", "n/a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TaobaoAd_x1 train/test CSV files to RecKit raw format."
    )
    parser.add_argument(
        "--train-path", type=Path, default=None, help="Path to train.csv"
    )
    parser.add_argument(
        "--test-path", type=Path, default=None, help="Path to test.csv (optional)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RAW_DIR, help="Output raw directory"
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help="Directory used to download and extract source data",
    )
    parser.add_argument(
        "--github-archive-url",
        default=DEFAULT_GITHUB_ARCHIVE_URL,
        help="GitHub archive URL to download and unzip first",
    )
    parser.add_argument(
        "--hf-train-url",
        default=DEFAULT_HF_TRAIN_URL,
        help="Fallback URL for train.csv",
    )
    parser.add_argument(
        "--hf-test-url", default=DEFAULT_HF_TEST_URL, help="Fallback URL for test.csv"
    )
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Disable auto download when raw files are missing",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Convert train only even if test.csv is unavailable",
    )
    parser.add_argument("--chunksize", type=int, default=500000, help="CSV chunk size")
    parser.add_argument(
        "--log-every-chunks",
        type=int,
        default=1,
        help="Print one progress log every N chunks while converting CSV files",
    )
    parser.add_argument("--encoding", default="utf-8", help="CSV encoding")
    parser.add_argument("--sep", default=",", help="CSV separator")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output files"
    )
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[prepare_taobaoad_x1] {message}", flush=True)


def _download_file(url: str, target_path: Path) -> tuple[bool, str | None]:
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, str(target_path))
        if target_path.exists() and target_path.stat().st_size > 0:
            return True, None
        return False, f"downloaded empty file from {url}"
    except HTTPError as exc:
        return False, f"HTTPError {exc.code} for {url}: {exc.reason}"
    except URLError as exc:
        return False, f"URLError for {url}: {exc.reason}"
    except Exception as exc:
        return False, f"download failed for {url}: {exc}"


def _unzip_file(zip_path: Path, extract_dir: Path) -> tuple[bool, str | None]:
    try:
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        return True, None
    except Exception as exc:
        return False, f"unzip failed for {zip_path}: {exc}"


def _download_with_fallback(
    urls: list[str], target_path: Path
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for url in urls:
        ok, err = _download_file(url, target_path)
        if ok:
            return True, errors
        if err:
            errors.append(err)
    return False, errors


def _find_file_recursive(root: Path, file_name: str) -> Path | None:
    if not root.exists():
        return None
    candidates = sorted(
        root.rglob(file_name),
        key=lambda p: (
            "Taobao" not in str(p) or "TaobaoAd_x1" not in str(p),
            len(str(p)),
        ),
    )
    return candidates[0] if candidates else None


def _resolve_local_data_paths(
    args: argparse.Namespace,
) -> tuple[Path | None, Path | None]:
    diagnostics: list[str] = []
    train_path = args.train_path if args.train_path is None else Path(args.train_path)
    test_path = args.test_path if args.test_path is None else Path(args.test_path)

    if train_path is not None and train_path.exists():
        if test_path is None or test_path.exists() or args.skip_test:
            _log(f"use local input files: train={train_path}, test={test_path}")
            return train_path, test_path

    if args.no_auto_download:
        diagnostics.append("auto download disabled via --no-auto-download")
        return train_path, test_path

    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    train_csv = download_dir / "train.csv"
    test_csv = download_dir / "test.csv"
    if train_csv.exists() and (test_csv.exists() or args.skip_test):
        diagnostics.append(f"reuse cached train/test in {download_dir}")
        _log(f"reuse cached download files from {download_dir}")
        return train_csv, (None if args.skip_test else test_csv)

    archive_path = download_dir / "reczoo_datasets_main.zip"
    archive_extract_dir = download_dir / "reczoo_datasets_main"
    if not archive_path.exists():
        _log(f"download GitHub archive: {args.github_archive_url}")
        ok, err = _download_file(str(args.github_archive_url), archive_path)
        diagnostics.append(f"github archive download: {'ok' if ok else 'failed'}")
        if err:
            diagnostics.append(err)
    else:
        diagnostics.append(f"github archive cache exists: {archive_path}")
        _log(f"reuse cached GitHub archive: {archive_path}")
    if archive_path.exists() and not archive_extract_dir.exists():
        _log(f"extract GitHub archive to {archive_extract_dir}")
        ok, err = _unzip_file(archive_path, archive_extract_dir)
        diagnostics.append(f"github archive unzip: {'ok' if ok else 'failed'}")
        if err:
            diagnostics.append(err)
    elif archive_extract_dir.exists():
        diagnostics.append(f"github archive already extracted: {archive_extract_dir}")
        _log(f"reuse extracted GitHub archive: {archive_extract_dir}")

    extracted_train = _find_file_recursive(archive_extract_dir, "train.csv")
    extracted_test = _find_file_recursive(archive_extract_dir, "test.csv")
    diagnostics.append(
        "github extracted files: "
        f"train={str(extracted_train) if extracted_train else 'None'}, "
        f"test={str(extracted_test) if extracted_test else 'None'}"
    )
    _log(
        "archive lookup result: "
        f"train={str(extracted_train) if extracted_train else 'None'}, "
        f"test={str(extracted_test) if extracted_test else 'None'}"
    )
    if extracted_train is not None and (extracted_test is not None or args.skip_test):
        if not train_csv.exists():
            train_csv.write_bytes(extracted_train.read_bytes())
        if not args.skip_test and extracted_test is not None and not test_csv.exists():
            test_csv.write_bytes(extracted_test.read_bytes())
        _log(
            f"resolved input files from extracted archive: train={train_csv}, test={None if args.skip_test else test_csv}"
        )
        return train_csv, (None if args.skip_test else test_csv)

    if not train_csv.exists():
        _log("download train.csv from HuggingFace mirrors")
        train_urls = [
            str(args.hf_train_url),
            str(args.hf_train_url).replace("?download=true", ""),
            "https://hf-mirror.com/datasets/reczoo/TaobaoAd_x1/resolve/main/train.csv",
        ]
        ok, errs = _download_with_fallback(train_urls, train_csv)
        diagnostics.append(f"hf train download: {'ok' if ok else 'failed'}")
        diagnostics.extend(errs)
    if not args.skip_test and not test_csv.exists():
        _log("download test.csv from HuggingFace mirrors")
        test_urls = [
            str(args.hf_test_url),
            str(args.hf_test_url).replace("?download=true", ""),
            "https://hf-mirror.com/datasets/reczoo/TaobaoAd_x1/resolve/main/test.csv",
        ]
        ok, errs = _download_with_fallback(test_urls, test_csv)
        diagnostics.append(f"hf test download: {'ok' if ok else 'failed'}")
        diagnostics.extend(errs)

    resolved_train = train_csv if train_csv.exists() else train_path
    resolved_test = (
        None if args.skip_test else (test_csv if test_csv.exists() else test_path)
    )

    if diagnostics:
        (download_dir / "download_diagnostics.log").write_text(
            "\n".join(diagnostics) + "\n", encoding="utf-8"
        )
        _log(f"wrote download diagnostics: {download_dir / 'download_diagnostics.log'}")

    if resolved_train is None or not Path(resolved_train).exists():
        raise FileNotFoundError(
            "train.csv not found after auto-download attempts. "
            f"See diagnostics: {download_dir / 'download_diagnostics.log'}"
        )
    if not args.skip_test and (
        resolved_test is None or not Path(resolved_test).exists()
    ):
        raise FileNotFoundError(
            "test.csv not found after auto-download attempts. "
            f"See diagnostics: {download_dir / 'download_diagnostics.log'}"
        )

    return resolved_train, resolved_test


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip().lower()
    return text in MISSING_TOKENS


def _as_str_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _as_int_series(series: pd.Series, default: int = 0) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").fillna(default)
    return out.astype("int64")


def _as_float_series(series: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").fillna(default)
    return out.astype("float64")


def _normalize_list_cell(value: Any) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, (list, tuple)):
        tokens = [str(item).strip() for item in value]
    else:
        text = str(value).strip()
        if text == "0":
            return ""
        tokens = [item.strip() for item in re.split(r"[\^,|]", text)]
    filtered = [
        token
        for token in tokens
        if token and token.lower() not in MISSING_TOKENS and token != "0"
    ]
    return ",".join(filtered)


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_to_actual = {name.lower(): name for name in columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in lower_to_actual:
            return lower_to_actual[key]
    return None


def _find_columns(columns: list[str]) -> dict[str, str | None]:
    return {
        "uid": _pick_column(columns, ["uid", "userid", "user", "nick", "user_id"]),
        "iid": _pick_column(columns, ["iid", "adgroup_id", "item_id"]),
        "timestamp": _pick_column(columns, ["timestamp", "time_stamp", "time"]),
        "clk": _pick_column(columns, ["clk", "click", "label", "action"]),
        "noclk": _pick_column(columns, ["noclk", "non_click"]),
        "pid": _pick_column(columns, ["pid", "scenario"]),
        "btag_his": _pick_column(columns, ["btag_his", "hist_btag"]),
        "cate_his": _pick_column(columns, ["cate_his", "hist_cate"]),
        "brand_his": _pick_column(columns, ["brand_his", "hist_brand"]),
        "cms_segid": _pick_column(columns, ["cms_segid"]),
        "cms_group_id": _pick_column(columns, ["cms_group_id"]),
        "final_gender_code": _pick_column(columns, ["final_gender_code", "gender"]),
        "age_level": _pick_column(columns, ["age_level"]),
        "pvalue_level": _pick_column(columns, ["pvalue_level"]),
        "shopping_level": _pick_column(columns, ["shopping_level"]),
        "occupation": _pick_column(columns, ["occupation"]),
        "new_user_class_level": _pick_column(columns, ["new_user_class_level"]),
        "cate_id": _pick_column(columns, ["cate_id", "cate"]),
        "campaign_id": _pick_column(columns, ["campaign_id"]),
        "customer": _pick_column(columns, ["customer", "customer_id"]),
        "brand": _pick_column(columns, ["brand"]),
        "price": _pick_column(columns, ["price"]),
    }


def _fallback_timestamps(
    uids: pd.Series, state: dict[str, int], offset: int
) -> pd.Series:
    values = []
    for uid in uids.tolist():
        raw_uid = str(uid)
        state[raw_uid] += 1
        values.append(offset + state[raw_uid])
    return pd.Series(values, index=uids.index, dtype="int64")


def _make_action(df: pd.DataFrame, column_map: dict[str, str | None]) -> pd.Series:
    clk_col = column_map.get("clk")
    noclk_col = column_map.get("noclk")
    if clk_col is not None:
        return (_as_int_series(df[clk_col], default=0) > 0).astype("int64")
    if noclk_col is not None:
        return (_as_int_series(df[noclk_col], default=1) == 0).astype("int64")
    raise ValueError(
        "No click label column found. Expected one of: clk/click/label/action or noclk"
    )


def _build_seq_frame(
    chunk: pd.DataFrame,
    column_map: dict[str, str | None],
    split_source: str,
    fallback_state: dict[str, int],
    timestamp_offset: int,
) -> pd.DataFrame:
    uid_col = column_map.get("uid")
    iid_col = column_map.get("iid")
    if uid_col is None or iid_col is None:
        raise ValueError(
            "Missing user or item ID column. Expected userid/user and adgroup_id/iid"
        )

    seq = pd.DataFrame()
    seq["uid"] = _as_str_series(chunk[uid_col])
    seq["iid"] = _as_str_series(chunk[iid_col])

    ts_col = column_map.get("timestamp")
    if ts_col is None:
        seq["timestamp"] = _fallback_timestamps(
            seq["uid"], fallback_state, timestamp_offset
        )
    else:
        ts = pd.to_numeric(chunk[ts_col], errors="coerce")
        missing_mask = ts.isna()
        ts = ts.fillna(0).astype("int64")
        if bool(missing_mask.any()):
            ts.loc[missing_mask] = _fallback_timestamps(
                seq.loc[missing_mask, "uid"], fallback_state, timestamp_offset
            ).values
        seq["timestamp"] = ts

    seq["action"] = _make_action(chunk, column_map)

    pid_col = column_map.get("pid")
    seq["pid"] = _as_int_series(chunk[pid_col], default=0) if pid_col is not None else 0

    for name in SEQ_LIST_COLUMNS:
        source = column_map.get(name)
        if source is None:
            seq[name] = ""
        else:
            seq[name] = chunk[source].apply(_normalize_list_cell)

    seq["split_source"] = split_source
    return seq


def _get_user_frame(
    chunk: pd.DataFrame, column_map: dict[str, str | None], seq: pd.DataFrame
) -> pd.DataFrame:
    user = pd.DataFrame()
    user["uid"] = seq["uid"].values
    user["timestamp"] = seq["timestamp"].values
    for name in USER_COLUMNS:
        source = column_map.get(name)
        if source is None:
            user[name] = 0
        else:
            user[name] = _as_int_series(chunk[source], default=0)
    return user


def _get_item_frame(
    chunk: pd.DataFrame, column_map: dict[str, str | None], seq: pd.DataFrame
) -> pd.DataFrame:
    item = pd.DataFrame()
    item["iid"] = seq["iid"].values
    item["timestamp"] = seq["timestamp"].values
    for name in ITEM_COLUMNS:
        source = column_map.get(name)
        if source is None:
            item[name] = 0.0 if name == "price" else 0
        elif name == "price":
            item[name] = _as_float_series(chunk[source], default=0.0)
        else:
            item[name] = _as_int_series(chunk[source], default=0)
    return item


def _update_latest(
    frame: pd.DataFrame,
    key_col: str,
    ts_col: str,
    value_cols: list[str],
    target: dict[str, tuple[int, dict[str, Any]]],
) -> None:
    normalized = frame.copy()
    normalized[ts_col] = pd.to_numeric(normalized[ts_col], errors="coerce").fillna(-1)
    latest = normalized.sort_values(ts_col).drop_duplicates(key_col, keep="last")
    for row in latest.itertuples(index=False):
        key = str(getattr(row, key_col))
        ts = int(getattr(row, ts_col))
        payload = {col: getattr(row, col) for col in value_cols}
        prev = target.get(key)
        if prev is None or ts >= prev[0]:
            target[key] = (ts, payload)


def _append_seq_csv(path: Path, seq: pd.DataFrame, write_header: bool) -> None:
    seq.to_csv(
        path, index=False, mode="w" if write_header else "a", header=write_header
    )


def _assert_valid_seq_timestamps(
    seq: pd.DataFrame, split_source: str, chunk_count: int
) -> None:
    if "timestamp" not in seq.columns:
        raise ValueError(
            f"seq chunk missing timestamp column: split={split_source}, chunk={chunk_count}"
        )

    numeric_ts = pd.to_numeric(seq["timestamp"], errors="coerce")
    invalid_mask = numeric_ts.isna()
    if not bool(invalid_mask.any()):
        return

    invalid_rows = seq.loc[
        invalid_mask,
        [
            col
            for col in ["uid", "iid", "timestamp", "split_source"]
            if col in seq.columns
        ],
    ].head(5)
    raise ValueError(
        "Invalid timestamp values detected before writing seq.csv: "
        f"split={split_source}, chunk={chunk_count}, invalid_rows={int(invalid_mask.sum())}. "
        f"Examples: {invalid_rows.to_dict(orient='records')}"
    )


def _write_side_tables(
    output_dir: Path,
    users: dict[str, tuple[int, dict[str, Any]]],
    items: dict[str, tuple[int, dict[str, Any]]],
) -> None:
    user_rows = []
    for uid, (_, payload) in users.items():
        row = {"uid": uid}
        row.update(payload)
        user_rows.append(row)
    item_rows = []
    for iid, (_, payload) in items.items():
        row = {"iid": iid}
        row.update(payload)
        item_rows.append(row)

    user_df = pd.DataFrame(user_rows)
    item_df = pd.DataFrame(item_rows)

    if not user_df.empty:
        user_df = user_df.sort_values("uid").reset_index(drop=True)
    if not item_df.empty:
        item_df = item_df.sort_values("iid").reset_index(drop=True)

    _log(f"write user_info.csv rows={len(user_df)}")
    user_df.to_csv(output_dir / "user_info.csv", index=False)
    _log(f"write item_fea.csv rows={len(item_df)}")
    item_df.to_csv(output_dir / "item_fea.csv", index=False)


def _write_data_format(output_dir: Path) -> None:
    rows = [
        ("seq.csv", "uid", "str", False),
        ("seq.csv", "iid", "str", False),
        ("seq.csv", "timestamp", "int", False),
        ("seq.csv", "action", "int", False),
        ("seq.csv", "pid", "int", False),
        ("seq.csv", "btag_his", "str", True),
        ("seq.csv", "cate_his", "int", True),
        ("seq.csv", "brand_his", "int", True),
        ("seq.csv", "split_source", "str", False),
    ]
    rows.extend(("user_info.csv", "uid", "str", False) for _ in [0])
    rows.extend(("user_info.csv", col, "int", False) for col in USER_COLUMNS)
    rows.extend(("item_fea.csv", "iid", "str", False) for _ in [0])
    for col in ITEM_COLUMNS:
        rows.append(("item_fea.csv", col, "float" if col == "price" else "int", False))

    df = pd.DataFrame(
        rows, columns=["file_name", "column_name", "data_type", "is_list"]
    )
    df["is_list"] = df["is_list"].map(lambda flag: "true" if flag else "false")
    _log(f"write data_format.csv rows={len(df)}")
    df.to_csv(output_dir / "data_format.csv", index=False)


def _process_file(
    file_path: Path,
    split_source: str,
    seq_path: Path,
    chunksize: int,
    log_every_chunks: int,
    encoding: str,
    sep: str,
    write_header: bool,
    user_latest: dict[str, tuple[int, dict[str, Any]]],
    item_latest: dict[str, tuple[int, dict[str, Any]]],
    fallback_state: dict[str, int],
    timestamp_offset: int,
) -> tuple[bool, dict[str, Any]]:
    row_count = 0
    action_sum = 0
    chunk_count = 0

    _log(f"start processing {split_source} file: {file_path}")
    reader = pd.read_csv(
        file_path, chunksize=chunksize, dtype=object, encoding=encoding, sep=sep
    )
    for chunk in reader:
        chunk_count += 1
        column_map = _find_columns(list(chunk.columns))
        seq = _build_seq_frame(
            chunk, column_map, split_source, fallback_state, timestamp_offset
        )
        _assert_valid_seq_timestamps(seq, split_source, chunk_count)
        _append_seq_csv(seq_path, seq, write_header)
        write_header = False

        user_frame = _get_user_frame(chunk, column_map, seq)
        item_frame = _get_item_frame(chunk, column_map, seq)
        _update_latest(user_frame, "uid", "timestamp", USER_COLUMNS, user_latest)
        _update_latest(item_frame, "iid", "timestamp", ITEM_COLUMNS, item_latest)

        row_count += int(len(seq))
        action_sum += int(seq["action"].sum())
        if log_every_chunks > 0 and chunk_count % log_every_chunks == 0:
            _log(
                f"{split_source} progress: chunks={chunk_count}, rows={row_count}, "
                f"positives={action_sum}, negatives={row_count - action_sum}"
            )

    _log(
        f"finished {split_source} file: chunks={chunk_count}, rows={row_count}, "
        f"positives={action_sum}, negatives={row_count - action_sum}"
    )
    return write_header, {
        "file": str(file_path),
        "split_source": split_source,
        "rows": row_count,
        "positives": action_sum,
        "negatives": row_count - action_sum,
        "chunks": chunk_count,
    }


def main() -> None:
    args = parse_args()
    _log(
        f"start conversion: output_dir={args.output_dir}, chunksize={args.chunksize}, "
        f"skip_test={args.skip_test}, force={args.force}"
    )
    train_path, test_path = _resolve_local_data_paths(args)
    if train_path is None or not Path(train_path).exists():
        raise FileNotFoundError("train.csv not found.")
    if not args.skip_test and (test_path is None or not Path(test_path).exists()):
        raise FileNotFoundError(
            "test.csv not found. Provide --test-path or use --skip-test."
        )
    _log(f"resolved input files: train={train_path}, test={test_path}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        for name in [
            "seq.csv",
            "user_info.csv",
            "item_fea.csv",
            "data_format.csv",
            "conversion_summary.json",
        ]:
            path = output_dir / name
            if path.exists():
                path.unlink()

    seq_path = output_dir / "seq.csv"
    if seq_path.exists() and not args.force:
        raise FileExistsError(f"{seq_path} already exists. Use --force to overwrite.")

    user_latest: dict[str, tuple[int, dict[str, Any]]] = {}
    item_latest: dict[str, tuple[int, dict[str, Any]]] = {}
    fallback_state: dict[str, int] = defaultdict(int)

    write_header = True
    reports = []

    write_header, report = _process_file(
        file_path=Path(train_path),
        split_source="train",
        seq_path=seq_path,
        chunksize=int(args.chunksize),
        log_every_chunks=max(1, int(args.log_every_chunks)),
        encoding=args.encoding,
        sep=args.sep,
        write_header=write_header,
        user_latest=user_latest,
        item_latest=item_latest,
        fallback_state=fallback_state,
        timestamp_offset=0,
    )
    reports.append(report)

    if test_path is not None and not args.skip_test:
        write_header, report = _process_file(
            file_path=Path(test_path),
            split_source="test",
            seq_path=seq_path,
            chunksize=int(args.chunksize),
            log_every_chunks=max(1, int(args.log_every_chunks)),
            encoding=args.encoding,
            sep=args.sep,
            write_header=write_header,
            user_latest=user_latest,
            item_latest=item_latest,
            fallback_state=fallback_state,
            timestamp_offset=2_000_000_000,
        )
        reports.append(report)

    _write_side_tables(output_dir, user_latest, item_latest)
    _write_data_format(output_dir)

    total_rows = sum(item["rows"] for item in reports)
    total_pos = sum(item["positives"] for item in reports)
    summary = {
        "output_dir": str(output_dir),
        "seq_rows": int(total_rows),
        "positives": int(total_pos),
        "negatives": int(total_rows - total_pos),
        "users": int(len(user_latest)),
        "items": int(len(item_latest)),
        "reports": reports,
    }
    (output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log(
        f"finished conversion: seq_rows={summary['seq_rows']}, users={summary['users']}, "
        f"items={summary['items']}, summary={output_dir / 'conversion_summary.json'}"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
