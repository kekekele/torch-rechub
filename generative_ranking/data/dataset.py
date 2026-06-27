import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from ..basic.features import DenseFeature, SequenceFeature, SparseFeature
from ..models.rankmixer_grouping import normalize_rankmixer_group_schema
from ..util.data import DataGenerator, pad_sequences


def ensure_source_files(config):
    source_cfg = dict(config.get("source", {}))
    tables_cfg = dict(source_cfg.get("tables", {}))
    data_dir = config.get("data_dir", "")
    missing = []
    for table_cfg in tables_cfg.values():
        table_path = _resolve_table_path(data_dir, table_cfg)
        if not pd.io.common.file_exists(table_path):
            missing.append(table_path)
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Dataset source files are required and must already exist:\n{missing_list}")


def _resolve_table_path(data_dir, table_cfg):
    path = table_cfg.get("path")
    if not path:
        raise ValueError(f"Table config must define path: {table_cfg}")
    return str(pd.io.common.stringify_path(f"{data_dir.rstrip('/\\')}/{path}" if data_dir else path))


def _normalize_read_csv_kwargs(table_cfg):
    kwargs = dict(table_cfg.get("read_csv", {}))
    if "header" in kwargs and kwargs["header"] == "none":
        kwargs["header"] = None
    return kwargs


def _apply_transform(frame, transform):
    transform_type = str(transform.get("type", "")).strip().lower()
    target = str(transform.get("target", "")).strip()
    source = str(transform.get("source", "")).strip()
    if not target:
        raise ValueError(f"Transform must define target: {transform}")

    if transform_type == "split_first":
        separator = str(transform.get("sep", "|"))
        default = transform.get("default", "")
        values = frame[source].fillna(default).astype(str)
        frame[target] = values.apply(lambda value: value.split(separator)[0] if value else default)
        return frame

    if transform_type == "binary_compare":
        op = str(transform.get("op", "gt")).lower()
        threshold = transform.get("threshold")
        if op == "gt":
            output = frame[source] > threshold
        elif op == "ge":
            output = frame[source] >= threshold
        elif op == "lt":
            output = frame[source] < threshold
        elif op == "le":
            output = frame[source] <= threshold
        elif op == "eq":
            output = frame[source] == threshold
        else:
            raise ValueError(f"Unsupported binary_compare op: {op}")
        frame[target] = output.astype(transform.get("dtype", "int32"))
        return frame

    if transform_type == "astype":
        frame[target] = frame[source].astype(transform.get("dtype", "float32"))
        return frame

    raise ValueError(f"Unsupported transform type: {transform_type}")


def _apply_transforms(frame, transforms):
    transformed = frame.copy()
    for transform in transforms or []:
        transformed = _apply_transform(transformed, transform)
    return transformed


def _read_source_tables(config):
    source_cfg = dict(config.get("source", {}))
    tables_cfg = dict(source_cfg.get("tables", {}))
    if not tables_cfg:
        raise ValueError("Config must define source.tables for generic table loading")

    data_dir = config.get("data_dir", "")
    tables = {}
    for table_name, table_cfg in tables_cfg.items():
        read_kwargs = _normalize_read_csv_kwargs(table_cfg)
        table = pd.read_csv(_resolve_table_path(data_dir, table_cfg), **read_kwargs)
        table = _apply_transforms(table, table_cfg.get("transforms", []))
        select_columns = table_cfg.get("select_columns")
        if select_columns:
            table = table[list(select_columns)].copy()
        tables[str(table_name)] = table
    return tables, source_cfg


def load_source_frame(config):
    tables, source_cfg = _read_source_tables(config)
    base_table_name = str(source_cfg.get("base_table", "interactions"))
    if base_table_name not in tables:
        raise ValueError(f"Base table '{base_table_name}' not found in source.tables")

    merged = tables[base_table_name].copy()
    for join_cfg in source_cfg.get("joins", []):
        right_name = str(join_cfg["right_table"])
        if right_name not in tables:
            raise ValueError(f"Join table '{right_name}' not found in source.tables")
        right = tables[right_name]
        select_columns = join_cfg.get("select_columns")
        if select_columns:
            right = right[list(select_columns)].copy()
        merged = merged.merge(
            right,
            how=join_cfg.get("how", "left"),
            left_on=join_cfg.get("left_on"),
            right_on=join_cfg.get("right_on"),
        )

    merged = _apply_transforms(merged, source_cfg.get("post_merge_transforms", []))
    select_columns = source_cfg.get("select_columns")
    if select_columns:
        merged = merged[list(select_columns)].copy()
    return merged


def _normalize_feature_entry(entry):
    if isinstance(entry, str):
        return {"name": entry, "source": entry}
    if not isinstance(entry, dict):
        raise TypeError(f"Feature entry must be a string or dict, got {type(entry)!r}")
    name = str(entry.get("name", entry.get("source", ""))).strip()
    source = str(entry.get("source", entry.get("name", ""))).strip()
    if not name or not source:
        raise ValueError(f"Invalid feature entry: {entry}")
    normalized = dict(entry)
    normalized["name"] = name
    normalized["source"] = source
    return normalized


def _normalize_sequence_entry(entry):
    normalized = _normalize_feature_entry(entry)
    normalized["shared_with"] = normalized.get("shared_with")
    normalized["rankmixer_pooling"] = str(normalized.get("rankmixer_pooling", "concat")).lower()
    normalized["dcn_pooling"] = str(normalized.get("dcn_pooling", "mean")).lower()
    return normalized


def normalize_three_table_spec(config):
    sample_builder = dict(config.get("sample_builder", {}))
    feature_cfg = dict(config.get("features", {}))
    semantic_schema = normalize_rankmixer_group_schema(config.get("semantic_schema", []))

    split_cfg = dict(sample_builder.get("split", {}))
    split_type = str(split_cfg.get("type", "global_time_ratio")).lower()
    split_ratios = tuple(split_cfg.get("ratios", (0.8, 0.1, 0.1)))
    if split_type != "global_time_ratio":
        raise ValueError(f"Unsupported split type: {split_type}")
    if len(split_ratios) != 3 or abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ValueError(f"split.ratios must contain three values summing to 1.0, got {split_ratios}")

    normalized = {
        "sample_builder": {
            "user_id_col": str(sample_builder.get("user_id_col", "user_id")),
            "timestamp_col": str(sample_builder.get("timestamp_col", "timestamp")),
            "label_col": str(sample_builder.get("label_col", "label")),
            "max_seq_len": int(sample_builder.get("max_seq_len", config.get("dataset_params", {}).get("max_seq_len", 50))),
            "history_positive_only": bool(sample_builder.get("history_filter", {}).get("positive_only", True)),
            "split_type": split_type,
            "split_ratios": split_ratios,
        },
        "features": {
            "user_sparse": [_normalize_feature_entry(item) for item in feature_cfg.get("user_sparse", [])],
            "user_dense": [_normalize_feature_entry(item) for item in feature_cfg.get("user_dense", [])],
            "item_sparse": [_normalize_feature_entry(item) for item in feature_cfg.get("item_sparse", [])],
            "item_dense": [_normalize_feature_entry(item) for item in feature_cfg.get("item_dense", [])],
            "sequence": [_normalize_sequence_entry(item) for item in feature_cfg.get("sequence", [])],
            "embedding_dim": int(feature_cfg.get("embedding_dim", 16)),
            "padding_idx": feature_cfg.get("padding_idx", 0),
        },
        "semantic_schema": semantic_schema,
    }
    return normalized


def encode_tabular_features(data, sparse_cols, dense_cols=None, label_col="label", timestamp_col="timestamp"):
    encoded = data.copy()
    for col in sparse_cols:
        encoder = LabelEncoder()
        encoded[col] = encoder.fit_transform(encoded[col].astype(str)) + 1
    for col in dense_cols or []:
        encoded[col] = encoded[col].astype(np.float32)
    encoded[label_col] = encoded[label_col].astype("int32")
    encoded[timestamp_col] = encoded[timestamp_col].astype("int64")
    return encoded


def build_sequence_samples(data, spec, desc="build ranking samples"):
    sample_cfg = spec["sample_builder"]
    feature_cfg = spec["features"]
    user_id_col = sample_cfg["user_id_col"]
    timestamp_col = sample_cfg["timestamp_col"]
    label_col = sample_cfg["label_col"]
    max_seq_len = sample_cfg["max_seq_len"]
    history_positive_only = sample_cfg["history_positive_only"]

    user_feature_specs = feature_cfg["user_sparse"] + feature_cfg["user_dense"]
    item_feature_specs = feature_cfg["item_sparse"] + feature_cfg["item_dense"]
    sequence_specs = feature_cfg["sequence"]

    samples = []
    ordered = data.sort_values([user_id_col, timestamp_col]).reset_index(drop=True)
    for _, hist in tqdm(ordered.groupby(user_id_col, sort=False), desc=desc):
        current_histories = {item["name"]: [] for item in sequence_specs}
        for row in hist.itertuples(index=False):
            has_history = any(current_histories[name] for name in current_histories)
            if has_history:
                sample = {
                    label_col: int(getattr(row, label_col)),
                    timestamp_col: int(getattr(row, timestamp_col)),
                }
                for feature in user_feature_specs:
                    sample[feature["name"]] = getattr(row, feature["source"])
                for feature in item_feature_specs:
                    sample[feature["name"]] = getattr(row, feature["source"])
                for feature in sequence_specs:
                    sample[feature["name"]] = list(current_histories[feature["name"]][-max_seq_len:])
                samples.append(sample)
            if not history_positive_only or int(getattr(row, label_col)) == 1:
                for feature in sequence_specs:
                    current_histories[feature["name"]].append(getattr(row, feature["source"]))

    if not samples:
        raise ValueError("No ranking samples were constructed from the normalized interaction table.")
    return pd.DataFrame(samples).sort_values(timestamp_col).reset_index(drop=True)


def split_samples(samples, spec):
    ratios = spec["sample_builder"]["split_ratios"]
    n_samples = len(samples)
    train_end = int(n_samples * ratios[0])
    val_end = int(n_samples * (ratios[0] + ratios[1]))
    train = samples.iloc[:train_end].copy()
    val = samples.iloc[train_end:val_end].copy()
    test = samples.iloc[val_end:].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Train/valid/test split produced an empty partition.")
    return train, val, test


def frame_to_model_input(frame, spec):
    sample_cfg = spec["sample_builder"]
    feature_cfg = spec["features"]
    label_col = sample_cfg["label_col"]
    timestamp_col = sample_cfg["timestamp_col"]
    max_seq_len = sample_cfg["max_seq_len"]

    frame = frame.copy()
    labels = frame.pop(label_col).to_numpy(dtype=np.float32)
    if timestamp_col in frame.columns:
        frame.drop(columns=[timestamp_col], inplace=True)

    x_dict = {}
    for feature in feature_cfg["user_sparse"] + feature_cfg["item_sparse"]:
        x_dict[feature["name"]] = frame[feature["name"]].to_numpy(dtype=np.int64)
    for feature in feature_cfg["user_dense"] + feature_cfg["item_dense"]:
        x_dict[feature["name"]] = frame[feature["name"]].to_numpy(dtype=np.float32)
    for feature in feature_cfg["sequence"]:
        x_dict[feature["name"]] = pad_sequences(
            frame[feature["name"]].tolist(),
            maxlen=max_seq_len,
            padding="pre",
            truncating="pre",
            value=feature_cfg["padding_idx"],
        ).astype(np.int64)
    return x_dict, labels


def build_feature_columns(encoded, spec):
    feature_cfg = spec["features"]
    embed_dim = feature_cfg["embedding_dim"]
    padding_idx = feature_cfg["padding_idx"]

    user_features = [
        SparseFeature(feature["name"], vocab_size=int(encoded[feature["source"]].max()) + 1, embed_dim=embed_dim, padding_idx=padding_idx)
        for feature in feature_cfg["user_sparse"]
    ]
    user_dense_features = [DenseFeature(feature["name"]) for feature in feature_cfg["user_dense"]]
    item_features = [
        SparseFeature(feature["name"], vocab_size=int(encoded[feature["source"]].max()) + 1, embed_dim=embed_dim, padding_idx=padding_idx)
        for feature in feature_cfg["item_sparse"]
    ]
    item_dense_features = [DenseFeature(feature["name"]) for feature in feature_cfg["item_dense"]]

    rankmixer_seq_features = []
    dcn_seq_features = []
    for feature in feature_cfg["sequence"]:
        vocab_size = int(encoded[feature["source"]].max()) + 1
        rankmixer_seq_features.append(
            SequenceFeature(
                feature["name"],
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                pooling=feature["rankmixer_pooling"],
                shared_with=feature.get("shared_with"),
                padding_idx=padding_idx,
            )
        )
        dcn_seq_features.append(
            SequenceFeature(
                feature["name"],
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                pooling=feature["dcn_pooling"],
                shared_with=feature.get("shared_with"),
                padding_idx=padding_idx,
            )
        )

    base_features = user_features + user_dense_features + item_features + item_dense_features
    return base_features, dcn_seq_features, rankmixer_seq_features, spec["semantic_schema"]


def prepare_three_table_dataset(config, normalized_data, dataset_name, desc="build ranking samples"):
    spec = normalize_three_table_spec(config)
    feature_cfg = spec["features"]
    sample_cfg = spec["sample_builder"]
    sparse_cols = sorted({feature["source"] for feature in feature_cfg["user_sparse"] + feature_cfg["item_sparse"] + feature_cfg["sequence"]})
    dense_cols = [feature["source"] for feature in feature_cfg["user_dense"] + feature_cfg["item_dense"]]
    encoded = encode_tabular_features(
        normalized_data,
        sparse_cols=sparse_cols,
        dense_cols=dense_cols,
        label_col=sample_cfg["label_col"],
        timestamp_col=sample_cfg["timestamp_col"],
    )
    samples = build_sequence_samples(encoded, spec, desc=desc)
    train, val, test = split_samples(samples, spec)
    train_x, train_y = frame_to_model_input(train, spec)
    val_x, val_y = frame_to_model_input(val, spec)
    test_x, test_y = frame_to_model_input(test, spec)
    generator = DataGenerator(train_x, train_y)
    train_dl, val_dl, test_dl = generator.generate_dataloader(
        x_val=val_x,
        y_val=val_y,
        x_test=test_x,
        y_test=test_y,
        batch_size=config["training"]["batch_size"],
    )
    base_features, dcn_seq_features, rankmixer_seq_features, semantic_schema = build_feature_columns(encoded, spec)
    print(f"{dataset_name} ranking samples: train={len(train_y)}, valid={len(val_y)}, test={len(test_y)}")
    return {
        "dataset": str(config["dataset"]),
        "train_dl": train_dl,
        "val_dl": val_dl,
        "test_dl": test_dl,
        "dcn_v2_features": base_features + dcn_seq_features,
        "rankmixer_features": base_features,
        "rankmixer_sequence_features": rankmixer_seq_features,
        "semantic_schema": semantic_schema,
    }


def prepare_three_table_dataset_from_config(config, dataset_name, desc="build ranking samples"):
    ensure_source_files(config)
    return prepare_three_table_dataset(config, load_source_frame(config), dataset_name=dataset_name, desc=desc)