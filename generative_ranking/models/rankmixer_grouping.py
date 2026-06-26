from collections import OrderedDict


def normalize_rankmixer_group_schema(group_schema, default_seq_pool_modes=("mean",)):
    normalized = []
    for index, raw_group in enumerate(group_schema or []):
        if not isinstance(raw_group, dict):
            raise TypeError(f"semantic schema group at index {index} must be a dict")
        name = str(raw_group.get("name", "")).strip() or f"group_{index}"
        features = [str(item).strip() for item in raw_group.get("features", []) if str(item).strip()]
        sequence_features = [str(item).strip() for item in raw_group.get("sequence_features", []) if str(item).strip()]
        raw_pool_modes = raw_group.get("pool_modes", default_seq_pool_modes)
        pool_modes = [str(item).strip().lower() for item in raw_pool_modes if str(item).strip()]
        if not features and not sequence_features:
            raise ValueError(f"semantic schema group '{name}' must define features or sequence_features")
        if sequence_features and not pool_modes:
            raise ValueError(f"semantic schema group '{name}' has sequence_features but no pool_modes")
        normalized.append(OrderedDict([
            ("name", name),
            ("features", features),
            ("sequence_features", sequence_features),
            ("pool_modes", tuple(pool_modes)),
        ]))
    return normalized


def build_rankmixer_semantic_groups(group_schema, default_seq_pool_modes=("mean",)):
    semantic_groups = []
    normalized_schema = normalize_rankmixer_group_schema(group_schema, default_seq_pool_modes=default_seq_pool_modes)
    for raw_group in normalized_schema:
        expanded = list(raw_group["features"])
        for seq_name in raw_group["sequence_features"]:
            for pool_mode in raw_group["pool_modes"]:
                expanded.append(f"seq::{seq_name}::{str(pool_mode).lower()}")
        semantic_groups.append(OrderedDict([("name", raw_group["name"]), ("features", expanded)]))
    return semantic_groups