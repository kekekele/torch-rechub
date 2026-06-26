from collections import OrderedDict


SCHEMA_TEMPLATE = [
    {
        "name": "user_profile",
        "features": ["user_id", "gender"],
    },
    {
        "name": "target_item",
        "features": ["target_item_id", "target_cate_id"],
    },
    {
        "name": "sequence_global",
        "sequence_features": ["hist_item_id", "hist_cate_id"],
        "pool_modes": ("mean",),
    },
    {
        "name": "sequence_target",
        "sequence_features": ["hist_item_id", "hist_cate_id"],
        "pool_modes": ("target",),
    },
]


def normalize_rankmixer_group_schema(group_schema, default_seq_pool_modes=("mean",)):
    """Normalize and validate dataset-level semantic group schema.

    Standard schema item format
    ---------------------------
    Each group must be a dict with the following keys:

    - ``name``: required semantic group name.
    - ``features``: optional explicit sparse/dense/target feature names.
    - ``sequence_features``: optional base sequence feature names such as
      ``hist_item_id``.
    - ``pool_modes``: optional sequence pooling modes for this group.

    At least one of ``features`` or ``sequence_features`` must be non-empty.
    """
    normalized = []
    for index, raw_group in enumerate(group_schema or []):
        if not isinstance(raw_group, dict):
            raise TypeError(f"semantic schema group at index {index} must be a dict")
        name = str(raw_group.get("name", "")).strip() or f"group_{index}"
        features = [str(item).strip() for item in raw_group.get("features", []) if str(item).strip()]
        sequence_features = [
            str(item).strip() for item in raw_group.get("sequence_features", []) if str(item).strip()
        ]
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
    """Expand a dataset-level semantic schema into RankMixer semantic groups.

    Parameters
    ----------
    group_schema : Sequence[dict]
        Ordered semantic group definitions. Each item supports:
        - ``name``: group name.
        - ``features``: explicit sparse/dense/target feature names.
        - ``sequence_features``: base sequence feature names such as ``hist_item_id``.
        - ``pool_modes``: optional pool modes for the sequence features in this group.
    default_seq_pool_modes : tuple[str], default=("mean",)
        Used when a group contains ``sequence_features`` but does not declare
        ``pool_modes`` explicitly.

    Returns
    -------
    list[dict]
        Semantic groups ready for ``RankMixer(..., semantic_groups=...)``.
    """
    semantic_groups = []
    normalized_schema = normalize_rankmixer_group_schema(
        group_schema,
        default_seq_pool_modes=default_seq_pool_modes,
    )
    for raw_group in normalized_schema:
        group_name = raw_group["name"]
        feature_names = list(raw_group["features"])
        seq_feature_names = list(raw_group["sequence_features"])
        pool_modes = raw_group["pool_modes"]
        expanded = list(feature_names)
        for seq_name in seq_feature_names:
            for pool_mode in pool_modes:
                expanded.append(f"seq::{seq_name}::{str(pool_mode).lower()}")
        semantic_groups.append(OrderedDict([
            ("name", group_name),
            ("features", expanded),
        ]))
    return semantic_groups