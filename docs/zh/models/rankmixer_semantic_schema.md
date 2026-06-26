---
title: RankMixer 语义分组范式
description: RankMixer semantic schema 的标准结构、字段约束与数据集接入方式
---

# RankMixer 语义分组范式

RankMixer 的语义 Tokenization 需要先将输入特征组织成一组稳定的语义组，再将每个语义组映射为一个 token。

为了让模型实现与具体数据集解耦，Torch-RecHub 中推荐将语义分组定义为一份数据层 schema，再由统一 helper 将 schema 展开为模型实际使用的 `semantic_groups`。

当前推荐使用：

```python
from torch_rechub.models.ranking import (
    SCHEMA_TEMPLATE,
    build_rankmixer_semantic_groups,
    normalize_rankmixer_group_schema,
)
```

## 1. 标准结构

每个语义组都使用一个 dict 表示，支持以下字段：

| 字段 | 是否必须 | 含义 |
| --- | --- | --- |
| `name` | 是 | 语义组名称，用于标识组含义 |
| `features` | 否 | 显式稀疏/稠密/目标特征名称列表 |
| `sequence_features` | 否 | 序列特征基名列表，例如 `hist_item_id` |
| `pool_modes` | 当存在 `sequence_features` 时必须 | 序列池化方式，例如 `("mean",)`、`("target",)` |

约束：

1. 每个 group 至少要提供 `features` 或 `sequence_features` 中的一种。
2. `sequence_features` 不直接写成 `seq::hist_item_id::mean` 这种最终 token 名。
3. `pool_modes` 只描述语义意图，真正的 token 名由 helper 自动展开。

## 2. 推荐模板

```python
semantic_schema = normalize_rankmixer_group_schema([
    {
        "name": "user_profile",
        "features": ["user_id", "gender", "age", "occupation"],
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
])
```

展开为模型使用的 semantic groups：

```python
semantic_groups = build_rankmixer_semantic_groups(
    semantic_schema,
    default_seq_pool_modes=("mean", "target"),
)
```

## 3. 数据集解耦原则

接入一个新数据集时，建议只改数据层，不改 RankMixer 主体实现：

1. 适配原始数据读取与预处理。
2. 构建 `SparseFeature`、`DenseFeature`、`SequenceFeature`。
3. 根据业务语义定义 `semantic_schema`。
4. 调用 `build_rankmixer_semantic_groups(...)` 生成模型输入。

这样模型层只关心“用户组 / 目标组 / 序列组”这类语义结构，不关心具体字段名来自 MovieLens、广告、搜索还是电商场景。

## 4. 设计建议

推荐优先按以下原则划分语义组：

1. 用户静态画像单独成组，例如用户 id、性别、年龄、地域。
2. 目标 item 与目标上下文特征单独成组，例如 item id、类目、品牌。
3. 序列长期偏好单独成组，例如历史 item 的 mean pooling。
4. 序列当前意图单独成组，例如历史序列的 target/last pooling。
5. 不同语义明显冲突的特征不要混到同一个组中。

## 5. 当前实现边界

当前 RankMixer 的语义分组范式已经实现了“schema -> semantic groups”的统一流程，但仍有两个边界需要注意：

1. 如果不显式传 `semantic_groups`，Tokenizer 仍会退回到“排序后均分切块”的 fallback 路径。
2. 当前范式强调数据集无关的语义组定义，但具体的组划分策略仍需要结合业务语义人工设计。