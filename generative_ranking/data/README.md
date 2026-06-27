# generative_ranking 数据设计

该目录用于存放独立 `generative_ranking` 包所需的数据集文件。

当前内置数据集：

- MovieLens-1M 目标三表文件：`generative_ranking/data/ml-1m/user_info.csv`、`item_fea.csv`、`seq.csv`

也可以在 `python -m generative_ranking.train` 或 `python -m generative_ranking.infer` 中通过 CLI 参数覆盖 `data_dir`。

## 目标

目标数据抽象是一套统一的三表范式：

- `user_info.csv`：每行一个用户，例如 `uid, age, gender, ...`
- `item_fea.csv`：每行一个物品，例如 `iid, category, brand, ...`
- `seq.csv`：每行一条交互，例如 `uid, iid, timestamp, action, ...`

工程目标是在保持训练和推理入口不变的前提下，仅通过修改配置切换数据集。

## 可行性结论

对于 CTR 或二分类 ranking 数据集，只要满足以下三个条件，这套方案就是可行的：

1. 用户、物品、交互数据都能归一化到三表结构中。
2. 监督标签可以通过声明式的规则从交互记录中推导出来。
3. 特征工程保持在声明式范围内，例如类别编码、稠密特征直通、序列截断以及语义分组。

如果数据集需要自定义文本处理、图像特征、复杂跨表聚合、多任务监督，或者非标准的 session 重建逻辑，那么这套方案就不能做到完全“只改配置”。

## 统一数据契约

数据层应输出与当前模型已使用的同一套 bundle 结构：

```python
{
	"dataset": "...",
	"train_dl": train_dl,
	"val_dl": val_dl,
	"test_dl": test_dl,
	"dcn_v2_features": [...],
	"rankmixer_features": [...],
	"rankmixer_sequence_features": [...],
	"semantic_schema": [...],
}
```

只要某个数据集适配器能够返回这套结构，当前的 model factory、trainer 和 inference 入口就不需要再写数据集分支。

## 分层设计

这套统一范式建议拆分为四层。

### 1. 数据源层

这一层负责读取三张源表，并完成字段名归一化。

必须承担的职责：

- 读取 `user_info`、`item_fea` 和 `seq`
- 应用文件格式设置，例如 `sep`、`header`、`encoding`
- 将数据集自有列名映射为统一语义角色
- 校验用户 ID、物品 ID、时间戳等关键字段是否存在

建议的配置字段：

```python
"source": {
	"user_table": {"path": "user_info.csv", "sep": ",", "encoding": "utf-8"},
	"item_table": {"path": "item_fea.csv", "sep": ",", "encoding": "utf-8"},
	"interaction_table": {"path": "seq.csv", "sep": ",", "encoding": "utf-8"},
	"columns": {
		"user_id": "uid",
		"item_id": "iid",
		"timestamp": "timestamp",
		"action": "action",
	},
}
```

这一层的输出是三个标准化后的 DataFrame：

- `users_df`
- `items_df`
- `interactions_df`

### 2. 样本构造层

这一层负责将交互日志转换成监督式 ranking 样本。

必须承担的职责：

- 按用户和时间对交互排序
- 定义哪些行为会进入历史序列
- 定义标签如何生成
- 构造 target 字段和历史序列字段
- 将样本切分为 train、val、test

建议的配置字段：

```python
"sample_builder": {
	"max_seq_len": 50,
	"label_rule": {"type": "expr", "expr": "action in ['click', 'buy']"},
	"history_filter": {"positive_only": True},
	"target_item_col": "iid",
	"sort_by": ["uid", "timestamp"],
	"split": {"type": "global_time_ratio", "ratios": [0.8, 0.1, 0.1]},
}
```

标准化后的样本表在概念上应类似如下结构：

```text
label, timestamp,
user-side static features ...,
target item features ...,
hist_item_id, hist_category_id, hist_brand_id, ...
```

当前 MovieLens 已经切换到统一三表协议，不再依赖专用读取脚本。

### 3. 特征声明层

这一层负责把标准化样本列转换成 `SparseFeature`、`DenseFeature` 和 `SequenceFeature` 定义。

必须承担的职责：

- 声明用户侧 sparse 和 dense 特征
- 声明物品侧 sparse 和 dense 特征
- 声明序列特征及其 embedding 共享关系
- 定义 DCNv2 和 RankMixer 各自使用的 pooling 方式

建议的配置字段：

```python
"features": {
	"user_sparse": ["uid", "gender", "age"],
	"user_dense": ["user_score"],
	"item_sparse": ["iid", "cate_id", "brand_id"],
	"item_dense": ["price"],
	"sequence": [
		{
			"name": "hist_item_id",
			"source": "iid",
			"shared_with": "target_item_id",
			"rankmixer_pooling": "concat",
			"dcn_pooling": "mean",
		},
		{
			"name": "hist_cate_id",
			"source": "cate_id",
			"shared_with": "target_cate_id",
			"rankmixer_pooling": "concat",
			"dcn_pooling": "mean",
		},
	],
	"embedding_dim": 16,
	"padding_idx": 0,
}
```

这一层非常适合配置化生成，因为当前的特征类本身已经是通用的。

### 4. 语义分组层

这一层描述 RankMixer tokenization 如何将特征组织成语义 token。

必须承担的职责：

- 定义具名语义组
- 将静态特征映射到各个语义组中
- 将序列特征按 `mean`、`target` 等 pool mode 映射到语义组中

建议的配置字段：

```python
"semantic_schema": [
	{"name": "user_profile", "features": ["uid", "gender", "age"]},
	{"name": "target_item", "features": ["target_item_id", "target_cate_id", "target_brand_id"]},
	{"name": "sequence_global", "sequence_features": ["hist_item_id", "hist_cate_id"], "pool_modes": ["mean"]},
	{"name": "sequence_target", "sequence_features": ["hist_item_id", "hist_cate_id"], "pool_modes": ["target"]},
]
```

这部分已经非常接近纯配置驱动，当前可通过 `normalize_rankmixer_group_schema()` 和 `build_rankmixer_semantic_groups()` 直接使用。

## 建议的统一配置形态

数据集部分可以统一标准化为一个配置对象：

```python
DATASET_CONFIG = {
	"dataset": "custom_three_table",
	"data_dir": "./generative_ranking/data/custom_dataset/",
	"source": {...},
	"sample_builder": {...},
	"features": {...},
	"semantic_schema": [...],
}
```

训练和模型部分则可以保持当前形态不变：

- `training`
- `dcn_v2`
- `rankmixer`

这意味着完整的新数据集接入流程可以收敛为：

1. Add raw files under a new data directory.
2. Add one new dataset config module.
3. Point `python -m generative_ranking.train --config ...` to that config.
4. Reuse the same inference entrypoint with the same config.

## 哪些部分必须泛化

如果要做到“只改配置切数据集”，以下逻辑必须从当前 MovieLens 专用实现中抽成通用能力：

- raw file reading
- key and column mapping
- label rule definition
- sequence history construction
- split strategy
- feature column generation
- semantic schema expansion from config

这些步骤应尽量收敛到统一的数据配置协议和通用数据构建器中，而不是回退到数据集专用脚本。

## 哪些部分可以继续留在模型层

以下行为应继续保留在模型层，而不是下沉到数据层：

- DCNv2 using sequence pooling such as `mean`
- RankMixer using sequence tokenization with `concat` plus semantic grouping
- MoE, token mixing, and encoder structure

数据层只需要暴露足够的元信息，供 model factory 一致地构造这些变体即可。

## 适用边界

这套统一范式最适合以下场景：

- CTR or binary ranking
- user-item interaction logs
- static user and item side features
- sequence features derived directly from interactions

但它本身不足以覆盖以下场景：

- multi-task datasets with multiple labels per sample
- listwise ranking pipelines
- datasets that require negative sampling coupled to training-time logic
- session graph construction or non-tabular modalities

对于这些场景，框架应允许插入自定义 sample builder hook，同时仍然复用统一的特征声明和语义分组 schema。

## 建议的落地路径

风险最低的迁移路径是：

1. Define a standardized dataset config protocol for the four layers above.
2. Refactor MovieLens into the first implementation of that protocol.
3. Introduce a generic three-table dataset builder under `generative_ranking/data/`.
4. Keep `prepare_dataset()` returning the same bundle structure used today.
5. Validate training and inference with MovieLens before onboarding new datasets.

## 总结

结论是肯定的：统一范式是可行的，而且当前代码在模型层和语义分组层已经比较接近目标形态。

真正缺失的抽象在数据层。一旦样本构造、特征声明和语义分组都由同一套数据集配置协议驱动，那么对于标准三表 ranking 数据，切换数据集就可以收敛为配置修改，而不再需要额外改代码。