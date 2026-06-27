# generative_ranking 数据协议

该目录文档用于说明 `generative_ranking` 当前采用的最终数据协议。

目标是让不同数据集在满足统一三表格式和统一配置契约的前提下，只通过配置即可接入训练与推理流程，而不再依赖数据集专用读取脚本。

## 1. 数据目标格式

当前推荐使用统一三表格式：

- `user_info.csv`
- `item_fea.csv`
- `seq.csv`

概念上分别对应：

- 用户表：每行一个用户及其静态特征
- 物品表：每行一个物品及其静态特征
- 交互表：每行一条用户与物品的行为记录

一个典型的数据目录结构如下：

```text
<your_data_dir>/
  user_info.csv
  item_fea.csv
  seq.csv
```

这三张表不要求固定列名，但要求列名、join key、标签字段和时间字段能够通过 `data.json` 正确描述。

## 2. 数据层输出契约

数据层最终需要输出下面这类 bundle 结构，供训练和推理入口复用：

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

只要某个数据集最终能产出这套结构，模型构建、训练和推理流程就不需要再写数据集特判。

## 3. 数据构建流程

当前数据构建逻辑由 `dataset.py` 负责，整体流程如下：

1. 读取 `source.tables` 中定义的原始表。
2. 按 `source.joins` 规则将用户表、物品表与交互表 merge 成统一交互表。
3. 应用单表或 merge 后的字段变换规则。
4. 根据 `sample_builder` 构造监督样本和历史序列。
5. 按时间顺序切分 `train / val / test`。
6. 根据 `features` 生成模型输入特征定义。
7. 根据 `semantic_schema` 生成 RankMixer 语义分组。

## 4. data.json 结构说明

每个数据集目录下推荐至少提供一份 `data.json`，用于描述数据协议。

典型位置：

```text
config/<dataset_name>/data.json
```

当前 `data.json` 主要包含以下部分：

- `dataset`
- `dataset_display_name`
- `data_dir`
- `source`
- `dataset_params`
- `sample_builder`
- `features`
- `semantic_schema`

### 4.1 dataset

- `dataset`：数据集标识，用于内部区分数据集。
- `dataset_display_name`：日志展示名称，可选。
- `data_dir`：数据根目录，`source.tables` 中的 `path` 会相对它进行解析。

### 4.2 source

`source` 用来描述原始表如何读取、如何 merge 成统一交互表。

当前支持：

- `base_table`：主表名，通常是交互表，例如 `interactions`
- `tables`：原始表读取配置
- `joins`：表关联规则
- `post_merge_transforms`：merge 后字段变换规则，可选
- `select_columns`：最终统一交互表保留字段，可选

#### source.tables

`source.tables` 是一个字典，每个键对应一张原始表，例如：

- `interactions`
- `users`
- `items`

每张表当前支持：

- `path`：文件路径，相对 `data_dir`
- `read_csv`：传给 `pandas.read_csv` 的参数
- `transforms`：单表读取后的字段变换规则，可选
- `select_columns`：该表保留字段，可选

例如：

```json
"interactions": {
  "path": "seq.csv",
  "read_csv": {
    "sep": ",",
    "header": 0,
    "encoding": "utf-8"
  },
  "select_columns": ["uid", "iid", "timestamp", "label"]
}
```

#### source.joins

`joins` 用来描述主表与其他表如何关联。每个 join 当前支持：

- `right_table`：右表名
- `left_on`：左表 join 键
- `right_on`：右表 join 键
- `how`：join 方式，通常为 `left`
- `select_columns`：该次 join 的右表保留字段，可选

#### source.transforms

当前代码支持以下字段变换类型：

- `split_first`：按分隔符拆分字符串并取第一段
- `binary_compare`：按比较规则生成二分类标签
- `astype`：类型转换

这些变换既可以用于单表读取后，也可以用于 merge 后。

#### select_columns 的行为

这是配置里最容易误解的字段之一。

当前实现中：

- 如果 `tables.<table_name>.select_columns` 不写，则该表读取后的所有字段都会保留
- 如果 `source.select_columns` 不写，则 merge 和变换后的所有字段都会保留

也就是说，`select_columns` 是字段裁剪规则，不是字段声明规则。

建议实践：

- 在单表层尽量写 `tables.<table_name>.select_columns`
- 在整表层尽量写 `source.select_columns`

这样做的好处是：

- 避免无关字段进入后续 merge
- 降低字段重名和冲突风险
- 让配置意图更清晰，排查问题更直接

### 4.3 dataset_params

当前主要使用：

- `max_seq_len`：历史序列最大长度

如果 `sample_builder.max_seq_len` 未显式给出，则会回退使用这里的值。

### 4.4 sample_builder

`sample_builder` 控制如何从统一交互表生成监督样本。

当前支持：

- `user_id_col`：用户列名
- `timestamp_col`：时间列名
- `label_col`：标签列名
- `max_seq_len`：历史序列最大长度
- `history_filter.positive_only`：是否仅让正样本进入历史序列
- `split.type`：当前仅支持 `global_time_ratio`
- `split.ratios`：例如 `[0.8, 0.1, 0.1]`

当前默认样本构造逻辑是：

- 先按用户和时间排序
- 对每个用户逐条构造样本
- 只有在已有历史时才生成监督样本
- 如果 `positive_only=true`，则只有正样本行为进入历史序列
- 所有样本构造完后，再按时间全局切分 `train / val / test`

### 4.5 features

`features` 用来声明哪些列会变成模型输入。

当前支持：

- `user_sparse`
- `user_dense`
- `item_sparse`
- `item_dense`
- `sequence`
- `embedding_dim`
- `padding_idx`

#### user_sparse / user_dense / item_sparse / item_dense

这些字段可以写成字符串，也可以写成对象。

字符串写法：

```json
"user_sparse": ["uid", "age"]
```

等价于：

```json
"user_sparse": [
  {"name": "uid", "source": "uid"},
  {"name": "age", "source": "age"}
]
```

对象写法适用于模型输入名和原始列名不同的场景，例如：

```json
{
  "name": "target_item_id",
  "source": "iid"
}
```

#### sequence

`sequence` 描述哪些列需要构造成历史序列特征。

当前每个 sequence 项支持：

- `name`：模型输入中的序列名
- `source`：历史来自哪一列
- `shared_with`：与哪个 target 特征共享 embedding
- `rankmixer_pooling`：RankMixer 的 sequence pooling，通常为 `concat`
- `dcn_pooling`：DCNv2 的 sequence pooling，通常为 `mean`

### 4.6 semantic_schema

`semantic_schema` 用于描述 RankMixer 的语义分组。

每个语义组当前支持：

- `name`：语义组名称
- `features`：静态特征名列表
- `sequence_features`：序列特征名列表
- `pool_modes`：例如 `mean`、`target`

这里写的是模型输入名，而不是原始 CSV 列名。

## 5. 通用约束与注意事项

### 5.1 字段一致性

以下字段在数据构建中最关键：

- 用户主键
- 物品主键
- 时间字段
- 标签字段
- 用于构造历史序列的交互字段

这些字段的列名和语义必须与 `data.json` 保持一致。

### 5.2 join key 一致性

所有 join key 必须满足：

- 类型一致
- 编码方式一致
- 语义一致

如果用户表、物品表、交互表中的 key 无法对齐，后续 merge 会直接影响样本完整性。

### 5.3 标签与时间字段

- `label` 决定监督目标
- `timestamp` 决定排序和数据切分

如果这两个字段缺失或语义错误，训练和评估结果都会不可信。

### 5.4 序列构造规则

样本构造逻辑会直接影响模型输入与最终指标，尤其是：

- 历史是否只保留正样本
- 历史序列最大长度是多少
- 样本切分是否按时间进行

这些都应明确写在 `sample_builder` 中，而不是依赖隐式约定。

### 5.5 配置复用边界

虽然这套协议支持较强的配置驱动，但并不是所有数据集都能做到完全“只改配置”。

以下场景通常仍需要扩展：

- 多任务标签
- listwise 排序
- 复杂负采样逻辑
- 图像、文本等非结构化特征
- 自定义 session 重建逻辑

当前协议最适合：

- CTR 或二分类 ranking
- 用户-物品交互日志
- 静态用户 / 物品特征
- 可直接从交互中构造历史序列的任务

## 6. 示例说明

工程中可以提供某个数据集作为配置示例，但示例数据集不是数据协议本身的一部分。

也就是说：

- 数据协议是通用的
- 示例配置只是帮助理解 `data.json` 应该怎么写

如果你接入新的数据集，应该以当前协议为准，而不是以某个示例数据集的字段命名为准。

## 7. 建议使用方式

建议按下面顺序接入新数据集：

1. 先把原始数据整理成 `user_info.csv / item_fea.csv / seq.csv` 三表格式。
2. 先编写 `data.json`，验证读取、join 和样本构造逻辑。
3. 再补充 `train.json` 和 `infer.json`。
4. 先单独训练一个模型，验证训练流程是否正确。
5. 再执行推理与指标检查，确认训练和推理使用的是同一套数据语义。
