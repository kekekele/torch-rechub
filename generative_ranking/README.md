# generative_ranking

## 1. 综述

`generative_ranking` 是一个面向 CTR / 二分类排序任务的工程，提供统一的数据协议、训练入口和推理入口，用于快速验证和比较不同 ranking 模型。

当前工程主要支持：

- `DCNv2`
- `RankMixer`

## 2. 代码目录说明

```text
generative_ranking/
├─ basic/
├─ config/
├─ data/
├─ models/
├─ util/
├─ train.py
├─ infer.py
└─ __init__.py
```

各目录和文件的作用如下：

- `basic/`
  存放基础网络组件和通用层实现，例如特征定义、EmbeddingLayer、LayerNorm、Token Mixing、Per-token FFN、Sparse MoE、回调和损失函数等。

- `config/`
  存放目录化配置。每个数据集通常对应一个目录，目录下推荐包含：
  - `data.json`：数据读取、样本构造、特征声明、语义分组
  - `train.json`：训练默认参数和模型训练配置
  - `infer.json`：推理默认参数

- `data/`
  存放通用数据构建逻辑。
  其中：
  - `dataset.py`：统一三表协议下的数据读取、merge、样本构造、特征生成
  - `registry.py`：根据配置决定如何准备数据集
  - `README.md`：数据协议与 `data.json` 字段详细说明

- `models/`
  存放模型定义与模型工厂。
  当前主要包括 `dcn_v2.py`、`rankmixer.py`、`factory.py`、`rankmixer_grouping.py`。

- `util/`
  存放通用工具模块，例如配置加载、DataLoader 构建、padding、运行时辅助函数等。

- `train.py`
  训练入口，负责加载配置、准备数据、构建模型、训练、保存 checkpoint，并在测试集上输出指标。

- `infer.py`
  推理入口，负责加载配置、重建模型、恢复 checkpoint、在指定数据切分上做评估，并可选导出预测结果。

## 3. 数据如何构建

### 3.1 目标数据格式

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

### 3.2 最小字段要求

这套工程不强制写死列名，但你提供的 CSV 字段必须与 `data.json` 配置一致。

通常需要至少具备以下几类信息：

- 用户主键，例如 `uid`
- 物品主键，例如 `iid`
- 时间字段，例如 `timestamp`
- 标签字段，例如 `label`
- 用户侧静态特征
- 物品侧静态特征
- 用于构造历史序列的交互字段

### 3.3 数据构建流程

当前数据构建逻辑由 `data/dataset.py` 负责，整体流程如下：

1. 读取 `source.tables` 中定义的原始表
2. 按 `joins` 规则把用户表、物品表和交互表 merge 成统一交互表
3. 应用字段裁剪与变换规则
4. 根据 `sample_builder` 构造监督样本和历史序列
5. 按时间顺序切分 `train / val / test`
6. 根据 `features` 生成模型输入特征定义
7. 根据 `semantic_schema` 生成 RankMixer 的语义分组

### 3.4 数据构建时需要注意什么

1. CSV 列名必须与配置一致。
   如果列名不同，需要同步修改 `data.json` 里的 `source`、`sample_builder`、`features`。

2. join key 必须可对齐。
   用户表、物品表和交互表之间的关联字段必须类型一致、语义一致。

3. `label` 和 `timestamp` 必须可用。
   当前样本构造依赖：
   - `label` 生成监督目标
   - `timestamp` 进行时间排序和数据切分

4. `select_columns` 建议显式写出。
   不写时默认保留该阶段所有字段，虽然不会直接报错，但容易带入冗余列或造成字段冲突。

5. 历史序列规则会直接影响训练样本。
   当前默认逻辑通常是：
   - 先按用户和时间排序
   - 有历史时才生成监督样本
   - 只让正样本进入历史序列

### 3.5 data.json 在哪里定义

每个数据集目录下的 `data.json` 负责定义数据协议，例如：

```text
config/<dataset_name>/data.json
```

它主要描述：

- 三张表如何读取
- 哪些表需要 join
- 样本如何构造
- 哪些列进入模型
- RankMixer 如何组织语义分组

更详细的 `data.json` 字段说明，请参考：

- `data/README.md`

## 4. 如何训练

### 4.1 基本命令

训练入口：

```bash
python -m generative_ranking.train --config <dataset_name>
```

例如，如果你已有对应的配置目录：

```bash
python -m generative_ranking.train --config movielens
```

如果只训练 RankMixer：

```bash
python -m generative_ranking.train --config movielens --model_name rankmixer
```

如果只训练 DCNv2：

```bash
python -m generative_ranking.train --config movielens --model_name dcn_v2
```

### 4.2 训练会发生什么

训练流程大致如下：

1. 加载 `data.json + train.json`
2. 准备 `train / val / test` dataloader
3. 构建指定模型
4. 在验证集上执行 early stop
5. 保存 checkpoint 到：

```text
<save_dir>/<model_name>/model.pth
```

6. 在测试集上输出 AUC

### 4.3 常用训练参数

训练参数既可以写在 `train.json` 中，也可以通过 CLI 覆盖。例如：

```bash
python -m generative_ranking.train --config movielens --model_name rankmixer --epoch 10 --batch_size 512 --learning_rate 1e-4
```

当前常用参数包括：

- `--config`：指定配置目录名或配置路径。
- `--model_name`：选择训练 `dcn_v2`、`rankmixer` 或 `all`。
- `--data_dir`：覆盖数据集根目录。
- `--device`：指定训练设备，例如 `cpu`、`cuda:0`。
- `--save_dir`：指定 checkpoint 和输出结果保存目录。
- `--seed`：指定随机种子，保证实验可复现。
- `--epoch`：指定训练轮数。
- `--learning_rate`：指定 Adam 优化器学习率。
- `--batch_size`：指定训练、验证、测试的批大小。
- `--weight_decay`：指定 Adam 权重衰减系数。
- `--max_seq_len`：指定历史序列最大长度。

RankMixer 还支持一组结构和正则参数，例如：

- `--rankmixer_d_model`：指定 RankMixer 的隐藏维度。
- `--rankmixer_num_layers`：指定 RankMixer 编码层数。
- `--rankmixer_num_tokens`：指定语义 token 数量。
- `--rankmixer_use_moe`：启用 MoE 前馈层。
- `--rankmixer_no_moe`：关闭 MoE，改用普通 FFN。
- `--rankmixer_moe_experts`：指定 MoE 专家数。
- `--rankmixer_moe_l1_coef`：指定 MoE 路由 L1 正则系数。
- `--rankmixer_moe_sparsity_ratio`：指定 MoE 稀疏性约束比例。
- `--rankmixer_moe_routing_type`：指定 MoE 路由类型。
- `--rankmixer_input_dropout`：指定输入层 dropout。
- `--rankmixer_token_mixing_dropout`：指定 token mixing 部分 dropout。
- `--rankmixer_ffn_dropout`：指定 FFN 或 MoE 专家内部 dropout。
- `--rankmixer_head_dropout`：指定预测头 dropout。

### 4.4 训练时的注意事项

1. 如果只是验证流程，建议先单独跑一个模型，不要一开始就使用 `model_name=all`。

2. `device`、`batch_size`、`max_seq_len` 会直接影响训练速度与显存占用。

3. 如果数据构造阶段报错，优先检查：
   - 文件路径
   - 列名
   - `label`
   - join key

4. 如果模型构建或训练报错，优先检查：
   - `features`
   - `semantic_schema`
   - RankMixer 结构参数是否与预期一致

## 5. 推理、指标与注意事项

### 5.1 基本命令

推理入口：

```bash
python -m generative_ranking.infer --config <dataset_name> --model_name rankmixer
```

例如：

```bash
python -m generative_ranking.infer --config movielens --model_name rankmixer
```

如果不显式传 `--checkpoint`，默认会从下面路径加载：

```text
<save_dir>/<model_name>/model.pth
```

### 5.2 导出预测结果

如果希望把预测结果写成 CSV：

```bash
python -m generative_ranking.infer --config movielens --model_name rankmixer --output_path ./outputs/rankmixer_test_predictions.csv
```

当前输出 CSV 格式为：

```csv
index,prediction
0,0.8123
1,0.1034
...
```

### 5.3 指标如何计算

当前训练和推理阶段的主要指标是 AUC。

具体行为是：

- 训练结束后，`train.py` 在测试集上计算一次 AUC
- 推理阶段，`infer.py` 在 `--split` 指定的数据切分上计算一次 AUC

默认 `--split=test`，即通常在测试集上做最终评估。

### 5.4 推理时需要注意什么

1. 推理时会读取：
   - `data.json`
   - `train.json`
   - `infer.json`

   这是因为推理阶段不仅需要数据配置，还需要训练时的模型结构参数来重建模型。

2. 训练和推理时的模型结构必须一致。
   尤其对于 RankMixer，需要保证以下内容一致：
   - `d_model`
   - `num_layers`
   - `num_tokens`
   - `seq_pool_modes`
   - `semantic_schema`

3. 训练和推理的数据字段语义也必须一致。
   特别是：
   - user/item key
   - `label`
   - item 侧关键特征
   - 历史序列构造方式

4. 如果 checkpoint 加载报错，优先检查：
   - 当前配置是否与训练时一致
   - `semantic_schema` 是否被修改过
   - `features` 中的 `name/source` 映射是否变化

## 6. 配置文件说明

当前每个数据集目录推荐包含三份配置：

- `data.json`
  定义数据协议：读取、merge、样本构造、特征、语义分组。

- `train.json`
  定义训练默认参数和模型结构默认值。

- `infer.json`
  定义推理默认参数，例如默认 `model_name`、默认 `split`、默认 `output_path`。

当前配置加载规则是：

- 训练：`data.json + train.json`
- 推理：`data.json + train.json + infer.json`

## 7. 环境与依赖

这套工程默认依赖至少包括：

- Python 3
- PyTorch
- numpy
- pandas
- scikit-learn
- tqdm

如果环境缺少这些依赖，训练、推理或数据准备都可能失败。

## 8. 常见问题

### 8.1 为什么训练跑不起来

优先检查：

- 数据文件是否存在
- `data_dir` 是否正确
- `data.json` 中列名是否与 CSV 一致
- `seq.csv` 是否已经带 `label`

### 8.2 为什么推理加载 checkpoint 失败

优先检查：

- 是否加载了与训练时同结构的配置
- 是否改过 RankMixer 结构参数
- 是否改过特征声明或语义分组

### 8.3 为什么指标与预期不一致

优先检查：

- `label` 是否按预期生成
- 关键 item 特征是否正确
- 历史序列是否按预期构造
- 切分方式是否仍与实验设定一致

## 9. 建议使用方式

如果你准备接入一个新数据集，建议按下面顺序做：

1. 先把原始数据转换为 `user_info.csv / item_fea.csv / seq.csv` 三表格式。
2. 先只写 `data.json`，验证数据读取、join 和样本构造是否正确。
3. 再补 `train.json` 与 `infer.json`。
4. 先单独训练 `rankmixer` 或 `dcn_v2`，不要一开始就用 `model_name=all`。
5. 训练通过后再跑推理，检查 AUC 和预测导出是否正常。
