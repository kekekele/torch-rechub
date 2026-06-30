# DCNv2

本文介绍 `DCNv2` 在 `generative_ranking` 中的实现方式、配置含义和使用注意点。

## 模型概述

`DCNv2` 是经典的显式特征交叉模型，目标是在 embedding 展平后的 dense 向量空间里，直接学习高阶特征交互。

当前实现位于 `generative_ranking/models/dcn_v2.py`，整体结构很直接：

- 先对输入特征做 embedding，并拼接成一个扁平向量
- 再进入 cross network 学习显式交叉项
- 根据 `model_structure`，选择是否再叠加一个 MLP 分支
- 最后通过线性输出层得到 CTR 概率

## 主要模块

### `EmbeddingLayer`

- 对输入特征做 embedding
- 输出会被展平成一个统一的 dense 向量
- `self.dims` 就是所有输入特征 embedding 维度之和

### `CrossNetV2` / `CrossNetMix`

当前实现支持两种 cross network：

- `CrossNetV2`
  - 标准 DCNv2 cross network
  - 直接在全维空间学习显式交叉

- `CrossNetMix`
  - low-rank mixture 版本
  - 用低秩分解和多专家结构提升表达能力与参数效率

在代码里：

- `use_low_rank_mixture=true` 时使用 `CrossNetMix`
- `use_low_rank_mixture=false` 时使用 `CrossNetV2`

### `MLP`

根据 `model_structure`，DCNv2 还可以带一个 DNN 分支：

- `crossnet_only`
  - 仅保留 cross network 输出

- `stacked`
  - 先过 cross network
  - 再把 cross 输出送入 MLP

- `parallel`
  - cross network 和 MLP 并行
  - 最后把两路输出拼接

当前 `generative_ranking` 默认使用的是更常见的 `parallel` 思路，但 `dcn_v2.py` 本身保留了另外两种结构入口。

## 前向流程

当前实现的 forward 逻辑可以概括为：

1. `embedding(x, features, squeeze_dim=True)` 得到扁平特征向量
2. `crossnet(embed_x)` 得到显式交叉输出
3. 根据 `model_structure` 决定：
   - 直接输出 `cross_out`
   - 或送入 `stacked_dnn`
   - 或与 `parallel_dnn(embed_x)` 拼接
4. 经过 `LR(final_out)` 输出 logit
5. `sigmoid` 转成 CTR 概率

## 配置参数说明

`generative_ranking/config/movielens/train.json` 中 `dcn_v2` 默认参数含义：

- `n_cross_layers`：cross network 的层数
- `mlp_params.dims`：DNN 分支每层隐藏维度，例如 `[256, 128]`
- `mlp_params.dropout`：DNN 分支的 dropout 比例
- `mlp_params.activation`：DNN 分支使用的激活函数，例如 `relu`

虽然当前默认配置里没有显式暴露以下字段，但 `dcn_v2.py` 实现本身还支持：

- `model_structure`：可选 `crossnet_only`、`stacked`、`parallel`
- `use_low_rank_mixture`：是否使用 `CrossNetMix`
- `low_rank`：low-rank mixture 的低秩维度
- `num_experts`：low-rank mixture 中专家数量

如果后续想把这些选项开放到 CLI 或 `train.json`，需要同步修改：

- `generative_ranking/train.py`
- `generative_ranking/config/*/train.json`

## 适用场景

相对 RankMixer 和 OneTrans，DCNv2 更偏向经典表格推荐模型：

- 更适合做结构清晰的 baseline
- 对静态 sparse/dense 特征交叉很直接
- 工程复杂度低，训练和调参成本通常也更低

如果当前目标是：

- 先验证数据协议是否正确
- 先建立一个稳定 baseline
- 对比生成式 / token 化模型的收益

那么 DCNv2 往往是最合适的参照模型。

## 设计注意点

- DCNv2 当前走的是 `bundle["dcn_v2_features"]`，与 RankMixer / OneTrans 使用的特征集合不是同一路。
- 该实现默认直接把 embedding 展平后送入 cross net，没有显式序列 token 化过程。
- 如果引入新的 feature schema，需要确认所有特征的 `embed_dim` 都能正确累计到 `self.dims`。
