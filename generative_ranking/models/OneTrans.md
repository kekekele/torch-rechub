# OneTrans

本文介绍 `OneTrans` 模型关键特性和使用说明。

## 模型概述

`OneTrans` 是一种混合序列 / 非序列融合结构，核心思路是：

- 将历史序列特征编码为 `S token` 序列（sequence tokens）
- 将静态特征与多值属性编码为 `NS token`（non-sequence tokens）
- 在 Transformer-like attention 中，对 `S token` 和 `NS token` 使用混合参数化的 Q/K/V 投影
- 通过金字塔堆栈（pyramid stack）逐步缩减 query 长度，保持 full KV 上下文

在当前实现中，`generative_ranking/models/onetrans.py` 提供了完整的模型构建与前向逻辑。

## 特性说明

- `MixedCausalAttention`
  - 对 `S token` 采用共享 Q/K/V 投影
  - 对每个 `NS token` 采用独立 Q/K/V 投影
  - 支持多种 attention mask：`paper_causal` / `origin` / `hard_mask` / `bimask_soft` / `bimask_hard`

- `OneTransBlock`
  - 使用 `RMSNorm` + attention + token-specific FFN
  - 可选 activation checkpoint 节省显存

- `OneTrans` 主干
  - `sequence_features`：用于构造真正的序列 token，要求 `pooling='concat'`
  - `item_sequence_features`：视为静态多值属性，先池化后作为 non-seq 特征
  - `use_sep_token`：控制是否插入分隔 token。当前默认可以关闭以保持简单 token 顺序

## 数据与特征处理

- `sparse_features`：稀疏离散特征，直接 embedding
- `dense_features`：连续数值特征，直接拼接
- `item_sequence_features`：虽然数据类型为序列特征，但本实现不将其作为 `S token` 输入；
  它们先通过 pooling 归约为一个固定向量，再进入 `NS token` tokenizer
- `sequence_features`：真正的历史序列输入，必须使用 `concat` pooling，以保留时间步 token 序列顺序

## 模型配置示例

可以在 `generative_ranking/config/movielens/train.json` 中配置 OneTrans 的默认参数，例如：

```json
"onetrans": {
  "max_seq_len": 50,
  "ns_len": 4,
  "d_model": 128,
  "num_heads": 4,
  "ffn_hidden": 256,
  "multi_num": 4,
  "num_pyramid_layers": 6,
  "pyramid_align": 32,
  "mask_type": "paper_causal",
  "use_sep_token": false,
  "use_checkpoint": false,
  "head_dropout": 0.0
}
```

## 使用说明

训练脚本和推理脚本会自动加载配置并构建 OneTrans 模型。

- 训练：使用 `generative_ranking/train.py` 并选择 `onetrans` 模型类型
- 推理：使用 `generative_ranking/infer.py` 并选择 `onetrans` 模型类型

## 设计注意点

- `item_sequence_features` 并不是 `OneTrans` 里的 S-token 历史序列，而是“非序列多值属性”的一种表达方式。
- 只有 `sequence_features` 才会被展开成真实的 `S token` 序列。
- `use_sep_token=false` 时，模型直接拼接 `[seq_tokens, ns_tokens]`；开启后会在它们中间插入一个可学习分隔 token。
