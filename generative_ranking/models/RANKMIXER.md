# RankMixer

本文介绍 `RankMixer` 模型关键特性和使用说明。

## 模型概述

`RankMixer` 的核心思路是：

- 先把稀疏特征、稠密特征、上下文多值特征以及历史序列摘要统一映射到同一语义空间
- 再通过 `SemanticTokenizer` 将这些 feature-level 表示压缩成固定数量的语义 token
- 使用参数自由的 token mixing 进行跨 token 交互
- 使用 per-token FFN 或 sparse MoE 进行 token 内变换
- 最后做 token pooling 并输出 CTR 预测

它不是标准 self-attention Transformer，而是更强调“固定 token 预算下的语义聚合 + 参数自由 mixing”。

## 主要模块

### `RankMixerBlock`

每个 block 包含两部分：

- `ParameterFreeTokenMixer`
  - 做 token 间混合
  - 不使用标准 attention 参数矩阵
  - 主要承担跨 token 信息交换

- `PerTokenFFN` 或 `PerTokenSparseMoE`
  - `use_moe=false` 时使用普通 per-token FFN
  - `use_moe=true` 时使用 sparse mixture-of-experts

同时支持：

- `ln_style="pre"` 或 `ln_style="post"`
- `token_mixing_dropout`
- `ffn_dropout`

### `RankMixerEncoder`

- 将多个 `RankMixerBlock` 叠起来
- 汇总每层的 `moe_loss`
- 可选最终 `LayerNorm`

### `RankMixer`

主模型负责：

- 构建 feature embedding
- 对不同类型特征做 pooling / projection
- 调用 `SemanticTokenizer` 生成基础 token
- 根据配置追加 sequence summary token 或 CLS token
- 经过 encoder 后做 pooling 和 CTR head 输出

## 特征处理逻辑

### 非序列与上下文特征

以下特征直接或池化后进入 feature map：

- `SparseFeature`
- `DenseFeature`
- `context_sequence_features`

其中 `context_sequence_features` 是出现在 `features` 里的 `SequenceFeature`。它们并不被当作历史 token 序列，而是先做 pooling，再作为上下文特征参与 tokenization。

### 历史序列特征

`sequence_features` 才是真正的用户历史序列特征。

要求：

- 必须使用 `pooling='concat'`
- 先取出逐步 embedding
- 再按照 `seq_pool_modes` 做摘要，例如 `mean` / `max` / `target`

这些摘要有两种进入模型的方式：

- `include_seq_in_tokenization=true`
  - 序列摘要与其他特征一起进入 `SemanticTokenizer`
- `include_seq_in_tokenization=false`
  - 序列摘要先独立投影，再作为额外 token 追加到 tokenizer 输出后面

## Token 组织方式

最终 token 数量由三部分组成：

- `num_tokens`：语义 tokenizer 产出的基础 token 数
- 额外 sequence token：仅在 `include_seq_in_tokenization=false` 时追加
- `CLS token`：仅在 `add_cls_token=true` 时追加

因此 `num_heads` 必须和最终 token 总数一致，这是当前 `RankMixer` 实现里的约束。

## MoE 技术点

当 `use_moe=true` 时，FFN 被 `PerTokenSparseMoE` 替换。相关参数含义如下：

- `moe_experts`：专家数量
- `moe_l1_coef`：routing 稀疏正则系数
- `moe_sparsity_ratio`：目标稀疏度
- `moe_use_dtsi`：是否启用 DTSI 风格约束
- `moe_routing_type`：路由函数类型，例如 `relu_dtsi`

训练时模型可返回 `(y_pred, moe_loss)`，上层 trainer 再把 `moe_loss` 合并到总损失。

## 配置参数说明

`generative_ranking/config/movielens/train.json` 中 `rankmixer` 默认参数含义：

- `d_model`：token 隐层维度
- `num_layers`：encoder block 层数
- `num_tokens`：semantic tokenizer 输出的基础 token 数
- `seq_pool_modes`：序列摘要方式列表，例如 `["mean", "target"]`
- `use_moe`：是否启用 sparse MoE FFN
- `moe_experts`：专家个数
- `moe_l1_coef`：MoE routing 的 L1 正则系数
- `moe_sparsity_ratio`：MoE 稀疏目标比例
- `moe_use_dtsi`：是否启用 DTSI 风格 routing 约束
- `moe_routing_type`：routing 形式
- `input_dropout`：encoder 前 dropout
- `token_mixing_dropout`：token mixing 内 dropout
- `ffn_dropout`：FFN / MoE 内 dropout
- `head_dropout`：输出头 dropout

## 设计注意点

- `context_sequence_features` 和 `sequence_features` 的语义不同，前者是上下文多值属性，后者才是用户历史序列。
- `RankMixer` 当前实现依赖统一维度投影和固定 token 预算，因此更像“feature-to-token compression”架构，而不是原始逐步序列 attention 架构。
- 若修改 token 组织方式，需要同步检查 `num_heads == token_count` 这一实现约束。
