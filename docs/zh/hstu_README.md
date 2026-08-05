# HSTU Vector Retrieval

HSTU 是 `generative_retrieval/vector_retrieval` 下的序列向量召回方法。它从 RecKit 统一原始数据出发，在方法目录内生成训练、验证和推理所需的中间格式。

特征 schema 由 `data_format.csv` 驱动，转换脚本会把最终 schema 写入 `meta.json`；训练和推理都只读取这个 schema。当前实现支持 query-context 条件召回：历史事件保留各自发生时的上下文，预测位置额外接收下一步/当前请求的上下文，用于学习 $P(\mathrm{next\_item} \mid \mathrm{history}, \mathrm{query\_context})$。

## 端到端流程图

下面给出一份更适合论文、技术报告和方案文档的精简框图文案。整体可归纳为五层：统一数据接口层、数据构建层、特征编码层、条件序列建模层、训练优化与召回输出层。重点突出三条关键机制：面向大规模样本的索引化表示、FiLM 用户条件调制、绝对时间周期特征与相对时间偏置的协同建模。

### 1. 论文/技术报告风格草图

```text
[统一原始数据: seq / user / item / data_format]
        |
        v
[字段选择与 schema 生成]
        |
        v
[索引映射与大数据构建]
(uid/iid 连续化 + 离散特征 fid 映射 + JSONL/offsets)
        |
        v
[事件序列构造与样本写出]
(user token + item event + context, train/valid/predict)
        |
        v
[Dataset 按 offset 读取并按 schema 打包特征]
        |
  +---------+---------+---------+---------+
  |                   |                   |
  v                   v                   v
[用户特征]           [物品特征]           [上下文/时间特征]
user id + feat       item id + feat       query context + periodic time
  |                   |                   |
  v                   v                   v
[userdnn]            [itemdnn]          [contextdnn + time_feat_proj]
  |                   |                   |
  +---------+---------+---------+---------+
        |
        v
[FiLM 用户条件调制]
cond(u) -> gamma, beta
seq = item_repr * (1 + gamma) + beta
        |
        v
[序列融合]
item repr + query context repr + time repr + position repr
        |
        v
[多层 HSTU]
causal self-attention + RAB(|ts_i-ts_j| -> bucket -> QK^T)
        |
        v
[序列表征输出]
        |
     +------+------+
     |             |
     v             v
[训练]               [推理]
next-item target     candidate retrieval with query context
CDN sampled softmax  Top-K recall
logQ correction
```

### 2. 层次结构与模块职责

1. `统一数据接口层`
  以 `seq / user / item / data_format` 作为统一原始输入接口，屏蔽不同业务数据源的原始格式差异，为后续转换、训练和推理提供稳定的数据契约。
2. `数据构建层`
  由三部分组成：`字段选择与 schema 生成`、`索引映射与大规模表示构建`、`条件事件序列构造与样本生成`。
  这一层的核心特点是：
  - 通过 `feature_columns + data_format.csv` 实现灵活的特征裁剪与 schema 显式化。
  - 通过 `uid/iid` 连续化、离散特征 `fid` 映射、`jsonl + offsets` 形成面向大规模样本的紧凑表示与随机访问能力。
  - 通过按时间排序、注入 user token、融合 item 特征与 event context，将表结构样本转化为模型可消费的条件事件序列。
3. `特征编码层`
  将输入拆为用户特征、物品特征、上下文特征和时间特征四路，并分别映射到统一表示空间：`userdnn` 负责用户条件表征，`itemdnn` 负责物品表示融合，`contextdnn` 负责 query/event context 编码，`time_feat_proj` 负责绝对时间周期特征投影。
4. `条件序列建模层`
  首先利用 FiLM 机制由用户表示生成 `gamma/beta`，对物品序列表示进行条件调制；随后将 item、context、time 和 position 表示融合，并输入多层 HSTU。该层同时建模两类时间信息：
  - 绝对时间：由 `timestamp` 解码出的 `hour_sin/cos`、`wday_sin/cos`。
  - 相对时间：由位置间时间差经分桶后形成的 RAB 偏置，并加到 $QK^\top$ 上。
5. `训练优化与召回输出层`
  由 `训练优化阶段` 与 `召回推理阶段` 两部分组成。前者面向参数学习与目标优化，后者面向候选编码、Top-K 检索与离线评估。

### 3. 训练阶段与推理阶段要点

1. `训练优化阶段`
  - 学习目标：围绕条件 next-item retrieval，建模 $P(\mathrm{next\_item} \mid \mathrm{history}, \mathrm{query\_context})$。
  - 输入组织：使用带历史行为、query context、正样本和负样本的训练序列。
  - 核心算法：采用 `CDN sampled softmax + logQ correction`，并结合全局负样本与批内负样本提升区分能力。
  - 优化特点：训练阶段关注序列表征和 item 表示的联合学习，而不是直接执行全候选召回。
  - 调度机制：可结合课程学习、负样本难度调度和 `action_nonzero_weight` 等策略控制训练重点。
2. `召回推理阶段`
  - 推理目标：在给定用户历史和当前 query context 的条件下执行 Top-K 向量召回。
  - 输入组织：使用 `predict_seq` 提供历史序列，使用 `predict_set` 作为候选 item 集，使用 `user_query_context` 注入当前请求条件。
  - 核心算法：先编码用户当前序列表征，再编码候选 item 表示，最后执行相似度检索得到 Top-K 结果。
  - 运行特点：推理阶段强调候选构建、批量编码与检索效率，而不是负采样优化。
  - 评估方式：离线场景下可结合 `infer_eval.csv` 计算 `Hit@K / NDCG@K`，检验召回效果。

### 4. 模块特点摘要

1. `统一数据接口层` 的特点是数据源解耦与格式统一，便于跨数据集复用同一条训练/推理链路。
2. `数据构建层` 的特点是“灵活 schema + 索引化压缩 + 条件事件序列化”，既能适配不同特征组合，也能支持大规模样本的高效组织。
3. `特征编码层` 的特点是多路异构特征分治建模，把 user、item、context、time 显式拆开，而不是在输入侧做简单拼接。
4. `条件序列建模层` 的特点是同时引入用户条件调制和双时间建模机制，既保留用户个性化偏好，又保留时间演化信息。
5. `训练优化与召回输出层` 的特点是训练目标与推理目标在任务定义上保持一致，但在算法形式上分别对应“采样式目标优化”和“候选集上的 Top-K 检索”。

### 5. 作图建议

1. 横向保留 5 个一级模块：`统一数据接口层`、`数据构建层`、`特征编码层`、`条件序列建模层`、`训练优化与召回输出层`。
2. `数据构建层` 内部可再细分为 3 个二级模块：`字段选择与 schema 生成`、`索引映射与大规模表示构建`、`条件事件序列构造与样本生成`。
3. `特征编码层` 内部可再细分为 4 个并行二级模块：`用户特征编码`、`物品特征编码`、`上下文特征编码`、`时间特征编码`。
4. `条件序列建模层` 内部可再细分为 3 个二级模块：`FiLM 用户条件调制`、`多源表示融合`、`HSTU 时序建模`。
5. `训练优化与召回输出层` 内部建议拆成 2 个并列二级模块：`训练优化阶段` 与 `召回推理阶段`，分别标注训练目标和推理目标。
6. `索引映射与大规模表示构建`、`FiLM 用户条件调制`、`时间特征编码 + RAB` 建议分别使用不同强调色，作为全图的三个重点模块。

图中几个实现重点可以直接对应到代码：

- 数据构建阶段通过 `indexer["u"] / indexer["i"] / indexer["f"]` 把原始用户、物品和离散特征重映射为紧凑整数 id，再配合 `jsonl + offsets` 存储，避免把整份大数据一次性驻留内存，适合大规模样本的顺序写出和随机读取。
- 特征构建分成 user、item、context、time 四路：user 侧进入 `userdnn`，item 侧进入 `itemdnn`，context 侧进入 `contextdnn`，时间戳则被拆成 `hour_sin/cos + wday_sin/cos` 后经 `time_feat_proj` 融合。
- FiLM 调制发生在用户塔和物品塔之间：先用用户向量生成 `gamma/beta`，再对整条 item 序列表示做 `seq = item_repr * (1 + gamma) + beta`，让不同用户条件直接控制同一物品序列编码。
- 时间信息有两层注入方式：一层是绝对时间周期特征直接加到 token embedding，另一层是 HSTU attention 内的相对时间偏置 RAB，对任意两位置的时间差分桶后加到 $QK^\top$。

## 关键机制与相对基础 HSTU 的增强

下面从 FiLM、双时间建模和 `logQ` 三个角度，说明当前实现为什么这样设计，以及它相对基础 HSTU 序列建模骨架额外增强了什么。

### 1. FiLM 用户条件调制

当前实现中，用户特征先经 `userdnn` 编码为用户条件向量，再通过 `self.cond` 生成两组调制参数 `gamma` 与 `beta`，最终对 item/event 主干表示执行：

```text
seq = item_repr * (1 + gamma) + beta
```

这一设计的核心目的，是在**不破坏 item 主干语义**的前提下，将用户偏好以逐维缩放和平移的方式注入序列表示。相较于把 user 和 item 特征直接拼接后再统一编码，FiLM 有三个直接优势：

1. `保持主干语义稳定`
  item 侧特征先由 `itemdnn` 独立建模，形成相对稳定的物品语义表示；用户条件只负责后续调制，而不是与 item 特征早期混杂，从而减少主干语义被个性化噪声破坏的风险。
2. `个性化调制更细粒度`
  `gamma` 控制逐维增强或抑制，`beta` 控制逐维偏移；同一个 item 表示可以在不同用户条件下被重新标定，而不需要为每类用户重新学习一套 item 编码器。
3. `适合序列级条件控制`
  代码中 `gamma/beta` 会广播到整条 item 序列，因此 FiLM 调制的不是单个 item，而是用户条件下的整段行为序列表示，更适合 query-context-conditioned retrieval。

一个直观例子是：同一件“运动鞋”对两个用户的语义重点可能不同。对偏潮流风格的用户，FiLM 可能放大与款式、品牌相关的表示维度；对偏实用导向的用户，FiLM 可能放大与耐用性、价格相关的表示维度。item 主体仍然是同一个 item，但序列编码会在用户条件下发生有控制的偏置。

相对基础 HSTU 而言，这里的增强点在于：基础 HSTU 更偏序列结构本身，而当前实现显式引入了**用户条件调制机制**，把模型从通用时序编码器扩展为面向个性化召回的条件序列编码器。

### 2. 双时间建模机制

当前实现对同一个 `timestamp` 做了两条并行建模路径：

1. `绝对时间周期特征`
  先将时间戳解码为 `hour_sin/cos` 与 `wday_sin/cos` 四维周期特征，再通过 `time_feat_proj` 加到 token embedding 上。这部分回答的是“行为发生在什么时间点”，建模的是日内周期和周内周期。
2. `相对时间偏置 RAB`
  在 attention 中对任意两个位置计算 $|ts_i - ts_j|$，再把时间间隔分桶并映射为可学习 bias，直接加到 $QK^\top$ 上。这部分回答的是“两个行为相隔多久”，建模的是时间距离和新近性。

这两条路径互补：绝对时间更适合表达长期周期习惯，相对时间更适合表达短期兴趣衰减与最近行为的影响强度。

可以用两个例子理解：

1. `绝对时间例子`
  用户每天早上 8 点打开 App 浏览早餐相关内容。周一 8:00 和周二 8:05 的行为虽然日期不同，但在周期时间特征空间里很接近，模型能学到“这是同一种晨间活跃模式”。
2. `相对时间例子`
  用户 30 天前看过手机壳，10 分钟前看过手机，当前要预测下一次点击。对当前预测位置而言，10 分钟前的行为和现在时间差更小，RAB 会更倾向于提高它的 attention 影响力；30 天前的行为仍可用，但通常影响更弱。

相对基础 HSTU 而言，这里的增强点在于：不仅保留了时间感知 attention 偏置，还额外加入了**显式的周期时间特征投影**。也就是说，当前实现同时建模“什么时候发生”和“相隔多久”这两类不同时间语义，而不是只依赖单一的时距偏置。

### 3. `logQ` 采样偏差校正

训练阶段使用的是 sampled softmax 风格的目标，而不是每一步都对全量 item 做 softmax。这样做的原因是召回场景中的 item 空间通常非常大，全量归一化代价高昂。当前实现进一步从 `item_freq.csv` 构造采样分布 $Q(i)$，并将其对数形式写成 `log_q_table`，在 `CDN_Softmax` 中用于重要性校正。

其核心思想可以写成：

$$
s'(i) = s(i) - \log Q(i)
$$

其中 $s(i)$ 是模型原始相似度分数，$Q(i)$ 是该 item 被采样为正样本或负样本的概率。这样做的原因是：如果热门 item 更容易被采样，而损失函数不做修正，模型就会被采样分布本身牵着走，过度围绕高频 item 学习判别边界。

一个直观例子是：

- 热门 item A 的采样概率较大，例如 $Q(A)=10^{-2}$
- 长尾 item B 的采样概率较小，例如 $Q(B)=10^{-5}$

若不做校正，A 在训练里会被频繁看到，模型容易把“采样频繁”误当成“语义更重要”。加入 `-logQ` 后，热门 item 会被减去更大的采样校正项，长尾 item 则得到相对更公平的比较机会，从而减轻采样偏置。

相对基础 HSTU 而言，这里的增强点不在序列结构，而在**训练目标设计**：当前实现把基础时序编码器进一步适配到大规模向量召回训练中，通过 sampled softmax、`logQ` 校正、全局负样本与批内负样本，使训练更高效，也更符合召回任务的分布特征。

### 4. 相对基础 HSTU 的整体增强总结

如果将基础 HSTU 看作“负责时序建模的主干骨架”，那么当前实现主要做了四类增强：

1. `条件化增强`
  通过 query context 建模和 FiLM 用户调制，把基础 HSTU 扩展为 query-context-conditioned、user-conditioned 的序列召回模型。
2. `时间增强`
  通过“绝对时间周期特征 + 相对时间偏置 RAB”的双时间机制，同时建模周期性与新近性。
3. `表示增强`
  通过 user/item/context/time 四路异构特征编码，以及可选 multimodal item embedding，将输入从单纯 item token 扩展为多源特征表示。
4. `训练增强`
  通过 sampled softmax、`logQ` 校正、全局负样本、批内负样本、课程式负样本难度调度和样本加权策略，把 HSTU 主干适配为更适合大规模召回优化的训练框架。

概括来说，基础 HSTU 解决的是“如何进行时序编码”，而当前实现进一步回答了“如何在个性化、多特征、时间感知和大规模召回训练场景下有效使用 HSTU”。

## 目录

```text
hstu/
├── data/process.py      # 统一 RecKit 数据 -> HSTU split 序列/indexer/cache
├── data/get_stat.py     # 从 train_seq.jsonl 统计 item_freq / item_last_ts
├── dataset.py           # HSTU 训练/推理 Dataset
├── model.py             # HSTU 向量召回模型
├── train.py             # 训练入口，支持 CPU/GPU/NPU 与分布式
├── infer.py             # Top-K 召回推理与指标
└── configs/
    ├── data.json        # 数据转换字段选择示例
    ├── train.json
    └── infer.json
```

## 1. 数据转换

```bash
PYTHONPATH=RecKit python -m reckit.generative_retrieval.vector_retrieval.hstu.data.process \
  --config RecKit/reckit/generative_retrieval/vector_retrieval/hstu/configs/data.json
```

数据转换默认读取 `hstu/configs/data.json`，其中统一管理 `data_root`、`output_dir`、列名映射、样本长度、训练滑窗和 `feature_columns`。命令行显式传入的参数会覆盖 config。

输出包括：

```text
outputs/vector_retrieval/hstu/
├── train_seq.jsonl
├── train_seq_offsets.pkl
├── valid_seq.jsonl
├── valid_seq_offsets.pkl
├── predict_seq.jsonl
├── predict_seq_offsets.pkl
├── predict_set.jsonl
├── infer_eval.csv
├── user_query_context.json
├── meta.json
├── common/
│   ├── indexer.pkl
│   └── item_feat_dict.json
└── cache/
    ├── item_freq.csv
    └── item_last_ts.json
```

三套序列文件的每一行都是一个序列样本，用户行为记录格式为：

```text
[user_reid, item_reid, user_feature, item_feature, action_type, timestamp]
```

其中 `item_feature` 在历史事件中包含 `item_fea.csv` 的物品特征和本次行为的 event context；`predict_set.jsonl`、正样本和负样本只使用物品自身特征，不携带目标事件上下文。

时间与长度切分策略：

- `train_seq.jsonl`：短序列优先进入训练；next-item label 在 Dataset 内部由相邻 item 生成。
- `valid_seq.jsonl`：当同一用户至少有 2 个 eligible target 时写入；Dataset 用样本内前缀预测样本内最后一个 item。
- `predict_seq.jsonl`：当同一用户至少有 3 个 eligible target 时写入；模型用该样本预测 `infer_eval.csv` 中的 `label_iid`，即原始完整序列的最后一个 item。
- `min_item_len` 约束训练前缀最少 item 数。若 `min_item_len=5`，长度 6 只进入 train，长度 7 进入 train/valid，长度 8 开始进入 train/valid/predict。
- `max_item_len > 0` 时，每条 jsonl 样本最多保留这么多个 item。它只限制写出的样本长度，不过滤完整序列，因此没有 `+2` 的换算；valid/predict 不滑窗，只保留最近 `max_item_len` 个 item。
- `train_sliding_window=true` 时，训练前缀会按 `max_item_len` 展开成多个窗口样本；每个窗口仍保留同样的 jsonl 记录格式，并携带 user token。关闭时，训练集也只保留最近 `max_item_len` 个 item。

### 字段选择配置

`data_format.csv` 描述原始数据的字段全集；`data.json` 中的 `feature_columns` 声明本次转换实际使用哪些用户、物品和上下文字段。训练和推理只读取转换后的 `feature_schema`，因此未在 config 中选中的字段不会进入后续流程。

`data.json` 示例：

```json
{
  "data_root": "data_preprocess/foursquare_global/processed_data_sample/NYC",
  "output_dir": "outputs/hstu-NYC/data",
  "min_item_len": 2,
  "max_item_len": 101,
  "user_col": "uid",
  "item_col": "iid",
  "timestamp_col": "timestamp",
  "action_col": "action",
  "train_sliding_window": true,
  "train_window_overlap": 20,
  "feature_columns": [
    {"file_name": "seq.csv", "column_name": "scene"},
    {"file_name": "user_info.csv", "column_name": "age"},
    {"file_name": "item_fea.csv", "column_name": "genres"}
  ]
}
```

`min_item_len` 和 `max_item_len` 都按 item 数计算，不包含 user token。`max_item_len <= 0` 表示不限制；若设为正数，必须不小于 `min_item_len`。`train_window_overlap` 控制相邻训练窗口重叠的 item 数。

长度参数建议保持：

```text
max_item_len = train.maxlen + 1
```

例如训练配置里 `maxlen=100` 时，数据配置里建议设 `max_item_len=101`。原因是 Dataset 会用最多 100 个历史 item 作为输入，并把下一个 item 作为 next-item target；因此一条训练样本最多需要 101 个 item。若 `max_item_len > maxlen + 1`，多出来的更早历史会在 Dataset 侧被截掉；若 `max_item_len < maxlen + 1`，不会丢数据，但模型输入会有更多 padding。

配置格式直接使用 `data_format.csv` 的 `file_name` 和 `column_name`：

```json
{
  "feature_columns": [
    {"file_name": "seq.csv", "column_name": "scene"},
    {"file_name": "user_info.csv", "column_name": "age"},
    {"file_name": "item_fea.csv", "column_name": "genres"}
  ]
}
```

也可以写成按文件分组的形式：

```json
{
  "feature_columns": {
    "seq.csv": ["scene", "device"],
    "user_info.csv": ["age", "gender"],
    "item_fea.csv": ["genres", "popularity"]
  }
}
```

`uid/iid/timestamp` 是序列结构列，会由转换脚本自动读取，不需要作为特征列配置。`action` 也会自动读取，并作为 context 特征参与建模。未传 `--config` 时，默认使用 `data_format.csv` 中所有可用的用户、物品和上下文字段。

基础字段处理：

| 字段 | 处理方式 |
| --- | --- |
| `uid` | 重新编码为从 1 开始的连续整数，作为 user token id，并可通过 `user_emb` 得到 user id embedding |
| `iid` | 重新编码为从 1 开始的连续整数，作为 item token id、正负样本 id 和候选 item id，并可通过 `item_emb` 得到 item id embedding |
| `timestamp` | 见下方「timestamp 格式」；不作为普通 feature 编入 `feature_schema`，训练时进入 `seq_time`，用于绝对时间特征与 HSTU 相对时间偏置 |
| `action` | 自动作为 context 特征编入 `feature_schema`，与 `scene/hour/device` 等字段一起用于 event context 和 query context。`action=0` 会作为普通取值编码，不再承担 padding 语义 |

上下文字段建模：

- `seq.csv` 中除 `uid/iid/timestamp` 以外的字段都会作为 context；`action` 自动加入 context，无需在 `feature_columns` 中显式配置。
- `event_context_t` 描述历史行为 `item_t` 发生时的上下文，例如 `action_t/scene_t/hour_t/device_t`。
- `query_context_{t+1}` 描述当前位置要预测的下一条行为，或线上当前请求条件，例如 `action_{t+1}/scene_{t+1}/hour_{t+1}/device_{t+1}`。
- 自回归训练位置 `t` 的输入为 `history events + query_context_{t+1}`，目标是 `item_{t+1}`。
- 正负样本和推理候选 item embedding 只使用物品自身特征，不携带目标事件 context，避免标签泄露。

#### timestamp 格式

`seq.csv` 中的 `timestamp` 列（可通过 `--timestamp-col` 指定列名）须满足：

| 要求 | 说明 |
| --- | --- |
| 类型 | 整数（`int`） |
| 单位 | **Unix epoch 秒**（自 1970-01-01 00:00:00 UTC 起的秒数） |
| 示例 | `1333476008`（对应 2012-04-03 左右） |

转换脚本会原样写入 jsonl，不做单位换算：

```python
ts = int(row.timestamp)  # 必须是秒，不支持毫秒自动 /1000
```

若原始数据是**毫秒**时间戳，须在写入 `seq.csv` 前自行除以 `1000`；否则小时/星期特征和相对时间偏置都会失真。

时区约定：模型从 timestamp 解码「几点、星期几」时，按该数值所代表的**绝对时刻**计算（注释建议为本地时区 epoch 秒）。若数据是 UTC 秒，则周期特征对应 UTC 时刻，而非用户本地时间。

网络中的两条用法（均由 `seq_time` 驱动，不经过 `feat2emb`）：

1. **绝对时间特征**：由 timestamp 算出 `hour_sin/cos`、`wday_sin/cos` 四维周期特征，经 `time_feat_proj` 加到 token embedding。
2. **相对时间偏置（RAB）**：attention 中对任意两位置计算 $|ts_i - ts_j|$，按对数分桶（默认最大跨度 30 天）得到可学习偏置，加到 $QK^\top$ 上。

`0` 保留给 padding、缺失值、未知值和冷启动。sparse / array 特征值会按字段分别映射为从 1 开始的连续整数；continuous 特征不做离散映射，缺失值填 `0.0`。

`data_format.csv` 到特征类型的映射：

| `data_type,is_list` | 特征类型 | 处理方式 |
| --- | --- | --- |
| `str,false` | sparse | 编码成 id，走 `Embedding` |
| `int,false` | sparse | 编码成 id，走 `Embedding` |
| `float,false` | continuous | 转为 float 后直接拼接 |
| `str,true` | array | 每个元素编码成 id，embedding 后 mean/bagging |
| `int,true` | array | 每个元素编码成 id，embedding 后 mean/bagging |
| `float,true` | continuous 向量 | 转为 float vector 后直接拼接 |

user 侧和 item 侧的同类特征底层处理一致：

| 特征类型 | user 侧 | item 侧 |
| --- | --- | --- |
| sparse | 编码成 id，走 embedding | 编码成 id，走 embedding |
| array | 元素编码成 id，embedding 后 pooling | 元素编码成 id，embedding 后 pooling |
| continuous | float / float vector 直接拼接 | float / float vector 直接拼接 |

不同文件的特征进入模型的位置不同：

- `user_info.csv`：作为 user token 放进序列，和 user id embedding 一起过 `userdnn`，得到用户向量；该向量通过 FiLM 的 `gamma/beta` 调制整条 item 序列表示。
- `item_fea.csv`：作为物品自身特征，和 item id embedding 一起过 `itemdnn` 得到 item 表示。历史 item、正样本、负样本和推理候选都会使用这部分特征。
- `seq.csv` context：在历史侧作为 `event_context_t` 融入对应历史 item event；在预测侧作为 `query_context_{t+1}` 经 `contextdnn` 编码后加到当前位置序列表征上。

因此，类型处理规则在 user、item、context 三侧是一致的，但用途不同：user 特征用于用户条件调制，item 特征用于物品表示，context 特征用于描述历史事件和当前请求条件。生成的 feature id 形如 `user_sparse_1`、`item_continual_1`、`item_array_1`，不再使用固定数字区间。

`predict_set.jsonl` 覆盖 indexer 中的全部 item，只包含物品自身特征；`user_query_context.json` 保存每个推理用户当前要预测的 query context；`infer_eval.csv` 只包含通过 split 过滤后实际出现在 `predict_seq.jsonl` 中的用户，标签是原始完整序列的最后一个 item。

`dataset.py` 依赖 `meta.json` 中的 `feature_schema` 初始化特征类型；如果缺少该字段，需要重新运行 `hstu/data/process.py` 生成数据。

如果统一原始数据里没有足够的用户/物品特征，也可以只基于 user/item id 训练。此时 `feature_schema` 可以为空，但 `meta.json` 仍应由当前数据转换脚本生成。

## 2. 训练

```bash
PYTHONPATH=RecKit python -m reckit.generative_retrieval.vector_retrieval.hstu.train \
  --config RecKit/reckit/generative_retrieval/vector_retrieval/hstu/configs/train.json \
  --data_path outputs/vector_retrieval/hstu \
  --ckpt_root outputs/vector_retrieval/hstu/checkpoints \
  --log_dir outputs/vector_retrieval/hstu/logs \
  --tf_dir outputs/vector_retrieval/hstu/tfevents
```

多卡时使用 `torchrun` 或 NPU 启动器注入 `RANK/WORLD_SIZE/LOCAL_RANK`，脚本会复用 `reckit.utils.distributed` 自动选择：

- NPU：`hccl`
- GPU：`nccl`
- CPU：`gloo`

`USER_CACHE_PATH` 未设置时会默认使用 `data_path/cache`。

训练默认读取 `hstu/configs/train.json`，命令行显式传入的参数会覆盖 config。评估和课程学习按训练总步数比例调度：

- `total_steps=0` 时自动使用 `len(train_loader) * num_epochs`。
- `total_steps>0` 时按 step 数停止训练；此时 `num_epochs` 不再作为训练停止条件。
- `eval_interval_steps=0` 时使用 `eval_interval_ratio * total_steps` 作为评估间隔；`eval_interval_ratio<=0` 会关闭周期性评估。
- `eval_at_epoch_end=true` 时每个 epoch 结束额外评估一次。
- `curriculum_schedule` 用 `start_ratio` 描述阶段边界，每个阶段设置 `hardtopk` 和 `glb_neg_ratio`。
- `action_nonzero_weight` 控制 query context 中原始 `action != 0` 样本的 loss 权重；默认 `1.0`，表示不加权。设置为 `2.0` 或 `3.0` 时，会让这类目标 item 的训练梯度占比更高。
- 评估会同时输出整体指标和 `action!=0` 子集指标，便于观察 `action_nonzero_weight` 影响的样本表现。
- `checkpoint_topk` 控制 eval checkpoint 保留数量，默认 `3`。脚本会分别按两个主指标维护 top-N：`eval_split=valid` 时是 `hit@K[context]` 和 `hit@K[action!=0]`；`eval_split=predict` 时是 `hit@K[predict]` 和 `hit@K[predict_action!=0]`。同一个 checkpoint 如果命中任一榜单就会保留；设置为 `0` 或负数时保留每次 eval 的 checkpoint。
- `train_split` 和 `eval_split` 可在 `train.json` 中显式设置：
  - `eval_split=valid`（默认）：使用 `valid_seq.jsonl`，标签由 Dataset 从序列中自动生成（预测样本内最后一个 item）。
  - `eval_split=predict`：使用 `predict_seq.jsonl`，并固定读取 `data_path/infer_eval.csv` 计算与最终推理一致的 Hit@K / NDCG@K。该文件由 `data/process.py` 生成，列名固定为 `uid` 和 `label_iid`；评估时按 `uid` 查表得到 ground-truth item，再与模型 Top-K 召回结果对比。
- eval checkpoint 目录会分开保存 `model.pt`、`model_config.json` 和 `ckpt.pt`，目录名形如 `global_stepXXXX`；训练结束还会额外保存一个 `final_stepXXXX`。
- 评估指标会作为 `event=eval` 的 JSONL 记录写入 `train.log`，并同时写入 `ckpt.pt` 的 `extra` 字段。
- `model.pt` 只含模型权重；`model_config.json` 记录构建模型所需的结构参数，例如 `maxlen`、`hidden_units`、`num_blocks`、`num_heads`、`mm_emb_id`；`ckpt.pt` 只含恢复训练所需的 optimizer/scheduler/RNG/step 等状态，并通过相对路径引用同目录的 `model.pt` 和 `model_config.json`，避免重复保存模型权重。
- `--resume` 传 checkpoint 目录即可，脚本会自动定位其中的 `ckpt.pt` 并在创建 Dataset/Model 前读取 `model_config.json`；如果传 checkpoint 根目录，则使用最新的 `ckpt.pt`。恢复后会从 epoch 内已完成 batch 之后继续训练。

## 3. 推理 / 指标

```bash
PYTHONPATH=RecKit python -m reckit.generative_retrieval.vector_retrieval.hstu.infer \
  --config RecKit/reckit/generative_retrieval/vector_retrieval/hstu/configs/infer.json \
  --data_path outputs/vector_retrieval/hstu \
  --model_output_path outputs/vector_retrieval/hstu/checkpoints/global_stepXXXX
```

推理会读取 `candidate_file` 指定的候选 JSONL 构建候选 item 向量，默认是 `data_path/predict_set.jsonl`；可以把它改成类目/业务裁剪后的候选集，绝对路径会直接读取，相对路径会按 `data_path` 拼接。存在 `infer_eval.csv` 时会计算 Hit@K / NDCG@K。`labels_file` 默认是 `infer_eval.csv`，可设为空字符串关闭指标计算；标签 CSV 的列名固定为 `uid` 和 `label_iid`。

推理默认读取 `hstu/configs/infer.json`，命令行显式传入的参数会覆盖 config。`--model_output_path` 传 checkpoint 目录即可，脚本会自动定位最新的 `model.pt`，并在创建 Dataset/Model 前读取同目录的 `model_config.json`；因此 `infer.json` 不需要再手动填写模型结构参数，只保留数据路径、checkpoint 目录、batch/chunk/topk 和指标相关参数。`ckpt.pt` 仅用于恢复训练，不能作为推理权重加载。`--output_file` 默认写入 `data_path/predictions.jsonl`，每行格式与 baseline 预测结果保持一致：

```json
{"uid": "原始uid", "topk": ["原始iid1", "原始iid2"], "target_item": ["原始label_iid"]}
```

由于输出字段与 baseline 一致，保存后的预测文件也可以用本目录的离线脚本重新计算
Hit@K / NDCG@K：

```bash
PYTHONPATH=RecKit python -m reckit.generative_retrieval.vector_retrieval.hstu.cal_metrics \
  --pred-file outputs/vector_retrieval/hstu/predictions.jsonl \
  --ks 1,5,10,20,50,100
```

这个脚本适用于 `uid/topk/target_item` 这类 Top-K ranking JSONL，因此适合当前
HSTU vector retrieval 的离线计算；如果其他 vector retrieval 方法也写出同样
字段，就可以直接复用。若字段名不同，可传 `--predictions-field` 和
`--targets-field`。
