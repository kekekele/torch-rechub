# prepare_taobaoad_x1.py 数据转换说明

本文档说明 [projects/ranking_taobao_adx1/prepare_taobaoad_x1.py](projects/ranking_taobao_adx1/prepare_taobaoad_x1.py) 如何把 TaobaoAd_x1 原始 `train.csv` 和 `test.csv` 转成 RecKit 排序任务使用的原始表。

## 脚本目标

TaobaoAd_x1 原始数据是广告点击样本表，每一行表示一次曝光样本，包含：

- 用户侧离散特征
- 广告侧离散特征
- 若干历史字段，如 `btag_his`、`cate_his`、`brand_his`
- 点击标签，如 `clk`

RecKit 排序模型需要的输入不是原始广告样本格式，而是三张原始表：

- `seq.csv`：样本主表
- `user_info.csv`：用户侧静态特征表
- `item_fea.csv`：物品侧静态特征表
- `data_format.csv`：字段类型说明表

脚本的工作就是把原始 `train/test` CSV 转成上述格式，并生成 `conversion_summary.json` 作为统计摘要。

## 输入

脚本支持两种输入来源：

1. 用户显式传入：
   - `--train-path`
   - `--test-path`
2. 如果本地不存在且未设置 `--no-auto-download`，脚本会自动尝试：
   - 先下载 RecZoo Datasets GitHub 压缩包并解压
   - 找不到时再回退到 HuggingFace 镜像下载 `train.csv` / `test.csv`

典型命令：

```bash
python projects/ranking_taobao_adx1/prepare_taobaoad_x1.py \
  --output-dir projects/ranking_taobao_adx1/outputs/raw/taobaoad_x1 \
  --force
```

如果源文件已经在本地：

```bash
python projects/ranking_taobao_adx1/prepare_taobaoad_x1.py \
  --train-path /path/to/train.csv \
  --test-path /path/to/test.csv \
  --output-dir projects/ranking_taobao_adx1/outputs/raw/taobaoad_x1 \
  --force
```

## 输出

脚本默认输出到：

- `projects/ranking_taobao_adx1/data_preprocess/taobao_adx1/processed`

实验中更常见的做法是显式指定：

- `projects/ranking_taobao_adx1/outputs/raw/taobaoad_x1`

最终会写出：

- `seq.csv`
- `user_info.csv`
- `item_fea.csv`
- `data_format.csv`
- `conversion_summary.json`

## 时间处理的核心逻辑

这是这个脚本里最容易误解、也是最关键的部分。

### 1. 原始数据可能没有显式时间戳

脚本会优先查找以下列名作为时间列：

- `timestamp`
- `time_stamp`
- `time`

如果原始文件中存在这些列，就优先使用原始时间。

如果原始文件里根本没有这些列，脚本不会放弃，而是按“文件行顺序隐含了访问顺序”来构造时间。

也就是说，这个脚本默认采用如下假设：

- 同一个文件内，样本行顺序本身就代表了时间先后
- 即使没有真实 wall-clock 时间，也可以构造一个单调递增的伪时间序列，供后续样本切分和历史构造使用

### 2. 无时间列时，按用户内出现顺序构造伪时间

当找不到时间列时，脚本调用 `_fallback_timestamps()` 生成时间。规则是：

- 维护一个 `fallback_state[uid]` 计数器
- 每遇到该用户一条新样本，计数器加 `1`
- 该行的伪时间戳取值为：

$$
\text{timestamp} = \text{timestamp\_offset} + \text{fallback\_state}[uid]
$$

因此，对同一个用户：

- 第 1 次出现，时间为 `offset + 1`
- 第 2 次出现，时间为 `offset + 2`
- 第 3 次出现，时间为 `offset + 3`

这个时间不是全局严格按文件行号递增，而是“按用户维度的出现顺序递增”。

### 3. train 和 test 使用不同时间区间

脚本对 `train.csv` 和 `test.csv` 调用 `_process_file()` 时使用了不同的 `timestamp_offset`：

- `train`: `timestamp_offset = 0`
- `test`: `timestamp_offset = 2_000_000_000`

这意味着即便原始数据没有时间列，脚本仍然会强制保证：

- train 构造出的伪时间在较小区间
- test 构造出的伪时间整体远大于 train

这样做的目的很直接：

- 保证 `test` 样本在时间上晚于 `train`
- 避免后续基于时间排序构造历史时，把测试样本错误地放到训练样本之前

这是一个非常重要的工程保护措施。

### 4. 有时间列但部分值缺失时，按缺失行单独回退

如果原始文件存在时间列，但其中某些行是空值或无法转成数值，脚本会：

- 对正常行保留原始时间
- 仅对缺失时间的那些行使用 `_fallback_timestamps()` 补齐

所以它不是“只要出现一个空值就整列重建时间”，而是“局部缺失局部回退”。

### 5. fallback 计数器跨 chunk 持续累积

脚本按 chunk 流式读取大文件，默认 `chunksize=500000`。但时间回退状态不是每个 chunk 清零，而是整个文件处理期间持续累积：

- 同一用户在第 1 个 chunk 出现一次，得到 `1`
- 在第 7 个 chunk 再出现时，会继续得到 `2`

这保证了大文件分块处理时，伪时间仍然与整份文件的用户出现顺序一致。

## 时间处理示例

### 示例 1：原始文件没有时间列

输入样例：

```csv
userid,adgroup_id,clk,pid,cms_segid,cate_id,brand
u1,i10,1,430548_1007,3,55,801
u2,i20,0,430548_1007,5,61,900
u1,i11,1,430548_1007,3,72,803
u1,i12,0,430548_1007,3,55,801
u2,i21,1,430548_1007,5,61,901
```

如果这份文件被当作 `train.csv` 处理，则生成的 `seq.csv` 关键列大致为：

```csv
uid,iid,timestamp,action,pid,split_source
u1,i10,1,1,0,train
u2,i20,1,0,0,train
u1,i11,2,1,0,train
u1,i12,3,0,0,train
u2,i21,2,1,0,train
```

注意这里：

- `u1` 的时间是 `1,2,3`
- `u2` 的时间是 `1,2`
- 它反映的是“同一用户在文件里的出现顺序”

### 示例 2：test.csv 无时间列

如果相同规则作用在 `test.csv`，由于 `offset=2_000_000_000`，则时间大致会变成：

```csv
uid,iid,timestamp,action,pid,split_source
u1,i99,2000000001,1,0,test
u2,i88,2000000001,0,0,test
u1,i77,2000000002,1,0,test
```

这样测试集天然晚于训练集。

### 示例 3：原始文件有时间列，但部分缺失

输入样例：

```csv
userid,adgroup_id,time_stamp,clk
u1,i10,1710000000,1
u1,i11,,0
u1,i12,1710000002,1
```

转换时：

- 第 1 行保留 `1710000000`
- 第 2 行因为缺失，回退生成一个伪时间
- 第 3 行保留 `1710000002`

也就是说，脚本会尽量保留原始时间信息，只在无法使用时补值。

## 输入列如何映射到输出列

脚本会自动识别常见列名变体。

### 主键和标签

- 用户列：`uid` / `userid` / `user` / `nick` / `user_id`
- 物品列：`iid` / `adgroup_id` / `item_id`
- 点击列：`clk` / `click` / `label` / `action`
- 非点击列：`noclk` / `non_click`

标签生成逻辑是：

- 如果存在 `clk` 类列，`clk > 0` 记作 `action = 1`
- 如果不存在 `clk`，但存在 `noclk`，则 `noclk == 0` 记作 `action = 1`

### 序列侧字段

输出到 `seq.csv` 的字段包括：

- `uid`
- `iid`
- `timestamp`
- `action`
- `pid`
- `btag_his`
- `cate_his`
- `brand_his`
- `split_source`

其中 `btag_his`、`cate_his`、`brand_his` 会经过 `_normalize_list_cell()` 规范化：

- 支持用 `^`、`,`、`|` 分隔
- 去掉空值、`nan`、`null`、`0` 等占位符
- 最终统一存成逗号分隔字符串

例如：

- 原始 `"12^18^25"` -> 输出 `"12,18,25"`
- 原始 `"0"` -> 输出空串

## user_info.csv 和 item_fea.csv 如何生成

脚本不是简单复制每一行，而是为每个用户、每个物品保留“最新时间对应的一份特征”。

### user_info.csv

对每个 `uid`：

- 取该用户在所有样本中时间最大的那一条记录
- 保存对应的用户特征

包含字段：

- `uid`
- `cms_segid`
- `cms_group_id`
- `final_gender_code`
- `age_level`
- `pvalue_level`
- `shopping_level`
- `occupation`
- `new_user_class_level`

### item_fea.csv

对每个 `iid`：

- 取该物品在所有样本中时间最大的那一条记录
- 保存对应的物品特征

包含字段：

- `iid`
- `cate_id`
- `campaign_id`
- `customer`
- `brand`
- `price`

## 输出样例

### seq.csv 样例

```csv
uid,iid,timestamp,action,pid,btag_his,cate_his,brand_his,split_source
u1,i10,1,1,0,"12,18","55,72","801,803",train
u1,i11,2,0,0,"12,18","55,72","801,803",train
u2,i20,1,1,0,"7,9","61,88","900,901",train
```

### user_info.csv 样例

```csv
uid,cms_segid,cms_group_id,final_gender_code,age_level,pvalue_level,shopping_level,occupation,new_user_class_level
u1,3,5,1,4,2,3,0,1
u2,7,9,2,3,1,2,1,2
```

### item_fea.csv 样例

```csv
iid,cate_id,campaign_id,customer,brand,price
i10,55,1021,3001,801,128.0
i11,72,1022,3005,803,88.5
i20,61,1033,3010,900,256.0
```

### conversion_summary.json 样例

```json
{
  "output_dir": "projects/ranking_taobao_adx1/outputs/raw/taobaoad_x1",
  "seq_rows": 3000000,
  "positives": 123456,
  "negatives": 2876544,
  "users": 456789,
  "items": 987654,
  "reports": [
    {
      "file": ".../train.csv",
      "split_source": "train",
      "rows": 2500000,
      "positives": 100000,
      "negatives": 2400000,
      "chunks": 5
    },
    {
      "file": ".../test.csv",
      "split_source": "test",
      "rows": 500000,
      "positives": 23456,
      "negatives": 476544,
      "chunks": 1
    }
  ]
}
```

## 对实验的实际影响

时间处理方式会直接影响后续排序实验：

- `process.py` 会基于 `timestamp` 构造历史行为顺序
- `run_kfold.py` 会基于 `timestamp` 做 blocked time split
- 如果时间无序，历史信息和切分都会被污染

因此，这个脚本的时间处理策略本质上是在做两件事：

1. 尽量保留原始时间信息
2. 当原始时间不存在时，用“文件中的隐含顺序”构造一个足够稳定的替代时间轴

## 使用建议

- 如果原始数据没有显式时间列，可以直接使用当前脚本，不需要额外补时间字段。
- 如果你确认原始 `train.csv` / `test.csv` 的行顺序已经代表真实曝光顺序，那么当前伪时间策略是合理的。
- 如果后续拿到更可靠的真实时间列，优先保留真实时间，不要再依赖 fallback。
- 如果怀疑源文件顺序被打乱，当前伪时间将不再可靠，这时应该先回到源数据确认排序依据。
