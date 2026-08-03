# Ranking Taobao Ad

This project benchmarks the three RecKit ranking models — **DCNv2**, **RankMixer**, and **OneTrans** — on the [飞桨 Taobao Ad dataset](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56).

## Data

Raw files (already on disk):

```
data_preprocess/飞桨_taobao_ad/
├── raw_sample.csv       # user, time_stamp, adgroup_id, pid, nonclk, clk
├── user_profile.csv     # userid + 8 user features
└── ad_feature.csv       # adgroup_id + cate_id, campaign_id, customer, brand, price
```

### Step 1 — Convert to RecKit raw format

```bash
PYTHONPATH=. python projects/ranking_taobao_ad/prepare_taobao_ad.py
```

Output:

```
projects/ranking_taobao_ad/outputs/raw/taobao_ad/
├── seq.csv          # uid, iid, timestamp, action, pid, split_source
├── user_info.csv    # uid + 8 user features
├── item_fea.csv     # iid + 5 item features
├── data_format.csv  # column metadata for RecKit
└── prepare_summary.json
```

Key decisions during conversion:
- Label: `clk=1` → `action=1`, `clk=0` → `action=0`
- `pid` (ad slot, string "xxxxxx_xxxx" format) is kept as a string category feature in `seq.csv`
- No explicit behavior history columns in this dataset; sequence history is built by RecKit's `sample_builder` from previous positive interactions in `seq.csv`
- `user_profile.csv` and `ad_feature.csv` are pure lookup tables (no timestamp); one row per user / item
- `split_source` is set to `"train"` for all rows; global time-ratio split (8/1/1) is applied by the model's process step

### Step 2 — Run the benchmark

```bash
PYTHONPATH=. python projects/ranking_taobao_ad/run_baseline.py \
  --models dcn_v2 rankmixer onetrans \
  --device cuda:0 \
  --force-process \
  --force-train
```

All outputs go under:

```
projects/ranking_taobao_ad/outputs/benchmark/
├── dcn_v2/seed_2026/
├── onetrans/seed_2026/
├── rankmixer/seed_2026/
├── summary.csv
└── summary_aggregated.json
```

## Feature Schema

| Source | Column | Type | Notes |
|--------|--------|------|-------|
| `seq.csv` | `pid` | str | Ad slot ID |
| `user_info.csv` | `cms_segid` | int | User segment |
| `user_info.csv` | `cms_group_id` | int | CMS group |
| `user_info.csv` | `final_gender_code` | int | 1=male, 2=female |
| `user_info.csv` | `age_level` | int | Age bucket |
| `user_info.csv` | `pvalue_level` | int | Purchasing power |
| `user_info.csv` | `shopping_level` | int | Shopping depth |
| `user_info.csv` | `occupation` | int | Is student |
| `user_info.csv` | `new_user_class_level` | int | New-user tier |
| `item_fea.csv` | `cate_id` | int | Product category |
| `item_fea.csv` | `campaign_id` | int | Ad campaign |
| `item_fea.csv` | `customer` | int | Advertiser ID |
| `item_fea.csv` | `brand` | int | Brand ID |
| `item_fea.csv` | `price` | float | Ad price |

## Model Configs

| Model | Config | Key parameters |
|-------|--------|---------------|
| DCNv2 | `dcn_v2/configs/train_dcn_v2.json` | `embedding_dim=16`, low-rank mixture cross net, `n_cross_layers=3`, `num_experts=4`, `lr=8e-4`, `weight_decay=5e-5`, `dropout=0.25` |
| RankMixer | `rankmixer/configs/train_rankmixer.json` | `d_model=64`, CLS pooling, enriched user/history/target blocks, `seq.timestamp` context, `lr=5e-4`, `weight_decay=5e-4`, dropout=0.2 |
| OneTrans | `onetrans/configs/train_onetrans.json` | `d_model=48`, 2 pyramid layers, enriched user/target/context blocks, `seq.timestamp` context, `lr=5e-4`, `weight_decay=5e-4` |

These are warm-start configs anchored to each model's best version summarized in `ranking_dataset_model_comparison.md`: DCNv2 uses the `processed_sample_v3` parameter set, OneTrans uses `processed_sample_v1`, and RankMixer uses `processed_sample_v1`. Only the feature blocks are adapted for this dataset, so `btag_his` / `cate_his` / `brand_his` stay removed while `seq.timestamp` is retained where the best TaobaoAd_x1 feature design benefited from time context. All models still use `global_time_ratio` split 8:1:1, and RankMixer / OneTrans keep `min_history_len=0` so cold-start users are not filtered out.

## Notes

- This dataset has no pre-computed behavior history (unlike taobao_adx1 which has `btag_his`, `cate_his`, `brand_his`). The sequence history used by RankMixer and OneTrans is derived from the users' previous positive interactions within `seq.csv` itself.
- `min_history_len=0` is used so that users without any prior clicks are not filtered out.
- The `pid` feature distinguishes ad slots but carries limited CTR signal — consider dropping it if it hurts generalization.
