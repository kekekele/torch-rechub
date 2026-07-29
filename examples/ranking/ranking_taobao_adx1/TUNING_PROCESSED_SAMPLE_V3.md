# TaobaoAd_x1 Processed Sample Tuning V3

This round only updates DCNv2 and RankMixer, and both are tuned from their best previous anchor instead of from the latest version.

## Anchor strategy

- DCNv2: start from the original baseline config because it still has the best valid and test AUC
- RankMixer: start from `processed_sample_v1` because it has the best test AUC among RankMixer variants
- OneTrans: unchanged because it is still the strongest model overall

## Config paths

- `projects/ranking_taobao_adx1/configs_processed_sample_v3/dcn_v2`
- `projects/ranking_taobao_adx1/configs_processed_sample_v3/rankmixer`

## What changed

### DCNv2

Small regularization-only move on top of the baseline:

- keep `embedding_dim=16`
- keep `n_cross_layers=3`
- keep low-rank mixture enabled
- `learning_rate: 1e-3 -> 8e-4`
- `weight_decay: 1e-5 -> 5e-5`
- `earlystop_patience: 3 -> 2`
- `dropout: 0.2 -> 0.25`

### RankMixer

Small move on top of `processed_sample_v1` instead of further simplifying from v2:

- keep broad user/item feature coverage
- keep `seq.timestamp`
- remove `seq.brand_his` from context
- reduce context tokens from `3 -> 2`
- increase target tokens from `4 -> 5` to keep total tokens divisible with CLS
- `history_summary_modes`: `mean,last,target_attention` -> `mean,target_attention`
- `learning_rate: 5e-4 -> 4e-4`
- `weight_decay: 5e-4 -> 7e-4`
- dropout: `0.2 -> 0.22`

## Commands

### DCNv2

```bash
python -m reckit.ranking.dcn_v2.process --config projects/ranking_taobao_adx1/configs_processed_sample_v3/dcn_v2/data.json
python -m reckit.ranking.dcn_v2.train --config projects/ranking_taobao_adx1/configs_processed_sample_v3/dcn_v2 --device cuda:0 --seed 2028
python -m reckit.ranking.dcn_v2.evaluate --config projects/ranking_taobao_adx1/configs_processed_sample_v3/dcn_v2 --split valid --device cuda:0 --seed 2028
python -m reckit.ranking.dcn_v2.evaluate --config projects/ranking_taobao_adx1/configs_processed_sample_v3/dcn_v2 --split test --device cuda:0 --seed 2028
```

### RankMixer

```bash
python -m reckit.ranking.rankmixer.process --config projects/ranking_taobao_adx1/configs_processed_sample_v3/rankmixer/data.json
python -m reckit.ranking.rankmixer.train --config projects/ranking_taobao_adx1/configs_processed_sample_v3/rankmixer --device cuda:0 --seed 2028
python -m reckit.ranking.rankmixer.evaluate --config projects/ranking_taobao_adx1/configs_processed_sample_v3/rankmixer --split valid --device cuda:0 --seed 2028
python -m reckit.ranking.rankmixer.evaluate --config projects/ranking_taobao_adx1/configs_processed_sample_v3/rankmixer --split test --device cuda:0 --seed 2028
```
