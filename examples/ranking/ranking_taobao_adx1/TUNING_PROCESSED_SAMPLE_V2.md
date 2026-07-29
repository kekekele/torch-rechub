# TaobaoAd_x1 Processed Sample Tuning V2

This round only updates DCNv2 and RankMixer.
OneTrans is kept unchanged because `processed_sample_v1` already showed a clear gain.

## Intent

- DCNv2: further reduce capacity and stop earlier
- RankMixer: reduce token count, remove weaker context features, keep only stronger user/item/context signals

## Config paths

- `projects/ranking_taobao_adx1/configs_processed_sample_v2/dcn_v2`
- `projects/ranking_taobao_adx1/configs_processed_sample_v2/rankmixer`

## Commands

### DCNv2

```bash
python -m reckit.ranking.dcn_v2.process --config projects/ranking_taobao_adx1/configs_processed_sample_v2/dcn_v2/data.json
python -m reckit.ranking.dcn_v2.train --config projects/ranking_taobao_adx1/configs_processed_sample_v2/dcn_v2 --device cuda:0 --seed 2028
python -m reckit.ranking.dcn_v2.evaluate --config projects/ranking_taobao_adx1/configs_processed_sample_v2/dcn_v2 --split valid --device cuda:0 --seed 2028
python -m reckit.ranking.dcn_v2.evaluate --config projects/ranking_taobao_adx1/configs_processed_sample_v2/dcn_v2 --split test --device cuda:0 --seed 2028
```

### RankMixer

```bash
python -m reckit.ranking.rankmixer.process --config projects/ranking_taobao_adx1/configs_processed_sample_v2/rankmixer/data.json
python -m reckit.ranking.rankmixer.train --config projects/ranking_taobao_adx1/configs_processed_sample_v2/rankmixer --device cuda:0 --seed 2028
python -m reckit.ranking.rankmixer.evaluate --config projects/ranking_taobao_adx1/configs_processed_sample_v2/rankmixer --split valid --device cuda:0 --seed 2028
python -m reckit.ranking.rankmixer.evaluate --config projects/ranking_taobao_adx1/configs_processed_sample_v2/rankmixer --split test --device cuda:0 --seed 2028
```

## What changed

### DCNv2

- `embedding_dim: 12 -> 8`
- `learning_rate: 5e-4 -> 3e-4`
- `weight_decay: 5e-4 -> 1e-3`
- `earlystop_patience: 4 -> 2`
- `mlp dims: [128, 64] -> [64, 32]`
- `dropout: 0.3 -> 0.4`

### RankMixer

- token count reduced from `16` total to `8` total including CLS
- `history_summary_modes`: `mean,last,target_attention` -> `mean,target_attention`
- drop weaker context history fields and keep only `seq.pid` and `seq.timestamp`
- `learning_rate: 5e-4 -> 3e-4`
- `weight_decay: 5e-4 -> 1e-3`
- dropout increased to `0.25`
