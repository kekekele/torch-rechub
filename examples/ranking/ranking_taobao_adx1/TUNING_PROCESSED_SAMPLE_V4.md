# TaobaoAd_x1 Processed Sample Tuning V4

This round only updates RankMixer and keeps the `processed_sample_v1` architecture and feature coverage intact.

## Why V4

`processed_sample_v1` is still the best RankMixer version on test AUC.
`processed_sample_v2` and `processed_sample_v3` both changed token layout or feature/context structure too much and regressed.
So V4 only makes small training-side adjustments on top of V1.

## Config path

- `projects/ranking_taobao_adx1/configs_processed_sample_v4/rankmixer`

## What changed from V1

- keep the same ordered blocks and token counts
- keep `history_summary_modes = [mean, last, target_attention]`
- keep all user/item/context features from V1
- `learning_rate: 5e-4 -> 4.5e-4`
- `weight_decay: 5e-4 -> 7e-4`
- `earlystop_patience: 4 -> 3`
- `input/token_mixing/ffn dropout: 0.2 -> 0.24`

## Commands

```bash
python -m reckit.ranking.rankmixer.process --config projects/ranking_taobao_adx1/configs_processed_sample_v4/rankmixer/data.json
python -m reckit.ranking.rankmixer.train --config projects/ranking_taobao_adx1/configs_processed_sample_v4/rankmixer --device cuda:0 --seed 2028
python -m reckit.ranking.rankmixer.evaluate --config projects/ranking_taobao_adx1/configs_processed_sample_v4/rankmixer --split valid --device cuda:0 --seed 2028
python -m reckit.ranking.rankmixer.evaluate --config projects/ranking_taobao_adx1/configs_processed_sample_v4/rankmixer --split test --device cuda:0 --seed 2028
```
