# TaobaoAd_x1 Processed Sample Tuning V1

This tuning set targets the sample built with:

```bash
python projects/ranking_taobao_adx1/sample_taobaoad_x1_raw.py \
  --input-dir /hy-tmp/RecKit-master/data_preprocess/taobao_adx1/processed \
  --days 2 \
  --max-rows 2000000 \
  --min-user-events 2 \
  --keep-positive-only-users \
  --force \
  --output-dir /hy-tmp/RecKit-master/data_preprocess/taobao_adx1/processed_sample
```

## Why this tuning set

Observed baseline behavior from logs:

- all three models reduce train loss quickly but valid/test AUC stays near 0.52-0.54
- valid AUC is consistently higher than test AUC, which indicates overfitting / split shift
- RankMixer and OneTrans currently use only a subset of available user/item/context features
- DCNv2 low-rank mixture experts show heavy collapse, so the mixture routing is likely too flexible for this sample

## Main changes

- lower learning rate from `1e-3` to `5e-4`
- increase weight decay from `1e-5` to `5e-4`
- reduce batch size from `1024` to `512`
- slightly increase max epochs and let early stop decide
- enrich RankMixer / OneTrans with more user/item/context features
- add `seq.timestamp` time-period features to RankMixer / OneTrans context blocks
- simplify DCNv2 and disable low-rank mixture routing
- simplify OneTrans capacity to reduce overfitting on the 2-day sample

## Config paths

- `projects/ranking_taobao_adx1/configs_processed_sample_v1/dcn_v2`
- `projects/ranking_taobao_adx1/configs_processed_sample_v1/rankmixer`
- `projects/ranking_taobao_adx1/configs_processed_sample_v1/onetrans`

## Commands

### DCNv2

```bash
python -m reckit.ranking.dcn_v2.process --config projects/ranking_taobao_adx1/configs_processed_sample_v1/dcn_v2/data.json
python -m reckit.ranking.dcn_v2.train --config projects/ranking_taobao_adx1/configs_processed_sample_v1/dcn_v2 --device cuda:0 --seed 2028
python -m reckit.ranking.dcn_v2.evaluate --config projects/ranking_taobao_adx1/configs_processed_sample_v1/dcn_v2 --split valid --device cuda:0 --seed 2028
python -m reckit.ranking.dcn_v2.evaluate --config projects/ranking_taobao_adx1/configs_processed_sample_v1/dcn_v2 --split test --device cuda:0 --seed 2028
```

### RankMixer

```bash
python -m reckit.ranking.rankmixer.process --config projects/ranking_taobao_adx1/configs_processed_sample_v1/rankmixer/data.json
python -m reckit.ranking.rankmixer.train --config projects/ranking_taobao_adx1/configs_processed_sample_v1/rankmixer --device cuda:0 --seed 2028
python -m reckit.ranking.rankmixer.evaluate --config projects/ranking_taobao_adx1/configs_processed_sample_v1/rankmixer --split valid --device cuda:0 --seed 2028
python -m reckit.ranking.rankmixer.evaluate --config projects/ranking_taobao_adx1/configs_processed_sample_v1/rankmixer --split test --device cuda:0 --seed 2028
```

### OneTrans

```bash
python -m reckit.ranking.onetrans.process --config projects/ranking_taobao_adx1/configs_processed_sample_v1/onetrans/data.json
python -m reckit.ranking.onetrans.train --config projects/ranking_taobao_adx1/configs_processed_sample_v1/onetrans --device cuda:0 --seed 2028
python -m reckit.ranking.onetrans.evaluate --config projects/ranking_taobao_adx1/configs_processed_sample_v1/onetrans --split valid --device cuda:0 --seed 2028
python -m reckit.ranking.onetrans.evaluate --config projects/ranking_taobao_adx1/configs_processed_sample_v1/onetrans --split test --device cuda:0 --seed 2028
```

## Extra information that would help the next tuning round

- processed sample `conversion_summary.json` or `sample_summary.json`
- per-split positive rate for train / valid / test after each model `process`
- item vocabulary size and user vocabulary size from each model `meta.json`
- average / p90 history length from processed tensors
- one full train log for the best and worst model after this tuning round
