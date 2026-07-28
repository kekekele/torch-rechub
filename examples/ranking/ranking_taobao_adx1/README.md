# Ranking TaobaoAd_x1

This directory is parallel to projects/ranking_taac2026 and is dedicated to TaobaoAd_x1 experiments for the three RecKit ranking models:

- DCNv2
- RankMixer
- OneTrans

## Experiment Design

TaobaoAd_x1 is a click-through ranking dataset with strong time dependence, so the recommended reliability path here is not random K-fold. Instead, this project provides a blocked time K-fold benchmark:

- sort built ranking samples by timestamp
- split the full timeline into `K` contiguous blocks
- each run uses one block as `test`
- one adjacent block is used as `valid`
- all remaining earlier and later blocks are used as `train`

This design is more conservative than a random fold split because it reduces temporal leakage and is closer to the actual ranking deployment setting.

## Structure

- dcn_v2/configs: model configs for DCNv2
- rankmixer/configs: model configs for RankMixer
- onetrans/configs: model configs for OneTrans
- outputs/raw/taobaoad_x1: expected RecKit raw data root
- run_taobaoad_x1.py: single-run benchmark orchestration
- run_kfold.py: blocked time K-fold benchmark orchestration

## Expected raw RecKit files

Place converted TaobaoAd_x1 data at:

- projects/ranking_taobao_adx1/outputs/raw/taobaoad_x1/seq.csv
- projects/ranking_taobao_adx1/outputs/raw/taobaoad_x1/user_info.csv
- projects/ranking_taobao_adx1/outputs/raw/taobaoad_x1/item_fea.csv
- projects/ranking_taobao_adx1/outputs/raw/taobaoad_x1/data_format.csv

Recommended core columns:

- seq.csv: uid, iid, timestamp, action, pid, btag_his, cate_his, brand_his
- user_info.csv: uid + user profile fields
- item_fea.csv: iid + ad/item profile fields

## Example commands

Prepare raw RecKit files:

```bash
python projects/ranking_taobao_adx1/prepare_taobaoad_x1.py --output-dir projects/ranking_taobao_adx1/outputs/raw/taobaoad_x1 --force
```

Single benchmark run for all three models:

```bash
python projects/ranking_taobao_adx1/run_taobaoad_x1.py \
	--models dcn_v2 rankmixer onetrans \
	--seeds 2026 \
	--device cuda:0 \
	--force-process \
	--force-train
```

Blocked time K-fold benchmark for all three models:

```bash
python projects/ranking_taobao_adx1/run_kfold.py \
	--models dcn_v2 rankmixer onetrans \
	--k 5 \
	--device cuda:0 \
	--valid-policy adjacent \
	--force-process \
	--force-train
```

Recommended quick validation on one model:

```bash
python projects/ranking_taobao_adx1/run_kfold.py \
	--models dcn_v2 \
	--k 5 \
	--device cuda:0 \
	--valid-policy adjacent \
	--force-process \
	--force-train
```

`run_kfold.py` writes per-fold configs, processed tensors, checkpoints, logs, and summaries to:

```text
projects/ranking_taobao_adx1/outputs/kfold/
```

Main result files are:

- `results.json`
- `fold_metrics.csv`
- `summary.csv`
- `summary.md`

Process data:

PYTHONPATH=. python -m reckit.ranking.dcn_v2.process --config projects/ranking_taobao_adx1/dcn_v2/configs/data.json
PYTHONPATH=. python -m reckit.ranking.rankmixer.process --config projects/ranking_taobao_adx1/rankmixer/configs/data.json
PYTHONPATH=. python -m reckit.ranking.onetrans.process --config projects/ranking_taobao_adx1/onetrans/configs/data.json

Train:

PYTHONPATH=. python -m reckit.ranking.dcn_v2.train --config projects/ranking_taobao_adx1/dcn_v2/configs
PYTHONPATH=. python -m reckit.ranking.rankmixer.train --config projects/ranking_taobao_adx1/rankmixer/configs
PYTHONPATH=. python -m reckit.ranking.onetrans.train --config projects/ranking_taobao_adx1/onetrans/configs

Evaluate:

PYTHONPATH=. python -m reckit.ranking.dcn_v2.evaluate --config projects/ranking_taobao_adx1/dcn_v2/configs --split test
PYTHONPATH=. python -m reckit.ranking.rankmixer.evaluate --config projects/ranking_taobao_adx1/rankmixer/configs --split test
PYTHONPATH=. python -m reckit.ranking.onetrans.evaluate --config projects/ranking_taobao_adx1/onetrans/configs --split test

## Notes

- The current `data.json` files point to `projects/ranking_taobao_adx1/outputs/raw/taobaoad_x1`, so the prepare command above uses the same output directory to stay aligned.
- `run_taobaoad_x1.py` is suitable for a standard single split benchmark.
- `run_kfold.py` is the recommended path when you need a more stable model comparison with mean/std across folds.
- `--valid-policy test` is supported for comparison with the optimistic TAAC2026-style setting, but `adjacent` is the preferred default for TaobaoAd_x1.
