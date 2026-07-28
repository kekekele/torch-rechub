# Ranking TaobaoAd_x1

This directory is parallel to projects/ranking_taac2026 and is dedicated to TaobaoAd_x1 experiments for the three RecKit ranking models:

- DCNv2
- RankMixer
- OneTrans

## Structure

- dcn_v2/configs: model configs for DCNv2
- rankmixer/configs: model configs for RankMixer
- onetrans/configs: model configs for OneTrans
- outputs/raw/taobaoad_x1: expected RecKit raw data root

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
