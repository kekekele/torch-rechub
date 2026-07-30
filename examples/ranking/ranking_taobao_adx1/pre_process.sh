#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/hy-tmp/RecKit-master"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR"

run_process() {
  local model="$1"
  local config="$2"

  echo "=================================================="
  echo "[START] model=${model}"
  echo "[CONFIG] ${config}"
  echo "[TIME] $(date '+%F %T')"
  echo "=================================================="

  python -m "reckit.ranking.${model}.process" --config "${config}"

  echo "=================================================="
  echo "[DONE] model=${model}"
  echo "[TIME] $(date '+%F %T')"
  echo "=================================================="
}

run_process "dcn_v2" "projects/ranking_taobao_adx1/dcn_v2/configs/data.json"
run_process "rankmixer" "projects/ranking_taobao_adx1/rankmixer/configs/data.json"
run_process "onetrans" "projects/ranking_taobao_adx1/onetrans/configs/data.json"

echo "All process jobs finished successfully."