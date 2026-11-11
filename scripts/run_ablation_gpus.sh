#!/usr/bin/env bash
# Matched ablation of configs/sirena.yml spread over several GPUs (one process per GPU).
# usage (project root): GPUS="0 1 2 3" bash scripts/run_ablation_gpus.sh
set -euo pipefail
GPUS=${GPUS:-"0 1 2 3"}
CONFIG=${CONFIG:-configs/sirena.yml}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV:-brian_ase_env}"
mkdir -p logs
N=$(echo $GPUS | wc -w)
first=$(echo $GPUS | cut -d' ' -f1)
# 1. MXtal cluster cache, built once (parallel shards would write the same cache file)
python scripts/run_offspring_ablation.py --config "$CONFIG" --gpu "$first" --prepare-only 2>&1 | tee logs/ablation_prepare.log
# 2. one shard per GPU
i=0; pids=()
for g in $GPUS; do
  python scripts/run_offspring_ablation.py --config "$CONFIG" --gpu "$g" --shard "$i/$N" > "logs/ablation_shard$i.log" 2>&1 &
  pids+=($!); i=$((i + 1))
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail -eq 0 ] || { echo "a shard failed, see logs/ablation_shard*.log"; exit 1; }
# 3. verify every run and write the analysis
python scripts/run_offspring_ablation.py --config "$CONFIG" --gpu "$first" --analyze-only 2>&1 | tee logs/ablation_analyze.log
