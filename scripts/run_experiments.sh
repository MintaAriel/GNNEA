#!/usr/bin/env bash
# All GNN experiments of the report. Runs JOBS trainings in parallel with THREADS torch threads each.
# usage: bash scripts/run_experiments.sh   (from the project root, ase_env active)
JOBS=${JOBS:-2}
THREADS=${THREADS:-4}
mkdir -p results/runs logs
{
for s in 0 1 2; do
  echo "A_concat_struct_s$s    --fusion concat_pool --seed $s"
  echo "B_poolcat_struct_s$s   --fusion pool_concat --seed $s"
  echo "C_concat_energy_s$s    --fusion concat_pool --energy-features --seed $s"
  echo "G_concat_struct_group_s$s --fusion concat_pool --split group --seed $s"
  echo "H_concat_energy_group_s$s --fusion concat_pool --energy-features --split group --seed $s"
done
echo "D_concat_separate_s0   --fusion concat_pool --separate-encoders --seed 0"
echo "E_concat_struct_gen    --fusion concat_pool --split generation"
echo "F_concat_energy_gen    --fusion concat_pool --energy-features --split generation"
echo "I_concat_struct_64_s0  --fusion concat_pool --node-fea 64 --edge-fea 64 --h-fea 128 --seed 0"
} | xargs -P "$JOBS" -L 1 bash -c 'name=$0; python scripts/03_train_gnn.py --threads '"$THREADS"' --out results/runs/$name "$@" > logs/$name.log 2>&1; echo "done $name"'
