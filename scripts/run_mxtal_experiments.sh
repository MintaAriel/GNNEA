#!/usr/bin/env bash
# MXtalTools-encoder experiments (09_train_mxtal.py), mirroring run_experiments.sh (CGCNN) + the edge-type flag.
# usage: bash scripts/run_mxtal_experiments.sh   (from the project root, ase_env active)
JOBS=${JOBS:-4}
THREADS=${THREADS:-2}
PY=${PY:-python}
mkdir -p results/runs logs
{
for s in 0 1 2; do
  echo "MA_concat_struct_s$s     --fusion concat_pool --seed $s"
  echo "MB_poolcat_struct_s$s    --fusion pool_concat --seed $s"
  echo "MC_concat_energy_s$s     --fusion concat_pool --energy-features --seed $s"
  echo "MG_concat_struct_group_s$s --fusion concat_pool --split group --seed $s"
  echo "MH_concat_energy_group_s$s --fusion concat_pool --energy-features --split group --seed $s"
  echo "MT_concat_edgetype_s$s   --fusion concat_pool --edge-type --seed $s"
done
echo "ME_concat_struct_gen     --fusion concat_pool --split generation"
echo "MF_concat_energy_gen     --fusion concat_pool --energy-features --split generation"
} | xargs -P "$JOBS" -L 1 bash -c 'name=$0; '"$PY"' scripts/09_train_mxtal.py --threads '"$THREADS"' --out results/runs/$name "$@" > logs/$name.log 2>&1; echo "done $name"'
