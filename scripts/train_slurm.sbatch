#!/bin/bash
# Run the same experiment grid on a cluster (GPU or many-core CPU node).
# Adapt partition / account / modules to the cluster, then:  sbatch scripts/train_slurm.sbatch
#SBATCH --job-name=GNNEA
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
# Conda environment and physical GPU are selected by configs/sirena.yml.
exec python3 scripts/run_offspring_ablation.py
