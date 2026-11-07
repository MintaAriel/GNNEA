#!/bin/bash
# Run the same experiment grid on a cluster (GPU or many-core CPU node).
# Adapt partition / account / modules to the cluster, then:  sbatch scripts/train_slurm.sbatch
#SBATCH --job-name=offspring-gnn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ase_env            # needs: torch (CUDA build on GPU nodes), torch_geometric, vesin, ase, ea
cd "$SLURM_SUBMIT_DIR"

# data/offspring.db must exist (scripts/01_build_db.py + 02_label.py, or copy it from the laptop)
# on a GPU every run uses the device automatically (--device cuda is the default when available)
JOBS=${JOBS:-4} THREADS=${THREADS:-4} bash scripts/run_experiments.sh
