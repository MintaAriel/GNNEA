#!/usr/bin/env bash
# Run the experiment grid on the shared GPU server (sirena@10.16.64.66, /home/Brian/GNN).
#   bash scripts/remote_gpu_run.sh            (from /home/Brian/GNN)
# GPU 7: CUDA_DEVICE_ORDER=PCI_BUS_ID makes CUDA indices match the nvidia-smi indices
# (by default CUDA orders "fastest first", which can differ from nvidia-smi).
set -e
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-7}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate brian_ase_env

python - <<'PY' || pip install torch torch_geometric vesin scikit-learn pandas ase
import torch, torch_geometric, vesin, sklearn, ase, pandas
assert torch.cuda.is_available()
PY
python -c "import torch; print('using', torch.cuda.get_device_name(0), 'pci', torch.cuda.get_device_properties(0).pci_bus_id)"
nvidia-smi --query-gpu=index,pci.bus_id,memory.used,utilization.gpu --format=csv

mkdir -p logs
JOBS=${JOBS:-4} THREADS=${THREADS:-2} nohup bash scripts/run_experiments.sh > logs/run_all.out 2>&1 &
echo "started, pid $!  (tail -f logs/run_all.out)"
