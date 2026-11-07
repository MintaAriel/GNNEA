#!/usr/bin/env bash
# Run the Sirena experiment; environment, GPU and training settings come from sirena.yml.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "${PYTHON:-python3}" "$repo_root/scripts/run_offspring_ablation.py" "$@"
