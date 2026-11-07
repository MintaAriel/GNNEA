# GNNEA

Graph neural networks for classifying good/bad Heredity offspring from a theophylline USPEX search.
The latest previous study is `report/mxtal_report.tex` (November 3): the corrected MXtal encoder matched pair-CGCNN from parent structures. The new experiment tests whether including the offspring geometry helps.

Use `configs/sirena.yml` for all current machine, environment, dataset, model, training and experiment settings. It inherits portable paths and the project name from `configs/base.yaml`. Relative paths resolve against the repository root. Other machine profiles retain their own external software paths.

On Sirena, use `brian_ase_env` and physical GPU **0**, matching the device index in nvtop/nvidia-smi. The launcher sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES` before importing torch. It automatically enters the configured conda environment when necessary.

```bash
conda activate brian_ase_env
python scripts/run_offspring_ablation.py --config configs/sirena.yml
# Recreate tables and figures from completed, verified runs:
python scripts/run_offspring_ablation.py --config configs/sirena.yml --analyze-only
```

For a fresh dataset, run `python scripts/01_build_db.py --config configs/sirena.yml` and then `python scripts/02_label.py --config configs/sirena.yml` before training. The builder requires `--overwrite` to replace an existing database.

`bash scripts/remote_gpu_run.sh` launches the same configured experiment in the foreground. Monitor it with `nvtop` and `tail -f logs/offspring_ablation.log` when redirecting its output to that log.

The new database is `data/offspring_gnnea.db`; the previous report database is preserved as `data/offspring.db`. Each offspring row stores relaxed ASE row atoms plus an explicit calculator-free `row.data.offspring_atoms` dictionary. `read_offspring_df` reconstructs an `ase.Atoms` object in its `offspring` column and also supports the original database schema.

```python
from offspring.db import read_offspring_df
from src.config import load_config, project_path
config = load_config()
df = read_offspring_df(project_path(config['dataset']['db']), operators=['Heredity'])
# df.p1, df.p2, df.offspring contain ASE Atoms objects.
```

The configured ablation contains 32 runs: CGCNN/MXtal, concat-pool/pool-concat, parents/parents+offspring, three parent-group split seeds and one generation split seed. Parent energies are disabled for this structure-only comparison. Both variants use identical labels, split IDs, training sampler seeds, hyperparameters and validation-loss checkpoint selection.

- `concat_pool`: concatenate the node embeddings of all enabled structures, then pool. Adding the child keeps the parameter count unchanged when the encoder is shared.
- `pool_concat`: append the pooled child embedding to `[g1 + g2, abs(g1 - g2), g1 * g2]`. This preserves parent symmetry and increases the head width.

**The available offspring geometry is already relaxed.** This tests post-relaxation classification. Screening candidates before relaxation requires unrelaxed offspring structures, which are absent from this dataset. Offspring energies are used to create labels and never supplied as input features. Parent-group splitting follows the previous report (unordered parent energies rounded to three decimals); it does not guarantee structure-disjoint sets. Generation splitting holds out future generations of the same search.

Outputs are under `results/offspring_ablation/`: `summary.csv`, `paired_deltas.csv` (paired parent-group bootstrap intervals), `prediction_changes.csv` (corrected/regressed decisions), `structure_id_reuse.csv`, `verification.json`, `offspring_comparison.{png,pdf}`, `offspring_roc.{png,pdf}`, and `report.txt`. A report copy is saved in `report/offspring_ablation.txt`. Every run retains its resolved YAML, split manifest, checkpoint, learning history, predictions and metrics with database/source fingerprints and GPU provenance. Resume accepts completed runs only when their fingerprints match.

Single runs support CLI overrides:

```bash
python scripts/03_train_gnn.py --offspring-structure --fusion pool_concat --out results/manual_cgcnn
python scripts/09_train_mxtal.py --offspring-structure --fusion concat_pool --out results/manual_mxtal
python -m unittest discover -s tests -v
```

Shared code lives in `src/`: configuration, environment/GPU selection, a common training loop and matched analysis. `offspring/` contains database/labels/splits and the CGCNN/MXtal graph and model implementations. The numbered analysis scripts and old shell experiment grids reproduce the earlier parent-only reports; their historical analysis paths target `data/offspring.db` and `results/runs/`.
