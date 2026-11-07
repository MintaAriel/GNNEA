# Predicting good / bad Heredity offspring with a pair-CGCNN

Data: python-USPEX (2026, `uspex_ver='2126'`) run on theophylline (C14N8H16O4 per 42-atom cell),
`~/Documents/uspex/theo_uspexpython/results1` (extracted from `theo_uspexpython.zip`, `the_final_test/results1`).

Environment: `conda activate ase_env` with `pip install -e ~/PycharmProjects/ASE-Evolutionary-Algorithm`
(`ea` module) plus `torch` (CPU wheel), `torch_geometric`, `vesin`, `scikit-learn`.

All commands are run from this folder.

| step                  | command                                                                                                                      | output                                                                                                                 |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| speed check           | `python scripts/00_benchmark_speed.py`                                                                                         | `results/benchmark_speed.txt`                                                                                            |
| Task 1: ASE db        | `python scripts/01_build_db.py [--results DIR]`                                                                                | `data/offspring.db`                                                                                                      |
| read the db           | `python scripts/read_db.py data/offspring.db --operators Heredity`                                                             | DataFrame `p1 p2 e1 e2 offspring_e operator generation p1_id p2_id ...`                                                  |
| Task 2: labels        | `python scripts/02_label.py`                                                                                                   | `label`, `limit` written in the db, `data/labelled_offspring.csv`, `data/generation_limits.csv`                                |
| Task 2: one GNN       | `python scripts/03_train_gnn.py --fusion concat_pool [--energy-features] [--split random|group|generation]`                   | `results/runs/<name>/`                                                                                                   |
| all GNN runs          | `bash scripts/run_experiments.sh` (cluster: `sbatch scripts/train_slurm.sbatch`)                                                 | `results/runs/*`                                                                                                         |
| baselines             | `python scripts/04_baselines.py`                                                                                               | `results/baselines.json`                                                                                                 |
| MXtalTools encoder    | `python scripts/09_train_mxtal.py --out results/runs/M_concat_struct [--fusion pool_concat] [--energy-features] [--edge-type]` | `results/runs/<name>/` (same files as 03)                                                                                |
| parent clusters (RDF) | `python scripts/10_parent_clusters.py [--d-cut X] [--all-individuals]`                                                         | `data/parent_clusters.csv`, `data/pair_rdf_distance.csv`, `results/parent_clusters.json`, `report/figures/parent_clusters.png` |
| all MXtal runs        | `bash scripts/run_mxtal_experiments.sh` (`--no-crystal-norm` in 09 = faithful head, collapses)                                   | `results/runs/M*`                                                                                                        |
| MXtal analysis        | `python scripts/11_mxtal_analysis.py`                                                                                          | `results/mxtal_{runs,summary,by_cluster}.csv`, `results/rdf_baselines.json`, `report/figures/mxtal_*`                        |
| MXtal report          | `cd report && tectonic mxtal_report.tex`                                                                                       | `report/mxtal_report.pdf`                                                                                                |
| figures               | `python scripts/05_make_figures.py`                                                                                            | `report/figures/*`, `results/gnn_summary.csv`                                                                              |
| Task 3: report        | `cd report && tectonic report.tex`                                                                                             | `report/report.pdf`                                                                                                      |

From python:

```python
from offspring.db import read_offspring_df
df = read_offspring_df('data/offspring.db', operators=['Heredity'])   # p1/p2/offspring are ase.Atoms
```

Package layout (`offspring/`): `mxtal.py` (MXtalTools crystal encoder + RDF port), `db.py` (ASE db build/read), `labels.py` (generation limit and labels),
`data.py` (labelled table, splits), `graphs.py` (periodic graphs, pair batching), `model.py` (pair-CGCNN).