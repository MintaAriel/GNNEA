"""Task 2: per-generation energy limit L_{i+1} = mean_i + z*std_i and good/bad offspring labels."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config import config_from_cli, configure_runtime, project_path

if __name__ == '__main__':
    config = config_from_cli()
    configure_runtime(config)
    from offspring.db import read_individuals_df, read_offspring_df
    from offspring.labels import generation_limits, label_offspring, write_labels_to_db
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=config['_config_path'])
    ap.add_argument('--db', default=config['dataset']['db'])
    ap.add_argument('--natoms', type=int, default=config['dataset']['natoms'])
    ap.add_argument('--z', type=float, default=config['dataset']['z'])
    ap.add_argument('--operators', nargs='*', default=config['dataset']['operators'])
    ap.add_argument('--out', default=str(project_path(config['paths']['data_dir']) / 'labelled_offspring_gnnea.csv'))
    a = ap.parse_args()
    a.db = project_path(a.db)
    a.out = project_path(a.out)

    ind = read_individuals_df(a.db)
    off = read_offspring_df(a.db, operators=a.operators, structures=False)
    lab = label_offspring(off, ind, natoms=a.natoms, z=a.z)
    write_labels_to_db(a.db, lab)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    generation_limits(ind, natoms=a.natoms, z=a.z).to_csv(a.out.with_name('generation_limits_gnnea.csv'))
    lab.to_csv(a.out, index=False)
    print(f'{len(lab)} labelled offspring, good fraction = {lab.label.mean():.3f}')
    print(lab.groupby('generation')['label'].agg(['count', 'mean']).T.round(2).to_string())
