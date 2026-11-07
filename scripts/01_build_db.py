"""Task 1: parse Individuals / BESTIndividuals / gatheredPOSCARS (USPEX 2126) into an ASE db."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config import config_from_cli, configure_runtime, project_path

if __name__ == '__main__':
    config = config_from_cli()
    configure_runtime(config)
    from offspring.db import build_offspring_db, read_offspring_df
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=config['_config_path'])
    ap.add_argument('--results', default=config['dataset']['results_dir'])
    ap.add_argument('--db', default=config['dataset']['db'])
    ap.add_argument('--uspex-ver', default=config['dataset']['uspex_ver'])
    ap.add_argument('--overwrite', action='store_true', help='replace the selected output database')
    a = ap.parse_args()
    path = project_path(a.db)
    if path.exists() and not a.overwrite:
        raise FileExistsError(f'{path} already exists; select a new --db or pass --overwrite.')
    path.parent.mkdir(parents=True, exist_ok=True)
    build_offspring_db(project_path(a.results), path, uspex_ver=a.uspex_ver,
                       project_name=config['project']['name'])
    df = read_offspring_df(path)
    print(f'{path}: {len(df)} offspring with ASE Atoms in the offspring column')
    print(df.drop(columns=['p1', 'p2', 'offspring']).head())
    print(df['operator'].value_counts())
