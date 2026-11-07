"""Task 1: parse Individuals / BESTIndividuals / gatheredPOSCARS (USPEX 2126) into an ASE db."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from offspring.db import build_offspring_db, read_offspring_df

DEFAULT_RESULTS = os.path.expanduser('~/Documents/uspex/theo_uspexpython/results1')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=DEFAULT_RESULTS)
    ap.add_argument('--db', default='data/offspring.db')
    ap.add_argument('--uspex-ver', default='2126')
    a = ap.parse_args()

    build_offspring_db(a.results, a.db, uspex_ver=a.uspex_ver)
    df = read_offspring_df(a.db)
    print(f'{a.db}: {len(df)} offspring')
    print(df.drop(columns=['p1', 'p2', 'offspring']).head())
    print(df['operator'].value_counts())
