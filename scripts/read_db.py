"""Read the offspring DataFrame back from the ASE db.

    python scripts/read_db.py data/offspring.db [--operators Heredity] [--csv out.csv]
or, from python:
    from offspring.db import read_offspring_df
    df = read_offspring_df('data/offspring.db', operators=['Heredity'])
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from offspring.db import read_offspring_df

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('db')
    ap.add_argument('--operators', nargs='*', default=None)
    ap.add_argument('--csv', default=None, help='save the table without structures')
    a = ap.parse_args()

    df = read_offspring_df(a.db, operators=a.operators)
    print(df.drop(columns=['p1', 'p2', 'offspring']).to_string(max_rows=20))
    print(df[['p1', 'p2']].head())
    if a.csv:
        df.drop(columns=['p1', 'p2', 'offspring']).to_csv(a.csv, index=False)
