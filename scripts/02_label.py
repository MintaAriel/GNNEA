"""Task 2: per-generation energy limit L_{i+1} = mean_i + z*std_i and good/bad offspring labels."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from offspring.db import read_individuals_df, read_offspring_df
from offspring.labels import Z_30, generation_limits, label_offspring, write_labels_to_db

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='data/offspring.db')
    ap.add_argument('--natoms', type=int, default=42)
    ap.add_argument('--z', type=float, default=Z_30)
    ap.add_argument('--operators', nargs='*', default=['Heredity'])
    ap.add_argument('--out', default='data/labelled_offspring.csv')
    a = ap.parse_args()

    ind = read_individuals_df(a.db)
    off = read_offspring_df(a.db, operators=a.operators, structures=False)
    lab = label_offspring(off, ind, natoms=a.natoms, z=a.z)
    write_labels_to_db(a.db, lab)

    generation_limits(ind, natoms=a.natoms, z=a.z).to_csv('data/generation_limits.csv')
    lab.to_csv(a.out, index=False)
    print(f'{len(lab)} labelled offspring, good fraction = {lab.label.mean():.3f}')
    print(lab.groupby('generation')['label'].agg(['count', 'mean']).T.round(2).to_string())
