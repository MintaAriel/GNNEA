"""Task 2: good/bad offspring labels from a per-generation energy limit.

    L_{i+1} = mean_i + z * std_i ,   z = -0.5244  (Phi(z) = 0.30)

mean_i / std_i are the energy statistics of the population of generation i (restricted to
structures with `natoms` atoms so that total energies are comparable). For a normal energy
distribution L_{i+1} is the 30th percentile, i.e. the boundary of the best (lowest) 30 % of the
previous generation. An offspring created in generation g is
    label = 1 (good)  if offspring_e <= L_g
    label = 0 (bad)   if offspring_e >  L_g
"""
import numpy as np
import pandas as pd
from ase.db import connect

Z_30 = -0.5244


def generation_limits(individuals, natoms=42, z=Z_30, exclude_operators=()):
    """DataFrame indexed by generation g with mean/std of g-1 and limit L_g."""
    pop = individuals[(individuals['natoms'] == natoms) & np.isfinite(individuals['offspring_e'])]
    if exclude_operators:
        pop = pop[~pop['operator'].isin(exclude_operators)]
    stats = pop.groupby('generation')['offspring_e'].agg(['mean', 'std', 'count'])
    stats['limit_next'] = stats['mean'] + z * stats['std']
    # limit used for offspring born in generation g comes from generation g-1
    lim = stats.copy()
    lim.index = lim.index + 1
    lim = lim.rename(columns={'mean': 'mean_prev', 'std': 'std_prev', 'count': 'n_prev',
                              'limit_next': 'limit'})
    return lim


def label_offspring(offspring, individuals, natoms=42, z=Z_30, exclude_operators=()):
    """Filter to pairs whose parents both have `natoms` atoms and add 'limit' and 'label'."""
    df = offspring[(offspring['natoms_p1'] == natoms) & (offspring['natoms_p2'] == natoms)].copy()
    lim = generation_limits(individuals, natoms=natoms, z=z, exclude_operators=exclude_operators)
    df['limit'] = df['generation'].map(lim['limit'])
    df = df[df['limit'].notna()]
    df['label'] = (df['offspring_e'] <= df['limit']).astype(int)
    return df.reset_index(drop=True)


def write_labels_to_db(db_path, labelled):
    """Store label/limit as key-value pairs of the offspring rows in the ASE database."""
    with connect(db_path) as db:
        for oid, label, limit in zip(labelled['offspring_id'], labelled['label'], labelled['limit']):
            row = db.get(uspex_id=int(oid))
            db.update(row.id, label=int(label), limit=float(limit))
