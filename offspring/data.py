"""Labelled pair table + reproducible train/val/test splits shared by the GNN and the baselines."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from offspring.db import read_individuals_df, read_offspring_df
from offspring.labels import Z_30, generation_limits, label_offspring

# parent energies as z-scores w.r.t. the previous generation (the population they were selected from)
ENERGY_FEATURES = ['z1', 'z2']


def load_labelled(db_path, natoms=42, z=Z_30, operators=('Heredity',), structures=True):
    ind = read_individuals_df(db_path)
    off = read_offspring_df(db_path, operators=list(operators), structures=structures)
    lab = label_offspring(off, ind, natoms=natoms, z=z)
    lim = generation_limits(ind, natoms=natoms, z=z)
    mean = lab['generation'].map(lim['mean_prev'])
    std = lab['generation'].map(lim['std_prev'])
    lab['z1'] = (lab['e1'] - mean) / std
    lab['z2'] = (lab['e2'] - mean) / std
    return lab


def split_indices(df, split='random', seed=0, val=0.15, test=0.15, gen_val=29, gen_test=35):
    """random: stratified 70/15/15 (seminar 8 uses train_test_split twice).
    generation: train on gens < gen_val, validate on [gen_val, gen_test), test on >= gen_test
    (closer to the real use case: predict future offspring from the past).
    group: like random but pairs of identical parents (same unordered parent energies; USPEX keeps
    copies of good structures under new ids) never end up in different subsets -> no leakage."""
    idx = np.arange(len(df))
    if split == 'random':
        tr, te = train_test_split(idx, test_size=test, random_state=seed, stratify=df['label'])
        tr, va = train_test_split(tr, test_size=val / (1 - test), random_state=seed,
                                  stratify=df['label'].values[tr])
    elif split == 'generation':
        g = df['generation'].values
        tr, va, te = idx[g < gen_val], idx[(g >= gen_val) & (g < gen_test)], idx[g >= gen_test]
    elif split == 'group':
        groups = parent_pair_groups(df)
        gss = GroupShuffleSplit(n_splits=1, test_size=test, random_state=seed)
        tr, te = next(gss.split(idx, groups=groups))
        gss = GroupShuffleSplit(n_splits=1, test_size=val / (1 - test), random_state=seed)
        a, b = next(gss.split(tr, groups=groups[tr]))
        tr, va = tr[a], tr[b]
    else:
        raise ValueError(split)
    return tr, va, te


def parent_pair_groups(df, decimals=3):
    e = np.sort(np.round(df[['e1', 'e2']].values, decimals), axis=1)
    return np.unique(e, axis=0, return_inverse=True)[1].ravel()
