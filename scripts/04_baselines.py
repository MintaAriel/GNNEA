"""Reference models on the same splits as the GNN + an estimate of the label noise ceiling."""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from offspring.data import load_labelled, split_indices
from offspring.db import read_individuals_df


def score(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                f1=f1_score(y, pred, zero_division=0), roc_auc=roc_auc_score(y, p))


if __name__ == '__main__':
    db = 'data/offspring.db'
    lab = load_labelled(db, structures=False)
    ind = read_individuals_df(db).set_index('uspex_id')
    for k in (1, 2):
        pid = lab[f'p{k}_id'].astype(int)
        lab[f'vol{k}'] = pid.map(ind['volume']).values / 42
        lab[f'sym{k}'] = pid.map(ind['symmetry']).values
    # symmetric (order-free) versions of the parent features
    lab['zmin'], lab['zmax'] = lab[['z1', 'z2']].min(axis=1), lab[['z1', 'z2']].max(axis=1)
    lab['vmin'], lab['vmax'] = lab[['vol1', 'vol2']].min(axis=1), lab[['vol1', 'vol2']].max(axis=1)

    feats = {'energy (z_min, z_max)': ['zmin', 'zmax'],
             'energy + volume + symmetry': ['zmin', 'zmax', 'vmin', 'vmax', 'sym1', 'sym2']}
    out = {}
    for split in ('random', 'group', 'generation'):
        seeds = (0,) if split == 'generation' else (0, 1, 2)
        res, conf = {}, {}
        for seed in seeds:
            tr, va, te = split_indices(lab, split=split, seed=seed)
            trv = np.concatenate([tr, va])  # no hyper-parameter search -> fit on train+val
            y = lab['label'].values
            models = {
                'majority': (DummyClassifier(strategy='most_frequent'), ['zmin']),
                'LogReg energy': (LogisticRegression(class_weight='balanced'), feats['energy (z_min, z_max)']),
                'LogReg energy+vol+sym': (LogisticRegression(class_weight='balanced', max_iter=1000),
                                          feats['energy + volume + symmetry']),
                'RandomForest energy+vol+sym': (RandomForestClassifier(300, min_samples_leaf=5,
                                                                       class_weight='balanced', random_state=seed),
                                                feats['energy + volume + symmetry']),
            }
            for name, (m, cols) in models.items():
                m.fit(lab[cols].values[trv], y[trv])
                p = m.predict_proba(lab[cols].values[te])[:, 1]
                res.setdefault(name, []).append(score(y[te], p))
                conf.setdefault(name, []).append(confusion_matrix(y[te], (p >= 0.5).astype(int), labels=[0, 1]))
            # rule: both parents already inside the best 30 % of the previous generation
            rule = (lab['zmax'].values[te] <= -0.5244).astype(float)
            res.setdefault('rule: both parents below limit', []).append(score(y[te], rule))
            conf.setdefault('rule: both parents below limit', []).append(
                confusion_matrix(y[te], rule.astype(int), labels=[0, 1]))
        out[split] = {k: {m: [np.mean([r[m] for r in v]), np.std([r[m] for r in v])] for m in v[0]}
                      for k, v in res.items()}
        # test confusion matrices [[TN, FP], [FN, TP]] summed over the seeds
        out[f'{split}_confusion'] = {k: np.sum(v, axis=0).tolist() for k, v in conf.items()}

    # label-noise ceiling: identical parent pairs (same unordered energies -> same structures, USPEX
    # keeps copies of the best structures under new ids) that produced different labels
    key = lab.apply(lambda r: tuple(sorted((round(r.e1, 3), round(r.e2, 3)))), axis=1)
    g = lab.groupby(key)['label'].agg(['sum', 'count'])
    best_possible = (np.maximum(g['sum'], g['count'] - g['sum'])).sum() / len(lab)
    dup = np.isclose(lab.offspring_e, lab.e1, atol=1e-3) | np.isclose(lab.offspring_e, lab.e2, atol=1e-3)
    out['analysis'] = dict(
        n_pairs=len(lab), good_fraction=lab.label.mean(),
        n_unique_parent_pairs=len(g), n_pairs_in_repeated_groups=int(g['count'][g['count'] > 1].sum()),
        n_conflicting_groups=int(((g['sum'] > 0) & (g['sum'] < g['count'])).sum()),
        max_accuracy_given_identical_inputs=best_possible,
        offspring_equal_to_a_parent=dup.mean(),
        p_good_if_copy=lab.label[dup].mean(), p_good_if_new=lab.label[~dup].mean(),
        n_unique_parent_structures=int(pd.concat([lab.p1_id, lab.p2_id]).nunique()),
        n_unique_parent_energies=int(pd.concat([lab.e1, lab.e2]).round(3).nunique()))
    print(json.dumps(out, indent=1))
    os.makedirs('results', exist_ok=True)
    with open('results/baselines.json', 'w') as f:
        json.dump(out, f, indent=1)
