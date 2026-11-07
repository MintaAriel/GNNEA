"""Results of the MXtalTools-encoder experiments (09) next to pair-CGCNN (03), the baselines (04) and the
RDF clustering of the parents (10).

1. per-run and per-experiment tables, MXtal vs CGCNN        -> results/mxtal_runs.csv, results/mxtal_summary.csv
2. baselines + RDF cluster features (same splits as 04)     -> results/rdf_baselines.json
3. test ROC-AUC split by same / different parent cluster    -> results/mxtal_by_cluster.csv
4. figures                                                  -> report/figures/mxtal_*.{pdf,png}

    python scripts/11_mxtal_analysis.py      (after 09 runs, 10_parent_clusters.py and 04_baselines.py)
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from offspring.data import load_labelled, split_indices
from offspring.db import read_individuals_df

FIG = 'report/figures'
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'figure.dpi': 200})

# experiment code -> (split, description); the CGCNN letter and the MXtal 'M' + letter share the setting
SETTINGS = {'A': ('random', 'concat-pool, structure'), 'B': ('random', 'pool-concat, structure'),
            'C': ('random', 'concat-pool + energies'), 'T': ('random', 'concat-pool + edge type'),
            'G': ('group', 'concat-pool, structure'), 'H': ('group', 'concat-pool + energies'),
            'E': ('generation', 'concat-pool, structure'), 'F': ('generation', 'concat-pool + energies')}


def exp_name(run):
    return run.rsplit('_s', 1)[0] if run[-3:-1] == '_s' else run


def score(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                f1=f1_score(y, pred, zero_division=0), roc_auc=roc_auc_score(y, p))


# ----------------------------------------------------------------------------- 1. run tables
rows = []
for f in sorted(glob.glob('results/runs/*/metrics.json') + glob.glob('results/runs_mxtal_nonorm/*/metrics.json')):
    m = json.load(open(f))
    run = os.path.basename(os.path.dirname(f))
    nonorm = 'runs_mxtal_nonorm' in f
    is_mx = m.get('encoder') == 'mxtal'
    code = run[1] if is_mx else run[0]
    if code not in SETTINGS:
        continue
    rows.append(dict(run=run, path=os.path.dirname(f), exp=exp_name(run),
                     encoder='MXtal, no norm' if nonorm else ('MXtal' if is_mx else 'CGCNN'), code=code,
                     split=m['config']['split'], setting=SETTINGS[code][1], params=m['n_params'],
                     best_epoch=m['best_epoch'], time_min=m['train_time_s'] / 60,
                     train_auc=m['train']['roc_auc'],
                     **{f'test_{k}': m['test'][k] for k in ('accuracy', 'balanced_accuracy', 'f1', 'roc_auc')}))
runs = pd.DataFrame(rows)
runs.to_csv('results/mxtal_runs.csv', index=False)
metric_cols = ['test_accuracy', 'test_balanced_accuracy', 'test_f1', 'test_roc_auc', 'train_auc', 'best_epoch',
               'time_min', 'params']
summary = runs.groupby(['code', 'encoder', 'split', 'setting'])[metric_cols].agg(['mean', 'std', 'count'])
summary.to_csv('results/mxtal_summary.csv')
fmt = runs.groupby(['split', 'setting', 'encoder'])[metric_cols[:5]].agg(
    lambda s: f'{s.mean():.3f}±{s.std():.3f}' if len(s) > 1 else f'{s.mean():.3f}')
print(fmt.to_string())

# ----------------------------------------------------------------------------- 2. baselines with RDF features
lab = load_labelled('data/offspring.db', structures=False)
ind = read_individuals_df('data/offspring.db').set_index('uspex_id')
for k in (1, 2):
    pid = lab[f'p{k}_id'].astype(int)
    lab[f'vol{k}'] = pid.map(ind['volume']).values / 42
    lab[f'sym{k}'] = pid.map(ind['symmetry']).values
lab['zmin'], lab['zmax'] = lab[['z1', 'z2']].min(axis=1), lab[['z1', 'z2']].max(axis=1)
lab['vmin'], lab['vmax'] = lab[['vol1', 'vol2']].min(axis=1), lab[['vol1', 'vol2']].max(axis=1)
pr = pd.read_csv('data/pair_rdf_distance.csv').set_index('offspring_id')
lab['rdf_dist'] = lab.offspring_id.map(pr['rdf_dist']).values
lab['same_cluster'] = lab.offspring_id.map(pr['same_cluster']).astype(float).values
cl = pd.read_csv('data/parent_clusters.csv').set_index('uspex_id')
csize = cl.cluster.map(cl.cluster.value_counts())
lab['csize_min'] = np.minimum(lab.p1_id.astype(int).map(csize).values, lab.p2_id.astype(int).map(csize).values)
lab['csize_max'] = np.maximum(lab.p1_id.astype(int).map(csize).values, lab.p2_id.astype(int).map(csize).values)
assert lab.rdf_dist.notna().all()

base = ['zmin', 'zmax', 'vmin', 'vmax', 'sym1', 'sym2']
rdf = ['rdf_dist', 'same_cluster', 'csize_min', 'csize_max']
models = {
    'LogReg RDF distance': (lambda s: LogisticRegression(class_weight='balanced'), ['rdf_dist']),
    'RF RDF features': (lambda s: RandomForestClassifier(300, min_samples_leaf=5, class_weight='balanced',
                                                         random_state=s), rdf),
    'RF energy+vol+sym': (lambda s: RandomForestClassifier(300, min_samples_leaf=5, class_weight='balanced',
                                                           random_state=s), base),
    'RF energy+vol+sym+RDF': (lambda s: RandomForestClassifier(300, min_samples_leaf=5, class_weight='balanced',
                                                               random_state=s), base + rdf),
}
rdf_base = {}
y = lab.label.values
for split in ('random', 'group', 'generation'):
    res = {}
    for seed in ((0,) if split == 'generation' else (0, 1, 2)):
        tr, va, te = split_indices(lab, split=split, seed=seed)
        trv = np.concatenate([tr, va])
        for name, (make, cols) in models.items():
            m = make(seed).fit(lab[cols].values[trv], y[trv])
            res.setdefault(name, []).append(score(y[te], m.predict_proba(lab[cols].values[te])[:, 1]))
    rdf_base[split] = {k: {mm: [float(np.mean([r[mm] for r in v])), float(np.std([r[mm] for r in v]))]
                           for mm in v[0]} for k, v in res.items()}
with open('results/rdf_baselines.json', 'w') as f:
    json.dump(rdf_base, f, indent=1)
print(json.dumps({s: {k: round(v['roc_auc'][0], 3) for k, v in d.items()} for s, d in rdf_base.items()}, indent=1))

# ----------------------------------------------------------------------------- 3. test AUC by parent cluster
by = []
for _, r in runs.iterrows():
    p = f'{r.path}/test_predictions.csv'
    if not os.path.exists(p):
        continue
    t = pd.read_csv(p)
    t['same'] = t.offspring_id.map(pr['same_cluster']).values
    run = r.run
    for name, sub in (('same cluster', t[t.same]), ('different cluster', t[~t.same]), ('all', t)):
        if sub.y.nunique() > 1:
            by.append(dict(run=run, exp=r.exp, encoder=r.encoder, code=r.code, subset=name, n=len(sub),
                           good_rate=sub.y.mean(), roc_auc=roc_auc_score(sub.y, sub.p_good),
                           accuracy=accuracy_score(sub.y, (sub.p_good >= .5).astype(int))))
by = pd.DataFrame(by)
by.to_csv('results/mxtal_by_cluster.csv', index=False)
if len(by):
    print(by.groupby(['code', 'encoder', 'subset'])[['n', 'good_rate', 'roc_auc', 'accuracy']].mean()
          .round(3).to_string())

# ----------------------------------------------------------------------------- 4. figures
# (a) test ROC-AUC per setting: CGCNN vs MXtal vs RF baselines
base_json = json.load(open('results/baselines.json'))
order = ['A', 'B', 'C', 'T', 'G', 'H', 'E', 'F']
fig, ax = plt.subplots(figsize=(7.0, 2.8))
xt, xl, labelled = [], [], set()
for i, code in enumerate(order):
    split, desc = SETTINGS[code]
    for j, (enc, col) in enumerate((('CGCNN', 'tab:gray'), ('MXtal, no norm', 'lightsteelblue'),
                                    ('MXtal', 'tab:blue'))):
        v = runs[(runs.code == code) & (runs.encoder == enc)].test_roc_auc
        if len(v):
            x = i + (j - 1) * 0.27
            ax.bar(x, v.mean(), 0.26, yerr=v.std() if len(v) > 1 else 0, color=col, capsize=2,
                   label=enc if code in ('A', 'T') and enc not in labelled else None)
            labelled.add(enc)
            ax.scatter(np.full(len(v), x), v, s=4, c='k', zorder=3)
    xt.append(i)
    xl.append(f'{code}\n{split}')
for split, x0, x1 in (('random', -0.5, 3.5), ('group', 3.5, 5.5), ('generation', 5.5, 7.5)):
    rf = base_json[split]['RandomForest energy+vol+sym']['roc_auc'][0]
    ax.hlines(rf, x0, x1, colors='tab:red', linestyles='--', lw=1, label='RF energy+vol+sym' if split == 'random'
              else None)
    rr = rdf_base[split]['RF energy+vol+sym+RDF']['roc_auc'][0]
    ax.hlines(rr, x0, x1, colors='tab:green', linestyles=':', lw=1.2, label='RF + RDF features' if split == 'random'
              else None)
ax.set_xticks(xt, xl, fontsize=7)
ax.set_ylabel('test ROC-AUC')
ax.set_ylim(0.25, 0.95)
ax.axhline(0.5, c='k', lw=0.5)
ax.legend(fontsize=6.5, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False)
fig.tight_layout()
fig.savefig(f'{FIG}/mxtal_results.pdf')
fig.savefig(f'{FIG}/mxtal_results.png')

# (b) learning curves, MXtal vs CGCNN on the same settings (seed 0)
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
for ax, (cg, mx, title) in zip(axs, (('A_concat_struct_s0', 'MA_concat_struct_s0', 'A: structure, random'),
                                     ('C_concat_energy_s0', 'MC_concat_energy_s0', 'C: + energies, random'),
                                     ('G_concat_struct_group_s0', 'MG_concat_struct_group_s0', 'G: structure, group'))):
    for p0, run, col, lbl in (('results/runs', cg, 'tab:gray', 'CGCNN'),
                              ('results/runs_mxtal_nonorm', mx, 'lightsteelblue', 'MXtal, no norm'),
                              ('results/runs', mx, 'tab:blue', 'MXtal')):
        p = f'{p0}/{run}/history.csv'
        if os.path.exists(p):
            h = pd.read_csv(p)
            ax.plot(h.epoch, h.train_loss, c=col, lw=1, label=f'{lbl} train')
            ax.plot(h.epoch, h.val_loss, c=col, lw=1, ls='--', label=f'{lbl} val')
    ax.set_title(title, fontsize=7)
    ax.set_xlabel('epoch')
axs[0].set_ylabel('weighted BCE')
axs[0].set_ylim(0.6, 1.3)
axs[0].legend(fontsize=5.5)
fig.tight_layout()
fig.savefig(f'{FIG}/mxtal_learning_curves.pdf')
fig.savefig(f'{FIG}/mxtal_learning_curves.png')

# (c) same / different cluster AUC
if len(by):
    g = by[by.code.isin(['A', 'C', 'G', 'H']) & (by.encoder != 'MXtal, no norm')].groupby(['code', 'encoder', 'subset']).roc_auc.mean().unstack()
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    g[['same cluster', 'different cluster', 'all']].plot.bar(ax=ax, width=0.75,
                                                            color=['tab:orange', 'tab:purple', '0.6'])
    ax.set_xticklabels([f'{c}\n{e}' for c, e in g.index], rotation=0, fontsize=6)
    ax.set_xlabel('')
    ax.set_ylabel('test ROC-AUC')
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, c='k', lw=0.5)
    ax.legend(fontsize=6, title='parents in', title_fontsize=6)
    fig.tight_layout()
    fig.savefig(f'{FIG}/mxtal_by_cluster.pdf')
    fig.savefig(f'{FIG}/mxtal_by_cluster.png')
print('done')
