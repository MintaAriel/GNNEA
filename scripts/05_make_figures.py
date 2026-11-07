"""Figures + summary tables for the report (reads results/ and data/)."""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from offspring.data import load_labelled
from offspring.db import read_individuals_df

FIG = 'report/figures'
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 200})

ind = read_individuals_df('data/offspring.db')
lab = load_labelled('data/offspring.db', structures=False)
lim = pd.read_csv('data/generation_limits.csv', index_col=0)

# 1. energy landscape per generation + limit
fig, ax = plt.subplots(figsize=(6, 3.2))
pop = ind[ind.natoms == 42]
ax.scatter(pop.generation, pop.offspring_e, s=3, c='0.7', label='population (42 atoms)')
for lbl, c, m in ((0, 'tab:red', 'bad'), (1, 'tab:blue', 'good')):
    s = lab[lab.label == lbl]
    ax.scatter(s.generation + 0.25, s.offspring_e, s=4, c=c, label=f'Heredity offspring ({m})')
ax.step(lim.index, lim['limit'], where='mid', c='k', lw=1, label=r'$L_g=\mu_{g-1}+z\sigma_{g-1}$')
ax.set_ylim(-284.0, -282.6)
ax.set_xlabel('generation')
ax.set_ylabel('energy (eV / 42-atom cell)')
ax.legend(fontsize=6, markerscale=2, loc='upper right')
fig.tight_layout()
fig.savefig(f'{FIG}/energy_vs_generation.pdf')

# 2. class balance per generation and copy fraction
dup = np.isclose(lab.offspring_e, lab.e1, atol=1e-3) | np.isclose(lab.offspring_e, lab.e2, atol=1e-3)
g = pd.DataFrame({'gen': lab.generation, 'good': lab.label, 'copy': dup}).groupby('gen').mean()
fig, ax = plt.subplots(figsize=(6, 2.4))
ax.plot(g.index, g.good, 'o-', ms=3, label='fraction good')
ax.plot(g.index, g['copy'], 's-', ms=3, label='fraction offspring = a parent')
ax.axhline(lab.label.mean(), c='tab:blue', ls=':', lw=1)
ax.set_xlabel('generation')
ax.set_ylim(0, 1)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(f'{FIG}/labels_per_generation.pdf')

# 3. P(good) vs parent energies
fig, ax = plt.subplots(figsize=(3.4, 3.0))
zmin, zmax = lab[['z1', 'z2']].min(axis=1), lab[['z1', 'z2']].max(axis=1)
sc = ax.scatter(zmin, zmax, c=lab.label, cmap='coolwarm_r', s=6, alpha=0.6)
ax.axvline(-0.5244, c='k', lw=0.6, ls='--')
ax.axhline(-0.5244, c='k', lw=0.6, ls='--')
ax.set_xlabel(r'$z_{\min}$ (better parent)')
ax.set_ylabel(r'$z_{\max}$ (worse parent)')
ax.set_title('blue = good offspring', fontsize=8)
fig.tight_layout()
fig.savefig(f'{FIG}/parents_z.pdf')

# 4. GNN results table + learning curves
rows = []
for f in sorted(glob.glob('results/runs/*/metrics.json')):
    m = json.load(open(f))
    name = os.path.basename(os.path.dirname(f))
    exp = name.rsplit('_s', 1)[0] if name[-3:-1] == '_s' else name
    rows.append(dict(run=name, exp=exp, split=m['config']['split'], params=m['n_params'],
                     best_epoch=m['best_epoch'], time_min=m['train_time_s'] / 60,
                     train_acc=m['train']['accuracy'], train_auc=m['train']['roc_auc'],
                     **{f'test_{k}': m['test'][k] for k in ('accuracy', 'balanced_accuracy', 'f1', 'roc_auc')}))
runs = pd.DataFrame(rows)
runs.to_csv('results/gnn_runs.csv', index=False)
summary = runs.drop(columns=['run', 'split']).groupby('exp').agg(['mean', 'std', 'count'])
summary.to_csv('results/gnn_summary.csv')
print(runs.groupby('exp')[[c for c in runs if c.startswith(('test_', 'train_'))] + ['best_epoch', 'time_min']]
      .agg(lambda s: f'{s.mean():.3f}±{s.std():.3f}' if len(s) > 1 else f'{s.mean():.3f}').to_string())

curves = [('A_concat_struct_s0', 'concat-pool, structure only'),
          ('C_concat_energy_s0', 'concat-pool + parent energies'),
          ('G_concat_struct_group_s0', 'concat-pool, group split')]
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))
for run, title in curves:
    p = f'results/runs/{run}/history.csv'
    if not os.path.exists(p):
        continue
    h = pd.read_csv(p)
    l, = axs[0].plot(h.epoch, h.train_loss, lw=1, label=f'{title} (train)')
    axs[0].plot(h.epoch, h.val_loss, lw=1, ls='--', c=l.get_color())
    axs[1].plot(h.epoch, h.val_auc, lw=1, c=l.get_color(), label=title)
axs[0].set_xlabel('epoch'); axs[0].set_ylabel('weighted BCE (— train, -- val)')
axs[1].set_xlabel('epoch'); axs[1].set_ylabel('validation ROC-AUC')
axs[1].legend(fontsize=6)
fig.tight_layout()
fig.savefig(f'{FIG}/learning_curves.pdf')

# 5. test confusion matrices (summed over seeds): GNN vs best baseline on each split
base = json.load(open('results/baselines.json'))
gnn_conf = {}
for f in glob.glob('results/runs/*/metrics.json'):
    name = os.path.basename(os.path.dirname(f))
    exp = name.rsplit('_s', 1)[0] if name[-3:-1] == '_s' else name
    gnn_conf[exp] = gnn_conf.get(exp, 0) + np.array(json.load(open(f))['test']['confusion'])

panels = {'random (3 seeds)': [('A_concat_struct', 'pair-CGCNN\nstructure'), ('C_concat_energy', 'pair-CGCNN\n+ energies'),
                               ('random:RandomForest energy+vol+sym', 'RandomForest')],
          'group (3 seeds)': [('G_concat_struct_group', 'pair-CGCNN\nstructure'), ('H_concat_energy_group', 'pair-CGCNN\n+ energies'),
                              ('group:RandomForest energy+vol+sym', 'RandomForest')],
          'generation': [('E_concat_struct_gen', 'pair-CGCNN\nstructure'), ('F_concat_energy_gen', 'pair-CGCNN\n+ energies'),
                         ('generation:RandomForest energy+vol+sym', 'RandomForest')]}
fig, axs = plt.subplots(3, 3, figsize=(6.0, 6.3))
conf_rows = []
for r, (split, models) in enumerate(panels.items()):
    for c, (key, title) in enumerate(models):
        if ':' in key:
            sp, m = key.split(':')
            cm = np.array(base[f'{sp}_confusion'][m])
        else:
            cm = gnn_conf[key]
        conf_rows.append(dict(split=split, model=title.replace('\n', ' '), TN=cm[0, 0], FP=cm[0, 1],
                              FN=cm[1, 0], TP=cm[1, 1]))
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax = axs[r, c]
        ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]}\n{100 * norm[i, j]:.0f}%', ha='center', va='center', fontsize=7,
                        color='white' if norm[i, j] > 0.6 else 'black')
        ax.set_xticks([0, 1], ['bad', 'good'])
        ax.set_yticks([0, 1], ['bad', 'good'])
        if r == 0:
            ax.set_title(title, fontsize=8)
        if c == 0:
            ax.set_ylabel(f'{split}\ntrue', fontsize=8)
        if r == 2:
            ax.set_xlabel('predicted', fontsize=8)
fig.tight_layout()
fig.savefig(f'{FIG}/confusion_matrices.pdf')
pd.DataFrame(conf_rows).to_csv('results/confusion_matrices.csv', index=False)
print(pd.DataFrame(conf_rows).to_string(index=False))
