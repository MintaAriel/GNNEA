"""Report figures drawn at their printed size (A4, 2 cm margins -> 6.69 in text width).

Fonts are 9 pt so that they match the 11 pt main text once printed; every figure is written as a vector
PDF (used by LaTeX) and a 300 dpi PNG.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from offspring.data import load_labelled
from offspring.db import read_individuals_df
from offspring.labels import Z_30

FIG = 'report/figures'
W = 6.69                         # text width, inches
OLIVE, DARK, GREY, LIGHT = '#8E9719', '#404040', '#8C8C8C', '#C8C8C8'
BLUE, ORANGE, GREEN = '#1f77b4', '#ff7f0e', '#2ca02c'
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'savefig.dpi': 300,
                     'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'{FIG}/{name}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


ind = read_individuals_df('data/offspring.db')
pop = ind[ind.natoms == 42]
lab = load_labelled('data/offspring.db', structures=False)
lim = pd.read_csv('data/generation_limits.csv', index_col=0)
base = json.load(open('results/baselines.json'))

# ------------------------------------------------------------------ Fig. 2: labels
GEN = 20
fig, axs = plt.subplots(1, 2, figsize=(W, 2.55), gridspec_kw={'width_ratios': [1.25, 1]})
ax = axs[0]
pp = pop[pop.offspring_e < -283.02]
ax.scatter(pp.generation, pp.offspring_e, s=3, c=LIGHT, lw=0, label='population')
bad, good = lab[lab.label == 0], lab[lab.label == 1]
ax.scatter(bad.generation + 0.25, bad.offspring_e, s=4, c=DARK, lw=0, label='bad child')
ax.scatter(good.generation + 0.25, good.offspring_e, s=5, c=OLIVE, lw=0, label='good child')
ax.step(lim.index, lim['limit'], where='mid', c='k', lw=1.2, label=r'limit $L_g$')
ax.set_ylim(-283.95, -282.72)  # empty band at the top for the legend
ax.set_xlabel('generation')
ax.set_ylabel('energy (eV / 42-atom cell)')
ax.set_title('(a) Heredity children and the limit', loc='left')
ax.legend(loc='upper right', ncol=2, markerscale=2.5, frameon=False, handletextpad=0.2, columnspacing=0.8)

ax = axs[1]
e = pop[pop.generation == GEN]['offspring_e'].values
row = lim.loc[GEN + 1]
mu, sd, L = row['mean_prev'], row['std_prev'], row['limit']
w = 0.02
bins = L + w * np.arange(np.floor((e.min() - L) / w) - 1, np.ceil((e.max() - L) / w) + 2)
_, edges, patches = ax.hist(e, bins=bins, density=True, color=LIGHT, edgecolor='white', lw=0.4)
for p, right in zip(patches, edges[1:]):
    if right <= L + 1e-9:
        p.set_facecolor(OLIVE)
x = np.linspace(bins[0], mu + 4 * sd, 400)
ax.plot(x, norm.pdf(x, mu, sd), c='k', lw=1.1)
xs = x[x <= L]
ax.fill_between(xs, norm.pdf(xs, mu, sd), facecolor='none', edgecolor='k', hatch='////', lw=0)
ax.axvline(L, c='k', ls='--', lw=0.9)
ax.set_xlim(bins[0], mu + 3.5 * sd)
ax.set_xlabel('energy (eV / 42-atom cell)')
ax.set_ylabel('density')
ax.set_title(f'(b) generation {GEN}: data vs normal model', loc='left')
ax.legend(handles=[Patch(color=OLIVE, label=f'below $L_{{{GEN + 1}}}$: {100 * (e <= L).mean():.0f} % of data'),
                   Patch(color=LIGHT, label='above $L$ (data)'),
                   Patch(facecolor='none', edgecolor='k', hatch='////', lw=0, label='below $L$: 30 % of normal'),
                   Line2D([], [], c='k', lw=1.1, label=r'normal, same $\mu,\sigma$')],
          loc='upper right', frameon=False, handlelength=1.2)
fig.tight_layout(w_pad=1.0)
save(fig, 'fig_labels')

# ------------------------------------------------------------------ Fig. 3: results (group split)
gnn = {}
for f in glob.glob('results/runs/*/metrics.json'):
    name = os.path.basename(os.path.dirname(f))
    exp = name.rsplit('_s', 1)[0] if name[-3:-1] == '_s' else name
    m = json.load(open(f))
    gnn.setdefault(exp, {'auc': [], 'cm': 0})
    gnn[exp]['auc'].append(m['test']['roc_auc'])
    gnn[exp]['cm'] = gnn[exp]['cm'] + np.array(m['test']['confusion'])

g = base['group']
models = [('majority (always bad)', g['majority']['roc_auc'], LIGHT),
          ('pair-CGCNN, structure', (np.mean(gnn['G_concat_struct_group']['auc']), np.std(gnn['G_concat_struct_group']['auc'])), OLIVE),
          ('pair-CGCNN + energies', (np.mean(gnn['H_concat_energy_group']['auc']), np.std(gnn['H_concat_energy_group']['auc'])), OLIVE),
          ('LogReg, 2 energies', g['LogReg energy']['roc_auc'], GREY),
          ('random forest', g['RandomForest energy+vol+sym']['roc_auc'], DARK)]
fig = plt.figure(figsize=(W, 2.35))
gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1, 1], wspace=0.55)
ax = fig.add_subplot(gs[0])
yy = np.arange(len(models))[::-1]
for y, (nm, (m, s), c) in zip(yy, models):
    ax.barh(y, m, xerr=s if s > 0 else None, color=c, height=0.65, error_kw=dict(lw=0.8, capsize=2))
    ax.text(m + max(s, 0) + 0.015, y, f'{m:.2f}', va='center', fontsize=8)
ax.set_yticks(yy, [m[0] for m in models])
ax.set_xlim(0.4, 1.0)
ax.axvline(0.5, c=GREY, lw=0.6, ls=':')
ax.set_xlabel('test ROC-AUC (group split, 3 seeds)')
ax.set_title('(a) ROC-AUC', loc='left')


def confusion(ax, cm, title):
    norm_ = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm_, cmap='Greens', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]}\n{100 * norm_[i, j]:.0f} %', ha='center', va='center', fontsize=8,
                    color='white' if norm_[i, j] > 0.6 else 'black')
    ax.set_xticks([0, 1], ['bad', 'good'])
    ax.set_yticks([0, 1], ['bad', 'good'])
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(title, loc='left')
    for s in ax.spines.values():
        s.set_visible(False)


confusion(fig.add_subplot(gs[1]), gnn['G_concat_struct_group']['cm'], '(b) pair-CGCNN')
confusion(fig.add_subplot(gs[2]), np.array(base['group_confusion']['RandomForest energy+vol+sym']), '(c) random forest')
save(fig, 'fig_results')

# ------------------------------------------------------------------ appendix: learning curves
runs = [('A_concat_struct_s0', 'structure only (random split)', BLUE),
        ('C_concat_energy_s0', '+ parent energies (random split)', ORANGE),
        ('G_concat_struct_group_s0', 'structure only (group split)', GREEN)]
fig, axs = plt.subplots(1, 2, figsize=(W, 2.5))
for run, label, c in runs:
    h = pd.read_csv(f'results/runs/{run}/history.csv')
    axs[0].plot(h.epoch, h.train_loss, c=c, lw=1.1)
    axs[0].plot(h.epoch, h.val_loss, c=c, lw=0.9, ls='--')
    axs[1].plot(h.epoch, h.val_auc, c=c, lw=1.0, label=label)
axs[0].set_ylim(0.55, 1.25)
axs[0].set_xlabel('epoch'); axs[0].set_ylabel('weighted BCE loss')
axs[0].legend(handles=[Line2D([], [], c='k', lw=1.1, label='train'),
                       Line2D([], [], c='k', lw=0.9, ls='--', label='validation')], loc='upper left', frameon=False)
axs[0].set_title('(a) loss', loc='left')
axs[1].set_xlabel('epoch'); axs[1].set_ylabel('validation ROC-AUC')
axs[1].legend(loc='lower right', frameon=False)
axs[1].set_title('(b) validation ROC-AUC', loc='left')
fig.tight_layout()
save(fig, 'figA_learning_curves')

# ------------------------------------------------------------------ appendix: class balance + parent z
dup = np.isclose(lab.offspring_e, lab.e1, atol=1e-3) | np.isclose(lab.offspring_e, lab.e2, atol=1e-3)
gg = pd.DataFrame({'gen': lab.generation, 'good': lab.label, 'copy': dup}).groupby('gen').mean()
fig, axs = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={'width_ratios': [1.5, 1]})
ax = axs[0]
ax.plot(gg.index, gg.good, 'o-', ms=2.5, lw=1, c=OLIVE, label='good children')
ax.plot(gg.index, gg['copy'], 's-', ms=2.5, lw=1, c=DARK, label='relaxed into a parent')
ax.axhline(lab.label.mean(), c=OLIVE, ls=':', lw=0.8)
ax.set_ylim(0, 1.12); ax.set_xlabel('generation'); ax.set_ylabel('fraction')
ax.legend(loc='upper center', ncol=2, frameon=False, handlelength=1.5, columnspacing=1.0)
ax.set_title('(a) per generation', loc='left')
ax = axs[1]
zmin, zmax = lab[['z1', 'z2']].min(axis=1), lab[['z1', 'z2']].max(axis=1)
for lbl, c in ((0, DARK), (1, OLIVE)):
    m = lab.label == lbl
    ax.scatter(zmin[m], zmax[m], s=4, c=c, alpha=0.6, lw=0, label='good' if lbl else 'bad')
ax.axvline(Z_30, c='k', lw=0.6, ls='--'); ax.axhline(Z_30, c='k', lw=0.6, ls='--')
ax.set_xlabel(r'$z_{\min}$ (better parent)'); ax.set_ylabel(r'$z_{\max}$ (worse parent)')
ax.legend(loc='upper left', frameon=False, markerscale=2.5)
ax.set_title('(b) parent energies vs label', loc='left')
fig.tight_layout()
save(fig, 'figA_labels_extra')

# ------------------------------------------------------------------ appendix: all confusion matrices
panels = [('random', 'A_concat_struct', 'C_concat_energy'), ('group', 'G_concat_struct_group', 'H_concat_energy_group'),
          ('generation', 'E_concat_struct_gen', 'F_concat_energy_gen')]
fig, axs = plt.subplots(3, 3, figsize=(W * 0.82, 5.6))
for r, (split, a, b) in enumerate(panels):
    cms = [gnn[a]['cm'], gnn[b]['cm'], np.array(base[f'{split}_confusion']['RandomForest energy+vol+sym'])]
    for c, (cm, t) in enumerate(zip(cms, ['pair-CGCNN', 'pair-CGCNN + energies', 'random forest'])):
        confusion(axs[r, c], cm, t if r == 0 else '')
        if c == 0:
            axs[r, c].set_ylabel(f'{split} split\ntrue')
        else:
            axs[r, c].set_ylabel('')
        if r < 2:
            axs[r, c].set_xlabel('')
fig.tight_layout()
save(fig, 'figA_confusion_all')
print('figures written to', FIG)
