"""Figure for Task 2: the normal assumption behind L = mu + z*sigma (30 % tail) vs a real generation."""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from offspring.db import read_individuals_df
from offspring.labels import Z_30

GEN = 20  # the population of generation GEN defines the limit L_{GEN+1}
OLIVE, GREY = '#8E9719', '#8C8C8C'

ind = read_individuals_df('data/offspring.db')
e = ind[(ind.natoms == 42) & (ind.generation == GEN)]['offspring_e'].values
lim = pd.read_csv('data/generation_limits.csv', index_col=0).loc[GEN + 1]
mu, sd, L = lim['mean_prev'], lim['std_prev'], lim['limit']
frac = (e <= L).mean()

plt.rcParams.update({'font.size': 10})
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), dpi=200)

# (a) assumed distribution, standardised
ax = axs[0]
z = np.linspace(-3.5, 3.5, 400)
ax.plot(z, norm.pdf(z), c='k', lw=1.5)
zz = z[z <= Z_30]
ax.fill_between(zz, norm.pdf(zz), color=OLIVE, alpha=0.6, label=f'lowest 30 %  (z ≤ {Z_30})')
ax.fill_between(z[z > Z_30], norm.pdf(z[z > Z_30]), color=GREY, alpha=0.25, label='other 70 %')
ax.axvline(Z_30, c='k', ls='--', lw=1)
ax.text(Z_30 - 0.1, 0.41, r'$L=\mu-0.5244\,\sigma$', ha='right', fontsize=9)
ax.text(-1.45, 0.07, '30 %', fontsize=13, weight='bold', ha='center')
ax.text(0.9, 0.12, '70 %', fontsize=13, weight='bold', ha='center')
ax.set_xlabel(r'$z=(E-\mu)/\sigma$')
ax.set_ylabel('probability density')
ax.set_ylim(0, 0.45)
ax.set_title('(a) assumed: normal energies', fontsize=10)
ax.legend(fontsize=8, loc='upper right', frameon=False)

# (b) a real generation
ax = axs[1]
w = 0.02  # bin width (eV); edges aligned so that L is a bin edge
bins = L + w * np.arange(np.floor((e.min() - L) / w) - 1, np.ceil((e.max() - L) / w) + 2)
counts, edges, patches = ax.hist(e, bins=bins, density=True, color=GREY, alpha=0.45, edgecolor='white')
for p, right in zip(patches, edges[1:]):
    if right <= L + 1e-9:
        p.set_facecolor(OLIVE); p.set_alpha(0.75)
x = np.linspace(bins[0], mu + 4 * sd, 400)
ax.plot(x, norm.pdf(x, mu, sd), c='k', lw=1.5, label=r'normal with same $\mu,\sigma$')
ax.axvline(L, c='k', ls='--', lw=1)
ax.axvline(mu, c=GREY, ls=':', lw=1)
ax.text(mu, ax.get_ylim()[1] * 0.93, r' $\mu$', color=GREY)
ax.text(L, ax.get_ylim()[1] * 0.93, rf' $L_{{{GEN + 1}}}$', color='k')
ax.text(0.97, 0.45, f'below $L$:\nnormal model 30 %\nthis generation {100 * frac:.0f} %',
        transform=ax.transAxes, ha='right', fontsize=9)
ax.set_xlabel('energy, eV per 42-atom cell')
ax.set_title(f'(b) actual: generation {GEN} (skewed)', fontsize=10)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[Patch(color=OLIVE, alpha=0.75, label=f'below $L$ (n = {(e <= L).sum()})'),
                   Patch(color=GREY, alpha=0.45, label=f'above $L$ (n = {(e > L).sum()})'),
                   Line2D([], [], c='k', lw=1.5, label=r'normal with same $\mu,\sigma$')],
          fontsize=8, loc='upper right', frameon=False, title=f'generation {GEN}, n = {len(e)}', title_fontsize=8)
ax.set_xlim(bins[0], mu + 4 * sd)

fig.tight_layout()
fig.savefig('report/figures/limit_distribution.pdf')
print(f'gen {GEN}: n={len(e)} mu={mu:.3f} sd={sd:.3f} L={L:.3f} below={frac:.3f}')
