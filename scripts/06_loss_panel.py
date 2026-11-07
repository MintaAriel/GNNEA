"""Standalone version of the report's Figure 4, left panel (train/validation loss), with a legend."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

runs = [('A_concat_struct_s0', 'concat-pool, structure only', 'tab:blue'),
        ('C_concat_energy_s0', 'concat-pool + parent energies', 'tab:orange'),
        ('G_concat_struct_group_s0', 'concat-pool, group split', 'tab:green')]

plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
for run, label, c in runs:
    h = pd.read_csv(f'results/runs/{run}/history.csv')
    ax.plot(h.epoch, h.train_loss, c=c, lw=1.6)
    ax.plot(h.epoch, h.val_loss, c=c, lw=1.2, ls='--')
ax.set_xlabel('epoch')
ax.set_ylabel('weighted BCE loss')
ax.grid(alpha=0.25)
ax.set_ylim(0.55, 1.45)  # headroom so the legend does not cover the validation spikes
run_handles = [Line2D([], [], c=c, lw=2, label=label) for _, label, c in runs]
style_handles = [Line2D([], [], c='k', lw=1.6, label='train'), Line2D([], [], c='k', lw=1.2, ls='--', label='validation')]
leg1 = ax.legend(handles=run_handles, loc='upper right', fontsize=9, title='run (seed 0)', title_fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=style_handles, loc='lower left', fontsize=9)
fig.tight_layout()
fig.savefig('learning_curve_loss.png')
fig.savefig('learning_curve_loss.pdf')
print('saved learning_curve_loss.png / .pdf')
