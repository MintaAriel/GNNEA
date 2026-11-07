"""Cluster the parent structures by their crystal RDF (MXtalTools analysis.crystal_rdf, ported in offspring/mxtal.py).

1. elementwise intermolecular RDF of every parent (H/C/N/O pair channels, 0-10 A, 100 bins)
2. all-pairs RDF earth-mover distance (compute_rdf_distance)
3. average-linkage hierarchical clusters at a distance threshold, and energy-ordered "anchors"
   (common.clustering.greedy_bottom_up_anchors): distinct low-energy packings
4. how the clusters relate to the offspring labels: good rate per parent cluster, same- vs
   different-cluster parents, and the parent-parent RDF distance as a one-feature predictor

    python scripts/10_parent_clusters.py [--d-cut 0.05] [--all-individuals]

Outputs: data/parent_rdf.pt (RDFs + distance matrix), data/parent_clusters.csv,
         data/pair_rdf_distance.csv, results/parent_clusters.json, report/figures/parent_clusters.{png,pdf}
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from offspring.data import load_labelled
from offspring.db import read_individuals_df
from offspring.mxtal import build_cluster, crystal_rdf, element_pairs, greedy_anchors, rdf_distance_matrix

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='data/offspring.db')
    ap.add_argument('--all-individuals', action='store_true',
                    help='cluster every 42-atom individual, not only the parents of labelled pairs')
    ap.add_argument('--rmax', type=float, default=10.0)
    ap.add_argument('--bins', type=int, default=100)
    ap.add_argument('--d-cut', type=float, default=None,
                    help='cluster / anchor distance threshold (default: 10th percentile of all distances)')
    ap.add_argument('--fig', default='report/figures/parent_clusters')
    a = ap.parse_args()

    lab = load_labelled(a.db)
    ind = read_individuals_df(a.db)
    ind = ind[ind.natoms == 42]
    if not a.all_individuals:
        parents = set(lab.p1_id.astype(int)) | set(lab.p2_id.astype(int))
        ind = ind[ind.uspex_id.isin(parents)]
    ind = ind.reset_index(drop=True)
    print(f'{len(ind)} structures')

    # ---- 1. RDFs
    pairs = element_pairs({int(z) for at in ind.atoms for z in at.numbers})
    rr = torch.linspace(0, a.rmax, a.bins)
    t0, rdfs, n_mol = time.time(), [], []
    for k, at in enumerate(ind.atoms):
        c = build_cluster(at, cutoff=a.rmax, max_num_neighbors=10_000)
        rdfs.append(crystal_rdf(c, pairs, rrange=(0, a.rmax), bins=a.bins))
        n_mol.append(c['n_mol'])
        if (k + 1) % 250 == 0:
            print(f'  rdf {k + 1}/{len(ind)}  ({time.time() - t0:.0f}s)', flush=True)
    rdfs = torch.stack(rdfs)
    ind['n_mol'] = n_mol
    if (ind.n_mol != 2).any():
        print(f'WARNING: {(ind.n_mol != 2).sum()} cells do not hold 2 whole molecules (n_mol != 2)')

    # ---- 2. distance matrix
    D = rdf_distance_matrix(rdfs, rr).numpy()
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0)
    torch.save(dict(uspex_id=ind.uspex_id.values, pairs=pairs, rr=rr, rdf=rdfs, D=torch.tensor(D)),
               os.path.join(os.path.dirname(a.db), 'parent_rdf.pt'))
    off = D[np.triu_indices(len(D), 1)]
    d_cut = a.d_cut if a.d_cut is not None else float(np.quantile(off, 0.10))
    print(f'RDF distance: median {np.median(off):.4f}, 10th pct {np.quantile(off, .1):.4f}, d_cut = {d_cut:.4f}')

    # ---- 3. clusters and anchors
    Z = linkage(squareform(D, checks=False), method='average')
    ind['cluster'] = fcluster(Z, t=d_cut, criterion='distance')
    energy = ind.offspring_e.values
    anchors, nearest = greedy_anchors(D, energy, d_cut)
    ind['anchor'] = ind.uspex_id.values[nearest]
    ind['is_anchor'] = np.isin(np.arange(len(ind)), anchors)
    # order cluster ids by size
    sizes = ind.cluster.value_counts()
    ind['cluster'] = ind.cluster.map({c: i for i, c in enumerate(sizes.index)})
    xy = TSNE(n_components=2, metric='precomputed', init='random', random_state=0,
              perplexity=min(30, len(ind) - 1)).fit_transform(D)
    ind['tsne_x'], ind['tsne_y'] = xy[:, 0], xy[:, 1]
    cols = ['uspex_id', 'generation', 'operator', 'offspring_e', 'cluster', 'anchor', 'is_anchor', 'n_mol',
            'tsne_x', 'tsne_y']
    ind[cols].to_csv('data/parent_clusters.csv', index=False)

    # ---- 4. relation to the offspring labels
    pos = {u: i for i, u in enumerate(ind.uspex_id)}
    cl = dict(zip(ind.uspex_id, ind.cluster))
    pl = lab[['offspring_id', 'generation', 'p1_id', 'p2_id', 'label']].copy()
    pl = pl[pl.p1_id.isin(pos) & pl.p2_id.isin(pos)]
    pl['rdf_dist'] = [D[pos[int(i)], pos[int(j)]] for i, j in zip(pl.p1_id, pl.p2_id)]
    pl['c1'] = pl.p1_id.astype(int).map(cl)
    pl['c2'] = pl.p2_id.astype(int).map(cl)
    pl['same_cluster'] = pl.c1 == pl.c2
    pl.to_csv('data/pair_rdf_distance.csv', index=False)

    long = pd.concat([pl[['c1', 'label']].rename(columns={'c1': 'cluster'}),
                      pl[['c2', 'label']].rename(columns={'c2': 'cluster'})])
    per_cluster = long.groupby('cluster')['label'].agg(['mean', 'count']).sort_values('count', ascending=False)
    auc = roc_auc_score(pl.label, -pl.rdf_dist)  # similar parents -> good offspring?
    summary = dict(
        n_structures=len(ind), d_cut=d_cut, n_clusters=int(ind.cluster.nunique()), n_anchors=int(len(anchors)),
        largest_clusters=sizes.head(10).tolist(),
        good_rate_same_cluster=float(pl[pl.same_cluster].label.mean()), n_same_cluster=int(pl.same_cluster.sum()),
        good_rate_diff_cluster=float(pl[~pl.same_cluster].label.mean()), n_diff_cluster=int((~pl.same_cluster).sum()),
        auc_minus_parent_rdf_distance=float(auc),
        good_rate_per_parent_cluster_top10={int(k): dict(rate=float(v['mean']), n=int(v['count']))
                                            for k, v in per_cluster.head(10).iterrows()})
    os.makedirs('results', exist_ok=True)
    with open('results/parent_clusters.json', 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))

    # ---- figure: t-SNE coloured by cluster / energy / generation, distance vs label
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    top = ind.cluster < 10
    ax[0].scatter(xy[~top, 0], xy[~top, 1], s=4, c='lightgrey')
    sc = ax[0].scatter(xy[top, 0], xy[top, 1], s=6, c=ind.cluster[top], cmap='tab10', vmin=0, vmax=9)
    ax[0].scatter(xy[anchors, 0], xy[anchors, 1], s=25, facecolors='none', edgecolors='k', lw=0.6)
    ax[0].set_title('(a) RDF clusters (10 largest, o = anchors)', fontsize=9)
    sc = ax[1].scatter(xy[:, 0], xy[:, 1], s=5, c=energy, cmap='viridis_r')
    fig.colorbar(sc, ax=ax[1], label='energy')
    ax[1].set_title('(b) parent energy', fontsize=9)
    sc = ax[2].scatter(xy[:, 0], xy[:, 1], s=5, c=ind.generation, cmap='plasma')
    fig.colorbar(sc, ax=ax[2], label='generation')
    ax[2].set_title('(c) generation', fontsize=9)
    bins = np.linspace(0, np.quantile(pl.rdf_dist, .99), 40)
    for lbl, name in ((0, 'bad'), (1, 'good')):
        ax[3].hist(pl.rdf_dist[pl.label == lbl], bins=bins, alpha=.6, density=True, label=name)
    ax[3].set_xlabel('RDF distance between the two parents')
    ax[3].set_title(f'(d) label vs parent distance (AUC {auc:.3f})', fontsize=9)
    ax[3].legend()
    for x in ax[:3]:
        x.set_xticks([]), x.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.fig), exist_ok=True)
    fig.savefig(a.fig + '.png', dpi=200)
    fig.savefig(a.fig + '.pdf')
    print('figure ->', a.fig + '.png')
