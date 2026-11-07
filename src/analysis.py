"""Matched test-set comparisons, cluster bootstrap intervals and exportable figures."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from offspring.data import load_labelled, parent_pair_groups
from src.config import project_path

METRICS = ('roc_auc', 'balanced_accuracy', 'average_precision', 'accuracy', 'precision', 'recall', 'f1')


def paired_bootstrap(y, baseline, offspring, groups, samples, seed, confidence):
    """Resample whole parent-pair groups, pairing both predictions in every replicate."""
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    deltas = []
    for _ in range(samples):
        idx = np.concatenate([members[int(i)] for i in rng.integers(len(members), size=len(members))])
        if np.unique(y[idx]).size == 2:
            deltas.append(roc_auc_score(y[idx], offspring[idx]) - roc_auc_score(y[idx], baseline[idx]))
    if not deltas:
        raise ValueError('No two-class bootstrap replicates; test sample is too small.')
    alpha = (1 - confidence) / 2
    return np.quantile(deltas, [alpha, 1 - alpha]).tolist()


def analyze(config, run_paths):
    out = project_path(config['experiments']['output_dir'])
    records, by_key = [], {}
    for path in run_paths:
        result = json.loads((path / 'metrics.json').read_text())
        c = result['config']
        key = (result['encoder'], c['fusion'], c['split'], c['seed'])
        by_key.setdefault(key, {})[c['offspring_structure']] = (path, result)
        records.append(dict(encoder=key[0], fusion=key[1], split=key[2], seed=key[3],
                            offspring_structure=c['offspring_structure'], best_epoch=result['best_epoch'],
                            offspring_geometry=result['provenance']['offspring_geometry'],
                            epochs_completed=result['epochs_completed'], n_params=result['n_params'],
                            train_time_s=result['train_time_s'], n_test=result['n_test'], **result['test']))
    pd.DataFrame(records).to_csv(out / 'runs.csv', index=False)
    ds, a = config['dataset'], config['analysis']
    lab = load_labelled(project_path(ds['db']), natoms=ds['natoms'], z=ds['z'], operators=ds['operators'], structures=False)
    lab['parent_group'] = parent_pair_groups(lab)
    group_lookup = lab.set_index('offspring_id')['parent_group']
    counts = lab.groupby('parent_group').label.agg(['sum', 'count'])
    conflicting = counts.index[(counts['sum'] > 0) & (counts['sum'] < counts['count'])]
    pairs, changes, reuse_rows = [], [], []
    for key, variants in by_key.items():
        base_path, base = variants[False]
        child_path, child = variants[True]
        pd.testing.assert_frame_equal(pd.read_csv(base_path / 'split_manifest.csv'),
                                      pd.read_csv(child_path / 'split_manifest.csv'))
        pred = pd.read_csv(base_path / 'test_predictions.csv').merge(
            pd.read_csv(child_path / 'test_predictions.csv'), on='offspring_id', suffixes=('_parents', '_offspring'),
            validate='one_to_one')
        if len(pred) != base['n_test'] or not np.array_equal(pred.y_parents, pred.y_offspring):
            raise ValueError('Baseline and offspring test IDs/labels differ.')
        groups = pred.offspring_id.map(group_lookup).to_numpy()
        manifest = pd.read_csv(base_path / 'split_manifest.csv')
        training_parent_ids = set(manifest.loc[manifest.subset == 'train', 'p1_id']) | set(
            manifest.loc[manifest.subset == 'train', 'p2_id'])
        training_child_ids = set(manifest.loc[manifest.subset == 'train', 'offspring_id'])
        pred['child_unrelaxed_id_seen_in_training'] = pred.offspring_id.isin(training_child_ids)
        pred['child_relaxed_id_seen_as_training_parent'] = pred.offspring_id.isin(training_parent_ids)
        lo, hi = paired_bootstrap(pred.y_parents.to_numpy(), pred.p_good_parents.to_numpy(),
            pred.p_good_offspring.to_numpy(), groups, a['bootstrap_samples'], a['seed'], a['confidence'])
        row = dict(encoder=key[0], fusion=key[1], split=key[2], seed=key[3], n_test=len(pred),
                   offspring_geometry=ds['offspring_geometry'],
                   auc_delta_ci_low=lo, auc_delta_ci_high=hi)
        for metric in METRICS:
            row[f'parents_{metric}'] = base['test'][metric]
            row[f'offspring_{metric}'] = child['test'][metric]
            row[f'delta_{metric}'] = child['test'][metric] - base['test'][metric]
        threshold = config['training']['threshold']
        pred['parents_correct'] = (pred.p_good_parents >= threshold).astype(int) == pred.y_parents
        pred['offspring_correct'] = (pred.p_good_offspring >= threshold).astype(int) == pred.y_parents
        pred['conflicting_parent_group'] = np.isin(groups, conflicting)
        pred['change'] = np.select([~pred.parents_correct & pred.offspring_correct,
                                   pred.parents_correct & ~pred.offspring_correct], ['corrected', 'regressed'], 'unchanged')
        row['corrected'] = int((pred.change == 'corrected').sum())
        row['regressed'] = int((pred.change == 'regressed').sum())
        for seen, name in ((False, 'child_id_unseen_as_relaxed_parent'), (True, 'child_id_seen_as_relaxed_parent')):
            subset = pred[pred.child_relaxed_id_seen_as_training_parent == seen]
            reuse = dict(encoder=key[0], fusion=key[1], split=key[2], seed=key[3], subset=name, n=len(subset),
                         offspring_geometry=ds['offspring_geometry'])
            for prefix in ('parents', 'offspring'):
                reuse[f'{prefix}_auc'] = (roc_auc_score(subset.y_parents, subset[f'p_good_{prefix}'])
                                          if subset.y_parents.nunique() == 2 else None)
                reuse[f'{prefix}_accuracy'] = float(subset[f'{prefix}_correct'].mean()) if len(subset) else None
            reuse_rows.append(reuse)
        for flag, name in ((True, 'conflicting'), (False, 'consistent')):
            subset = pred[pred.conflicting_parent_group == flag]
            row[f'{name}_n'] = len(subset)
            row[f'{name}_accuracy_delta'] = (float(subset.offspring_correct.mean() - subset.parents_correct.mean())
                                             if len(subset) else None)
        pairs.append(row)
        pred['encoder'], pred['fusion'], pred['split'], pred['seed'] = key
        pred['offspring_geometry'] = ds['offspring_geometry']
        changes.append(pred)
    paired = pd.DataFrame(pairs)
    paired.to_csv(out / 'paired_deltas.csv', index=False)
    pd.concat(changes, ignore_index=True).to_csv(out / 'prediction_changes.csv', index=False)
    pd.DataFrame(reuse_rows).to_csv(out / 'structure_id_reuse.csv', index=False)
    summary_rows = []
    for key, subset in paired.groupby(['encoder', 'fusion', 'split'], sort=False):
        row = dict(encoder=key[0], fusion=key[1], split=key[2], n_seeds=len(subset),
                   offspring_geometry=ds['offspring_geometry'],
                   positive_auc_seeds=int((subset.delta_roc_auc > 0).sum()))
        for metric in METRICS:
            for prefix in ('parents', 'offspring', 'delta'):
                values = subset[f'{prefix}_{metric}']
                row[f'{prefix}_{metric}_mean'] = values.mean()
                row[f'{prefix}_{metric}_std'] = values.std(ddof=1) if len(values) > 1 else None
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / 'summary.csv', index=False)
    plot_results(config, paired, summary, by_key)
    write_report(config, summary, paired, lab, records)


def plot_results(config, paired, summary, by_key):
    out = project_path(config['experiments']['output_dir'])
    splits = list(config['experiments']['splits'])
    fig, axes = plt.subplots(1, len(splits), figsize=(6.5 * len(splits), 4.8), squeeze=False)
    for ax, split in zip(axes.flat, splits):
        subset = summary[summary.split == split].reset_index(drop=True)
        x = np.arange(len(subset))
        for offset, prefix, label, color in ((-0.18, 'parents', 'Parents', '#6d8199'),
                                              (0.18, 'offspring', 'Parents + offspring', '#008b80')):
            means = subset[f'{prefix}_roc_auc_mean']
            errors = subset[f'{prefix}_roc_auc_std'].fillna(0)
            ax.bar(x + offset, means, width=0.34, yerr=errors, capsize=3, label=label, color=color)
            for i, row in subset.iterrows():
                values = paired[(paired.encoder == row.encoder) & (paired.fusion == row.fusion) & (paired.split == split)]
                ax.scatter(np.full(len(values), i + offset), values[f'{prefix}_roc_auc'], color='black', s=14, zorder=3)
        ax.set_xticks(x, [f"{r.encoder.upper()}\n{r.fusion.replace('_', '-') }" for r in subset.itertuples()])
        ax.set_ylim(0.45, 1.0)
        ax.axhline(0.5, color='grey', linestyle=':', linewidth=1)
        ax.set_ylabel('Test ROC-AUC')
        ax.set_title(f'{split.capitalize()} split')
        ax.legend(fontsize=9)
    fig.suptitle(f"{config['project']['name']}: adding {config['dataset']['offspring_geometry']} offspring structure\n"
                 'Bars: mean; error bars: sample SD across seeds; dots: runs', fontsize=12)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'offspring_comparison.{extension}', dpi=180)
    plt.close(fig)
    encoders = list(dict.fromkeys(key[0] for key in by_key))
    fig, axes = plt.subplots(len(splits), len(encoders), figsize=(5.5 * len(encoders), 4.3 * len(splits)), squeeze=False)
    for row, split in enumerate(splits):
        for col, encoder in enumerate(encoders):
            ax = axes[row, col]
            for key, variants in by_key.items():
                if key[0] != encoder or key[2] != split or key[3] != config['experiments']['splits'][split][0]:
                    continue
                for child, (path, _) in variants.items():
                    pred = pd.read_csv(path / 'test_predictions.csv')
                    fpr, tpr, _ = roc_curve(pred.y, pred.p_good)
                    ax.plot(fpr, tpr, linestyle='-' if child else '--',
                            label=f"{key[1]} {'+ child' if child else 'parents'} ({roc_auc_score(pred.y, pred.p_good):.3f})")
            ax.plot([0, 1], [0, 1], ':', color='grey')
            ax.set(xlabel='False positive rate', ylabel='True positive rate', title=f'{encoder.upper()}, {split}, first seed')
            ax.legend(fontsize=8)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'offspring_roc.{extension}', dpi=180)
    plt.close(fig)


def write_report(config, summary, paired, lab, records):
    out = project_path(config['experiments']['output_dir'])
    lines = [f"{config['project']['name']} offspring structure ablation", '',
        f"Configuration: {config['_config_path']}",
        f"Environment: {config['env']['conda_env']}; physical GPU: {config['runtime']['gpu']}",
        f"Dataset: {len(lab)} labelled Heredity offspring; good fraction {lab.label.mean():.4f}",
        f"Source results: {project_path(config['dataset']['results_dir'])}; USPEX version {config['dataset']['uspex_ver']}",
        f"Unrelaxed offspring source: {project_path(config['dataset']['results_dir']) / config['dataset']['offspring_structures']}",
        f"Parent cell size: {config['dataset']['natoms']} atoms; good = child energy <= mean(previous generation) "
        f"+ ({config['dataset']['z']}) * std(previous generation)",
        f"Completed training runs: {len(records)}", '',
        'Question: does adding offspring geometry improve classification on matched held-out samples?',
        'Only atomic numbers, positions and cells are encoded. Offspring energy is used for labels, never as a model input.',
        f"Parent-energy features enabled: {config['training']['energy_features']} (controlled in sirena.yml).", '',
        'concat_pool: concatenate node embeddings of the two parents and (when enabled) child, then pool.',
        'With shared weights, concat_pool is symmetric across all three structures and loses the child role.',
        'pool_concat: append the pooled child embedding to [g1+g2, abs(g1-g2), g1*g2].',
        'pool_concat retains the distinction between parent-pair features and child features.',
        'The encoder is shared across structures by default. Parent order remains symmetric.',
        'All paired runs have identical train/validation/test IDs and training sampler seeds.',
        'Checkpoint selection and early stopping use validation loss only; test data never select a checkpoint.',
        'MXtal normalization is fit/recalibrated on training data only.', '',
        'Test ROC-AUC (mean +/- sample SD where multiple seeds are configured):']
    for row in summary.itertuples():
        base_sd = f' +/- {row.parents_roc_auc_std:.3f}' if row.n_seeds > 1 else ''
        child_sd = f' +/- {row.offspring_roc_auc_std:.3f}' if row.n_seeds > 1 else ''
        lines.append(f'{row.encoder:6s} {row.fusion:12s} {row.split:10s}: '
                     f'{row.parents_roc_auc_mean:.3f}{base_sd} -> {row.offspring_roc_auc_mean:.3f}{child_sd}; '
                     f'delta {row.delta_roc_auc_mean:+.3f}; improves in {row.positive_auc_seeds}/{row.n_seeds} seeds; '
                     f'balanced accuracy {row.parents_balanced_accuracy_mean:.3f} -> {row.offspring_balanced_accuracy_mean:.3f}')
    lines += ['', 'How predictions change (each matched seed):']
    for row in paired.itertuples():
        lines.append(f'{row.encoder} {row.fusion} {row.split} seed={row.seed}: '
                     f'corrected {row.corrected}, regressed {row.regressed} of {row.n_test}; '
                     f'AUC delta {row.delta_roc_auc:+.3f}, paired parent-group bootstrap '
                     f"{config['analysis']['confidence']:.0%} interval [{row.auc_delta_ci_low:+.3f}, {row.auc_delta_ci_high:+.3f}]")
    lines += ['', 'Interpretation and limits:',
        'Child geometry comes exclusively from the configured unrelaxed POSCAR collection. No relaxed child geometry is used as an offspring input.',
        'Parent geometry and target energies are relaxed. This tests whether unrelaxed candidate geometry predicts the later relaxation outcome.',
        'Graph caches distinguish unrelaxed child inputs from relaxed parents even when they share a USPEX ID.',
        'The group split groups unordered parent energies rounded to 3 decimals, following the previous report; it is not a fully structure-disjoint split.',
        'The generation split holds out later generations but can revisit the same packings. Neither split proves transfer to a new USPEX run or molecule.',
        'structure_id_reuse.csv separates child IDs seen/unseen as relaxed training parents. A shared ID here does not imply the same geometry state.',
        'prediction_changes.csv separately checks whether an unrelaxed test child ID occurs among unrelaxed training children. Different IDs can still represent copied packings.',
        'Bootstrap intervals resample parent-pair groups within each fixed test set; they do not measure uncertainty across USPEX searches.',
        'The conflicting-parent-group subsets in paired_deltas.csv are descriptive: groups producing both labels in this dataset.',
        'concat_pool keeps the same parameter count with the child; pool_concat has a wider head. Changes in parameter counts are recorded in runs.csv.',
        'A small seed count and one molecular search limit broad claims, even when observed deltas are positive.', '',
        f'Full metrics and paired uncertainty: {out / "paired_deltas.csv"}',
        f'Per-offspring corrections/regressions: {out / "prediction_changes.csv"}',
        f'Figure: {out / "offspring_comparison.png"}', f'ROC curves: {out / "offspring_roc.png"}']
    report = '\n'.join(lines) + '\n'
    (out / 'report.txt').write_text(report)
    report_path = project_path(config['analysis']['report_path'])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
