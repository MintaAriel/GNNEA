"""One training/evaluation implementation for both encoders and both input ablations."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
import torch
from torch_geometric.loader import DataLoader
import yaml
from ase.db import connect

from offspring.data import ENERGY_FEATURES, load_labelled, split_indices
from offspring.graphs import make_pair_dataset
from offspring.model import PairCGCNN
from offspring.mxtal import DEFAULT_ATOM_FEATURES, PairMXtal, atom_feature_stats, make_pair_cluster_dataset
from src.config import ROOT, config_from_cli, project_path

_DATASETS = {}


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_digest():
    paths = sorted((ROOT / 'offspring').glob('*.py')) + [Path(__file__), ROOT / 'src/config.py']
    return hashlib.sha256(''.join(file_digest(path) for path in paths).encode()).hexdigest()


def fingerprint(config, encoder):
    return hashlib.sha256(json.dumps(dict(config=config, encoder=encoder,
        database_sha256=file_digest(project_path(config['dataset']['db'])),
        source_sha256=source_digest()), sort_keys=True).encode()).hexdigest()


def train(model, iterator, optimizer, criterion, device='cpu'):
    model.train()
    loss_sum, seen = 0., 0
    for batch in iterator:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch), batch.y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * batch.y.size(0)
        seen += batch.y.size(0)
    return loss_sum / seen


@torch.no_grad()
def evaluate(model, iterator, criterion, device='cpu'):
    model.eval()
    loss_sum, probs, ys, ids = 0., [], [], []
    for batch in iterator:
        batch = batch.to(device)
        logits = model(batch)
        loss_sum += criterion(logits, batch.y).item() * batch.y.size(0)
        probs.append(torch.sigmoid(logits).cpu())
        ys.append(batch.y.cpu())
        ids.append(batch.offspring_id.cpu())
    return (loss_sum / len(iterator.dataset), torch.cat(probs).numpy(),
            torch.cat(ys).numpy(), torch.cat(ids).numpy())


def metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    return dict(accuracy=float(accuracy_score(y, pred)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)), f1=float(f1_score(y, pred, zero_division=0)),
                roc_auc=float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
                average_precision=float(average_precision_score(y, p)),
                confusion=confusion_matrix(y, pred, labels=[0, 1]).tolist(), threshold=threshold)


@torch.no_grad()
def recalibrate_bn(model, loader, device='cpu'):
    """Recompute MXtal BatchNorm statistics on training data, preserving each layer's momentum."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in bns]
    model.eval()
    for module in bns:
        module.reset_running_stats()
        module.momentum = None
        module.train()
    if bns:
        for batch in loader:
            model(batch.to(device))
    for module, momentum in zip(bns, momenta):
        module.momentum = momentum
    model.eval()


def prepare_data(config, encoder):
    ds, model_cfg, train_cfg = config['dataset'], config[encoder], config['training']
    metadata = connect(project_path(ds['db'])).metadata
    geometry = metadata.get('offspring_geometry')
    if geometry != 'unrelaxed' or geometry != ds['offspring_geometry']:
        raise ValueError(f"Configured child geometry {ds['offspring_geometry']} differs from database {geometry}.")
    source = project_path(ds['results_dir']) / ds['offspring_structures']
    if metadata.get('offspring_structures_sha256') != file_digest(source):
        raise ValueError('Unrelaxed offspring source differs from the database provenance; rebuild the database.')
    db_hash = file_digest(project_path(ds['db']))
    key = json.dumps([db_hash, encoder, model_cfg, ds, train_cfg['energy_features'],
                      train_cfg['offspring_structure']], sort_keys=True)
    if key in _DATASETS:
        return _DATASETS[key]
    lab = load_labelled(project_path(ds['db']), natoms=ds['natoms'], z=ds['z'], operators=ds['operators'])
    if lab.offspring.isna().any():
        raise ValueError('The database lacks unrelaxed geometry for some labelled offspring.')
    gcols = ENERGY_FEATURES if train_cfg['energy_features'] else ()
    child = train_cfg['offspring_structure']
    cache_dir = project_path(ds['cache_dir']) / db_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    if encoder == 'cgcnn':
        data = make_pair_dataset(lab, model_cfg['r_cut'], model_cfg['max_neighbors'], gcols,
                                 offspring_structure=child)
        cache = None
    else:
        inside = model_cfg.get('inside', 'cell')
        suffix = '' if inside == 'cell' else f'_{inside}'          # keep the existing whole-cell caches valid
        cache_path = cache_dir / f"mxtal_c{model_cfg['cutoff']:g}_n{model_cfg['max_neighbors']}{suffix}.pt"
        cache = torch.load(cache_path, weights_only=False) if cache_path.exists() else {}
        n_before = len(cache)
        data, cache = make_pair_cluster_dataset(lab, model_cfg['cutoff'], model_cfg['max_neighbors'],
                                               gcols, cache=cache, offspring_structure=child, inside=inside)
        if len(cache) != n_before:
            torch.save(cache, cache_path)
    _DATASETS[key] = lab, data, cache
    return lab, data, cache


def make_model(config, encoder, lab, tr, cache):
    t, c = config['training'], config[encoder]
    common = dict(fusion=t['fusion'], shared_encoder=t['shared_encoder'],
                  n_global=len(ENERGY_FEATURES) if t['energy_features'] else 0,
                  h_fea_len=t['h_fea'], dropout=t['dropout'], offspring_structure=t['offspring_structure'])
    if encoder == 'cgcnn':
        columns = ('p1', 'p2', 'offspring') if t['offspring_structure'] else ('p1', 'p2')
        max_z = max(int(atoms.numbers.max()) for column in columns for atoms in lab[column])
        return PairCGCNN(n_species=max_z + 1, node_fea_len=c['node_fea'], edge_fea_len=c['edge_fea'],
                         n_conv=c['n_conv'], r_cut=c['r_cut'], **common)
    ids = {('relaxed', int(i)) for col in ('p1_id', 'p2_id') for i in lab[col].iloc[tr]}
    if t['offspring_structure']:
        ids.update(('unrelaxed', int(i)) for i in lab.offspring_id.iloc[tr])
    z_train = torch.cat([cache[i]['z'][:cache[i]['n_inside']] for i in sorted(ids)]).numpy()
    architecture = argparse.Namespace(activation=c['activation'], graph=argparse.Namespace(
        node_dim=c['node_dim'], message_dim=c['message_dim'], embedding_dim=c['embedding_dim'],
        num_convs=c['num_convs'], fcs_per_gc=c['fcs_per_gc'], num_radial=c['num_radial'],
        num_input_classes=c['num_input_classes'], norm=None if c['graph_norm'] == 'none' else c['graph_norm'],
        dropout=c['gnn_dropout'], atom_type_embedding_dim=c['atom_type_embedding_dim'],
        radial_embedding=c['radial_embedding'], cutoff=c['cutoff'], max_num_neighbors=c['max_neighbors'],
        envelope_exponent=c['envelope_exponent']), fc=argparse.Namespace(
        hidden_dim=c['fc_dim'], num_layers=c['fc_layers'], dropout=c['gnn_dropout'], norm=c['fc_norm']))
    return PairMXtal(architecture, DEFAULT_ATOM_FEATURES, atom_feature_stats(z_train),
                     edge_type=c['edge_type'], crystal_norm=c['crystal_norm'], **common)


def run_training(config, encoder, out):
    t, runtime = config['training'], config['runtime']
    torch.set_num_threads(runtime['threads'])
    random.seed(t['seed'])
    np.random.seed(t['seed'])
    torch.manual_seed(t['seed'])
    if runtime['device'].startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('The configured CUDA device is unavailable; refusing to fall back to CPU.')
        torch.cuda.manual_seed_all(t['seed'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run_fingerprint = fingerprint(config, encoder)
    with (out / 'resolved_config.yml').open('w') as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    lab, data, cache = prepare_data(config, encoder)
    tr, va, te = split_indices(lab, split=t['split'], seed=t['seed'], **config['split'])
    for name, indices in zip(('train', 'val', 'test'), (tr, va, te)):
        if not len(indices) or lab.label.iloc[indices].nunique() < 2:
            raise ValueError(f'{name} must contain both classes; check dataset/split configuration.')
    manifest = lab[['offspring_id', 'generation', 'label', 'p1_id', 'p2_id']].copy()
    manifest['subset'] = ''
    for name, indices in zip(('train', 'val', 'test'), (tr, va, te)):
        manifest.loc[indices, 'subset'] = name
    manifest.to_csv(out / 'split_manifest.csv', index=False)
    prefix = 'numbers' if encoder == 'cgcnn' else 'z'
    suffixes = ['1', '2'] + (['offspring'] if t['offspring_structure'] else [])
    follow = [f'{prefix}_{s}' for s in suffixes]
    generator = torch.Generator().manual_seed(t['seed'])
    subsets = [[data[int(i)] for i in idx] for idx in (tr, va, te)]
    train_loader = DataLoader(subsets[0], batch_size=t['batch_size'], shuffle=True, follow_batch=follow,
                              drop_last=encoder == 'mxtal' and len(tr) % t['batch_size'] == 1,
                              generator=generator)
    eval_loaders = [DataLoader(subset, batch_size=t['eval_batch_size'], follow_batch=follow) for subset in subsets]
    calibration_loader = DataLoader(subsets[0], batch_size=t['batch_size'], follow_batch=follow,
                                     drop_last=len(tr) % t['batch_size'] == 1)
    device = runtime['device']
    model = make_model(config, encoder, lab, tr, cache).to(device)
    labels = lab.label.values[tr]
    pos_weight = torch.tensor((labels == 0).sum() / (labels == 1).sum(), dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=t['lr_factor'], patience=t['lr_patience'])
    history, best, best_epoch, started = [], float('inf'), 0, time.monotonic()
    print(f"{encoder} {t['fusion']} child={t['offspring_structure']} {t['split']} seed={t['seed']} "
          f'train/val/test={len(tr)}/{len(va)}/{len(te)}', flush=True)
    for epoch in range(t['epochs']):
        loss = train(model, train_loader, optimizer, criterion, device)
        if encoder == 'mxtal':
            recalibrate_bn(model, calibration_loader, device)
        va_loss, va_p, va_y, _ = evaluate(model, eval_loaders[1], criterion, device)
        if not np.isfinite(loss + va_loss):
            raise RuntimeError(f'Nonfinite loss at epoch {epoch}')
        scheduler.step(va_loss)
        m = metrics(va_y, va_p, t['threshold'])
        history.append(dict(epoch=epoch, train_loss=loss, val_loss=va_loss, val_acc=m['accuracy'],
                            val_bacc=m['balanced_accuracy'], val_auc=m['roc_auc'], lr=optimizer.param_groups[0]['lr']))
        print(f"epoch {epoch:3d} train={loss:.4f} val={va_loss:.4f} val_auc={m['roc_auc']:.3f} "
              f'elapsed={time.monotonic() - started:.0f}s', flush=True)
        if va_loss < best:
            best, best_epoch = va_loss, epoch
            torch.save(model.state_dict(), out / 'best_model.pt')
        elif epoch - best_epoch >= t['patience']:
            break
    model.load_state_dict(torch.load(out / 'best_model.pt', map_location=device, weights_only=True))
    flat_config = dict(t, **config[encoder], db=str(project_path(config['dataset']['db'])), out=str(out), device=device)
    result = dict(config=flat_config, encoder=encoder, fingerprint=run_fingerprint,
                  n_params=sum(p.numel() for p in model.parameters()), best_epoch=best_epoch,
                  epochs_completed=len(history), train_time_s=time.monotonic() - started,
                  n_train=len(tr), n_val=len(va), n_test=len(te),
                  provenance=dict(database_sha256=file_digest(project_path(config['dataset']['db'])),
                    source_sha256=source_digest(), conda_env=config['env']['conda_env'], torch=torch.__version__,
                    cuda=torch.version.cuda, physical_gpu=runtime['gpu'],
                    gpu_name=torch.cuda.get_device_name(0) if device.startswith('cuda') else None,
                    offspring_geometry=config['dataset']['offspring_geometry'],
                    offspring_structures_source=str(project_path(config['dataset']['results_dir']) /
                                                    config['dataset']['offspring_structures']),
                    offspring_structures_sha256=file_digest(project_path(config['dataset']['results_dir']) /
                                                           config['dataset']['offspring_structures'])))
    for name, loader in zip(('train', 'val', 'test'), eval_loaders):
        loss, probs, y, ids = evaluate(model, loader, criterion, device)
        result[name] = dict(metrics(y, probs, t['threshold']), loss=loss)
        pd.DataFrame(dict(offspring_id=ids, y=y, p_good=probs)).to_csv(out / f'{name}_predictions.csv', index=False)
    pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
    with (out / 'metrics.json').open('w') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(dict(best_epoch=best_epoch, test=result['test'])), flush=True)
    del model, optimizer
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return result


def training_main(encoder):
    config = config_from_cli()
    parser = argparse.ArgumentParser(description=f'GNNEA {encoder} classifier')
    parser.add_argument('--config', default=config['_config_path'])
    parser.add_argument('--db')
    parser.add_argument('--out')
    parser.add_argument('--device')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--separate-encoders', action='store_true', default=None)
    parser.add_argument('--no-crystal-norm', action='store_true', default=None)
    for key, value in {**config['training'], **config[encoder]}.items():
        option = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            parser.add_argument(option, action=argparse.BooleanOptionalAction, default=None)
        elif value is not None:
            parser.add_argument(option, type=type(value), default=None)
    args = vars(parser.parse_args())
    config = deepcopy(config)
    for section in ('training', encoder):
        for key in config[section]:
            if args.get(key) is not None:
                config[section][key] = args[key]
    for key in ('threads', 'device'):
        if args[key] is not None:
            config['runtime'][key] = args[key]
    if args['db']:
        config['dataset']['db'] = args['db']
    if args['separate_encoders']:
        config['training']['shared_encoder'] = False
    if encoder == 'mxtal' and args['no_crystal_norm']:
        config['mxtal']['crystal_norm'] = None
    out = project_path(args['out'] or str(Path(config['paths']['results_dir']) / encoder))
    return run_training(config, encoder, out)
