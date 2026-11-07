"""Task 2 with MXtalTools molecular-crystal encoders (offspring/mxtal.py) instead of CGCNN.

Same data, labels, splits, loss, early stopping and outputs as 03_train_gnn.py, so results are directly
comparable (results/runs/<name>/metrics.json etc.). Clusters are cached in data/mxtal_clusters_<cutoff>.pt.

    python scripts/09_train_mxtal.py --out results/runs/M_concat_struct
    python scripts/09_train_mxtal.py --fusion pool_concat --energy-features --edge-type --out results/runs/M_...
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
from offspring.data import ENERGY_FEATURES, load_labelled, split_indices
from offspring.mxtal import DEFAULT_ATOM_FEATURES, PairMXtal, atom_feature_stats, make_pair_cluster_dataset

# reuse the train / evaluate / metrics functions of the CGCNN script
_spec = importlib.util.spec_from_file_location('train_gnn', os.path.join(os.path.dirname(__file__), '03_train_gnn.py'))
_tg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tg)
train, evaluate, metrics = _tg.train, _tg.evaluate, _tg.metrics

FOLLOW = ['z_1', 'z_2']


@torch.no_grad()
def recalibrate_bn(model, loader, device='cpu'):
    """Recompute the BatchNorm running statistics exactly (cumulative average) on the training pairs with the current
    weights, as torch.optim.swa_utils.update_bn does. The crystal embeddings share a large component that changes
    while the encoder trains; the exponential running mean lags behind it and the eval-mode predictions then flip
    to one class. Dropout is kept off during the pass."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.eval()
    for m in bns:
        m.train()
    for batch in loader:
        model(batch.to(device))
    for m in bns:
        m.momentum = 0.1


def cached_dataset(lab, cutoff, max_nb, gcols, path):
    """Pair dataset; the per-parent clusters are stored in `path` and reused on the next run."""
    cache = torch.load(path, weights_only=False) if os.path.exists(path) else {}
    n_before, t0 = len(cache), time.time()
    data, cache = make_pair_cluster_dataset(lab, cutoff, max_nb, gcols, cache=cache)
    if len(cache) > n_before:
        torch.save(cache, path)
        print(f'built {len(cache) - n_before} clusters in {time.time() - t0:.0f}s -> {path}')
    return data, cache


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='data/offspring.db')
    ap.add_argument('--out', default='results/runs/mxtal')
    ap.add_argument('--split', default='random', choices=['random', 'generation', 'group'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fusion', default='concat_pool', choices=['concat_pool', 'pool_concat'])
    ap.add_argument('--separate-encoders', action='store_true')
    ap.add_argument('--energy-features', action='store_true', help='append parent energy z-scores')
    ap.add_argument('--edge-type', action='store_true',
                    help='append an intra/intermolecular flag to the edge features (not in MXtalTools)')
    ap.add_argument('--no-crystal-norm', action='store_true',
                    help='faithful MXtalTools head without BatchNorm on the crystal embeddings (collapses, see mxtal.py)')
    # MolCrystalScalarGNN (defaults: MXtalTools configs/experiments/base/discriminator.yaml, scaled down for CPU)
    ap.add_argument('--cutoff', type=float, default=6.0)
    ap.add_argument('--max-neighbors', type=int, default=100)
    ap.add_argument('--node-dim', type=int, default=64)
    ap.add_argument('--message-dim', type=int, default=32)
    ap.add_argument('--embedding-dim', type=int, default=64)
    ap.add_argument('--num-convs', type=int, default=2)
    ap.add_argument('--num-radial', type=int, default=12)
    ap.add_argument('--fc-layers', type=int, default=2)
    ap.add_argument('--fc-dim', type=int, default=64)
    ap.add_argument('--graph-norm', default='graph layer', choices=['graph layer', 'layer', 'graph', 'batch', 'none'])
    ap.add_argument('--gnn-dropout', type=float, default=0.05)
    ap.add_argument('--h-fea', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    lab = load_labelled(a.db)
    gcols = ENERGY_FEATURES if a.energy_features else ()
    cache_path = os.path.join(os.path.dirname(a.db), f'mxtal_clusters_c{a.cutoff:g}_n{a.max_neighbors}.pt')
    data, cache = cached_dataset(lab, a.cutoff, a.max_neighbors, gcols, cache_path)
    print('molecules per parent cell:', pd.Series({k: v['n_mol'] for k, v in cache.items()}).value_counts().to_dict())
    tr, va, te = split_indices(lab, split=a.split, seed=a.seed)
    train_data, val_data, test_data = ([data[i] for i in s] for s in (tr, va, te))
    print(f'train/val/test = {len(tr)}/{len(va)}/{len(te)}  good fraction train = {lab.label.values[tr].mean():.3f}')

    # BatchNorm (crystal_norm) cannot train on a last batch of one pair
    train_loader = DataLoader(train_data, batch_size=a.batch_size, shuffle=True, follow_batch=FOLLOW,
                              drop_last=len(train_data) % a.batch_size == 1)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False, follow_batch=FOLLOW)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, follow_batch=FOLLOW)

    # node-feature standardization from the atoms of the training parents (MXtalTools: dataset statistics)
    train_ids = set(lab.p1_id.values[tr].astype(int)) | set(lab.p2_id.values[tr].astype(int))
    z_train = torch.cat([cache[i]['z'][:cache[i]['n_inside']] for i in train_ids]).numpy()
    norm = None if a.graph_norm == 'none' else a.graph_norm
    config = Namespace(
        activation='gelu',
        graph=Namespace(node_dim=a.node_dim, message_dim=a.message_dim, embedding_dim=a.embedding_dim,
                        num_convs=a.num_convs, fcs_per_gc=1, num_radial=a.num_radial, num_input_classes=101,
                        norm=norm, dropout=a.gnn_dropout, atom_type_embedding_dim=12, radial_embedding='bessel',
                        cutoff=a.cutoff, max_num_neighbors=a.max_neighbors, envelope_exponent=5),
        fc=Namespace(hidden_dim=a.fc_dim, num_layers=a.fc_layers, dropout=a.gnn_dropout, norm='layer'))
    model = PairMXtal(config, DEFAULT_ATOM_FEATURES, atom_feature_stats(z_train), fusion=a.fusion,
                      shared_encoder=not a.separate_encoders, n_global=len(gcols), h_fea_len=a.h_fea,
                      dropout=a.dropout, edge_type=a.edge_type,
                      crystal_norm=None if a.no_crystal_norm else 'batch').to(a.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'PairMXtal: {n_params} parameters')

    y_tr = lab.label.values[tr]
    pos_weight = torch.tensor((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(a.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    history, best, best_epoch, t0 = [], np.inf, 0, time.time()
    for epoch in range(a.epochs):
        tr_loss = train(model, train_loader, optimizer, criterion, a.device)
        recalibrate_bn(model, train_loader, a.device)
        va_loss, va_p, va_y, _ = evaluate(model, val_loader, criterion, a.device)
        scheduler.step(va_loss)
        va_m = metrics(va_y, va_p)
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss, val_acc=va_m['accuracy'],
                            val_bacc=va_m['balanced_accuracy'], val_auc=va_m['roc_auc'],
                            lr=optimizer.param_groups[0]['lr']))
        print(f"epoch {epoch:3d}  train {tr_loss:.4f}  val {va_loss:.4f}  val_acc {va_m['accuracy']:.3f}  "
              f"val_auc {va_m['roc_auc']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if va_loss < best:
            best, best_epoch = va_loss, epoch
            torch.save(model.state_dict(), os.path.join(a.out, 'best_model.pt'))
        elif epoch - best_epoch >= a.patience:
            print(f'early stopping at epoch {epoch}, best epoch {best_epoch}')
            break

    model.load_state_dict(torch.load(os.path.join(a.out, 'best_model.pt')))  # includes the recalibrated statistics
    _, va_p, va_y, _ = evaluate(model, val_loader, criterion, a.device)
    te_loss, te_p, te_y, te_ids = evaluate(model, test_loader, criterion, a.device)
    tr_loss, tr_p, tr_y, _ = evaluate(model, DataLoader(train_data, batch_size=128, follow_batch=FOLLOW),
                                      criterion, a.device)
    result = dict(config=vars(a), encoder='mxtal', n_params=n_params, best_epoch=best_epoch,
                  train_time_s=time.time() - t0, n_train=len(tr), n_val=len(va), n_test=len(te),
                  train=metrics(tr_y, tr_p), val=metrics(va_y, va_p), test=metrics(te_y, te_p))
    print(json.dumps({k: result[k] for k in ('best_epoch', 'test')}, indent=1))

    pd.DataFrame(history).to_csv(os.path.join(a.out, 'history.csv'), index=False)
    pd.DataFrame({'offspring_id': te_ids, 'y': te_y, 'p_good': te_p}).to_csv(
        os.path.join(a.out, 'test_predictions.csv'), index=False)
    with open(os.path.join(a.out, 'metrics.json'), 'w') as f:
        json.dump(result, f, indent=1)
