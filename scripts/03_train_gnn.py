"""Task 2: train the pair-CGCNN good/bad offspring classifier (train/evaluate loop of seminar 8)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from offspring.data import ENERGY_FEATURES, load_labelled, split_indices
from offspring.graphs import make_pair_dataset
from offspring.model import PairCGCNN

FOLLOW = ['numbers_1', 'numbers_2']


def train(model, iterator, optimizer, criterion, device='cpu'):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        batch = batch.to(device)
        loss = criterion(model(batch), batch.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.y.size(0)
    return epoch_loss / len(iterator.dataset)


def evaluate(model, iterator, criterion, device='cpu'):
    model.eval()
    epoch_loss, probs, ys, ids = 0, [], [], []
    with torch.no_grad():
        for batch in iterator:
            batch = batch.to(device)
            logits = model(batch)
            epoch_loss += criterion(logits, batch.y).item() * batch.y.size(0)
            probs.append(torch.sigmoid(logits).cpu())
            ys.append(batch.y.cpu())
            ids.append(batch.offspring_id.cpu())
    return (epoch_loss / len(iterator.dataset), torch.cat(probs).numpy(), torch.cat(ys).numpy(),
            torch.cat(ids).numpy())


def metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred),
                f1=f1_score(y, pred), roc_auc=roc_auc_score(y, p) if len(set(y)) > 1 else float('nan'),
                confusion=confusion_matrix(y, pred, labels=[0, 1]).tolist(), threshold=threshold)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='data/offspring.db')
    ap.add_argument('--out', default='results/gnn')
    ap.add_argument('--split', default='random', choices=['random', 'generation', 'group'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fusion', default='concat_pool', choices=['concat_pool', 'pool_concat'])
    ap.add_argument('--separate-encoders', action='store_true')
    ap.add_argument('--energy-features', action='store_true', help='append parent energy z-scores')
    ap.add_argument('--node-fea', type=int, default=32)
    ap.add_argument('--edge-fea', type=int, default=32)
    ap.add_argument('--h-fea', type=int, default=64)
    ap.add_argument('--n-conv', type=int, default=3)
    ap.add_argument('--r-cut', type=float, default=6.0)
    ap.add_argument('--max-neighbors', type=int, default=12)
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
    data = make_pair_dataset(lab, r_cut=a.r_cut, n_max_neighbors=a.max_neighbors, global_cols=gcols)
    tr, va, te = split_indices(lab, split=a.split, seed=a.seed)
    train_data, val_data, test_data = ([data[i] for i in s] for s in (tr, va, te))
    print(f'train/val/test = {len(tr)}/{len(va)}/{len(te)}  good fraction train = {lab.label.values[tr].mean():.3f}')

    train_loader = DataLoader(train_data, batch_size=a.batch_size, shuffle=True, follow_batch=FOLLOW)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=False, follow_batch=FOLLOW)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, follow_batch=FOLLOW)

    model = PairCGCNN(node_fea_len=a.node_fea, edge_fea_len=a.edge_fea, h_fea_len=a.h_fea, n_conv=a.n_conv,
                      r_cut=a.r_cut, fusion=a.fusion, shared_encoder=not a.separate_encoders,
                      n_global=len(gcols), dropout=a.dropout).to(a.device)
    n_params = sum(p.numel() for p in model.parameters())

    # class imbalance (~34 % good): weight positives by n_neg / n_pos
    y_tr = lab.label.values[tr]
    pos_weight = torch.tensor((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(a.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    history, best, best_epoch, t0 = [], np.inf, 0, time.time()
    for epoch in range(a.epochs):
        tr_loss = train(model, train_loader, optimizer, criterion, a.device)
        va_loss, va_p, va_y, _ = evaluate(model, val_loader, criterion, a.device)
        scheduler.step(va_loss)
        va_m = metrics(va_y, va_p)
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss, val_acc=va_m['accuracy'],
                            val_bacc=va_m['balanced_accuracy'], val_auc=va_m['roc_auc'],
                            lr=optimizer.param_groups[0]['lr']))
        print(f"epoch {epoch:3d}  train {tr_loss:.4f}  val {va_loss:.4f}  val_acc {va_m['accuracy']:.3f}  "
              f"val_auc {va_m['roc_auc']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if va_loss < best:  # seminar 8 save_checkpoint
            best, best_epoch = va_loss, epoch
            torch.save(model.state_dict(), os.path.join(a.out, 'best_model.pt'))
        elif epoch - best_epoch >= a.patience:
            print(f'early stopping at epoch {epoch}, best epoch {best_epoch}')
            break

    model.load_state_dict(torch.load(os.path.join(a.out, 'best_model.pt')))
    _, va_p, va_y, _ = evaluate(model, val_loader, criterion, a.device)
    te_loss, te_p, te_y, te_ids = evaluate(model, test_loader, criterion, a.device)
    tr_loss, tr_p, tr_y, _ = evaluate(model, DataLoader(train_data, batch_size=256, follow_batch=FOLLOW),
                                      criterion, a.device)
    result = dict(config=vars(a), n_params=n_params, best_epoch=best_epoch, train_time_s=time.time() - t0,
                  n_train=len(tr), n_val=len(va), n_test=len(te),
                  train=metrics(tr_y, tr_p), val=metrics(va_y, va_p), test=metrics(te_y, te_p))
    print(json.dumps({k: result[k] for k in ('best_epoch', 'test')}, indent=1))

    pd.DataFrame(history).to_csv(os.path.join(a.out, 'history.csv'), index=False)
    pd.DataFrame({'offspring_id': te_ids, 'y': te_y, 'p_good': te_p}).to_csv(
        os.path.join(a.out, 'test_predictions.csv'), index=False)
    with open(os.path.join(a.out, 'metrics.json'), 'w') as f:
        json.dump(result, f, indent=1)
