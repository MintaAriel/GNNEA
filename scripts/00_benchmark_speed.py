"""Check whether training is feasible on this machine (CPU): graph building + epoch timings."""
import os
import sys
import time

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from offspring.db import read_offspring_df
from offspring.graphs import make_pair_dataset
from offspring.labels import label_offspring
from offspring.db import read_individuals_df
from offspring.model import PairCGCNN

db = 'data/offspring.db'
t = time.time()
lab = label_offspring(read_offspring_df(db, operators=['Heredity']), read_individuals_df(db))
t_read = time.time() - t
t = time.time()
data = make_pair_dataset(lab)
t_graph = time.time() - t
print(f'threads={torch.get_num_threads()}  read={t_read:.1f}s  graphs({len(data)} pairs)={t_graph:.1f}s')
print('example pair:', data[0])

loader = DataLoader(data, batch_size=64, shuffle=True, follow_batch=['numbers_1', 'numbers_2'])
for fusion in ('concat_pool', 'pool_concat'):
    model = PairCGCNN(fusion=fusion)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.BCEWithLogitsLoss()
    t = time.time()
    for batch in loader:
        opt.zero_grad()
        loss = crit(model(batch), batch.y)
        loss.backward()
        opt.step()
    dt = time.time() - t
    n_par = sum(p.numel() for p in model.parameters())
    print(f'{fusion}: params={n_par}  one epoch over all {len(data)} pairs = {dt:.2f}s '
          f'-> 200 epochs ~ {200 * dt / 60:.1f} min')
