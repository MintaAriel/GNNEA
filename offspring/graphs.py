"""Periodic crystal graphs (seminar 8 `atoms2data`) and parent-pair graph objects."""
import numpy as np
import torch
from torch_geometric.data import Data
from vesin import NeighborList


def atoms2graph(atoms, r_cut=6.0, n_max_neighbors=12):
    """Neighbours within r_cut (periodic), truncated to the n_max_neighbors nearest (CGCNN uses 12)."""
    calculator = NeighborList(cutoff=r_cut, full_list=True)
    i, j, ij_length = calculator.compute(points=atoms.positions, box=atoms.cell.array,
                                         periodic=bool(atoms.pbc.any()), quantities='ijd')
    if n_max_neighbors is not None:
        keep = []
        for atom_idx in np.unique(i):
            idx = np.where(i == atom_idx)[0]
            keep.extend(idx[np.argsort(ij_length[idx])][:n_max_neighbors])
        keep = np.array(keep)
        i, j, ij_length = i[keep], j[keep], ij_length[keep]
    return dict(numbers=torch.LongTensor(atoms.get_atomic_numbers()),
                edge_index=torch.stack([torch.LongTensor(i.astype(np.int64)),
                                        torch.LongTensor(j.astype(np.int64))]),
                edge_length=torch.tensor(ij_length, dtype=torch.float32))


class PairData(Data):
    """Parent graphs (suffixes _1, _2), optionally with a child graph (_offspring).

    Edge indices of each parent are shifted by the number of atoms of that same parent when
    batching; include numbers_offspring in follow_batch when the child is enabled.
    """

    def __inc__(self, key, value, *args, **kwargs):
        for suffix in ('1', '2', 'offspring'):
            if key == f'edge_index_{suffix}':
                return self[f'numbers_{suffix}'].size(0)
        return super().__inc__(key, value, *args, **kwargs)


def make_pair_dataset(df, r_cut=6.0, n_max_neighbors=12, global_cols=(), offspring_structure=False):
    """One PairData per row of a labelled offspring df (needs p1, p2, p1_id, p2_id, label).

    Graphs of parents are cached by uspex id because the same parent appears in many pairs.
    global_cols: optional per-pair scalar features (e.g. parent energies) stored in `u`.
    offspring_structure: append the calculator-free child graph; no child energy enters the input.
    """
    cache = {}

    def graph(uid, atoms):
        if uid not in cache:
            cache[uid] = atoms2graph(atoms, r_cut=r_cut, n_max_neighbors=n_max_neighbors)
        return cache[uid]

    data = []
    for row in df.itertuples(index=False):
        g1 = graph(int(row.p1_id), row.p1)
        g2 = graph(int(row.p2_id), row.p2)
        d = PairData(numbers_1=g1['numbers'], edge_index_1=g1['edge_index'], edge_length_1=g1['edge_length'],
                     numbers_2=g2['numbers'], edge_index_2=g2['edge_index'], edge_length_2=g2['edge_length'],
                     y=torch.tensor([float(row.label)]),
                     offspring_id=torch.tensor([int(row.offspring_id)]))
        if offspring_structure:
            child = graph(int(row.offspring_id), row.offspring)
            for key, value in child.items():
                d[f'{key}_offspring'] = value
        if global_cols:
            d.u = torch.tensor([[float(getattr(row, c)) for c in global_cols]], dtype=torch.float32)
        data.append(d)
    return data
