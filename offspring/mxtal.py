"""Self-contained port of the MXtalTools molecular-crystal encoder and crystal RDF analysis.

Copied / adapted from ~/skoltech/ms_thesis/repos/MXtalTools (no import of mxtaltools needed: the
package needs torch_sparse / torch_cluster and a CIF-based ingestion chain that we do not have).

What is ported (MXtalTools source -> here)
------------------------------------------
crystal_building.utils.ucell2cluster / MolCrystalBuilding.build_cluster
        -> molecules_from_atoms + build_cluster
           Our USPEX cells are P1 cells holding whole molecules (theophylline: 42 atoms = 2 x C7H8N4O2),
           so the "asymmetric unit" is the whole unit cell (Z' = number of molecules, sg 1). Molecules are
           found from covalent radii under PBC (crystal_steps.find_molecules(bonds='distance')) and made
           whole; the cluster is the reference cell (aux_ind 0, "inside") plus its periodic images.
           Image atoms within `cutoff` of an inside atom are kept (aux_ind 1); the rest (aux_ind 2 in
           MXtalTools, they are in no edge) are dropped here.
models.functions.radial_graph.build_radial_graph
        -> radial_edges (torch.cdist instead of torch_cluster.radius; nearest max_num_neighbors kept)
models.modules.basis_functions.{BesselBasisLayer, Envelope, GaussianEmbedding}
models.modules.augmented_softmax_aggregator.AugSoftmaxAggregation
models.modules.components.{Normalization, Activation, scalarMLP, EmbeddingBlock}
models.modules.graph_convolution.MConv
models.graph_models.graph_neural_network.MolCrystalScalarGNN
        -> identical layers. `periodize_molecular_crystal` copies the features of every inside atom onto
           all its images; MXtalTools does it with `.repeat(n_repeats)` (needs the cluster to be whole
           copies of the inside block), here with a gather `x[image_src]` (same result, allows pruning).
models.graph_models.molecule_graph_model.MolecularCrystalGraphModel
models.graph_models.base_graph_model.BaseGraphModel.{featurize_input_graph, standardize}
models.task_models.crystal_models.MolecularCrystalModel
        -> MXtalCrystalEncoder (atom-property featurization + GNN + softmax pooling + MLP)
analysis.crystal_rdf.{crystal_rdf (mode='elementwise'), smooth_1d_histogram, smooth_cutoff,
                      compute_rdf_distance, earth_movers_distance_torch}
        -> crystal_rdf, rdf_distance, rdf_distance_matrix (chunked, all pairs at once)

Notes on fidelity
-----------------
* In MolCrystalScalarGNN the intra- and intermolecular edges are concatenated and go through the SAME
  convolution, so the encoder is not told which edges are intermolecular. `edge_type=True` (off by default,
  not in MXtalTools) appends a 0/1 intermolecular flag to the radial basis.
* BesselBasisLayer zeroes the envelope where dist/cutoff > cutoff (a typo for > 1 in MXtalTools); here > 1.
  Identical on edges <= cutoff, which are the only ones built.
* The quirk `x[:, -1] = aux_ind` after each periodization is kept (it marks image atoms in one channel).
"""
from argparse import Namespace
from math import pi as PI
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ase.data import atomic_masses, covalent_radii
from torch import Tensor
from torch_geometric import nn as gnn
from torch_geometric.data import Data
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from vesin import NeighborList

# ---------------------------------------------------------------------------------------------------------------
# atom property tables (mxtaltools.constants.atom_properties) for the elements we can meet; index = Z
# ---------------------------------------------------------------------------------------------------------------
_N_Z = 100
VDW_RADII = np.zeros(_N_Z)
ELECTRONEGATIVITY = np.full(_N_Z, -1.0)
GROUP = np.zeros(_N_Z)
PERIOD = np.zeros(_N_Z)
for _z, _r, _en, _g, _p in [(1, 1.2, 2.2, 1, 1), (6, 1.7, 2.55, 14, 2), (7, 1.6, 3.04, 15, 2),
                            (8, 1.55, 3.44, 16, 2), (9, 1.47, 3.98, 17, 2), (15, 1.8, 2.19, 15, 3),
                            (16, 1.8, 2.58, 16, 3), (17, 1.75, 3.16, 17, 3), (35, 1.85, 2.96, 17, 4)]:
    VDW_RADII[_z], ELECTRONEGATIVITY[_z], GROUP[_z], PERIOD[_z] = _r, _en, _g, _p
ATOM_WEIGHTS = np.asarray(atomic_masses[:_N_Z], dtype=float)
ATOM_PROPERTIES = dict(vdw_radii=VDW_RADII, atom_weight=ATOM_WEIGHTS, electronegativity=ELECTRONEGATIVITY,
                       group=GROUP, period=PERIOD)
DEFAULT_ATOM_FEATURES = ['atomic_number', 'vdw_radii', 'atom_weight', 'electronegativity', 'group', 'period']


# ===============================================================================================================
# 1. crystal building: molecules, cluster of periodic images, radial graph
# ===============================================================================================================
def molecules_from_atoms(atoms, bond_scale=(1.15, 1.35)):
    """Find molecules under PBC and make them whole (crystal_steps.find_molecules, bonds='distance').

    Atoms i, j are bonded when |x_i - x_j - t| < s_i r_i + s_j r_j for a lattice translation t (ASE covalent
    radii; s = bond_scale[0] for heavy atoms, bond_scale[1] for H, as MXtalTools' generator benchmark).
    Each molecule is unwrapped along its bond graph and translated so its centroid lies inside the cell.

    Returns pos [N, 3] (whole molecules), z [N], mol_ind [N] (0..n_mol-1), atoms sorted by molecule.
    """
    z = atoms.get_atomic_numbers()
    cell = atoms.cell.array
    scale = np.where(z == 1, bond_scale[1], bond_scale[0])
    radii = covalent_radii[z] * scale
    i, j, S, d = NeighborList(cutoff=2 * radii.max(), full_list=True).compute(
        points=atoms.positions, box=cell, periodic=True, quantities='ijSd')
    bonded = d < radii[i] + radii[j]
    i, j, S = i[bonded], j[bonded], S[bonded]
    shift = S @ cell                                       # x_j + shift is the image of j bonded to i

    n = len(z)
    neighbors = [[] for _ in range(n)]
    for a, b, s in zip(i, j, shift):
        neighbors[a].append((b, s))
    pos = atoms.positions.copy()
    mol_ind = -np.ones(n, dtype=int)
    n_mol = 0
    for seed in range(n):                                  # BFS: unwrap each molecule along its bonds
        if mol_ind[seed] >= 0:
            continue
        mol_ind[seed] = n_mol
        stack = [seed]
        while stack:
            a = stack.pop()
            for b, s in neighbors[a]:
                if mol_ind[b] < 0:
                    mol_ind[b] = n_mol
                    pos[b] = pos[a] + (atoms.positions[b] + s - atoms.positions[a])
                    stack.append(b)
        n_mol += 1

    inv = np.linalg.inv(cell)
    for m in range(n_mol):                                 # centroid of every molecule inside [0, 1)
        sel = mol_ind == m
        frac_c = pos[sel].mean(0) @ inv
        pos[sel] -= np.floor(frac_c) @ cell
    order = np.argsort(mol_ind, kind='stable')
    return pos[order], z[order], mol_ind[order]


def _translations(cell, reach):
    """All lattice translations n (|n_i| <= ceil(reach / d_i)), d_i = interplanar spacing (get_cart_translations)."""
    vol = abs(np.linalg.det(cell))
    d = np.array([vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) for k in range(3)])
    nmax = np.ceil(reach / d).astype(int)
    grid = np.stack(np.meshgrid(*[np.arange(-m, m + 1) for m in nmax], indexing='ij'), -1).reshape(-1, 3)
    grid = grid[np.argsort(np.abs(grid).sum(1), kind='stable')]   # (0, 0, 0) first
    return grid


def radial_edges(pos, mol_ind, n_inside, cutoff, max_num_neighbors=100):
    """build_radial_graph(aux_ind=...) for one cluster whose first n_inside nodes are the inside nodes.

    edge_index:       intramolecular edges among inside nodes (both directions, no self loops)
    edge_index_inter: source = image node, or inside node of another molecule; target = inside node
    Row 0 is the source, row 1 the target (flow='source_to_target'). Per inside target at most
    max_num_neighbors sources are kept (the nearest; torch_cluster keeps an arbitrary subset).
    """
    pos = torch.as_tensor(pos, dtype=torch.float64)
    mol_ind = torch.as_tensor(mol_ind)
    d = torch.cdist(pos[:n_inside], pos)                     # [inside, all]
    d[torch.arange(n_inside), torch.arange(n_inside)] = float('inf')
    k = min(max_num_neighbors, d.shape[1])
    dk, src = torch.topk(d, k, dim=1, largest=False)
    tgt = torch.arange(n_inside)[:, None].expand_as(src)
    keep = dk <= cutoff
    src, tgt = src[keep], tgt[keep]
    intra = (src < n_inside) & (mol_ind[src] == mol_ind[tgt])
    return (torch.stack([src[intra], tgt[intra]]).long(),
            torch.stack([src[~intra], tgt[~intra]]).long())


def build_cluster(atoms, cutoff=6.0, max_num_neighbors=100, bond_scale=(1.15, 1.35)):
    """ucell2cluster + build_radial_graph for one ase.Atoms crystal (P1 cell holding whole molecules).

    Returns a dict of tensors (one graph):
        z [N], pos [N, 3]         inside atoms first (the unit cell, molecules whole), then image atoms
        mol_ind [N]               molecule index in the cluster (images of molecule m get new indices)
        aux_ind [N]               0 inside, 1 image atom within `cutoff` of an inside atom
        image_src [N]             index of the inside atom each node is a copy of (itself for inside nodes)
        edge_index, edge_index_inter   see radial_edges
        n_mol, n_inside           ints (n_mol = Z' in MXtalTools terms, the cell is taken as P1)
    """
    pos0, z0, mol0 = molecules_from_atoms(atoms, bond_scale=bond_scale)
    cell = atoms.cell.array
    n_in, n_mol = len(z0), int(mol0.max()) + 1
    reach = cutoff + 2 * np.max([np.linalg.norm(pos0[mol0 == m] - pos0[mol0 == m].mean(0), axis=1).max()
                                 for m in range(n_mol)]) + 1.0
    T = _translations(cell, reach) @ cell                    # [n_cells, 3], zero translation first
    img = pos0[None] + T[:, None]                            # [n_cells, n_in, 3]
    dmin = torch.cdist(torch.as_tensor(img.reshape(-1, 3)), torch.as_tensor(pos0)).min(1).values.numpy()
    dmin = dmin.reshape(len(T), n_in)
    keep = dmin <= cutoff
    keep[0] = True                                           # the reference cell itself

    cell_ind, atom_ind = np.nonzero(keep)                    # row-major: cell 0 (inside) first
    pos = img[cell_ind, atom_ind]
    z = z0[atom_ind]
    mol_ind = mol0[atom_ind] + n_mol * cell_ind
    aux_ind = (cell_ind > 0).astype(int)
    edge_index, edge_index_inter = radial_edges(pos, mol_ind, n_in, cutoff, max_num_neighbors)
    return dict(z=torch.as_tensor(z, dtype=torch.long), pos=torch.as_tensor(pos, dtype=torch.float32),
                mol_ind=torch.as_tensor(mol_ind, dtype=torch.long), aux_ind=torch.as_tensor(aux_ind, dtype=torch.long),
                image_src=torch.as_tensor(atom_ind, dtype=torch.long),
                edge_index=edge_index, edge_index_inter=edge_index_inter, n_mol=n_mol, n_inside=n_in)


class PairClusterData(Data):
    """Parent clusters (_1, _2) and optional child (_offspring).
    Include z_offspring in follow_batch when using the child."""
    _NODE_INDEX = ('edge_index', 'edge_index_inter', 'image_src')

    def __inc__(self, key, value, *args, **kwargs):
        for k in self._NODE_INDEX:
            for suffix in ('1', '2', 'offspring'):
                if key == f'{k}_{suffix}':
                    return self[f'z_{suffix}'].size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key.startswith('edge_index'):
            return -1
        return super().__cat_dim__(key, value, *args, **kwargs)


_CLUSTER_KEYS = ('z', 'pos', 'aux_ind', 'image_src', 'edge_index', 'edge_index_inter')


def make_pair_cluster_dataset(df, cutoff=6.0, max_num_neighbors=100, global_cols=(), cache=None, verbose=True,
                              offspring_structure=False):
    """One PairClusterData per row of a labelled offspring df (needs p1, p2, p1_id, p2_id, label, offspring_id).
    Clusters are cached by (geometry state, uspex_id); pass `cache` (key -> build_cluster
    output) to reuse clusters built earlier. Returns (data, cache)."""
    cache = {} if cache is None else cache

    def cluster(uid, atoms):
        if uid not in cache:
            cache[uid] = build_cluster(atoms, cutoff=cutoff, max_num_neighbors=max_num_neighbors)
            if verbose and len(cache) % 200 == 0:
                print(f'  built {len(cache)} clusters', flush=True)
        return cache[uid]

    data = []
    for row in df.itertuples(index=False):
        c1 = cluster(('relaxed', int(row.p1_id)), row.p1)
        c2 = cluster(('relaxed', int(row.p2_id)), row.p2)
        kw = {f'{k}_1': c1[k] for k in _CLUSTER_KEYS}
        kw.update({f'{k}_2': c2[k] for k in _CLUSTER_KEYS})
        if offspring_structure:
            child = cluster(('unrelaxed', int(row.offspring_id)), row.offspring)
            kw.update({f'{k}_offspring': child[k] for k in _CLUSTER_KEYS})
        d = PairClusterData(**kw, y=torch.tensor([float(row.label)]),
                            offspring_id=torch.tensor([int(row.offspring_id)]))
        if global_cols:
            d.u = torch.tensor([[float(getattr(row, c)) for c in global_cols]], dtype=torch.float32)
        data.append(d)
    return data, cache


# ===============================================================================================================
# 2. network modules (verbatim ports)
# ===============================================================================================================
class Envelope(nn.Module):
    def __init__(self, exponent):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x0 = x.pow(p - 1)
        x1 = x0 * x
        return 1. / x + a * x0 + b * x1 + c * x1 * x


class BesselBasisLayer(nn.Module):
    """envelope(d / cutoff) * sin(freq * d / cutoff), learnable freq (DimeNet)."""

    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        self.register_buffer('cutoff', torch.tensor(float(cutoff)))
        self.envelope = Envelope(envelope_exponent)
        self.freq = nn.Parameter(torch.arange(1, num_radial + 1, dtype=torch.float32) * PI)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        env = self.envelope(dist) * (dist <= 1)
        return env * (self.freq * dist).sin()


class GaussianEmbedding(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer('offset', offset)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2

    def forward(self, dist):
        return torch.exp(self.coeff * (dist.view(-1, 1) - self.offset.view(1, -1)) ** 2)


class AugSoftmaxAggregation(Aggregation):
    """sum_i x_i * (softmax_i(t * x_i) + b), learnable per-channel temperature t and bias b."""

    def __init__(self, temperature=1.0, channels=1, bias=0.1):
        super().__init__()
        self.channels = channels
        self.t = nn.Parameter(torch.full((channels,), float(temperature)))
        self.b = nn.Parameter(torch.full((channels,), float(bias)))

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        t, b = self.t.view(-1, self.channels), self.b.view(-1, self.channels)
        alpha = softmax(x * t, index, ptr, dim_size, dim)
        return self.reduce(x * (alpha + b), index, ptr, dim_size, dim, reduce='sum')


class Normalization(nn.Module):
    def __init__(self, norm, filters):
        super().__init__()
        self.norm_type = norm
        self.norm = {'batch': lambda: nn.BatchNorm1d(filters), 'graph layer': lambda: gnn.LayerNorm(filters),
                     'layer': lambda: nn.LayerNorm(filters), 'graph': lambda: gnn.GraphNorm(filters),
                     None: nn.Identity}[norm]()

    def forward(self, x, batch=None):
        if batch is not None and self.norm_type not in ('batch', 'layer', None):
            return self.norm(x, batch)
        return self.norm(x)


_ACT = {'relu': F.relu, 'gelu': F.gelu, 'leaky relu': F.leaky_relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid}


class scalarMLP(nn.Module):
    """Residual MLP: x = x + dropout(act(norm(linear(x)))) per layer, then an optional output projection."""

    def __init__(self, layers, filters, input_dim, output_dim, activation='gelu', dropout=0.0, norm=None):
        super().__init__()
        dims = [input_dim] + [filters] * layers
        self.same_depth = len(set(dims)) == 1
        if not self.same_depth:
            self.residue_adjust = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(layers)])
        self.fc_layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layers)])
        self.fc_norms = nn.ModuleList([Normalization(norm, dims[i + 1]) for i in range(layers)])
        self.fc_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(layers)])
        self.act = _ACT[activation]
        self.output_layer = nn.Linear(dims[-1], output_dim, bias=False) if output_dim != dims[-1] else nn.Identity()

    def forward(self, x, batch=None):
        for i, (lin, norm, drop) in enumerate(zip(self.fc_layers, self.fc_norms, self.fc_dropouts)):
            res = x if self.same_depth else self.residue_adjust[i](x)
            x = res + drop(self.act(norm(lin(x), batch=batch)))
        return self.output_layer(x)


class EmbeddingBlock(nn.Module):
    """Column 0 (atomic number) -> nn.Embedding, concatenated with the other columns -> Linear."""

    def __init__(self, node_dim, num_input_classes, num_scalar_input_features, atom_type_embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(num_input_classes + 1, atom_type_embedding_dim)
        self.linear = nn.Linear(atom_type_embedding_dim + num_scalar_input_features - 1, node_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return self.linear(torch.cat([self.embeddings(x[:, 0].long()), x[:, 1:]], dim=-1))


class MConv(MessagePassing):
    """x_i += W( AugSoftmax_j act( W_g [W_s x_i, W_t x_j, W_e e_ij] ) )."""

    def __init__(self, message_dim, node_dim, edge_embedding_dim, norm=None, activation_fn='gelu'):
        super().__init__(aggr=AugSoftmaxAggregation(temperature=1, bias=0.1, channels=message_dim))
        self.message_dim = message_dim
        self.edge2message = nn.Linear(edge_embedding_dim, message_dim, bias=False)
        self.source_node2message = nn.Linear(node_dim, message_dim, bias=False)
        self.tgt_node2message = nn.Linear(node_dim, message_dim, bias=False)
        self.generate_message = nn.Linear(3 * message_dim, message_dim, bias=False)
        self.norm = Normalization(norm, message_dim)
        self.act = _ACT[activation_fn]
        self.message2node = nn.Linear(message_dim, node_dim, bias=False)

    def forward(self, x, edge_index, edge_attr):
        # MXtalTools computes per edge  generate_message(cat[W_s x_i, W_t x_j, W_e e_ij])  with bias-free layers.
        # That equals  G_s W_s x_i + G_t W_t x_j + G_e W_e e_ij  (G = the column blocks of generate_message), so the
        # node terms are computed once per node and gathered per edge: same function and parameters, ~10x cheaper.
        M = self.message_dim
        G = self.generate_message.weight
        s = F.linear(self.source_node2message(x), G[:, :M])
        t = F.linear(self.tgt_node2message(x), G[:, M:2 * M])
        e = F.linear(edge_attr, G[:, 2 * M:] @ self.edge2message.weight)   # one per-edge matmul instead of two
        out = self.propagate(edge_index=edge_index, s=s, t=t, e=e, size=(x.size(0), x.size(0)))
        return x + self.message2node(out)

    def message(self, s_i, t_j, e):
        return self.act(self.norm(s_i + t_j + e))


class MolCrystalScalarGNN(nn.Module):
    """Message passing over the inside/image cluster; image atoms are re-synchronized with their inside atom
    after every convolution. Returns the embeddings of the inside atoms."""

    def __init__(self, input_node_dim, node_dim, fcs_per_gc, message_dim, embedding_dim, num_convs, num_radial,
                 num_input_classes=101, cutoff=6.0, envelope_exponent=5, activation='gelu',
                 atom_type_embedding_dim=5, norm=None, dropout=0.0, radial_embedding='bessel', edge_type=False,
                 **unused):
        super().__init__()
        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        else:
            self.rbf = GaussianEmbedding(0.0, cutoff, num_radial)
        self.edge_type = edge_type
        edge_dim = num_radial + int(edge_type)
        self.init_node_embedding = EmbeddingBlock(node_dim, num_input_classes, input_node_dim, atom_type_embedding_dim)
        self.zeroth_fc_block = scalarMLP(fcs_per_gc, node_dim, node_dim, node_dim, activation, dropout, norm)
        self.interaction_blocks = nn.ModuleList([MConv(message_dim, node_dim, edge_dim, None, activation)
                                                 for _ in range(num_convs)])
        self.fc_blocks = nn.ModuleList([scalarMLP(fcs_per_gc, node_dim, node_dim, node_dim, activation, dropout, norm)
                                        for _ in range(num_convs)])
        self.output_layer = nn.Linear(node_dim, embedding_dim, bias=False) if node_dim != embedding_dim \
            else nn.Identity()

    def forward(self, x, pos, batch, aux_ind, image_src, edge_index, edge_index_inter):
        x = self.init_node_embedding(x)
        x = self.zeroth_fc_block(x, batch=batch)
        inside = torch.nonzero(aux_ind == 0).flatten()
        ei = torch.cat([edge_index, edge_index_inter], dim=1)         # all edges in one convolution
        dist = (pos[ei[0]] - pos[ei[1]]).norm(dim=-1)
        rbf = self.rbf(dist)
        if self.edge_type:
            flag = torch.cat([torch.zeros(edge_index.size(1)), torch.ones(edge_index_inter.size(1))]).to(rbf)
            rbf = torch.cat([rbf, flag[:, None]], dim=-1)
        aux_col = aux_ind.to(x.dtype)[:, None]
        for conv, fc in zip(self.interaction_blocks, self.fc_blocks):
            x = conv(x, ei, rbf)
            x = x.index_copy(0, inside, fc(x[inside], batch=batch[inside]))
            x = x[image_src]                                          # periodize: images <- inside atom
            x = torch.cat([x[:, :-1], aux_col], dim=-1)               # x[:, -1] = aux_ind (MXtalTools)
        return self.output_layer(x[inside]), inside


# ===============================================================================================================
# 3. crystal encoder (MolecularCrystalModel + MolecularCrystalGraphModel) and pair classifier
# ===============================================================================================================
DISCRIMINATOR_CONFIG = Namespace(  # configs/experiments/base/discriminator.yaml, model section
    activation='gelu',
    graph=Namespace(node_dim=128, message_dim=64, embedding_dim=128, num_convs=2, fcs_per_gc=1, num_radial=12,
                    num_input_classes=101, norm='graph layer', dropout=0.05, atom_type_embedding_dim=12,
                    radial_embedding='bessel', cutoff=6.0, max_num_neighbors=100, envelope_exponent=5),
    fc=Namespace(hidden_dim=128, num_layers=2, dropout=0.05, norm='layer'))


def atom_feature_stats(z, atom_features=DEFAULT_ATOM_FEATURES):
    """(mean, std) per node-feature column over the atoms z (column 0 = atomic number is left unscaled)."""
    table = np.stack([np.arange(_N_Z)] + [ATOM_PROPERTIES[f] for f in atom_features[1:]], 1)[np.asarray(z)]
    stats = np.stack([table.mean(0), table.std(0) + 1e-6], 1)
    stats[0] = (0, 1)
    return torch.tensor(stats, dtype=torch.float32)


class MXtalCrystalEncoder(nn.Module):
    """One embedding per crystal: atom-property featurization -> MolCrystalScalarGNN -> AugSoftmax pooling over
    the inside atoms -> optional scalarMLP. Output width: config.fc.hidden_dim (or graph.embedding_dim)."""

    def __init__(self, config=DISCRIMINATOR_CONFIG, atom_features=DEFAULT_ATOM_FEATURES,
                 node_standardization=None, edge_type=False, pool=True):
        super().__init__()
        assert atom_features[0] == 'atomic_number'
        table = [np.arange(_N_Z)] + [ATOM_PROPERTIES[f] for f in atom_features[1:]]
        self.register_buffer('atom_properties_tensor', torch.tensor(np.stack(table, 1), dtype=torch.float32))
        if node_standardization is None:
            node_standardization = torch.tensor([[0., 1.]] * len(atom_features))
        self.register_buffer('node_standardization_tensor', node_standardization)
        g = config.graph
        self.graph_net = MolCrystalScalarGNN(input_node_dim=len(atom_features), activation=config.activation,
                                             edge_type=edge_type, **vars(g))
        self.pool = pool
        self.global_pool = AugSoftmaxAggregation(temperature=1, bias=0.1, channels=g.embedding_dim)
        if config.fc.num_layers > 0:
            self.gnn_mlp = scalarMLP(config.fc.num_layers, config.fc.hidden_dim, g.embedding_dim,
                                     config.fc.hidden_dim, config.activation, config.fc.dropout, config.fc.norm)
            self.out_dim = config.fc.hidden_dim
        else:
            self.gnn_mlp = nn.Identity()
            self.out_dim = g.embedding_dim
        self.node_dim = g.embedding_dim

    def featurize(self, z):
        x = self.atom_properties_tensor[z]
        return (x - self.node_standardization_tensor[:, 0]) / self.node_standardization_tensor[:, 1]

    def node_embeddings(self, z, pos, batch, aux_ind, image_src, edge_index, edge_index_inter):
        """Embeddings of the inside atoms [n_inside, embedding_dim] and their batch index."""
        h, inside = self.graph_net(self.featurize(z), pos, batch, aux_ind, image_src, edge_index, edge_index_inter)
        return h, batch[inside]

    def forward(self, z, pos, batch, aux_ind, image_src, edge_index, edge_index_inter, num_graphs):
        h, b = self.node_embeddings(z, pos, batch, aux_ind, image_src, edge_index, edge_index_inter)
        return self.gnn_mlp(self.global_pool(h, b, dim_size=num_graphs))


class PairMXtal(nn.Module):
    """Pair classifier with MXtalTools crystal encoders (same fusion options as offspring.model.PairCGCNN).

    fusion='concat_pool' : inside-atom embeddings of both parents are pooled together per pair (symmetric).
    fusion='pool_concat' : each parent -> crystal embedding g1, g2, then [g1+g2, |g1-g2|, g1*g2].
    offspring_structure=True includes child nodes in concat_pool or appends its crystal
    embedding in pool_concat; the encoder weights are shared by default.
    crystal_norm='batch' : BatchNorm1d on the crystal / pair embedding before the fusion (not in MXtalTools).
        All parents hold the same two molecules, so the softmax-sum pooled embeddings are dominated by a component
        shared by every crystal (at initialization they differ by ~3 % of their size between crystals) and the
        faithful model (crystal_norm=None) collapses to a constant prediction. Normalizing over the batch removes
        the shared component, as the BatchNorm layers of CGCNN do.
    """

    def __init__(self, config=DISCRIMINATOR_CONFIG, atom_features=DEFAULT_ATOM_FEATURES, node_standardization=None,
                 fusion='concat_pool', shared_encoder=True, n_global=0, h_fea_len=64, dropout=0.2, edge_type=False,
                 crystal_norm='batch', offspring_structure=False):
        super().__init__()
        if fusion not in ('concat_pool', 'pool_concat'):
            raise ValueError(fusion)
        self.fusion, self.n_global = fusion, n_global
        self.offspring_structure = offspring_structure
        make = lambda: MXtalCrystalEncoder(config, atom_features, node_standardization, edge_type)
        self.enc1 = make()
        self.enc2 = self.enc1 if shared_encoder else make()
        if offspring_structure:
            self.enc_offspring = self.enc1 if shared_encoder else make()
        if fusion == 'concat_pool':
            self.pair_pool = AugSoftmaxAggregation(temperature=1, bias=0.1, channels=self.enc1.node_dim)
            self.pair_mlp = self.enc1.gnn_mlp if shared_encoder else make().gnn_mlp
            pooled = self.enc1.out_dim
        else:
            pooled = 3 * self.enc1.out_dim
            if offspring_structure:
                pooled += self.enc1.out_dim
        self.crystal_norm = nn.BatchNorm1d(self.enc1.out_dim) if crystal_norm == 'batch' else nn.Identity()
        self.head = nn.Sequential(nn.Linear(pooled + n_global, h_fea_len), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(h_fea_len, 1))

    @staticmethod
    def _args(data, s):
        return (data[f'z_{s}'], data[f'pos_{s}'], data[f'z_{s}_batch'], data[f'aux_ind_{s}'],
                data[f'image_src_{s}'], data[f'edge_index_{s}'], data[f'edge_index_inter_{s}'])

    def forward(self, data):
        n = data.y.size(0)
        if self.fusion == 'concat_pool':
            h1, b1 = self.enc1.node_embeddings(*self._args(data, 1))
            h2, b2 = self.enc2.node_embeddings(*self._args(data, 2))
            hs, bs = [h1, h2], [b1, b2]
            if self.offspring_structure:
                hc, bc = self.enc_offspring.node_embeddings(*self._args(data, 'offspring'))
                hs.append(hc)
                bs.append(bc)
            g = self.crystal_norm(self.pair_mlp(self.pair_pool(torch.cat(hs), torch.cat(bs), dim_size=n)))
        else:
            g1 = self.crystal_norm(self.enc1(*self._args(data, 1), num_graphs=n))
            g2 = self.crystal_norm(self.enc2(*self._args(data, 2), num_graphs=n))
            g = torch.cat([g1 + g2, (g1 - g2).abs(), g1 * g2], dim=-1)
            if self.offspring_structure:
                gc = self.crystal_norm(self.enc_offspring(*self._args(data, 'offspring'), num_graphs=n))
                g = torch.cat([g, gc], dim=-1)
        if self.n_global:
            g = torch.cat([g, data.u], dim=-1)
        return self.head(g).squeeze(-1)


# ===============================================================================================================
# 4. crystal RDF + EMD distance (analysis.crystal_rdf), for clustering structures
# ===============================================================================================================
def smooth_cutoff(d, rc, ron):
    x = ((d - ron) / (rc - ron)).clamp(0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(torch.pi * x)) * (d <= rc)


def element_pairs(elements):
    elements = sorted(elements)
    return [(a, b) for i, a in enumerate(elements) for b in elements[i:]]


def crystal_rdf(cluster, pairs, rrange=(0.0, 10.0), bins=100, sigma=None):
    """Elementwise intermolecular RDF of one cluster (crystal_rdf(mode='elementwise') + smooth_1d_histogram).

    Uses the intermolecular edges (edge_index_inter) of a cluster built with cutoff >= rrange[1]. Each distance
    is spread with a Gaussian (sigma = bin width), histogrammed per element-pair channel, divided by the shell
    volume and by Z' (molecules per cell), and damped over the last 1 A. Returns [len(pairs), bins].
    """
    rmin, rmax = rrange
    centers = torch.linspace(rmin, rmax, bins, dtype=torch.float64)
    edges = torch.linspace(rmin, rmax, bins + 1, dtype=torch.float64)
    sigma = sigma or (rmax - rmin) / bins
    ei = cluster['edge_index_inter']
    pos, z = cluster['pos'].double(), cluster['z']
    d = (pos[ei[0]] - pos[ei[1]]).norm(dim=-1)
    za, zb = torch.minimum(z[ei[0]], z[ei[1]]), torch.maximum(z[ei[0]], z[ei[1]])
    lut = {p: k for k, p in enumerate(pairs)}
    chan = torch.tensor([lut.get((int(a), int(b)), -1) for a, b in zip(za, zb)], dtype=torch.long)
    ok = chan >= 0
    w = torch.exp(-0.5 * ((d[ok, None] - centers[None]) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    w = w * (rmax - rmin) / bins
    hist = torch.zeros(len(pairs), bins, dtype=torch.float64).index_add_(0, chan[ok], w)
    shell = (4 / 3) * torch.pi * ((edges[:-1] + torch.diff(edges)) ** 3 - edges[:-1] ** 3)
    rdf = hist / shell[None] * smooth_cutoff(edges[1:], rmax, rmax - 1)[None]
    return (rdf / cluster['n_mol']).float()


def rdf_distance(rdf1, rdf2, rr):
    """compute_rdf_distance: per channel EMD of the normalized RDFs (x bin width), averaged over active channels."""
    bw = rr[1] - rr[0]
    p1 = rdf1 / (rdf1.sum(-1, keepdim=True) + 1e-10)
    p2 = rdf2 / (rdf2.sum(-1, keepdim=True) + 1e-10)
    emd = (torch.cumsum(p1, -1) - torch.cumsum(p2, -1)).abs().sum(-1) * bw
    active = (rdf1.sum(-1) > 1e-12) | (rdf2.sum(-1) > 1e-12)
    return (emd * active).sum(-1) / active.sum(-1).clamp_min(1)


def rdf_distance_matrix(rdfs, rr, chunk=256):
    """All-pairs rdf_distance for rdfs [n, channels, bins] -> [n, n] (EMD = L1 distance between CDFs)."""
    rdfs = rdfs.double()
    bw = float(rr[1] - rr[0])
    cdf = torch.cumsum(rdfs / (rdfs.sum(-1, keepdim=True) + 1e-10), -1)       # [n, C, B]
    act = (rdfs.sum(-1) > 1e-12).double()                                     # [n, C]
    n, C, _ = rdfs.shape
    D = torch.zeros(n, n, dtype=torch.float64)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        num = torch.zeros(e - s, n, dtype=torch.float64)
        den = torch.zeros(e - s, n, dtype=torch.float64)
        for c in range(C):
            a = 1 - (1 - act[s:e, c, None]) * (1 - act[None, :, c])          # active in either
            num += torch.cdist(cdf[s:e, c], cdf[:, c], p=1) * bw * a
            den += a
        D[s:e] = num / den.clamp_min(1)
    return D.float()


def greedy_anchors(D, energies, d_cut, e_cut=None):
    """common.clustering.greedy_bottom_up_anchors on a precomputed distance matrix: visit structures by ascending
    energy, keep one as a new anchor (a distinct low-energy "polymorph") if it is farther than d_cut from all
    anchors so far. Returns anchor indices and, for every structure, the index of its nearest anchor."""
    order = np.argsort(energies)
    anchors = [order[0]]
    for i in order[1:]:
        if e_cut is not None and energies[i] > e_cut:
            continue
        if np.all(D[i, anchors] > d_cut):
            anchors.append(i)
    anchors = np.array(anchors)
    return anchors, anchors[np.argmin(D[:, anchors], axis=1)]
