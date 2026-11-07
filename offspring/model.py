"""Pair-CGCNN: CGCNN encoders (seminar 8) for both parents + fusion/pooling + classifier head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


class GaussianFilter(nn.Module):
    """Edge length -> Gaussian basis expansion (seminar 8)."""

    def __init__(self, r_min=0.5, r_max=6.0, num_edge_features=64):
        super().__init__()
        self.register_buffer('_filter', torch.linspace(r_min, r_max, num_edge_features))
        self.register_buffer('step', torch.tensor((r_max - r_min) / num_edge_features))

    def forward(self, edge_length):
        diff = edge_length[:, None] - self._filter
        return torch.exp(-0.5 * torch.square(diff / self.step))


class GraphConvLayer(nn.Module):
    """CGCNN convolution: v_i += BN( sum_j sigmoid(z W_f + b_f) * g(z W_s + b_s) ), z = v_i + v_j + u_ij."""

    def __init__(self, node_fea_len, edge_fea_len):
        super().__init__()
        self.lin_f = nn.Linear(2 * node_fea_len + edge_fea_len, node_fea_len)
        self.lin_s = nn.Linear(2 * node_fea_len + edge_fea_len, node_fea_len)
        self.norm = nn.BatchNorm1d(node_fea_len)

    def forward(self, node_attrs, edge_index, edge_attrs):
        source, target = edge_index
        z = torch.cat([node_attrs[source], node_attrs[target], edge_attrs], dim=-1)
        z = torch.sigmoid(self.lin_f(z)) * F.leaky_relu(self.lin_s(z), negative_slope=0.01)
        total_message = scatter(z, source, dim=0, dim_size=node_attrs.size(0), reduce='sum')
        return node_attrs + self.norm(total_message)


class CGCNNEncoder(nn.Module):
    """Embedding(Z) -> BN -> n_conv x GraphConvLayer -> node embeddings."""

    def __init__(self, n_species=9, node_fea_len=64, edge_fea_len=64, n_conv=3, r_cut=6.0):
        super().__init__()
        self.embedding = nn.Embedding(n_species, node_fea_len)
        self.filter = GaussianFilter(r_max=r_cut, num_edge_features=edge_fea_len)
        self.norm = nn.BatchNorm1d(node_fea_len)
        self.convs = nn.ModuleList([GraphConvLayer(node_fea_len, edge_fea_len) for _ in range(n_conv)])

    def forward(self, numbers, edge_index, edge_length):
        h = self.norm(self.embedding(numbers))
        e = self.filter(edge_length)
        for conv in self.convs:
            h = conv(h, edge_index, e)
        return h


class PairCGCNN(nn.Module):
    """Predict whether the offspring of (parent 1, parent 2) is good (logit output).

    fusion='concat_pool' : node embeddings of both parents are concatenated into one node set
                           and pooled per pair (mean || max)  -> symmetric in the parents.
    fusion='pool_concat' : each parent is pooled to h1, h2, then [h1+h2, |h1-h2|, h1*h2].
    shared_encoder=False uses two independent encoders (parent-order dependent).
    n_global > 0 appends per-pair scalar features (e.g. parent energies) to the pooled vector.
    offspring_structure=True adds child nodes before concat_pool, or appends the pooled child
    to the symmetric parent representation in pool_concat. Child and parents share weights
    unless shared_encoder=False. Only the parents are required to be exchangeable.
    """

    def __init__(self, n_species=9, node_fea_len=64, edge_fea_len=64, h_fea_len=128, n_conv=3,
                 r_cut=6.0, fusion='concat_pool', shared_encoder=True, n_global=0, dropout=0.2,
                 offspring_structure=False):
        super().__init__()
        if fusion not in ('concat_pool', 'pool_concat'):
            raise ValueError(fusion)
        self.fusion = fusion
        self.n_global = n_global
        self.offspring_structure = offspring_structure
        self.enc1 = CGCNNEncoder(n_species, node_fea_len, edge_fea_len, n_conv, r_cut)
        self.enc2 = self.enc1 if shared_encoder else CGCNNEncoder(n_species, node_fea_len, edge_fea_len,
                                                                 n_conv, r_cut)
        if offspring_structure:
            self.enc_offspring = self.enc1 if shared_encoder else CGCNNEncoder(
                n_species, node_fea_len, edge_fea_len, n_conv, r_cut)
        self.lin_atom = nn.Linear(node_fea_len, h_fea_len)
        pooled = 2 * h_fea_len if fusion == 'concat_pool' else 3 * 2 * h_fea_len
        if offspring_structure and fusion == 'pool_concat':
            pooled += 2 * h_fea_len
        self.head = nn.Sequential(nn.Linear(pooled + n_global, h_fea_len), nn.Softplus(),
                                  nn.Dropout(dropout), nn.Linear(h_fea_len, 1))

    @staticmethod
    def _pool(h, batch, n):
        return torch.cat([scatter(h, batch, dim=0, dim_size=n, reduce='mean'),
                          scatter(h, batch, dim=0, dim_size=n, reduce='max')], dim=-1)

    def forward(self, data):
        n = data.y.size(0)
        h1 = self.enc1(data.numbers_1, data.edge_index_1, data.edge_length_1)
        h2 = self.enc2(data.numbers_2, data.edge_index_2, data.edge_length_2)
        h1 = F.softplus(self.lin_atom(h1))
        h2 = F.softplus(self.lin_atom(h2))
        if self.offspring_structure:
            hc = F.softplus(self.lin_atom(self.enc_offspring(
                data.numbers_offspring, data.edge_index_offspring, data.edge_length_offspring)))

        if self.fusion == 'concat_pool':
            h = torch.cat([h1, h2], dim=0)
            batch = torch.cat([data.numbers_1_batch, data.numbers_2_batch], dim=0)
            if self.offspring_structure:
                h = torch.cat([h, hc], dim=0)
                batch = torch.cat([batch, data.numbers_offspring_batch], dim=0)
            g = self._pool(h, batch, n)
        else:
            g1 = self._pool(h1, data.numbers_1_batch, n)
            g2 = self._pool(h2, data.numbers_2_batch, n)
            g = torch.cat([g1 + g2, (g1 - g2).abs(), g1 * g2], dim=-1)
            if self.offspring_structure:
                g = torch.cat([g, self._pool(hc, data.numbers_offspring_batch, n)], dim=-1)

        if self.n_global:
            g = torch.cat([g, data.u], dim=-1)
        return self.head(g).squeeze(-1)
