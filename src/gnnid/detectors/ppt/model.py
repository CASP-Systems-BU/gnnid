"""PPT-GNN model: stacked temporal-then-spatial hetero layers (paper §3.2).

Each layer applies, per edge type e, the paper's update
W1_e·h_v + W2_e·mean({h_u : u in N_e(v)}) — a SAGEConv with mean aggregation
and root weight — first over the temporal relations, then over the spatial
ones, summing across relations (HeteroConv aggr="sum" = the paper's ⊕₂) with
LeakyReLU between passes. Temporal-first maps both node types into the hidden
space before spatial mixing, as in the paper.

PPTModel adds the 2-layer MLP role classifier over ENTITY nodes.
PPTLinkPredictor wraps the same encoder with per-relation concat-MLP edge
decoders for self-supervised pre-training (relations are asymmetric and
connect different node types, so dot-product decoders don't fit); decoders
are discarded after pre-training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

from .graph import ALL_EDGE_TYPES, SPATIAL_EDGE_TYPES, TEMPORAL_EDGE_TYPES


def _key(et: tuple[str, str, str]) -> str:
    return "__".join(et)


class PPTLayer(nn.Module):
    def __init__(self, ent_in: int, ev_in: int, out: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        dims = {"entity": ent_in, "event": ev_in}
        self.temporal = HeteroConv(
            {et: SAGEConv((dims[et[0]], dims[et[2]]), out, aggr="mean")
             for et in TEMPORAL_EDGE_TYPES}, aggr="sum")
        self.spatial = HeteroConv(
            {et: SAGEConv((out, out), out, aggr="mean")
             for et in SPATIAL_EDGE_TYPES}, aggr="sum")

    def forward(self, x_dict, edge_index_dict):
        h = self.temporal(x_dict, edge_index_dict)
        h = {k: F.leaky_relu(v) for k, v in h.items()}
        h = self.spatial(h, edge_index_dict)
        return {k: F.dropout(F.leaky_relu(v), p=self.dropout,
                             training=self.training) for k, v in h.items()}


class PPTEncoder(nn.Module):
    def __init__(self, ent_dim: int, ev_dim: int, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.ent_dim, self.ev_dim = ent_dim, ev_dim
        self.hidden, self.num_layers, self.dropout = hidden, layers, dropout
        mods = [PPTLayer(ent_dim, ev_dim, hidden, dropout)]
        mods += [PPTLayer(hidden, hidden, hidden, dropout)
                 for _ in range(layers - 1)]
        self.layers = nn.ModuleList(mods)

    def forward(self, x_dict, edge_index_dict) -> dict[str, torch.Tensor]:
        h = x_dict
        for layer in self.layers:
            h = layer(h, edge_index_dict)
        return h


class PPTModel(nn.Module):
    """Encoder + role-classification head over entity nodes."""

    def __init__(self, ent_dim: int, ev_dim: int, hidden: int, layers: int,
                 num_classes: int, dropout: float):
        super().__init__()
        self.arch = {"ent_dim": ent_dim, "ev_dim": ev_dim, "hidden": hidden,
                     "layers": layers, "num_classes": num_classes,
                     "dropout": dropout}
        self.encoder = PPTEncoder(ent_dim, ev_dim, hidden, layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes))

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(data.x_dict, data.edge_index_dict)
        ent = h["entity"]
        return self.head(ent), ent


class PPTLinkPredictor(nn.Module):
    """Encoder + per-relation edge decoders for link-prediction pre-training."""

    def __init__(self, encoder: PPTEncoder):
        super().__init__()
        self.encoder = encoder
        h = encoder.hidden
        self.decoders = nn.ModuleDict({
            _key(et): nn.Sequential(nn.Linear(2 * h, h), nn.LeakyReLU(),
                                    nn.Linear(h, 1))
            for et in ALL_EDGE_TYPES})

    def decode(self, h_dict: dict[str, torch.Tensor], et: tuple,
               edge_index: torch.Tensor) -> torch.Tensor:
        src, _, dst = et
        z = torch.cat([h_dict[src][edge_index[0]],
                       h_dict[dst][edge_index[1]]], dim=1)
        return self.decoders[_key(et)](z).squeeze(-1)
