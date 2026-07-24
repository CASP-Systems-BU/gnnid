"""FlashSAGE: 2-layer GraphSAGE role classifier (FLASH §4.3.1).

forward returns (logits, embedding): the embedding (last hidden) is what the
XGBoost stage concatenates with the Word2Vec vector.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FlashSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int, embed: int, num_classes: int,
                 dropout: float = 0.25):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, embed, aggr="mean")
        self.head = nn.Linear(embed, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        emb = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(emb), emb
