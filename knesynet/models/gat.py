from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class HomogeneousGAT(nn.Module):
    def __init__(self, num_entities, dim=64, heads=4, layers=2, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.embedding = nn.Embedding(num_entities, dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(GATConv(dim, dim // heads, heads=heads, concat=True, dropout=dropout))
        self.dropout = dropout

    def forward(self, edge_index):
        x = self.embedding.weight
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
