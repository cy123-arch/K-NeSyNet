from __future__ import annotations
import torch
from torch import nn

class UnimodalMLP(nn.Module):
    def __init__(self, input_dim, n_drugs=80, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_drugs), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)
