from __future__ import annotations
import torch
from torch import nn

class BilinearNeuralScorer(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.W = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, patient, drug):
        # patient [B,D], drug [B,M,D]
        logits = torch.einsum("bd,de,bme->bm", patient, self.W, drug)
        return torch.sigmoid(logits)

class AdaptiveGate(nn.Module):
    def __init__(self, patient_dim=64):
        super().__init__()
        self.fc = nn.Linear(patient_dim + 2, 1)

    def forward(self, patient, neural_score, symbolic_score):
        # patient [B,D], scores [B,M]
        B, M = neural_score.shape
        p = patient[:, None, :].expand(B, M, patient.shape[-1])
        x = torch.cat([p, neural_score[..., None], symbolic_score[..., None]], dim=-1)
        return torch.sigmoid(self.fc(x)).squeeze(-1)

def fuse_scores(neural_score, symbolic_score, gate):
    """Equation: gate * neural + (1-gate) * sigmoid(symbolic)."""
    return gate * neural_score + (1.0 - gate) * torch.sigmoid(symbolic_score)
