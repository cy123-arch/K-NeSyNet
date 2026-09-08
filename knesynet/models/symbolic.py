from __future__ import annotations
import torch
from torch import nn

class SymbolicCombiner(nn.Module):
    """Implements s_symbolic = s_guide + s_target + s_contra."""
    def forward(self, guide, target, contra):
        return guide + target + contra
