from __future__ import annotations
import torch
import torch.nn.functional as F

def composite_loss(scores, labels, contraindication_indicator, model, alpha=1.0, beta=1e-5, eps=1e-8):
    bce = F.binary_cross_entropy(scores, labels.float())
    c = contraindication_indicator.float()
    safety = (c * scores).sum() / (c.sum() + eps) if c.sum() > 0 else scores.new_tensor(0.0)
    l2 = scores.new_tensor(0.0)
    for p in model.parameters():
        if p.requires_grad:
            l2 = l2 + p.pow(2).sum()
    total = bce + alpha * safety + beta * l2
    return total, {"bce": bce.detach(), "safety": safety.detach(), "l2": l2.detach()}
