from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class GenomicEncoder(nn.Module):
    def __init__(self, gene_count=10586, projected_dim=256):
        super().__init__()
        self.fc = nn.Linear(gene_count, projected_dim)
        self.norm = nn.LayerNorm(projected_dim)
    def forward(self, x):
        return F.relu(self.norm(self.fc(x)))

class ModalityProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
    def forward(self, x):
        return F.relu(self.norm(self.fc(x)))

class MaskAwareModalityFusion(nn.Module):
    def __init__(self, image_dim=512, text_dim=768, genomic_dim=256, embedding_dim=64):
        super().__init__()
        self.image = ModalityProjection(image_dim, embedding_dim)
        self.text = ModalityProjection(text_dim, embedding_dim)
        self.genomic = ModalityProjection(genomic_dim, embedding_dim)
        self.att = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.normal_(self.att, std=0.02)

    def forward(self, image, text, genomic, mask):
        # mask: [B, 3] in image/text/genomic order.
        hs = torch.stack([self.image(image), self.text(text), self.genomic(genomic)], dim=1)
        logits = torch.einsum("bmd,d->bm", hs, self.att)
        logits = logits.masked_fill(mask <= 0, float("-inf"))
        weights = torch.softmax(logits, dim=1)
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
        patient = torch.einsum("bm,bmd->bd", weights, hs)
        return patient, weights
