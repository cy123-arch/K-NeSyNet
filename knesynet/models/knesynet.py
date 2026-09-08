from __future__ import annotations
import torch
from torch import nn
from .encoders import GenomicEncoder, MaskAwareModalityFusion
from .gat import HomogeneousGAT
from .symbolic import SymbolicCombiner
from .fusion import BilinearNeuralScorer, AdaptiveGate, fuse_scores

class KNeSyNet(nn.Module):
    def __init__(
        self,
        num_entities: int,
        gene_count: int = 10586,
        embedding_dim: int = 64,
        gat_heads: int = 4,
        gat_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.genomic_encoder = GenomicEncoder(gene_count, 256)
        self.patient_encoder = MaskAwareModalityFusion(512, 768, 256, embedding_dim)
        self.graph = HomogeneousGAT(num_entities, embedding_dim, gat_heads, gat_layers, dropout)
        self.neural_scorer = BilinearNeuralScorer(embedding_dim)
        self.symbolic = SymbolicCombiner()
        self.gate = AdaptiveGate(embedding_dim)

    def forward(
        self,
        image_features,
        text_features,
        genomic_multihot,
        modality_mask,
        edge_index,
        drug_node_indices,
        guide_scores,
        target_scores,
        contra_scores,
    ):
        genomic = self.genomic_encoder(genomic_multihot)
        patient, modality_weights = self.patient_encoder(
            image_features, text_features, genomic, modality_mask
        )
        entity_embeddings = self.graph(edge_index)
        drug = entity_embeddings[drug_node_indices][None, :, :].expand(patient.shape[0], -1, -1)
        neural = self.neural_scorer(patient, drug)
        symbolic = self.symbolic(guide_scores, target_scores, contra_scores)
        gate = self.gate(patient, neural, symbolic)
        fused = fuse_scores(neural, symbolic, gate)
        return {
            "score": fused,
            "neural": neural,
            "symbolic": symbolic,
            "gate": gate,
            "modality_weights": modality_weights,
        }
