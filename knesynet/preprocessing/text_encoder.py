from __future__ import annotations
from typing import List
import torch
from torch import nn

class ClinicalBERTCaseEncoder(nn.Module):
    def __init__(self, checkpoint="emilyalsentzer/Bio_ClinicalBERT", max_tokens=512, overlap=64, fine_tune=True):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint)
        self.max_tokens = max_tokens
        self.overlap = overlap
        if not fine_tune:
            for p in self.model.parameters():
                p.requires_grad = False

    def _segments(self, text: str):
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        usable = self.max_tokens - 2
        step = max(1, usable - self.overlap)
        for start in range(0, max(1, len(ids)), step):
            chunk = ids[start:start + usable]
            if not chunk:
                break
            yield self.tokenizer.prepare_for_model(
                chunk, add_special_tokens=True, return_tensors="pt", truncation=True,
                max_length=self.max_tokens
            )
            if start + usable >= len(ids):
                break

    def encode_documents(self, documents: List[str], device=None):
        device = device or next(self.parameters()).device
        doc_embeddings = []
        for text in documents:
            seg_embeddings = []
            for batch in self._segments(text):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = self.model(**batch).last_hidden_state[:, 0]
                seg_embeddings.append(out.squeeze(0))
            if seg_embeddings:
                doc_embeddings.append(torch.stack(seg_embeddings).mean(0))
        if not doc_embeddings:
            return torch.zeros(768, device=device)
        return torch.stack(doc_embeddings).mean(0)
