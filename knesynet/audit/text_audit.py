from __future__ import annotations
import pandas as pd

def summarize_document_audit(path):
    df = pd.read_csv(path)
    return {
        "documents": len(df),
        "temporal_eligible": int(df["temporal_eligible"].astype(bool).sum()),
        "documents_with_terms": int(df["treatment_terms_detected"].fillna("").astype(str).ne("").sum()),
        "sentences_removed": int(df["sentences_removed"].fillna(0).astype(int).sum()),
    }
