from __future__ import annotations
import pandas as pd

def audit_heldout_treatment_edges(edges_csv, validation_ids, test_ids):
    e = pd.read_csv(edges_csv)
    rel = e["relation"].astype(str)
    tx = e[rel.str.contains("treated_with", regex=False)].copy()
    def count(ids):
        ids = set(map(str, ids))
        return int((tx["src"].astype(str).isin(ids) | tx["dst"].astype(str).isin(ids)).sum())
    return {
        "validation_treatment_edges": count(validation_ids),
        "test_treatment_edges": count(test_ids),
    }
