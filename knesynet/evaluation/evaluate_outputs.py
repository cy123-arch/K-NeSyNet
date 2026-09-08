from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .metrics import (
    candidate_restricted_threshold_predictions, cr_thresholded_f1,
    sample_jaccard, conventional_topk_metrics, full_thresholded_f1, ndcg_at_k
)

def parse_binary_matrix(series, n):
    rows = []
    for x in series:
        vals = [int(v) for v in str(x).split(";") if str(v) != ""]
        if len(vals) != n:
            raise ValueError(f"expected {n} binary values, got {len(vals)}")
        rows.append(vals)
    return np.asarray(rows, dtype=int)

def evaluate_patient_output_csv(path, threshold, k=10):
    df = pd.read_csv(path)
    drug_cols = [c for c in df.columns if c.startswith("score_")]
    true_cols = [c for c in df.columns if c.startswith("true_")]
    scores = df[drug_cols].to_numpy(float)
    y_true = df[true_cols].to_numpy(int)
    pred = candidate_restricted_threshold_predictions(scores, k, threshold)
    out = {
        "cr_thresholded_f1_at_k": cr_thresholded_f1(y_true, scores, k, threshold),
        "jaccard": sample_jaccard(y_true, pred),
        "ndcg_at_k": ndcg_at_k(y_true, scores, k),
        "full_thresholded_f1": full_thresholded_f1(y_true, scores, threshold),
    }
    out.update(conventional_topk_metrics(y_true, scores, k))
    return out
