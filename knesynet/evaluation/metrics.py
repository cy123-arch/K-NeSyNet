from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

def topk_indices(scores, k):
    scores = np.asarray(scores)
    return np.argpartition(-scores, kth=k-1, axis=1)[:, :k]

def candidate_restricted_threshold_predictions(scores, k, threshold):
    scores = np.asarray(scores)
    idx = topk_indices(scores, k)
    pred = np.zeros_like(scores, dtype=np.int8)
    rows = np.arange(scores.shape[0])[:, None]
    selected_scores = scores[rows, idx]
    pred[rows, idx] = (selected_scores >= threshold).astype(np.int8)
    return pred

def cr_thresholded_f1(y_true, scores, k, threshold):
    pred = candidate_restricted_threshold_predictions(scores, k, threshold)
    return f1_score(y_true, pred, average="samples", zero_division=0)

def sample_jaccard(y_true, pred):
    return jaccard_score(y_true, pred, average="samples", zero_division=0)

def conventional_topk_metrics(y_true, scores, k):
    y_true = np.asarray(y_true)
    idx = topk_indices(scores, k)
    pred = np.zeros_like(y_true, dtype=np.int8)
    pred[np.arange(y_true.shape[0])[:, None], idx] = 1
    p = precision_score(y_true, pred, average="samples", zero_division=0)
    r = recall_score(y_true, pred, average="samples", zero_division=0)
    f = f1_score(y_true, pred, average="samples", zero_division=0)
    return {"precision_at_k": p, "recall_at_k": r, "f1_at_k": f}

def full_thresholded_f1(y_true, scores, threshold):
    pred = (np.asarray(scores) >= threshold).astype(np.int8)
    return f1_score(y_true, pred, average="samples", zero_division=0)

def ndcg_at_k(y_true, scores, k):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    vals = []
    for t, s in zip(y_true, scores):
        order = np.argsort(-s)[:k]
        rel = t[order].astype(float)
        dcg = sum(v / np.log2(i + 2) for i, v in enumerate(rel))
        ideal_hits = min(int(t.sum()), k)
        if ideal_hits == 0:
            vals.append(0.0)
            continue
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        vals.append(dcg / idcg)
    return float(np.mean(vals))
