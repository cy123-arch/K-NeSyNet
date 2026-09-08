import numpy as np
from knesynet.evaluation.metrics import candidate_restricted_threshold_predictions, conventional_topk_metrics

def test_candidate_restriction():
    scores = np.array([[0.9,0.8,0.7,0.6]])
    pred = candidate_restricted_threshold_predictions(scores, k=2, threshold=0.75)
    assert pred.tolist() == [[1,1,0,0]]

def test_conventional_topk_shape():
    y = np.array([[1,0,0,1],[0,1,0,0]])
    s = np.array([[0.9,0.1,0.2,0.8],[0.3,0.9,0.2,0.1]])
    m = conventional_topk_metrics(y, s, 2)
    assert set(m) == {"precision_at_k","recall_at_k","f1_at_k"}
