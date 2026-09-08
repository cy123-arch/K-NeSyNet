import numpy as np
from sklearn.metrics import f1_score

def select_threshold(y_true, scores, candidates=None):
    candidates = np.asarray(candidates if candidates is not None else np.linspace(0.05, 0.95, 91))
    best_t, best_f1 = None, -1.0
    for t in candidates:
        pred = (scores >= t).astype(int)
        f = f1_score(y_true, pred, average="samples", zero_division=0)
        if f > best_f1:
            best_t, best_f1 = float(t), float(f)
    return best_t, best_f1
