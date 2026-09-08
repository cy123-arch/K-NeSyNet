import numpy as np

def cvr_at_k(contra_flags, k):
    a = np.asarray(contra_flags)[:, :k]
    return float(a.mean())

def cgc_at_k(guideline_flags, k):
    a = np.asarray(guideline_flags)[:, :k]
    return float(a.mean())
