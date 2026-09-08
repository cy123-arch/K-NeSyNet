from __future__ import annotations
import math
import pandas as pd

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))

def expected_fused(neural, guide, target, contra, gate):
    symbolic = float(guide) + float(target) + float(contra)
    return float(gate) * float(neural) + (1.0 - float(gate)) * sigmoid(symbolic)

def audit_table(path: str, tolerance=1e-6):
    df = pd.read_csv(path)
    needed = {"drug","neural","guide","target","contra","gate","fused"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    rows = []
    for r in df.itertuples(index=False):
        exp = expected_fused(r.neural, r.guide, r.target, r.contra, r.gate)
        rows.append({
            "drug": r.drug,
            "reported": float(r.fused),
            "expected": exp,
            "delta": float(r.fused) - exp,
            "pass": abs(float(r.fused) - exp) <= tolerance,
        })
    return pd.DataFrame(rows)
