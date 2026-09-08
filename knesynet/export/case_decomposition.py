from __future__ import annotations
import pandas as pd
import torch

def export_case_decomposition(drugs, neural, guide, target, contra, gate, fused, path):
    symbolic = guide + target + contra
    sig_symbolic = torch.sigmoid(symbolic)
    df = pd.DataFrame({
        "drug": drugs,
        "neural": neural.detach().cpu().numpy(),
        "guide": guide.detach().cpu().numpy(),
        "target": target.detach().cpu().numpy(),
        "contra": contra.detach().cpu().numpy(),
        "symbolic_raw": symbolic.detach().cpu().numpy(),
        "symbolic_sigmoid": sig_symbolic.detach().cpu().numpy(),
        "gate": gate.detach().cpu().numpy(),
        "fused": fused.detach().cpu().numpy(),
    })
    df.to_csv(path, index=False)
    return df
