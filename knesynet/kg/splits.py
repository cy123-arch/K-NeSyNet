from __future__ import annotations
import pandas as pd

def remove_heldout_treatment_edges(edges: pd.DataFrame, heldout_patient_ids, treatment_relation="treated_with"):
    heldout = set(map(str, heldout_patient_ids))
    e = edges.copy()
    src = e["src"].astype(str)
    dst = e["dst"].astype(str)
    rel = e["relation"].astype(str)
    treatment_like = rel.isin([treatment_relation, f"reverse_{treatment_relation}", f"{treatment_relation}_reverse"])
    incident = src.isin(heldout) | dst.isin(heldout)
    removed = e[treatment_like & incident].copy()
    kept = e[~(treatment_like & incident)].copy()
    return kept.reset_index(drop=True), removed.reset_index(drop=True)

def strictly_inductive_graph(edges: pd.DataFrame, allowed_patient_ids, patient_ids):
    patient_ids = set(map(str, patient_ids))
    allowed = set(map(str, allowed_patient_ids))
    disallowed = patient_ids - allowed
    e = edges.copy()
    mask = ~(
        e["src"].astype(str).isin(disallowed) |
        e["dst"].astype(str).isin(disallowed)
    )
    return e[mask].reset_index(drop=True)

def describe_patient_presence(edges: pd.DataFrame, patient_ids):
    p = set(map(str, patient_ids))
    touched = set(edges["src"].astype(str)) | set(edges["dst"].astype(str))
    return {"present": len(p & touched), "total": len(p), "absent": len(p - touched)}
