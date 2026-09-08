from __future__ import annotations
import pandas as pd
from collections import defaultdict
from .drug_normalization import DrugNormalizer

def build_patient_labels(
    treatments_csv: str,
    aliases_csv: str,
    vocabulary_csv: str,
    case_col: str = "case_id",
    drug_col: str = "therapeutic_agent",
    diagnosis_col: str = "diagnosis_date",
    treatment_col: str = "treatment_date",
):
    tx = pd.read_csv(treatments_csv)
    vocab = pd.read_csv(vocabulary_csv)
    drug_set = set(vocab["drug"].astype(str))
    normalizer = DrugNormalizer.from_csv(aliases_csv)

    tx[diagnosis_col] = pd.to_datetime(tx[diagnosis_col], errors="coerce")
    tx[treatment_col] = pd.to_datetime(tx[treatment_col], errors="coerce")

    labels = defaultdict(set)
    audit = []
    for r in tx.itertuples(index=False):
        case_id = str(getattr(r, case_col))
        diagnosis = getattr(r, diagnosis_col)
        treatment = getattr(r, treatment_col)
        raw = getattr(r, drug_col)

        temporal_ok = pd.notna(diagnosis) and pd.notna(treatment) and treatment >= diagnosis
        mapped = normalizer.split_and_map(raw) if temporal_ok else []
        mapped = [d for d in mapped if d in drug_set]

        for d in mapped:
            labels[case_id].add(d)

        audit.append({
            "case_id": case_id,
            "raw_agent": raw,
            "temporal_ok": bool(temporal_ok),
            "mapped_drugs": ";".join(mapped),
        })

    label_rows = [{"case_id": k, "true_labels": ";".join(sorted(v))}
                  for k, v in sorted(labels.items())]
    return pd.DataFrame(label_rows), pd.DataFrame(audit)
