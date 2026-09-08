from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd

def pseudonymize(case_id, salt="knesynet"):
    return hashlib.sha256(f"{salt}:{case_id}".encode()).hexdigest()[:20]

def export_patient_outputs(
    case_ids,
    project_ids,
    drug_names,
    y_true,
    scores,
    threshold,
    guideline_flags,
    contraindication_flags,
    path,
    k=10,
):
    rows = []
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    for i, case_id in enumerate(case_ids):
        order = np.argsort(-scores[i])
        top = order[:k]
        thresholded = [drug_names[j] for j in top if scores[i, j] >= threshold]
        true = [drug_names[j] for j in np.where(y_true[i] > 0)[0]]
        rows.append({
            "case_id": pseudonymize(case_id),
            "project_id": project_ids[i],
            "true_labels": ";".join(true),
            "top10_drugs": ";".join(drug_names[j] for j in top),
            "top10_scores": ";".join(f"{scores[i,j]:.8f}" for j in top),
            "thresholded_positive_drugs": ";".join(thresholded),
            "guideline_flags_top10": ";".join(str(int(guideline_flags[i,j])) for j in top),
            "contraindication_flags_top10": ";".join(str(int(contraindication_flags[i,j])) for j in top),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
