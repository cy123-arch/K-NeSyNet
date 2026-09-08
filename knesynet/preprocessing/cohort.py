from __future__ import annotations
import pandas as pd

def build_cohort(case_manifest_csv: str, projects, label_csv: str):
    cases = pd.read_csv(case_manifest_csv)
    labels = pd.read_csv(label_csv)
    cases = cases[cases["project_id"].isin(projects)].copy()
    cases = cases.drop_duplicates("case_id")
    cases = cases.merge(labels, on="case_id", how="inner")
    available_cols = [c for c in ["imaging_available", "text_available", "genomics_available"] if c in cases.columns]
    if available_cols:
        cases = cases[cases[available_cols].fillna(0).astype(int).sum(axis=1) >= 1]
    return cases.reset_index(drop=True)

def assert_disjoint_splits(df, split_col="split", case_col="case_id"):
    groups = {s: set(g[case_col].astype(str)) for s, g in df.groupby(split_col)}
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            overlap = groups[a] & groups[b]
            if overlap:
                raise AssertionError(f"split overlap between {a} and {b}: {len(overlap)}")
