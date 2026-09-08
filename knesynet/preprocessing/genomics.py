from __future__ import annotations
import pandas as pd
import numpy as np

DEFAULT_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins", "Splice_Site",
    "Translation_Start_Site", "Nonstop_Mutation",
}

def maf_to_multihot(
    maf_path: str,
    gene_vocabulary_csv: str,
    case_col: str = "case_id",
    gene_col: str = "Hugo_Symbol",
    class_col: str = "Variant_Classification",
    retained_classes=None,
):
    retained_classes = set(retained_classes or DEFAULT_CLASSES)
    maf = pd.read_csv(maf_path, sep=None, engine="python", low_memory=False)
    genes = pd.read_csv(gene_vocabulary_csv)["gene"].astype(str).tolist()
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    maf = maf[
        maf[class_col].isin(retained_classes) &
        maf[gene_col].astype(str).isin(gene_to_idx)
    ]

    case_ids = sorted(maf[case_col].astype(str).unique())
    case_to_idx = {c: i for i, c in enumerate(case_ids)}
    x = np.zeros((len(case_ids), len(genes)), dtype=np.float32)

    for r in maf.itertuples(index=False):
        c = str(getattr(r, case_col))
        g = str(getattr(r, gene_col))
        x[case_to_idx[c], gene_to_idx[g]] = 1.0

    return case_ids, x
