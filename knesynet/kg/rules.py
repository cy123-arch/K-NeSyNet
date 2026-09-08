from __future__ import annotations
import pandas as pd
from collections import defaultdict

class RuleTables:
    def __init__(self, guideline_csv, drug_targets_csv, contraindication_csv, eps=1e-8):
        self.eps = eps
        self.guidelines = pd.read_csv(guideline_csv)
        self.targets = defaultdict(set)
        self.contra = defaultdict(set)

        dt = pd.read_csv(drug_targets_csv)
        for r in dt.itertuples(index=False):
            self.targets[str(r.drug)].add(str(r.gene))

        ct = pd.read_csv(contraindication_csv)
        for r in ct.itertuples(index=False):
            if hasattr(r, "gene"):
                self.contra[str(r.drug)].add(str(r.gene))

    def guideline_score(self, drug, cancer_type, stage):
        g = self.guidelines
        mask = (
            g["drug"].astype(str).eq(str(drug)) &
            g["cancer_type"].astype(str).eq(str(cancer_type))
        )
        if "stage" in g.columns:
            stage_col = g["stage"].fillna("*").astype(str)
            mask &= stage_col.isin([str(stage), "*"])
        return float(mask.any())

    def target_score(self, drug, mutated_genes):
        t = self.targets.get(str(drug), set())
        if not t:
            return 0.0
        return len(t & set(mutated_genes)) / (len(t) + self.eps)

    def contraindication_score(self, drug, mutated_genes):
        c = self.contra.get(str(drug), set())
        if not c:
            return 0.0
        return -len(c & set(mutated_genes)) / (len(c) + self.eps)

    def vector(self, drug, cancer_type, stage, mutated_genes):
        guide = self.guideline_score(drug, cancer_type, stage)
        target = self.target_score(drug, mutated_genes)
        contra = self.contraindication_score(drug, mutated_genes)
        return guide, target, contra
