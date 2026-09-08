from __future__ import annotations
import re
import pandas as pd
from dataclasses import dataclass
from typing import Iterable

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _norm(x):
    return re.sub(r"\s+", " ", str(x or "").strip().lower())

@dataclass
class FrozenTreatmentLexicon:
    terms: tuple[str, ...]

    @classmethod
    def from_csv(cls, path):
        df = pd.read_csv(path)
        terms=set()
        for col in ["normalized_term","alias","canonical_drug","brand_name","brand","salt_form","abbreviation","regimen_term"]:
            if col in df.columns:
                terms.update(_norm(v) for v in df[col].dropna().astype(str))
        terms.discard("")
        return cls(tuple(sorted(terms,key=len,reverse=True)))

    def matches(self,text):
        t=_norm(text)
        return [term for term in self.terms if term in t]

def temporal_eligible(document_date,index_date):
    d=pd.to_datetime(document_date,errors="coerce")
    i=pd.to_datetime(index_date,errors="coerce")
    return bool(pd.notna(d) and pd.notna(i) and d <= i)

def filter_for_prediction_time(text, lexicon: FrozenTreatmentLexicon,
    post_markers: Iterable[str] = (
        "post-treatment","after chemotherapy","following treatment",
        "response to therapy","treatment response",
    )):
    kept=[]; removed=[]; all_terms=set(); post_any=False
    for sent in SENTENCE_SPLIT.split(str(text or "")):
        sent=sent.strip()
        if not sent: continue
        low=_norm(sent)
        terms=lexicon.matches(low)
        post=any(_norm(m) in low for m in post_markers)
        if terms or post:
            removed.append(sent); all_terms.update(terms); post_any = post_any or post
        else:
            kept.append(sent)
    return " ".join(kept).strip(), {
        "treatment_term_detected": int(bool(all_terms)),
        "n_treatment_terms": len(all_terms),
        "n_sentences_removed": len(removed),
        "post_treatment_content_detected": int(post_any),
    }
