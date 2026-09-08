from __future__ import annotations
import re
import hashlib
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, List, Tuple

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _norm(x: str) -> str:
    return re.sub(r"\s+", " ", str(x or "").lower()).strip()

@dataclass
class TreatmentLexicon:
    terms: set[str]

    @classmethod
    def from_alias_csv(cls, path: str) -> "TreatmentLexicon":
        df = pd.read_csv(path)
        terms = set()
        for col in ["alias", "canonical_drug", "brand", "regimen_term"]:
            if col in df.columns:
                terms.update(_norm(x) for x in df[col].dropna().astype(str))
        terms.discard("")
        return cls(terms)

    def matched_terms(self, text: str) -> List[str]:
        t = _norm(text)
        return sorted(term for term in self.terms if term and term in t)

def filter_document(
    text: str,
    lexicon: TreatmentLexicon,
    post_treatment_markers: Iterable[str] = (
        "post-treatment", "after chemotherapy", "following treatment",
        "response to therapy", "treatment response",
    ),
):
    kept, removed = [], []
    for sent in SENTENCE_SPLIT.split(str(text or "")):
        s = sent.strip()
        if not s:
            continue
        matches = lexicon.matched_terms(s)
        post = any(_norm(m) in _norm(s) for m in post_treatment_markers)
        if matches or post:
            removed.append((s, matches, post))
        else:
            kept.append(s)
    return " ".join(kept), removed

def document_audit_row(case_id, document_id, document_date, index_date, raw_text, filtered_text, removed):
    h = hashlib.sha256(str(document_id).encode()).hexdigest()[:16]
    temporal_eligible = pd.to_datetime(document_date, errors="coerce") <= pd.to_datetime(index_date, errors="coerce")
    terms = sorted({t for _, matches, _ in removed for t in matches})
    return {
        "case_id_hash": hashlib.sha256(str(case_id).encode()).hexdigest()[:16],
        "document_id_hash": h,
        "document_date": document_date,
        "index_date": index_date,
        "temporal_eligible": bool(temporal_eligible),
        "treatment_terms_detected": ";".join(terms),
        "sentences_removed": len(removed),
        "final_characters": len(filtered_text),
    }
