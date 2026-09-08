from __future__ import annotations
import re
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Iterable, List

_SPACE = re.compile(r"\s+")
_DOSE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|units?)\b", re.I)

def normalize_surface(text: str) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("–", "-").replace("—", "-")
    text = _DOSE.sub(" ", text)
    text = re.sub(r"[(),;/]+", " ", text)
    text = _SPACE.sub(" ", text).strip()
    return text

@dataclass
class DrugNormalizer:
    alias_to_generic: Dict[str, str]

    @classmethod
    def from_csv(cls, path: str) -> "DrugNormalizer":
        df = pd.read_csv(path)
        required = {"alias", "canonical_drug"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"drug alias table missing columns: {sorted(missing)}")
        mapping = {}
        for row in df.itertuples(index=False):
            mapping[normalize_surface(row.alias)] = str(row.canonical_drug).strip()
        return cls(mapping)

    def map_one(self, raw: str) -> str | None:
        key = normalize_surface(raw)
        return self.alias_to_generic.get(key)

    def split_and_map(self, raw: str) -> List[str]:
        # Combination strings should be represented explicitly in the alias table when possible.
        pieces = re.split(r"\s*(?:\+|&|\band\b)\s*", str(raw), flags=re.I)
        out = []
        for p in pieces:
            mapped = self.map_one(p)
            if mapped and mapped not in out:
                out.append(mapped)
        if not out:
            mapped = self.map_one(raw)
            if mapped:
                out.append(mapped)
        return out
