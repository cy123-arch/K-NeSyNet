from __future__ import annotations
import pandas as pd
import torch
from dataclasses import dataclass

@dataclass
class GraphBundle:
    entity_to_idx: dict
    idx_to_entity: list
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    relation_to_idx: dict
    edge_table: pd.DataFrame

def build_graph(entity_csv: str, edge_csv: str) -> GraphBundle:
    entities = pd.read_csv(entity_csv)
    edges = pd.read_csv(edge_csv)
    required_e = {"entity_id", "entity_type"}
    required_r = {"src", "relation", "dst"}
    if not required_e.issubset(entities.columns):
        raise ValueError(f"entities need {required_e}")
    if not required_r.issubset(edges.columns):
        raise ValueError(f"edges need {required_r}")

    ids = entities["entity_id"].astype(str).tolist()
    entity_to_idx = {e: i for i, e in enumerate(ids)}
    rels = sorted(edges["relation"].astype(str).unique())
    relation_to_idx = {r: i for i, r in enumerate(rels)}

    valid = edges["src"].astype(str).isin(entity_to_idx) & edges["dst"].astype(str).isin(entity_to_idx)
    edges = edges.loc[valid].copy()
    src = [entity_to_idx[str(x)] for x in edges["src"]]
    dst = [entity_to_idx[str(x)] for x in edges["dst"]]
    et = [relation_to_idx[str(x)] for x in edges["relation"]]
    return GraphBundle(
        entity_to_idx=entity_to_idx,
        idx_to_entity=ids,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_type=torch.tensor(et, dtype=torch.long),
        relation_to_idx=relation_to_idx,
        edge_table=edges.reset_index(drop=True),
    )
