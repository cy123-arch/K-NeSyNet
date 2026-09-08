from __future__ import annotations
import numpy as np
import torch

@torch.no_grad()
def collect_scores(model, loader, edge_index, drug_node_indices, device):
    model.eval()
    scores, labels = [], []
    case_ids, projects = [], []
    guides, contras = [], []
    decompositions = []
    for batch in loader:
        tensor_batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}
        out = model(
            tensor_batch["image"], tensor_batch["text"], tensor_batch["genomic"], tensor_batch["mask"],
            edge_index, drug_node_indices,
            tensor_batch["guide"], tensor_batch["target"], tensor_batch["contra"],
        )
        scores.append(out["score"].cpu().numpy())
        labels.append(tensor_batch["labels"].cpu().numpy())
        guides.append(tensor_batch["guide"].cpu().numpy())
        contras.append((tensor_batch["contra"] < 0).float().cpu().numpy())
        case_ids.extend(batch["case_id"])
        projects.extend(batch["project_id"])
        decompositions.append({
            "neural": out["neural"].cpu().numpy(),
            "symbolic": out["symbolic"].cpu().numpy(),
            "gate": out["gate"].cpu().numpy(),
        })
    return {
        "scores": np.concatenate(scores, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "guide_flags": np.concatenate(guides, axis=0),
        "contra_flags": np.concatenate(contras, axis=0),
        "case_ids": list(case_ids),
        "project_ids": list(projects),
        "decomposition": decompositions,
    }
