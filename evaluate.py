from __future__ import annotations
import argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader

from knesynet.config import load_config
from knesynet.data import ModelInputDataset
from knesynet.models.knesynet import KNeSyNet
from knesynet.training.inference import collect_scores
from knesynet.evaluation.metrics import (
    candidate_restricted_threshold_predictions, cr_thresholded_f1,
    sample_jaccard, ndcg_at_k, conventional_topk_metrics, full_thresholded_f1
)
from knesynet.utils import device_from_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--inputs", default="data/model_inputs.npz")
    ap.add_argument("--graph", default="data/graph.pt")
    ap.add_argument("--drug-node-map", default="data/drug_node_indices.npy")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = device_from_config()
    ds = ModelInputDataset(args.inputs, args.split)
    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

    graph = torch.load(args.graph, map_location="cpu")
    drug_nodes = torch.tensor(np.load(args.drug_node_map), dtype=torch.long, device=device)
    mcfg = cfg["model"]
    model = KNeSyNet(
        len(graph["entity_to_idx"]), int(cfg["data"]["gene_count"]),
        int(mcfg["embedding_dim"]), int(mcfg["gat_heads"]),
        int(mcfg["gat_layers"]), float(mcfg["dropout"])
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    out = collect_scores(model, loader, graph["edge_index"].to(device), drug_nodes, device)
    threshold = float(ckpt["threshold"])
    k = int(cfg["evaluation"]["top_k"])
    pred = candidate_restricted_threshold_predictions(out["scores"], k, threshold)

    metrics = {
        "candidate_restricted_thresholded_f1_at_k": cr_thresholded_f1(out["labels"], out["scores"], k, threshold),
        "jaccard": sample_jaccard(out["labels"], pred),
        "ndcg_at_k": ndcg_at_k(out["labels"], out["scores"], k),
        "full_80_thresholded_f1": full_thresholded_f1(out["labels"], out["scores"], threshold),
    }
    metrics.update(conventional_topk_metrics(out["labels"], out["scores"], k))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
