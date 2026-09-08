from __future__ import annotations
import argparse, pandas as pd, numpy as np, torch
from torch.utils.data import DataLoader
from knesynet.config import load_config
from knesynet.data import ModelInputDataset
from knesynet.models.knesynet import KNeSyNet
from knesynet.training.inference import collect_scores
from knesynet.export.patient_outputs import export_patient_outputs
from knesynet.utils import device_from_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--inputs", default="data/model_inputs.npz")
    ap.add_argument("--graph", default="data/graph.pt")
    ap.add_argument("--drug-node-map", default="data/drug_node_indices.npy")
    ap.add_argument("--drug-vocabulary", default="data/drug_vocabulary.csv")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="outputs/patient_level_test_predictions.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = device_from_config()
    ds = ModelInputDataset(args.inputs, "test")
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
    o = collect_scores(model, loader, graph["edge_index"].to(device), drug_nodes, device)

    drugs = pd.read_csv(args.drug_vocabulary)["drug"].astype(str).tolist()
    export_patient_outputs(
        o["case_ids"], o["project_ids"], drugs, o["labels"], o["scores"],
        float(ckpt["threshold"]), o["guide_flags"], o["contra_flags"], args.out,
        k=int(cfg["evaluation"]["top_k"])
    )
    print(args.out)

if __name__ == "__main__":
    main()
