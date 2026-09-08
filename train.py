from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from knesynet.config import load_config
from knesynet.data import ModelInputDataset
from knesynet.models.knesynet import KNeSyNet
from knesynet.training.losses import composite_loss
from knesynet.training.trainer import train_epoch, EarlyStopper
from knesynet.training.inference import collect_scores
from knesynet.evaluation.thresholds import select_threshold
from knesynet.evaluation.metrics import cr_thresholded_f1
from knesynet.utils import seed_everything, device_from_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--inputs", default="data/model_inputs.npz")
    ap.add_argument("--graph", default="data/graph.pt")
    ap.add_argument("--drug-node-map", default="data/drug_node_indices.npy")
    ap.add_argument("--out", default="checkpoints/best.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    device = device_from_config()

    train_ds = ModelInputDataset(args.inputs, "train")
    val_ds = ModelInputDataset(args.inputs, "validation")
    bs = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    graph = torch.load(args.graph, map_location="cpu")
    edge_index = graph["edge_index"].to(device)
    drug_nodes = torch.tensor(np.load(args.drug_node_map), dtype=torch.long, device=device)

    mcfg = cfg["model"]
    model = KNeSyNet(
        num_entities=len(graph["entity_to_idx"]),
        gene_count=int(cfg["data"]["gene_count"]),
        embedding_dim=int(mcfg["embedding_dim"]),
        gat_heads=int(mcfg["gat_heads"]),
        gat_layers=int(mcfg["gat_layers"]),
        dropout=float(mcfg["dropout"]),
    ).to(device)

    tcfg = cfg["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(tcfg["max_epochs"]))
    stopper = EarlyStopper(int(tcfg["early_stopping_patience"]))

    def loss_fn(scores, labels, contra_indicator, model_):
        return composite_loss(
            scores, labels, contra_indicator, model_,
            alpha=float(tcfg["alpha_safety"]),
            beta=float(tcfg["beta_l2"]),
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(tcfg["max_epochs"]) + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, edge_index, drug_nodes)
        val = collect_scores(model, val_loader, edge_index, drug_nodes, device)
        threshold, _ = select_threshold(val["labels"], val["scores"])
        val_f1 = cr_thresholded_f1(
            val["labels"], val["scores"], int(cfg["evaluation"]["top_k"]), threshold
        )
        improved, should_stop = stopper.step(val_f1)
        if improved:
            torch.save({
                "model_state": model.state_dict(),
                "threshold": threshold,
                "config": cfg,
                "epoch": epoch,
                "validation_cr_f1": val_f1,
            }, args.out)
        scheduler.step()
        print(f"epoch={epoch:03d} loss={train_loss:.6f} val_cr_f1={val_f1:.6f} threshold={threshold:.4f}")
        if should_stop:
            break

if __name__ == "__main__":
    main()
