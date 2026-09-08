from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class EarlyStopper:
    patience: int = 12
    best: float = float("-inf")
    bad_epochs: int = 0

    def step(self, metric):
        if metric > self.best:
            self.best = float(metric)
            self.bad_epochs = 0
            return True, False
        self.bad_epochs += 1
        return False, self.bad_epochs >= self.patience

def _to_device(batch, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

def train_epoch(model, loader, optimizer, loss_fn, device, edge_index, drug_node_indices):
    """Train one epoch using the experiment-level graph objects.

    edge_index and drug_node_indices are global graph quantities and are therefore
    supplied once per epoch rather than being duplicated inside every patient batch.
    """
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(
            batch["image"], batch["text"], batch["genomic"], batch["mask"],
            edge_index, drug_node_indices,
            batch["guide"], batch["target"], batch["contra"],
        )
        loss, _ = loss_fn(out["score"], batch["labels"], batch["contra_indicator"], model)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        n += 1
    return total / max(n, 1)
