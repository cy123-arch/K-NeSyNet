from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

class ModelInputDataset(Dataset):
    """
    NPZ schema:
      image        float32 [N,512]
      text         float32 [N,768]
      genomic      float32 [N,10586]
      mask         float32 [N,3]
      labels       float32 [N,80]
      guide        float32 [N,80]
      target       float32 [N,80]
      contra       float32 [N,80]
      contra_indicator float32 [N,80]
      case_id      str [N]
      project_id   str [N]
      split        str [N]
    """
    def __init__(self, npz_path, split=None):
        z = np.load(npz_path, allow_pickle=True)
        required = [
            "image","text","genomic","mask","labels","guide","target","contra",
            "contra_indicator","case_id","project_id","split"
        ]
        missing = [k for k in required if k not in z]
        if missing:
            raise ValueError(f"model input NPZ missing: {missing}")
        self.arr = {k: z[k] for k in required}
        if split is None:
            self.indices = np.arange(len(self.arr["case_id"]))
        else:
            self.indices = np.where(self.arr["split"].astype(str) == str(split))[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        return {
            "image": torch.tensor(self.arr["image"][j], dtype=torch.float32),
            "text": torch.tensor(self.arr["text"][j], dtype=torch.float32),
            "genomic": torch.tensor(self.arr["genomic"][j], dtype=torch.float32),
            "mask": torch.tensor(self.arr["mask"][j], dtype=torch.float32),
            "labels": torch.tensor(self.arr["labels"][j], dtype=torch.float32),
            "guide": torch.tensor(self.arr["guide"][j], dtype=torch.float32),
            "target": torch.tensor(self.arr["target"][j], dtype=torch.float32),
            "contra": torch.tensor(self.arr["contra"][j], dtype=torch.float32),
            "contra_indicator": torch.tensor(self.arr["contra_indicator"][j], dtype=torch.float32),
            "case_id": str(self.arr["case_id"][j]),
            "project_id": str(self.arr["project_id"][j]),
        }
