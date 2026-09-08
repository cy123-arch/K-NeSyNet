from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image

class FrozenResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.transform = weights.transforms()
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_tiles(self, tiles, device=None):
        device = device or next(self.parameters()).device
        batch = torch.stack([self.transform(img.convert("RGB")) for img in tiles]).to(device)
        z = self.backbone(batch).flatten(1)
        return z

def mean_pool_case(slide_embeddings):
    if not slide_embeddings:
        return torch.zeros(2048)
    return torch.stack(slide_embeddings).mean(0)

def sample_tiles_from_image(image_path: str, tile_size=224, max_tiles=256):
    # Lightweight interface for pre-extracted raster slides/tiles.
    # Whole-slide SVS extraction can be supplied through OpenSlide using the same return contract.
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    tiles = []
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            crop = img.crop((x, y, x + tile_size, y + tile_size))
            arr = np.asarray(crop)
            if arr.mean() > 245:
                continue
            tiles.append(crop)
            if len(tiles) >= max_tiles:
                return tiles
    return tiles
