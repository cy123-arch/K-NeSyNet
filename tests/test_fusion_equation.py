import torch
from knesynet.models.fusion import fuse_scores

def test_letrozole_formula_example():
    neural = torch.tensor([[0.4]])
    symbolic = torch.tensor([[1.0]])
    gate = torch.tensor([[0.0]])
    out = fuse_scores(neural, symbolic, gate)
    assert torch.allclose(out, torch.sigmoid(torch.tensor([[1.0]])))

def test_score_bounds():
    neural = torch.rand(4, 80)
    symbolic = torch.randn(4, 80)
    gate = torch.rand(4, 80)
    out = fuse_scores(neural, symbolic, gate)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
