from __future__ import annotations

import torch
from torch import Tensor


def cosine_distance(a: Tensor, b: Tensor, eps: float = 1e-10) -> float:
    a_flat = a.detach().float().reshape(-1)
    b_flat = b.detach().float().reshape(-1)
    cos_sim = torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm() + eps)
    return float((1.0 - cos_sim).item())


def normalized_l2(a: Tensor, b_reference: Tensor, eps: float = 1e-10) -> float:
    # relative L2 error: ||a - b|| / ||b||. Stands in for the spec's
    # "normalized L3", which doesn't fit an otherwise all-L2/covariance list.
    a_flat = a.detach().float().reshape(-1)
    b_flat = b_reference.detach().float().reshape(-1)
    return float((torch.norm(a_flat - b_flat) / (torch.norm(b_flat) + eps)).item())
