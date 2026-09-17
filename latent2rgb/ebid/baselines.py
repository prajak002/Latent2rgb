"""
Single-path baseline metrics for Experiment 1 (Hussain's list, 2026-09-XX):
  1. Raw Latent L2 drift      -- latent2rgb.metrics.latent_drift (already exists)
  2. Cosine distance          -- cosine_distance() below
  3. Normalized L3            -- see NOTE below; implemented as normalized_l2()
  4. Ensemble variance        -- ebid.ensemble.ensemble_spread_stats
  5. Pixel error              -- latent2rgb.metrics.pixel_error (already exists)
  6. Effective rank           -- ebid.ensemble.ensemble_spread_stats
  7. Mahalanobis distance     -- ebid.ensemble.ensemble_spread_stats

NOTE on baseline 3: "Normalized L3" as written is ambiguous -- an L3
(cubic) norm on a 1024-dim ViT embedding is not a standard representation
metric, and every other item on the list is an L2-family or covariance-
family statistic. We implemented the standard scale-invariant reading --
L2 distance divided by the reference vector's own L2 norm, i.e. relative
L2 error -- and labeled the column `normalized_l2` rather than silently
presenting it as literal L3. Flag to Hussain to confirm or correct.
"""

from __future__ import annotations

import torch
from torch import Tensor


def cosine_distance(a: Tensor, b: Tensor, eps: float = 1e-10) -> float:
    a_flat = a.detach().float().reshape(-1)
    b_flat = b.detach().float().reshape(-1)
    cos_sim = torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm() + eps)
    return float((1.0 - cos_sim).item())


def normalized_l2(a: Tensor, b_reference: Tensor, eps: float = 1e-10) -> float:
    """||a - b|| / ||b|| -- see module NOTE re: "Normalized L3"."""
    a_flat = a.detach().float().reshape(-1)
    b_flat = b_reference.detach().float().reshape(-1)
    return float((torch.norm(a_flat - b_flat) / (torch.norm(b_flat) + eps)).item())
