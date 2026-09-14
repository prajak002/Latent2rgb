"""
Spectral entropy of a latent token set -- the EBID state-distribution
measurement, generalized from PCC's grid-histogram Shannon entropy
(pcc/scripts/metrics.py: shannon_entropy_grid) to continuous ViT-style
tokens.

A token set [N, D] (N tokens, D embedding dims) has no natural discrete
histogram. Instead we take the singular values of the centered token
matrix, normalize their squares to a probability distribution over N-1
components, and compute Shannon entropy over that distribution. This is
the standard "effective rank" / spectral-entropy diagnostic used to detect
representation collapse: entropy is maximal when variance is spread evenly
across many directions (rich structure) and collapses toward zero as the
tokens converge onto a low-rank subspace (representation collapse).

This is a state-distribution measurement, not a distance metric -- it
takes ONE token set and returns one number, unlike latent_drift/pixel_error
which compare two. That's the point: it's meant to be computed on the
predicted latent alone, without needing a ground-truth counterpart, which
is exactly the situation Horizon Ladder's rolled-forward-into-the-future
latents are in (no real image exists to check them against).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor


def spectral_entropy(tokens: Tensor, eps: float = 1e-12) -> float:
    """
    tokens: [N, D]. Returns Shannon entropy (nats) of the normalized
    squared singular value spectrum of the row-centered token matrix.
    Ranges [0, log(min(N, D) - 1)]; report alongside that max for context
    when N or D varies across calls (it does here: N depends on
    tokens_per_tubelet, fixed per model, so this is stable within one
    model family but not necessarily comparable across two).
    """
    x = tokens.detach()
    if x.dtype not in (torch.float32, torch.float64):
        x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    # singular values of the centered [N, D] matrix
    s = torch.linalg.svdvals(x)
    p = (s ** 2)
    total = p.sum()
    if total <= eps:
        return 0.0
    p = p / total
    p = p.clamp_min(eps)
    return float(-(p * torch.log(p)).sum().item())


def max_spectral_entropy(n_tokens: int, embed_dim: int) -> float:
    """log(rank) upper bound for spectral_entropy given token-set shape."""
    rank = max(min(n_tokens, embed_dim) - 1, 1)
    return float(np.log(rank))


@dataclass
class EntropyTrace:
    xs: List[float]           # the index axis (horizon k, or reinjection step j)
    entropy: List[float]      # spectral_entropy at each x
    entropy_rate: List[float]  # smoothed d(entropy)/d(x)


def _smooth(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(y) < w:
        return y
    w = int(w) if int(w) % 2 == 1 else int(w) + 1
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


def entropy_trace(xs: Sequence[float], token_sets: Sequence[Tensor], smooth_w: int = 3) -> EntropyTrace:
    """
    xs must be sorted ascending (horizon steps or reinjection j-values).
    token_sets[i] is the [N, D] latent for xs[i].
    """
    xs_arr = np.asarray(xs, dtype=float)
    ent = np.asarray([spectral_entropy(t) for t in token_sets], dtype=float)
    if len(xs_arr) >= 2:
        rate = np.gradient(ent, xs_arr)
        rate = _smooth(rate, smooth_w)
    else:
        rate = np.zeros_like(ent)
    return EntropyTrace(xs=list(xs_arr), entropy=list(ent), entropy_rate=list(rate))
