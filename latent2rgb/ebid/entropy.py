from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor


def spectral_entropy(tokens: Tensor, eps: float = 1e-12) -> float:
    # tokens: [N, D]. Shannon entropy (nats) of the normalized squared
    # singular values of the row-centered token matrix.
    x = tokens.detach()
    if x.dtype not in (torch.float32, torch.float64):
        x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = (s ** 2)
    total = p.sum()
    if total <= eps:
        return 0.0
    p = p / total
    p = p.clamp_min(eps)
    return float(-(p * torch.log(p)).sum().item())


def max_spectral_entropy(n_tokens: int, embed_dim: int) -> float:
    rank = max(min(n_tokens, embed_dim) - 1, 1)
    return float(np.log(rank))


@dataclass
class EntropyTrace:
    xs: List[float]
    entropy: List[float]
    entropy_rate: List[float]


def _smooth(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(y) < w:
        return y
    w = int(w) if int(w) % 2 == 1 else int(w) + 1
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


def entropy_trace(xs: Sequence[float], token_sets: Sequence[Tensor], smooth_w: int = 3) -> EntropyTrace:
    xs_arr = np.asarray(xs, dtype=float)
    ent = np.asarray([spectral_entropy(t) for t in token_sets], dtype=float)
    if len(xs_arr) >= 2:
        rate = np.gradient(ent, xs_arr)
        rate = _smooth(rate, smooth_w)
    else:
        rate = np.zeros_like(ent)
    return EntropyTrace(xs=list(xs_arr), entropy=list(ent), entropy_rate=list(rate))
