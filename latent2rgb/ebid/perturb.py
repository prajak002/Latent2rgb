from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from torch import Tensor


def perturb_tokens(tokens: Tensor, epsilon: float, generator: torch.Generator | None = None) -> Tensor:
    # Gaussian noise scaled to epsilon * per-token std. epsilon=0 -> unchanged.
    if epsilon <= 0:
        return tokens
    std = tokens.detach().std()
    if generator is not None:
        # a CPU generator can't draw directly into an MPS tensor -- sample
        # on the generator's own device, then move.
        noise = torch.randn(tokens.shape, generator=generator, device=generator.device, dtype=tokens.dtype)
        noise = noise.to(tokens.device)
    else:
        noise = torch.randn_like(tokens)
    return tokens + epsilon * std * noise


@dataclass
class ScalingFitResult:
    epsilons: List[float]
    entropy_deficit: List[float]
    slope: float
    intercept: float
    r_squared: float


def fit_entropy_deficit_scaling(epsilons: Sequence[float], entropy_deficit: Sequence[float]) -> ScalingFitResult:
    x = np.asarray(epsilons, dtype=float)
    y = np.asarray(entropy_deficit, dtype=float)
    if len(x) < 2:
        return ScalingFitResult(epsilons=list(x), entropy_deficit=list(y), slope=float("nan"),
                                 intercept=float("nan"), r_squared=float("nan"))
    A = np.stack([x, np.ones_like(x)], axis=1)
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return ScalingFitResult(epsilons=list(x), entropy_deficit=list(y), slope=float(slope),
                             intercept=float(intercept), r_squared=r2)
