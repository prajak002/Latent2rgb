"""
Controlled Monte Carlo latent perturbation -- the "chaos" sweep, adapted
from pcc_ebid_sim.py's pressure/control/chaos sweep (there: a swept
`chaos` amplitude injecting Gaussian noise into a 2D grid each step).
V-JEPA2's StatePredictor.rollout is a single-shot query, not a per-step
loop (see interfaces.py), so there is no "inject noise every step" point
to hook into for this model family. The nearest architecturally valid
analog is a ONE-TIME perturbation of the re-injected state at the
reinjection point (mirrors Stage F's decode/re-encode round trip, which is
already a one-time perturbation of exactly this kind) -- so `epsilon` here
plays the role PCC's chaos amplitude plays, but as a controlled knob layered
on top of the existing re-injection mechanism rather than a new incompatible
per-step loop.

This is also where EBID's actual quantitative claim gets tested, not just
its lead-time heuristic: EBID predicts (github.com/HussainAther/pcc,
RECOMMENDED_STRUCTURE.md) that entropy deficit scales with the SQUARE of
distance past a Hopf instability threshold -- r_* = sqrt((sigma0-mu)/kappa),
entropy deficit ~ r_*^2, i.e. linear in (sigma0-mu). The analog tested here:
sweep epsilon (the perturbation "distance past equilibrium"), and ask
whether entropy_deficit(epsilon) is well-fit by a similar power law, rather
than only asking whether entropy warns before pixel error does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from torch import Tensor


def perturb_tokens(tokens: Tensor, epsilon: float, generator: torch.Generator | None = None) -> Tensor:
    """
    tokens: [N, D]. Adds Gaussian noise scaled to epsilon * per-token std,
    so epsilon is a dimensionless fraction of the token set's own spread
    (comparable across clips/models with different latent scales) --
    epsilon=0 returns tokens unchanged.
    """
    if epsilon <= 0:
        return tokens
    std = tokens.detach().std()
    noise = torch.randn(tokens.shape, generator=generator, device=tokens.device, dtype=tokens.dtype) if generator is not None \
        else torch.randn_like(tokens)
    return tokens + epsilon * std * noise


@dataclass
class ScalingFitResult:
    epsilons: List[float]
    entropy_deficit: List[float]
    slope: float        # linear fit: entropy_deficit ~ slope * epsilon + intercept
    intercept: float
    r_squared: float     # goodness of the linear (EBID: quadratic-in-amplitude == linear-in-(sigma0-mu)) fit


def fit_entropy_deficit_scaling(epsilons: Sequence[float], entropy_deficit: Sequence[float]) -> ScalingFitResult:
    """
    Linear least-squares fit of entropy_deficit vs epsilon. EBID's closed-form
    result is entropy deficit ~ (sigma0-mu) once past the Hopf threshold --
    i.e. LINEAR in distance-past-threshold, given entropy deficit is already
    quadratic in limit-cycle amplitude and amplitude is sqrt(distance).
    epsilon here plays the role of (sigma0-mu): a linear fit with high R^2
    is the direct empirical analog of that scaling law, not just a
    correlation check.
    """
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
