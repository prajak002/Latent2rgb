"""
Ensemble construction and ensemble-spread statistics -- Experiment 1
(Hussain Ather, PCC/EBID collaboration): "Does entropy-based ensemble
spread grow with V-JEPA2 rollout horizon? Does EBID predict pixel-space
failure better than latent L2?"

V-JEPA2's predictor is frozen and deterministic (no dropout, no sampling),
so there is no native notion of an ensemble the way an MC-dropout or a
diffusion sampler has one. The ensemble here is constructed the way
ensemble weather forecasting constructs one: perturb the initial condition
(here, the encoded context tokens) with small Monte Carlo noise, and treat
the resulting spread of rollout outcomes as the ensemble. Member 0 is
always the unperturbed ("pure") rollout, matching what latent2rgb's other
stages already treat as the reference prediction.

All three ensemble statistics below (spectral entropy, effective rank,
Mahalanobis distance) come from ONE economy SVD of the centered,
flattened ensemble -- cheap, and it's the same SVD-entropy machinery as
ebid/entropy.py, just applied with ensemble members as rows instead of
latent tokens as rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .entropy import spectral_entropy
from .perturb import perturb_tokens


def build_ensemble_predictions(
    predictor, context_tokens: Tensor, k: int, m_members: int, epsilon: float,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    context_tokens: [N_ctx, D] (single clip, no batch dim).
    Returns predicted tokens for horizon k, ensemble-batched: [M, N, D].
    Member 0 is always the unperturbed rollout (epsilon=0).
    """
    members = [context_tokens]
    for _ in range(m_members - 1):
        members.append(perturb_tokens(context_tokens, epsilon, generator=generator))
    batched_context = torch.stack(members, dim=0)  # [M, N_ctx, D]
    with torch.no_grad():
        pred = predictor.rollout(batched_context, None, k)  # [M, N, D]
    return pred


@dataclass
class EnsembleSpreadStats:
    entropy: float           # EBID: spectral entropy of the ensemble's spread (grows = ensemble disagrees more)
    effective_rank: float    # exp(entropy) -- Roy & Vetterli's effective rank of the perturbation covariance
    ensemble_variance: float  # mean per-dimension variance across members (baseline 4)
    mahalanobis: float       # distance of `reference` from the ensemble mean, in the ensemble's own metric (baseline 7)


def ensemble_spread_stats(ensemble: Tensor, reference: Tensor, eps: float = 1e-10) -> EnsembleSpreadStats:
    """
    ensemble: [M, N, D] -- M perturbed rollout outcomes at one horizon.
    reference: [N, D] -- typically the true (encoded ground-truth) latent;
               Mahalanobis measures how far the truth sits from the
               ensemble's own spread, in the ensemble's own covariance
               metric (standard ensemble-forecast verification).
    """
    m = ensemble.shape[0]
    x = ensemble.detach().float().reshape(m, -1)          # [M, N*D]
    ref = reference.detach().float().reshape(-1)           # [N*D]

    ent = spectral_entropy(x, eps=eps)
    eff_rank = float(torch.exp(torch.tensor(ent)).item())

    mean = x.mean(dim=0)                                    # [N*D]
    centered = x - mean
    ens_var = float(centered.var(dim=0, unbiased=True).mean().item()) if m > 1 else 0.0

    # Mahalanobis distance in the ensemble's own (rank <= M-1) subspace via
    # economy SVD -- avoids inverting a singular D x D covariance directly.
    if m > 1:
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)  # S: [r], Vt: [r, N*D]
        diff = ref - mean
        proj = Vt @ diff                                    # [r]
        eigvals = (S ** 2) / max(m - 1, 1)
        maha_sq = torch.sum((proj ** 2) / (eigvals + eps))
        maha = float(torch.sqrt(maha_sq.clamp_min(0.0)).item())
    else:
        maha = float("nan")

    return EnsembleSpreadStats(entropy=ent, effective_rank=eff_rank, ensemble_variance=ens_var, mahalanobis=maha)
