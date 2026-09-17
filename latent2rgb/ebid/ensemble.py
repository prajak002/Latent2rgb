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
    # context_tokens: [N_ctx, D]. Returns [M, N, D]; member 0 is always the
    # unperturbed (epsilon=0) rollout.
    members = [context_tokens]
    for _ in range(m_members - 1):
        members.append(perturb_tokens(context_tokens, epsilon, generator=generator))
    batched_context = torch.stack(members, dim=0)
    with torch.no_grad():
        pred = predictor.rollout(batched_context, None, k)
    return pred


@dataclass
class EnsembleSpreadStats:
    entropy: float
    effective_rank: float
    ensemble_variance: float
    mahalanobis: float


def ensemble_spread_stats(ensemble: Tensor, reference: Tensor, eps: float = 1e-10) -> EnsembleSpreadStats:
    # ensemble: [M, N, D]. reference: [N, D], e.g. the true latent -- mahalanobis
    # measures how far it sits from the ensemble's own spread.
    m = ensemble.shape[0]
    x = ensemble.detach().float().reshape(m, -1)
    ref = reference.detach().float().reshape(-1)

    ent = spectral_entropy(x, eps=eps)
    eff_rank = float(torch.exp(torch.tensor(ent)).item())

    mean = x.mean(dim=0)
    centered = x - mean
    ens_var = float(centered.var(dim=0, unbiased=True).mean().item()) if m > 1 else 0.0

    if m > 1:
        # economy SVD avoids inverting a singular D x D covariance directly
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        diff = ref - mean
        proj = Vt @ diff
        eigvals = (S ** 2) / max(m - 1, 1)
        maha_sq = torch.sum((proj ** 2) / (eigvals + eps))
        maha = float(torch.sqrt(maha_sq.clamp_min(0.0)).item())
    else:
        maha = float("nan")

    return EnsembleSpreadStats(entropy=ent, effective_rank=eff_rank, ensemble_variance=ens_var, mahalanobis=maha)
