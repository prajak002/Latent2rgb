"""
Stage E honesty checks and Stage F full instruments. Everything here is
generic: it takes an Encoder/Decoder/StatePredictor/ClipSource satisfying
interfaces.py and returns numbers. None of it is specific to the synthetic
backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import Tensor

from .data import RolloutBuilder
from .interfaces import ClipSource, Decoder, Encoder, StatePredictor
from .metrics import pixel_l2


# ---------------------------------------------------------------------------
# Stage E, Check 1: context leakage
# ---------------------------------------------------------------------------


@dataclass
class LeakageResult:
    same_clip_error: float
    cross_clip_error: float
    leakage_gap: float  # cross_clip_error - same_clip_error; should be >> 0


def context_leakage_check(
    decoder: Decoder,
    encoder: Encoder,
    clip_source: ClipSource,
    clip_id_a: str,
    clip_id_b: str,
    t: int,
) -> LeakageResult:
    """
    Decode the latent from clip_id_a's frame at time t, but compare it
    against clip_id_b's frame at time t (a different clip). If the decoder
    is leaking context (e.g. memorizing/hallucinating clip identity instead
    of reading the latent), decoding clip_a's latent will still look
    unrelated to clip_b -- same_clip_error should be low and cross_clip_error
    high. A decoder that ignores the latent and just reproduces a generic
    "plausible" output would show a small leakage_gap.
    """
    batch_a = clip_source.get_clip(clip_id_a, t, k_max=0)
    batch_b = clip_source.get_clip(clip_id_b, t, k_max=0)
    frame_a = batch_a.frames[0]
    frame_b = batch_b.frames[0]

    with torch.no_grad():
        latent_a = encoder.encode(frame_a.unsqueeze(0))
        decoded_a = decoder.decode(latent_a).squeeze(0)

    same = pixel_l2(decoded_a, frame_a)
    cross = pixel_l2(decoded_a, frame_b)
    return LeakageResult(same_clip_error=same, cross_clip_error=cross, leakage_gap=cross - same)


# ---------------------------------------------------------------------------
# Stage E, Check 2: snapping
# ---------------------------------------------------------------------------


@dataclass
class SnapResult:
    alphas: List[float]
    pixel_errors_vs_endpoint_a: List[float]
    max_second_derivative: float  # large value => snapping rather than smooth interpolation


def snapping_check(
    decoder: Decoder,
    encoder: Encoder,
    frame_a: Tensor,
    frame_b: Tensor,
    n_steps: int = 9,
) -> SnapResult:
    """
    Interpolate linearly between encode(frame_a) and encode(frame_b) in
    latent space, decode each interpolant, and measure decoded output
    against frame_a as alpha goes 0->1. A well-behaved decoder degrades
    smoothly (roughly monotonic, low curvature). A decoder that "snaps" to
    one plausible real frame or the other shows a plateau-then-jump: near-
    zero second derivative almost everywhere, with a spike at the snap
    point.
    """
    with torch.no_grad():
        za = encoder.encode(frame_a.unsqueeze(0))
        zb = encoder.encode(frame_b.unsqueeze(0))

    alphas = [i / (n_steps - 1) for i in range(n_steps)]
    errors = []
    for alpha in alphas:
        z = (1 - alpha) * za + alpha * zb
        with torch.no_grad():
            decoded = decoder.decode(z).squeeze(0)
        errors.append(pixel_l2(decoded, frame_a))

    # discrete second derivative of the error curve; a snap shows up as a
    # large spike relative to neighboring segments
    second_deriv = [
        abs(errors[i + 1] - 2 * errors[i] + errors[i - 1]) for i in range(1, len(errors) - 1)
    ]
    max_curv = max(second_deriv) if second_deriv else 0.0

    return SnapResult(alphas=alphas, pixel_errors_vs_endpoint_a=errors, max_second_derivative=max_curv)


# ---------------------------------------------------------------------------
# Stage F: counterfactual action divergence
# ---------------------------------------------------------------------------


@dataclass
class CounterfactualResult:
    k: int
    cf_latent_delta: float
    cf_pixel_delta: float
    ratio: float  # cf_pixel_delta / cf_latent_delta; near 0 => decoder overwriting with prior


def counterfactual_action_divergence(
    predictor: StatePredictor,
    decoder: Decoder,
    latent_t: Tensor,
    actions_a: Tensor,
    actions_b: Tensor,
    k: int,
) -> CounterfactualResult:
    """
    Roll the SAME starting latent forward under two different action
    sequences and compare both the resulting latents and their decoded
    frames. actions_a, actions_b: [1, k, A].

    Interpretation:
      - cf_pixel_delta << cf_latent_delta: the decoder is not faithfully
        rendering the action-induced state difference -- it's overwriting
        predicted state with a generic prior.
      - cf_pixel_delta >> cf_latent_delta (after accounting for pixel/latent
        scale): the decoder may be inventing detail not supported by the
        state difference.
    """
    with torch.no_grad():
        latent_a = predictor.rollout(latent_t, actions_a, k)
        latent_b = predictor.rollout(latent_t, actions_b, k)
        decoded_a = decoder.decode(latent_a).squeeze(0)
        decoded_b = decoder.decode(latent_b).squeeze(0)

    latent_delta = pixel_l2(latent_a, latent_b)
    pixel_delta = pixel_l2(decoded_a, decoded_b)
    ratio = pixel_delta / latent_delta if latent_delta > 1e-8 else float("nan")
    return CounterfactualResult(k=k, cf_latent_delta=latent_delta, cf_pixel_delta=pixel_delta, ratio=ratio)


# ---------------------------------------------------------------------------
# Stage F: re-injection divergence (the headline instrument)
# ---------------------------------------------------------------------------


@dataclass
class ReinjectionResult:
    k_reinject: int
    j_values: List[int]
    trajectory_distance: List[float]  # distance(pure-latent path, re-injected path) at each j
    divergence_rate: float  # slope of trajectory_distance vs j (linear fit)


def reinjection_divergence(
    encoder: Encoder,
    decoder: Decoder,
    predictor: StatePredictor,
    latent_t: Tensor,
    actions: Tensor,
    k_reinject: int,
    j_values: Sequence[int],
) -> ReinjectionResult:
    """
    Roll forward to k_reinject in pure latent space, decode, re-encode
    (round-tripping through pixels once), then resume the rollout from the
    re-encoded state. Compare against the pure-latent rollout (never
    decoded) at j additional steps past the re-injection point.

    This is strictly stronger than a single-shot decode/re-encode/compare:
    it measures how much a one-time pixel round-trip perturbs everything
    that happens afterward, i.e. whether the decoder's error compounds
    through the dynamics rather than staying a fixed one-time discrepancy.

    actions: [1, k_reinject + max(j_values), A]
    """
    j_max = max(j_values)
    total_actions = actions.shape[1]
    assert total_actions >= k_reinject + j_max, "not enough actions for requested horizon"

    with torch.no_grad():
        latent_at_reinject = predictor.rollout(latent_t, actions[:, :k_reinject], k_reinject)
        decoded = decoder.decode(latent_at_reinject)
        reencoded = encoder.encode(decoded)

        remaining_actions = actions[:, k_reinject : k_reinject + j_max]

        distances = []
        for j in j_values:
            pure_path = predictor.rollout(latent_t, actions[:, : k_reinject + j], k_reinject + j)
            reinjected_path = predictor.rollout(reencoded, remaining_actions[:, :j], j)
            distances.append(pixel_l2(pure_path, reinjected_path))

    # linear fit of distance vs j for the divergence rate
    if len(j_values) >= 2:
        xs = torch.tensor(j_values, dtype=torch.float32)
        ys = torch.tensor(distances, dtype=torch.float32)
        x_mean, y_mean = xs.mean(), ys.mean()
        slope = ((xs - x_mean) * (ys - y_mean)).sum() / ((xs - x_mean) ** 2).sum().clamp_min(1e-8)
        rate = slope.item()
    else:
        rate = float("nan")

    return ReinjectionResult(
        k_reinject=k_reinject, j_values=list(j_values), trajectory_distance=distances, divergence_rate=rate
    )
