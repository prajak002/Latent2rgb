"""
Core metrics, generic across any (Encoder, Decoder, StatePredictor).
Every number downstream is reported relative to `floor` -- see Block 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor


def l2(a: Tensor, b: Tensor) -> float:
    return torch.sqrt(torch.sum((a - b) ** 2)).item()


def pixel_l2(a: Tensor, b: Tensor) -> float:
    """Mean per-pixel L2 (RMSE-like), invariant to image resolution."""
    return torch.sqrt(torch.mean((a - b) ** 2)).item()


def latent_drift(predicted_latent: Tensor, true_latent: Tensor) -> float:
    return l2(predicted_latent, true_latent)


def pixel_error(decoded_frame: Tensor, true_frame: Tensor) -> float:
    return pixel_l2(decoded_frame, true_frame)


def decoder_floor(decoder, encoder, tubelet_clips: Tensor) -> float:
    """
    floor = decoder reconstruction error on REAL tubelets (encode then
    decode, no rollout involved). tubelet_clips: [N, C, T, H, W], T=tubelet_size.
    """
    with torch.no_grad():
        latents = encoder.encode(tubelet_clips)
        recon = decoder.decode(latents)
        return pixel_l2(recon, tubelet_clips)


def excess_over_floor(pixel_err: float, floor: float) -> float:
    return pixel_err - floor


@dataclass
class SeparationResult:
    slope: float
    intercept: float
    fit_horizons: List[int]
    eval_horizon: int
    predicted_pixel_error_at_eval: float
    actual_pixel_error_at_eval: float
    residual: float  # actual - predicted; the headline separation statistic


def fit_separation_statistic(
    df, fit_k: List[int], eval_k: int, x_col: str = "latent_drift", y_col: str = "pixel_error"
) -> SeparationResult:
    """
    Fit pixel_error ~ a * latent_drift + b using only rows at low horizons
    (fit_k), then report the residual between that fit's prediction and the
    actually-observed pixel_error at eval_k (a high horizon).

    A residual near zero means pixel_error is fully explained by
    latent_drift at every horizon -- the decoder adds no information beyond
    what latent-space evaluation already tells you (Stage D's "stop"
    outcome). A large positive residual means pixel_error grows faster than
    latent_drift predicts -- decoder-side failure compounding with horizon,
    which is the diagnostic's reason to exist.
    """
    fit_rows = df[df["k"].isin(fit_k)]
    x = fit_rows[x_col].to_numpy()
    y = fit_rows[y_col].to_numpy()
    A = np.stack([x, np.ones_like(x)], axis=1)
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)

    eval_rows = df[df["k"] == eval_k]
    actual = eval_rows[y_col].mean()
    x_eval = eval_rows[x_col].mean()
    predicted = slope * x_eval + intercept
    residual = actual - predicted

    return SeparationResult(
        slope=float(slope),
        intercept=float(intercept),
        fit_horizons=fit_k,
        eval_horizon=eval_k,
        predicted_pixel_error_at_eval=float(predicted),
        actual_pixel_error_at_eval=float(actual),
        residual=float(residual),
    )
