from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _smooth(y: np.ndarray, w: int = 3) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if w <= 1 or len(y) < w:
        return y
    w = int(w) if int(w) % 2 == 1 else int(w) + 1
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


def warn_index_from_trace(
    xs: Sequence[float],
    entropy: Sequence[float],
    *,
    k_consecutive: int = 2,
    threshold: Optional[float] = None,
    threshold_quantile: float = 0.10,
    smooth_w: int = 3,
) -> Tuple[Optional[float], float]:
    # First x (increasing order) after which smoothed d(entropy)/dx stays
    # below `threshold` for k_consecutive points. threshold=None auto-picks
    # the threshold_quantile quantile of the trace's own rate, clipped at 0.
    xs_arr = np.asarray(xs, dtype=float)
    h_arr = np.asarray(entropy, dtype=float)
    if xs_arr.size < 3:
        return None, np.nan

    sdot = np.gradient(h_arr, xs_arr)
    sdot = _smooth(sdot, smooth_w)

    if threshold is None:
        finite = sdot[np.isfinite(sdot)]
        if finite.size == 0:
            return None, np.nan
        used_thr = min(0.0, float(np.quantile(finite, threshold_quantile)))
    else:
        used_thr = float(threshold)

    consec = 0
    for i in range(sdot.size):
        v = sdot[i]
        if np.isfinite(v) and v < used_thr:
            consec += 1
            if consec >= k_consecutive:
                return float(xs_arr[i - (k_consecutive - 1)]), used_thr
        else:
            consec = 0
    return None, used_thr


@dataclass
class LeadTimeResult:
    crash_x: Optional[float]
    entropy_warn_x: Optional[float]
    lead: Optional[float]
    entropy_threshold_used: float


def compare_lead_time(
    xs: Sequence[float],
    entropy: Sequence[float],
    reference_signal: Sequence[float],
    *,
    reference_crash_threshold: float,
    k_consecutive: int = 2,
    entropy_threshold: Optional[float] = None,
    entropy_threshold_quantile: float = 0.10,
    smooth_w: int = 3,
) -> LeadTimeResult:
    xs_arr = np.asarray(xs, dtype=float)
    ref = np.asarray(reference_signal, dtype=float)
    crash_idx = np.where(ref >= reference_crash_threshold)[0]
    crash_x = float(xs_arr[crash_idx[0]]) if crash_idx.size else None

    warn_x, used_thr = warn_index_from_trace(
        xs, entropy,
        k_consecutive=k_consecutive,
        threshold=entropy_threshold,
        threshold_quantile=entropy_threshold_quantile,
        smooth_w=smooth_w,
    )

    lead = (crash_x - warn_x) if (crash_x is not None and warn_x is not None) else None
    return LeadTimeResult(crash_x=crash_x, entropy_warn_x=warn_x, lead=lead, entropy_threshold_used=used_thr)
