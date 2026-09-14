"""
Lead-time detection: ported from github.com/HussainAther/pcc,
scripts/lead_time.py (warn_time_from_trace). There it asks how many
timesteps before a replicator system's population extinction the smoothed
entropy-rate dS/dt first drops below a threshold. Here "timestep" is
replaced by whatever index axis the caller supplies (horizon k, or
reinjection step j), and "extinction" is replaced by a caller-supplied
crash detector on Horizon Ladder's own signal (excess_over_floor).

Kept close to the original argument names/shape on purpose, so a reviewer
who knows the PCC repo can see directly that this is the same method
applied to a different system, not a different method with a similar
name.
"""

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
    """
    Same logic as PCC's warn_time_from_trace: compute d(entropy)/d(xs),
    smooth it, and find the first index (reading in increasing-xs order)
    after which the rate stays below `threshold` for `k_consecutive`
    points in a row. Returns (warn_x, threshold_used); warn_x is None if
    the rate never crosses.

    threshold=None auto-picks the `threshold_quantile` quantile of the
    observed rate values (PCC default: 0.10), clipped at 0 -- i.e. "rate
    is unusually negative relative to this trace's own history."
    """
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
    crash_x: Optional[float]         # where the reference signal (e.g. pixel_error) crossed its own crash threshold
    entropy_warn_x: Optional[float]  # where entropy-rate crossed its threshold
    lead: Optional[float]            # crash_x - entropy_warn_x; positive = entropy warned first
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
    """
    reference_signal: e.g. excess_over_floor(x) for x in xs -- the thing
    Horizon Ladder already reports. crash_x = first x where
    reference_signal >= reference_crash_threshold (mirrors PCC's
    first_time_below, just polarity-flipped since pixel error rising is
    the bad direction here, not a population dropping).
    """
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
