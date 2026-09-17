"""CV+ conformal calibration (Barber, Candes, Ramdas, Tibshirani 2021,
"Predictive inference with the jackknife+", Algorithm 2 / Section 5's
K-fold CV+ variant) on outputs/experiment1_results.csv, leaving out one
CLIP at a time (not one row) -- rows within a clip share the same latent
rollout and aren't independent, so the fold has to be the clip, matching
the clustering already used in stats_significance.py.

This directly answers the critique that split-conformal COVERAGE is
guaranteed by construction regardless of whether a candidate proxy carries
any information -- coverage alone can't distinguish an informative proxy
from an uninformative one. The informative quantity is interval WIDTH
(efficiency): a proxy-conditioned interval that's no narrower than a
proxy-free baseline (which ignores the candidate metric and predicts the
marginal mean of pixel_error) is not adding information, regardless of its
coverage. CV+ gives a valid (if conservative: coverage >= 1-2*alpha, not
exactly 1-alpha) marginal coverage guarantee for ANY score, which is the
correct, defensible framing -- unlike claiming "90% coverage" as if it were
evidence of informativeness on its own.

NOT called "functional conformal prediction" (Foresight, arXiv 2606.23085):
that method calibrates over curve-valued trajectories. This dataset is
scalar (one proxy value, one pixel_error value) per (clip, k) row, so plain
CV+ is the correctly-scoped tool -- citing FCP here would overstate the
methodological connection.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

ALPHA = 0.1  # target 1-alpha=90% coverage; CV+ guarantees >= 1-2*alpha=80%

CANDIDATE_COLUMNS = [
    "latent_l2_drift", "cosine_distance", "normalized_l2",
    "ensemble_variance", "effective_rank", "mahalanobis", "ebid_entropy",
]


def fit_line(x: np.ndarray, y: np.ndarray):
    if len(np.unique(x)) < 2:
        return 0.0, float(y.mean())
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def cv_plus(x: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha: float = ALPHA):
    """Leave-one-GROUP-out CV+. Returns per-row (width, covered) plus mean width/coverage."""
    n = len(x)
    unique_groups = np.unique(groups)
    loo_model = {}
    for g in unique_groups:
        mask = groups != g
        loo_model[g] = fit_line(x[mask], y[mask])

    resid = np.empty(n)
    for i in range(n):
        slope, intercept = loo_model[groups[i]]
        resid[i] = abs(y[i] - (slope * x[i] + intercept))

    # for each target point j, pool {f_{-g(i)}(x_j) +/- resid[i]} over ALL i
    k = int(np.floor(alpha * (n + 1)))  # order-statistic index (0-based from the extreme end)
    widths = np.empty(n)
    covered = np.empty(n, dtype=bool)
    all_slopes = np.array([loo_model[g][0] for g in groups])   # per-row model params, reused as f_{-g(i)}
    all_intercepts = np.array([loo_model[g][1] for g in groups])
    for j in range(n):
        fitted_at_xj = all_slopes * x[j] + all_intercepts  # f_{-g(i)}(x_j) for every i
        lo_vals = np.sort(fitted_at_xj - resid)
        hi_vals = np.sort(fitted_at_xj + resid)
        lower = lo_vals[k] if k < n else lo_vals[0]
        upper = hi_vals[n - 1 - k] if k < n else hi_vals[-1]
        widths[j] = upper - lower
        covered[j] = lower <= y[j] <= upper

    return {
        "mean_width": float(widths.mean()),
        "coverage": float(covered.mean()),
        "n": n,
        "n_groups": len(unique_groups),
        "alpha": alpha,
        "guaranteed_coverage_floor": 1 - 2 * alpha,
    }


def main():
    df = pd.read_csv("outputs/experiment1_results.csv")
    groups = (df["dataset"] + "/" + df["clip_id"]).to_numpy()
    y = df["pixel_error"].to_numpy()

    results = {}

    baseline_x = np.zeros(len(df))  # proxy-free: degenerates to per-fold constant (group mean)
    baseline = cv_plus(baseline_x, y, groups)
    results["proxy_free_baseline"] = baseline
    print(f"proxy-free baseline: mean_width={baseline['mean_width']:.5f}  coverage={baseline['coverage']:.3f}  "
          f"(n={baseline['n']}, {baseline['n_groups']} clip-folds, guaranteed >= {baseline['guaranteed_coverage_floor']:.0%})")

    for col in CANDIDATE_COLUMNS:
        x = df[col].to_numpy()
        r = cv_plus(x, y, groups)
        r["width_vs_baseline_pct"] = 100.0 * (r["mean_width"] - baseline["mean_width"]) / baseline["mean_width"]
        results[col] = r
        tag = "NARROWER (informative)" if r["width_vs_baseline_pct"] < -1.0 else "no better than proxy-free"
        print(f"{col:20s} mean_width={r['mean_width']:.5f}  coverage={r['coverage']:.3f}  "
              f"width_vs_baseline={r['width_vs_baseline_pct']:+.1f}%  -> {tag}")

    with open("outputs/conformal_calibration.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote outputs/conformal_calibration.json")


if __name__ == "__main__":
    main()
