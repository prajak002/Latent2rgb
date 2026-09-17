"""Cluster-bootstrap confidence intervals on the headline numbers already
reported (Stage D, Experiments 1/3/4), resampling by clip_id (not by row --
rows within a clip aren't independent observations). Answers the reviewer
question "is r=0.12 at n=24 clips distinguishable from zero?" with an actual
interval instead of a bare point estimate.

Reads only files scripts/pilot.py and experiment{1,3,4}_*.py already wrote --
no new model inference.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

N_BOOT = 5000
SEED = 0


def cluster_bootstrap(df: pd.DataFrame, clip_col: str, statistic_fn, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    clip_ids = df[clip_col].unique()
    n = len(clip_ids)
    point = statistic_fn(df)
    boot_stats = []
    for _ in range(n_boot):
        sample_ids = rng.choice(clip_ids, size=n, replace=True)
        # concat, allowing repeats, of each sampled clip's full row set
        parts = [df[df[clip_col] == cid] for cid in sample_ids]
        boot_df = pd.concat(parts, ignore_index=True)
        try:
            boot_stats.append(statistic_fn(boot_df))
        except Exception:
            continue
    boot_stats = np.array(boot_stats)
    lo, hi = np.percentile(boot_stats, [2.5, 97.5])
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi), "n_clips": n, "n_boot_used": len(boot_stats)}


def wilson_ci(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    results = {}

    # --- Stage D: pixel_error's horizon-range vs. latent_drift's, ratio, with CI ---
    df_d = pd.read_csv("stage_d_results.csv")

    def range_ratio(df):
        g = df.groupby("k")[["latent_drift", "pixel_error"]].mean()
        ld_range = (g["latent_drift"].max() - g["latent_drift"].min()) / g["latent_drift"].mean() * 100
        pe_range = (g["pixel_error"].max() - g["pixel_error"].min()) / g["pixel_error"].mean() * 100
        return pe_range / ld_range if ld_range else float("nan")

    results["stage_d_range_ratio"] = cluster_bootstrap(df_d, "clip_id", range_ratio)

    # --- Experiment 1: candidate-metric correlations vs. pixel_error, with CI ---
    df1 = pd.read_csv("outputs/experiment1_results.csv")
    metric_cols = [
        "latent_l2_drift", "cosine_distance", "normalized_l2",
        "ensemble_variance", "effective_rank", "mahalanobis", "ebid_entropy",
    ]
    corr_results = {}
    for col in metric_cols:
        def r_stat(df, col=col):
            sub = df[[col, "pixel_error"]].dropna()
            return float(stats.pearsonr(sub[col], sub["pixel_error"])[0])
        corr_results[col] = cluster_bootstrap(df1, "clip_id", r_stat)
    results["experiment1_correlations"] = corr_results

    # --- Experiment 3: lead-time success rate, Wilson CI ---
    df3 = pd.read_csv("outputs/experiment3_results.csv")
    n_crashed = df3["crash_k"].notna().sum()
    n_led = df3["lead"].notna().sum()
    lo, hi = wilson_ci(int(n_led), int(n_crashed)) if n_crashed else (0.0, 0.0)
    results["experiment3_lead_time_rate"] = {
        "successes": int(n_led), "n_crashed": int(n_crashed),
        "rate": (n_led / n_crashed if n_crashed else None),
        "wilson_ci_95": [lo, hi],
    }

    # --- Experiment 4: entropy-vs-epsilon slope per mode, with CI (sign-flip check) ---
    df4 = pd.read_csv("outputs/experiment4_results.csv")
    slope_results = {}
    for mode in ["pure", "corrected"]:
        sub_mode = df4[df4["mode"] == mode]

        def slope_stat(df):
            g = df.groupby("epsilon")["entropy"].mean().reset_index()
            slope, _, _, _, _ = stats.linregress(g["epsilon"], g["entropy"])
            return float(slope)
        slope_results[mode] = cluster_bootstrap(sub_mode, "clip_id", slope_stat)
    results["experiment4_entropy_epsilon_slope"] = slope_results
    results["experiment4_sign_flip_ci_disjoint"] = (
        slope_results["pure"]["ci_hi"] < slope_results["corrected"]["ci_lo"]
        or slope_results["corrected"]["ci_hi"] < slope_results["pure"]["ci_lo"]
    )

    with open("outputs/stats_significance.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print("\nwrote outputs/stats_significance.json")


if __name__ == "__main__":
    main()
