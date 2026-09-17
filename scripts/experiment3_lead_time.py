"""Experiment 3: does EBID entropy give early warning of rollout failure?

Reuses Experiment 1's per-clip trace across horizons (k=2,4,8,16,32):
for each clip, does the entropy-rate drop (ebid.lead_time.warn_index_from_trace)
happen at a smaller k than pixel_error first crosses a crash threshold?
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from latent2rgb.ebid.lead_time import compare_lead_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", type=str, default="outputs/experiment1_results.csv")
    ap.add_argument("--out-csv", type=str, default="outputs/experiment3_results.csv")
    ap.add_argument("--out-json", type=str, default="outputs/experiment3_summary.json")
    ap.add_argument("--crash-quantile", type=float, default=0.75)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    crash_threshold = float(df["pixel_error"].quantile(args.crash_quantile))
    print(f"crash threshold (pixel_error, q={args.crash_quantile}): {crash_threshold:.4f}")

    rows = []
    for (dataset, clip_id), g in df.groupby(["dataset", "clip_id"]):
        g = g.sort_values("k")
        result = compare_lead_time(
            xs=g["k"].to_numpy(),
            entropy=g["ebid_entropy"].to_numpy(),
            reference_signal=g["pixel_error"].to_numpy(),
            reference_crash_threshold=crash_threshold,
        )
        rows.append({
            "dataset": dataset, "clip_id": clip_id,
            "crash_k": result.crash_x,
            "entropy_warn_k": result.entropy_warn_x,
            "lead": result.lead,
            "entropy_threshold_used": result.entropy_threshold_used,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv} ({len(out)} clips)")

    n_crashed = out["crash_k"].notna().sum()
    n_warned = out["entropy_warn_k"].notna().sum()
    n_led = (out["lead"].notna() & (out["lead"] > 0)).sum()
    lead_vals = out.loc[out["lead"].notna(), "lead"]

    summary = {
        "crash_threshold": crash_threshold,
        "n_clips": len(out),
        "n_clips_crashed": int(n_crashed),
        "n_clips_entropy_warned": int(n_warned),
        "n_clips_entropy_led_before_crash": int(n_led),
        "mean_lead": float(lead_vals.mean()) if len(lead_vals) else None,
        "median_lead": float(lead_vals.median()) if len(lead_vals) else None,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{n_crashed}/{len(out)} clips crossed the crash threshold at some k")
    print(f"{n_warned}/{len(out)} clips had an entropy-rate warning at some k")
    print(f"{n_led}/{len(out)} clips: entropy warned at a smaller k than the crash (lead > 0)")
    if len(lead_vals):
        print(f"lead (crash_k - warn_k) mean={summary['mean_lead']:.2f}, median={summary['median_lead']:.2f}")
    print(f"\nwrote {args.out_json}")
    print("\nRAW NUMBERS ONLY ABOVE.")


if __name__ == "__main__":
    main()
