"""Experiment 5: interactions and regime boundaries in Experiment 4's sweep.

Two questions against outputs/experiment4_results.csv (dataset, clip_id,
mode, epsilon, j, entropy, pixel_error):

1. Does context-correction (pure vs corrected) flip sign as a function of j
   or epsilon -- i.e. is there a horizon/perturbation regime where
   correction helps vs one where it hurts?
2. Does the entropy~epsilon slope (EBID's scaling claim) change with j --
   i.e. is there a horizon past which the linear fit gets stronger/weaker?
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from latent2rgb.ebid.perturb import fit_entropy_deficit_scaling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", type=str, default="outputs/experiment4_results.csv")
    ap.add_argument("--out-json", type=str, default="outputs/experiment5_summary.json")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # (1) mode gap: corrected - pure pixel_error, mean over clips, by (epsilon, j)
    pure = df[df["mode"] == "pure"].groupby(["epsilon", "j"])["pixel_error"].mean()
    corrected = df[df["mode"] == "corrected"].groupby(["epsilon", "j"])["pixel_error"].mean()
    gap = (corrected - pure).rename("mode_gap").reset_index()
    print("--- mode gap (corrected - pure) mean pixel_error, by (epsilon, j) ---")
    print(gap.to_string(index=False))

    sign_flips = []
    for eps, g in gap.groupby("epsilon"):
        g = g.sort_values("j")
        signs = np.sign(g["mode_gap"].to_numpy())
        for i in range(1, len(signs)):
            if signs[i] != signs[i - 1] and signs[i] != 0 and signs[i - 1] != 0:
                sign_flips.append({"epsilon": float(eps), "j_before": int(g["j"].iloc[i - 1]), "j_after": int(g["j"].iloc[i])})

    # (2) entropy~epsilon slope/R^2 by (mode, j), pooled across clips (mean entropy per epsilon)
    fits = []
    for (mode, j), g in df.groupby(["mode", "j"]):
        means = g.groupby("epsilon")["entropy"].mean()
        eps_sorted = sorted(means.index)
        deficits = [means[e] - means[eps_sorted[0]] for e in eps_sorted]
        fit = fit_entropy_deficit_scaling(eps_sorted, deficits)
        fits.append({"mode": mode, "j": int(j), "slope": fit.slope, "r_squared": fit.r_squared})

    fits_df = pd.DataFrame(fits).sort_values(["mode", "j"])
    print("\n--- entropy-deficit-vs-epsilon linear fit, by (mode, j) ---")
    print(fits_df.to_string(index=False))

    with open(args.out_json, "w") as f:
        json.dump({
            "mode_gap_by_epsilon_j": gap.to_dict(orient="records"),
            "mode_gap_sign_flips": sign_flips,
            "entropy_scaling_fit_by_mode_j": fits_df.to_dict(orient="records"),
        }, f, indent=2)

    print(f"\n{len(sign_flips)} sign flip(s) in the context-correction mode gap across adjacent j, at fixed epsilon:")
    for sf in sign_flips:
        print(f"  epsilon={sf['epsilon']}: sign flips between j={sf['j_before']} and j={sf['j_after']}")

    print(f"\nwrote {args.out_json}")
    print("\nRAW NUMBERS ONLY ABOVE.")


if __name__ == "__main__":
    main()
