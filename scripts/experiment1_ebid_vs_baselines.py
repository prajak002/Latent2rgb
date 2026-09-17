"""Experiment 1/2: does ensemble spread grow with rollout horizon, and does
EBID predict pixel-space failure better than the other baselines?"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from latent2rgb.data import RolloutBuilder
from latent2rgb.decoder import MinimalDecoder
from latent2rgb.ebid.baselines import cosine_distance, normalized_l2
from latent2rgb.ebid.ensemble import build_ensemble_predictions, ensemble_spread_stats
from latent2rgb.interfaces import HORIZONS
from latent2rgb.metrics import latent_drift as latent_drift_fn, pixel_error as pixel_error_fn
from latent2rgb.video_source import VideoDirClipSource
from latent2rgb.vjepa_adapter import TUBELET_SIZE, VJEPA2Encoder, VJEPA2Predictor, load_encoder_predictor

M_MEMBERS = 6
EPSILON = 0.05
PERTURB_SEED = 7

CANDIDATE_COLUMNS = [
    ("latent_l2_drift", "1. Raw Latent L2 drift"),
    ("cosine_distance", "2. Cosine distance"),
    ("normalized_l2", "3. Normalized L3 (interpreted, see baselines.py)"),
    ("ensemble_variance", "4. Ensemble variance"),
    ("effective_rank", "6. Effective rank (perturbation covariance)"),
    ("mahalanobis", "7. Mahalanobis distance"),
    ("ebid_entropy", "8. EBID entropy-based ensemble spread (candidate)"),
]


def run_dataset(name, clip_source, encoder, predictor, decoder, device, ids, min_t=8):
    builder = RolloutBuilder(clip_source, encoder, predictor, horizons=HORIZONS, device=device)
    gen = torch.Generator(device="cpu").manual_seed(PERTURB_SEED)
    rows = []
    for clip_id in ids:
        length = clip_source.clip_length(clip_id)
        max_t = length - HORIZONS[-1] - 2
        if max_t < min_t:
            continue
        t = min_t
        sample = builder.build(clip_id, t)

        for k in HORIZONS:
            if k not in sample.predicted_tokens:
                continue
            pred_pure = sample.predicted_tokens[k]      # [N, D] -- ensemble member 0
            true_tok = sample.true_tubelet_tokens[k]
            true_frame = sample.true_frames[k]

            ensemble = build_ensemble_predictions(
                predictor, sample.context_tokens, k, M_MEMBERS, EPSILON, generator=gen,
            )  # [M, N, D], member 0 == pred_pure by construction

            stats = ensemble_spread_stats(ensemble, reference=true_tok)

            with torch.no_grad():
                decoded = decoder.decode(pred_pure.unsqueeze(0)).squeeze(0)
            decoded_last_frame = decoded[:, -1]

            rows.append({
                "dataset": name, "clip_id": clip_id, "t": t, "k": k,
                "latent_l2_drift": latent_drift_fn(pred_pure, true_tok),
                "cosine_distance": cosine_distance(pred_pure, true_tok),
                "normalized_l2": normalized_l2(pred_pure, true_tok),
                "ensemble_variance": stats.ensemble_variance,
                "effective_rank": stats.effective_rank,
                "mahalanobis": stats.mahalanobis,
                "ebid_entropy": stats.entropy,
                "pixel_error": pixel_error_fn(decoded_last_frame, true_frame),
            })
        print(f"  {name}/{clip_id}: done ({len([r for r in rows if r['clip_id']==clip_id])} horizons)")
    return pd.DataFrame(rows)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/vitl.pt")
    ap.add_argument("--decoder-path", type=str, default="decoder.pt")
    ap.add_argument("--num-clips", type=int, default=12)
    ap.add_argument("--out-csv", type=str, default="outputs/experiment1_results.csv")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  M_MEMBERS={M_MEMBERS}  EPSILON={EPSILON}")

    raw_encoder, raw_predictor = load_encoder_predictor(args.checkpoint, device=device)
    encoder = VJEPA2Encoder(raw_encoder, device=device)
    predictor = VJEPA2Predictor(raw_predictor, device=device)
    decoder = MinimalDecoder(embed_dim=encoder.embed_dim, grid_size=16, patch_size=16, tubelet_size=TUBELET_SIZE).to(device)
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
    decoder.eval()

    dfs = []
    for name, video_dir, ext, offset in [
        ("ssv2", "data/ssv2/videos", ("webm",), 600),
        ("kinetics_mini", "data/kinetics_mini", ("mp4",), 0),
    ]:
        clip_source = VideoDirClipSource(video_dir, extensions=ext)
        ids = list(clip_source.clip_ids())[offset: offset + args.num_clips]
        print(f"\n=== {name}, {len(ids)} clips ===")
        dfs.append(run_dataset(name, clip_source, encoder, predictor, decoder, device, ids))

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv} ({len(df)} rows)")

    print("\n--- Q1: does ensemble spread grow with horizon? (per-k means, pooled both datasets) ---")
    print(df.groupby("k")[["ebid_entropy", "effective_rank", "ensemble_variance", "pixel_error"]].mean())

    print("\n--- Q2: does EBID predict pixel-space failure better than the baselines? ---")
    print("Pearson r of each candidate signal vs. pixel_error, pooled across both datasets, all (clip,k):\n")
    results = []
    for col, label in CANDIDATE_COLUMNS:
        r = pearson(df[col].to_numpy(), df["pixel_error"].to_numpy())
        results.append({"metric": label, "column": col, "pearson_r_vs_pixel_error": r, "abs_r": abs(r) if not np.isnan(r) else -1})
    ranked = pd.DataFrame(results).sort_values("abs_r", ascending=False).drop(columns="abs_r")
    print(ranked.to_string(index=False))

    with open("outputs/experiment1_summary.json", "w") as f:
        json.dump({
            "m_members": M_MEMBERS, "epsilon": EPSILON,
            "per_k_means": df.groupby("k")[["ebid_entropy", "effective_rank", "ensemble_variance", "pixel_error"]].mean().to_dict(),
            "pearson_r_ranked": ranked.to_dict(orient="records"),
        }, f, indent=2)
    print("\nwrote outputs/experiment1_summary.json")
    print("\nRAW NUMBERS ONLY ABOVE -- no interpretation beyond the two questions asked.")


if __name__ == "__main__":
    main()
