"""Experiment 1 equivalent for the action-free LeWM-style family: same six
candidate-metrics-vs-pixel_error battery as experiment1_ebid_vs_baselines.py
/ experiment1_second_family.py, on the from-scratch-trained third family.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from latent2rgb import select_device
from latent2rgb.data import RolloutBuilder
from latent2rgb.decoder import MinimalDecoder
from latent2rgb.ebid.baselines import cosine_distance, normalized_l2
from latent2rgb.ebid.ensemble import build_ensemble_predictions, ensemble_spread_stats
from latent2rgb.interfaces import HORIZONS
from latent2rgb.lewm_adapter import EMBED_DIM, TUBELET_SIZE, LeWMEncoder, Projector, build_vit_tiny_from_scratch
from latent2rgb.lewm_predictor import CausalPredictor
from latent2rgb.metrics import latent_drift as latent_drift_fn, pixel_error as pixel_error_fn
from latent2rgb.video_source import VideoDirClipSource

M_MEMBERS = 6
EPSILON = 0.05
PERTURB_SEED = 7

CANDIDATE_COLUMNS = [
    ("latent_l2_drift", "1. Raw Latent L2 drift"),
    ("cosine_distance", "2. Cosine distance"),
    ("normalized_l2", "3. Normalized L2"),
    ("ensemble_variance", "4. Ensemble variance"),
    ("effective_rank", "6. Effective rank (perturbation covariance)"),
    ("mahalanobis", "7. Mahalanobis distance"),
    ("ebid_entropy", "8. EBID entropy-based ensemble spread"),
]


def pearson(x, y):
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load_family(device):
    vit = build_vit_tiny_from_scratch().to(device)
    vit.load_state_dict(torch.load("lewm_encoder_vit.pt", map_location=device))
    vit.eval()
    enc_projector = Projector(EMBED_DIM).to(device)
    enc_projector.load_state_dict(torch.load("lewm_encoder.pt", map_location=device))
    enc_projector.eval()
    encoder = LeWMEncoder(vit, enc_projector, device=device)

    predictor = CausalPredictor(embed_dim=EMBED_DIM).to(device)
    predictor.projector = Projector(EMBED_DIM).to(device)
    predictor.load_state_dict(torch.load("lewm_predictor.pt", map_location=device))
    predictor.eval()

    decoder = MinimalDecoder(embed_dim=EMBED_DIM, grid_size=1, patch_size=256, tubelet_size=TUBELET_SIZE).to(device)
    decoder.load_state_dict(torch.load("decoder_lewm.pt", map_location=device))
    decoder.eval()
    return encoder, predictor, decoder


def main():
    device = select_device()
    print(f"device: {device}")

    with open("floor_lewm.json") as f:
        floor_data = json.load(f)
    val_ids_by_source = floor_data["val_ids"]

    encoder, predictor, decoder = load_family(device)

    sources = {
        "ssv2": VideoDirClipSource("data/ssv2/videos", extensions=("webm",)),
        "kinetics_mini": VideoDirClipSource("data/kinetics_mini", extensions=("mp4",)),
    }
    clip_id_to_source_name = {}
    for cid in val_ids_by_source:
        for name, src in sources.items():
            if cid in src.clip_ids():
                clip_id_to_source_name[cid] = name
                break

    gen = torch.Generator(device="cpu").manual_seed(PERTURB_SEED)
    rows = []
    for clip_id in val_ids_by_source:
        name = clip_id_to_source_name.get(clip_id)
        if name is None:
            continue
        clip_source = sources[name]
        length = clip_source.clip_length(clip_id)
        if length - HORIZONS[-1] - 2 < 8:
            continue
        builder = RolloutBuilder(clip_source, encoder, predictor, horizons=HORIZONS, device=device)
        t = 8
        sample = builder.build(clip_id, t)

        for k in HORIZONS:
            if k not in sample.predicted_tokens:
                continue
            pred_pure = sample.predicted_tokens[k]
            true_tok = sample.true_tubelet_tokens[k]
            true_frame = sample.true_frames[k]

            ensemble = build_ensemble_predictions(predictor, sample.context_tokens, k, M_MEMBERS, EPSILON, generator=gen)
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
        print(f"  {name}/{clip_id}: done")

    df = pd.DataFrame(rows)
    df.to_csv("outputs/experiment1_lewm_family_results.csv", index=False)
    print(f"\nwrote outputs/experiment1_lewm_family_results.csv ({len(df)} rows)")

    print("\nPearson r of each candidate signal vs. pixel_error (LeWM-style family):\n")
    results = []
    for col, label in CANDIDATE_COLUMNS:
        r = pearson(df[col].to_numpy(), df["pixel_error"].to_numpy())
        results.append({"metric": label, "column": col, "pearson_r_vs_pixel_error": r, "abs_r": abs(r) if not np.isnan(r) else -1})
    ranked = pd.DataFrame(results).sort_values("abs_r", ascending=False).drop(columns="abs_r")
    print(ranked.to_string(index=False))

    with open("outputs/experiment1_lewm_family_summary.json", "w") as f:
        json.dump({
            "m_members": M_MEMBERS, "epsilon": EPSILON,
            "n_clips": df["clip_id"].nunique(),
            "pearson_r_ranked": ranked.to_dict(orient="records"),
        }, f, indent=2)
    print("\nwrote outputs/experiment1_lewm_family_summary.json")


if __name__ == "__main__":
    main()
