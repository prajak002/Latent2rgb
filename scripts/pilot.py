"""
Stage D, real components: the pilot. 20 clips. For each clip and each
horizon k, causally query the frozen V-JEPA2 predictor for the tubelet at
t+k, decode it, and compute latent_drift[k], pixel_error[k],
excess_over_floor[k] against real SSv2 ground truth.

REMINDER (see vjepa_adapter.py and interfaces.py): this queries the base
V-JEPA2 predictor in a causal (context=past, target=future) pattern it was
never trained on. Report every number here as characterizing the frozen
predictor under that out-of-distribution regime, not as a clean measurement
of "how good is this model's forward dynamics."
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from latent2rgb import select_device
from latent2rgb.data import RolloutBuilder
from latent2rgb.decoder import MinimalDecoder
from latent2rgb.interfaces import HORIZONS
from latent2rgb.metrics import (
    fit_separation_statistic,
    latent_drift as latent_drift_fn,
    pixel_error as pixel_error_fn,
)
from latent2rgb.ssv2 import SSv2ClipSource
from latent2rgb.vjepa_adapter import TUBELET_SIZE, VJEPA2Encoder, VJEPA2Predictor, load_encoder_predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/vitl.pt")
    ap.add_argument("--video-dir", type=str, default="data/ssv2/videos")
    ap.add_argument("--num-clips", type=int, default=20)
    ap.add_argument("--clip-offset", type=int, default=150)  # avoid overlap with Stage C's training clips
    ap.add_argument("--floor-json", type=str, default="floor.json")
    ap.add_argument("--decoder-path", type=str, default="decoder.pt")
    ap.add_argument("--out-csv", type=str, default="stage_d_results.csv")
    ap.add_argument("--fit-k", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--eval-k", type=int, default=32)
    ap.add_argument("--min-t", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = select_device()
    print(f"device: {device}")

    with open(args.floor_json) as f:
        floor_data = json.load(f)
    aggregate_floor = floor_data["aggregate_floor"]

    torch.manual_seed(args.seed)
    clip_source = SSv2ClipSource(args.video_dir)
    all_ids = list(clip_source.clip_ids())
    pilot_ids = all_ids[args.clip_offset : args.clip_offset + args.num_clips]
    print(f"pilot clips ({len(pilot_ids)}): {pilot_ids}")

    raw_encoder, raw_predictor = load_encoder_predictor(args.checkpoint, device=device)
    encoder = VJEPA2Encoder(raw_encoder, device=device)
    predictor = VJEPA2Predictor(raw_predictor, device=device)

    decoder = MinimalDecoder(embed_dim=encoder.embed_dim, grid_size=16, patch_size=16, tubelet_size=TUBELET_SIZE).to(device)
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
    decoder.eval()

    builder = RolloutBuilder(clip_source, encoder, predictor, horizons=HORIZONS, device=device)

    rows = []
    for clip_id in pilot_ids:
        length = clip_source.clip_length(clip_id)
        max_t = length - HORIZONS[-1] - 2
        if max_t < args.min_t:
            print(f"skipping {clip_id}: too short (length={length})")
            continue
        t = args.min_t
        sample = builder.build(clip_id, t)
        for k in HORIZONS:
            if k not in sample.predicted_tokens:
                continue
            true_tok = sample.true_tubelet_tokens[k]
            pred_tok = sample.predicted_tokens[k]
            true_frame = sample.true_frames[k]

            with torch.no_grad():
                decoded = decoder.decode(pred_tok.unsqueeze(0)).squeeze(0)  # [C,T,H,W]
            decoded_last_frame = decoded[:, -1]  # later frame of the decoded tubelet

            ld = latent_drift_fn(pred_tok, true_tok)
            pe = pixel_error_fn(decoded_last_frame, true_frame)
            rows.append(
                {
                    "clip_id": clip_id,
                    "t": t,
                    "k": k,
                    "latent_drift": ld,
                    "pixel_error": pe,
                    "excess_over_floor": pe - aggregate_floor,
                }
            )
        print(f"{clip_id}: done ({len([r for r in rows if r['clip_id']==clip_id])} horizons)")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print(f"\nfloor (from Stage C, real) = {aggregate_floor:.5f}")
    print(f"\nRAW per-horizon means (real V-JEPA2 + real SSv2, OOD causal query):")
    print(df.groupby("k")[["latent_drift", "pixel_error", "excess_over_floor"]].mean())
    print(f"\nRAW full dataframe:")
    print(df.to_string())

    if len(df["k"].unique()) >= 2 and all(k in df["k"].values for k in args.fit_k):
        sep = fit_separation_statistic(df, fit_k=args.fit_k, eval_k=args.eval_k)
        print(f"\n--- separation statistic (fit_k={args.fit_k}, eval_k={args.eval_k}) ---")
        print(f"pixel_error = {sep.slope:.4f} * latent_drift + {sep.intercept:.4f}")
        print(f"predicted={sep.predicted_pixel_error_at_eval:.5f}  actual={sep.actual_pixel_error_at_eval:.5f}  "
              f"residual={sep.residual:.5f}")
    else:
        print("\nnot enough horizon coverage to fit separation statistic -- report raw numbers only")

    print(f"\nwrote {args.out_csv}")
    print("\nRAW NUMBERS ONLY ABOVE -- no interpretation. See dataframe/means for the actual decision.")


if __name__ == "__main__":
    main()
