"""Stage D equivalent for the second model family (DINOv2 encoder + trained
DeltaMLPPredictor, see latent2rgb/dinov2_adapter.py, delta_predictor.py,
scripts/train_second_family.py): does the same latent_drift-vs-pixel_error
divergence Stage D found on V-JEPA2 reproduce on a differently-architected
latent-only predictor? Mirrors scripts/pilot.py exactly so the two are
directly comparable -- only the (Encoder, StatePredictor, Decoder) triple
changes.
"""

from __future__ import annotations

import json

import pandas as pd
import torch

from latent2rgb import select_device
from latent2rgb.data import RolloutBuilder
from latent2rgb.decoder import MinimalDecoder
from latent2rgb.delta_predictor import DeltaMLPPredictor
from latent2rgb.dinov2_adapter import DINOv2Encoder, TUBELET_SIZE, load_dinov2
from latent2rgb.interfaces import HORIZONS
from latent2rgb.metrics import fit_separation_statistic, latent_drift as latent_drift_fn, pixel_error as pixel_error_fn
from latent2rgb.video_source import VideoDirClipSource


def main():
    device = select_device()
    print(f"device: {device}")

    with open("floor_dinov2.json") as f:
        floor_data = json.load(f)
    aggregate_floor = floor_data["aggregate_floor"]
    val_ids_by_source = floor_data["val_ids"]

    dinov2 = load_dinov2(device=device)
    encoder = DINOv2Encoder(dinov2, device=device)
    predictor = DeltaMLPPredictor(embed_dim=encoder.embed_dim).to(device)
    predictor.load_state_dict(torch.load("predictor_dinov2.pt", map_location=device))
    predictor.eval()
    decoder = MinimalDecoder(embed_dim=encoder.embed_dim, grid_size=16, patch_size=16, tubelet_size=TUBELET_SIZE).to(device)
    decoder.load_state_dict(torch.load("decoder_dinov2.pt", map_location=device))
    decoder.eval()

    sources = {
        "ssv2": VideoDirClipSource("data/ssv2/videos", extensions=("webm",)),
        "kinetics_mini": VideoDirClipSource("data/kinetics_mini", extensions=("mp4",)),
    }
    clip_id_to_source = {}
    for cid in val_ids_by_source:
        for name, src in sources.items():
            if cid in src.clip_ids():
                clip_id_to_source[cid] = src
                break

    rows = []
    for clip_id in val_ids_by_source:
        clip_source = clip_id_to_source.get(clip_id)
        if clip_source is None:
            continue
        builder = RolloutBuilder(clip_source, encoder, predictor, horizons=HORIZONS, device=device)
        length = clip_source.clip_length(clip_id)
        max_t = length - HORIZONS[-1] - 2
        if max_t < 8:
            print(f"skipping {clip_id}: too short")
            continue
        t = 8
        sample = builder.build(clip_id, t)
        for k in HORIZONS:
            if k not in sample.predicted_tokens:
                continue
            true_tok = sample.true_tubelet_tokens[k]
            pred_tok = sample.predicted_tokens[k]
            true_frame = sample.true_frames[k]
            with torch.no_grad():
                decoded = decoder.decode(pred_tok.unsqueeze(0)).squeeze(0)
            decoded_last_frame = decoded[:, -1]
            ld = latent_drift_fn(pred_tok, true_tok)
            pe = pixel_error_fn(decoded_last_frame, true_frame)
            rows.append({"clip_id": clip_id, "t": t, "k": k, "latent_drift": ld, "pixel_error": pe, "excess_over_floor": pe - aggregate_floor})
        print(f"{clip_id}: done")

    df = pd.DataFrame(rows)
    df.to_csv("stage_d_results_second_family.csv", index=False)

    print(f"\nfloor (DINOv2 second family) = {aggregate_floor:.5f}")
    print("\nper-horizon means (DINOv2 encoder + trained DeltaMLPPredictor):")
    print(df.groupby("k")[["latent_drift", "pixel_error", "excess_over_floor"]].mean())

    if len(df["k"].unique()) >= 2 and all(k in df["k"].values for k in [2, 4]):
        sep = fit_separation_statistic(df, fit_k=[2, 4], eval_k=32)
        print(f"\nseparation statistic (fit_k=[2,4], eval_k=32): residual={sep.residual:.5f}")

    print("\nwrote stage_d_results_second_family.csv")


if __name__ == "__main__":
    main()
