"""
Stage B smoke test, real components: real SSv2 clips, real V-JEPA2 ViT-L
encoder+predictor. Prints real shapes at every step. No synthetic backend
anywhere in this file.
"""

from __future__ import annotations

import argparse

import torch

from latent2rgb.data import RolloutBuilder
from latent2rgb.interfaces import HORIZONS
from latent2rgb.ssv2 import SSv2ClipSource
from latent2rgb.vjepa_adapter import VJEPA2Encoder, VJEPA2Predictor, load_encoder_predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/vitl.pt")
    ap.add_argument("--video-dir", type=str, default="data/ssv2/videos")
    ap.add_argument("--t", type=int, default=8)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    print("loading real V-JEPA2 ViT-L encoder+predictor from checkpoint...")
    raw_encoder, raw_predictor = load_encoder_predictor(args.checkpoint, device=device)
    encoder = VJEPA2Encoder(raw_encoder, device=device)
    predictor = VJEPA2Predictor(raw_predictor, device=device)
    print(f"encoder.embed_dim: {encoder.embed_dim}")
    print(f"encoder.tokens_per_tubelet: {encoder.tokens_per_tubelet}")

    print("\nloading SSv2 clip source...")
    clip_source = SSv2ClipSource(args.video_dir)
    ids = clip_source.clip_ids()
    print(f"num clips available: {len(ids)}")
    print(f"clip_length({ids[0]}): {clip_source.clip_length(ids[0])}")

    print(f"\nhorizons: {HORIZONS}")
    builder = RolloutBuilder(clip_source, encoder, predictor, horizons=HORIZONS, device=device)

    sample = builder.build(ids[0], t=args.t)
    print(f"\nsample.clip_id={sample.clip_id} t={sample.t}")
    print(f"context_tokens.shape: {tuple(sample.context_tokens.shape)}")
    for k in HORIZONS:
        if k not in sample.predicted_tokens:
            print(f"k={k:>2}  SKIPPED (target tubelet overlaps context or exceeds fetched frames)")
            continue
        tf = sample.true_frames[k]
        tt = sample.true_tubelet_tokens[k]
        pt = sample.predicted_tokens[k]
        print(
            f"k={k:>2}  true_frame.shape={tuple(tf.shape)}  true_frame.range=[{tf.min():.3f},{tf.max():.3f}]  "
            f"true_tubelet_tokens.shape={tuple(tt.shape)}  predicted_tokens.shape={tuple(pt.shape)}"
        )

    print("\niterating a few samples via iter_samples...")
    gen = torch.Generator().manual_seed(0)
    n = 0
    for s in builder.iter_samples(ids[:3], starts_per_clip=1, rng=gen, min_t=8):
        n += 1
        print(f"  sample {n}: clip={s.clip_id} t={s.t} n_horizons_ok={len(s.predicted_tokens)}")
    print(f"iter_samples produced {n} samples")

    print("\nSMOKE TEST PASSED (real V-JEPA2 + real SSv2 clips)")


if __name__ == "__main__":
    main()
