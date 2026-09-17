"""Computes latent_drift and pixel_error at each k for the exact clips in
outputs/videos/manifest.json (dump_rollout_videos.py's output), so the
results gallery can annotate each video pair with the numbers that make
the divergence visible instead of asking the viewer to eyeball it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from latent2rgb import select_device
from latent2rgb.data import RolloutBuilder
from latent2rgb.decoder import MinimalDecoder
from latent2rgb.metrics import latent_drift as latent_drift_fn, pixel_error as pixel_error_fn
from latent2rgb.video_source import VideoDirClipSource
from latent2rgb.vjepa_adapter import TUBELET_SIZE, VJEPA2Encoder, VJEPA2Predictor, load_encoder_predictor

T = 8
K_VALUES = list(range(2, 33, 2))
DATASET_DIRS = {
    "ssv2": ("data/ssv2/videos", ("webm",)),
    "kinetics_mini": ("data/kinetics_mini", ("mp4",)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/vitl.pt")
    ap.add_argument("--decoder-path", type=str, default="decoder.pt")
    ap.add_argument("--manifest", type=str, default="outputs/videos/manifest.json")
    ap.add_argument("--out-json", type=str, default="outputs/videos/rollout_metrics.json")
    args = ap.parse_args()

    device = select_device()
    print(f"device: {device}")

    with open(args.manifest) as f:
        manifest = json.load(f)

    raw_encoder, raw_predictor = load_encoder_predictor(args.checkpoint, device=device)
    encoder = VJEPA2Encoder(raw_encoder, device=device)
    predictor = VJEPA2Predictor(raw_predictor, device=device)
    decoder = MinimalDecoder(embed_dim=encoder.embed_dim, grid_size=16, patch_size=16, tubelet_size=TUBELET_SIZE).to(device)
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
    decoder.eval()

    sources = {}
    results = []
    for entry in manifest:
        name = entry["dataset"]
        if name not in sources:
            video_dir, ext = DATASET_DIRS[name]
            sources[name] = VideoDirClipSource(video_dir, extensions=ext)
        clip_source = sources[name]
        builder = RolloutBuilder(clip_source, encoder, predictor, horizons=K_VALUES, device=device)

        sample = builder.build(entry["clip_id"], T)
        per_k = []
        for k in K_VALUES:
            if k not in sample.predicted_tokens:
                continue
            true_tok = sample.true_tubelet_tokens[k]
            pred_tok = sample.predicted_tokens[k]
            true_frame = sample.true_frames[k]
            with torch.no_grad():
                decoded = decoder.decode(pred_tok.unsqueeze(0)).squeeze(0)
            decoded_frame = decoded[:, -1]
            per_k.append({
                "k": k,
                "latent_drift": latent_drift_fn(pred_tok, true_tok),
                "pixel_error": pixel_error_fn(decoded_frame, true_frame),
            })
        results.append({"tag": entry["tag"], "dataset": name, "clip_id": entry["clip_id"], "per_k": per_k})
        print(f"  {entry['tag']}: {len(per_k)} horizons")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
