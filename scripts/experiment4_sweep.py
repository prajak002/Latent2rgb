"""Experiment 4: horizon (j) x perturbation (epsilon) x context-correction
(pure vs. decode/re-encode) sweep, built on Stage F's reinjection mechanism.

For each clip: roll out to a fixed reinjection point (k_reinject), take that
prediction either raw ("pure") or through a decode->re-encode round trip
("corrected"), perturb it with epsilon noise, extend context with it, and
roll forward j more steps. Records entropy and real pixel error at every
(mode, epsilon, j) combination, so Experiment 5 can look for interactions
and Experiment 6 can test transitivity across conditions.
"""

from __future__ import annotations

import argparse
import itertools
import json

import pandas as pd
import torch

from latent2rgb import select_device
from latent2rgb.decoder import MinimalDecoder
from latent2rgb.ebid.entropy import spectral_entropy
from latent2rgb.ebid.perturb import perturb_tokens
from latent2rgb.metrics import pixel_error as pixel_error_fn
from latent2rgb.video_source import VideoDirClipSource
from latent2rgb.vjepa_adapter import TUBELET_SIZE, VJEPA2Encoder, VJEPA2Predictor, load_encoder_predictor

T = 8
K_REINJECT = 8
J_VALUES = [2, 4, 6, 8, 12, 16, 24, 32]
EPSILONS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
MODES = ["pure", "corrected"]
PERTURB_SEED = 1234


def true_frame_for(frames: torch.Tensor, base_t: int, k_total: int) -> torch.Tensor:
    target_slot = (base_t + k_total) // TUBELET_SIZE
    target_end = target_slot * TUBELET_SIZE + TUBELET_SIZE
    return frames[target_end - 1]


def run_dataset(name, clip_source, encoder, predictor, decoder, device, ids):
    rows = []
    max_j = J_VALUES[-1]
    for clip_id in ids:
        length = clip_source.clip_length(clip_id)
        needed = T + K_REINJECT + max_j + 2
        if length < needed:
            continue
        batch = clip_source.get_clip(clip_id, T, K_REINJECT + max_j + 1)
        frames = batch.frames.to(device)

        context = encoder.encode(frames[:T].permute(1, 0, 2, 3).unsqueeze(0)).squeeze(0)
        with torch.no_grad():
            p1_pure = predictor.rollout(context.unsqueeze(0), None, K_REINJECT).squeeze(0)
            decoded_p1 = decoder.decode(p1_pure.unsqueeze(0))
            p1_corrected = encoder.encode(decoded_p1).squeeze(0)

        bases = {"pure": p1_pure, "corrected": p1_corrected}
        gen = torch.Generator(device=device if device != "mps" else "cpu").manual_seed(PERTURB_SEED)

        for mode, eps in itertools.product(MODES, EPSILONS):
            base = bases[mode]
            perturbed = perturb_tokens(base, eps, generator=gen if device != "mps" else None)
            ext_context = torch.cat([context, perturbed], dim=0).unsqueeze(0)
            for j in J_VALUES:
                with torch.no_grad():
                    next_tok = predictor.rollout(ext_context, None, j).squeeze(0)
                    decoded = decoder.decode(next_tok.unsqueeze(0)).squeeze(0)
                true_frame = true_frame_for(frames, T + K_REINJECT, j)
                rows.append({
                    "dataset": name, "clip_id": clip_id, "mode": mode, "epsilon": eps, "j": j,
                    "entropy": spectral_entropy(next_tok),
                    "pixel_error": pixel_error_fn(decoded[:, -1], true_frame),
                })
        print(f"  {name}/{clip_id}: {len(MODES) * len(EPSILONS) * len(J_VALUES)} rows done")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/vitl.pt")
    ap.add_argument("--decoder-path", type=str, default="decoder.pt")
    ap.add_argument("--num-clips", type=int, default=20)
    ap.add_argument("--out-csv", type=str, default="outputs/experiment4_results.csv")
    ap.add_argument("--out-json", type=str, default="outputs/experiment4_summary.json")
    args = ap.parse_args()

    device = select_device()
    print(f"device: {device}  modes={MODES}  epsilons={EPSILONS}  j_values={J_VALUES}")

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
        print(f"\n=== {name}, {len(ids)} candidate clips ===")
        dfs.append(run_dataset(name, clip_source, encoder, predictor, decoder, device, ids))

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv} ({len(df)} rows)")

    print("\n--- mean entropy / pixel_error by (mode, epsilon), pooled over j and clips ---")
    grouped = df.groupby(["mode", "epsilon"])[["entropy", "pixel_error"]].mean()
    print(grouped)

    with open(args.out_json, "w") as f:
        json.dump({
            "t": T, "k_reinject": K_REINJECT, "j_values": J_VALUES, "epsilons": EPSILONS, "modes": MODES,
            "mean_by_mode_epsilon": grouped.reset_index().to_dict(orient="records"),
        }, f, indent=2)
    print(f"wrote {args.out_json}")
    print("\nRAW NUMBERS ONLY ABOVE.")


if __name__ == "__main__":
    main()
