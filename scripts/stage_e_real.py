"""
Stage E, real components: the two honesty checks that validate whether the
decoder is trustworthy at all, run against the real V-JEPA2 encoder/decoder
on real SSv2 + Kinetics-mini tubelets. This directly answers the question
raised by the blurry live-feed images and the tightly-clustered trajectory
plot: is the decoder actually reading the latent, or has it partially
collapsed to a generic output regardless of input?

Adapted from latent2rgb/instruments.py, which was written against a
single-frame Encoder.encode(frame) signature -- the real V-JEPA2 encoder
needs a 2-frame tubelet, so this builds real 2-frame tubelets instead of
duplicating one frame.
"""

from __future__ import annotations

import json

import torch

from latent2rgb.decoder import MinimalDecoder
from latent2rgb.metrics import pixel_l2
from latent2rgb.video_source import VideoDirClipSource
from latent2rgb.vjepa_adapter import TUBELET_SIZE, VJEPA2Encoder, load_encoder_predictor


def get_tubelet(clip_source, clip_id, start):
    # get_clip returns frames[0 : start+k_max+1] (full prefix from clip
    # start, per the causal-context contract RolloutBuilder needs) -- slice
    # out just this tubelet's 2 frames.
    batch = clip_source.get_clip(clip_id, start, TUBELET_SIZE - 1)
    tubelet = batch.frames[start : start + TUBELET_SIZE]
    return tubelet.permute(1, 0, 2, 3)  # [C, T, H, W]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    raw_encoder, _ = load_encoder_predictor("checkpoints/vitl.pt", device=device)
    encoder = VJEPA2Encoder(raw_encoder, device=device)
    decoder = MinimalDecoder(embed_dim=encoder.embed_dim, grid_size=16, patch_size=16, tubelet_size=TUBELET_SIZE).to(device)
    decoder.load_state_dict(torch.load("decoder.pt", map_location=device))
    decoder.eval()

    ssv2 = VideoDirClipSource("data/ssv2/videos", extensions=("webm",))
    kinetics = VideoDirClipSource("data/kinetics_mini", extensions=("mp4",))

    results = {}

    # --- CHECK 1: context leakage, several pairs, both datasets ---
    print("=" * 70)
    print("CHECK 1: context leakage (real encoded latents, no predictor involved)")
    print("=" * 70)
    leakage_rows = []
    ssv2_ids = list(ssv2.clip_ids())[400:420]
    kinetics_ids = list(kinetics.clip_ids())[:20]
    for name, ids, source in [("ssv2", ssv2_ids, ssv2), ("kinetics_mini", kinetics_ids, kinetics)]:
        for i in range(0, len(ids) - 1, 2):
            id_a, id_b = ids[i], ids[i + 1]
            try:
                tub_a = get_tubelet(source, id_a, 8).to(device)
                tub_b = get_tubelet(source, id_b, 8).to(device)
            except AssertionError:
                continue
            with torch.no_grad():
                latent_a = encoder.encode(tub_a.unsqueeze(0))
                decoded_a = decoder.decode(latent_a).squeeze(0)
            same = pixel_l2(decoded_a, tub_a)
            cross = pixel_l2(decoded_a, tub_b)
            leakage_rows.append({"dataset": name, "clip_a": id_a, "clip_b": id_b, "same": same, "cross": cross, "gap": cross - same})
            print(f"  {name}: {id_a} vs {id_b}  same={same:.4f}  cross={cross:.4f}  gap={cross-same:+.4f}")

    avg_gap = sum(r["gap"] for r in leakage_rows) / len(leakage_rows)
    avg_same = sum(r["same"] for r in leakage_rows) / len(leakage_rows)
    results["leakage"] = {"rows": leakage_rows, "avg_gap": avg_gap, "avg_same": avg_same}
    print(f"\navg same-clip error: {avg_same:.4f}   avg leakage gap (cross-same): {avg_gap:.4f}")

    # --- CHECK 2: snapping, interpolate between two REAL different clips ---
    print("\n" + "=" * 70)
    print("CHECK 2: snapping (interpolate between two real, different tubelets)")
    print("=" * 70)
    snap_rows = []
    pairs = [(ssv2_ids[0], ssv2_ids[1], ssv2), (kinetics_ids[0], kinetics_ids[1], kinetics)]
    for id_a, id_b, source in pairs:
        tub_a = get_tubelet(source, id_a, 8).to(device)
        tub_b = get_tubelet(source, id_b, 8).to(device)
        with torch.no_grad():
            za = encoder.encode(tub_a.unsqueeze(0))
            zb = encoder.encode(tub_b.unsqueeze(0))
        n_steps = 9
        errors = []
        for i in range(n_steps):
            alpha = i / (n_steps - 1)
            z = (1 - alpha) * za + alpha * zb
            with torch.no_grad():
                decoded = decoder.decode(z).squeeze(0)
            errors.append(pixel_l2(decoded, tub_a))
        second_deriv = [abs(errors[i+1] - 2*errors[i] + errors[i-1]) for i in range(1, len(errors)-1)]
        max_curv = max(second_deriv)
        snap_rows.append({"clip_a": id_a, "clip_b": id_b, "errors": errors, "max_curvature": max_curv})
        print(f"  {id_a} -> {id_b}: errors={[f'{e:.4f}' for e in errors]}")
        print(f"    max_second_derivative={max_curv:.5f}")

    results["snapping"] = snap_rows

    with open("outputs/stage_e_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote outputs/stage_e_results.json")


if __name__ == "__main__":
    main()
