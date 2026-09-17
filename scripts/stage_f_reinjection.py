"""
Stage F, re-injection divergence -- real components, both datasets. This is
Gap 3 (compounding-leak measurement) and does NOT need actions, unlike
counterfactual faithfulness (Gap 2), which genuinely requires an
action-conditioned model V-JEPA2 base doesn't have -- see the writeup.

Mechanism, adapted to a causal-masked-completion predictor used
autoregressively (rather than a native step function):
  1. context = encode(real frames [0..t])                          (4 tubelets)
  2. query k_reinject -> predicted tubelet P1 (never decoded)       ("pure" chain)
  3. decode P1 -> pixels -> re-encode -> P1_reencoded               (one pixel round-trip)
  4. extend context two ways: context_pure = context + P1,
                               context_reinj = context + P1_reencoded
  5. for j in {2, 4}: query k=j from each extended context, compare
     the two resulting predictions. Growing distance with j = the
     decode error compounds through the dynamics rather than staying
     a fixed one-time discrepancy -- strictly stronger than a single
     decode-reencode-compare.
"""

from __future__ import annotations

import json

import torch

from latent2rgb import select_device
from latent2rgb.decoder import MinimalDecoder
from latent2rgb.metrics import pixel_l2
from latent2rgb.video_source import VideoDirClipSource
from latent2rgb.vjepa_adapter import TUBELET_SIZE, VJEPA2Encoder, VJEPA2Predictor, load_encoder_predictor

T = 8
K_REINJECT = 8
J_VALUES = [2, 4]
N_CLIPS_PER_DATASET = 25


def encode_context(encoder, frames, context_len, device):
    clip = frames[:context_len].permute(1, 0, 2, 3).unsqueeze(0).to(device)
    return encoder.encode(clip).squeeze(0)


def run_dataset(name, clip_source, encoder, predictor, decoder, device, ids):
    rows = []
    immediate_deltas, downstream = [], {j: [] for j in J_VALUES}

    for clip_id in ids:
        length = clip_source.clip_length(clip_id)
        needed = T + K_REINJECT + max(J_VALUES) + 2
        if length < needed:
            continue
        batch = clip_source.get_clip(clip_id, T, K_REINJECT + max(J_VALUES) + 1)
        frames = batch.frames.to(device)

        context = encode_context(encoder, frames, T, device)  # [n_ctx, D]

        with torch.no_grad():
            p1 = predictor.rollout(context.unsqueeze(0), None, K_REINJECT).squeeze(0)  # [256, D]
            decoded_p1 = decoder.decode(p1.unsqueeze(0))  # [1, C, T, H, W]
            p1_reencoded = encoder.encode(decoded_p1).squeeze(0)  # [256, D]

        immediate_delta = pixel_l2(p1, p1_reencoded)
        immediate_deltas.append(immediate_delta)

        context_pure = torch.cat([context, p1], dim=0)
        context_reinj = torch.cat([context, p1_reencoded], dim=0)

        row = {"clip_id": clip_id, "immediate_delta": immediate_delta, "downstream": {}}
        for j in J_VALUES:
            with torch.no_grad():
                pure_next = predictor.rollout(context_pure.unsqueeze(0), None, j).squeeze(0)
                reinj_next = predictor.rollout(context_reinj.unsqueeze(0), None, j).squeeze(0)
            d = pixel_l2(pure_next, reinj_next)
            downstream[j].append(d)
            row["downstream"][j] = d
        rows.append(row)
        print(f"  {name}/{clip_id}: immediate={immediate_delta:.3f}  " +
              "  ".join(f"j={j}:{row['downstream'][j]:.3f}" for j in J_VALUES))

    n = len(rows)
    avg_immediate = sum(immediate_deltas) / n if n else float("nan")
    avg_downstream = {j: sum(downstream[j]) / n for j in J_VALUES} if n else {}

    # divergence rate: slope of avg_downstream vs j (simple two-point slope
    # when len(J_VALUES)==2, else least squares)
    js = J_VALUES
    ys = [avg_downstream[j] for j in js]
    if len(js) >= 2:
        x_mean = sum(js) / len(js)
        y_mean = sum(ys) / len(ys)
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(js, ys))
        den = sum((x - x_mean) ** 2 for x in js)
        rate = num / den if den > 1e-8 else float("nan")
    else:
        rate = float("nan")

    return {
        "n": n,
        "avg_immediate_delta": avg_immediate,
        "avg_downstream": avg_downstream,
        "divergence_rate": rate,
        "rows": rows,
    }


def main():
    device = select_device()
    print(f"device: {device}")

    raw_encoder, raw_predictor = load_encoder_predictor("checkpoints/vitl.pt", device=device)
    encoder = VJEPA2Encoder(raw_encoder, device=device)
    predictor = VJEPA2Predictor(raw_predictor, device=device)
    decoder = MinimalDecoder(embed_dim=encoder.embed_dim, grid_size=16, patch_size=16, tubelet_size=TUBELET_SIZE).to(device)
    decoder.load_state_dict(torch.load("decoder.pt", map_location=device))
    decoder.eval()

    results = {}
    for name, video_dir, ext, offset in [
        ("ssv2", "data/ssv2/videos", ("webm",), 600),
        ("kinetics_mini", "data/kinetics_mini", ("mp4",), 0),
    ]:
        clip_source = VideoDirClipSource(video_dir, extensions=ext)
        ids = list(clip_source.clip_ids())[offset : offset + N_CLIPS_PER_DATASET]
        print(f"\n=== {name} ({len(ids)} candidate clips) ===")
        results[name] = run_dataset(name, clip_source, encoder, predictor, decoder, device, ids)
        r = results[name]
        print(f"\n{name}: n={r['n']}  avg_immediate_delta={r['avg_immediate_delta']:.4f}  "
              f"avg_downstream={ {j: round(v,4) for j,v in r['avg_downstream'].items()} }  "
              f"divergence_rate={r['divergence_rate']:.4f}")

    with open("outputs/stage_f_results.json", "w") as f:
        json.dump({"k_reinject": K_REINJECT, "j_values": J_VALUES, "t": T, "datasets": results}, f, indent=2)
    print("\nwrote outputs/stage_f_results.json")


if __name__ == "__main__":
    main()
