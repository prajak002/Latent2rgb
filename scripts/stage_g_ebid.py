"""
Stage G: EBID entropy-rate lead-time test, real V-JEPA2, real data --
Track B of the Horizon Ladder x EBID integration (see
github.com/HussainAther/pcc for the entropy-rate/lead-time method this
adapts; Track A, a ground-truth-anchored calibration of the same method on
Stream-DiffVSR, needs a CUDA machine and is not run here -- see
docs/ebid_integration.md).

Question: does the spectral entropy of the predicted latent token set --
computed from the predicted latent ALONE, no ground truth needed -- carry
an earlier warning of degradation than the horizon-indexed pixel_error
Stage D already reports? Two independent probes of that question, sharing
one loaded model:

  Part 1 (horizon-indexed): mirrors pilot.py. For each clip, track
  spectral_entropy(predicted_tokens[k]) across HORIZONS alongside the
  existing latent_drift[k]/pixel_error[k]. Compare entropy-rate's warning
  index against a pixel_error crash threshold via ebid.lead_time.

  Part 2 (reinjection-indexed): mirrors stage_f_reinjection.py. Track
  spectral_entropy along the pure-latent vs re-injected trajectories at
  each j, and ask whether entropy DIVERGENCE between the two paths crosses
  threshold before their pixel_l2 divergence (the existing "downstream"
  numbers) does.

  Part 3 (Hopf-scaling): EBID's actual closed-form claim, not just its
  lead-time heuristic -- entropy deficit scales linearly with distance
  past a Hopf instability threshold. Sweep a controlled perturbation
  amplitude epsilon at the re-injection point and fit entropy_deficit(j)
  vs epsilon; report the linear fit R^2 per clip.

RAW NUMBERS ONLY reported below -- no interpretation, same policy as every
other stage here.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from latent2rgb import select_device
from latent2rgb.data import RolloutBuilder
from latent2rgb.decoder import MinimalDecoder
from latent2rgb.ebid.entropy import entropy_trace, spectral_entropy
from latent2rgb.ebid.lead_time import compare_lead_time
from latent2rgb.interfaces import HORIZONS
from latent2rgb.metrics import (
    latent_drift as latent_drift_fn,
    pixel_error as pixel_error_fn,
    pixel_l2,
)
from latent2rgb.video_source import VideoDirClipSource
from latent2rgb.vjepa_adapter import TUBELET_SIZE, VJEPA2Encoder, VJEPA2Predictor, load_encoder_predictor

T = 8
K_REINJECT = 8
J_VALUES = [2, 4]
N_CLIPS_PER_DATASET = 25


# ---------------------------------------------------------------------------
# Part 1: horizon-indexed entropy (extends Stage D / pilot.py)
# ---------------------------------------------------------------------------


def run_horizon_entropy(name, clip_source, encoder, predictor, decoder, device, ids,
                         aggregate_floor, crash_multiple, min_t=8):
    builder = RolloutBuilder(clip_source, encoder, predictor, horizons=HORIZONS, device=device)
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
            pred_tok = sample.predicted_tokens[k]
            true_tok = sample.true_tubelet_tokens[k]
            true_frame = sample.true_frames[k]
            with torch.no_grad():
                decoded = decoder.decode(pred_tok.unsqueeze(0)).squeeze(0)
            decoded_last_frame = decoded[:, -1]

            ld = latent_drift_fn(pred_tok, true_tok)
            pe = pixel_error_fn(decoded_last_frame, true_frame)
            ent = spectral_entropy(pred_tok)
            rows.append({
                "dataset": name, "clip_id": clip_id, "t": t, "k": k,
                "latent_drift": ld, "pixel_error": pe,
                "excess_over_floor": pe - aggregate_floor,
                "spectral_entropy": ent,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, []

    crash_threshold = crash_multiple * aggregate_floor
    leads = []
    for clip_id, g in df.groupby("clip_id"):
        g = g.sort_values("k")
        result = compare_lead_time(
            xs=g["k"].tolist(),
            entropy=g["spectral_entropy"].tolist(),
            reference_signal=g["excess_over_floor"].tolist(),
            reference_crash_threshold=crash_threshold,
            k_consecutive=1,  # only 5 horizon points; can't require 2 consecutive
            smooth_w=1,       # no room to smooth over 5 points
        )
        leads.append({
            "dataset": name, "clip_id": clip_id,
            "crash_k": result.crash_x, "entropy_warn_k": result.entropy_warn_x,
            "lead": result.lead, "entropy_threshold_used": result.entropy_threshold_used,
        })
    return df, leads


# ---------------------------------------------------------------------------
# Part 2: reinjection-indexed entropy (extends Stage F / stage_f_reinjection.py)
# ---------------------------------------------------------------------------


def encode_context(encoder, frames, context_len, device):
    clip = frames[:context_len].permute(1, 0, 2, 3).unsqueeze(0).to(device)
    return encoder.encode(clip).squeeze(0)


def run_reinjection_entropy(name, clip_source, encoder, predictor, decoder, device, ids):
    rows = []
    for clip_id in ids:
        length = clip_source.clip_length(clip_id)
        needed = T + K_REINJECT + max(J_VALUES) + 2
        if length < needed:
            continue
        batch = clip_source.get_clip(clip_id, T, K_REINJECT + max(J_VALUES) + 1)
        frames = batch.frames.to(device)
        context = encode_context(encoder, frames, T, device)

        with torch.no_grad():
            p1 = predictor.rollout(context.unsqueeze(0), None, K_REINJECT).squeeze(0)
            decoded_p1 = decoder.decode(p1.unsqueeze(0))
            p1_reencoded = encoder.encode(decoded_p1).squeeze(0)

        entropy_p1 = spectral_entropy(p1)
        entropy_p1_reencoded = spectral_entropy(p1_reencoded)
        immediate_pixel_delta = pixel_l2(p1, p1_reencoded)

        context_pure = torch.cat([context, p1], dim=0)
        context_reinj = torch.cat([context, p1_reencoded], dim=0)

        row = {
            "dataset": name, "clip_id": clip_id,
            "entropy_p1": entropy_p1, "entropy_p1_reencoded": entropy_p1_reencoded,
            "immediate_pixel_delta": immediate_pixel_delta,
            "j_values": J_VALUES, "entropy_pure": {}, "entropy_reinj": {},
            "entropy_divergence": {}, "pixel_divergence": {},
        }
        for j in J_VALUES:
            with torch.no_grad():
                pure_next = predictor.rollout(context_pure.unsqueeze(0), None, j).squeeze(0)
                reinj_next = predictor.rollout(context_reinj.unsqueeze(0), None, j).squeeze(0)
            e_pure = spectral_entropy(pure_next)
            e_reinj = spectral_entropy(reinj_next)
            row["entropy_pure"][j] = e_pure
            row["entropy_reinj"][j] = e_reinj
            row["entropy_divergence"][j] = abs(e_reinj - e_pure)
            row["pixel_divergence"][j] = pixel_l2(pure_next, reinj_next)
        rows.append(row)
        print(f"  {name}/{clip_id}: entropy_p1={entropy_p1:.4f} entropy_p1_reencoded={entropy_p1_reencoded:.4f} "
              + "  ".join(f"j={j}: entropy_div={row['entropy_divergence'][j]:.4f} "
                          f"pixel_div={row['pixel_divergence'][j]:.4f}" for j in J_VALUES))

    if not rows:
        return {"n": 0}

    def avg(key_path_fn):
        vals = [key_path_fn(r) for r in rows]
        return sum(vals) / len(vals)

    avg_entropy_divergence = {j: avg(lambda r, j=j: r["entropy_divergence"][j]) for j in J_VALUES}
    avg_pixel_divergence = {j: avg(lambda r, j=j: r["pixel_divergence"][j]) for j in J_VALUES}

    return {
        "n": len(rows),
        "avg_entropy_p1": avg(lambda r: r["entropy_p1"]),
        "avg_entropy_p1_reencoded": avg(lambda r: r["entropy_p1_reencoded"]),
        "avg_immediate_pixel_delta": avg(lambda r: r["immediate_pixel_delta"]),
        "avg_entropy_divergence": avg_entropy_divergence,
        "avg_pixel_divergence": avg_pixel_divergence,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Part 3: EBID Hopf-scaling test -- controlled Monte Carlo perturbation sweep
#
# EBID's actual closed-form claim (pcc/RECOMMENDED_STRUCTURE.md) is not just
# "entropy warns early" -- it's that entropy DEFICIT scales linearly with
# distance past a Hopf instability threshold (sigma0 - mu), because deficit
# ~ limit-cycle-amplitude^2 and amplitude ~ sqrt(sigma0-mu). We test the
# direct analog here: sweep a perturbation amplitude epsilon (plays the role
# of sigma0-mu) at the Stage F re-injection point, and fit entropy_deficit(j)
# vs epsilon. A strong linear fit (high R^2) is the empirical signature of
# that same scaling relationship in a real learned latent rollout, not a toy
# replicator system. A weak/no fit is a real negative result -- the model's
# degradation may not pass through a comparable regime at all.
# ---------------------------------------------------------------------------

EPSILONS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
PERTURB_SEED = 1234


def run_hopf_scaling(name, clip_source, encoder, predictor, decoder, device, ids):
    from latent2rgb.ebid.perturb import fit_entropy_deficit_scaling, perturb_tokens

    j = J_VALUES[-1]  # fixed downstream horizon; only epsilon varies
    rows = []
    for clip_id in ids:
        length = clip_source.clip_length(clip_id)
        needed = T + K_REINJECT + j + 2
        if length < needed:
            continue
        batch = clip_source.get_clip(clip_id, T, K_REINJECT + j + 1)
        frames = batch.frames.to(device)
        context = encode_context(encoder, frames, T, device)

        with torch.no_grad():
            p1 = predictor.rollout(context.unsqueeze(0), None, K_REINJECT).squeeze(0)
            pure_next = predictor.rollout(
                torch.cat([context, p1], dim=0).unsqueeze(0), None, j
            ).squeeze(0)
        entropy_pure_next = spectral_entropy(pure_next)

        gen = torch.Generator(device=p1.device if p1.device.type != "mps" else "cpu").manual_seed(PERTURB_SEED)
        entropy_deficit = []
        pixel_deficit = []
        for eps in EPSILONS:
            p1_pert = perturb_tokens(p1, eps, generator=gen if p1.device.type != "mps" else None)
            with torch.no_grad():
                pert_next = predictor.rollout(
                    torch.cat([context, p1_pert], dim=0).unsqueeze(0), None, j
                ).squeeze(0)
            entropy_deficit.append(spectral_entropy(pert_next) - entropy_pure_next)
            pixel_deficit.append(pixel_l2(pert_next, pure_next))

        fit = fit_entropy_deficit_scaling(EPSILONS, entropy_deficit)
        rows.append({
            "dataset": name, "clip_id": clip_id, "j": j,
            "epsilons": EPSILONS, "entropy_deficit": entropy_deficit, "pixel_deficit": pixel_deficit,
            "slope": fit.slope, "intercept": fit.intercept, "r_squared": fit.r_squared,
        })
        print(f"  {name}/{clip_id}: entropy_deficit(eps)={[round(v,4) for v in entropy_deficit]}  "
              f"slope={fit.slope:.4f} R^2={fit.r_squared:.3f}")

    if not rows:
        return {"n": 0}

    r2_vals = [r["r_squared"] for r in rows if not np.isnan(r["r_squared"])]
    return {
        "n": len(rows),
        "j": j,
        "epsilons": EPSILONS,
        "mean_r_squared": float(np.mean(r2_vals)) if r2_vals else float("nan"),
        "mean_slope": float(np.mean([r["slope"] for r in rows])),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/vitl.pt")
    ap.add_argument("--decoder-path", type=str, default="decoder.pt")
    ap.add_argument("--floor-json", type=str, default="floor.json")
    ap.add_argument("--num-clips", type=int, default=10)
    ap.add_argument("--crash-multiple", type=float, default=1.5,
                     help="crash threshold for Part 1 = crash_multiple * floor, in excess_over_floor units "
                          "(i.e. pixel_error - floor >= (crash_multiple-1)*floor)")
    ap.add_argument("--out-csv", type=str, default="outputs/stage_g_horizon_entropy.csv")
    ap.add_argument("--out-json", type=str, default="outputs/stage_g_reinjection_entropy.json")
    ap.add_argument("--out-hopf-json", type=str, default="outputs/stage_g_hopf_scaling.json")
    args = ap.parse_args()

    device = select_device()
    print(f"device: {device}")

    with open(args.floor_json) as f:
        aggregate_floor = json.load(f)["aggregate_floor"]

    raw_encoder, raw_predictor = load_encoder_predictor(args.checkpoint, device=device)
    encoder = VJEPA2Encoder(raw_encoder, device=device)
    predictor = VJEPA2Predictor(raw_predictor, device=device)
    decoder = MinimalDecoder(embed_dim=encoder.embed_dim, grid_size=16, patch_size=16, tubelet_size=TUBELET_SIZE).to(device)
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
    decoder.eval()

    all_horizon_dfs = []
    all_leads = []
    reinjection_results = {}
    hopf_results = {}

    for name, video_dir, ext, offset in [
        ("ssv2", "data/ssv2/videos", ("webm",), 600),
        ("kinetics_mini", "data/kinetics_mini", ("mp4",), 0),
    ]:
        clip_source = VideoDirClipSource(video_dir, extensions=ext)
        ids = list(clip_source.clip_ids())[offset: offset + args.num_clips]
        print(f"\n=== {name}: Part 1 (horizon-indexed entropy), {len(ids)} clips ===")
        df, leads = run_horizon_entropy(
            name, clip_source, encoder, predictor, decoder, device, ids,
            aggregate_floor, args.crash_multiple,
        )
        all_horizon_dfs.append(df)
        all_leads.extend(leads)

        print(f"\n=== {name}: Part 2 (reinjection-indexed entropy), {len(ids)} clips ===")
        reinjection_results[name] = run_reinjection_entropy(
            name, clip_source, encoder, predictor, decoder, device, ids,
        )

        print(f"\n=== {name}: Part 3 (Hopf-scaling perturbation sweep), {len(ids)} clips ===")
        hopf_results[name] = run_hopf_scaling(
            name, clip_source, encoder, predictor, decoder, device, ids,
        )

    horizon_df = pd.concat(all_horizon_dfs, ignore_index=True) if all_horizon_dfs else pd.DataFrame()
    horizon_df.to_csv(args.out_csv, index=False)

    print(f"\n--- Part 1 raw per-(dataset,k) means ---")
    if not horizon_df.empty:
        print(horizon_df.groupby(["dataset", "k"])[["latent_drift", "pixel_error", "spectral_entropy"]].mean())

    print(f"\n--- Part 1 lead-time per clip (crash_multiple={args.crash_multiple}) ---")
    leads_df = pd.DataFrame(all_leads)
    if not leads_df.empty:
        print(leads_df.to_string())
        n_with_lead = leads_df["lead"].notna().sum()
        print(f"\nclips with both a crash_k and an entropy_warn_k: {n_with_lead}/{len(leads_df)}")
        if n_with_lead:
            print(f"mean lead (positive = entropy warned earlier, in horizon-k units): "
                  f"{leads_df['lead'].dropna().mean():.3f}")

    with open(args.out_json, "w") as f:
        json.dump({
            "k_reinject": K_REINJECT, "j_values": J_VALUES, "t": T,
            "datasets": reinjection_results,
        }, f, indent=2)

    print(f"\n--- Part 2 summary ---")
    for name, r in reinjection_results.items():
        if r.get("n", 0) == 0:
            print(f"{name}: no valid clips")
            continue
        print(f"{name}: n={r['n']}  avg_entropy_divergence={ {j: round(v,4) for j,v in r['avg_entropy_divergence'].items()} }  "
              f"avg_pixel_divergence={ {j: round(v,4) for j,v in r['avg_pixel_divergence'].items()} }")

    with open(args.out_hopf_json, "w") as f:
        json.dump({"j": J_VALUES[-1], "epsilons": EPSILONS, "datasets": hopf_results}, f, indent=2)

    print(f"\n--- Part 3 summary (EBID Hopf-scaling: entropy_deficit vs perturbation epsilon) ---")
    for name, r in hopf_results.items():
        if r.get("n", 0) == 0:
            print(f"{name}: no valid clips")
            continue
        print(f"{name}: n={r['n']}  mean_R^2={r['mean_r_squared']:.3f}  mean_slope={r['mean_slope']:.4f}")

    print(f"\nwrote {args.out_csv}")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_hopf_json}")
    print("\nRAW NUMBERS ONLY ABOVE -- no interpretation. This is Track B (V-JEPA2 side) only; "
          "see docs/ebid_integration.md for Track A (Stream-DiffVSR, ground-truth-anchored calibration, "
          "not run in this environment).")


if __name__ == "__main__":
    main()
