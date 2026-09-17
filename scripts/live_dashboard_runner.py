"""
Background worker for the live dashboard. Runs Stage D across TWO disjoint
datasets -- Something-Something v2 (hand-object manipulation, 12fps) and
Kinetics-mini (full-body sports/outdoor actions, 25fps) -- on the same
frozen V-JEPA2 ViT-L encoder/predictor/decoder, and writes cumulative
per-dataset results to outputs/live_results.json after every clip.

The point of running two datasets: a pattern that only shows up on SSv2
could be an artifact of that dataset (occlusion-heavy, close-up, low fps).
If the same qualitative shape shows up on Kinetics-mini too -- a genuinely
different domain, fps, and camera convention -- that's real evidence the
finding is about the frozen predictor, not about SSv2.

Also computes a 2D PCA projection of true vs predicted latent trajectories
(mean-pooled over each tubelet's 256 tokens down to one 1024-dim vector per
k, then PCA'd to 2D across everything collected so far) for the most
recently processed clip from each dataset.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

from latent2rgb import select_device
from latent2rgb.data import RolloutBuilder
from latent2rgb.decoder import MinimalDecoder
from latent2rgb.interfaces import HORIZONS
from latent2rgb.metrics import latent_drift as latent_drift_fn, pixel_error as pixel_error_fn
from latent2rgb.video_source import VideoDirClipSource
from latent2rgb.vjepa_adapter import TUBELET_SIZE, VJEPA2Encoder, VJEPA2Predictor, load_encoder_predictor

OUT_PATH = Path("outputs/live_results.json")
FRAMES_DIR = Path("outputs/live_frames")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def save_frame(tensor, path: Path):
    arr = (tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    tmp = path.with_suffix(".tmp.png")
    Image.fromarray(arr).save(tmp)
    tmp.replace(path)


CHECKPOINT = "checkpoints/vitl.pt"
DECODER_PATH = "decoder.pt"
FLOOR_JSON = "floor.json"
T = 8
HISTORY_SIZE = 8  # how many recent full-horizon clips to keep browsable per dataset

DATASETS = {
    "ssv2": {
        "video_dir": "data/ssv2/videos",
        "extensions": ("webm",),
        "n_clips": 150,
        "offset": 300,  # disjoint from Stage C's train/val clips and the original 20-clip pilot
    },
    "kinetics_mini": {
        "video_dir": "data/kinetics_mini",
        "extensions": ("mp4",),
        "n_clips": 100,
        "offset": 0,
    },
}


def atomic_write(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(path)


def compute_by_k(rows, dataset=None):
    by_k = {}
    for k in HORIZONS:
        vals_ld = [r["latent_drift"] for r in rows if r["k"] == k and (dataset is None or r["dataset"] == dataset)]
        vals_pe = [r["pixel_error"] for r in rows if r["k"] == k and (dataset is None or r["dataset"] == dataset)]
        if vals_ld:
            by_k[k] = {
                "n": len(vals_ld),
                "latent_drift_mean": float(np.mean(vals_ld)),
                "latent_drift_std": float(np.std(vals_ld)),
                "pixel_error_mean": float(np.mean(vals_pe)),
                "pixel_error_std": float(np.std(vals_pe)),
            }
    return by_k


def main():
    device = select_device()
    print(f"device: {device}")

    with open(FLOOR_JSON) as f:
        floor = json.load(f)["aggregate_floor"]

    raw_encoder, raw_predictor = load_encoder_predictor(CHECKPOINT, device=device)
    encoder = VJEPA2Encoder(raw_encoder, device=device)
    predictor = VJEPA2Predictor(raw_predictor, device=device)
    decoder = MinimalDecoder(embed_dim=encoder.embed_dim, grid_size=16, patch_size=16, tubelet_size=TUBELET_SIZE).to(device)
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
    decoder.eval()

    builders = {}
    per_dataset_ids = {}
    clip_sources = {}
    for name, cfg in DATASETS.items():
        clip_source = VideoDirClipSource(cfg["video_dir"], extensions=cfg["extensions"])
        clip_sources[name] = clip_source
        builders[name] = RolloutBuilder(clip_source, encoder, predictor, horizons=HORIZONS, device=device)
        per_dataset_ids[name] = list(clip_source.clip_ids())[cfg["offset"] : cfg["offset"] + cfg["n_clips"]]

    # interleave round-robin across datasets so the live comparison fills in
    # together rather than one dataset finishing before the next starts
    jobs = []
    max_len = max(len(v) for v in per_dataset_ids.values())
    for i in range(max_len):
        for name in DATASETS:
            if i < len(per_dataset_ids[name]):
                jobs.append((name, per_dataset_ids[name][i], clip_sources[name]))

    target_n = len(jobs)
    print(f"total jobs across datasets: {target_n}")

    rows = []
    last_trajectory = {name: None for name in DATASETS}
    pooled = {name: {k: {"true": [], "pred": []} for k in HORIZONS} for name in DATASETS}

    started_at = time.time()
    processed = 0
    skipped = 0
    next_seq = {name: 0 for name in DATASETS}
    frame_history = {name: [] for name in DATASETS}  # oldest -> newest, list of {seq, clip_id}

    for name, clip_id, clip_source in jobs:
        length = clip_source.clip_length(clip_id)
        if length - HORIZONS[-1] - 2 < T:
            skipped += 1
            continue
        try:
            sample = builders[name].build(clip_id, T)
        except Exception as e:
            print(f"skip {name}/{clip_id}: {e}")
            skipped += 1
            continue

        traj_true, traj_pred = [], []
        decoded_by_k = {}
        for k in HORIZONS:
            if k not in sample.predicted_tokens:
                continue
            true_tok = sample.true_tubelet_tokens[k]
            pred_tok = sample.predicted_tokens[k]
            true_frame = sample.true_frames[k]

            with torch.no_grad():
                decoded = decoder.decode(pred_tok.unsqueeze(0)).squeeze(0)
            decoded_last_frame = decoded[:, -1]
            decoded_by_k[k] = decoded_last_frame

            ld = latent_drift_fn(pred_tok, true_tok)
            pe = pixel_error_fn(decoded_last_frame, true_frame)
            rows.append({"dataset": name, "clip_id": clip_id, "k": k, "latent_drift": ld, "pixel_error": pe})

            true_vec = true_tok.mean(dim=0).cpu().numpy()
            pred_vec = pred_tok.mean(dim=0).cpu().numpy()
            pooled[name][k]["true"].append(true_vec)
            pooled[name][k]["pred"].append(pred_vec)
            traj_true.append((k, true_vec))
            traj_pred.append((k, pred_vec))

        # only treat a clip as "the live feed" when every horizon resolved --
        # a partial clip (short, near the length cutoff) makes a degenerate
        # 1-2 point trajectory/feed that looks broken rather than informative
        if len(traj_true) == len(HORIZONS):
            last_trajectory[name] = {"clip_id": clip_id, "true": traj_true, "pred": traj_pred}

            seq = next_seq[name]
            next_seq[name] += 1
            prefix = f"{name}_{seq}"
            batch = clip_source.get_clip(clip_id, T, 0)
            save_frame(batch.frames[0], FRAMES_DIR / f"{prefix}_context.png")
            for k in HORIZONS:
                save_frame(sample.true_frames[k], FRAMES_DIR / f"{prefix}_k{k}_true.png")
                save_frame(decoded_by_k[k], FRAMES_DIR / f"{prefix}_k{k}_rollout.png")

            frame_history[name].append({"seq": seq, "clip_id": clip_id})
            if len(frame_history[name]) > HISTORY_SIZE:
                old = frame_history[name].pop(0)
                old_prefix = f"{name}_{old['seq']}"
                for f in FRAMES_DIR.glob(f"{old_prefix}_*.png"):
                    f.unlink(missing_ok=True)

        processed += 1

        by_k_all = compute_by_k(rows)
        by_k_per_dataset = {name2: compute_by_k(rows, dataset=name2) for name2 in DATASETS}

        trajectories = {}
        for name2 in DATASETS:
            all_true = [v for k in HORIZONS for v in pooled[name2][k]["true"]]
            all_pred = [v for k in HORIZONS for v in pooled[name2][k]["pred"]]
            if len(all_true) >= 10 and last_trajectory[name2] is not None:
                X = np.stack(all_true + all_pred)
                pca = PCA(n_components=2)
                pca.fit(X)
                X_proj = pca.transform(X)
                # robust (5th/95th percentile) bounds, not strict min/max --
                # a couple of outlier tubelets otherwise blow out the scale
                # and squash every real trajectory into a corner
                x_lo, x_hi = np.percentile(X_proj[:, 0], [5, 95])
                y_lo, y_hi = np.percentile(X_proj[:, 1], [5, 95])
                bounds = {
                    "x_min": float(x_lo), "x_max": float(x_hi),
                    "y_min": float(y_lo), "y_max": float(y_hi),
                }
                lt = last_trajectory[name2]
                proj_true = [(k, pca.transform(v[None, :])[0].tolist()) for k, v in lt["true"]]
                proj_pred = [(k, pca.transform(v[None, :])[0].tolist()) for k, v in lt["pred"]]
                trajectories[name2] = {
                    "clip_id": lt["clip_id"],
                    "true": proj_true,
                    "pred": proj_pred,
                    "bounds": bounds,
                    "explained_variance": pca.explained_variance_ratio_.tolist(),
                }

        atomic_write(OUT_PATH, {
            "updated_at": time.time(),
            "elapsed_s": time.time() - started_at,
            "processed": processed,
            "skipped": skipped,
            "target_n": target_n,
            "floor": floor,
            "by_k_all": by_k_all,
            "by_k_per_dataset": by_k_per_dataset,
            "trajectories": trajectories,
            "frame_history": frame_history,
            "datasets": list(DATASETS.keys()),
        })
        print(f"[{processed}/{target_n}] {name}/{clip_id}  (skipped {skipped})")

    print("done")


if __name__ == "__main__":
    main()
