"""
Stage C, real components: train MinimalDecoder on real SSv2 tubelets
(encode-then-decode, no rollout). Real V-JEPA2 encoder is frozen throughout
-- only the decoder's parameters are optimized. Plain pixel MSE loss only.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn

from latent2rgb.decoder import MinimalDecoder
from latent2rgb.metrics import pixel_l2
from latent2rgb.ssv2 import SSv2ClipSource
from latent2rgb.vjepa_adapter import TOKENS_PER_TUBELET, TUBELET_SIZE, VJEPA2Encoder, load_encoder_predictor


def collect_tubelets(clip_source, clip_ids, tubelets_per_clip: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    out = []  # (clip_id, tubelet_clip [C,T,H,W])
    for clip_id in clip_ids:
        length = clip_source.clip_length(clip_id)
        max_slot = length // TUBELET_SIZE - 1
        if max_slot < 0:
            continue
        n = min(tubelets_per_clip, max_slot + 1)
        slots = torch.randperm(max_slot + 1, generator=gen)[:n].tolist()
        for slot in slots:
            start = slot * TUBELET_SIZE
            # get_clip returns frames[0 : t+k_max+1] (full prefix from clip
            # start, per the causal-context contract RolloutBuilder needs) --
            # slice out just this tubelet's 2 frames.
            batch = clip_source.get_clip(clip_id, t=start, k_max=TUBELET_SIZE - 1)
            tubelet = batch.frames[start : start + TUBELET_SIZE]
            clip = tubelet.permute(1, 0, 2, 3)  # [C, T, H, W]
            out.append((clip_id, clip))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/vitl.pt")
    ap.add_argument("--video-dir", type=str, default="data/ssv2/videos")
    ap.add_argument("--num-clips", type=int, default=150)
    ap.add_argument("--tubelets-per-clip", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", type=str, default="floor.json")
    ap.add_argument("--decoder-out", type=str, default="decoder.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    clip_source = SSv2ClipSource(args.video_dir)
    all_ids = list(clip_source.clip_ids())[: args.num_clips]
    n_val = max(1, int(len(all_ids) * args.val_frac))
    val_ids = all_ids[:n_val]
    train_ids = all_ids[n_val:]
    print(f"train clips: {len(train_ids)}  val clips: {len(val_ids)}")

    raw_encoder, _ = load_encoder_predictor(args.checkpoint, device=device)
    encoder = VJEPA2Encoder(raw_encoder, device=device)

    t0 = time.time()
    train_data = collect_tubelets(clip_source, train_ids, args.tubelets_per_clip, args.seed)
    val_data = collect_tubelets(clip_source, val_ids, args.tubelets_per_clip, args.seed + 1)
    print(f"collected {len(train_data)} train tubelets, {len(val_data)} val tubelets in {time.time()-t0:.1f}s")

    decoder = MinimalDecoder(embed_dim=encoder.embed_dim, grid_size=16, patch_size=16, tubelet_size=TUBELET_SIZE).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    train_clips = torch.stack([c for _, c in train_data]).to(device)
    t0 = time.time()
    with torch.no_grad():
        train_latents = encoder.encode(train_clips)
    print(f"encoded {train_clips.shape[0]} train tubelets in {time.time()-t0:.1f}s -> latents {tuple(train_latents.shape)}")

    for epoch in range(args.epochs):
        perm = torch.randperm(train_clips.shape[0])
        total_loss = 0.0
        for i in range(0, len(perm), 16):
            idx = perm[i : i + 16]
            latents = train_latents[idx]
            targets = train_clips[idx]
            recon = decoder(latents)
            loss = nn.functional.mse_loss(recon, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch+1}/{args.epochs}  train_mse={total_loss/len(perm):.5f}")

    torch.save(decoder.state_dict(), args.decoder_out)

    per_clip = {}
    by_clip: dict[str, list[torch.Tensor]] = {}
    for clip_id, clip in val_data:
        by_clip.setdefault(clip_id, []).append(clip)
    all_val_clips = []
    for clip_id, clips in by_clip.items():
        clips_t = torch.stack(clips).to(device)
        floor = pixel_l2(decoder.decode(encoder.encode(clips_t)).detach(), clips_t)
        per_clip[clip_id] = floor
        all_val_clips.append(clips_t)

    aggregate_clips = torch.cat(all_val_clips)
    with torch.no_grad():
        aggregate_floor = pixel_l2(decoder.decode(encoder.encode(aggregate_clips)), aggregate_clips)

    result = {
        "aggregate_floor": aggregate_floor,
        "per_clip_floor": per_clip,
        "val_ids": val_ids,
        "train_ids": train_ids,
        "config": vars(args),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\naggregate_floor (val pixel L2, real SSv2 + real V-JEPA2 encoder) = {aggregate_floor:.5f}")
    print(f"wrote {args.out} and {args.decoder_out}")


if __name__ == "__main__":
    main()
