"""Trains the second-family predictor (DeltaMLPPredictor) and decoder on
real SSv2 + Kinetics-mini clips, using the frozen DINOv2 encoder. This is
the one training step this whole repo does beyond the original minimal
decoder: DINOv2 ships no video predictor, so unlike V-JEPA2's frozen
pretrained predictor, this one has to be fit here to have anything to query
at all. See latent2rgb/dinov2_adapter.py and delta_predictor.py for why this
is still a fair, if lightweight, "second architecture" check.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn

from latent2rgb import select_device
from latent2rgb.decoder import MinimalDecoder
from latent2rgb.delta_predictor import DeltaMLPPredictor
from latent2rgb.dinov2_adapter import DINOv2Encoder, TUBELET_SIZE, load_dinov2
from latent2rgb.interfaces import HORIZONS
from latent2rgb.metrics import pixel_l2
from latent2rgb.video_source import VideoDirClipSource


def collect_context_and_targets(clip_source, encoder, clip_ids, t, horizons, device):
    """Mirrors RolloutBuilder.build()'s windowing but skips the predictor
    call entirely -- this is training data collection, not evaluation."""
    samples = []
    k_max = horizons[-1]
    for clip_id in clip_ids:
        length = clip_source.clip_length(clip_id)
        if length - k_max - 2 < t:
            continue
        batch = clip_source.get_clip(clip_id, t, k_max + 1)
        frames = batch.frames.to(device)

        context_len = 2 * ((t + 1) // 2)
        context_clip = frames[:context_len].permute(1, 0, 2, 3).unsqueeze(0)
        with torch.no_grad():
            context_tokens = encoder.encode(context_clip).squeeze(0)

        targets = {}
        for k in horizons:
            target_slot = (t + k) // TUBELET_SIZE
            num_ctx_tubelets = context_len // TUBELET_SIZE
            if target_slot < num_ctx_tubelets:
                continue
            target_start = target_slot * TUBELET_SIZE
            target_end = target_start + TUBELET_SIZE
            if target_end > frames.shape[0]:
                continue
            target_clip = frames[target_start:target_end].permute(1, 0, 2, 3).unsqueeze(0)
            with torch.no_grad():
                true_tok = encoder.encode(target_clip).squeeze(0)
            targets[k] = true_tok
        if targets:
            samples.append({"clip_id": clip_id, "context_tokens": context_tokens, "targets": targets})
    return samples


def collect_tubelets(clip_source, clip_ids, tubelets_per_clip, seed, device):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for clip_id in clip_ids:
        length = clip_source.clip_length(clip_id)
        max_slot = length // TUBELET_SIZE - 1
        if max_slot < 0:
            continue
        n = min(tubelets_per_clip, max_slot + 1)
        slots = torch.randperm(max_slot + 1, generator=gen)[:n].tolist()
        for slot in slots:
            start = slot * TUBELET_SIZE
            batch = clip_source.get_clip(clip_id, t=start, k_max=TUBELET_SIZE - 1)
            tubelet = batch.frames[start:start + TUBELET_SIZE]
            out.append(tubelet.permute(1, 0, 2, 3).to(device))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips-per-dataset", type=int, default=16)
    ap.add_argument("--val-clips-per-dataset", type=int, default=4)
    ap.add_argument("--min-t", type=int, default=8)
    ap.add_argument("--predictor-epochs", type=int, default=40)
    ap.add_argument("--predictor-lr", type=float, default=1e-3)
    ap.add_argument("--decoder-epochs", type=int, default=30)
    ap.add_argument("--decoder-lr", type=float, default=1e-3)
    ap.add_argument("--predictor-out", type=str, default="predictor_dinov2.pt")
    ap.add_argument("--decoder-out", type=str, default="decoder_dinov2.pt")
    ap.add_argument("--floor-out", type=str, default="floor_dinov2.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = select_device()
    print(f"device: {device}")
    torch.manual_seed(args.seed)

    dinov2 = load_dinov2(device=device)
    encoder = DINOv2Encoder(dinov2, device=device)

    train_ids, val_ids = {}, {}
    for name, video_dir, ext, offset in [
        ("ssv2", "data/ssv2/videos", ("webm",), 800),
        ("kinetics_mini", "data/kinetics_mini", ("mp4",), 20),
    ]:
        clip_source = VideoDirClipSource(video_dir, extensions=ext)
        ids = list(clip_source.clip_ids())[offset: offset + args.num_clips_per_dataset + args.val_clips_per_dataset]
        train_ids[name] = (clip_source, ids[:args.num_clips_per_dataset])
        val_ids[name] = (clip_source, ids[args.num_clips_per_dataset:])

    # --- collect predictor training data ---
    t0 = time.time()
    train_samples = []
    for name, (clip_source, ids) in train_ids.items():
        s = collect_context_and_targets(clip_source, encoder, ids, args.min_t, HORIZONS, device)
        print(f"  {name}: {len(s)} training samples")
        train_samples.extend(s)
    print(f"collected {len(train_samples)} predictor-training samples in {time.time()-t0:.1f}s")

    # --- train predictor ---
    predictor = DeltaMLPPredictor(embed_dim=encoder.embed_dim).to(device)
    opt = torch.optim.Adam(predictor.parameters(), lr=args.predictor_lr)

    pairs = [(s["context_tokens"], k, tok) for s in train_samples for k, tok in s["targets"].items()]
    print(f"training predictor on {len(pairs)} (context, k, target) pairs")
    for epoch in range(args.predictor_epochs):
        perm = torch.randperm(len(pairs))
        total_loss = 0.0
        for idx in perm.tolist():
            ctx, k, target = pairs[idx]
            pred = predictor.rollout(ctx.unsqueeze(0), None, k).squeeze(0)
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  predictor epoch {epoch+1}/{args.predictor_epochs}  train_mse={total_loss/len(pairs):.5f}")
    torch.save(predictor.state_dict(), args.predictor_out)
    print(f"wrote {args.predictor_out}")

    # --- train decoder (encode-decode only, no rollout -- same as Stage C) ---
    all_train_ids = [cid for _, ids in train_ids.values() for cid in ids]
    all_val_ids = [cid for _, ids in val_ids.values() for cid in ids]
    id_to_source = {}
    for name, (clip_source, ids) in list(train_ids.items()) + list(val_ids.items()):
        for cid in ids:
            id_to_source[cid] = clip_source

    def collect_for(ids):
        out = []
        for cid in ids:
            out.extend(collect_tubelets(id_to_source[cid], [cid], tubelets_per_clip=3, seed=args.seed, device=device))
        return out

    train_clips = torch.stack(collect_for(all_train_ids))
    val_clips = torch.stack(collect_for(all_val_ids))
    print(f"decoder training: {train_clips.shape[0]} train tubelets, {val_clips.shape[0]} val tubelets")

    decoder = MinimalDecoder(embed_dim=encoder.embed_dim, grid_size=16, patch_size=16, tubelet_size=TUBELET_SIZE).to(device)
    opt2 = torch.optim.Adam(decoder.parameters(), lr=args.decoder_lr)
    with torch.no_grad():
        train_latents = encoder.encode(train_clips)

    for epoch in range(args.decoder_epochs):
        perm = torch.randperm(train_clips.shape[0])
        total_loss = 0.0
        for i in range(0, len(perm), 16):
            idx = perm[i:i + 16]
            recon = decoder(train_latents[idx])
            loss = nn.functional.mse_loss(recon, train_clips[idx])
            opt2.zero_grad()
            loss.backward()
            opt2.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  decoder epoch {epoch+1}/{args.decoder_epochs}  train_mse={total_loss/len(perm):.5f}")
    torch.save(decoder.state_dict(), args.decoder_out)

    with torch.no_grad():
        aggregate_floor = pixel_l2(decoder.decode(encoder.encode(val_clips)), val_clips)
    with open(args.floor_out, "w") as f:
        json.dump({"aggregate_floor": aggregate_floor, "train_ids": all_train_ids, "val_ids": all_val_ids, "config": vars(args)}, f, indent=2)

    print(f"\naggregate_floor (val pixel L2, real clips, DINOv2 encoder) = {aggregate_floor:.5f}")
    print(f"wrote {args.decoder_out}, {args.floor_out}")


if __name__ == "__main__":
    main()
