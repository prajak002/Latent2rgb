"""Trains the action-free LeWM-style encoder+predictor jointly, end-to-end,
from raw pixels (Algorithm 3, arXiv 2603.19312) on real SSv2 + Kinetics-mini
clips, then trains a diagnostic decoder the same way every other family in
this repo does (encode-decode on real tubelets only, never on rollouts).

Unlike scripts/train_second_family.py (frozen DINOv2 + a trained predictor
head), THIS script trains the encoder itself -- the defining difference of
the LeWM recipe. That makes this run meaningfully slower per step; see
README Limitations for the undertraining-confound caveat this raises, and
why the SIGReg/prediction loss curves are reported alongside any downstream
result rather than only the downstream result.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn

from latent2rgb import select_device
from latent2rgb.decoder import MinimalDecoder
from latent2rgb.lewm_adapter import (
    EMBED_DIM, TUBELET_SIZE, LeWMEncoder, Projector, build_vit_tiny_from_scratch,
)
from latent2rgb.lewm_predictor import CausalPredictor, sigreg_term
from latent2rgb.metrics import pixel_l2
from latent2rgb.video_source import VideoDirClipSource


def collect_windows(clip_source, clip_ids, h_seq: int, windows_per_clip: int, seed: int, device):
    """Each window: h_seq*TUBELET_SIZE consecutive raw frames -> [T,C,H,W]."""
    gen = torch.Generator().manual_seed(seed)
    win_len = h_seq * TUBELET_SIZE
    out = []
    for i, clip_id in enumerate(clip_ids):
        print(f"    decoding clip {i+1}/{len(clip_ids)}: {clip_id}", flush=True)
        t0 = time.time()
        length = clip_source.clip_length(clip_id)
        print(f"      length={length} frames, decoded in {time.time()-t0:.1f}s", flush=True)
        max_start = length - win_len - 1
        if max_start < 0:
            continue
        n = min(windows_per_clip, max_start + 1)
        starts = torch.randperm(max_start + 1, generator=gen)[:n].tolist()
        for start in starts:
            batch = clip_source.get_clip(clip_id, t=start, k_max=win_len - 1)
            window = batch.frames[start:start + win_len].to(device)  # [win_len, C, H, W]
            out.append(window)
    return out


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
    ap.add_argument("--val-clips-per-dataset", type=int, default=6)
    ap.add_argument("--h-seq", type=int, default=4, help="tubelets per training window")
    ap.add_argument("--windows-per-clip", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sigreg-lambda", type=float, default=0.1)
    ap.add_argument("--decoder-epochs", type=int, default=30)
    ap.add_argument("--decoder-lr", type=float, default=1e-3)
    ap.add_argument("--encoder-out", type=str, default="lewm_encoder.pt")
    ap.add_argument("--predictor-out", type=str, default="lewm_predictor.pt")
    ap.add_argument("--decoder-out", type=str, default="decoder_lewm.pt")
    ap.add_argument("--floor-out", type=str, default="floor_lewm.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = select_device()
    print(f"device: {device}")
    torch.manual_seed(args.seed)

    vit = build_vit_tiny_from_scratch().to(device)
    enc_projector = Projector(EMBED_DIM).to(device)
    encoder = LeWMEncoder(vit, enc_projector, device=device)
    n_enc_params = sum(p.numel() for p in vit.parameters()) + sum(p.numel() for p in enc_projector.parameters())

    predictor = CausalPredictor(embed_dim=EMBED_DIM).to(device)
    predictor.projector = Projector(EMBED_DIM).to(device)
    n_pred_params = sum(p.numel() for p in predictor.parameters())
    print(f"encoder params: {n_enc_params/1e6:.1f}M   predictor params: {n_pred_params/1e6:.1f}M")

    train_ids, val_ids = {}, {}
    for name, video_dir, ext, offset in [
        ("ssv2", "data/ssv2/videos", ("webm",), 900),
        ("kinetics_mini", "data/kinetics_mini", ("mp4",), 40),
    ]:
        clip_source = VideoDirClipSource(video_dir, extensions=ext)
        ids = list(clip_source.clip_ids())[offset: offset + args.num_clips_per_dataset + args.val_clips_per_dataset]
        train_ids[name] = (clip_source, ids[:args.num_clips_per_dataset])
        val_ids[name] = (clip_source, ids[args.num_clips_per_dataset:])

    t0 = time.time()
    windows = []
    for name, (clip_source, ids) in train_ids.items():
        w = collect_windows(clip_source, ids, args.h_seq, args.windows_per_clip, args.seed, device)
        print(f"  {name}: {len(w)} training windows", flush=True)
        windows.extend(w)
    print(f"collected {len(windows)} windows ({args.h_seq} tubelets each) in {time.time()-t0:.1f}s", flush=True)

    params = list(vit.parameters()) + list(enc_projector.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    print("starting training loop...", flush=True)
    t0 = time.time()
    for step in range(args.steps):
        step_t0 = time.time()
        idx = torch.randint(0, len(windows), (min(args.batch_size, len(windows)),))
        batch = torch.stack([windows[i] for i in idx])  # [B, win_len, C, H, W]
        clip_in = batch.permute(0, 2, 1, 3, 4)            # [B, C, win_len, H, W]

        emb = encoder.encode(clip_in)                     # [B, h_seq, D] -- gradients flow (encoder not frozen)
        if step == 0:
            print(f"  step 0: encoded batch in {time.time()-step_t0:.2f}s, emb shape {tuple(emb.shape)}", flush=True)
        next_emb = predictor(emb)                         # [B, h_seq, D]
        if step == 0:
            print(f"  step 0: predictor forward done at {time.time()-step_t0:.2f}s", flush=True)

        pred_loss = nn.functional.mse_loss(emb[:, 1:], next_emb[:, :-1])
        sigreg_loss = torch.stack([sigreg_term(emb[:, t]) for t in range(emb.shape[1])]).mean()
        if step == 0:
            print(f"  step 0: losses computed at {time.time()-step_t0:.2f}s", flush=True)
        loss = pred_loss + args.sigreg_lambda * sigreg_loss

        opt.zero_grad()
        loss.backward()
        if step == 0:
            print(f"  step 0: backward done at {time.time()-step_t0:.2f}s", flush=True)
        opt.step()

        if step < 3 or (step + 1) % 25 == 0:
            print(f"  step {step+1}/{args.steps}  pred_loss={pred_loss.item():.5f}  sigreg={sigreg_loss.item():.5f}  "
                  f"total={loss.item():.5f}  step_time={time.time()-step_t0:.2f}s", flush=True)
    print(f"encoder+predictor training: {time.time()-t0:.1f}s", flush=True)

    torch.save(vit.state_dict(), args.encoder_out.replace(".pt", "_vit.pt"))
    torch.save(enc_projector.state_dict(), args.encoder_out)
    torch.save(predictor.state_dict(), args.predictor_out)
    print(f"wrote {args.encoder_out}, {args.encoder_out.replace('.pt','_vit.pt')}, {args.predictor_out}")

    # --- diagnostic decoder: encode-decode only on real tubelets, encoder now frozen ---
    for p in vit.parameters():
        p.requires_grad_(False)
    for p in enc_projector.parameters():
        p.requires_grad_(False)
    vit.eval()
    enc_projector.eval()

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

    decoder = MinimalDecoder(embed_dim=EMBED_DIM, grid_size=1, patch_size=256, tubelet_size=TUBELET_SIZE).to(device)
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
        json.dump({
            "aggregate_floor": aggregate_floor, "train_ids": all_train_ids, "val_ids": all_val_ids,
            "n_encoder_params": n_enc_params, "n_predictor_params": n_pred_params,
            "config": vars(args),
        }, f, indent=2)

    print(f"\naggregate_floor (val pixel L2, real clips, LeWM-style encoder, grid_size=1) = {aggregate_floor:.5f}")
    print(f"wrote {args.decoder_out}, {args.floor_out}")


if __name__ == "__main__":
    main()
