"""
Dump PNGs comparing: real frame | floor reconstruction (encode+decode, no
rollout) | rolled-forward decode (the OOD causal predictor query) for a
handful of pilot clips/horizons. Appendix-style evidence that the decoder
works at all -- not the headline result (that's the curves/statistics).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from latent2rgb import select_device
from latent2rgb.data import RolloutBuilder
from latent2rgb.decoder import MinimalDecoder
from latent2rgb.interfaces import HORIZONS
from latent2rgb.ssv2 import SSv2ClipSource
from latent2rgb.vjepa_adapter import TUBELET_SIZE, VJEPA2Encoder, VJEPA2Predictor, load_encoder_predictor


def to_pil(frame: torch.Tensor) -> Image.Image:
    arr = (frame.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/vitl.pt")
    ap.add_argument("--video-dir", type=str, default="data/ssv2/videos")
    ap.add_argument("--decoder-path", type=str, default="decoder.pt")
    ap.add_argument("--clip-ids", type=str, nargs="+", default=["105489", "10564", "105928", "106046"])
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 8, 32])
    ap.add_argument("--out-dir", type=str, default="outputs/frames")
    args = ap.parse_args()

    device = select_device()
    raw_encoder, raw_predictor = load_encoder_predictor(args.checkpoint, device=device)
    encoder = VJEPA2Encoder(raw_encoder, device=device)
    predictor = VJEPA2Predictor(raw_predictor, device=device)
    decoder = MinimalDecoder(embed_dim=encoder.embed_dim, grid_size=16, patch_size=16, tubelet_size=TUBELET_SIZE).to(device)
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
    decoder.eval()

    clip_source = SSv2ClipSource(args.video_dir)
    builder = RolloutBuilder(clip_source, encoder, predictor, horizons=HORIZONS, device=device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for clip_id in args.clip_ids:
        length = clip_source.clip_length(clip_id)
        if length - HORIZONS[-1] - 2 < args.t:
            print(f"skip {clip_id}: too short ({length})")
            continue
        sample = builder.build(clip_id, args.t)

        # frame at t itself, for reference
        t_frame = clip_source.get_clip(clip_id, args.t, 0).frames[0]
        to_pil(t_frame).save(out_dir / f"{clip_id}_t{args.t}_context.png")

        for k in args.ks:
            if k not in sample.predicted_tokens:
                continue
            true_frame = sample.true_frames[k]
            true_tok = sample.true_tubelet_tokens[k]
            pred_tok = sample.predicted_tokens[k]

            with torch.no_grad():
                floor_recon = decoder.decode(true_tok.unsqueeze(0)).squeeze(0)[:, -1]  # encode+decode, no rollout
                rollout_decode = decoder.decode(pred_tok.unsqueeze(0)).squeeze(0)[:, -1]  # the OOD causal query

            base = f"{clip_id}_t{args.t}_k{k}"
            to_pil(true_frame).save(out_dir / f"{base}_true.png")
            to_pil(floor_recon).save(out_dir / f"{base}_floor_recon.png")
            to_pil(rollout_decode).save(out_dir / f"{base}_rollout_decode.png")
            manifest.append({"clip_id": clip_id, "t": args.t, "k": k})
            print(f"saved {base}_{{true,floor_recon,rollout_decode}}.png")

    print(f"\nwrote {len(manifest)} triples to {out_dir}")


if __name__ == "__main__":
    main()
