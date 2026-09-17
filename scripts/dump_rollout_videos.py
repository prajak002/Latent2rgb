"""Renders true-vs-rollout video pairs for the results gallery: for a clip,
sample horizons k=2,4,...,32 (every 2 frames), decode the single-shot
rollout prediction at each k, and pair it against the real frame at that
same timestamp. Sequencing by increasing k is also increasing real time, so
this is a coarsely-sampled real playback vs. the model's future predictions
at each of those timestamps -- encoded as looping mp4s for a drag-compare
gallery.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch
from PIL import Image

from latent2rgb import select_device
from latent2rgb.decoder import MinimalDecoder
from latent2rgb.video_source import VideoDirClipSource
from latent2rgb.vjepa_adapter import TUBELET_SIZE, VJEPA2Encoder, VJEPA2Predictor, load_encoder_predictor

T = 8
K_VALUES = list(range(2, 33, 2))  # 2,4,...,32


def frame_to_pil(frame: torch.Tensor) -> Image.Image:
    arr = (frame.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def encode_mp4(frames_dir: Path, out_path: Path, fps: int = 4):
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "%03d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ], check=True, capture_output=True)


def build_clip_videos(clip_id, clip_source, encoder, predictor, decoder, device, out_dir: Path, tag: str):
    length = clip_source.clip_length(clip_id)
    needed = T + K_VALUES[-1] + 2
    if length < needed:
        return False

    batch = clip_source.get_clip(clip_id, T, K_VALUES[-1] + 1)
    frames = batch.frames.to(device)
    context = encoder.encode(frames[:T].permute(1, 0, 2, 3).unsqueeze(0)).squeeze(0)

    context_dir = out_dir / f"{tag}_context_frames"
    true_dir = out_dir / f"{tag}_true_frames"
    pred_dir = out_dir / f"{tag}_rollout_frames"
    context_dir.mkdir(parents=True, exist_ok=True)
    true_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for i in range(T):
        frame_to_pil(frames[i]).save(context_dir / f"{i:03d}.png")

    for i, k in enumerate(K_VALUES):
        target_slot = (T + k) // TUBELET_SIZE
        target_end = target_slot * TUBELET_SIZE + TUBELET_SIZE
        true_frame = frames[target_end - 1]

        with torch.no_grad():
            pred_tok = predictor.rollout(context.unsqueeze(0), None, k).squeeze(0)
            decoded = decoder.decode(pred_tok.unsqueeze(0)).squeeze(0)
        pred_frame = decoded[:, -1]

        frame_to_pil(true_frame).save(true_dir / f"{i:03d}.png")
        frame_to_pil(pred_frame).save(pred_dir / f"{i:03d}.png")

    encode_mp4(context_dir, out_dir / f"{tag}_context.mp4")
    encode_mp4(true_dir, out_dir / f"{tag}_true.mp4")
    encode_mp4(pred_dir, out_dir / f"{tag}_rollout.mp4")
    shutil.rmtree(context_dir)
    shutil.rmtree(true_dir)
    shutil.rmtree(pred_dir)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/vitl.pt")
    ap.add_argument("--decoder-path", type=str, default="decoder.pt")
    ap.add_argument("--num-clips", type=int, default=4)
    ap.add_argument("--out-dir", type=str, default="outputs/videos")
    args = ap.parse_args()

    device = select_device()
    print(f"device: {device}  k_values={K_VALUES}")

    raw_encoder, raw_predictor = load_encoder_predictor(args.checkpoint, device=device)
    encoder = VJEPA2Encoder(raw_encoder, device=device)
    predictor = VJEPA2Predictor(raw_predictor, device=device)
    decoder = MinimalDecoder(embed_dim=encoder.embed_dim, grid_size=16, patch_size=16, tubelet_size=TUBELET_SIZE).to(device)
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
    decoder.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for name, video_dir, ext, offset in [
        ("ssv2", "data/ssv2/videos", ("webm",), 700),
        ("kinetics_mini", "data/kinetics_mini", ("mp4",), 0),
    ]:
        clip_source = VideoDirClipSource(video_dir, extensions=ext)
        ids = list(clip_source.clip_ids())[offset: offset + 40]
        made = 0
        for clip_id in ids:
            if made >= args.num_clips:
                break
            tag = f"{name}_{clip_id.replace('/', '_')}"
            ok = build_clip_videos(clip_id, clip_source, encoder, predictor, decoder, device, out_dir, tag)
            if ok:
                made += 1
                manifest.append({"dataset": name, "clip_id": clip_id, "tag": tag})
                print(f"  {tag}: done ({made}/{args.num_clips})")

    import json
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {len(manifest)} clip video pairs to {out_dir}, manifest at {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
