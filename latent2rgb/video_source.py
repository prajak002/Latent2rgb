"""
Generic ClipSource for a directory of video files (webm, mp4, whatever
OpenCV can decode). Same decode/resize/crop logic used for SSv2; factored
out because Stage D now runs on more than one dataset to check whether the
flat-latent-drift pattern is SSv2-specific or general.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor

from .interfaces import ClipBatch, ClipSource


class VideoDirClipSource(ClipSource):
    def __init__(self, video_dir: str, crop_size: int = 256, extensions: Sequence[str] = ("webm", "mp4")):
        self.video_dir = video_dir
        self.crop_size = crop_size
        paths: List[str] = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(video_dir, "**", f"*.{ext}"), recursive=True))
        self._paths = sorted(paths)
        assert self._paths, f"no video files found under {video_dir}"
        # id = path relative to video_dir, extension stripped -- keeps ids
        # unique even when class-subfolder structure repeats basenames.
        self._id_to_path = {
            os.path.relpath(p, video_dir).rsplit(".", 1)[0]: p for p in self._paths
        }
        self._frame_cache: Dict[str, Tensor] = {}

    def clip_ids(self) -> Sequence[str]:
        return list(self._id_to_path.keys())

    def _decode(self, clip_id: str) -> Tensor:
        if clip_id in self._frame_cache:
            return self._frame_cache[clip_id]
        path = self._id_to_path[clip_id]
        cap = cv2.VideoCapture(path)
        frames: List[np.ndarray] = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(frame_bgr)
        cap.release()
        assert frames, f"decoded 0 frames from {path}"

        processed = []
        for frame_bgr in frames:
            h, w = frame_bgr.shape[:2]
            short = min(h, w)
            scale = self.crop_size / short
            new_w, new_h = round(w * scale), round(h * scale)
            resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            top = (new_h - self.crop_size) // 2
            left = (new_w - self.crop_size) // 2
            cropped = resized[top : top + self.crop_size, left : left + self.crop_size]
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            processed.append(rgb)

        arr = np.stack(processed).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        self._frame_cache[clip_id] = tensor
        return tensor

    def clip_length(self, clip_id: str) -> int:
        return self._decode(clip_id).shape[0]

    def get_clip(self, clip_id: str, t: int, k_max: int) -> ClipBatch:
        frames = self._decode(clip_id)
        n = t + k_max + 1
        assert n <= frames.shape[0], (
            f"clip {clip_id} has {frames.shape[0]} frames, need {n} (t={t}, k_max={k_max})"
        )
        return ClipBatch(clip_id=clip_id, t=t, frames=frames[:n], actions=None)
