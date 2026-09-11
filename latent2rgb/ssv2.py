"""Thin alias kept for existing scripts/imports -- see video_source.py,
which SSv2ClipSource now shares with every other dataset."""

from __future__ import annotations

from .video_source import VideoDirClipSource


class SSv2ClipSource(VideoDirClipSource):
    def __init__(self, video_dir: str, crop_size: int = 256):
        super().__init__(video_dir, crop_size=crop_size, extensions=("webm",))
