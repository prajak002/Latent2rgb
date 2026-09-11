"""
Data backbone: given any (ClipSource, Encoder, StatePredictor) triple that
satisfies the interfaces in interfaces.py, produce the per-horizon rollout
samples every diagnostic stage consumes.

Causal windowing for a tubelet_size=2 encoder (see vjepa_adapter.py):
  - context = frames[0 : context_len], context_len = largest even number
    <= t+1 (never includes frame t+1 or later -- strictly causal).
  - for horizon k, target_slot = (t+k) // 2, covering real frames
    {2*target_slot, 2*target_slot+1}. We use frame 2*target_slot+1 (the
    later of the pair) as the representative ground-truth frame for
    pixel-space comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
from torch import Tensor

from .interfaces import ClipSource, Encoder, HORIZONS, StatePredictor

TUBELET_SIZE = 2


@dataclass
class RolloutSample:
    clip_id: str
    t: int
    true_frames: Dict[int, Tensor]        # k -> representative frame at target tubelet, [C, H, W]
    true_tubelet_tokens: Dict[int, Tensor]  # k -> encode(real 2-frame tubelet at target), [N, D]
    predicted_tokens: Dict[int, Tensor]   # k -> predictor output at target tubelet, [N, D]
    context_tokens: Tensor                # [N_ctx, D]


class RolloutBuilder:
    def __init__(
        self,
        clip_source: ClipSource,
        encoder: Encoder,
        predictor: StatePredictor,
        horizons: Sequence[int] = HORIZONS,
        device: str = "cpu",
    ):
        self.clip_source = clip_source
        self.encoder = encoder
        self.predictor = predictor
        self.horizons = tuple(sorted(horizons))
        self.k_max = self.horizons[-1]
        self.device = device

    @torch.no_grad()
    def build(self, clip_id: str, t: int) -> RolloutSample:
        # +1 headroom: when t+k is even, the target tubelet is {t+k, t+k+1},
        # which needs one more frame than a naive t+k_max+1 fetch provides.
        batch = self.clip_source.get_clip(clip_id, t, self.k_max + 1)
        frames = batch.frames.to(self.device)  # [t+k_max+2, C, H, W]

        context_len = 2 * ((t + 1) // 2)
        assert context_len >= 2, "need at least one full tubelet (2 frames) of context; increase t"
        context_clip = frames[:context_len].permute(1, 0, 2, 3).unsqueeze(0)  # [1,C,T,H,W]
        context_tokens = self.encoder.encode(context_clip).squeeze(0)  # [N_ctx, D]

        true_frames = {}
        true_tubelet_tokens = {}
        predicted_tokens = {}
        for k in self.horizons:
            target_slot = (t + k) // TUBELET_SIZE
            num_ctx_tubelets = context_len // TUBELET_SIZE
            if target_slot < num_ctx_tubelets:
                continue  # k too small relative to tubelet_size at this t; caller should skip
            target_start = target_slot * TUBELET_SIZE
            target_end = target_start + TUBELET_SIZE
            if target_end > frames.shape[0]:
                continue  # not enough frames fetched for this horizon
            target_clip = frames[target_start:target_end].permute(1, 0, 2, 3).unsqueeze(0)
            true_tok = self.encoder.encode(target_clip).squeeze(0)  # [N, D]

            pred = self.predictor.rollout(context_tokens.unsqueeze(0), actions=None, k=k)
            pred = pred.squeeze(0)  # [N, D]

            true_frames[k] = frames[target_end - 1]  # later frame of the pair
            true_tubelet_tokens[k] = true_tok
            predicted_tokens[k] = pred

        return RolloutSample(
            clip_id=clip_id,
            t=t,
            true_frames=true_frames,
            true_tubelet_tokens=true_tubelet_tokens,
            predicted_tokens=predicted_tokens,
            context_tokens=context_tokens,
        )

    def iter_samples(self, clip_ids: Sequence[str], starts_per_clip: int, rng: torch.Generator, min_t: int = 3):
        """Yield RolloutSample for `starts_per_clip` random valid t per clip."""
        for clip_id in clip_ids:
            length = self.clip_source.clip_length(clip_id)
            max_t = length - self.k_max - 2
            if max_t < min_t:
                continue
            n = min(starts_per_clip, max_t - min_t + 1)
            ts = (torch.randperm(max_t - min_t + 1, generator=rng)[:n] + min_t).tolist()
            for t in ts:
                yield self.build(clip_id, t)
