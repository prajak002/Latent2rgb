"""
Model-agnostic interfaces for the diagnostic protocol.

REVISED after reading V-JEPA2's actual source (vendor/vjepa2): a ViT-based
video encoder's "latent" for one instant is not a flat vector, and its
predictor does not take a single-frame latent_t -- it takes a token
sequence for a CONTEXT WINDOW of frames. That is reflected here.

Shape convention used everywhere in this package:
    frame        : float tensor [B, C, H, W], values in [0, 1]
    clip         : float tensor [B, C, T, H, W], values in [0, 1], T even
                   (tubelet_size=2 for V-JEPA2 -- see StatePredictor)
    latent       : float tensor [B, N, D] -- a token sequence. N depends on
                   how many frames were encoded (context grows over time),
                   D is the model's embedding dim. Never assume a fixed N.
    action       : float tensor [B, A] or None (no action-conditioning)

IMPORTANT, carried over from the V-JEPA2 adapter report: the base V-JEPA2
predictor was pretrained ONLY for masked spatiotemporal completion within an
already-fully-observed clip (context and target both sampled from across the
whole clip). It was never trained on a strictly-causal "context = past only,
target = future only" mask pattern. StatePredictor.rollout below implements
exactly that causal pattern anyway, because it's architecturally valid to
construct -- but every result produced through it characterizes the frozen
predictor under an out-of-distribution querying regime, not its native task.
Say so wherever these numbers are reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class Encoder(Protocol):
    """Maps a real RGB clip to the predictor's native token sequence."""

    def encode(self, clip: Tensor) -> Tensor:
        """clip: [B, C, T, H, W] (T even) -> tokens: [B, N, D]"""
        ...

    @property
    def embed_dim(self) -> int:
        ...

    @property
    def tokens_per_tubelet(self) -> int:
        """Number of tokens produced per tubelet_size-frame time slot
        (i.e. spatial grid size: (H/patch_size)*(W/patch_size))."""
        ...


@runtime_checkable
class Decoder(Protocol):
    """Maps latents back to RGB frames. This is the instrument under test."""

    def decode(self, latents: Tensor) -> Tensor:
        """latent: [B, N, D] (one tubelet's worth of tokens, N=tokens_per_tubelet)
        -> frames: [B, C, tubelet_size, H, W]"""
        ...


@runtime_checkable
class StatePredictor(Protocol):
    """
    The frozen, latent-only dynamics model. NEVER fine-tuned by any stage
    in this package -- every stage takes a StatePredictor and only calls
    .rollout on it.
    """

    def rollout(
        self, context_tokens: Tensor, actions: Optional[Tensor], k: int
    ) -> Tensor:
        """
        context_tokens : [B, N_ctx, D] -- encoder output for frames [0..t],
                          N_ctx = num_context_tubelets * tokens_per_tubelet.
        actions  : [B, k_max, A] or None -- actions for steps t..t+k-1.
                   Implementations that are not action-conditioned must
                   accept actions=None and ignore it.
        k        : how many raw frames forward to predict (t+k). Because
                   V-JEPA2's tubelet_size=2, the actual query resolves to
                   the tubelet containing frame t+k, which may jointly
                   represent frames {t+k-1, t+k} -- see the adapter.
        returns  : predicted tokens for that tubelet, [B, tokens_per_tubelet, D]

        Contract: calling rollout(context_tokens, actions, k) is a single
        query, not a k-step loop -- V-JEPA2's predictor answers "what's at
        this target position" in one forward pass given context. Other
        StatePredictor implementations may loop internally instead; callers
        must not assume statefulness across calls.
        """
        ...


@dataclass
class ClipBatch:
    """
    Everything one (clip, start index t) sample needs. Unlike a simpler
    single-frame-latent setup, V-JEPA2's predictor needs the FULL prefix
    [0..t] as context (not just frame t), so frames spans from clip start
    through t+k_max, not just from t onward.
    """

    clip_id: str
    t: int
    frames: Tensor            # [t+k_max+1, C, H, W] -- frames[0] is clip start
    actions: Optional[Tensor] # [k_max, A] or None, actions for steps t..t+k_max-1


@runtime_checkable
class ClipSource(Protocol):
    """Supplies ground-truth clips. Implement this for a real dataset."""

    def clip_ids(self) -> Sequence[str]:
        ...

    def get_clip(self, clip_id: str, t: int, k_max: int) -> ClipBatch:
        """Return frames[t..t+k_max] and actions[t..t+k_max-1] for one clip."""
        ...

    def clip_length(self, clip_id: str) -> int:
        ...


# k=1 dropped for the real V-JEPA2 adapter: tubelet_size=2 means the
# smallest resolvable target unit is a 2-frame tubelet, so k=1 from an
# even t would target a tubelet already partly inside the context window
# (not a genuine future prediction). See vjepa_adapter.py.
HORIZONS: Sequence[int] = (2, 4, 8, 16, 32)
