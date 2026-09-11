"""
Real adapter around V-JEPA2 ViT-L/16, built by importing the vendored
source directly (vendor/vjepa2) rather than reimplementing the model.

Read directly from source (not summarized) before writing this:
  - vendor/vjepa2/src/hub/backbones.py      (_make_vjepa2_model, vjepa2_vit_large)
  - vendor/vjepa2/src/models/vision_transformer.py  (encoder forward, RoPE, PatchEmbed3D)
  - vendor/vjepa2/src/models/predictor.py   (VisionTransformerPredictor.forward)
  - vendor/vjepa2/src/masks/utils.py        (apply_masks index convention)
  - vendor/vjepa2/notebooks/vjepa2_demo.py, energy_landscape_example.ipynb

KEY FACTS ESTABLISHED FROM SOURCE:
  - encoder(x): x is [B, C, T, H, W] (T even, tubelet_size=2). Plain call,
    no masks, RoPE-based positions computed dynamically from actual T/H/W --
    NOT tied to a fixed nominal clip length. Output: [B, N, D],
    N = (T/2)*(H/16)*(W/16), D=1024 for ViT-L.
  - PatchEmbed3D flattens Conv3d output as .flatten(2) on [B,D,T',H',W'],
    i.e. token index = t_slot*(H'*W') + h*W' + w (time-major, row-major
    within each time slot). This is what makes masks_x/masks_y index math
    below correct.
  - predictor.forward(x, masks_x, masks_y): x is ALREADY-GATHERED context
    tokens (caller applies apply_masks to the encoder output before calling).
    masks_x/masks_y are [B, K] index tensors into the flattened token grid,
    used only to (a) place RoPE positions per Block call and (b) restore
    output order. The predictor was constructed with num_frames=64,
    tubelet_size=2, img_size=256, patch_size=16 -> grid_depth=32,
    grid_height=grid_width=16 -> nominal grid of 32*16*16=8192 positions.
    We only ever populate a small subset of that nominal grid (real context
    + one query tubelet); RoPE handles the rest since it computes relative
    positions from the index values we hand it, not from a fixed table walk.
  - out_embed_dim of the predictor defaults to encoder embed_dim (1024),
    confirmed via predictor_proj construction when teacher_embed_dim is
    unset -- so predicted tokens live in the same space as encoder output,
    and our latent_drift L2 comparison is dimensionally valid.
  - CRITICAL LIMITATION (reported to the user, repeated here so it isn't
    lost): nowhere in this repo is the base predictor called this way. Its
    only real use is masked completion within an already-observed clip
    (training masks span the whole temporal extent). The causal masks_x/
    masks_y construction below (context=past, target=future) is
    architecturally valid but UNPRECEDENTED -- an out-of-distribution query.
    Every number produced through this adapter should be reported as
    testing the frozen predictor under that OOD regime, not as a clean
    measurement of "how good is this model's forward dynamics."
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn

_VENDOR_ROOT = Path(__file__).resolve().parent.parent / "vendor" / "vjepa2"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from src.hub import backbones as _backbones  # noqa: E402
from src.masks.utils import apply_masks  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)

IMG_SIZE = 256
PATCH_SIZE = 16
TUBELET_SIZE = 2
NUM_FRAMES_NOMINAL = 64  # matches the predictor's constructed grid_depth=32
GRID_HW = IMG_SIZE // PATCH_SIZE  # 16
TOKENS_PER_TUBELET = GRID_HW * GRID_HW  # 256
GRID_DEPTH = NUM_FRAMES_NOMINAL // TUBELET_SIZE  # 32


def load_encoder_predictor(checkpoint_path: str, device: str = "cpu"):
    """
    Builds the real vjepa2_vit_large encoder+predictor and loads the
    checkpoint from a LOCAL file (the vendored hub code points
    VJEPA_BASE_URL at localhost in this checkout -- we bypass torch.hub
    entirely and replicate its (verified-by-reading) loading logic here).
    """
    from src.models import predictor as vit_predictor, vision_transformer as vit_encoder

    encoder = vit_encoder.__dict__["vit_large"](
        patch_size=PATCH_SIZE,
        img_size=(IMG_SIZE, IMG_SIZE),
        num_frames=NUM_FRAMES_NOMINAL,
        tubelet_size=TUBELET_SIZE,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
    )
    predictor = vit_predictor.__dict__["vit_predictor"](
        img_size=(IMG_SIZE, IMG_SIZE),
        patch_size=PATCH_SIZE,
        use_mask_tokens=True,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=384,
        out_embed_dim=None,
        num_frames=NUM_FRAMES_NOMINAL,
        tubelet_size=TUBELET_SIZE,
        depth=12,
        num_heads=12,
        num_mask_tokens=10,
        use_rope=True,
        uniform_power=False,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder_sd = _backbones._clean_backbone_key(state_dict["target_encoder"])
    msg_e = encoder.load_state_dict(encoder_sd, strict=False)
    predictor_sd = _backbones._clean_backbone_key(state_dict["predictor"])
    msg_p = predictor.load_state_dict(predictor_sd, strict=False)
    print(f"[vjepa_adapter] encoder load_state_dict: {msg_e}")
    print(f"[vjepa_adapter] predictor load_state_dict: {msg_p}")

    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in predictor.parameters():
        p.requires_grad_(False)
    return encoder, predictor


class VJEPA2Encoder(nn.Module):
    """Wraps the real encoder. Handles ImageNet normalization internally so
    the rest of the pipeline stays in raw [0,1] pixel space."""

    def __init__(self, encoder, device: str = "cpu"):
        super().__init__()
        self.encoder = encoder
        self.device = device

    @property
    def embed_dim(self) -> int:
        return self.encoder.embed_dim

    @property
    def tokens_per_tubelet(self) -> int:
        return TOKENS_PER_TUBELET

    @torch.no_grad()
    def encode(self, clip: Tensor) -> Tensor:
        """clip: [B, C, T, H, W] in [0,1], T even -> tokens [B, N, D]"""
        assert clip.shape[2] % TUBELET_SIZE == 0, "clip length must be a multiple of tubelet_size"
        clip = clip.to(self.device)
        normed = (clip - IMAGENET_MEAN.to(self.device)) / IMAGENET_STD.to(self.device)
        return self.encoder(normed)


class VJEPA2Predictor(nn.Module):
    """
    Implements the causal masks_x/masks_y query described in the module
    docstring. rollout(context_tokens, actions, k) predicts the tubelet
    containing frame t+k, where t is implied by context_tokens' length
    (context_tokens must cover an integer number of tubelets, temporal
    slots 0..num_context_tubelets-1).
    """

    def __init__(self, predictor, device: str = "cpu"):
        super().__init__()
        self.predictor = predictor
        self.device = device

    @torch.no_grad()
    def rollout(self, context_tokens: Tensor, actions: Optional[Tensor], k: int) -> Tensor:
        B, n_ctx, D = context_tokens.shape
        assert n_ctx % TOKENS_PER_TUBELET == 0, "context_tokens must cover whole tubelets"
        num_ctx_tubelets = n_ctx // TOKENS_PER_TUBELET
        t = num_ctx_tubelets * TUBELET_SIZE - 1  # last real frame index in context
        target_slot = (t + k) // TUBELET_SIZE
        assert target_slot < GRID_DEPTH, (
            f"target_slot={target_slot} exceeds predictor's nominal grid_depth={GRID_DEPTH}; "
            f"reduce k or t"
        )
        assert target_slot >= num_ctx_tubelets, (
            f"target_slot={target_slot} overlaps context (num_ctx_tubelets={num_ctx_tubelets}); "
            f"k={k} too small relative to tubelet_size at this t"
        )

        device = context_tokens.device
        ctx_idx = torch.arange(0, num_ctx_tubelets * TOKENS_PER_TUBELET, device=device)
        ctx_idx = ctx_idx.unsqueeze(0).repeat(B, 1)  # [B, n_ctx]

        tgt_idx = torch.arange(
            target_slot * TOKENS_PER_TUBELET, (target_slot + 1) * TOKENS_PER_TUBELET, device=device
        )
        tgt_idx = tgt_idx.unsqueeze(0).repeat(B, 1)  # [B, TOKENS_PER_TUBELET]

        pred = self.predictor(context_tokens, [ctx_idx], [tgt_idx])  # [B, TOKENS_PER_TUBELET, D]
        return pred
