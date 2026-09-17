"""
Third model family: an action-free LeWM-style JEPA (Maes, Le Lidec, Scieur,
LeCun, Balestriero, "LeWorldModel: Stable End-to-End Joint-Embedding
Predictive Architecture from Pixels", arXiv 2603.19312, June 2026).

This is deliberately NOT the DINOv2 stand-in in dinov2_adapter.py: that
pairs a FROZEN, independently-pretrained image encoder with a small trained
predictor. LeWM's whole point is training the encoder and predictor
JOINTLY, end-to-end, from raw pixels, with no pretrained representation at
all -- a genuinely different paradigm, and the "real second family" this
project's Limitations section called for. Config matches the paper's
released numbers exactly: ViT-Tiny encoder (patch14, depth=12, heads=3,
embed_dim=192, ~5M params, ImageNet-scale init but NOT pretrained -- weights
start random and are trained here), a 1-layer MLP+BatchNorm projector (the
paper uses this because the final ViT LayerNorm would fight the SIGReg
anti-collapse objective).

Two honesty notes, see README Limitations:
  1. LeWM is action-conditioned (actions via AdaLN in the predictor); SSv2
     and Kinetics-mini carry no action labels, so this omits the AdaLN
     action pathway entirely. That makes this an "action-free LeWM-style
     JEPA", not a reimplementation -- call it that, not "LeWM".
  2. LeWM encodes a whole frame into ONE embedding vector (the [CLS] token,
     projected), not a spatial patch-token grid like V-JEPA2/DINOv2's 256
     tokens/tubelet. tokens_per_tubelet=1 here is a real architectural
     difference this family introduces, not a bug to reconcile with the
     other two adapters.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, nn

TUBELET_SIZE = 2
EMBED_DIM = 192
INPUT_SIZE = 224
PATCH_SIZE = 14
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Projector(nn.Module):
    """1-layer MLP + BatchNorm, per the paper: maps the ViT's final-layer
    CLS embedding (which has already been through LayerNorm, hostile to
    SIGReg) into a fresh representation space."""

    def __init__(self, dim: int = EMBED_DIM):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        x = self.fc(x.reshape(-1, shape[-1]))
        x = self.bn(x)
        return x.reshape(shape)


def build_vit_tiny_from_scratch() -> VisionTransformer:
    """No pretrained=True anywhere -- weights are randomly initialized and
    trained jointly with the predictor, matching LeWM's headline claim."""
    return VisionTransformer(
        img_size=INPUT_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
        depth=12, num_heads=3, num_classes=0, global_pool="token",
    )


class LeWMEncoder(nn.Module):
    """Encoder protocol: clip [B, C, T=2, H, W] in [0,1] -> tokens [B, num_tubelets, 192].
    tokens_per_tubelet=1 -- see module docstring. Trainable (not frozen):
    the training script back-props through this, unlike the other two
    adapters' frozen encoders.
    """

    def __init__(self, vit: VisionTransformer, projector: Projector, device: str = "cpu"):
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.device = device

    @property
    def embed_dim(self) -> int:
        return EMBED_DIM

    @property
    def tokens_per_tubelet(self) -> int:
        return 1

    def encode(self, clip: Tensor) -> Tensor:
        assert clip.shape[2] % TUBELET_SIZE == 0, "clip length must be a multiple of tubelet_size"
        clip = clip.to(self.device)
        B, C, T, H, W = clip.shape
        num_tubelets = T // TUBELET_SIZE

        frames = clip.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        if (H, W) != (INPUT_SIZE, INPUT_SIZE):
            frames = F.interpolate(frames, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
        normed = (frames - IMAGENET_MEAN.to(self.device)) / IMAGENET_STD.to(self.device)

        feats = self.vit.forward_features(normed)   # [B*T, 1+num_patches, 192]
        cls = feats[:, 0, :]                          # [B*T, 192] -- the [CLS] embedding
        cls = self.projector(cls)
        cls = cls.reshape(B, T, self.embed_dim)

        out = []
        for slot in range(num_tubelets):
            pair = cls[:, slot * TUBELET_SIZE:(slot + 1) * TUBELET_SIZE]  # [B,2,192]
            out.append(pair.mean(dim=1, keepdim=True))  # mean-pool the tubelet's 2 frames -> [B,1,192]
        return torch.cat(out, dim=1)  # [B, num_tubelets, 192]
