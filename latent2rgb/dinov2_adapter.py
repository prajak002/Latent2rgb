"""
Second model family for Horizon Ladder's generalization check: an
independently-pretrained frozen image encoder (DINOv2 ViT-S/14, no video
pretraining, no joint spatiotemporal tokenization) paired with a small
predictor TRAINED HERE (see delta_predictor.py) -- unlike V-JEPA2, DINOv2
ships no video predictor at all, so there is nothing to keep frozen there.

This is a genuinely different paradigm from V-JEPA2's adapter, not a
relabeled copy: per-frame 2D patch tokens (no Conv3d tubelet fusion), mean-
pooled across a tubelet's 2 frames to land on the same [B, 256, D] tubelet-
token convention the rest of the pipeline (data.py, decoder.py, metrics.py)
already assumes -- so RolloutBuilder and the EBID instruments run against
this pair completely unmodified.

Honesty note (see README "Protocol" / "Second model family"): this is a
disk/time-light stand-in for "a second independently pretrained SOTA video
world model," not that model. DINOv2 was never pretrained to model temporal
dynamics; the temporal fusion (mean pool) and the predictor (delta_predictor.py)
are both built here, not pretrained. Treat results from this pair as "does
the measurement gap reproduce on a differently-architected latent-only
predictor we trained," not as evidence about DINOv2 or any other shipped
video world model.
"""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

TUBELET_SIZE = 2
GRID_HW = 16                     # 224 / patch_size(14) = 16
TOKENS_PER_TUBELET = GRID_HW * GRID_HW  # 256, matches vjepa_adapter's convention
INPUT_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_dinov2(device: str = "cpu"):
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=INPUT_SIZE, num_classes=0)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


class DINOv2Encoder(nn.Module):
    """Encoder protocol: clip [B, C, T=2, H, W] in [0,1] -> tokens [B, 256, 384].

    Each of the tubelet's 2 frames is patch-encoded independently (DINOv2 has
    no temporal mixing), then mean-pooled across time -- the only place any
    of this pipeline's temporal information for this encoder is fused,
    unlike V-JEPA2's Conv3d PatchEmbed3D which fuses temporally inside the
    encoder itself.
    """

    def __init__(self, dinov2, device: str = "cpu"):
        super().__init__()
        self.dinov2 = dinov2
        self.device = device

    @property
    def embed_dim(self) -> int:
        return 384

    @property
    def tokens_per_tubelet(self) -> int:
        return TOKENS_PER_TUBELET

    @torch.no_grad()
    def encode(self, clip: Tensor) -> Tensor:
        assert clip.shape[2] % TUBELET_SIZE == 0, "clip length must be a multiple of tubelet_size"
        clip = clip.to(self.device)
        B, C, T, H, W = clip.shape
        num_tubelets = T // TUBELET_SIZE

        frames = clip.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        if (H, W) != (INPUT_SIZE, INPUT_SIZE):
            frames = F.interpolate(frames, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
        normed = (frames - IMAGENET_MEAN.to(self.device)) / IMAGENET_STD.to(self.device)

        feats = self.dinov2.forward_features(normed)  # [B*T, 257, 384]
        patch_tokens = feats[:, 1:, :]                 # drop CLS -> [B*T, 256, 384]
        patch_tokens = patch_tokens.reshape(B, T, TOKENS_PER_TUBELET, self.embed_dim)

        out = []
        for slot in range(num_tubelets):
            pair = patch_tokens[:, slot * TUBELET_SIZE:(slot + 1) * TUBELET_SIZE]  # [B,2,256,384]
            out.append(pair.mean(dim=1))  # mean-pool across the tubelet's 2 frames -> [B,256,384]
        return torch.cat(out, dim=1)  # [B, num_tubelets*256, 384]
