"""
Minimal-capacity decoder: one tubelet's worth of tokens -> a 2-frame RGB
tubelet. This is the measurement instrument, not a deliverable. Deliberately
low capacity (a single per-token linear "unpatchify", the literal inverse of
V-JEPA2's PatchEmbed3D Conv3d) and no perceptual/adversarial/diffusion
losses -- see Block 0.

Token layout matches PatchEmbed3D exactly (see vjepa_adapter.py docstring):
token index = h*grid_w + w for a single time slot, grid_h=grid_w=16 for
256px/patch16. We invert that with einops rather than a transposed conv,
since a single Linear + fold is the minimal decodable instrument.
"""

from __future__ import annotations

from einops import rearrange
from torch import Tensor, nn


class MinimalDecoder(nn.Module):
    def __init__(self, embed_dim: int, grid_size: int = 16, patch_size: int = 16, tubelet_size: int = 2, out_channels: int = 3):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.out_channels = out_channels
        out_dim = out_channels * tubelet_size * patch_size * patch_size
        self.proj = nn.Linear(embed_dim, out_dim)
        self.act = nn.Sigmoid()

    def decode(self, latents: Tensor) -> Tensor:
        """latents: [B, N, D], N=grid_size*grid_size -> [B, C, T, H, W]"""
        x = self.proj(latents)
        x = self.act(x)
        frames = rearrange(
            x,
            "b (h w) (c t ph pw) -> b c t (h ph) (w pw)",
            h=self.grid_size,
            w=self.grid_size,
            c=self.out_channels,
            t=self.tubelet_size,
            ph=self.patch_size,
            pw=self.patch_size,
        )
        return frames

    def forward(self, latents: Tensor) -> Tensor:
        return self.decode(latents)
