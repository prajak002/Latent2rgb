"""
A small, TRAINED StatePredictor for the DINOv2 second-family check
(dinov2_adapter.py) -- DINOv2 ships no predictor at all, so unlike
VJEPA2Predictor (frozen, pretrained, never fine-tuned by this package),
this one is fit here, on the same real clips, with plain MSE regression to
future tubelet tokens (see scripts/train_second_family.py).

Architecture, deliberately simple (this is a baseline predictor, not a
research contribution in itself -- see README "Second model family"):
predict the target tubelet as the last-seen context tubelet's tokens plus a
per-token residual delta, where the delta is produced by a small shared MLP
conditioned on (that token, a global mean-pooled context summary, a
sinusoidal embedding of horizon k). Single forward pass per call, satisfying
StatePredictor.rollout's "one query, not a loop" contract.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

TOKENS_PER_TUBELET = 256


def sinusoidal_k_embedding(k: Tensor, dim: int) -> Tensor:
    """k: [B] int/float -> [B, dim]"""
    device = k.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device).float() / half)
    args = k.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim] (dim even)


class DeltaMLPPredictor(nn.Module):
    def __init__(self, embed_dim: int = 384, k_embed_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.k_embed_dim = k_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + k_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def rollout(self, context_tokens: Tensor, actions, k) -> Tensor:
        """context_tokens: [B, N_ctx, D], N_ctx = num_context_tubelets * 256.
        k: int or [B] tensor. actions: ignored (not action-conditioned).
        returns: [B, 256, D] predicted target-tubelet tokens.
        """
        B, N_ctx, D = context_tokens.shape
        last_tubelet = context_tokens[:, -TOKENS_PER_TUBELET:, :]           # [B, 256, D]
        global_ctx = context_tokens.mean(dim=1, keepdim=True).expand(-1, TOKENS_PER_TUBELET, -1)  # [B, 256, D]

        if not torch.is_tensor(k):
            k = torch.full((B,), float(k), device=context_tokens.device)
        k_embed = sinusoidal_k_embedding(k, self.k_embed_dim)              # [B, k_embed_dim]
        k_embed = k_embed.unsqueeze(1).expand(-1, TOKENS_PER_TUBELET, -1)  # [B, 256, k_embed_dim]

        mlp_in = torch.cat([last_tubelet, global_ctx, k_embed], dim=-1)
        delta = self.mlp(mlp_in)
        return last_tubelet + delta
