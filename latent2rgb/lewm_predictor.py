"""
The action-free LeWM-style predictor and its SIGReg anti-collapse
regularizer, per arXiv 2603.19312 Sec. 3.1 / Appendix A / Algorithm 3.

Predictor: a causal transformer (paper: ViT-S backbone with learned
positional embeddings and causal masking over the observation history; here
adapted to operate over TUBELET embeddings rather than raw ViT patches,
since it consumes lewm_adapter.LeWMEncoder's output). No action
conditioning (see lewm_adapter.py's honesty note) -- the paper's AdaLN
action pathway is simply absent.

SIGReg (Appendix A): projects a batch of embeddings onto M random unit
directions and applies the univariate Epps-Pulley normality test to each
1D projection; by the Cramer-Wold theorem, matching every 1D marginal
implies matching the full joint distribution to an isotropic Gaussian.
Implemented here exactly per Eq. (6)/(EP)/(SIGReg) in the paper:
    h^(m)   = Z u^(m),  u^(m) ~ Uniform(S^{D-1})
    T^(m)   = integral w(t) |phi_N(t; h^(m)) - phi_0(t)|^2 dt
    phi_N(t; h) = (1/N) sum_n exp(i t h_n)      (empirical char. function)
    phi_0(t)    = exp(-t^2/2)                    (standard Gaussian's)
    w(t)        = exp(-t^2 / (2*lambda^2)),  lambda=1
The paper integrates via quadrature with nodes in [0.2, 4]; the integrand
is even in t (phi_0 is real, |.|^2 of an even/odd-real/imag pair is even),
so integrating [0.2,4] and doubling approximates the full (-inf,inf)
integral, skipping t=0 where phi_N(0)=phi_0(0)=1 contributes nothing.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

TOKENS_PER_TUBELET = 1


def sigreg_term(Z: Tensor, num_directions: int = 256, num_quad_nodes: int = 17, lam: float = 1.0) -> Tensor:
    """Z: [N, D] (N samples -- here, one time-step's batch of embeddings).
    Returns a scalar: the SIGReg regularization loss for this batch."""
    N, D = Z.shape
    device = Z.device
    U = torch.randn(D, num_directions, device=device)
    U = U / U.norm(dim=0, keepdim=True)          # [D, M], unit directions

    H = Z @ U                                       # [N, M] -- h^(m) for each m
    t = torch.linspace(0.2, 4.0, num_quad_nodes, device=device)  # [T]

    # empirical characteristic function phi_N(t; h) at each quadrature node,
    # for every direction m at once: angle[t, n, m] = t * h_{n,m}
    angle = t.view(-1, 1, 1) * H.unsqueeze(0)        # [T, N, M]
    ecf_re = angle.cos().mean(dim=1)                 # [T, M]
    ecf_im = angle.sin().mean(dim=1)                 # [T, M]

    phi0 = torch.exp(-0.5 * t**2)                    # [T], standard Gaussian char. fn (real)
    sq_dev = (ecf_re - phi0.view(-1, 1)) ** 2 + ecf_im**2   # [T, M]

    w = torch.exp(-0.5 * (t / lam) ** 2)              # [T]
    integrand = w.view(-1, 1) * sq_dev                # [T, M]
    # trapezoid rule over t in [0.2,4], doubled for the symmetric [-4,-0.2] half
    per_direction = 2.0 * torch.trapezoid(integrand, t, dim=0)  # [M]
    return per_direction.mean()


class CausalPredictor(nn.Module):
    """Causal transformer over a sequence of tubelet embeddings. No action
    conditioning. forward(seq) -> next-step predictions at every position
    (teacher-forcing training, per Algorithm 3); rollout(...) chains this
    internally for multi-step inference, closed-loop (its own predictions
    feed back in), matching the paper's planning-time behavior (Fig. 4).
    """

    def __init__(self, embed_dim: int = 192, depth: int = 6, num_heads: int = 16,
                 dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.projector = None  # set externally to share the encoder's Projector class instance semantics

    def forward(self, seq: Tensor) -> Tensor:
        """seq: [B, T, D] tubelet embeddings -> [B, T, D] next-step predictions (position i predicts step i+1)."""
        B, T, D = seq.shape
        x = seq + self.pos_embed[:, :T]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=seq.device)
        out = self.transformer(x, mask=causal_mask, is_causal=True)
        if self.projector is not None:
            out = self.projector(out)
        return out

    def rollout(self, context_tokens: Tensor, actions, k: int) -> Tensor:
        """context_tokens: [B, N_ctx, D], N_ctx = num_context_tubelets (tokens_per_tubelet=1).
        k: raw frames forward (interfaces.py convention) -- chained internally
        as k // TUBELET_SIZE tubelet-steps, closed-loop (own predictions feed
        back in), matching StatePredictor's "single query, may loop
        internally" contract. actions: ignored (action-free variant).
        returns: [B, 1, D] the final predicted tubelet's embedding.
        """
        from .lewm_adapter import TUBELET_SIZE
        num_steps = k // TUBELET_SIZE
        seq = context_tokens
        for _ in range(num_steps):
            pred_seq = self.forward(seq)
            next_token = pred_seq[:, -1:, :]   # the prediction FOR the position after the last context token
            seq = torch.cat([seq, next_token], dim=1)
        return seq[:, -1:, :]
