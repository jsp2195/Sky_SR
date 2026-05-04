"""Conditional U-Net for DDPM epsilon prediction.

Inputs:
    x_t:  noised clean image, [B, 3, H, W]
    t:    timestep indices,   [B]
    cond: degraded image,     [B, 3, H, W]

The model concatenates x_t and cond along channels (6 in) and predicts
epsilon (3 out). Sinusoidal timestep embedding is projected per-block
via a small MLP and added to feature maps. No attention in v1.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional embedding for diffusion timesteps."""
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _norm(c: int) -> nn.Module:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class TimedConvBlock(nn.Module):
    """ConvBlock with FiLM-like additive injection of a time embedding."""

    def __init__(self, c_in: int, c_out: int, t_dim: int):
        super().__init__()
        self.n1 = _norm(c_in)
        self.c1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, c_out)
        self.n2 = _norm(c_out)
        self.c2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class DDPMUNet(nn.Module):
    """Conditional U-Net predicting epsilon from (x_t, t, cond)."""

    def __init__(
        self,
        in_channels: int = 3,
        cond_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 4),
        t_embed_dim: int = 256,
    ):
        super().__init__()
        chs = [base_channels * m for m in channel_mults]
        self.t_embed_dim = t_embed_dim

        # Time-embedding MLP (sinusoidal -> hidden -> hidden).
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, t_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(t_embed_dim * 4, t_embed_dim),
        )

        in_ch = in_channels + cond_channels  # 6 by default
        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        self.enc_blocks = nn.ModuleList([
            TimedConvBlock(chs[0], chs[0], t_embed_dim),
            TimedConvBlock(chs[0], chs[1], t_embed_dim),
            TimedConvBlock(chs[1], chs[2], t_embed_dim),
            TimedConvBlock(chs[2], chs[3], t_embed_dim),  # bottleneck
        ])

        self.up_convs = nn.ModuleList([
            nn.ConvTranspose2d(chs[3], chs[2], 4, stride=2, padding=1),
            nn.ConvTranspose2d(chs[2], chs[1], 4, stride=2, padding=1),
            nn.ConvTranspose2d(chs[1], chs[0], 4, stride=2, padding=1),
        ])
        self.dec_blocks = nn.ModuleList([
            TimedConvBlock(chs[2] + chs[2], chs[2], t_embed_dim),
            TimedConvBlock(chs[1] + chs[1], chs[1], t_embed_dim),
            TimedConvBlock(chs[0] + chs[0], chs[0], t_embed_dim),
        ])

        self.out_norm = _norm(chs[0])
        self.out_conv = nn.Conv2d(chs[0], out_channels, 3, padding=1)
        # zero-init head: predicts ~0 epsilon at start, stable training.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t, self.t_embed_dim)
        t_emb = self.t_mlp(t_emb)

        h = torch.cat([x_t, cond], dim=1)
        h = self.in_conv(h)

        skips: list[torch.Tensor] = []
        h = self.enc_blocks[0](h, t_emb); skips.append(h)
        h = F.avg_pool2d(h, 2)
        h = self.enc_blocks[1](h, t_emb); skips.append(h)
        h = F.avg_pool2d(h, 2)
        h = self.enc_blocks[2](h, t_emb); skips.append(h)
        h = F.avg_pool2d(h, 2)
        h = self.enc_blocks[3](h, t_emb)  # bottleneck

        for up, blk, sk in zip(self.up_convs, self.dec_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != sk.shape[-2:]:
                h = F.interpolate(h, size=sk.shape[-2:], mode="nearest")
            h = blk(torch.cat([h, sk], dim=1), t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))
