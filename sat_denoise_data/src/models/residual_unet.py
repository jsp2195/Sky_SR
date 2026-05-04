"""Small residual U-Net for deterministic denoising / restoration.

Forward signature:
    clean_pred = model(degraded) = degraded + residual

CPU-friendly: 3 down/up stages, GroupNorm + SiLU.
With base_channels=32 and channel_mults=(1, 2, 4, 4), encoder produces
feature maps at H, H/2, H/4, H/8; decoder mirrors back to H.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(c: int) -> nn.Module:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.n1 = _norm(c_in)
        self.c1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.n2 = _norm(c_out)
        self.c2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class ResidualUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4, 4),
    ):
        super().__init__()
        chs = [base_channels * m for m in channel_mults]  # 4 levels: H, H/2, H/4, H/8

        # Encoder: produce a feature map at each level.
        self.in_conv = nn.Conv2d(in_channels, chs[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList([
            ConvBlock(chs[0], chs[0]),     # level 0 (H)
            ConvBlock(chs[0], chs[1]),     # level 1 (H/2)
            ConvBlock(chs[1], chs[2]),     # level 2 (H/4)
            ConvBlock(chs[2], chs[3]),     # level 3 (H/8) - bottleneck
        ])

        # Decoder: mirror back. At each level, upsample then ConvBlock(cat).
        self.up_convs = nn.ModuleList([
            nn.ConvTranspose2d(chs[3], chs[2], 4, stride=2, padding=1),
            nn.ConvTranspose2d(chs[2], chs[1], 4, stride=2, padding=1),
            nn.ConvTranspose2d(chs[1], chs[0], 4, stride=2, padding=1),
        ])
        self.dec_blocks = nn.ModuleList([
            ConvBlock(chs[2] + chs[2], chs[2]),
            ConvBlock(chs[1] + chs[1], chs[1]),
            ConvBlock(chs[0] + chs[0], chs[0]),
        ])

        self.out_norm = _norm(chs[0])
        self.out_conv = nn.Conv2d(chs[0], out_channels, 3, padding=1)
        # zero-init residual head for near-identity start.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(x)
        skips: list[torch.Tensor] = []
        h = self.enc_blocks[0](h);  skips.append(h)         # H
        h = F.avg_pool2d(h, 2)
        h = self.enc_blocks[1](h);  skips.append(h)         # H/2
        h = F.avg_pool2d(h, 2)
        h = self.enc_blocks[2](h);  skips.append(h)         # H/4
        h = F.avg_pool2d(h, 2)
        h = self.enc_blocks[3](h)                           # H/8 bottleneck

        for up, blk, sk in zip(self.up_convs, self.dec_blocks, reversed(skips)):
            h = up(h)
            # Skip-connection contract: encoder/decoder must match exactly.
            # No silent interpolation fallback. Input H,W must be divisible by 8.
            if h.shape[-2:] != sk.shape[-2:]:
                raise RuntimeError(
                    f"ResidualUNet skip mismatch: up={tuple(h.shape[-2:])} "
                    f"sk={tuple(sk.shape[-2:])}. Input H,W must be multiples of 8."
                )
            h = blk(torch.cat([h, sk], dim=1))

        h = self.out_conv(F.silu(self.out_norm(h)))
        return x + h
