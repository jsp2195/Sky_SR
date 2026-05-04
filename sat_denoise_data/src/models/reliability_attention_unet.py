"""Reliability-aware attention U-Net for deterministic RGB restoration.

The model predicts an RGB residual. Callers restore with:
    restored = degraded_rgb + residual

When ``predict_uncertainty`` is enabled, a one-channel positive uncertainty
map is returned for diagnostics or optional heteroscedastic training.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(c: int) -> nn.Module:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class ResidualBlock(nn.Module):
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


class ChannelSpatialAttention(nn.Module):
    """Compact CBAM-style attention for bottleneck features."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        x = x * torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(self.spatial(torch.cat([avg_map, max_map], dim=1)))


class AttentionGate(nn.Module):
    """Attention-gate skip features using the decoder gating signal."""

    def __init__(self, skip_channels: int, gate_channels: int):
        super().__init__()
        inter = max(8, min(skip_channels, gate_channels) // 2)
        self.skip_proj = nn.Conv2d(skip_channels, inter, 1, bias=False)
        self.gate_proj = nn.Conv2d(gate_channels, inter, 1, bias=False)
        self.psi = nn.Conv2d(inter, 1, 1)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attn = torch.sigmoid(self.psi(F.silu(self.skip_proj(skip) + self.gate_proj(gate))))
        return skip * attn


class ReliabilityAttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        base_channels: int = 48,
        channel_mults: Sequence[int] = (1, 2, 4, 4),
        predict_uncertainty: bool = False,
        bottleneck_attention: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.predict_uncertainty = bool(predict_uncertainty)
        chs = [base_channels * int(m) for m in channel_mults]
        if len(chs) != 4:
            raise ValueError("ReliabilityAttentionUNet expects four channel_mults levels")

        self.in_conv = nn.Conv2d(in_channels, chs[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList([
            ResidualBlock(chs[0], chs[0]),
            ResidualBlock(chs[0], chs[1]),
            ResidualBlock(chs[1], chs[2]),
            ResidualBlock(chs[2], chs[3]),
        ])
        self.bottleneck_attn = ChannelSpatialAttention(chs[3]) if bottleneck_attention else nn.Identity()

        self.up_convs = nn.ModuleList([
            nn.ConvTranspose2d(chs[3], chs[2], 4, stride=2, padding=1),
            nn.ConvTranspose2d(chs[2], chs[1], 4, stride=2, padding=1),
            nn.ConvTranspose2d(chs[1], chs[0], 4, stride=2, padding=1),
        ])
        self.skip_gates = nn.ModuleList([
            AttentionGate(chs[2], chs[2]),
            AttentionGate(chs[1], chs[1]),
            AttentionGate(chs[0], chs[0]),
        ])
        self.dec_blocks = nn.ModuleList([
            ResidualBlock(chs[2] + chs[2], chs[2]),
            ResidualBlock(chs[1] + chs[1], chs[1]),
            ResidualBlock(chs[0] + chs[0], chs[0]),
        ])

        self.out_norm = _norm(chs[0])
        self.residual_head = nn.Conv2d(chs[0], out_channels, 3, padding=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

        self.uncertainty_head = None
        if self.predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(chs[0], chs[0], 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(chs[0], 1, 3, padding=1),
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.in_conv(x)
        skips: list[torch.Tensor] = []
        h = self.enc_blocks[0](h); skips.append(h)
        h = F.avg_pool2d(h, 2)
        h = self.enc_blocks[1](h); skips.append(h)
        h = F.avg_pool2d(h, 2)
        h = self.enc_blocks[2](h); skips.append(h)
        h = F.avg_pool2d(h, 2)
        h = self.bottleneck_attn(self.enc_blocks[3](h))

        for up, gate, block, skip in zip(self.up_convs, self.skip_gates, self.dec_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError(
                    f"ReliabilityAttentionUNet skip mismatch: up={tuple(h.shape[-2:])} "
                    f"skip={tuple(skip.shape[-2:])}. Input H,W must be multiples of 8."
                )
            h = block(torch.cat([h, gate(skip, h)], dim=1))

        feat = F.silu(self.out_norm(h))
        out = {"residual": self.residual_head(feat)}
        if self.uncertainty_head is not None:
            out["uncertainty"] = F.softplus(self.uncertainty_head(feat)) + 1e-4
        return out
