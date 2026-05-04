"""Restoration loss bundle.

Combines:
    - L1(pred, target)
    - 1 - SSIM(pred, target)            (structural similarity)
    - L1 on Sobel gradient magnitude   (edge preservation)
    - optional total-variation penalty on the residual

All inputs are float tensors in [-1, 1] of shape (B, 3, H, W).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gauss_kernel(window: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    coords = torch.arange(window, dtype=torch.float32) - (window - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    k2 = g[:, None] * g[None, :]
    return k2.expand(channels, 1, window, window).contiguous()


def ssim_value(a: torch.Tensor, b: torch.Tensor, window: int = 11) -> torch.Tensor:
    c = a.shape[1]
    kernel = _gauss_kernel(window, 1.5, c).to(a.device, a.dtype)
    pad = window // 2
    mu_a = F.conv2d(a, kernel, padding=pad, groups=c)
    mu_b = F.conv2d(b, kernel, padding=pad, groups=c)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sig_a = F.conv2d(a * a, kernel, padding=pad, groups=c) - mu_a2
    sig_b = F.conv2d(b * b, kernel, padding=pad, groups=c) - mu_b2
    sig_ab = F.conv2d(a * b, kernel, padding=pad, groups=c) - mu_ab
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_ab + C1) * (2 * sig_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sig_a + sig_b + C2)
    return num.div(den.clamp_min(1e-8)).mean()


_SOBEL_X = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
_SOBEL_Y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)


def edge_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = a.shape[1]
    kx = _SOBEL_X.to(a.device, a.dtype).expand(c, 1, 3, 3).contiguous()
    ky = _SOBEL_Y.to(a.device, a.dtype).expand(c, 1, 3, 3).contiguous()
    gxa = F.conv2d(a, kx, padding=1, groups=c)
    gya = F.conv2d(a, ky, padding=1, groups=c)
    gxb = F.conv2d(b, kx, padding=1, groups=c)
    gyb = F.conv2d(b, ky, padding=1, groups=c)
    mag_a = torch.sqrt(gxa * gxa + gya * gya + 1e-8)
    mag_b = torch.sqrt(gxb * gxb + gyb * gyb + 1e-8)
    return F.l1_loss(mag_a, mag_b)


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


class RestorationLoss(nn.Module):
    def __init__(
        self,
        w_l1: float = 1.0,
        w_ssim: float = 0.2,
        w_edge: float = 0.1,
        w_tv: float = 0.0,
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_edge = w_edge
        self.w_tv = w_tv

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_l1 = F.l1_loss(pred, target)
        loss_ssim = 1.0 - ssim_value(pred, target)
        loss_edge = edge_loss(pred, target)
        loss_tv = tv_loss(residual) if (self.w_tv > 0 and residual is not None) else pred.new_zeros(())
        total = (
            self.w_l1 * loss_l1
            + self.w_ssim * loss_ssim
            + self.w_edge * loss_edge
            + self.w_tv * loss_tv
        )
        parts = {
            "l1": float(loss_l1.detach()),
            "ssim_loss": float(loss_ssim.detach()),
            "edge": float(loss_edge.detach()),
            "tv": float(loss_tv.detach()),
            "total": float(total.detach()),
        }
        return total, parts
