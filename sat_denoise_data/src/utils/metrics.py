"""PSNR / MAE / SSIM. Inputs are float tensors in [-1, 1] or [0, 1].

Auto-detect range by max abs value. SSIM uses a small Gaussian kernel
implementation (no external dependency).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _to_unit(x: torch.Tensor) -> torch.Tensor:
    """Map either [-1, 1] or [0, 1] inputs to [0, 1]."""
    x = x.detach().float()
    if x.min() < -1e-3:  # treat as [-1, 1]
        x = (x + 1.0) * 0.5
    return x.clamp(0.0, 1.0)


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    a = _to_unit(pred)
    b = _to_unit(target)
    mse = F.mse_loss(a, b).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


@torch.no_grad()
def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    a = _to_unit(pred)
    b = _to_unit(target)
    return F.l1_loss(a, b).item()


def _gauss_kernel(window: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    coords = torch.arange(window, dtype=torch.float32) - (window - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    k2 = g[:, None] * g[None, :]
    k2 = k2.expand(channels, 1, window, window).contiguous()
    return k2


@torch.no_grad()
def ssim(pred: torch.Tensor, target: torch.Tensor, window: int = 11) -> float:
    a = _to_unit(pred)
    b = _to_unit(target)
    if a.dim() == 3:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
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
    return num.div(den.clamp_min(1e-8)).mean().item()


@torch.no_grad()
def all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "psnr": psnr(pred, target),
        "mae": mae(pred, target),
        "ssim": ssim(pred, target),
    }
