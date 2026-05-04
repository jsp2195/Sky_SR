"""Residual target and frequency-band utilities for residual DDPMs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _gaussian_kernel_2d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(round(float(sigma) * 3.0)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-(coords ** 2) / (2.0 * float(sigma) ** 2))
    g = g / g.sum().clamp_min(1e-12)
    return (g[:, None] * g[None, :]).contiguous()


def gaussian_blur_tensor(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Depthwise Gaussian blur for BCHW tensors, preserving shape."""
    if x.ndim != 4:
        raise ValueError(f"expected [B,C,H,W], got shape {tuple(x.shape)}")
    sigma = float(sigma)
    if sigma <= 0.0:
        return x
    kernel = _gaussian_kernel_2d(sigma, x.device, x.dtype)
    k = int(kernel.shape[0])
    c = int(x.shape[1])
    weight = kernel.view(1, 1, k, k).expand(c, 1, k, k).contiguous()
    pad = k // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x_pad, weight, groups=c)


def lowpass_tensor(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Low-frequency component of a BCHW tensor."""
    return gaussian_blur_tensor(x, sigma)


def highpass_tensor(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """High-frequency/detail component of a BCHW tensor."""
    if float(sigma) <= 0.0:
        return x
    return x - lowpass_tensor(x, sigma)


def residual_target_config(cfg: dict[str, Any]) -> dict[str, float | str]:
    return {
        "residual_target_type": str(cfg.get("residual_target_type", "pixel") or "pixel"),
        "detail_sigma": float(cfg.get("detail_sigma", 1.0)),
        "lowpass_sigma": float(cfg.get("lowpass_sigma", 4.0)),
        "lowpass_weight": float(cfg.get("lowpass_weight", 1.0)),
        "highpass_weight": float(cfg.get("highpass_weight", 1.0)),
    }


def build_residual_target(clean: torch.Tensor, restored_base: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    """Build the unscaled residual target selected by cfg.residual_target_type."""
    rc = residual_target_config(cfg)
    kind = str(rc["residual_target_type"])
    detail_sigma = float(rc["detail_sigma"])
    lowpass_sigma = float(rc["lowpass_sigma"])
    lowpass_weight = float(rc["lowpass_weight"])
    highpass_weight = float(rc["highpass_weight"])

    if kind == "pixel":
        return clean - restored_base
    if kind == "lowpass":
        return lowpass_tensor(clean, lowpass_sigma) - lowpass_tensor(restored_base, lowpass_sigma)
    if kind == "highpass":
        return highpass_tensor(clean, detail_sigma) - highpass_tensor(restored_base, detail_sigma)
    if kind == "multiband":
        low = lowpass_tensor(clean, lowpass_sigma) - lowpass_tensor(restored_base, lowpass_sigma)
        high = highpass_tensor(clean, detail_sigma) - highpass_tensor(restored_base, detail_sigma)
        return lowpass_weight * low + highpass_weight * high
    raise ValueError(
        f"unknown residual_target_type {kind!r}; expected pixel, lowpass, highpass, or multiband"
    )


def gradient_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dx_pred = pred[..., :, 1:] - pred[..., :, :-1]
    dx_tgt = target[..., :, 1:] - target[..., :, :-1]
    dy_pred = pred[..., 1:, :] - pred[..., :-1, :]
    dy_tgt = target[..., 1:, :] - target[..., :-1, :]
    return 0.5 * ((dx_pred - dx_tgt).abs().mean() + (dy_pred - dy_tgt).abs().mean())


def detail_metric_values(pred: torch.Tensor, target: torch.Tensor, cfg: dict[str, Any]) -> dict[str, float]:
    rc = residual_target_config(cfg)
    hp_pred = highpass_tensor(pred, float(rc["detail_sigma"]))
    hp_target = highpass_tensor(target, float(rc["detail_sigma"]))
    lp_pred = lowpass_tensor(pred, float(rc["lowpass_sigma"]))
    lp_target = lowpass_tensor(target, float(rc["lowpass_sigma"]))
    return {
        "highpass_mae": float((hp_pred - hp_target).abs().mean().item()),
        "lowpass_mae": float((lp_pred - lp_target).abs().mean().item()),
        "gradient_mae": float(gradient_mae(pred, target).item()),
    }
