"""Train the conditional DDPM restoration model.

Two modes (config: ``diffusion_target``):

* ``clean`` (default, original behaviour): the diffusion target is the clean
  image, conditioning is the degraded image.
      x0       = clean
      cond     = degraded                  # 3 channels
      pred     = model(x_t, t, cond)
      loss     = MSE(pred, eps)
* ``residual``: the diffusion target is a *scaled* residual that a frozen,
  already trained residual U-Net leaves on the table. Conditioning is the
  concatenation of the degraded image and the U-Net base prediction.
      restored_base    = unet(degraded)    # frozen, no_grad
      residual         = clean - restored_base
      x0               = residual / residual_scale
      cond             = concat(degraded, restored_base)   # 6 channels
      pred             = model(x_t, t, cond)
      loss             = MSE(pred, eps)
  Sampling reconstructs the residual via residual_scale, optionally clips
  and blends conservatively:
      sampled_residual = clamp(scale * sampled_x0, -residual_clip, residual_clip)
      final            = clamp(restored_base + residual_blend * sampled_residual, -1, 1)

  ``residual_scale`` may be a positive float, or ``"auto"``: in auto mode the
  scale is estimated from a small subset of training samples at startup
  (robust standard deviation of the unscaled residual, clamped to a safe
  range) and the resolved value is stored in the saved config.

Saves ckpt_last.pt, ckpt_best.pt (by val MSE), train_log.jsonl, sample grids.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.patch_degradation_dataset import (
    FLAT_STRENGTH_KEYS,
    PatchDegradationDataset,
    patch_degradation_collate,
    resolve_profile,
)
from src.models.ddpm_unet import DDPMUNet
from src.models.diffusion import GaussianDiffusion
from src.models.residual_unet import ResidualUNet
from src.utils.checkpoint import is_better, load_checkpoint, save_checkpoint
from src.utils.diagnostic_grids import (
    save_labeled_gated_residual_refinement_grid,
    save_labeled_masked_completion_grid,
    save_labeled_residual_refinement_grid,
    save_labeled_restoration_grid,
)
from src.utils.metrics import all_metrics
from src.utils.residual_gating import (
    build_completion_mask,
    build_residual_gate,
    compose_gated_residual,
    compose_masked_completion,
    filter_batch_by_indices,
    localized_mask_keep,
    mask_area_filter_config,
)
from src.utils.residual_targets import (
    build_residual_target,
    detail_metric_values,
    residual_target_config,
)
from src.utils.seed import set_seed
from src.utils.training_plots import write_all_plots
from collections import Counter, defaultdict


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--patch_dir", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--val_fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sampling_steps", type=int, default=None)
    ap.add_argument(
        "--diffusion_target",
        choices=["clean", "residual", "masked_completion"],
        default=None,
        help="Diffusion target: clean, residual, or masked_completion.",
    )
    ap.add_argument(
        "--base_ckpt",
        default=None,
        help="Path to a trained residual-U-Net checkpoint. Required when "
             "diffusion_target=residual.",
    )
    ap.add_argument(
        "--residual_scale",
        default=None,
        help="Residual normalization scale: a positive float or 'auto'. "
             "Residual mode only.",
    )
    ap.add_argument(
        "--residual_clip",
        type=float,
        default=None,
        help="Hard clip on the sampled residual at inference (residual mode).",
    )
    ap.add_argument(
        "--residual_blend",
        type=float,
        default=None,
        help="Default blend factor for diagnostic sample grids (residual mode).",
    )
    ap.add_argument(
        "--masked_completion_target",
        choices=["residual", "clean"],
        default=None,
        help="Masked-completion x0 target: residual (default) or clean.",
    )
    return ap.parse_args()


def load_cfg(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    for k in [
        "manifest", "patch_dir", "output_dir",
        "batch_size", "epochs", "lr",
        "num_workers", "max_samples", "val_fraction", "seed",
        "sampling_steps",
        "diffusion_target", "base_ckpt",
        "residual_scale", "residual_clip", "residual_blend",
        "masked_completion_target",
    ]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    cfg.setdefault("diffusion_target", "clean")
    if cfg["diffusion_target"] not in ("clean", "residual", "masked_completion"):
        raise ValueError(
            "diffusion_target must be 'clean', 'residual', or 'masked_completion', "
            f"got {cfg['diffusion_target']!r}"
        )
    if cfg["diffusion_target"] in ("residual", "masked_completion") and not cfg.get("base_ckpt"):
        raise ValueError(f"diffusion_target={cfg['diffusion_target']!r} requires --base_ckpt or cfg.base_ckpt")
    # Conservative residual defaults. Apply only in residual mode but always
    # set the keys so checkpoints record them; clean mode ignores them.
    cfg.setdefault("residual_scale", "auto" if cfg["diffusion_target"] == "residual" else 1.0)
    cfg.setdefault("residual_clip", 0.10)
    cfg.setdefault("residual_blend", 0.25)
    cfg.setdefault("eval_blends", [0.0, 0.1, 0.25, 0.5, 1.0])
    cfg.setdefault("save_step_train_batch_samples", True)
    cfg.setdefault("save_step_fixed_val_samples", False)
    cfg.setdefault("residual_gate_mode", "none")
    cfg.setdefault("residual_gate_hard_modes", [
        "mask_dropout",
        "mixed_structured",
        "lowfreq_atmospheric_bias",
        "blur_downsample_upsample",
    ])
    cfg.setdefault("residual_gate_blur", 0)
    cfg.setdefault("residual_gate_dilate", 0)
    cfg.setdefault("residual_gate_min_value", 0.0)
    cfg.setdefault("residual_gate_default_for_hard_modes", 1.0)
    cfg.setdefault("residual_target_type", "pixel")
    cfg.setdefault("detail_sigma", 1.0)
    cfg.setdefault("lowpass_sigma", 4.0)
    cfg.setdefault("lowpass_weight", 1.0)
    cfg.setdefault("highpass_weight", 1.0)
    cfg.setdefault("ddpm_condition_use_mask", False)
    cfg.setdefault("ddpm_condition_use_reliability", False)
    cfg.setdefault("masked_completion_target", "residual")
    if cfg["masked_completion_target"] not in ("residual", "clean"):
        raise ValueError("masked_completion_target must be 'residual' or 'clean'")
    cfg.setdefault("masked_loss_min_weight", 0.0)
    cfg.setdefault("masked_context_consistency_weight", 0.0)
    cfg.setdefault("mask_area_filter_enabled", False)
    cfg.setdefault("mask_area_min", 0.0)
    cfg.setdefault("mask_area_max", 1.0)
    return cfg


def _ddpm_cond_channels(cfg: dict[str, Any]) -> int:
    if str(cfg.get("diffusion_target", "clean")) == "clean":
        return 3
    extra = int(bool(cfg.get("ddpm_condition_use_mask", False)))
    extra += int(bool(cfg.get("ddpm_condition_use_reliability", False)))
    return 6 + extra


def _conditioning_extras(
    batch: dict[str, Any] | None,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    b: int,
    h: int,
    w: int,
) -> list[torch.Tensor]:
    extras: list[torch.Tensor] = []
    specs = [
        ("ddpm_condition_use_mask", "degradation_mask", 0.0),
        ("ddpm_condition_use_reliability", "reliability_map", 1.0),
    ]
    for flag, key, default in specs:
        if not bool(cfg.get(flag, False)):
            continue
        val = None if batch is None else batch.get(key)
        if val is None:
            x = torch.full((b, 1, h, w), float(default), device=device, dtype=torch.float32)
        else:
            x = val.to(device=device, dtype=torch.float32)
            if x.ndim == 3:
                x = x.unsqueeze(1)
            if x.shape[1] != 1:
                x = x[:, :1]
            if x.shape[-2:] != (h, w):
                x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        extras.append(x.clamp(0.0, 1.0))
    return extras


def _build_degradation_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Pull degradation_profile, structured strengths, and any flat-key
    overrides out of a training config and bundle them as kwargs for
    ``PatchDegradationDataset``."""
    flat = {k: cfg[k] for k in FLAT_STRENGTH_KEYS if k in cfg}
    out: dict[str, Any] = {}
    if cfg.get("degradation_profile") is not None:
        out["degradation_profile"] = str(cfg["degradation_profile"])
    if cfg.get("degradation_strengths") is not None:
        out["strengths"] = cfg["degradation_strengths"]
    if flat:
        out["flat_overrides"] = flat
    return out


def _filtered_degradation_probs(
    cfg: dict[str, Any],
    allowed: list[str] | tuple[str, ...] | None,
) -> dict[str, float] | None:
    if not allowed:
        return None
    if cfg.get("degradation_probs") is not None:
        base_probs = {str(k): float(v) for k, v in cfg["degradation_probs"].items()}
    else:
        base_probs, _ = resolve_profile(str(cfg.get("degradation_profile") or "balanced"))
    if "mask_dropout_probability" in cfg:
        base_probs["mask_dropout"] = float(cfg["mask_dropout_probability"])
    allowed_set = {str(k) for k in allowed}
    filtered = {k: float(v) for k, v in base_probs.items() if k in allowed_set and float(v) > 0.0}
    missing = sorted(allowed_set - set(filtered.keys()))
    if not filtered:
        raise ValueError(f"ddpm degradation filter has no positive-probability modes: {sorted(allowed_set)}")
    if missing:
        print(f"[train-ddpm] degradation filter ignored zero/missing modes: {missing}")
    return filtered


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction)))
    return idx[n_val:].tolist(), idx[:n_val].tolist()


def to01(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1.0) * 0.5).clamp(0.0, 1.0)


def sample_labels(indices: list[int], kinds: list[str], paths: list[str]) -> list[str]:
    labels = []
    for idx, kind, path in zip(indices, kinds, paths):
        labels.append(f"idx {idx} | {kind} | {os.path.basename(path)}")
    return labels


def batch_sample_labels(kinds: list[str], paths: list[str]) -> list[str]:
    labels = []
    for i, (kind, path) in enumerate(zip(kinds, paths)):
        labels.append(f"batch {i} | {kind} | {os.path.basename(path)}")
    return labels


def load_frozen_base_unet(base_ckpt: str, device: torch.device) -> ResidualUNet:
    """Load a residual U-Net from a checkpoint and put it in frozen eval mode."""
    state = load_checkpoint(base_ckpt, map_location=str(device))
    base_cfg = state.get("config", {}) or {}
    base = ResidualUNet(
        base_channels=int(base_cfg.get("base_channels", 32)),
        channel_mults=tuple(base_cfg.get("channel_mults", [1, 2, 4, 4])),
    ).to(device)
    base.load_state_dict(state["model"])
    base.eval()
    for p in base.parameters():
        p.requires_grad = False
    return base


@torch.no_grad()
def compute_targets(
    clean: torch.Tensor,
    deg: torch.Tensor,
    *,
    mode: str,
    base_unet: ResidualUNet | None,
    cfg: dict[str, Any] | None = None,
    batch: dict[str, Any] | None = None,
    residual_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return (x0_target, cond, restored_base) for the given diffusion mode.

    * clean mode: x0=clean, cond=deg, restored_base=None.
    * residual mode: x0 = residual_target / residual_scale,
      cond = cat(deg, restored_base), restored_base also returned for
      diagnostics. The scaling brings the diffusion target close to unit
      variance so eps prediction has useful signal.
    * masked_completion mode: cond is the same base-aware conditioning as
      residual mode. x0 is either residual/restored_base or clean, selected
      by cfg.masked_completion_target. Final outputs are composed separately.
    """
    if mode == "clean":
        return clean, deg, None
    if base_unet is None:
        raise RuntimeError(f"{mode} mode requires a loaded base U-Net")
    cfg = cfg or {}
    restored_base = base_unet(deg).clamp(-1.0, 1.0)
    if mode == "residual":
        residual = build_residual_target(clean, restored_base, cfg)
        x0 = residual / max(float(residual_scale), 1e-6)
    elif mode == "masked_completion":
        target = str(cfg.get("masked_completion_target", "residual"))
        if target == "residual":
            x0 = (clean - restored_base) / max(float(residual_scale), 1e-6)
        elif target == "clean":
            x0 = clean
        else:
            raise ValueError(f"unknown masked_completion_target {target!r}")
    else:
        raise ValueError(f"unknown diffusion mode {mode!r}")
    extras = _conditioning_extras(
        batch, cfg, deg.device, b=int(deg.shape[0]), h=int(deg.shape[-2]), w=int(deg.shape[-1])
    )
    cond = torch.cat([deg, restored_base, *extras], dim=1)
    return x0, cond, restored_base


@torch.no_grad()
def estimate_residual_scale(
    base_unet: ResidualUNet,
    loader,
    device,
    *,
    max_batches: int = 8,
    method: str = "std",
    floor: float = 5e-3,
    ceil: float = 0.5,
    cfg: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float]]:
    """Estimate a robust scale for the U-Net residual from a few batches.

    Method choices:
      * ``std``     – standard deviation of the residual.
      * ``mad``     – 1.4826 * median absolute deviation (≈ Gaussian std).
      * ``p95``     – 95th-percentile absolute residual.

    The estimate is clamped to ``[floor, ceil]`` to avoid near-zero blow-ups
    or absurdly large scales. Stats are returned alongside the resolved
    scale for logging.
    """
    abs_vals: list[torch.Tensor] = []
    sq_vals: list[torch.Tensor] = []
    n_pixels = 0
    seen = 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        clean = batch["clean"].to(device)
        deg = batch["degraded"].to(device)
        restored_base = base_unet(deg).clamp(-1.0, 1.0)
        residual = build_residual_target(clean, restored_base, cfg or {}).flatten()
        abs_vals.append(residual.abs().detach().cpu())
        sq_vals.append((residual ** 2).detach().cpu())
        n_pixels += residual.numel()
        seen += int(clean.shape[0])
    if not abs_vals:
        return float(min(max(0.05, floor), ceil)), {
            "samples": 0, "method": method, "abs_mean": 0.0,
            "std": 0.0, "mad": 0.0, "p95": 0.0,
        }
    a = torch.cat(abs_vals)
    s = torch.cat(sq_vals)
    abs_mean = float(a.mean())
    std = float(s.mean().sqrt())  # residual mean is ~0, so sqrt(E[r^2]) ≈ std
    mad = float((a - a.median()).abs().median()) * 1.4826
    p95 = float(a.kthvalue(max(1, int(0.95 * a.numel())))[0])
    if method == "std":
        scale = std
    elif method == "mad":
        scale = mad
    elif method == "p95":
        scale = p95
    elif method == "abs_mean":
        scale = abs_mean
    else:
        raise ValueError(f"unknown residual_scale method {method!r}")
    scale = float(min(max(scale, floor), ceil))
    return scale, {
        "samples": seen, "method": method,
        "abs_mean": abs_mean, "std": std, "mad": mad, "p95": p95,
        "resolved": scale,
    }


def resolve_residual_scale(
    cfg_value,
    base_unet: ResidualUNet | None,
    train_loader,
    device,
    *,
    cfg: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float] | None]:
    """Resolve cfg.residual_scale to a numeric value. ``auto`` triggers
    estimation from a small subset of training samples."""
    if isinstance(cfg_value, str) and cfg_value.lower() == "auto":
        if base_unet is None:
            raise RuntimeError("residual_scale='auto' requires a base U-Net")
        scale, stats = estimate_residual_scale(base_unet, train_loader, device, cfg=cfg)
        return scale, stats
    val = float(cfg_value)
    if val <= 0:
        raise ValueError(f"residual_scale must be > 0, got {val}")
    return val, None


def masked_eps_losses(
    pred: torch.Tensor,
    eps: torch.Tensor,
    mask: torch.Tensor,
    *,
    min_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (training_loss, true_mask_only_loss) for masked completion."""
    mse = (pred - eps) ** 2
    weight = mask.to(device=pred.device, dtype=pred.dtype)
    if weight.ndim == 3:
        weight = weight.unsqueeze(1)
    if weight.shape[1] != 1:
        weight = weight[:, :1]
    if weight.shape[-2:] != pred.shape[-2:]:
        weight = F.interpolate(weight, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    weight = weight.clamp(0.0, 1.0).expand_as(mse)
    mask_loss = (weight * mse).sum() / weight.sum().clamp_min(1e-8)
    if float(min_weight) > 0.0:
        train_weight = weight.clamp_min(float(min_weight))
        train_loss = (train_weight * mse).sum() / train_weight.sum().clamp_min(1e-8)
    else:
        train_loss = mask_loss
    return train_loss, mask_loss


@torch.no_grad()
def evaluate_eps_mse(
    model,
    diffusion,
    loader,
    device,
    *,
    mode: str = "clean",
    base_unet: ResidualUNet | None = None,
    residual_scale: float = 1.0,
    residual_clip: float = 0.10,
    residual_blend: float = 0.25,
    gate_cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute eps MSE plus x0-based diagnostics.

    In residual mode the diffusion target x0 is the *scaled* residual. The
    diagnostics also report the unscaled residual (residual_target_l1),
    the model's x0-from-eps reconstruction error in the same unscaled space
    (residual_x0_l1), and a one-shot ``final_from_x0`` PSNR estimate which
    blends the x0-derived residual onto the U-Net base. This is a much
    cheaper proxy than full p_sample_loop sampling.
    """
    model.eval()
    losses: list[float] = []
    x0_l1: list[float] = []
    res_target_l1: list[float] = []
    res_x0_l1: list[float] = []
    final_from_x0_psnr: list[float] = []
    final_from_x0_mae: list[float] = []
    gate_area_fracs: list[float] = []
    mask_only_losses: list[float] = []
    mask_area_fracs: list[float] = []
    inside_mask_mae: list[float] = []
    outside_context_l1: list[float] = []
    outside_mask_mae_to_base: list[float] = []
    nonzero_gate_samples = 0
    outside_gate_max_diff = 0.0
    outside_mask_max_diff = 0.0
    detail_base: dict[str, list[float]] = defaultdict(list)
    detail_refined: dict[str, list[float]] = defaultdict(list)
    mask_filter_seen = 0
    mask_filter_kept = 0
    mask_filter_skipped_batches = 0
    for batch in loader:
        if mode == "masked_completion":
            mask0 = build_completion_mask(batch, device, int(batch["clean"].shape[-2]), int(batch["clean"].shape[-1]))
            keep, area = localized_mask_keep(mask0, gate_cfg or {})
            mask_filter_seen += int(keep.numel())
            mask_filter_kept += int(keep.sum().item())
            if not bool(keep.any().item()):
                mask_filter_skipped_batches += 1
                continue
            batch = filter_batch_by_indices(batch, keep)
        clean = batch["clean"].to(device)
        deg = batch["degraded"].to(device)
        b = clean.shape[0]
        x0, cond, restored_base = compute_targets(
            clean, deg, mode=mode, base_unet=base_unet, cfg=gate_cfg or {},
            batch=batch, residual_scale=residual_scale,
        )
        t = torch.randint(0, diffusion.timesteps, (b,), device=device, dtype=torch.long)
        eps = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, eps)
        pred = model(x_t, t, cond)
        if mode == "masked_completion":
            mask = build_completion_mask(batch, device, int(x0.shape[-2]), int(x0.shape[-1]))
            train_loss, mask_loss = masked_eps_losses(
                pred, eps, mask, min_weight=float((gate_cfg or {}).get("masked_loss_min_weight", 0.0) or 0.0)
            )
            losses.append(float(train_loss.item()))
            mask_only_losses.append(float(mask_loss.item()))
            mask_area_fracs.extend([float(v) for v in mask.flatten(1).mean(dim=1).detach().cpu()])
        else:
            losses.append(F.mse_loss(pred, eps).item())
        x0_pred = diffusion.predict_x0_from_eps(x_t, t, pred)
        x0_l1.append(F.l1_loss(x0_pred, x0).item())

        if mode == "masked_completion" and restored_base is not None:
            target = str((gate_cfg or {}).get("masked_completion_target", "residual"))
            final = compose_masked_completion(
                restored_base, x0_pred, mask,
                target=target,
                residual_scale=residual_scale,
                residual_clip=residual_clip,
                residual_blend=residual_blend,
            )
            outside = (1.0 - mask).to(restored_base.dtype).expand_as(restored_base)
            inside = mask.to(restored_base.dtype).expand_as(restored_base)
            if float(outside.sum().item()) > 0.0:
                diff = (final - restored_base).abs() * outside
                outside_mask_max_diff = max(outside_mask_max_diff, float(diff.max().item()))
                outside_context_l1.append(float(diff.sum().item() / outside.sum().clamp_min(1e-8).item()))
                outside_mask_mae_to_base.append(float(diff.sum().item() / outside.sum().clamp_min(1e-8).item()))
            if float(inside.sum().item()) > 0.0:
                inside_mask_mae.append(float(((to01(final) - to01(clean)).abs() * inside).sum().item() / inside.sum().clamp_min(1e-8).item()))
            mse = F.mse_loss(to01(final), to01(clean)).item()
            psnr_db = 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)
            final_from_x0_psnr.append(float(psnr_db))
            final_from_x0_mae.append(F.l1_loss(to01(final), to01(clean)).item())

        if mode == "residual" and restored_base is not None:
            residual_target = build_residual_target(clean, restored_base, gate_cfg or {})
            residual_x0 = x0_pred * float(residual_scale)
            res_target_l1.append(residual_target.abs().mean().item())
            res_x0_l1.append((residual_x0 - residual_target).abs().mean().item())
            sampled_residual = residual_x0.clamp(-residual_clip, residual_clip)
            gate = build_residual_gate(batch, gate_cfg or {"residual_gate_mode": "none"}, device)
            final = compose_gated_residual(
                restored_base, sampled_residual, gate, residual_blend,
            )
            outside = (gate <= 1e-6).to(restored_base.dtype).expand_as(restored_base)
            if float(outside.sum().item()) > 0.0:
                outside_gate_max_diff = max(
                    outside_gate_max_diff,
                    float(((final - restored_base).abs() * outside).max().item()),
                )
            gate_area = gate.flatten(1).mean(dim=1)
            gate_area_fracs.extend([float(v) for v in gate_area.detach().cpu()])
            nonzero_gate_samples += int((gate.flatten(1).amax(dim=1) > 0).sum().item())
            for k, v in detail_metric_values(restored_base, clean, gate_cfg or {}).items():
                detail_base[k].append(v)
            for k, v in detail_metric_values(final, clean, gate_cfg or {}).items():
                detail_refined[k].append(v)
            mse = F.mse_loss(((final + 1) * 0.5).clamp(0, 1),
                             ((clean + 1) * 0.5).clamp(0, 1)).item()
            psnr_db = 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)
            final_from_x0_psnr.append(float(psnr_db))
            final_from_x0_mae.append(F.l1_loss(((final + 1) * 0.5).clamp(0, 1),
                                               ((clean + 1) * 0.5).clamp(0, 1)).item())
    model.train()
    out = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "x0_l1": float(np.mean(x0_l1)) if x0_l1 else 0.0,
    }
    if res_target_l1:
        out.update({
            "residual_target_l1": float(np.mean(res_target_l1)),
            "residual_x0_l1": float(np.mean(res_x0_l1)),
            "final_from_x0_psnr": float(np.mean(final_from_x0_psnr)),
            "final_from_x0_mae": float(np.mean(final_from_x0_mae)),
            "mean_gate_area_fraction": float(np.mean(gate_area_fracs)) if gate_area_fracs else 0.0,
            "percent_samples_with_nonzero_gate": 100.0 * nonzero_gate_samples / max(1, len(gate_area_fracs)),
            "outside_gate_max_diff": outside_gate_max_diff,
        })
        for metric_name in ("highpass_mae", "lowpass_mae", "gradient_mae"):
            out[f"{metric_name}_base"] = float(np.mean(detail_base[metric_name])) if detail_base[metric_name] else 0.0
            out[f"{metric_name}_refined"] = (
                float(np.mean(detail_refined[metric_name])) if detail_refined[metric_name] else 0.0
            )
    if mask_area_fracs:
        out.update({
            "mask_only_loss": float(np.mean(mask_only_losses)) if mask_only_losses else 0.0,
            "mean_mask_area_fraction": float(np.mean(mask_area_fracs)),
            "inside_mask_mae": float(np.mean(inside_mask_mae)) if inside_mask_mae else 0.0,
            "outside_context_l1": float(np.mean(outside_context_l1)) if outside_context_l1 else 0.0,
            "outside_mask_mae_to_base": (
                float(np.mean(outside_mask_mae_to_base)) if outside_mask_mae_to_base else 0.0
            ),
            "outside_mask_max_diff": outside_mask_max_diff,
            "mask_area_filter_pass_fraction": mask_filter_kept / max(1, mask_filter_seen),
            "mask_area_filter_skipped_batches": mask_filter_skipped_batches,
            "final_from_x0_psnr": float(np.mean(final_from_x0_psnr)) if final_from_x0_psnr else 0.0,
            "final_from_x0_mae": float(np.mean(final_from_x0_mae)) if final_from_x0_mae else 0.0,
        })
    return out


@torch.no_grad()
def save_sample_grid(
    path: str,
    diffusion: GaussianDiffusion,
    model,
    deg: torch.Tensor,
    clean: torch.Tensor,
    sampling_steps: int,
    *,
    title: str = "DDPM samples",
    sample_labels: list[str] | None = None,
    mode: str = "clean",
    base_unet: ResidualUNet | None = None,
    residual_scale: float = 1.0,
    residual_clip: float = 0.10,
    residual_blend: float = 0.25,
    sampling_mode: str = "ddpm",
    gate_batch: dict[str, Any] | None = None,
    gate_cfg: dict[str, Any] | None = None,
) -> bool:
    """Save a labeled diagnostic grid for the current diffusion mode.

    clean mode: 4 columns: degraded | DDPM output | clean | abs error
    residual mode: 5 columns: degraded | U-Net base | DDPM refined | clean | abs error
    masked_completion mode: degraded | mask | reliability | U-Net base | final | clean | abs error
    Residual mode applies residual_scale, residual_clip, residual_blend so the
    grid reflects the conservative inference path actually used at eval time.
    """
    n = min(4, deg.shape[0])
    deg_n = deg[:n]
    clean_n = clean[:n]
    if mode in ("residual", "masked_completion"):
        if base_unet is None:
            print(f"[train-ddpm] {mode} sampling requires a base U-Net; skipping grid")
            return False
        restored_base = base_unet(deg_n).clamp(-1.0, 1.0)
        if gate_batch is None:
            gate_batch = {"degraded": deg_n, "clean": clean_n}
        extras = _conditioning_extras(
            gate_batch, gate_cfg or {}, deg_n.device, b=n, h=int(deg_n.shape[-2]), w=int(deg_n.shape[-1])
        )
        cond = torch.cat([deg_n, restored_base, *extras], dim=1)
        sample_shape = (n, 3, deg_n.shape[-2], deg_n.shape[-1])
        # Sampled x0 lives in scaled-residual space, roughly unit-bounded.
        clip_min, clip_max = -1.0, 1.0
    else:
        cond = deg_n
        sample_shape = cond.shape
        clip_min, clip_max = -1.0, 1.0
    try:
        if sampling_mode == "ddim":
            out = diffusion.ddim_sample_loop(
                model, cond, shape=sample_shape, sampling_steps=sampling_steps,
                progress=False, x0_clip_min=clip_min, x0_clip_max=clip_max,
            )
        elif sampling_mode == "mean":
            out = diffusion.p_sample_loop(
                model, cond, shape=sample_shape, sampling_steps=sampling_steps,
                mean_only=True, progress=False,
                x0_clip_min=clip_min, x0_clip_max=clip_max,
            )
        else:
            out = diffusion.p_sample_loop(
                model, cond, shape=sample_shape, sampling_steps=sampling_steps,
                progress=False, x0_clip_min=clip_min, x0_clip_max=clip_max,
            )
    except Exception as e:
        print(f"[train-ddpm] sampling failed, skipping grid: {e}")
        return False
    if mode == "masked_completion":
        mask = build_completion_mask(gate_batch or {}, deg_n.device, int(deg_n.shape[-2]), int(deg_n.shape[-1]))[:n]
        target = str((gate_cfg or {}).get("masked_completion_target", "residual"))
        refined = compose_masked_completion(
            restored_base, out, mask,
            target=target,
            residual_scale=residual_scale,
            residual_clip=residual_clip,
            residual_blend=residual_blend,
        )
        mask_area = float(mask.flatten(1).mean().item())
        full_title = (
            f"{title} | diffusion_target=masked_completion "
            f"masked_completion_target={target} sampling_mode={sampling_mode} "
            f"sampling_steps={sampling_steps} mask_area_fraction={mask_area:.4f} "
            f"residual_scale={residual_scale:.4f}"
        )
        save_labeled_masked_completion_grid(
            path,
            deg_n,
            mask,
            restored_base,
            refined,
            clean_n,
            title=full_title,
            sample_labels=sample_labels,
            reliability=gate_batch.get("reliability_map")[:n] if gate_batch and gate_batch.get("reliability_map") is not None else None,
        )
    elif mode == "residual":
        sampled_residual = (out * float(residual_scale)).clamp(-residual_clip, residual_clip)
        gate = build_residual_gate(gate_batch, gate_cfg or {"residual_gate_mode": "none"}, deg_n.device)[:n]
        refined = compose_gated_residual(
            restored_base, sampled_residual, gate, residual_blend,
        )
        gate_mode = str((gate_cfg or {}).get("residual_gate_mode", "none"))
        rc = residual_target_config(gate_cfg or {})
        full_title = (
            f"{title} | sampling_mode={sampling_mode} "
            f"residual_target_type={rc['residual_target_type']} "
            f"detail_sigma={float(rc['detail_sigma']):.2f} "
            f"lowpass_sigma={float(rc['lowpass_sigma']):.2f} "
            f"residual_scale={residual_scale:.4f} residual_clip={residual_clip:.3f} "
            f"residual_blend={residual_blend:.2f} blend={residual_blend:.2f} "
            f"residual_gate_mode={gate_mode}"
        )
        if gate_mode == "none":
            save_labeled_residual_refinement_grid(
                path,
                deg_n,
                restored_base,
                refined,
                clean_n,
                title=full_title,
                sample_labels=sample_labels,
            )
        else:
            save_labeled_gated_residual_refinement_grid(
                path,
                deg_n,
                restored_base,
                gate,
                refined,
                clean_n,
                title=full_title,
                sample_labels=sample_labels,
                reliability=gate_batch.get("reliability_map")[:n] if gate_batch.get("reliability_map") is not None else None,
            )
    else:
        save_labeled_restoration_grid(
            path,
            deg_n,
            out,
            clean_n,
            output_label="DDPM output",
            title=f"{title} | mode={sampling_mode}",
            sample_labels=sample_labels,
        )
    return True


def main() -> int:
    args = parse_args()
    cfg = load_cfg(args)
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    sample_dir = os.path.join(out_dir, "samples")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.jsonl")

    deg_kwargs = _build_degradation_kwargs(cfg)
    train_filter = cfg.get("ddpm_train_degradation_types")
    val_filter = cfg.get("ddpm_val_degradation_types", train_filter)
    train_probs = _filtered_degradation_probs(cfg, train_filter)
    val_probs = _filtered_degradation_probs(cfg, val_filter)
    if train_filter:
        print(f"[train-ddpm] training degradation filter={list(train_filter)} probs={train_probs}")
    if val_filter:
        print(f"[train-ddpm] validation degradation filter={list(val_filter)} probs={val_probs}")
    full = PatchDegradationDataset(
        manifest_path=cfg["manifest"],
        patch_dir=cfg["patch_dir"],
        probs=train_probs if train_probs is not None else cfg.get("degradation_probs"),
        max_samples=cfg.get("max_samples"),
        image_size=cfg.get("image_size"),
        return_degradation_mask=bool(cfg.get("return_degradation_mask", False)),
        return_reliability_map=bool(cfg.get("return_reliability_map", False)),
        **deg_kwargs,
    )
    if len(full) == 0:
        print("[train-ddpm] empty dataset"); return 1
    print(f"[train-ddpm] degradation_profile={full.degradation_profile or 'balanced'}  "
          f"types={list(full.probs.keys())}")
    val_full = PatchDegradationDataset(
        manifest_path=cfg["manifest"],
        patch_dir=cfg["patch_dir"],
        probs=val_probs if val_probs is not None else cfg.get("degradation_probs"),
        max_samples=cfg.get("max_samples"),
        image_size=cfg.get("image_size"),
        return_degradation_mask=bool(cfg.get("return_degradation_mask", False)),
        return_reliability_map=bool(cfg.get("return_reliability_map", False)),
        **deg_kwargs,
    )

    train_idx, val_idx = split_indices(len(full), float(cfg["val_fraction"]), int(cfg["seed"]))
    train_set = Subset(full, train_idx)
    val_set = Subset(val_full, val_idx)
    train_batches = len(train_set) // int(cfg["batch_size"])
    val_batches = (len(val_set) + int(cfg["batch_size"]) - 1) // int(cfg["batch_size"])
    print(f"[train-ddpm] eligible train samples={len(train_set)} batches={train_batches}  "
          f"val samples={len(val_set)} batches={val_batches}")

    # Fixed (deterministic) val samples for cross-epoch comparison.
    n_fixed = int(cfg.get("fixed_val_samples", 4))
    fixed_view = PatchDegradationDataset(
        manifest_path=cfg["manifest"],
        patch_dir=cfg["patch_dir"],
        probs=val_probs if val_probs is not None else cfg.get("degradation_probs"),
        max_samples=cfg.get("max_samples"),
        image_size=cfg.get("image_size"),
        deterministic=True,
        seed=int(cfg["seed"]),
        return_degradation_mask=bool(cfg.get("return_degradation_mask", False)),
        return_reliability_map=bool(cfg.get("return_reliability_map", False)),
        **deg_kwargs,
    )
    fixed_clean_list, fixed_deg_list, fixed_mask_list, fixed_reliability_list = [], [], [], []
    fixed_kinds, fixed_params, fixed_paths = [], [], []
    fixed_indices = val_idx[: max(1, min(n_fixed, len(val_idx)))]
    for i in fixed_indices:
        s = fixed_view[i]
        fixed_clean_list.append(s["clean"])
        fixed_deg_list.append(s["degraded"])
        if "degradation_mask" in s:
            fixed_mask_list.append(s["degradation_mask"])
        if "reliability_map" in s:
            fixed_reliability_list.append(s["reliability_map"])
        fixed_kinds.append(s["degradation_type"])
        fixed_params.append(s.get("degradation_params", {}))
        fixed_paths.append(s.get("path", ""))

    train_loader = DataLoader(
        train_set, batch_size=int(cfg["batch_size"]), shuffle=True,
        num_workers=int(cfg["num_workers"]), drop_last=True, persistent_workers=False,
        collate_fn=patch_degradation_collate,
    )
    val_loader = DataLoader(
        val_set, batch_size=int(cfg["batch_size"]), shuffle=False,
        num_workers=int(cfg["num_workers"]), drop_last=False, persistent_workers=False,
        collate_fn=patch_degradation_collate,
    )

    diffusion = GaussianDiffusion(
        timesteps=int(cfg.get("timesteps", 1000)),
        schedule=str(cfg.get("beta_schedule", "cosine")),
    ).to(device)

    mode = str(cfg.get("diffusion_target", "clean"))
    base_unet: ResidualUNet | None = None
    residual_scale = 1.0
    residual_clip = float(cfg.get("residual_clip", 0.10))
    residual_blend = float(cfg.get("residual_blend", 0.25))
    if mode in ("residual", "masked_completion"):
        base_ckpt = cfg["base_ckpt"]
        base_unet = load_frozen_base_unet(base_ckpt, device)
        n_base_params = sum(p.numel() for p in base_unet.parameters())
        print(f"[train-ddpm] {mode} mode: loaded frozen base U-Net "
              f"({n_base_params:,} params) from {base_ckpt}")
        # Resolve residual_scale (numeric or 'auto'). Persist resolved value
        # back into cfg so it travels with the checkpoint.
        residual_scale, scale_stats = resolve_residual_scale(
            cfg.get("residual_scale", "auto"), base_unet, train_loader, device, cfg=cfg,
        )
        cfg["residual_scale"] = float(residual_scale)
        if scale_stats is not None:
            print(f"[train-ddpm] residual_scale=auto resolved to {residual_scale:.5f} "
                  f"(method={scale_stats['method']}, samples={scale_stats['samples']}, "
                  f"abs_mean={scale_stats['abs_mean']:.5f}, std={scale_stats['std']:.5f}, "
                  f"mad={scale_stats['mad']:.5f}, p95={scale_stats['p95']:.5f})")
        else:
            print(f"[train-ddpm] residual_scale={residual_scale:.5f} (from config)")
        if mode == "residual":
            rc = residual_target_config(cfg)
            print(f"[train-ddpm] residual_target_type={rc['residual_target_type']}  "
                  f"detail_sigma={float(rc['detail_sigma']):.2f}  "
                  f"lowpass_sigma={float(rc['lowpass_sigma']):.2f}  "
                  f"lowpass_weight={float(rc['lowpass_weight']):.2f}  "
                  f"highpass_weight={float(rc['highpass_weight']):.2f}  "
                  f"residual_scale={residual_scale:.5f}")
        else:
            print(f"[train-ddpm] masked_completion_target={cfg.get('masked_completion_target', 'residual')}  "
                  f"masked_loss_min_weight={float(cfg.get('masked_loss_min_weight', 0.0)):.3f}  "
                  f"masked_context_consistency_weight={float(cfg.get('masked_context_consistency_weight', 0.0)):.3f}  "
                  f"residual_scale={residual_scale:.5f}")
        print(f"[train-ddpm] residual_clip={residual_clip:.4f}  "
              f"residual_blend={residual_blend:.3f}")
        print(f"[train-ddpm] residual_gate_mode={cfg.get('residual_gate_mode', 'none')}  "
              f"return_degradation_mask={bool(cfg.get('return_degradation_mask', False))}  "
              f"return_reliability_map={bool(cfg.get('return_reliability_map', False))}")

    fixed_clean = torch.stack(fixed_clean_list, dim=0).to(device) if fixed_clean_list else None
    fixed_deg = torch.stack(fixed_deg_list, dim=0).to(device) if fixed_deg_list else None
    fixed_mask = torch.stack(fixed_mask_list, dim=0).to(device) if fixed_mask_list else None
    fixed_reliability = (
        torch.stack(fixed_reliability_list, dim=0).to(device) if fixed_reliability_list else None
    )
    fixed_labels = sample_labels(fixed_indices, fixed_kinds, fixed_paths)
    fixed_gate_batch: dict[str, Any] | None = None
    if fixed_clean is not None:
        fixed_gate_batch = {
            "clean": fixed_clean,
            "degraded": fixed_deg,
            "degradation_type": fixed_kinds,
        }
        if fixed_mask is not None:
            fixed_gate_batch["degradation_mask"] = fixed_mask
        if fixed_reliability is not None:
            fixed_gate_batch["reliability_map"] = fixed_reliability
    if fixed_clean is not None:
        print(f"[train-ddpm] fixed val samples cached: {fixed_clean.shape[0]} kinds={fixed_kinds}")

    cond_channels = _ddpm_cond_channels(cfg)
    model = DDPMUNet(
        in_channels=3,
        cond_channels=cond_channels,
        out_channels=3,
        base_channels=int(cfg.get("base_channels", 32)),
        channel_mults=tuple(cfg.get("channel_mults", [1, 2, 4, 4])),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train-ddpm] device={device}  mode={mode}  cond_ch={cond_channels}  "
          f"params={n_params:,}  train={len(train_set)}  val={len(val_set)}  "
          f"T={diffusion.timesteps}")

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))

    sample_every = int(cfg.get("sample_every", 500))
    sampling_steps = int(cfg.get("sampling_steps", 100))
    save_step_train_batch_samples = bool(
        cfg.get("save_step_train_batch_samples", cfg.get("save_train_batch_samples", True))
    )
    save_step_fixed_val_samples = bool(cfg.get("save_step_fixed_val_samples", False))
    log_f = open(log_path, "w")
    best_val: float | None = None
    step = 0
    t0 = time.time()

    type_counter: Counter = Counter()
    raw_type_counter: Counter = Counter()
    mask_filter_seen = 0
    mask_filter_kept = 0
    mask_filter_skipped_batches = 0
    sampling_disabled = False  # disabled if first attempt fails badly

    for epoch in range(int(cfg["epochs"])):
        model.train()
        epoch_train_losses: list[float] = []
        for batch in train_loader:
            for k in batch.get("degradation_type", []):
                raw_type_counter[str(k)] += 1
            if mode == "masked_completion":
                mask0 = build_completion_mask(
                    batch, device, int(batch["clean"].shape[-2]), int(batch["clean"].shape[-1])
                )
                keep, area = localized_mask_keep(mask0, cfg)
                mask_filter_seen += int(keep.numel())
                mask_filter_kept += int(keep.sum().item())
                if bool(cfg.get("mask_area_filter_enabled", False)) and not bool(keep.all().item()):
                    rejected = int((~keep).sum().item())
                    print(
                        f"[train-ddpm] mask_area_filter kept={int(keep.sum().item())}/{int(keep.numel())} "
                        f"rejected={rejected} area_min={float(area.min().item()):.4f} "
                        f"area_mean={float(area.mean().item()):.4f} area_max={float(area.max().item()):.4f}"
                    )
                if not bool(keep.any().item()):
                    mask_filter_skipped_batches += 1
                    print("[train-ddpm] skipping batch: no masks passed mask_area_filter")
                    continue
                batch = filter_batch_by_indices(batch, keep)
            clean = batch["clean"].to(device)
            deg = batch["degraded"].to(device)
            b = clean.shape[0]

            for k in batch.get("degradation_type", []):
                type_counter[str(k)] += 1

            x0, cond, _ = compute_targets(
                clean, deg, mode=mode, base_unet=base_unet,
                cfg=cfg, batch=batch,
                residual_scale=residual_scale,
            )
            t = torch.randint(0, diffusion.timesteps, (b,), device=device, dtype=torch.long)
            eps = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, t, eps)
            pred = model(x_t, t, cond)
            mask_area_fraction = None
            mask_only_loss_value = None
            if mode == "masked_completion":
                mask = build_completion_mask(batch, device, int(x0.shape[-2]), int(x0.shape[-1]))
                loss, mask_only_loss = masked_eps_losses(
                    pred, eps, mask, min_weight=float(cfg.get("masked_loss_min_weight", 0.0) or 0.0)
                )
                mask_only_loss_value = float(mask_only_loss.detach())
                mask_area_fraction = float(mask.flatten(1).mean().detach().cpu())
            else:
                loss = F.mse_loss(pred, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_train_losses.append(float(loss.detach()))

            if step % 20 == 0:
                rec = {"step": step, "epoch": epoch, "phase": "train",
                       "train_loss": float(loss.detach()),
                       "loss": float(loss.detach()),
                       "residual_target_type": str(cfg.get("residual_target_type", "pixel")),
                       "residual_scale": float(residual_scale),
                       "elapsed_s": round(time.time() - t0, 1)}
                if mode == "masked_completion":
                    rec["masked_completion_target"] = str(cfg.get("masked_completion_target", "residual"))
                    rec["mask_only_loss"] = mask_only_loss_value
                    rec["mask_area_fraction"] = mask_area_fraction
                    rec["mask_area_filter_pass_fraction"] = mask_filter_kept / max(1, mask_filter_seen)
                    rec["mask_area_filter_skipped_batches"] = mask_filter_skipped_batches
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                if mode == "masked_completion":
                    print(
                        f"[train-ddpm] ep{epoch} step{step} masked_mse_eps={float(loss):.4f} "
                        f"mask_only={mask_only_loss_value:.4f} mask_area_fraction={mask_area_fraction:.4f}"
                    )
                else:
                    print(f"[train-ddpm] ep{epoch} step{step} mse_eps={float(loss):.4f}")

            if sample_every > 0 and step > 0 and step % sample_every == 0 and not sampling_disabled:
                if save_step_train_batch_samples:
                    ok = save_sample_grid(
                        os.path.join(sample_dir, f"train_batch_step_{step:06d}_labeled.png"),
                        diffusion, model, deg, clean, sampling_steps,
                        title=f"DDPM train batch samples | step {step:06d}",
                        sample_labels=batch_sample_labels(
                            list(batch.get("degradation_type", [])),
                            list(batch.get("path", [])),
                        ),
                        mode=mode, base_unet=base_unet,
                        residual_scale=residual_scale,
                        residual_clip=residual_clip,
                        residual_blend=residual_blend,
                        gate_batch=batch,
                        gate_cfg=cfg,
                    )
                    if not ok:
                        sampling_disabled = True
                if (
                    save_step_fixed_val_samples
                    and fixed_clean is not None
                    and not sampling_disabled
                ):
                    ok = save_sample_grid(
                        os.path.join(sample_dir, f"fixed_val_step_{step:06d}_labeled.png"),
                        diffusion, model, fixed_deg, fixed_clean, sampling_steps,
                        title=f"DDPM fixed validation samples | step {step:06d}",
                        sample_labels=fixed_labels,
                        mode=mode, base_unet=base_unet,
                        residual_scale=residual_scale,
                        residual_clip=residual_clip,
                        residual_blend=residual_blend,
                        gate_batch=fixed_gate_batch,
                        gate_cfg=cfg,
                    )
                    if not ok:
                        sampling_disabled = True
            step += 1

        val_metrics = evaluate_eps_mse(
            model, diffusion, val_loader, device,
            mode=mode, base_unet=base_unet,
            residual_scale=residual_scale,
            residual_clip=residual_clip,
            residual_blend=residual_blend,
            gate_cfg=cfg,
        )
        epoch_train_loss = float(np.mean(epoch_train_losses)) if epoch_train_losses else 0.0
        rec = {
            "step": step, "epoch": epoch, "phase": "val",
            "train_loss": epoch_train_loss,
            "val_loss": val_metrics["loss"], "loss": val_metrics["loss"],
            "x0_l1": val_metrics["x0_l1"],
            "residual_target_type": str(cfg.get("residual_target_type", "pixel")),
            "residual_scale": float(residual_scale),
            "degradation_type_counts": dict(type_counter),
            "raw_degradation_type_counts": dict(raw_type_counter),
            "mask_area_filter_pass_fraction": mask_filter_kept / max(1, mask_filter_seen),
            "mask_area_filter_skipped_batches": mask_filter_skipped_batches,
            "elapsed_s": round(time.time() - t0, 1),
        }
        for k in ("residual_target_l1", "residual_x0_l1",
                  "final_from_x0_psnr", "final_from_x0_mae",
                  "mean_gate_area_fraction", "percent_samples_with_nonzero_gate",
                  "outside_gate_max_diff",
                  "mask_only_loss", "mean_mask_area_fraction", "inside_mask_mae",
                  "outside_context_l1", "outside_mask_mae_to_base", "outside_mask_max_diff",
                  "mask_area_filter_pass_fraction", "mask_area_filter_skipped_batches",
                  "highpass_mae_base", "highpass_mae_refined",
                  "lowpass_mae_base", "lowpass_mae_refined",
                  "gradient_mae_base", "gradient_mae_refined"):
            if k in val_metrics:
                rec[k] = val_metrics[k]
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        if mode == "masked_completion":
            print(
                f"[val-ddpm] ep{epoch} loss={val_metrics['loss']:.4f}  "
                f"mask_only_loss={val_metrics.get('mask_only_loss', 0.0):.4f}  "
                f"x0_l1={val_metrics['x0_l1']:.4f}  "
                f"final_from_x0_psnr={val_metrics.get('final_from_x0_psnr', 0.0):.2f}dB  "
                f"mask_area_fraction={val_metrics.get('mean_mask_area_fraction', 0.0):.4f}  "
                f"filter_pass={val_metrics.get('mask_area_filter_pass_fraction', 0.0):.4f}  "
                f"skipped_batches={val_metrics.get('mask_area_filter_skipped_batches', 0)}  "
                f"outside_mask_max_diff={val_metrics.get('outside_mask_max_diff', 0.0):.8f}"
            )
        elif "residual_x0_l1" in val_metrics:
            print(
                f"[val-ddpm] ep{epoch} loss={val_metrics['loss']:.4f}  "
                f"x0_l1={val_metrics['x0_l1']:.4f}  "
                f"res_target_l1={val_metrics['residual_target_l1']:.4f}  "
                f"res_x0_l1={val_metrics['residual_x0_l1']:.4f}  "
                f"final_from_x0_psnr={val_metrics['final_from_x0_psnr']:.2f}dB  "
                f"gate_area={val_metrics.get('mean_gate_area_fraction', 0.0):.4f}  "
                f"outside_gate_max_diff={val_metrics.get('outside_gate_max_diff', 0.0):.8f}"
            )
        else:
            print(f"[val-ddpm] ep{epoch} loss={val_metrics['loss']:.4f}  "
                  f"x0_l1={val_metrics['x0_l1']:.4f}")

        # Fixed val sample grid per epoch (only if sampling is healthy and runtime is reasonable).
        if fixed_clean is not None and not sampling_disabled:
            t_sample0 = time.time()
            ok = save_sample_grid(
                os.path.join(sample_dir, f"fixed_val_epoch_{epoch + 1:04d}_labeled.png"),
                diffusion, model, fixed_deg, fixed_clean, sampling_steps,
                title=f"DDPM fixed validation samples | epoch {epoch + 1:04d}",
                sample_labels=fixed_labels,
                mode=mode, base_unet=base_unet,
                residual_scale=residual_scale,
                residual_clip=residual_clip,
                residual_blend=residual_blend,
                gate_batch=fixed_gate_batch,
                gate_cfg=cfg,
            )
            elapsed = time.time() - t_sample0
            if not ok:
                print("[train-ddpm] fixed-epoch sampling failed; disabling further sampling")
                sampling_disabled = True
            else:
                print(f"[train-ddpm] saved epoch sample grid (sampling took {elapsed:.1f}s)")
                if elapsed > 600:
                    print("[train-ddpm] sampling > 600s; disabling further per-epoch sampling")
                    sampling_disabled = True
        elif sampling_disabled:
            print("[train-ddpm] skipping per-epoch sample grid (sampling disabled)")

        save_every = int(cfg.get("save_every", 1))
        state = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "step": step,
            "val": val_metrics,
            "config": cfg,
        }
        if save_every > 0 and (epoch % save_every == 0 or epoch == int(cfg["epochs"]) - 1):
            save_checkpoint(os.path.join(out_dir, "ckpt_last.pt"), state)
        v = val_metrics["loss"]
        if is_better(v, best_val, "min"):
            best_val = v
            save_checkpoint(os.path.join(out_dir, "ckpt_best.pt"), state)
            print(f"[val-ddpm] new best loss={v:.4f}")

    log_f.close()

    # End-of-run plots (loss curve + degradation distribution; metric curves
    # are not produced by DDPM training because epsilon-MSE is the only loss).
    plot_results = write_all_plots(
        out_dir, log_path,
        type_counter=type_counter,
        param_history=None,
        title_prefix="DDPM ",
    )
    print(f"[train-ddpm] plots: {plot_results}")
    print(f"[train-ddpm] done in {time.time() - t0:.1f}s. outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
