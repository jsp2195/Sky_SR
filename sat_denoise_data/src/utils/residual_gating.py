"""Residual-DDPM spatial gating utilities."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_HARD_GATE_MODES = [
    "mask_dropout",
    "mixed_structured",
    "lowfreq_atmospheric_bias",
    "blur_downsample_upsample",
]


def _image_shape_from_batch(batch: dict[str, Any]) -> tuple[int, int, int]:
    for key in ("degraded", "clean"):
        if key in batch:
            x = batch[key]
            return int(x.shape[0]), int(x.shape[-2]), int(x.shape[-1])
    if "degradation_mask" in batch:
        x = batch["degradation_mask"]
        return int(x.shape[0]), int(x.shape[-2]), int(x.shape[-1])
    raise ValueError("batch must contain degraded, clean, or degradation_mask")


def _mask_from_batch(batch: dict[str, Any], device: torch.device, h: int, w: int) -> torch.Tensor | None:
    mask = batch.get("degradation_mask")
    if mask is None:
        return None
    mask = mask.to(device=device, dtype=torch.float32)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    if mask.shape[-2:] != (h, w):
        mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
    return mask.clamp(0.0, 1.0)


def _smooth_gate(gate: torch.Tensor, *, dilate: int, blur: int) -> torch.Tensor:
    if dilate > 0:
        k = 2 * int(dilate) + 1
        gate = F.max_pool2d(gate, kernel_size=k, stride=1, padding=int(dilate))
    if blur > 0:
        k = 2 * int(blur) + 1
        gate = F.avg_pool2d(gate, kernel_size=k, stride=1, padding=int(blur))
    return gate.clamp(0.0, 1.0)


def build_residual_gate(batch: dict[str, Any], cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Build a residual-DDPM gate of shape [B,1,H,W].

    ``residual_gate_mode='none'`` returns all ones and preserves legacy
    residual-DDPM behavior. Other modes use degradation metadata and optional
    spatial masks to decide where the DDPM correction may modify the base.
    """
    b, h, w = _image_shape_from_batch(batch)
    mode = str(cfg.get("residual_gate_mode", "none") or "none")
    kinds = [str(k) for k in batch.get("degradation_type", [""] * b)]
    if len(kinds) < b:
        kinds.extend([""] * (b - len(kinds)))
    mask = _mask_from_batch(batch, device, h, w)

    if mode == "none":
        gate = torch.ones(b, 1, h, w, device=device, dtype=torch.float32)
    elif mode == "metadata":
        gate = mask if mask is not None else torch.zeros(b, 1, h, w, device=device, dtype=torch.float32)
    elif mode == "mask_dropout":
        gate = torch.zeros(b, 1, h, w, device=device, dtype=torch.float32)
        if mask is not None:
            for i, kind in enumerate(kinds[:b]):
                if kind == "mask_dropout":
                    gate[i] = mask[i]
    elif mode == "hard_modes":
        hard_modes = set(cfg.get("residual_gate_hard_modes", DEFAULT_HARD_GATE_MODES))
        fallback = float(cfg.get("residual_gate_default_for_hard_modes", 1.0))
        gate = torch.zeros(b, 1, h, w, device=device, dtype=torch.float32)
        for i, kind in enumerate(kinds[:b]):
            if kind not in hard_modes:
                continue
            if mask is not None and float(mask[i].max().item()) > 0.0:
                gate[i] = mask[i]
            else:
                gate[i].fill_(fallback)
    else:
        raise ValueError(
            f"unknown residual_gate_mode {mode!r}; expected none, metadata, hard_modes, or mask_dropout"
        )

    gate = gate.clamp(0.0, 1.0)
    gate = _smooth_gate(
        gate,
        dilate=int(cfg.get("residual_gate_dilate", 0) or 0),
        blur=int(cfg.get("residual_gate_blur", 0) or 0),
    )
    min_value = float(cfg.get("residual_gate_min_value", 0.0) or 0.0)
    if min_value > 0.0:
        gate = torch.where(gate > 0.0, gate.clamp_min(min_value), gate)
    return gate.clamp(0.0, 1.0).to(device=device, dtype=torch.float32)


def compose_gated_residual(
    restored_base: torch.Tensor,
    sampled_residual: torch.Tensor,
    gate: torch.Tensor,
    residual_blend: float,
    *,
    valid_min: float = -1.0,
    valid_max: float = 1.0,
) -> torch.Tensor:
    """Apply a gated residual correction and clamp to the valid image range."""
    return (
        restored_base + gate.to(restored_base.device, restored_base.dtype) * float(residual_blend) * sampled_residual
    ).clamp(valid_min, valid_max)


def build_completion_mask(
    batch: dict[str, Any],
    device: torch.device,
    h: int,
    w: int,
    *,
    require: bool = True,
) -> torch.Tensor:
    """Build a masked-completion eligibility mask [B,1,H,W].

    Priority:
      1. degradation_mask, where 1 means DDPM-eligible.
      2. 1 - reliability_map, where 1 means unreliable / low confidence.

    Masked completion is undefined without either metadata map. Callers may
    pass ``require=False`` only for synthetic unit tests.
    """
    b = _image_shape_from_batch(batch)[0]
    mask = batch.get("degradation_mask")
    source = "degradation_mask"
    if mask is None:
        reliability = batch.get("reliability_map")
        if reliability is not None:
            mask = 1.0 - reliability
            source = "1 - reliability_map"
    if mask is None:
        msg = (
            "masked_completion requires either degradation_mask or reliability_map; "
            "set return_degradation_mask=true or return_reliability_map=true"
        )
        if require:
            raise ValueError(msg)
        print(f"[masked-completion] WARNING: {msg}; using an all-zero mask")
        return torch.zeros(b, 1, h, w, device=device, dtype=torch.float32)
    mask = mask.to(device=device, dtype=torch.float32)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    if mask.shape[-2:] != (h, w):
        mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
    mask = (mask.clamp(0.0, 1.0) > 1e-6).to(dtype=torch.float32)
    # Attach lightweight debug metadata for callers that want to log it.
    mask._completion_mask_source = source  # type: ignore[attr-defined]
    return mask


def mask_area_filter_config(cfg: dict[str, Any]) -> tuple[bool, float, float]:
    enabled = bool(cfg.get("mask_area_filter_enabled", False))
    return enabled, float(cfg.get("mask_area_min", 0.0)), float(cfg.get("mask_area_max", 1.0))


def mask_area_fraction(mask: torch.Tensor) -> torch.Tensor:
    """Return per-sample area fraction for [B,1,H,W] masks."""
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return mask.float().flatten(1).mean(dim=1)


def localized_mask_keep(mask: torch.Tensor, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (keep_bool, area_fraction) for configured localized mask filtering."""
    enabled, min_area, max_area = mask_area_filter_config(cfg)
    area = mask_area_fraction(mask)
    if not enabled:
        return torch.ones_like(area, dtype=torch.bool), area
    keep = (area >= float(min_area)) & (area <= float(max_area))
    return keep, area


def filter_batch_by_indices(batch: dict[str, Any], keep: torch.Tensor) -> dict[str, Any]:
    """Filter tensor/list batch fields by a boolean keep vector."""
    keep_cpu = keep.detach().cpu().bool()
    idx = keep_cpu.nonzero(as_tuple=False).flatten().tolist()
    out: dict[str, Any] = {}
    for key, val in batch.items():
        if torch.is_tensor(val):
            out[key] = val[keep.to(device=val.device)]
        elif isinstance(val, list):
            out[key] = [val[i] for i in idx]
        else:
            out[key] = val
    return out


def compose_masked_completion(
    restored_base: torch.Tensor,
    sampled_x0: torch.Tensor,
    mask: torch.Tensor,
    *,
    target: str = "residual",
    residual_scale: float = 1.0,
    residual_clip: float | None = None,
    residual_blend: float = 1.0,
    valid_min: float = -1.0,
    valid_max: float = 1.0,
) -> torch.Tensor:
    """Compose a masked-completion DDPM sample with exact outside-mask preservation."""
    m = mask.to(device=restored_base.device, dtype=restored_base.dtype)
    if m.ndim == 3:
        m = m.unsqueeze(1)
    if m.shape[1] != 1:
        m = m[:, :1]
    if m.shape[-2:] != restored_base.shape[-2:]:
        m = F.interpolate(m, size=restored_base.shape[-2:], mode="bilinear", align_corners=False)
    m = m.clamp(0.0, 1.0).expand_as(restored_base)
    if target == "residual":
        residual = sampled_x0 * float(residual_scale)
        if residual_clip is not None:
            residual = residual.clamp(-float(residual_clip), float(residual_clip))
        inside = (restored_base + float(residual_blend) * residual).clamp(valid_min, valid_max)
    elif target == "clean":
        inside = sampled_x0.clamp(valid_min, valid_max)
    else:
        raise ValueError(f"unknown masked_completion_target {target!r}; expected residual or clean")
    return restored_base * (1.0 - m) + inside * m
