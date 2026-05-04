"""Dataset of clean Earth-imagery patches paired with synthetic degradations.

Reads `data/manifests/patches.jsonl`, loads RGB PNGs, normalizes to [-1, 1],
and creates a degraded version of each patch on-the-fly.

Returns dicts:
    {
        "clean":            tensor (3, H, W) in [-1, 1],
        "degraded":         tensor (3, H, W) in [-1, 1],
        "degradation_type": str,
        "path":             str,
    }

Synthetic remote-sensing-style degradations (proxy for InSAR-style noise):
    - speckle_multiplicative
    - blur                       (multilook-like smoothing)
    - downsample_upsample
    - blur_downsample_upsample
    - lowfreq_atmospheric_bias   (smooth low-frequency offset / haze)
    - mixed_structured           (combination of two of the above)
    - gaussian_additive          (minority fallback only)
    - mask_dropout               (smooth blob / rectangular missing area; RGB proxy)

Degradation strength ranges are configurable via a ``strengths`` dict
keyed by degradation type. The defaults match the original hardcoded
behaviour. Two named profiles are provided:

    * ``balanced`` (default) – the original probabilities and ranges.
    * ``hard``                – wider strength ranges, more weight on
                                structured + low-freq bias + mixed
                                degradations, and downsample biased toward 4x.

A profile resolves to ``(probs, strengths)`` defaults; user-supplied
``probs`` and ``strengths`` deep-merge on top of the profile defaults so
fine-grained overrides are possible from a YAML config.
"""

from __future__ import annotations

import json
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Synthetic degradations
# ---------------------------------------------------------------------------

DEGRADATION_TYPES = [
    "speckle_multiplicative",
    "blur",
    "downsample_upsample",
    "blur_downsample_upsample",
    "lowfreq_atmospheric_bias",
    "mixed_structured",
    "gaussian_additive",
    "mask_dropout",
]

DEFAULT_PROBS = {
    "speckle_multiplicative": 0.25,
    "blur": 0.15,
    "downsample_upsample": 0.15,
    "blur_downsample_upsample": 0.15,
    "lowfreq_atmospheric_bias": 0.15,
    "mixed_structured": 0.10,
    "gaussian_additive": 0.05,
    "mask_dropout": 0.0,
}

# Default strengths reproduce the original hardcoded ranges so any caller
# that does not pass ``strengths`` keeps existing behaviour exactly.
DEFAULT_STRENGTHS: dict[str, dict] = {
    "speckle_multiplicative":   {"sigma_min": 0.06, "sigma_max": 0.20},
    "gaussian_additive":        {"sigma_min": 0.04, "sigma_max": 0.18},
    "blur":                     {"sigma_min": 0.8,  "sigma_max": 2.0},
    "downsample_upsample":      {"scale_choices": [2, 4]},
    "blur_downsample_upsample": {
        "blur_sigma_min": 0.8, "blur_sigma_max": 1.6,
        "scale_choices": [2, 4],
    },
    "lowfreq_atmospheric_bias": {
        "amplitude_min": 0.05, "amplitude_max": 0.20,
    },
    "mixed_structured": {
        "blur_sigma_min": 0.8, "blur_sigma_max": 1.6,
        "scale_choices": [2, 4],
        "noise_sigma_min": 0.03, "noise_sigma_max": 0.10,
        "lowfreq_prob": 0.5,
    },
    "mask_dropout": {
        "n_masks_min": 1, "n_masks_max": 2,
        "size_min_frac": 0.08, "size_max_frac": 0.25,
        # one of "rect", "blob", "mixed" (random per sample)
        "shape": "mixed",
        # "zero" (replace with 0 ≈ mid-gray in [-1, 1]) or "noise"
        "fill": "zero",
        "feather_px": 6,
    },
}

# ------------------------------- profiles ----------------------------------
HARD_PROBS = {
    "speckle_multiplicative":   0.26,
    "blur":                     0.04,
    "downsample_upsample":      0.04,
    "blur_downsample_upsample": 0.18,
    "lowfreq_atmospheric_bias": 0.18,
    "mixed_structured":         0.20,
    "gaussian_additive":        0.03,
    "mask_dropout":             0.07,
}

HARD_STRENGTHS: dict[str, dict] = {
    "speckle_multiplicative":   {"sigma_min": 0.10, "sigma_max": 0.30},
    "gaussian_additive":        {"sigma_min": 0.06, "sigma_max": 0.25},
    "blur":                     {"sigma_min": 1.0,  "sigma_max": 3.0},
    # 4 appears 4x more often than 2.
    "downsample_upsample":      {"scale_choices": [2, 4, 4, 4, 4]},
    "blur_downsample_upsample": {
        "blur_sigma_min": 1.0, "blur_sigma_max": 3.0,
        "scale_choices": [2, 4, 4, 4, 4],
    },
    "lowfreq_atmospheric_bias": {
        "amplitude_min": 0.10, "amplitude_max": 0.35,
    },
    "mixed_structured": {
        "blur_sigma_min": 1.0, "blur_sigma_max": 3.0,
        "scale_choices": [2, 4, 4, 4, 4],
        "noise_sigma_min": 0.05, "noise_sigma_max": 0.15,
        "lowfreq_prob": 0.7,
    },
    "mask_dropout": {
        "n_masks_min": 1, "n_masks_max": 3,
        "size_min_frac": 0.10, "size_max_frac": 0.30,
        "shape": "mixed",
        "fill": "zero",
        "feather_px": 8,
    },
}

PROFILES: dict[str, dict] = {
    "balanced": {"probs": DEFAULT_PROBS, "strengths": DEFAULT_STRENGTHS},
    "hard":     {"probs": HARD_PROBS,    "strengths": HARD_STRENGTHS},
}

FLAT_STRENGTH_KEYS = {
    "speckle_sigma_min": ("speckle_multiplicative", "sigma_min"),
    "speckle_sigma_max": ("speckle_multiplicative", "sigma_max"),
    "gaussian_sigma_min": ("gaussian_additive", "sigma_min"),
    "gaussian_sigma_max": ("gaussian_additive", "sigma_max"),
    "blur_sigma_min": ("__all_blur__", "min"),
    "blur_sigma_max": ("__all_blur__", "max"),
    "lowfreq_amplitude_min": ("lowfreq_atmospheric_bias", "amplitude_min"),
    "lowfreq_amplitude_max": ("lowfreq_atmospheric_bias", "amplitude_max"),
    "downsample_scale_choices": ("__all_downsample__", "scale_choices"),
    "mixed_strength_min": ("mixed_structured", "noise_sigma_min"),
    "mixed_strength_max": ("mixed_structured", "noise_sigma_max"),
    "mask_dropout_probability": ("__prob__", "mask_dropout"),
    "mask_dropout_strength_min": ("mask_dropout", "size_min_frac"),
    "mask_dropout_strength_max": ("mask_dropout", "size_max_frac"),
}


def degradation_cfg_from_training_cfg(cfg: dict) -> dict:
    """Extract degradation settings from a training/eval YAML config."""
    out: dict = {
        "degradation_profile": cfg.get("degradation_profile", "balanced"),
        "degradation_probs": cfg.get("degradation_probs"),
        "degradation_strengths": cfg.get("degradation_strengths"),
    }
    for flat_key in FLAT_STRENGTH_KEYS:
        if flat_key in cfg:
            out[flat_key] = cfg[flat_key]
    return out


def _flat_strength_overrides(cfg: Optional[dict]) -> tuple[dict, dict[str, float]]:
    strengths: dict = {}
    probs: dict[str, float] = {}
    if not cfg:
        return strengths, probs
    for flat_key, (kind, key) in FLAT_STRENGTH_KEYS.items():
        if flat_key not in cfg:
            continue
        value = cfg[flat_key]
        if kind == "__prob__":
            probs[key] = float(value)
        elif kind == "__all_downsample__":
            for target in ("downsample_upsample", "blur_downsample_upsample", "mixed_structured"):
                strengths.setdefault(target, {})[key] = list(value)
        elif kind == "__all_blur__":
            blur_key = "sigma_min" if key == "min" else "sigma_max"
            structured_key = "blur_sigma_min" if key == "min" else "blur_sigma_max"
            strengths.setdefault("blur", {})[blur_key] = value
            for target in ("blur_downsample_upsample", "mixed_structured"):
                strengths.setdefault(target, {})[structured_key] = value
        else:
            strengths.setdefault(kind, {})[key] = value
    return strengths, probs


def _deep_merge(base: dict, override: Optional[dict]) -> dict:
    """Recursive merge: override sub-dicts on per-key basis. Values in
    ``override`` win for scalar / list keys."""
    if not override:
        return {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    out: dict = {}
    keys = set(base.keys()) | set(override.keys())
    for k in keys:
        bv = base.get(k)
        ov = override.get(k)
        if isinstance(bv, dict) and isinstance(ov, dict):
            out[k] = _deep_merge(bv, ov)
        elif ov is not None:
            out[k] = ov
        else:
            out[k] = dict(bv) if isinstance(bv, dict) else bv
    return out


def resolve_profile(
    name: Optional[str],
    *,
    probs_override: Optional[dict] = None,
    strengths_override: Optional[dict] = None,
) -> tuple[dict, dict]:
    """Return ``(probs, strengths)`` for the requested profile, with optional
    per-key overrides deep-merged on top."""
    if name is None:
        base_probs = dict(DEFAULT_PROBS)
        base_strengths = DEFAULT_STRENGTHS
    else:
        if name not in PROFILES:
            raise ValueError(
                f"unknown degradation_profile {name!r}; "
                f"choose one of {sorted(PROFILES.keys())}"
            )
        base_probs = dict(PROFILES[name]["probs"])
        base_strengths = PROFILES[name]["strengths"]
    probs = dict(base_probs)
    if probs_override:
        for k, v in probs_override.items():
            probs[_ALIASES.get(k, k)] = float(v)
    strengths = _deep_merge(base_strengths, strengths_override)
    _validate_strengths(strengths)
    return probs, strengths


# Backwards-compatible aliases for older configs.
_ALIASES = {
    "gaussian": "gaussian_additive",
    "speckle": "speckle_multiplicative",
    "blur_ds_us": "blur_downsample_upsample",
    "blur_ds_noise_us": "mixed_structured",
}


def _validate_strengths(strengths: dict) -> None:
    range_pairs = [
        ("speckle_multiplicative", "sigma_min", "sigma_max"),
        ("gaussian_additive", "sigma_min", "sigma_max"),
        ("blur", "sigma_min", "sigma_max"),
        ("blur_downsample_upsample", "blur_sigma_min", "blur_sigma_max"),
        ("lowfreq_atmospheric_bias", "amplitude_min", "amplitude_max"),
        ("mixed_structured", "blur_sigma_min", "blur_sigma_max"),
        ("mixed_structured", "noise_sigma_min", "noise_sigma_max"),
        ("mask_dropout", "size_min_frac", "size_max_frac"),
        ("mask_dropout", "n_masks_min", "n_masks_max"),
    ]
    for kind, lo_key, hi_key in range_pairs:
        if kind not in strengths:
            continue
        lo = strengths[kind].get(lo_key)
        hi = strengths[kind].get(hi_key)
        if lo is None or hi is None:
            continue
        if float(lo) < 0 or float(hi) < float(lo):
            raise ValueError(f"invalid degradation range for {kind}: {lo_key}={lo}, {hi_key}={hi}")

    for kind in ("downsample_upsample", "blur_downsample_upsample", "mixed_structured"):
        choices = strengths.get(kind, {}).get("scale_choices")
        if choices is None:
            continue
        if not choices:
            raise ValueError(f"{kind}.scale_choices must not be empty")
        if any(int(v) < 1 for v in choices):
            raise ValueError(f"{kind}.scale_choices must contain positive integers")
        strengths[kind]["scale_choices"] = [int(v) for v in choices]


def _gauss_kernel(sigma: float, ksize: int) -> torch.Tensor:
    coords = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    k2 = g[:, None] * g[None, :]
    return k2


def _gauss_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """x: (C, H, W) in [-1, 1]."""
    if sigma <= 0:
        return x
    ksize = max(3, int(sigma * 4) | 1)
    kernel = _gauss_kernel(sigma, ksize).to(x.device, x.dtype)
    c = x.shape[0]
    kernel = kernel.expand(c, 1, ksize, ksize).contiguous()
    return F.conv2d(x.unsqueeze(0), kernel, padding=ksize // 2, groups=c).squeeze(0)


def _down_up(x: torch.Tensor, scale: int) -> torch.Tensor:
    c, h, w = x.shape
    small = F.interpolate(
        x.unsqueeze(0), size=(h // scale, w // scale), mode="bilinear", align_corners=False
    )
    up = F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False)
    return up.squeeze(0).clamp(-1.0, 1.0)


def _lowfreq_bias(
    x: torch.Tensor,
    rng: random.Random,
    amp_min: float = 0.05,
    amp_max: float = 0.20,
) -> tuple[torch.Tensor, dict]:
    """Add a smooth low-frequency multi-channel bias (atmospheric / haze proxy).

    A small (e.g. 8x8) noise field per channel is upsampled bicubically to the
    full patch resolution, scaled, and added to the image. ``amp_min`` /
    ``amp_max`` control the per-channel amplitude range.
    """
    c, h, w = x.shape
    grid = max(4, min(16, h // 32))
    amp = rng.uniform(float(amp_min), float(amp_max))
    noise = torch.randn(1, c, grid, grid) * amp
    bias = F.interpolate(noise, size=(h, w), mode="bicubic", align_corners=False)
    out = (x + bias.squeeze(0).to(x.device, x.dtype)).clamp(-1.0, 1.0)
    return out, {"amplitude": float(amp), "grid": int(grid)}


def _make_blob_mask(
    h: int, w: int, cy: int, cx: int, ry: int, rx: int, feather_px: int
) -> torch.Tensor:
    """Smoothly-feathered ellipsoidal mask in [0, 1] of shape (H, W)."""
    yy = torch.arange(h, dtype=torch.float32).view(h, 1)
    xx = torch.arange(w, dtype=torch.float32).view(1, w)
    # normalized squared distance; <=1 inside ellipse.
    d = ((yy - cy) / max(1.0, float(ry))) ** 2 + ((xx - cx) / max(1.0, float(rx))) ** 2
    inside = (d <= 1.0).float()
    if feather_px > 0:
        # Gaussian blur the binary mask to feather the edge.
        mk = inside.unsqueeze(0).unsqueeze(0)
        sigma = max(1.0, float(feather_px) / 2.0)
        ksize = max(3, int(sigma * 4) | 1)
        kernel = _gauss_kernel(sigma, ksize).expand(1, 1, ksize, ksize).contiguous()
        mk = F.conv2d(mk, kernel, padding=ksize // 2)
        return mk.squeeze(0).squeeze(0).clamp(0.0, 1.0)
    return inside


def _make_rect_mask(
    h: int, w: int, cy: int, cx: int, ry: int, rx: int, feather_px: int
) -> torch.Tensor:
    """Feathered rectangular mask in [0, 1]."""
    mk = torch.zeros(h, w, dtype=torch.float32)
    y0 = max(0, cy - ry); y1 = min(h, cy + ry)
    x0 = max(0, cx - rx); x1 = min(w, cx + rx)
    mk[y0:y1, x0:x1] = 1.0
    if feather_px > 0:
        sigma = max(1.0, float(feather_px) / 2.0)
        ksize = max(3, int(sigma * 4) | 1)
        kernel = _gauss_kernel(sigma, ksize).expand(1, 1, ksize, ksize).contiguous()
        mk = F.conv2d(mk.unsqueeze(0).unsqueeze(0), kernel, padding=ksize // 2)
        return mk.squeeze(0).squeeze(0).clamp(0.0, 1.0)
    return mk


def _apply_mask_dropout(
    x: torch.Tensor, rng: random.Random, params: dict, *, return_mask: bool = False,
) -> tuple[torch.Tensor, dict] | tuple[torch.Tensor, dict, torch.Tensor]:
    """Drop / corrupt rectangular and/or blob regions of the image.

    RGB proxy only – this is not a true InSAR mask. Configurable via
    ``params``: ``n_masks_min/max``, ``size_min_frac/size_max_frac``,
    ``shape`` ('rect'|'blob'|'mixed'), ``fill`` ('zero'|'noise'),
    ``feather_px``.
    """
    c, h, w = x.shape
    n_min = int(params.get("n_masks_min", 1))
    n_max = int(params.get("n_masks_max", 2))
    size_min = float(params.get("size_min_frac", 0.08))
    size_max = float(params.get("size_max_frac", 0.25))
    shape = str(params.get("shape", "mixed"))
    fill = str(params.get("fill", "zero"))
    feather = int(params.get("feather_px", 6))
    n = rng.randint(max(1, n_min), max(n_min, n_max))
    out = x.clone()
    combined_mask = torch.zeros(h, w, dtype=x.dtype, device=x.device)
    masks_meta: list[dict] = []
    for _ in range(n):
        sz = rng.uniform(size_min, size_max)
        ry = max(2, int(round(sz * h * 0.5)))
        rx = max(2, int(round(sz * w * 0.5)))
        cy = rng.randint(ry, h - ry)
        cx = rng.randint(rx, w - rx)
        sh = shape if shape in ("rect", "blob") else rng.choice(["rect", "blob"])
        if sh == "blob":
            mk = _make_blob_mask(h, w, cy, cx, ry, rx, feather).to(x.device, x.dtype)
        else:
            mk = _make_rect_mask(h, w, cy, cx, ry, rx, feather).to(x.device, x.dtype)
        if fill == "noise":
            replacement = torch.randn_like(out) * 0.5
        else:
            replacement = torch.zeros_like(out)
        # Broadcast mask over channels.
        mk3 = mk.unsqueeze(0).expand_as(out)
        out = out * (1.0 - mk3) + replacement * mk3
        combined_mask = torch.maximum(combined_mask, mk)
        masks_meta.append({
            "shape": sh, "cy": int(cy), "cx": int(cx),
            "ry": int(ry), "rx": int(rx), "size_frac": float(sz),
        })
    out = out.clamp(-1.0, 1.0)
    meta = {
        "n_masks": int(n), "shape": shape, "fill": fill,
        "feather_px": int(feather), "masks": masks_meta,
    }
    if return_mask:
        return out, meta, combined_mask.unsqueeze(0).clamp(0.0, 1.0)
    return out, meta


def degrade(
    clean: torch.Tensor,
    kind: str,
    *,
    rng: Optional[random.Random] = None,
    strengths: Optional[dict[str, dict]] = None,
    return_mask: bool = False,
) -> tuple[torch.Tensor, dict] | tuple[torch.Tensor, dict, Optional[torch.Tensor]]:
    """Apply a synthetic degradation. clean is (3, H, W) in [-1, 1].

    ``strengths`` is a mapping ``kind -> {param: value}``. If absent, the
    range and choice defaults are taken from ``DEFAULT_STRENGTHS`` so existing
    callers keep their original behaviour.

    Returns:
        (degraded_tensor, params_dict)
        If ``return_mask=True``, returns ``(degraded_tensor, params_dict, mask)``
        where mask is ``[1,H,W]`` for spatial degradations or ``None``.

    params_dict is JSON-serializable (floats / ints / strings / lists). Names
    are stable per kind so callers can build per-type histograms.
    """
    rng = rng or random
    kind = _ALIASES.get(kind, kind)
    s_all = strengths or DEFAULT_STRENGTHS
    s = s_all.get(kind, DEFAULT_STRENGTHS.get(kind, {}))

    if kind == "gaussian_additive":
        sigma = rng.uniform(float(s.get("sigma_min", 0.04)),
                            float(s.get("sigma_max", 0.18)))
        out = (clean + torch.randn_like(clean) * sigma).clamp(-1.0, 1.0)
        result = (out, {"sigma": float(sigma)})
        return (*result, None) if return_mask else result

    if kind == "speckle_multiplicative":
        sigma = rng.uniform(float(s.get("sigma_min", 0.06)),
                            float(s.get("sigma_max", 0.20)))
        x01 = (clean + 1.0) * 0.5
        noise = torch.randn_like(x01) * sigma
        y01 = (x01 * (1.0 + noise)).clamp(0.0, 1.0)
        result = (y01 * 2.0 - 1.0, {"sigma": float(sigma)})
        return (*result, None) if return_mask else result

    if kind == "blur":
        bs = rng.uniform(float(s.get("sigma_min", 0.8)),
                         float(s.get("sigma_max", 2.0)))
        result = (_gauss_blur(clean, bs).clamp(-1.0, 1.0), {"blur_sigma": float(bs)})
        return (*result, None) if return_mask else result

    if kind == "downsample_upsample":
        choices = list(s.get("scale_choices", [2, 4]))
        scale = int(rng.choice(choices))
        result = (_down_up(clean, scale), {"scale": int(scale)})
        return (*result, None) if return_mask else result

    if kind == "blur_downsample_upsample":
        bs = rng.uniform(float(s.get("blur_sigma_min", 0.8)),
                         float(s.get("blur_sigma_max", 1.6)))
        choices = list(s.get("scale_choices", [2, 4]))
        scale = int(rng.choice(choices))
        result = (_down_up(_gauss_blur(clean, bs), scale), {
            "blur_sigma": float(bs), "scale": int(scale),
        })
        return (*result, None) if return_mask else result

    if kind == "lowfreq_atmospheric_bias":
        result = _lowfreq_bias(
            clean, rng,
            amp_min=float(s.get("amplitude_min", 0.05)),
            amp_max=float(s.get("amplitude_max", 0.20)),
        )
        return (*result, None) if return_mask else result

    if kind == "mixed_structured":
        bs = rng.uniform(float(s.get("blur_sigma_min", 0.8)),
                         float(s.get("blur_sigma_max", 1.6)))
        choices = list(s.get("scale_choices", [2, 4]))
        scale = int(rng.choice(choices))
        nsigma = rng.uniform(float(s.get("noise_sigma_min", 0.03)),
                             float(s.get("noise_sigma_max", 0.10)))
        c, h, w = clean.shape
        blurred = _gauss_blur(clean, bs).unsqueeze(0)
        small = F.interpolate(blurred, size=(h // scale, w // scale),
                              mode="bilinear", align_corners=False)
        small = small + torch.randn_like(small) * nsigma
        up = F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False)
        out = up.squeeze(0).clamp(-1.0, 1.0)
        params = {"blur_sigma": float(bs), "scale": int(scale),
                  "noise_sigma": float(nsigma), "with_lowfreq": False}
        if rng.random() < float(s.get("lowfreq_prob", 0.5)):
            lf_strengths = s_all.get("lowfreq_atmospheric_bias", {})
            out, lp = _lowfreq_bias(
                out, rng,
                amp_min=float(lf_strengths.get("amplitude_min", 0.05)),
                amp_max=float(lf_strengths.get("amplitude_max", 0.20)),
            )
            params["with_lowfreq"] = True
            params["lowfreq_amplitude"] = lp["amplitude"]
            params["lowfreq_grid"] = lp["grid"]
        result = (out, params)
        return (*result, None) if return_mask else result

    if kind == "mask_dropout":
        return _apply_mask_dropout(clean, rng, s, return_mask=return_mask)

    raise ValueError(f"unknown degradation kind: {kind}")


def _severity01(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return float(max(0.0, min(1.0, (float(value) - float(lo)) / (float(hi) - float(lo)))))


def _reliability_from_degradation(
    kind: str,
    params: dict,
    strengths: dict[str, dict],
    h: int,
    w: int,
    mask: Optional[torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (degradation_mask, reliability_map), both [1,H,W].

    The maps are RGB-proxy confidence metadata: no InSAR-specific physics or
    losses are encoded here.
    """
    kind = _ALIASES.get(kind, kind)
    if mask is not None:
        m = mask.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        if m.ndim == 2:
            m = m.unsqueeze(0)
        if m.shape[-2:] != (h, w):
            m = F.interpolate(m.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
        return m.clamp(0.0, 1.0), (1.0 - m).clamp(0.0, 1.0)

    s = strengths.get(kind, DEFAULT_STRENGTHS.get(kind, {}))
    degradation_mask = torch.zeros(1, h, w, dtype=dtype, device=device)
    reliability_value = 1.0

    if kind == "lowfreq_atmospheric_bias":
        sev = _severity01(
            float(params.get("amplitude", s.get("amplitude_min", 0.05))),
            float(s.get("amplitude_min", 0.05)),
            float(s.get("amplitude_max", 0.20)),
        )
        reliability_value = 0.85 - 0.45 * sev
        degradation_mask.fill_(1.0)
    elif kind == "blur_downsample_upsample":
        blur_sev = _severity01(
            float(params.get("blur_sigma", s.get("blur_sigma_min", 0.8))),
            float(s.get("blur_sigma_min", 0.8)),
            float(s.get("blur_sigma_max", 1.6)),
        )
        scale = float(params.get("scale", 1))
        scale_choices = [float(v) for v in s.get("scale_choices", [2, 4])]
        scale_sev = _severity01(scale, min(scale_choices), max(scale_choices))
        sev = 0.5 * blur_sev + 0.5 * scale_sev
        reliability_value = 0.88 - 0.38 * sev
        degradation_mask.fill_(1.0)
    elif kind == "mixed_structured":
        blur_sev = _severity01(
            float(params.get("blur_sigma", s.get("blur_sigma_min", 0.8))),
            float(s.get("blur_sigma_min", 0.8)),
            float(s.get("blur_sigma_max", 1.6)),
        )
        noise_sev = _severity01(
            float(params.get("noise_sigma", s.get("noise_sigma_min", 0.03))),
            float(s.get("noise_sigma_min", 0.03)),
            float(s.get("noise_sigma_max", 0.10)),
        )
        sev = 0.5 * blur_sev + 0.5 * noise_sev
        if bool(params.get("with_lowfreq", False)):
            sev = max(sev, 0.65)
        reliability_value = 0.86 - 0.42 * sev
        degradation_mask.fill_(1.0)
    elif kind in ("blur", "downsample_upsample"):
        reliability_value = 0.90
    elif kind in ("speckle_multiplicative", "gaussian_additive"):
        reliability_value = 0.95

    reliability = torch.full((1, h, w), float(reliability_value), dtype=dtype, device=device)
    return degradation_mask.clamp(0.0, 1.0), reliability.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PatchDegradationDataset(Dataset):
    """Yields (clean, degraded) pairs from `data/manifests/patches.jsonl`."""

    def __init__(
        self,
        manifest_path: str,
        patch_dir: str,
        probs: Optional[dict[str, float]] = None,
        max_samples: Optional[int] = None,
        deterministic: bool = False,
        seed: int = 0,
        image_size: Optional[int] = None,
        degradation_profile: Optional[str] = "balanced",
        strengths: Optional[dict[str, dict]] = None,
        flat_overrides: Optional[dict] = None,
        return_degradation_mask: bool = False,
        return_reliability_map: bool = False,
    ):
        """Yields (clean, degraded) pairs from a patches manifest.

        ``degradation_profile`` selects a named profile ('balanced' default,
        'hard' for stronger / harder distribution). ``probs`` and
        ``strengths`` may be passed to override profile defaults; both
        deep-merge on top.

        ``flat_overrides`` accepts a dict like
        ``{"speckle_sigma_max": 0.3, "downsample_scale_choices": [4, 4, 2]}``
        for ergonomic per-key overrides from a flat YAML config; see
        ``FLAT_STRENGTH_KEYS`` for the supported names.
        """
        self.patch_dir = patch_dir
        self.deterministic = deterministic
        self.seed = seed
        self.image_size = image_size
        self.degradation_profile = degradation_profile
        self.return_degradation_mask = bool(return_degradation_mask)
        self.return_reliability_map = bool(return_reliability_map)

        flat_strengths, flat_probs = _flat_strength_overrides(flat_overrides)
        merged_strengths = _deep_merge(strengths or {}, flat_strengths)

        # Resolve profile probs/strengths (with deep-merged user strengths).
        profile_probs, resolved_strengths = resolve_profile(
            degradation_profile,
            probs_override=None,
            strengths_override=merged_strengths or None,
        )
        # Explicit ``probs`` arg overrides profile probs entirely; flat_probs
        # may add/override individual entries on top of either source.
        raw_probs = dict(probs) if probs is not None else dict(profile_probs)
        for k, v in flat_probs.items():
            raw_probs[k] = float(v)
        # Resolve any legacy aliases in supplied probs.
        resolved: dict[str, float] = {}
        for k, v in raw_probs.items():
            resolved[_ALIASES.get(k, k)] = resolved.get(_ALIASES.get(k, k), 0.0) + float(v)
        # Drop zero-weight entries to keep self._types compact.
        resolved = {k: v for k, v in resolved.items() if v > 0.0}
        types = list(resolved.keys())
        weights = np.array([resolved[k] for k in types], dtype=np.float64)
        if weights.sum() <= 0:
            raise ValueError("degradation probs sum to 0")
        weights = weights / weights.sum()
        self.probs = resolved
        self.strengths = resolved_strengths
        self._types = types
        self._weights = weights

        records: list[dict] = []
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records.append(rec)
        if max_samples is not None:
            records = records[:max_samples]
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def _resolve(self, path: str) -> str:
        if os.path.isabs(path) and os.path.exists(path):
            return path
        if os.path.exists(path):
            return path
        return os.path.join(self.patch_dir, os.path.basename(path))

    def _load(self, path: str) -> torch.Tensor:
        full = self._resolve(path)
        with Image.open(full) as im:
            arr = np.array(im.convert("RGB"), dtype=np.uint8)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        t = t * 2.0 - 1.0  # [-1, 1]
        if self.image_size is not None and (t.shape[-1] != self.image_size or t.shape[-2] != self.image_size):
            t = F.interpolate(
                t.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return t

    def _pick_kind(self, idx: int) -> str:
        if self.deterministic:
            rng = random.Random(self.seed + idx)
            return rng.choices(self._types, weights=self._weights.tolist(), k=1)[0]
        return random.choices(self._types, weights=self._weights.tolist(), k=1)[0]

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        path = rec["patch_path"]
        clean = self._load(path)
        kind = self._pick_kind(idx)
        if self.deterministic:
            torch.manual_seed(self.seed + idx)
            rng = random.Random(self.seed + idx)
        else:
            rng = None
        need_metadata_maps = self.return_degradation_mask or self.return_reliability_map
        if need_metadata_maps:
            degraded, params, mask = degrade(
                clean, kind, rng=rng, strengths=self.strengths, return_mask=True,
            )
        else:
            degraded, params = degrade(clean, kind, rng=rng, strengths=self.strengths)
            mask = None
        sample = {
            "clean": clean,
            "degraded": degraded,
            "degradation_type": kind,
            "degradation_params": params,
            "path": path,
        }
        if need_metadata_maps:
            h, w = clean.shape[-2:]
            mask_map, reliability_map = _reliability_from_degradation(
                kind, params, self.strengths, h, w, mask,
                dtype=clean.dtype, device=clean.device,
            )
            if self.return_degradation_mask:
                sample["degradation_mask"] = mask_map.float().clamp(0.0, 1.0)
            if self.return_reliability_map:
                sample["reliability_map"] = reliability_map.float().clamp(0.0, 1.0)
        return sample


def patch_degradation_collate(batch: list[dict]) -> dict:
    """Collate patch/degradation samples without merging param dictionaries.

    Each degradation type has its own parameter schema, so keep params as a
    per-sample list instead of using PyTorch's default nested-dict collation.
    """
    out = {
        "clean": torch.stack([sample["clean"] for sample in batch], dim=0),
        "degraded": torch.stack([sample["degraded"] for sample in batch], dim=0),
        "degradation_type": [str(sample["degradation_type"]) for sample in batch],
        "degradation_params": [sample.get("degradation_params", {}) for sample in batch],
        "path": [str(sample["path"]) for sample in batch],
    }
    if any("degradation_mask" in sample for sample in batch):
        masks = []
        for sample in batch:
            if "degradation_mask" in sample:
                masks.append(sample["degradation_mask"].float())
            else:
                h, w = sample["clean"].shape[-2:]
                masks.append(torch.zeros(1, h, w, dtype=torch.float32))
        out["degradation_mask"] = torch.stack(masks, dim=0)
    if any("reliability_map" in sample for sample in batch):
        maps = []
        for sample in batch:
            if "reliability_map" in sample:
                maps.append(sample["reliability_map"].float())
            else:
                h, w = sample["clean"].shape[-2:]
                maps.append(torch.ones(1, h, w, dtype=torch.float32))
        out["reliability_map"] = torch.stack(maps, dim=0)
    return out
