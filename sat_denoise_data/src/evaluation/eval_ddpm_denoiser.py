"""Evaluate the conditional DDPM restoration model.

Two evaluation modes are supported, selected by the checkpoint's saved config
(``diffusion_target``):

* ``clean``: classic conditional DDPM. Saves 4-column grids
  (degraded | DDPM output | clean | abs error) and reports degraded vs clean
  and DDPM-output vs clean.
* ``residual``: residual-DDPM refinement on top of a frozen residual U-Net.
  Saves 5-column grids (degraded | U-Net base | DDPM refined | clean |
  abs error) and reports three metric sets: degraded vs clean, U-Net base vs
  clean, and DDPM-refined vs clean for each value in ``--eval_blends``.
  blend=0.0 must reproduce the U-Net base output exactly (verified at
  startup on a small batch).

For residual mode, the base U-Net checkpoint must be supplied via
``--base_ckpt`` or recoverable from the saved config's ``base_ckpt`` field.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

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
from src.utils.checkpoint import load_checkpoint
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
)
from src.utils.residual_targets import detail_metric_values, residual_target_config
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--patch_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--sampling_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--base_ckpt",
        default=None,
        help="Frozen residual-U-Net checkpoint. Required for residual-mode "
             "DDPM evaluation (overrides the path stored in the DDPM ckpt).",
    )
    ap.add_argument(
        "--sampling_mode",
        choices=["ddpm", "ddim", "mean"],
        default="ddpm",
        help="Sampler. 'ddpm' is full ancestral; 'ddim' is deterministic "
             "(eta=0); 'mean' returns the posterior mean (no per-step noise).",
    )
    ap.add_argument(
        "--residual_scale",
        default=None,
        help="Override residual_scale. Float; defaults to value stored in "
             "the DDPM checkpoint config.",
    )
    ap.add_argument(
        "--residual_clip",
        type=float,
        default=None,
        help="Override residual_clip used at inference (residual mode).",
    )
    ap.add_argument(
        "--residual_blend",
        type=float,
        default=None,
        help="Override the *default* (visual-grid) blend factor.",
    )
    ap.add_argument(
        "--eval_blends",
        default=None,
        help="Comma-separated blend values for the sweep, e.g. '0.0,0.1,0.25,0.5,1.0'.",
    )
    ap.add_argument(
        "--num_samples_per_input",
        type=int,
        default=1,
        help="Generate K diffusion samples per input for uncertainty estimates. Default: 1.",
    )
    return ap.parse_args()


def _build_degradation_kwargs(cfg: dict) -> dict:
    flat = {k: cfg[k] for k in FLAT_STRENGTH_KEYS if k in cfg}
    out = {}
    if cfg.get("degradation_profile") is not None:
        out["degradation_profile"] = str(cfg["degradation_profile"])
    if cfg.get("degradation_strengths") is not None:
        out["strengths"] = cfg["degradation_strengths"]
    if flat:
        out["flat_overrides"] = flat
    return out


def _filtered_degradation_probs(cfg: dict, allowed: list[str] | tuple[str, ...] | None) -> dict[str, float] | None:
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
    if not filtered:
        raise ValueError(f"DDPM eval degradation filter has no positive-probability modes: {sorted(allowed_set)}")
    return filtered


def to01(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1.0) * 0.5).clamp(0.0, 1.0)


def _avg(d: dict[str, list[float]]) -> dict[str, float]:
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in d.items()}


def _region_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float] | None:
    mask = mask.to(pred.device, pred.dtype)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if float(mask.sum().item()) <= 0.0:
        return None
    pred01 = to01(pred)
    target01 = to01(target)
    weight = mask.expand_as(pred01)
    denom = weight.sum().clamp_min(1e-8)
    mse = (((pred01 - target01) ** 2) * weight).sum() / denom
    mae = ((pred01 - target01).abs() * weight).sum() / denom
    psnr = 99.0 if float(mse.item()) <= 1e-12 else float(10.0 * torch.log10(1.0 / mse).item())
    return {"psnr": psnr, "mae": float(mae.item())}


def _region_mean(x: torch.Tensor, mask: torch.Tensor) -> float | None:
    mask = mask.to(x.device, x.dtype)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1 and x.shape[1] != 1:
        mask = mask.expand_as(x)
    denom = mask.sum()
    if float(denom.item()) <= 0.0:
        return None
    return float((x * mask).sum().item() / denom.clamp_min(1e-8).item())


def _load_base_unet(base_ckpt: str, device: torch.device) -> ResidualUNet:
    state = load_checkpoint(base_ckpt, map_location=str(device))
    bc = state.get("config", {}) or {}
    base = ResidualUNet(
        base_channels=int(bc.get("base_channels", 32)),
        channel_mults=tuple(bc.get("channel_mults", [1, 2, 4, 4])),
    ).to(device)
    base.load_state_dict(state["model"])
    base.eval()
    for p in base.parameters():
        p.requires_grad = False
    return base


def _parse_blends(spec: Optional[str], default: list[float]) -> list[float]:
    if spec is None:
        return list(default)
    out: list[float] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def _resolve_residual_scale(cfg_value, override) -> float:
    if override is not None:
        return float(override)
    if cfg_value is None:
        return 1.0
    if isinstance(cfg_value, str):
        if cfg_value.lower() == "auto":
            # Should never happen: training resolves 'auto' before saving.
            # Fall back to a conservative 0.05 and log a warning.
            print("[eval-ddpm] WARNING: cfg.residual_scale='auto' not resolved; using 0.05")
            return 0.05
        return float(cfg_value)
    return float(cfg_value)


def _ddpm_cond_channels(cfg: dict) -> int:
    if str(cfg.get("diffusion_target", "clean")) == "clean":
        return 3
    extra = int(bool(cfg.get("ddpm_condition_use_mask", False)))
    extra += int(bool(cfg.get("ddpm_condition_use_reliability", False)))
    return 6 + extra


def _conditioning_extras(
    batch: dict,
    cfg: dict,
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
        val = batch.get(key)
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


def _sample_residual_x0(
    diffusion: GaussianDiffusion,
    model,
    cond: torch.Tensor,
    shape: tuple[int, int, int, int],
    *,
    sampling_mode: str,
    sampling_steps: int,
) -> torch.Tensor:
    if sampling_mode == "ddim":
        return diffusion.ddim_sample_loop(
            model, cond, shape=shape, sampling_steps=sampling_steps,
            progress=True, x0_clip_min=-1.0, x0_clip_max=1.0,
        )
    if sampling_mode == "mean":
        return diffusion.p_sample_loop(
            model, cond, shape=shape, sampling_steps=sampling_steps,
            mean_only=True, progress=True,
            x0_clip_min=-1.0, x0_clip_max=1.0,
        )
    return diffusion.p_sample_loop(
        model, cond, shape=shape, sampling_steps=sampling_steps,
        progress=True, x0_clip_min=-1.0, x0_clip_max=1.0,
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_checkpoint(args.ckpt, map_location=str(device))
    cfg = state.get("config", {})
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
    cfg.setdefault("mask_area_filter_enabled", False)
    cfg.setdefault("mask_area_min", 0.0)
    cfg.setdefault("mask_area_max", 1.0)

    mode = str(cfg.get("diffusion_target", "clean"))
    if mode not in ("clean", "residual", "masked_completion"):
        raise ValueError(f"unknown diffusion_target {mode!r} in checkpoint config")

    diffusion = GaussianDiffusion(
        timesteps=int(cfg.get("timesteps", 1000)),
        schedule=str(cfg.get("beta_schedule", "cosine")),
    ).to(device)

    cond_channels = _ddpm_cond_channels(cfg)
    model = DDPMUNet(
        in_channels=3,
        cond_channels=cond_channels,
        out_channels=3,
        base_channels=int(cfg.get("base_channels", 32)),
        channel_mults=tuple(cfg.get("channel_mults", [1, 2, 4, 4])),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    base_unet: ResidualUNet | None = None
    residual_scale = 1.0
    residual_clip = float(cfg.get("residual_clip", 0.10))
    if args.residual_clip is not None:
        residual_clip = float(args.residual_clip)
    default_blend = float(cfg.get("residual_blend", 0.25))
    if args.residual_blend is not None:
        default_blend = float(args.residual_blend)
    eval_blends = _parse_blends(args.eval_blends, cfg.get("eval_blends", [0.0, default_blend]))
    # Always ensure the configured default blend is in the sweep so the
    # visual grid has a candidate to show.
    if default_blend not in eval_blends:
        eval_blends = sorted(set(eval_blends + [default_blend]))

    if mode in ("residual", "masked_completion"):
        base_ckpt = args.base_ckpt or cfg.get("base_ckpt")
        if not base_ckpt:
            raise ValueError(
                f"{mode}-mode DDPM requires --base_ckpt or cfg.base_ckpt in the DDPM checkpoint"
            )
        base_unet = _load_base_unet(base_ckpt, device)
        residual_scale = _resolve_residual_scale(cfg.get("residual_scale"), args.residual_scale)
        print(f"[eval-ddpm] {mode} mode: loaded frozen base U-Net from {base_ckpt}")
        if mode == "residual":
            rc = residual_target_config(cfg)
            print(f"[eval-ddpm] residual_target_type={rc['residual_target_type']}  "
                  f"detail_sigma={float(rc['detail_sigma']):.2f}  "
                  f"lowpass_sigma={float(rc['lowpass_sigma']):.2f}  "
                  f"lowpass_weight={float(rc['lowpass_weight']):.2f}  "
                  f"highpass_weight={float(rc['highpass_weight']):.2f}  "
                  f"cond_ch={cond_channels}")
            print(f"[eval-ddpm] residual_scale={residual_scale:.5f}  "
                  f"residual_clip={residual_clip:.4f}  "
                  f"default_residual_blend={default_blend:.3f}  "
                  f"eval_blends={eval_blends}  "
                  f"sampling_mode={args.sampling_mode}  "
                  f"gate_mode={cfg.get('residual_gate_mode', 'none')}")
        else:
            print(f"[eval-ddpm] masked_completion_target={cfg.get('masked_completion_target', 'residual')}  "
                  f"residual_scale={residual_scale:.5f} residual_clip={residual_clip:.4f}  "
                  f"sampling_mode={args.sampling_mode} num_samples_per_input={args.num_samples_per_input}  "
                  f"cond_ch={cond_channels}")

    eval_filter = cfg.get("ddpm_val_degradation_types", cfg.get("ddpm_train_degradation_types"))
    eval_probs = _filtered_degradation_probs(cfg, eval_filter)
    if eval_filter:
        print(f"[eval-ddpm] degradation filter={list(eval_filter)} probs={eval_probs}")
    full = PatchDegradationDataset(
        manifest_path=args.manifest,
        patch_dir=args.patch_dir,
        probs=eval_probs if eval_probs is not None else cfg.get("degradation_probs"),
        **_build_degradation_kwargs(cfg),
        max_samples=args.max_samples,
        deterministic=True,
        seed=args.seed,
        image_size=cfg.get("image_size"),
        return_degradation_mask=bool(cfg.get("return_degradation_mask", False)),
        return_reliability_map=bool(cfg.get("return_reliability_map", False)),
    )
    n = len(full)
    print(f"[eval-ddpm] device={device}  mode={mode}  samples={n}  ckpt={args.ckpt}  "
          f"sampling_steps={args.sampling_steps}")
    loader = DataLoader(
        full, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=patch_degradation_collate,
    )

    # Per-sample metric accumulators.
    overall_in: dict[str, list[float]] = defaultdict(list)
    overall_base: dict[str, list[float]] = defaultdict(list)  # residual/masked mode only
    by_type_in: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_type_base: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    # For clean mode: the single "out" set.
    # For residual mode: one "out" set per blend in eval_blends.
    overall_out_per_blend: dict[float, dict[str, list[float]]] = {
        bl: defaultdict(list) for bl in eval_blends
    } if mode == "residual" else {1.0: defaultdict(list)}
    by_type_out_per_blend: dict[float, dict[str, dict[str, list[float]]]] = {
        bl: defaultdict(lambda: defaultdict(list)) for bl in eval_blends
    } if mode == "residual" else {1.0: defaultdict(lambda: defaultdict(list))}
    overall_ungated_per_blend: dict[float, dict[str, list[float]]] = {
        bl: defaultdict(list) for bl in eval_blends
    } if mode == "residual" else {}
    by_type_ungated_per_blend: dict[float, dict[str, dict[str, list[float]]]] = {
        bl: defaultdict(lambda: defaultdict(list)) for bl in eval_blends
    } if mode == "residual" else {}
    gated_region_per_blend: dict[float, dict[str, list[float]]] = {
        bl: defaultdict(list) for bl in eval_blends
    } if mode == "residual" else {}
    outside_gate_per_blend: dict[float, dict[str, list[float]]] = {
        bl: defaultdict(list) for bl in eval_blends
    } if mode == "residual" else {}
    by_type_gated_region_per_blend: dict[float, dict[str, dict[str, list[float]]]] = {
        bl: defaultdict(lambda: defaultdict(list)) for bl in eval_blends
    } if mode == "residual" else {}
    by_type_outside_gate_per_blend: dict[float, dict[str, dict[str, list[float]]]] = {
        bl: defaultdict(lambda: defaultdict(list)) for bl in eval_blends
    } if mode == "residual" else {}

    grids: list[str] = []
    per_sample: list[dict] = []

    blend_zero_max_diff = 0.0  # residual mode: must be ~0
    outside_gate_max_diff = 0.0
    gate_area_fracs: list[float] = []
    nonzero_gate_samples = 0
    detail_base: dict[str, list[float]] = defaultdict(list)
    detail_refined_per_blend: dict[float, dict[str, list[float]]] = {
        bl: defaultdict(list) for bl in eval_blends
    } if mode == "residual" else {}
    masked_region: dict[str, list[float]] = defaultdict(list)
    outside_mask: dict[str, list[float]] = defaultdict(list)
    mask_area_fracs: list[float] = []
    mask_area_by_type: dict[str, list[float]] = defaultdict(list)
    uncertainty_stats: dict[str, list[float]] = defaultdict(list)
    mask_filter_seen = 0
    mask_filter_kept = 0
    mask_filter_skipped_batches = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if mode == "masked_completion":
                mask0 = build_completion_mask(
                    batch, device, int(batch["clean"].shape[-2]), int(batch["clean"].shape[-1])
                )
                keep, area0 = localized_mask_keep(mask0, cfg)
                mask_filter_seen += int(keep.numel())
                mask_filter_kept += int(keep.sum().item())
                if bool(cfg.get("mask_area_filter_enabled", False)) and not bool(keep.all().item()):
                    print(
                        f"[eval-ddpm] mask_area_filter kept={int(keep.sum().item())}/{int(keep.numel())} "
                        f"area_min={float(area0.min().item()):.4f} "
                        f"area_mean={float(area0.mean().item()):.4f} area_max={float(area0.max().item()):.4f}"
                    )
                if not bool(keep.any().item()):
                    mask_filter_skipped_batches += 1
                    print("[eval-ddpm] skipping batch: no masks passed mask_area_filter")
                    continue
                batch = filter_batch_by_indices(batch, keep)
            clean = batch["clean"].to(device)
            deg = batch["degraded"].to(device)
            b = clean.shape[0]

            if mode == "residual":
                restored_base = base_unet(deg).clamp(-1.0, 1.0)
                cond = torch.cat([
                    deg,
                    restored_base,
                    *_conditioning_extras(batch, cfg, device, b=b, h=int(deg.shape[-2]), w=int(deg.shape[-1])),
                ], dim=1)
                sampled_x0 = _sample_residual_x0(
                    diffusion, model, cond, deg.shape,
                    sampling_mode=args.sampling_mode,
                    sampling_steps=args.sampling_steps,
                )
                sampled_residual = (sampled_x0 * residual_scale).clamp(
                    -residual_clip, residual_clip
                )
                gate = build_residual_gate(batch, cfg, device)
                outs_per_blend: dict[float, torch.Tensor] = {}
                ungated_per_blend: dict[float, torch.Tensor] = {}
                for bl in eval_blends:
                    ungated = (restored_base + float(bl) * sampled_residual).clamp(-1.0, 1.0)
                    ungated_per_blend[bl] = ungated
                    outs_per_blend[bl] = compose_gated_residual(
                        restored_base, sampled_residual, gate, float(bl),
                    )
                for k, v in detail_metric_values(restored_base, clean, cfg).items():
                    detail_base[k].append(v)
                for bl, out in outs_per_blend.items():
                    for k, v in detail_metric_values(out, clean, cfg).items():
                        detail_refined_per_blend[bl][k].append(v)
                # Verify blend=0 reproduces the U-Net base exactly.
                if 0.0 in outs_per_blend:
                    diff = (outs_per_blend[0.0] - restored_base).abs().max().item()
                    blend_zero_max_diff = max(blend_zero_max_diff, diff)
                outside = (gate <= 1e-6).to(restored_base.dtype).expand_as(restored_base)
                if float(outside.sum().item()) > 0.0:
                    outside_diff = ((outs_per_blend[default_blend] - restored_base).abs() * outside).max().item()
                    outside_gate_max_diff = max(outside_gate_max_diff, outside_diff)
                gate_area = gate.flatten(1).mean(dim=1)
                gate_area_fracs.extend([float(v) for v in gate_area.detach().cpu()])
                nonzero_gate_samples += int((gate.flatten(1).amax(dim=1) > 0).sum().item())
            elif mode == "masked_completion":
                restored_base = base_unet(deg).clamp(-1.0, 1.0)
                cond = torch.cat([
                    deg,
                    restored_base,
                    *_conditioning_extras(batch, cfg, device, b=b, h=int(deg.shape[-2]), w=int(deg.shape[-1])),
                ], dim=1)
                mask = build_completion_mask(batch, device, int(deg.shape[-2]), int(deg.shape[-1]))
                mask_area = mask.flatten(1).mean(dim=1)
                mask_area_fracs.extend([float(v) for v in mask_area.detach().cpu()])
                for kind, area_val in zip(batch["degradation_type"], mask_area.detach().cpu().tolist()):
                    mask_area_by_type[str(kind)].append(float(area_val))
                target = str(cfg.get("masked_completion_target", "residual"))
                sample_count = max(1, int(args.num_samples_per_input))
                final_samples: list[torch.Tensor] = []
                for _ in range(sample_count):
                    sampled_x0 = _sample_residual_x0(
                        diffusion, model, cond, deg.shape,
                        sampling_mode=args.sampling_mode,
                        sampling_steps=args.sampling_steps,
                    )
                    final_samples.append(compose_masked_completion(
                        restored_base, sampled_x0, mask,
                        target=target,
                        residual_scale=residual_scale,
                        residual_clip=residual_clip,
                        residual_blend=default_blend,
                    ))
                sample_stack = torch.stack(final_samples, dim=0)
                default_out = sample_stack.mean(dim=0)
                uncertainty = sample_stack.var(dim=0, unbiased=False).mean(dim=1, keepdim=True)
                if sample_count > 1:
                    save_image(
                        make_grid(to01(default_out), nrow=b),
                        os.path.join(args.output_dir, f"grid_{bi:03d}_sample_mean.png"),
                    )
                    uncertainty_vis = uncertainty / uncertainty.amax().clamp_min(1e-8)
                    save_image(
                        make_grid(uncertainty_vis.expand(-1, 3, -1, -1).clamp(0, 1), nrow=b),
                        os.path.join(args.output_dir, f"grid_{bi:03d}_sample_variance.png"),
                    )
                outs_per_blend = {1.0: default_out}
                ungated_per_blend = {}
                gate = mask

                outside = (1.0 - mask).to(restored_base.dtype).expand_as(restored_base)
                if float(outside.sum().item()) > 0.0:
                    diff = (default_out - restored_base).abs() * outside
                    outside_mask_max = float(diff.max().item())
                    outside_mask["outside_mask_max_diff"].append(outside_mask_max)
                    outside_mask["outside_mask_mae_to_base"].append(
                        float(diff.sum().item() / outside.sum().clamp_min(1e-8).item())
                    )
                    outside_gate_max_diff = max(outside_gate_max_diff, outside_mask_max)
                inside_metrics = _region_metrics(default_out, clean, mask)
                if inside_metrics is not None:
                    for k, v in inside_metrics.items():
                        masked_region[f"inside_mask_{k}"].append(v)
                outside_metrics = _region_metrics(default_out, clean, 1.0 - mask)
                if outside_metrics is not None:
                    for k, v in outside_metrics.items():
                        outside_mask[f"outside_mask_{k}"].append(v)
                inside_unc = _region_mean(uncertainty, mask)
                outside_unc = _region_mean(uncertainty, 1.0 - mask)
                if inside_unc is not None:
                    uncertainty_stats["mean_uncertainty_inside_mask"].append(inside_unc)
                if outside_unc is not None:
                    uncertainty_stats["mean_uncertainty_outside_mask"].append(outside_unc)
            else:
                restored_base = None
                gate = None
                ungated_per_blend = {}
                outs_per_blend = {
                    1.0: _sample_residual_x0(
                        diffusion, model, deg, deg.shape,
                        sampling_mode=args.sampling_mode,
                        sampling_steps=args.sampling_steps,
                    )
                }

            # Per-sample metrics.
            for i in range(b):
                kind = batch["degradation_type"][i]
                m_in = all_metrics(deg[i:i + 1], clean[i:i + 1])
                for k in ("psnr", "mae", "ssim"):
                    overall_in[k].append(m_in[k])
                    by_type_in[kind][k].append(m_in[k])
                row_rec = {
                    "path": batch["path"][i],
                    "degradation_type": kind,
                    "psnr_in": m_in["psnr"],
                    "mae_in": m_in["mae"],
                    "ssim_in": m_in["ssim"],
                }
                if mode in ("residual", "masked_completion"):
                    m_base = all_metrics(restored_base[i:i + 1], clean[i:i + 1])
                    for k in ("psnr", "mae", "ssim"):
                        overall_base[k].append(m_base[k])
                        by_type_base[kind][k].append(m_base[k])
                    row_rec.update({
                        "psnr_base": m_base["psnr"],
                        "mae_base": m_base["mae"],
                        "ssim_base": m_base["ssim"],
                    })
                for bl, out in outs_per_blend.items():
                    m_out = all_metrics(out[i:i + 1], clean[i:i + 1])
                    for k in ("psnr", "mae", "ssim"):
                        overall_out_per_blend[bl][k].append(m_out[k])
                        by_type_out_per_blend[bl][kind][k].append(m_out[k])
                    suffix = f"blend{bl:g}" if mode == "residual" else "out"
                    prefix = f"gated_{suffix}" if mode == "residual" else suffix
                    row_rec[f"psnr_{prefix}"] = m_out["psnr"]
                    row_rec[f"mae_{prefix}"] = m_out["mae"]
                    row_rec[f"ssim_{prefix}"] = m_out["ssim"]
                    if mode == "residual":
                        m_ungated = all_metrics(ungated_per_blend[bl][i:i + 1], clean[i:i + 1])
                        for k in ("psnr", "mae", "ssim"):
                            overall_ungated_per_blend[bl][k].append(m_ungated[k])
                            by_type_ungated_per_blend[bl][kind][k].append(m_ungated[k])
                        row_rec[f"psnr_ungated_{suffix}"] = m_ungated["psnr"]
                        row_rec[f"mae_ungated_{suffix}"] = m_ungated["mae"]
                        row_rec[f"ssim_ungated_{suffix}"] = m_ungated["ssim"]
                        rg = _region_metrics(out[i:i + 1], clean[i:i + 1], gate[i:i + 1])
                        if rg is not None:
                            for k, v in rg.items():
                                gated_region_per_blend[bl][k].append(v)
                                by_type_gated_region_per_blend[bl][kind][k].append(v)
                                row_rec[f"{k}_gate_region_{suffix}"] = v
                        ro = _region_metrics(out[i:i + 1], clean[i:i + 1], 1.0 - gate[i:i + 1])
                        if ro is not None:
                            for k, v in ro.items():
                                outside_gate_per_blend[bl][k].append(v)
                                by_type_outside_gate_per_blend[bl][kind][k].append(v)
                                row_rec[f"{k}_outside_gate_{suffix}"] = v
                    elif mode == "masked_completion":
                        rg = _region_metrics(out[i:i + 1], clean[i:i + 1], gate[i:i + 1])
                        if rg is not None:
                            for k, v in rg.items():
                                row_rec[f"inside_mask_{k}"] = v
                        ro = _region_metrics(out[i:i + 1], clean[i:i + 1], 1.0 - gate[i:i + 1])
                        if ro is not None:
                            for k, v in ro.items():
                                row_rec[f"outside_mask_{k}"] = v
                        outside_i = (1.0 - gate[i:i + 1]).to(out.dtype).expand_as(out[i:i + 1])
                        if float(outside_i.sum().item()) > 0.0:
                            diff_i = (out[i:i + 1] - restored_base[i:i + 1]).abs() * outside_i
                            row_rec["outside_mask_max_diff"] = float(diff_i.max().item())
                            row_rec["outside_mask_mae_to_base"] = float(
                                diff_i.sum().item() / outside_i.sum().clamp_min(1e-8).item()
                            )
                            row_rec["mask_area_fraction"] = float(gate[i:i + 1].flatten(1).mean().item())
                per_sample.append(row_rec)

            sample_labels = [
                f"{batch['degradation_type'][i]} | {os.path.basename(batch['path'][i])}"
                for i in range(b)
            ]
            if mode == "masked_completion":
                out = outs_per_blend[1.0]
                err = (out - clean).abs()
                err_vis = (err / (err.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)).clamp(0, 1)
                mask_area_batch = float(gate.flatten(1).mean().item())
                row = torch.cat([
                    to01(deg),
                    gate.expand(-1, 3, -1, -1).clamp(0, 1),
                    to01(restored_base),
                    to01(out),
                    to01(clean),
                    err_vis,
                ], dim=0)
                grid_path = os.path.join(args.output_dir, f"grid_{bi:03d}.png")
                save_image(make_grid(row, nrow=b), grid_path)
                grids.append(grid_path)
                title = (
                    f"Masked-completion DDPM eval batch {bi:03d} | "
                    f"diffusion_target=masked_completion "
                    f"masked_completion_target={cfg.get('masked_completion_target', 'residual')} "
                    f"sampling_mode={args.sampling_mode} sampling_steps={args.sampling_steps} "
                    f"mask_area_fraction={mask_area_batch:.4f} residual_scale={residual_scale:.4f}"
                )
                save_labeled_masked_completion_grid(
                    os.path.join(args.output_dir, f"grid_{bi:03d}_labeled.png"),
                    deg, gate, restored_base, out, clean,
                    title=title, sample_labels=sample_labels,
                    reliability=batch.get("reliability_map"),
                    uncertainty=uncertainty if int(args.num_samples_per_input) > 1 else None,
                )
            elif mode == "residual":
                # Default visual grid uses the configured default_blend.
                default_out = outs_per_blend[default_blend]
                err = (default_out - clean).abs()
                err_vis = (err / (err.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)).clamp(0, 1)
                gate_vis = gate.expand(-1, 3, -1, -1).clamp(0, 1)
                row = torch.cat([to01(deg), to01(restored_base), gate_vis, to01(default_out), to01(clean), err_vis], dim=0)
                grid_path = os.path.join(args.output_dir, f"grid_{bi:03d}.png")
                save_image(make_grid(row, nrow=b), grid_path)
                grids.append(grid_path)
                title = (
                    f"Residual-DDPM eval batch {bi:03d} | "
                    f"sampling_mode={args.sampling_mode} "
                    f"residual_target_type={cfg.get('residual_target_type', 'pixel')} "
                    f"detail_sigma={float(cfg.get('detail_sigma', 1.0)):.2f} "
                    f"lowpass_sigma={float(cfg.get('lowpass_sigma', 4.0)):.2f} "
                    f"residual_scale={residual_scale:.4f} residual_clip={residual_clip:.3f} "
                    f"residual_blend={default_blend:.2f} blend={default_blend:.2f} "
                    f"residual_gate_mode={cfg.get('residual_gate_mode', 'none')}"
                )
                if str(cfg.get("residual_gate_mode", "none")) == "none":
                    save_labeled_residual_refinement_grid(
                        os.path.join(args.output_dir, f"grid_{bi:03d}_labeled.png"),
                        deg, restored_base, default_out, clean,
                        title=title, sample_labels=sample_labels,
                    )
                else:
                    save_labeled_gated_residual_refinement_grid(
                        os.path.join(args.output_dir, f"grid_{bi:03d}_labeled.png"),
                        deg, restored_base, gate, default_out, clean,
                        title=title, sample_labels=sample_labels,
                        reliability=batch.get("reliability_map"),
                    )
            else:
                out = outs_per_blend[1.0]
                err = (out - clean).abs()
                err_vis = (err / (err.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)).clamp(0, 1)
                row = torch.cat([to01(deg), to01(out), to01(clean), err_vis], dim=0)
                grid = make_grid(row, nrow=b)
                grid_path = os.path.join(args.output_dir, f"grid_{bi:03d}.png")
                save_image(grid, grid_path)
                grids.append(grid_path)
                save_labeled_restoration_grid(
                    os.path.join(args.output_dir, f"grid_{bi:03d}_labeled.png"),
                    deg, out, clean,
                    output_label="DDPM output",
                    title=(f"DDPM evaluation batch {bi:03d} | "
                           f"mode={args.sampling_mode}"),
                    sample_labels=sample_labels,
                )

    # ----- aggregate -----
    summary_in = _avg(overall_in)
    summary: dict[str, object] = {
        "mode": mode,
        "count": len(per_sample),
        "sampling_mode": args.sampling_mode,
        "psnr_in": summary_in.get("psnr", 0.0),
        "mae_in": summary_in.get("mae", 0.0),
        "ssim_in": summary_in.get("ssim", 0.0),
    }
    if mode in ("residual", "masked_completion"):
        summary_base = _avg(overall_base)
        summary.update({
            "psnr_base": summary_base.get("psnr", 0.0),
            "mae_base": summary_base.get("mae", 0.0),
            "ssim_base": summary_base.get("ssim", 0.0),
        })
    if mode == "masked_completion":
        masked_avg = _avg(masked_region)
        outside_avg = _avg(outside_mask)
        uncertainty_avg = _avg(uncertainty_stats)
        summary.update({
            "masked_completion_target": cfg.get("masked_completion_target", "residual"),
            "residual_scale": residual_scale,
            "residual_clip": residual_clip,
            "default_residual_blend": default_blend,
            "mean_mask_area_fraction": sum(mask_area_fracs) / len(mask_area_fracs) if mask_area_fracs else 0.0,
            "mask_area_min": min(mask_area_fracs) if mask_area_fracs else 0.0,
            "mask_area_max": max(mask_area_fracs) if mask_area_fracs else 0.0,
            "mask_area_filter_enabled": bool(cfg.get("mask_area_filter_enabled", False)),
            "mask_area_filter_pass_fraction": mask_filter_kept / max(1, mask_filter_seen),
            "mask_area_filter_skipped_batches": mask_filter_skipped_batches,
            "inside_mask": masked_avg,
            "outside_mask": outside_avg,
            "outside_mask_max_diff": max(outside_mask.get("outside_mask_max_diff", [0.0])),
            "outside_mask_mae_to_base": outside_avg.get("outside_mask_mae_to_base", 0.0),
            "num_samples_per_input": max(1, int(args.num_samples_per_input)),
            "uncertainty": {
                "mean_uncertainty_inside_mask": uncertainty_avg.get("mean_uncertainty_inside_mask", 0.0),
                "mean_uncertainty_outside_mask": uncertainty_avg.get("mean_uncertainty_outside_mask", 0.0),
            },
        })
    if mode == "residual":
        summary.update({
            "residual_scale": residual_scale,
            "residual_clip": residual_clip,
            "residual_target_type": cfg.get("residual_target_type", "pixel"),
            "detail_sigma": float(cfg.get("detail_sigma", 1.0)),
            "lowpass_sigma": float(cfg.get("lowpass_sigma", 4.0)),
            "lowpass_weight": float(cfg.get("lowpass_weight", 1.0)),
            "highpass_weight": float(cfg.get("highpass_weight", 1.0)),
            "default_residual_blend": default_blend,
            "eval_blends": eval_blends,
            "blend_zero_max_diff_vs_base": blend_zero_max_diff,
            "residual_gate_mode": cfg.get("residual_gate_mode", "none"),
            "outside_gate_max_diff": outside_gate_max_diff,
            "mean_gate_area_fraction": sum(gate_area_fracs) / len(gate_area_fracs) if gate_area_fracs else 0.0,
            "nonzero_gate_sample_fraction": nonzero_gate_samples / max(1, len(gate_area_fracs)),
            "percent_samples_with_nonzero_gate": 100.0 * nonzero_gate_samples / max(1, len(gate_area_fracs)),
            "nonzero_gate_samples": nonzero_gate_samples,
        })
        for metric_name in ("highpass_mae", "lowpass_mae", "gradient_mae"):
            summary[f"{metric_name}_base"] = (
                sum(detail_base[metric_name]) / len(detail_base[metric_name])
                if detail_base[metric_name] else 0.0
            )

    # Per-blend summary.
    per_blend_summary: dict[str, dict] = {}
    best_psnr = (-1.0, float("-inf"))
    best_mae = (-1.0, float("inf"))
    for bl, acc in overall_out_per_blend.items():
        a = _avg(acc)
        rec = {
            "psnr_out": a.get("psnr", 0.0),
            "mae_out": a.get("mae", 0.0),
            "ssim_out": a.get("ssim", 0.0),
        }
        if mode == "residual":
            rec["delta_psnr_over_base"] = rec["psnr_out"] - summary["psnr_base"]
            rec["delta_mae_over_base"] = summary["mae_base"] - rec["mae_out"]
            rec["delta_ssim_over_base"] = rec["ssim_out"] - summary["ssim_base"]
            ungated = _avg(overall_ungated_per_blend[bl])
            rec["ungated_psnr_out"] = ungated.get("psnr", 0.0)
            rec["ungated_mae_out"] = ungated.get("mae", 0.0)
            rec["ungated_ssim_out"] = ungated.get("ssim", 0.0)
            rec["gate_region"] = _avg(gated_region_per_blend[bl])
            rec["outside_gate"] = _avg(outside_gate_per_blend[bl])
            for metric_name in ("highpass_mae", "lowpass_mae", "gradient_mae"):
                vals = detail_refined_per_blend[bl][metric_name]
                rec[f"{metric_name}_refined"] = sum(vals) / len(vals) if vals else 0.0
                rec[f"{metric_name}_base"] = summary.get(f"{metric_name}_base", 0.0)
            if rec["psnr_out"] > best_psnr[1]:
                best_psnr = (bl, rec["psnr_out"])
            if rec["mae_out"] < best_mae[1]:
                best_mae = (bl, rec["mae_out"])
        per_blend_summary[f"{bl:g}"] = rec
    summary["per_blend"] = per_blend_summary
    if mode == "residual":
        summary["best_blend_by_psnr"] = best_psnr[0]
        summary["best_blend_by_mae"] = best_mae[0]

    # Per-type breakdown (always per-blend in residual; single set in clean).
    by_type: dict[str, dict] = {}
    for kind in by_type_in:
        a = _avg(by_type_in[kind])
        rec = {
            "count": len(by_type_in[kind]["psnr"]),
            "psnr_in": a["psnr"], "mae_in": a["mae"], "ssim_in": a["ssim"],
        }
        if mode in ("residual", "masked_completion") and kind in by_type_base:
            c = _avg(by_type_base[kind])
            rec.update({
                "psnr_base": c["psnr"], "mae_base": c["mae"], "ssim_base": c["ssim"],
            })
        if mode == "masked_completion" and kind in mask_area_by_type:
            vals = mask_area_by_type[kind]
            rec["mask_area_fraction"] = {
                "min": min(vals),
                "mean": sum(vals) / len(vals),
                "max": max(vals),
            }
        per_blend_kind: dict[str, dict] = {}
        for bl, acc in by_type_out_per_blend.items():
            if kind not in acc:
                continue
            b = _avg(acc[kind])
            sub = {
                "psnr_out": b["psnr"], "mae_out": b["mae"], "ssim_out": b["ssim"],
                "delta_psnr": b["psnr"] - a["psnr"],
                "delta_mae": a["mae"] - b["mae"],
                "delta_ssim": b["ssim"] - a["ssim"],
            }
            if mode == "residual" and "psnr_base" in rec:
                sub["delta_psnr_over_base"] = b["psnr"] - rec["psnr_base"]
                sub["delta_mae_over_base"] = rec["mae_base"] - b["mae"]
                sub["delta_ssim_over_base"] = b["ssim"] - rec["ssim_base"]
                if kind in by_type_ungated_per_blend.get(bl, {}):
                    u = _avg(by_type_ungated_per_blend[bl][kind])
                    sub["ungated_psnr_out"] = u.get("psnr", 0.0)
                    sub["ungated_mae_out"] = u.get("mae", 0.0)
                    sub["ungated_ssim_out"] = u.get("ssim", 0.0)
                if kind in by_type_gated_region_per_blend.get(bl, {}):
                    sub["gate_region"] = _avg(by_type_gated_region_per_blend[bl][kind])
                if kind in by_type_outside_gate_per_blend.get(bl, {}):
                    sub["outside_gate"] = _avg(by_type_outside_gate_per_blend[bl][kind])
            elif mode == "masked_completion" and "psnr_base" in rec:
                sub["delta_psnr_over_base"] = b["psnr"] - rec["psnr_base"]
                sub["delta_mae_over_base"] = rec["mae_base"] - b["mae"]
                sub["delta_ssim_over_base"] = b["ssim"] - rec["ssim_base"]
            per_blend_kind[f"{bl:g}"] = sub
        rec["per_blend"] = per_blend_kind
        by_type[kind] = rec

    out_json = os.path.join(args.output_dir, "metrics.json")
    with open(out_json, "w") as f:
        json.dump({
            "summary": summary,
            "by_degradation_type": by_type,
            "grids": grids,
            "per_sample": per_sample,
        }, f, indent=2)

    grouped_json = os.path.join(args.output_dir, "grouped_metrics_by_degradation_type.json")
    with open(grouped_json, "w") as f:
        json.dump(by_type, f, indent=2)

    if mode == "masked_completion":
        clean_rec = per_blend_summary[next(iter(per_blend_summary))]
        print(
            f"[eval-ddpm] PSNR in/base/final: {summary['psnr_in']:.2f} / "
            f"{summary['psnr_base']:.2f} / {clean_rec['psnr_out']:.2f}  "
            f"MAE in/base/final: {summary['mae_in']:.4f} / "
            f"{summary['mae_base']:.4f} / {clean_rec['mae_out']:.4f}"
        )
        inside = summary.get("inside_mask", {})
        outside = summary.get("outside_mask", {})
        unc = summary.get("uncertainty", {})
        print(
            f"[eval-ddpm] mask_area_fraction={summary.get('mean_mask_area_fraction', 0.0):.4f}  "
            f"min={summary.get('mask_area_min', 0.0):.4f} max={summary.get('mask_area_max', 0.0):.4f}  "
            f"filter_pass={summary.get('mask_area_filter_pass_fraction', 0.0):.4f}  "
            f"inside_mask_mae={inside.get('inside_mask_mae', 0.0):.4f}  "
            f"inside_mask_psnr={inside.get('inside_mask_psnr', 0.0):.2f}  "
            f"outside_mask_max_diff={summary.get('outside_mask_max_diff', 0.0):.8f}  "
            f"outside_mask_mae_to_base={summary.get('outside_mask_mae_to_base', 0.0):.8f}"
        )
        print(
            f"[eval-ddpm] uncertainty inside/outside: "
            f"{unc.get('mean_uncertainty_inside_mask', 0.0):.8f} / "
            f"{unc.get('mean_uncertainty_outside_mask', 0.0):.8f}"
        )
        print("[eval-ddpm] by degradation_type (PSNR in -> base -> final):")
        for kind, m in sorted(by_type.items()):
            sub = next(iter(m.get("per_blend", {1: {}}).values()), {})
            print(
                f"  {kind:28s} n={m['count']:3d}  "
                f"in={m['psnr_in']:.2f}  base={m.get('psnr_base', 0.0):.2f}  "
                f"final={sub.get('psnr_out', 0.0):.2f}  "
                f"mask_area={m.get('mask_area_fraction', {}).get('mean', 0.0):.4f}"
            )
    elif mode == "residual":
        print(
            f"[eval-ddpm] PSNR in/base: {summary['psnr_in']:.2f} / "
            f"{summary['psnr_base']:.2f}  "
            f"MAE in/base: {summary['mae_in']:.4f} / "
            f"{summary['mae_base']:.4f}  "
            f"SSIM in/base: {summary['ssim_in']:.4f} / "
            f"{summary['ssim_base']:.4f}"
        )
        print(f"[eval-ddpm] blend=0 max |out - base| = {blend_zero_max_diff:.6f} "
              f"(must be ~0; verifies sampling pipeline is conservative)")
        print(f"[eval-ddpm] outside_gate_max_diff={outside_gate_max_diff:.8f}  "
              f"mean_gate_area_fraction={summary['mean_gate_area_fraction']:.4f}  "
              f"nonzero_gate_samples={nonzero_gate_samples}/{max(1, len(gate_area_fracs))}")
        print(f"[eval-ddpm] detail base: "
              f"highpass_mae={summary.get('highpass_mae_base', 0.0):.5f}  "
              f"lowpass_mae={summary.get('lowpass_mae_base', 0.0):.5f}  "
              f"gradient_mae={summary.get('gradient_mae_base', 0.0):.5f}")
        for bl, rec in per_blend_summary.items():
            print(
                f"[eval-ddpm] gated blend={float(bl):.3f}  "
                f"PSNR={rec['psnr_out']:.2f} (Δ_base={rec.get('delta_psnr_over_base', 0):+.2f})  "
                f"MAE={rec['mae_out']:.4f} (Δ_base={rec.get('delta_mae_over_base', 0):+.4f})  "
                f"SSIM={rec['ssim_out']:.4f}  "
                f"ungated_PSNR={rec.get('ungated_psnr_out', 0.0):.2f}  "
                f"hp_mae={rec.get('highpass_mae_refined', 0.0):.5f}  "
                f"lp_mae={rec.get('lowpass_mae_refined', 0.0):.5f}"
            )
        print(f"[eval-ddpm] best blend by PSNR: {summary['best_blend_by_psnr']}  "
              f"by MAE: {summary['best_blend_by_mae']}")
        print("[eval-ddpm] by degradation_type (PSNR base -> per-blend):")
        for kind, m in sorted(by_type.items()):
            base_psnr = m.get("psnr_base", float("nan"))
            blend_strs = "  ".join(
                f"b{bl}:{rec['psnr_out']:.2f}" for bl, rec in m.get("per_blend", {}).items()
            )
            print(
                f"  {kind:28s} n={m['count']:3d}  "
                f"in={m['psnr_in']:.2f}  base={base_psnr:.2f}  {blend_strs}"
            )
    else:
        clean_rec = per_blend_summary[next(iter(per_blend_summary))]
        print(f"[eval-ddpm] PSNR in/out: {summary['psnr_in']:.2f} / "
              f"{clean_rec['psnr_out']:.2f}")
        print(f"[eval-ddpm] MAE  in/out: {summary['mae_in']:.4f} / "
              f"{clean_rec['mae_out']:.4f}")
        print(f"[eval-ddpm] SSIM in/out: {summary['ssim_in']:.4f} / "
              f"{clean_rec['ssim_out']:.4f}")
        print("[eval-ddpm] by degradation_type:")
        for kind, m in sorted(by_type.items()):
            sub = next(iter(m.get("per_blend", {1: {}}).values()), {})
            print(f"  {kind:28s} n={m['count']:3d}  "
                  f"PSNR {m['psnr_in']:.2f}->{sub.get('psnr_out', 0.0):.2f}  "
                  f"MAE {m['mae_in']:.4f}->{sub.get('mae_out', 0.0):.4f}  "
                  f"SSIM {m['ssim_in']:.4f}->{sub.get('ssim_out', 0.0):.4f}")
    print(f"[eval-ddpm] wrote {len(grids)} grids and {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
