"""Write a small localized masked-completion dataset preview grid."""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml
from torchvision.utils import make_grid, save_image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.patch_degradation_dataset import FLAT_STRENGTH_KEYS, PatchDegradationDataset
from src.training.train_ddpm_denoiser import _filtered_degradation_probs
from src.utils.residual_gating import build_completion_mask, localized_mask_keep


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_ddpm_masked_completion_localized.yaml")
    ap.add_argument("--output", default="outputs/previews/masked_completion_localized_preview.png")
    ap.add_argument("--num_samples", type=int, default=4)
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


def to01(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1.0) * 0.5).clamp(0.0, 1.0)


def main() -> int:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    probs = _filtered_degradation_probs(cfg, cfg.get("ddpm_train_degradation_types"))
    ds = PatchDegradationDataset(
        manifest_path=cfg["manifest"],
        patch_dir=cfg["patch_dir"],
        probs=probs if probs is not None else cfg.get("degradation_probs"),
        max_samples=max(int(cfg.get("max_samples", 128)), int(args.num_samples) * 4),
        image_size=cfg.get("image_size", 256),
        deterministic=True,
        seed=int(cfg.get("seed", 42)),
        return_degradation_mask=True,
        return_reliability_map=True,
        **_build_degradation_kwargs(cfg),
    )
    rows = []
    areas = []
    for i in range(len(ds)):
        s = ds[i]
        batch = {
            "clean": s["clean"].unsqueeze(0),
            "degraded": s["degraded"].unsqueeze(0),
            "degradation_mask": s["degradation_mask"].unsqueeze(0),
            "reliability_map": s["reliability_map"].unsqueeze(0),
            "degradation_type": [s["degradation_type"]],
        }
        mask = build_completion_mask(batch, torch.device("cpu"), int(s["clean"].shape[-2]), int(s["clean"].shape[-1]))
        keep, area = localized_mask_keep(mask, cfg)
        if not bool(keep.item()):
            continue
        mask3 = mask[0].expand(3, -1, -1)
        reliability3 = s["reliability_map"].clamp(0, 1).expand(3, -1, -1)
        rows.extend([to01(s["degraded"]), mask3, reliability3, to01(s["clean"])])
        areas.append(float(area.item()))
        if len(areas) >= int(args.num_samples):
            break
    if not rows:
        raise RuntimeError("no localized masks passed mask_area_filter for preview")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_image(make_grid(torch.stack(rows, dim=0), nrow=4), args.output)
    print(
        f"[preview] wrote {args.output} samples={len(areas)} "
        f"mask_area_min={min(areas):.4f} mean={sum(areas)/len(areas):.4f} max={max(areas):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
