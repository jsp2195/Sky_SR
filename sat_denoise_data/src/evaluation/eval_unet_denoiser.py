"""Evaluate the deterministic residual U-Net denoiser.

Loads a checkpoint, runs deterministic restoration on a subset of patches,
saves PNG grids (degraded / restored / clean / |error|) and a metrics JSON
with metrics aggregated overall and grouped by degradation_type.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.patch_degradation_dataset import (
    FLAT_STRENGTH_KEYS,
    PatchDegradationDataset,
    patch_degradation_collate,
)
from src.models.residual_unet import ResidualUNet
from src.utils.checkpoint import load_checkpoint
from src.utils.diagnostic_grids import save_labeled_restoration_grid
from src.utils.metrics import all_metrics
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--patch_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def to01(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1.0) * 0.5).clamp(0.0, 1.0)


def _avg(d: dict[str, list[float]]) -> dict[str, float]:
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in d.items()}


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


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_checkpoint(args.ckpt, map_location=str(device))
    cfg = state.get("config", {})
    model = ResidualUNet(
        base_channels=int(cfg.get("base_channels", 64)),
        channel_mults=tuple(cfg.get("channel_mults", [1, 2, 4, 4])),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    full = PatchDegradationDataset(
        manifest_path=args.manifest,
        patch_dir=args.patch_dir,
        probs=cfg.get("degradation_probs"),
        **_build_degradation_kwargs(cfg),
        max_samples=args.max_samples,
        deterministic=True,
        seed=args.seed,
        image_size=cfg.get("image_size"),
    )
    n = len(full)
    print(f"[eval] device={device}  samples={n}  ckpt={args.ckpt}")
    loader = DataLoader(
        full, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=patch_degradation_collate,
    )

    overall_in: dict[str, list[float]] = defaultdict(list)
    overall_out: dict[str, list[float]] = defaultdict(list)
    by_type_in: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_type_out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    grids: list[str] = []
    per_sample: list[dict] = []

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            clean = batch["clean"].to(device)
            deg = batch["degraded"].to(device)
            pred = model(deg).clamp(-1, 1)
            err = (pred - clean).abs()
            err_vis = (err / (err.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)).clamp(0, 1)

            b = clean.shape[0]
            for i in range(b):
                kind = batch["degradation_type"][i]
                m_in = all_metrics(deg[i:i + 1], clean[i:i + 1])
                m_out = all_metrics(pred[i:i + 1], clean[i:i + 1])
                for k in ("psnr", "mae", "ssim"):
                    overall_in[k].append(m_in[k])
                    overall_out[k].append(m_out[k])
                    by_type_in[kind][k].append(m_in[k])
                    by_type_out[kind][k].append(m_out[k])
                per_sample.append({
                    "path": batch["path"][i],
                    "degradation_type": kind,
                    "psnr_in": m_in["psnr"], "psnr_out": m_out["psnr"],
                    "mae_in": m_in["mae"], "mae_out": m_out["mae"],
                    "ssim_in": m_in["ssim"], "ssim_out": m_out["ssim"],
                })

            row = torch.cat([to01(deg), to01(pred), to01(clean), err_vis], dim=0)
            grid = make_grid(row, nrow=b)
            grid_path = os.path.join(args.output_dir, f"grid_{bi:03d}.png")
            save_image(grid, grid_path)
            grids.append(grid_path)
            sample_labels = [
                f"{batch['degradation_type'][i]} | {os.path.basename(batch['path'][i])}"
                for i in range(b)
            ]
            labeled_grid_path = os.path.join(args.output_dir, f"grid_{bi:03d}_labeled.png")
            save_labeled_restoration_grid(
                labeled_grid_path,
                deg,
                pred,
                clean,
                output_label="restored output",
                title=f"U-Net evaluation batch {bi:03d}",
                sample_labels=sample_labels,
            )

    summary_in = _avg(overall_in)
    summary_out = _avg(overall_out)
    summary = {
        "count": len(per_sample),
        "psnr_in": summary_in.get("psnr", 0.0),
        "psnr_out": summary_out.get("psnr", 0.0),
        "mae_in": summary_in.get("mae", 0.0),
        "mae_out": summary_out.get("mae", 0.0),
        "ssim_in": summary_in.get("ssim", 0.0),
        "ssim_out": summary_out.get("ssim", 0.0),
    }
    summary["delta_psnr"] = summary["psnr_out"] - summary["psnr_in"]
    summary["delta_mae"] = summary["mae_in"] - summary["mae_out"]
    summary["delta_ssim"] = summary["ssim_out"] - summary["ssim_in"]

    by_type: dict[str, dict[str, float]] = {}
    for kind in by_type_in:
        a = _avg(by_type_in[kind])
        b = _avg(by_type_out[kind])
        by_type[kind] = {
            "count": len(by_type_in[kind]["psnr"]),
            "psnr_in": a["psnr"], "psnr_out": b["psnr"], "delta_psnr": b["psnr"] - a["psnr"],
            "mae_in": a["mae"], "mae_out": b["mae"], "delta_mae": a["mae"] - b["mae"],
            "ssim_in": a["ssim"], "ssim_out": b["ssim"], "delta_ssim": b["ssim"] - a["ssim"],
        }

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

    print(f"[eval] PSNR in/out: {summary['psnr_in']:.2f} / {summary['psnr_out']:.2f}  "
          f"(Δ={summary['delta_psnr']:+.2f} dB)")
    print(f"[eval] MAE  in/out: {summary['mae_in']:.4f} / {summary['mae_out']:.4f}")
    print(f"[eval] SSIM in/out: {summary['ssim_in']:.4f} / {summary['ssim_out']:.4f}")
    print("[eval] by degradation_type:")
    for kind, m in sorted(by_type.items()):
        print(f"  {kind:28s} n={m['count']:3d}  "
              f"PSNR {m['psnr_in']:.2f}->{m['psnr_out']:.2f}  "
              f"MAE {m['mae_in']:.4f}->{m['mae_out']:.4f}  "
              f"SSIM {m['ssim_in']:.4f}->{m['ssim_out']:.4f}")
    print(f"[eval] wrote {len(grids)} grids and {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
