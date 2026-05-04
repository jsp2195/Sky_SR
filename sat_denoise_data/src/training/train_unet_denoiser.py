"""Train the deterministic residual U-Net denoiser.

Usage:
    python -m src.training.train_unet_denoiser --config configs/train_unet_denoiser.yaml

CLI flags override the YAML config when provided.

Diagnostics produced per run:
    outputs/<run>/ckpt_last.pt
    outputs/<run>/ckpt_best.pt
    outputs/<run>/train_log.jsonl
    outputs/<run>/samples/epoch_<NNNN>.png    (fixed val samples per epoch)
    outputs/<run>/samples/step_<NNNNNN>.png   (rolling training samples)
    outputs/<run>/plots/loss_curve.png
    outputs/<run>/plots/metric_curves.png
    outputs/<run>/plots/degradation_type_distribution.png
    outputs/<run>/plots/degradation_param_histograms.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.patch_degradation_dataset import (
    FLAT_STRENGTH_KEYS,
    PatchDegradationDataset,
    patch_degradation_collate,
)
from src.losses.restoration_losses import RestorationLoss
from src.models.residual_unet import ResidualUNet
from src.utils.checkpoint import is_better, save_checkpoint
from src.utils.diagnostic_grids import save_labeled_restoration_grid
from src.utils.metrics import all_metrics
from src.utils.seed import set_seed
from src.utils.training_plots import write_all_plots


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
    return ap.parse_args()


def load_cfg(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    for k in [
        "manifest", "patch_dir", "output_dir",
        "batch_size", "epochs", "lr",
        "num_workers", "max_samples", "val_fraction", "seed",
    ]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    return cfg


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


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction)))
    return idx[n_val:].tolist(), idx[:n_val].tolist()


def to01(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1.0) * 0.5).clamp(0.0, 1.0)


def evaluate(model, loader, device, loss_fn) -> dict[str, float]:
    model.eval()
    sums = {"loss": 0.0, "psnr_in": 0.0, "psnr_out": 0.0,
            "mae_in": 0.0, "mae_out": 0.0, "ssim_in": 0.0, "ssim_out": 0.0}
    n = 0
    with torch.no_grad():
        for batch in loader:
            clean = batch["clean"].to(device)
            deg = batch["degraded"].to(device)
            pred = model(deg)
            total, _ = loss_fn(pred, clean, residual=pred - deg)
            m_in = all_metrics(deg, clean)
            m_out = all_metrics(pred, clean)
            b = clean.shape[0]
            sums["loss"] += float(total) * b
            sums["psnr_in"] += m_in["psnr"] * b
            sums["psnr_out"] += m_out["psnr"] * b
            sums["mae_in"] += m_in["mae"] * b
            sums["mae_out"] += m_out["mae"] * b
            sums["ssim_in"] += m_in["ssim"] * b
            sums["ssim_out"] += m_out["ssim"] * b
            n += b
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


def collect_fixed_val_samples(dataset, indices: list[int], device: torch.device):
    """Load fixed validation samples once and cache tensors/metadata.

    The dataset view is deterministic, so degradation type and noise are
    generated once here and never resampled for epoch visualizations.
    """
    cleans, degs, kinds, params, paths = [], [], [], [], []
    for idx in indices:
        s = dataset[idx]
        cleans.append(s["clean"])
        degs.append(s["degraded"])
        kinds.append(s["degradation_type"])
        params.append(s.get("degradation_params", {}))
        paths.append(s.get("path", ""))
    return (
        torch.stack(cleans, dim=0).to(device),
        torch.stack(degs, dim=0).to(device),
        kinds,
        params,
        paths,
    )


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


@torch.no_grad()
def save_unet_labeled_grid(
    path: str,
    model: torch.nn.Module,
    degraded: torch.Tensor,
    clean: torch.Tensor,
    *,
    title: str,
    labels: list[str],
) -> None:
    model.eval()
    pred = model(degraded).clamp(-1, 1)
    save_labeled_restoration_grid(
        path,
        degraded,
        pred,
        clean,
        output_label="restored",
        title=title,
        sample_labels=labels,
    )


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
    full = PatchDegradationDataset(
        manifest_path=cfg["manifest"],
        patch_dir=cfg["patch_dir"],
        probs=cfg.get("degradation_probs"),
        max_samples=cfg.get("max_samples"),
        image_size=cfg.get("image_size"),
        **deg_kwargs,
    )
    if len(full) == 0:
        print("[train] empty dataset"); return 1
    print(f"[train] degradation_profile={full.degradation_profile or 'balanced'}  "
          f"types={list(full.probs.keys())}")

    train_idx, val_idx = split_indices(len(full), float(cfg["val_fraction"]), int(cfg["seed"]))
    train_set = Subset(full, train_idx)
    val_set = Subset(full, val_idx)

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

    # Fixed validation samples for cross-epoch comparison.
    n_fixed = int(cfg.get("fixed_val_samples", 8))
    # Use a deterministic-degradation view of the same val records.
    fixed_view = PatchDegradationDataset(
        manifest_path=cfg["manifest"],
        patch_dir=cfg["patch_dir"],
        probs=cfg.get("degradation_probs"),
        max_samples=cfg.get("max_samples"),
        image_size=cfg.get("image_size"),
        deterministic=True,
        seed=int(cfg["seed"]),
        **deg_kwargs,
    )
    fixed_indices = val_idx[: max(1, min(n_fixed, len(val_idx)))]
    fixed_clean, fixed_deg, fixed_kinds, fixed_params, fixed_paths = collect_fixed_val_samples(
        fixed_view, fixed_indices, device,
    )
    fixed_labels = sample_labels(fixed_indices, fixed_kinds, fixed_paths)
    print(f"[train] fixed val samples cached: {fixed_clean.shape[0]} kinds={fixed_kinds}")

    model = ResidualUNet(
        base_channels=int(cfg.get("base_channels", 32)),
        channel_mults=tuple(cfg.get("channel_mults", [1, 2, 4, 4])),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] device={device}  params={n_params:,}  train={len(train_set)}  val={len(val_set)}")

    lcfg = cfg.get("loss", {}) or {}
    loss_fn = RestorationLoss(
        w_l1=float(lcfg.get("lambda_l1", lcfg.get("w_l1", 1.0))),
        w_ssim=float(lcfg.get("lambda_ssim", lcfg.get("w_ssim", 0.0))),
        w_edge=float(lcfg.get("lambda_grad", lcfg.get("w_edge", 0.25))),
        w_tv=float(lcfg.get("lambda_tv", lcfg.get("w_tv", 0.0))),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))

    sample_every = int(cfg.get("sample_every_steps", 100))
    save_step_train_batch_samples = bool(cfg.get("save_step_train_batch_samples", True))
    save_step_fixed_val_samples = bool(cfg.get("save_step_fixed_val_samples", False))
    log_f = open(log_path, "w")
    best_val: float | None = None
    step = 0
    t0 = time.time()

    type_counter: Counter = Counter()
    param_history: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for epoch in range(int(cfg["epochs"])):
        model.train()
        epoch_train_losses: list[float] = []
        for batch in train_loader:
            clean = batch["clean"].to(device)
            deg = batch["degraded"].to(device)
            pred = model(deg)
            residual = pred - deg
            total, parts = loss_fn(pred, clean, residual=residual)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_train_losses.append(float(parts["total"]))

            # Record degradation distribution from this batch.
            kinds = batch.get("degradation_type", [])
            for k in kinds:
                type_counter[str(k)] += 1
            params_field = batch.get("degradation_params", [])
            if isinstance(params_field, list):
                for kind, params in zip(kinds, params_field):
                    if not isinstance(params, dict):
                        continue
                    for pname, v in params.items():
                        if isinstance(v, (int, float)):
                            param_history[str(kind)][str(pname)].append(float(v))

            if step % 20 == 0:
                rec = {"step": step, "epoch": epoch, "phase": "train",
                       "train_loss": float(parts["total"]),
                       "loss": float(parts["total"]),
                       "l1": parts["l1"], "ssim_loss": parts.get("ssim_loss", 0.0),
                       "edge": parts.get("edge", 0.0), "tv": parts.get("tv", 0.0),
                       "elapsed_s": round(time.time() - t0, 1)}
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                print(f"[train] ep{epoch} step{step} loss={parts['total']:.4f} l1={parts['l1']:.4f}")

            if sample_every > 0 and step % sample_every == 0:
                if save_step_train_batch_samples:
                    save_labeled_restoration_grid(
                        os.path.join(sample_dir, f"train_batch_step_{step:06d}_labeled.png"),
                        deg,
                        pred.clamp(-1, 1),
                        clean,
                        output_label="restored",
                        title=f"U-Net train batch samples | step {step:06d}",
                        sample_labels=batch_sample_labels(
                            list(batch.get("degradation_type", [])),
                            list(batch.get("path", [])),
                        ),
                    )
                if save_step_fixed_val_samples:
                    save_unet_labeled_grid(
                        os.path.join(sample_dir, f"fixed_val_step_{step:06d}_labeled.png"),
                        model,
                        fixed_deg,
                        fixed_clean,
                        title=f"U-Net fixed validation samples | step {step:06d}",
                        labels=fixed_labels,
                    )
                model.train()
            step += 1

        # End-of-epoch validation.
        val_metrics = evaluate(model, val_loader, device, loss_fn)
        epoch_train_loss = float(np.mean(epoch_train_losses)) if epoch_train_losses else 0.0
        rec = {
            "step": step, "epoch": epoch, "phase": "val",
            "train_loss": epoch_train_loss,
            "val_loss": val_metrics["loss"], "loss": val_metrics["loss"],
            "degraded_psnr": val_metrics["psnr_in"], "restored_psnr": val_metrics["psnr_out"],
            "degraded_mae": val_metrics["mae_in"], "restored_mae": val_metrics["mae_out"],
            "degraded_ssim": val_metrics["ssim_in"], "restored_ssim": val_metrics["ssim_out"],
            "psnr_in": val_metrics["psnr_in"], "psnr_out": val_metrics["psnr_out"],
            "mae_in": val_metrics["mae_in"], "mae_out": val_metrics["mae_out"],
            "ssim_in": val_metrics["ssim_in"], "ssim_out": val_metrics["ssim_out"],
            "degradation_type_counts": dict(type_counter),
            "elapsed_s": round(time.time() - t0, 1),
        }
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        print(f"[val] ep{epoch} loss={val_metrics['loss']:.4f}  "
              f"PSNR in/out={val_metrics['psnr_in']:.2f}/{val_metrics['psnr_out']:.2f}  "
              f"MAE in/out={val_metrics['mae_in']:.4f}/{val_metrics['mae_out']:.4f}")

        # Fixed val sample grid after each epoch.
        save_unet_labeled_grid(
            os.path.join(sample_dir, f"fixed_val_epoch_{epoch + 1:04d}_labeled.png"),
            model,
            fixed_deg,
            fixed_clean,
            title=f"U-Net fixed validation samples | epoch {epoch + 1:04d}",
            labels=fixed_labels,
        )
        model.train()

        state = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "step": step,
            "val": val_metrics,
            "config": cfg,
        }
        save_checkpoint(os.path.join(out_dir, "ckpt_last.pt"), state)
        v = val_metrics["loss"]
        if is_better(v, best_val, "min"):
            best_val = v
            save_checkpoint(os.path.join(out_dir, "ckpt_best.pt"), state)
            print(f"[val] new best loss={v:.4f}")

    log_f.close()

    # End-of-run plots.
    plot_results = write_all_plots(
        out_dir, log_path,
        type_counter=type_counter,
        param_history={t: dict(p) for t, p in param_history.items()},
        title_prefix="U-Net ",
    )
    print(f"[train] plots: {plot_results}")
    print(f"[train] done in {time.time() - t0:.1f}s. outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
