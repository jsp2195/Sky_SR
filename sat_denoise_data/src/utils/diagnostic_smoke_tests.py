"""Small smoke tests for diagnostic grids and plots.

This module writes debug artifacts only; it does not read or modify datasets.
"""

from __future__ import annotations

import json
import os

import torch

from src.utils.diagnostic_grids import save_labeled_restoration_grid
from src.utils.training_plots import write_all_plots


def run_labeled_grid_smoke(path: str = "outputs/debug_labeled_grid.png") -> bool:
    torch.manual_seed(123)
    clean = torch.rand(4, 3, 64, 64) * 2.0 - 1.0
    degraded = (clean + torch.randn_like(clean) * 0.15).clamp(-1.0, 1.0)
    restored = (clean + torch.randn_like(clean) * 0.05).clamp(-1.0, 1.0)
    labels = [f"sample {i} | debug_degradation" for i in range(clean.shape[0])]
    save_labeled_restoration_grid(
        path,
        degraded,
        restored,
        clean,
        output_label="restored output",
        title="Diagnostic labeled grid smoke test",
        sample_labels=labels,
    )
    train_batch_path = "outputs/debug_samples/train_batch_step_000300_labeled.png"
    fixed_val_path = "outputs/debug_samples/fixed_val_epoch_0001_labeled.png"
    save_labeled_restoration_grid(
        train_batch_path,
        degraded,
        restored,
        clean,
        output_label="restored",
        title="U-Net train batch samples | step 000300",
        sample_labels=labels,
    )
    save_labeled_restoration_grid(
        fixed_val_path,
        degraded,
        restored,
        clean,
        output_label="restored",
        title="U-Net fixed validation samples | epoch 0001",
        sample_labels=labels,
    )
    return os.path.exists(path) and os.path.exists(train_batch_path) and os.path.exists(fixed_val_path)


def run_plot_smoke(out_dir: str = "outputs/debug_plots") -> bool:
    """Verify loss/metric plots write, with the log-scale loss_curve path
    exercised including NaN / non-positive / missing-field robustness.
    """
    import math
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.jsonl")
    rows = []
    # Inject a NaN train_loss at step 0 and a non-positive (zero) val_loss at
    # epoch 1 to exercise log-scale filtering. Epoch 2 also drops train_loss
    # entirely to mimic an older log.
    nan = float("nan")
    rows.append({"step": 0, "epoch": 0, "phase": "train", "loss": nan, "train_loss": nan})
    for epoch in range(3):
        rows.append({
            "step": epoch * 10 + 1,
            "epoch": epoch,
            "phase": "train",
            "train_loss": 0.30 / (epoch + 1),
            "loss": 0.30 / (epoch + 1),
        })
        val_row = {
            "step": (epoch + 1) * 10,
            "epoch": epoch,
            "phase": "val",
            "val_loss": (0.0 if epoch == 1 else 0.32 / (epoch + 1)),
            "loss":     (0.0 if epoch == 1 else 0.32 / (epoch + 1)),
            "psnr_in": 22.0 + epoch,
            "psnr_out": 23.5 + epoch,
            "mae_in": 0.08 - epoch * 0.01,
            "mae_out": 0.06 - epoch * 0.01,
            "ssim_in": 0.55 + epoch * 0.04,
            "ssim_out": 0.60 + epoch * 0.05,
        }
        if epoch != 2:
            val_row["train_loss"] = 0.28 / (epoch + 1)
        rows.append(val_row)
    with open(log_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    res_log = write_all_plots(out_dir, log_path, title_prefix="Debug ")
    loss_log = os.path.join(out_dir, "plots", "loss_curve.png")
    metrics_png = os.path.join(out_dir, "plots", "metric_curves.png")

    # Also exercise the linear fallback path: a log with no positive losses
    # must not crash and must produce a plot.
    fallback_dir = os.path.join(out_dir, "fallback")
    os.makedirs(fallback_dir, exist_ok=True)
    fb_log = os.path.join(fallback_dir, "train_log.jsonl")
    with open(fb_log, "w") as f:
        for epoch in range(2):
            f.write(json.dumps({
                "step": epoch, "epoch": epoch, "phase": "train",
                "loss": 0.0, "train_loss": 0.0,
            }) + "\n")
            f.write(json.dumps({
                "step": epoch + 1, "epoch": epoch, "phase": "val",
                "loss": 0.0, "val_loss": 0.0, "train_loss": 0.0,
            }) + "\n")
    write_all_plots(fallback_dir, fb_log, title_prefix="Fallback ")
    fb_loss = os.path.join(fallback_dir, "plots", "loss_curve.png")

    return (
        bool(res_log.get("loss_curve")) and bool(res_log.get("metric_curves"))
        and os.path.exists(loss_log) and os.path.exists(metrics_png)
        and os.path.exists(fb_loss)
    )


def main() -> int:
    grid_ok = run_labeled_grid_smoke()
    plots_ok = run_plot_smoke()
    print(f"debug_labeled_grid.png exists: {grid_ok}")
    print(f"debug plot PNGs exist: {plots_ok}")
    return 0 if grid_ok and plots_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
