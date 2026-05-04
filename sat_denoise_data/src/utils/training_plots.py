"""Training-run diagnostic plots.

Each function takes plain Python lists/dicts (no torch) and writes a PNG.
matplotlib is required; if missing, every function logs a warning and
returns False so training is never blocked by a plotting failure.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Iterable, Optional


def _import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        print(f"[plots] matplotlib unavailable, skipping plot: {e}")
        return None


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ---------------------------------------------------------------------------
# Loss / metric curves
# ---------------------------------------------------------------------------


def _filter_positive_pairs(
    xs: Iterable, ys: Iterable
) -> tuple[list[float], list[float]]:
    """Drop (x, y) pairs where y is NaN or y <= 0. Used before plotting on a
    log-scale y-axis. Inputs that are not finite numbers are also dropped."""
    out_x: list[float] = []
    out_y: list[float] = []
    for x, y in zip(list(xs), list(ys)):
        try:
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(yf) or yf <= 0.0:
            continue
        out_x.append(x)
        out_y.append(yf)
    return out_x, out_y


def _filter_finite_pairs(
    xs: Iterable, ys: Iterable
) -> tuple[list[float], list[float]]:
    """Drop (x, y) pairs where y is NaN/inf. Allows zeros and negatives."""
    out_x: list[float] = []
    out_y: list[float] = []
    for x, y in zip(list(xs), list(ys)):
        try:
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(yf):
            continue
        out_x.append(x)
        out_y.append(yf)
    return out_x, out_y


def plot_loss_curve(
    out_path: str,
    train_steps: Iterable[int],
    train_loss: Iterable[float],
    train_epochs: Optional[Iterable[int]] = None,
    epoch_train_loss: Optional[Iterable[float]] = None,
    val_epochs: Optional[Iterable[int]] = None,
    val_loss: Optional[Iterable[float]] = None,
    title: str = "Loss",
    log_scale: bool = True,
) -> bool:
    """Plot loss curves. With ``log_scale=True`` (default) the y-axis is
    logarithmic and non-positive / NaN entries are filtered before plotting.
    If after filtering no series has a single positive value, falls back to
    a linear y-axis and prints a warning so the plot is still useful.

    Tolerates older logs with missing fields: any of ``train_loss``,
    ``epoch_train_loss``, ``val_loss`` may be empty and the plot still draws.
    """
    plt = _import_mpl()
    if plt is None:
        return False
    _ensure_dir(out_path)
    plt.rcParams.update({"font.size": 11})

    # Filter inputs by mode. On log scale, drop non-positive and NaN values.
    if log_scale:
        ts, tl = _filter_positive_pairs(train_steps, train_loss)
        te, etl = _filter_positive_pairs(
            train_epochs or [], epoch_train_loss or [],
        )
        ve, vl = _filter_positive_pairs(val_epochs or [], val_loss or [])
        any_positive = bool(tl or etl or vl)
        if not any_positive:
            print("[plots] loss_curve: no positive loss values found; "
                  "falling back to linear y-axis")
            log_scale = False
            ts, tl = _filter_finite_pairs(train_steps, train_loss)
            te, etl = _filter_finite_pairs(
                train_epochs or [], epoch_train_loss or [],
            )
            ve, vl = _filter_finite_pairs(val_epochs or [], val_loss or [])
    else:
        ts, tl = _filter_finite_pairs(train_steps, train_loss)
        te, etl = _filter_finite_pairs(train_epochs or [], epoch_train_loss or [])
        ve, vl = _filter_finite_pairs(val_epochs or [], val_loss or [])

    fig, ax = plt.subplots(figsize=(8, 5))
    if ts and tl:
        ax.plot(ts, tl, label="train step loss", color="#8fbce6",
                linewidth=0.9, alpha=0.45)
    if te and etl:
        ax.plot(te, etl, label="train loss / epoch", color="#1f77b4",
                marker="o", linewidth=1.8)
    if ve and vl:
        ax.plot(ve, vl, label="validation loss / epoch", color="#d62728",
                marker="o", linewidth=1.8)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Loss (log scale)")
    else:
        ax.set_ylabel("loss")
    ax.set_xlabel("epoch")
    ax.set_title(title)
    ax.grid(True, which="both" if log_scale else "major", alpha=0.3)
    if ts or te or ve:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_metric_curves(
    out_path: str,
    epochs: Iterable[int],
    metrics: dict[str, Iterable[float]],
    title: str = "Validation metrics",
) -> bool:
    """metrics: dict mapping label -> per-epoch values."""
    plt = _import_mpl()
    if plt is None:
        return False
    _ensure_dir(out_path)
    eps = list(epochs)
    if not eps:
        return False
    plt.rcParams.update({"font.size": 11})
    pairs = [
        ("PSNR (dB)", [k for k in metrics if "psnr" in k.lower()]),
        ("MAE",       [k for k in metrics if "mae" in k.lower()]),
        ("SSIM",      [k for k in metrics if "ssim" in k.lower()]),
    ]
    if not any(keys for _, keys in pairs):
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, vals_iter in metrics.items():
            vals = list(vals_iter)
            if len(vals) == len(eps):
                ax.plot(eps, vals, marker="o", label=k, linewidth=1.8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("metric value")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return True

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (label, keys) in zip(axes, pairs):
        for k in keys:
            vals = list(metrics[k])
            if len(vals) != len(eps):
                continue
            style = "--" if k.endswith("_in") or "deg" in k.lower() else "-"
            ax.plot(eps, vals, style, marker="o", label=k, linewidth=1.2)
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Degradation distribution
# ---------------------------------------------------------------------------


def plot_degradation_type_distribution(
    out_path: str,
    counts: dict[str, int],
    title: str = "Degradation type distribution",
) -> bool:
    plt = _import_mpl()
    if plt is None:
        return False
    if not counts:
        return False
    _ensure_dir(out_path)
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    keys = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(keys, vals, color="#4c72b0")
    ax.set_ylabel("count")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def plot_degradation_param_histograms(
    out_path: str,
    params_by_type: dict[str, dict[str, list[float]]],
    title: str = "Degradation parameter histograms",
) -> bool:
    """params_by_type: {type -> {param_name -> list[float]}}.

    Skips empty parameter lists; if everything is empty, no plot is written.
    """
    plt = _import_mpl()
    if plt is None:
        return False
    flat: list[tuple[str, str, list[float]]] = []
    for t, params in params_by_type.items():
        for name, vals in params.items():
            vals = [float(v) for v in vals if isinstance(v, (int, float))]
            if vals:
                flat.append((t, name, vals))
    if not flat:
        return False
    _ensure_dir(out_path)
    n = len(flat)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.0 * rows), squeeze=False)
    for ax, (t, name, vals) in zip(axes.flat, flat):
        ax.hist(vals, bins=20, color="#4c72b0")
        ax.set_title(f"{t} :: {name}", fontsize=9)
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(flat):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Train-log helpers
# ---------------------------------------------------------------------------


def collect_train_log(log_path: str) -> dict:
    """Reads a JSONL training log and groups rows by phase."""
    train_steps, train_loss = [], []
    train_by_epoch: dict[int, list[float]] = {}
    epoch_train_loss = []
    val_loss = []
    val_epoch_idx = []
    val_psnr_in, val_psnr_out = [], []
    val_mae_in, val_mae_out = [], []
    val_ssim_in, val_ssim_out = [], []
    val_x0_l1 = []
    if not os.path.exists(log_path):
        return {}
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            phase = rec.get("phase")
            if phase == "train" and "step" in rec and ("loss" in rec or "train_loss" in rec):
                loss = float(rec.get("loss", rec.get("train_loss", 0.0)))
                epoch = int(rec.get("epoch", 0))
                train_steps.append(float(epoch))
                train_loss.append(loss)
                train_by_epoch.setdefault(epoch, []).append(loss)
            elif phase == "train" and "total" in rec and "step" in rec:
                loss = float(rec["total"])
                epoch = int(rec.get("epoch", 0))
                train_steps.append(float(epoch))
                train_loss.append(loss)
                train_by_epoch.setdefault(epoch, []).append(loss)
            elif phase == "val":
                epoch = int(rec.get("epoch", len(val_epoch_idx)))
                val_epoch_idx.append(epoch)
                if "train_loss" in rec:
                    epoch_train_loss.append(float(rec["train_loss"]))
                else:
                    vals = train_by_epoch.get(epoch, [])
                    epoch_train_loss.append(sum(vals) / len(vals) if vals else 0.0)
                val_loss.append(float(rec.get("loss", rec.get("total", 0.0))))
                val_psnr_in.append(float(rec.get("psnr_in", 0.0)))
                val_psnr_out.append(float(rec.get("psnr_out", 0.0)))
                val_mae_in.append(float(rec.get("mae_in", 0.0)))
                val_mae_out.append(float(rec.get("mae_out", 0.0)))
                val_ssim_in.append(float(rec.get("ssim_in", 0.0)))
                val_ssim_out.append(float(rec.get("ssim_out", 0.0)))
                val_x0_l1.append(float(rec.get("x0_l1", 0.0)))
    return {
        "train_steps": train_steps,
        "train_loss": train_loss,
        "train_epochs": val_epoch_idx,
        "epoch_train_loss": epoch_train_loss,
        "val_epochs": val_epoch_idx,
        "val_loss": val_loss,
        "psnr_in": val_psnr_in,
        "psnr_out": val_psnr_out,
        "mae_in": val_mae_in,
        "mae_out": val_mae_out,
        "ssim_in": val_ssim_in,
        "ssim_out": val_ssim_out,
        "x0_l1": val_x0_l1,
    }


def write_all_plots(
    out_dir: str,
    log_path: str,
    *,
    type_counter: Optional[Counter] = None,
    param_history: Optional[dict[str, dict[str, list[float]]]] = None,
    title_prefix: str = "",
    loss_log_scale: bool = True,
) -> dict[str, bool]:
    """Convenience wrapper: read log, write loss/metric/distribution plots.

    ``loss_log_scale`` (default True) controls the y-axis of loss_curve.png
    only. Validation metrics (PSNR/MAE/SSIM) stay linear.
    """
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    log = collect_train_log(log_path)
    results: dict[str, bool] = {}

    results["loss_curve"] = plot_loss_curve(
        os.path.join(plots_dir, "loss_curve.png"),
        train_steps=log.get("train_steps", []),
        train_loss=log.get("train_loss", []),
        train_epochs=log.get("train_epochs", []),
        epoch_train_loss=log.get("epoch_train_loss", []),
        val_epochs=log.get("val_epochs", []),
        val_loss=log.get("val_loss", []),
        title=f"{title_prefix}loss".strip() or "loss",
        log_scale=loss_log_scale,
    )

    has_restoration_metrics = any(log.get(k, []) and any(v != 0.0 for v in log.get(k, []))
                                  for k in ("psnr_in", "psnr_out", "mae_in", "mae_out"))
    if has_restoration_metrics:
        metric_dict = {
            "degraded PSNR":  log.get("psnr_in", []),
            "restored PSNR":  log.get("psnr_out", []),
            "degraded MAE":   log.get("mae_in", []),
            "restored MAE":   log.get("mae_out", []),
            "degraded SSIM":  log.get("ssim_in", []),
            "restored SSIM":  log.get("ssim_out", []),
        }
    else:
        metric_dict = {
            "train epsilon-MSE": log.get("epoch_train_loss", []),
            "validation epsilon-MSE": log.get("val_loss", []),
            "x0_l1": log.get("x0_l1", []),
        }
    results["metric_curves"] = plot_metric_curves(
        os.path.join(plots_dir, "metric_curves.png"),
        epochs=log.get("val_epochs", []),
        metrics=metric_dict,
        title=f"{title_prefix}validation metrics".strip() or "validation metrics",
    )

    if type_counter is not None and len(type_counter) > 0:
        results["degradation_type_distribution"] = plot_degradation_type_distribution(
            os.path.join(plots_dir, "degradation_type_distribution.png"),
            counts=dict(type_counter),
        )
    if param_history:
        results["degradation_param_histograms"] = plot_degradation_param_histograms(
            os.path.join(plots_dir, "degradation_param_histograms.png"),
            params_by_type=param_history,
        )
    return results
