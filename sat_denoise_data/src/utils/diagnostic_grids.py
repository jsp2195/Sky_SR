"""Labeled diagnostic image grids for restoration outputs."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _to_uint8_image(x: torch.Tensor) -> Image.Image:
    """Convert a CHW tensor in [-1, 1] or [0, 1] to a PIL RGB image."""
    t = x.detach().float().cpu()
    if t.min() < -1e-3:
        t = (t + 1.0) * 0.5
    t = t.clamp(0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _error_image(pred: torch.Tensor, clean: torch.Tensor) -> Image.Image:
    err = (pred.detach().float().cpu() - clean.detach().float().cpu()).abs()
    denom = err.max().clamp_min(1e-6)
    err = (err / denom).clamp(0.0, 1.0)
    arr = (err.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _mask_image(mask: torch.Tensor) -> Image.Image:
    t = mask.detach().float().cpu()
    if t.ndim == 3:
        t = t[:1]
    if t.ndim == 2:
        t = t.unsqueeze(0)
    t = t.clamp(0.0, 1.0).expand(3, -1, -1)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_text(text: str, max_width: int, font: ImageFont.ImageFont) -> str:
    if len(text) <= 4:
        return text
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    suffix = "..."
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid] + suffix
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + suffix


def save_labeled_restoration_grid(
    path: str,
    degraded: torch.Tensor,
    output: torch.Tensor,
    clean: torch.Tensor,
    *,
    output_label: str,
    title: str,
    sample_labels: Sequence[str] | None = None,
) -> None:
    """Save rows=samples and columns=degraded/output/clean/absolute error."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    degraded = degraded.detach().cpu()
    output = output.detach().cpu()
    clean = clean.detach().cpu()
    n = min(degraded.shape[0], output.shape[0], clean.shape[0])
    if n <= 0:
        raise ValueError("cannot save an empty diagnostic grid")

    tile_w = int(clean.shape[-1])
    tile_h = int(clean.shape[-2])
    label_w = 180
    header_h = 56
    row_label_h = 28
    pad = 8
    cols = ["degraded", output_label, "clean", "abs error"]
    grid_w = label_w + len(cols) * tile_w + (len(cols) + 1) * pad
    grid_h = header_h + n * (tile_h + row_label_h + pad) + pad

    canvas = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    label_font = _font(14)
    small_font = _font(12)
    draw.text((pad, 12), title, fill=(20, 20, 20), font=title_font)

    y0 = header_h
    for ci, col in enumerate(cols):
        x = label_w + pad + ci * (tile_w + pad)
        draw.text((x, header_h - 24), col, fill=(20, 20, 20), font=label_font)

    labels = list(sample_labels or [])
    for ri in range(n):
        y = y0 + ri * (tile_h + row_label_h + pad)
        label = labels[ri] if ri < len(labels) else f"sample {ri}"
        label = _fit_text(label, label_w - 2 * pad, small_font)
        draw.text((pad, y + 6), label, fill=(20, 20, 20), font=small_font)

        tiles = [
            _to_uint8_image(degraded[ri]),
            _to_uint8_image(output[ri]),
            _to_uint8_image(clean[ri]),
            _error_image(output[ri], clean[ri]),
        ]
        for ci, tile in enumerate(tiles):
            x = label_w + pad + ci * (tile_w + pad)
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=(210, 210, 210))

        meta_y = y + tile_h + 4
        draw.text((label_w + pad, meta_y), label, fill=(70, 70, 70), font=small_font)

    canvas.save(path)


def save_labeled_residual_refinement_grid(
    path: str,
    degraded: torch.Tensor,
    unet_base: torch.Tensor,
    ddpm_refined: torch.Tensor,
    clean: torch.Tensor,
    *,
    title: str,
    sample_labels: Sequence[str] | None = None,
) -> None:
    """Five-column grid for residual-DDPM refinement diagnostics.

    Columns: degraded | U-Net base | DDPM refined | clean | abs error
    where abs error is |DDPM refined - clean|, normalised per row.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    degraded = degraded.detach().cpu()
    unet_base = unet_base.detach().cpu()
    ddpm_refined = ddpm_refined.detach().cpu()
    clean = clean.detach().cpu()
    n = min(
        degraded.shape[0], unet_base.shape[0], ddpm_refined.shape[0], clean.shape[0]
    )
    if n <= 0:
        raise ValueError("cannot save an empty diagnostic grid")

    tile_w = int(clean.shape[-1])
    tile_h = int(clean.shape[-2])
    label_w = 180
    header_h = 56
    row_label_h = 28
    pad = 8
    cols = ["degraded", "U-Net base", "DDPM refined", "clean", "abs error"]
    grid_w = label_w + len(cols) * tile_w + (len(cols) + 1) * pad
    grid_h = header_h + n * (tile_h + row_label_h + pad) + pad

    canvas = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    label_font = _font(14)
    small_font = _font(12)
    draw.text((pad, 12), title, fill=(20, 20, 20), font=title_font)

    y0 = header_h
    for ci, col in enumerate(cols):
        x = label_w + pad + ci * (tile_w + pad)
        draw.text((x, header_h - 24), col, fill=(20, 20, 20), font=label_font)

    labels = list(sample_labels or [])
    for ri in range(n):
        y = y0 + ri * (tile_h + row_label_h + pad)
        label = labels[ri] if ri < len(labels) else f"sample {ri}"
        label = _fit_text(label, label_w - 2 * pad, small_font)
        draw.text((pad, y + 6), label, fill=(20, 20, 20), font=small_font)

        tiles = [
            _to_uint8_image(degraded[ri]),
            _to_uint8_image(unet_base[ri]),
            _to_uint8_image(ddpm_refined[ri]),
            _to_uint8_image(clean[ri]),
            _error_image(ddpm_refined[ri], clean[ri]),
        ]
        for ci, tile in enumerate(tiles):
            x = label_w + pad + ci * (tile_w + pad)
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=(210, 210, 210))

        meta_y = y + tile_h + 4
        draw.text((label_w + pad, meta_y), label, fill=(70, 70, 70), font=small_font)

    canvas.save(path)


def save_labeled_gated_residual_refinement_grid(
    path: str,
    degraded: torch.Tensor,
    unet_base: torch.Tensor,
    gate: torch.Tensor,
    ddpm_gated_refined: torch.Tensor,
    clean: torch.Tensor,
    *,
    title: str,
    sample_labels: Sequence[str] | None = None,
    reliability: torch.Tensor | None = None,
) -> None:
    """Six-column grid for gated residual-DDPM diagnostics.

    Columns: degraded | U-Net base | gate mask | DDPM gated refined | clean | abs error
    where abs error is |DDPM gated refined - clean|, normalised per row.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    degraded = degraded.detach().cpu()
    unet_base = unet_base.detach().cpu()
    gate = gate.detach().cpu()
    reliability = reliability.detach().cpu() if reliability is not None else None
    ddpm_gated_refined = ddpm_gated_refined.detach().cpu()
    clean = clean.detach().cpu()
    n = min(
        degraded.shape[0], unet_base.shape[0], gate.shape[0],
        ddpm_gated_refined.shape[0], clean.shape[0],
    )
    if n <= 0:
        raise ValueError("cannot save an empty diagnostic grid")

    tile_w = int(clean.shape[-1])
    tile_h = int(clean.shape[-2])
    label_w = 180
    header_h = 56
    row_label_h = 28
    pad = 8
    cols = ["degraded", "U-Net base", "gate mask", "DDPM gated", "clean", "abs error"]
    if reliability is not None:
        cols = ["degraded", "U-Net base", "reliability", "gate mask", "DDPM gated", "clean", "abs error"]
    grid_w = label_w + len(cols) * tile_w + (len(cols) + 1) * pad
    grid_h = header_h + n * (tile_h + row_label_h + pad) + pad

    canvas = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    label_font = _font(14)
    small_font = _font(12)
    draw.text((pad, 12), title, fill=(20, 20, 20), font=title_font)

    y0 = header_h
    for ci, col in enumerate(cols):
        x = label_w + pad + ci * (tile_w + pad)
        draw.text((x, header_h - 24), col, fill=(20, 20, 20), font=label_font)

    labels = list(sample_labels or [])
    for ri in range(n):
        y = y0 + ri * (tile_h + row_label_h + pad)
        label = labels[ri] if ri < len(labels) else f"sample {ri}"
        label = _fit_text(label, label_w - 2 * pad, small_font)
        draw.text((pad, y + 6), label, fill=(20, 20, 20), font=small_font)

        tiles = [_to_uint8_image(degraded[ri]), _to_uint8_image(unet_base[ri])]
        if reliability is not None:
            tiles.append(_mask_image(reliability[ri]))
        tiles.extend([
            _mask_image(gate[ri]),
            _to_uint8_image(ddpm_gated_refined[ri]),
            _to_uint8_image(clean[ri]),
            _error_image(ddpm_gated_refined[ri], clean[ri]),
        ])
        for ci, tile in enumerate(tiles):
            x = label_w + pad + ci * (tile_w + pad)
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=(210, 210, 210))

        meta_y = y + tile_h + 4
        draw.text((label_w + pad, meta_y), label, fill=(70, 70, 70), font=small_font)

    canvas.save(path)


def save_labeled_masked_completion_grid(
    path: str,
    degraded: torch.Tensor,
    mask: torch.Tensor,
    unet_base: torch.Tensor,
    ddpm_final: torch.Tensor,
    clean: torch.Tensor,
    *,
    title: str,
    sample_labels: Sequence[str] | None = None,
    reliability: torch.Tensor | None = None,
    uncertainty: torch.Tensor | None = None,
) -> None:
    """Diagnostic grid for masked-completion DDPM.

    Columns: degraded | mask | reliability | U-Net base | diffusion final |
    clean | uncertainty_or_abs_error.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    degraded = degraded.detach().cpu()
    mask = mask.detach().cpu()
    unet_base = unet_base.detach().cpu()
    ddpm_final = ddpm_final.detach().cpu()
    clean = clean.detach().cpu()
    reliability = reliability.detach().cpu() if reliability is not None else None
    uncertainty = uncertainty.detach().cpu() if uncertainty is not None else None
    n = min(degraded.shape[0], mask.shape[0], unet_base.shape[0], ddpm_final.shape[0], clean.shape[0])
    if n <= 0:
        raise ValueError("cannot save an empty diagnostic grid")

    tile_w = int(clean.shape[-1])
    tile_h = int(clean.shape[-2])
    label_w = 180
    header_h = 56
    row_label_h = 28
    pad = 8
    tail = "uncertainty" if uncertainty is not None else "abs error"
    cols = ["degraded", "mask", "reliability", "U-Net base", "diffusion final", "clean", tail]
    grid_w = label_w + len(cols) * tile_w + (len(cols) + 1) * pad
    grid_h = header_h + n * (tile_h + row_label_h + pad) + pad

    canvas = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    label_font = _font(14)
    small_font = _font(12)
    draw.text((pad, 12), title, fill=(20, 20, 20), font=title_font)

    y0 = header_h
    for ci, col in enumerate(cols):
        x = label_w + pad + ci * (tile_w + pad)
        draw.text((x, header_h - 24), col, fill=(20, 20, 20), font=label_font)

    labels = list(sample_labels or [])
    for ri in range(n):
        y = y0 + ri * (tile_h + row_label_h + pad)
        label = labels[ri] if ri < len(labels) else f"sample {ri}"
        label = _fit_text(label, label_w - 2 * pad, small_font)
        draw.text((pad, y + 6), label, fill=(20, 20, 20), font=small_font)

        reliability_tile = (
            _mask_image(reliability[ri]) if reliability is not None
            else _mask_image(torch.ones_like(mask[ri]))
        )
        tail_tile = _mask_image(uncertainty[ri]) if uncertainty is not None else _error_image(ddpm_final[ri], clean[ri])
        tiles = [
            _to_uint8_image(degraded[ri]),
            _mask_image(mask[ri]),
            reliability_tile,
            _to_uint8_image(unet_base[ri]),
            _to_uint8_image(ddpm_final[ri]),
            _to_uint8_image(clean[ri]),
            tail_tile,
        ]
        for ci, tile in enumerate(tiles):
            x = label_w + pad + ci * (tile_w + pad)
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=(210, 210, 210))

        meta_y = y + tile_h + 4
        draw.text((label_w + pad, meta_y), label, fill=(70, 70, 70), font=small_font)

    canvas.save(path)
