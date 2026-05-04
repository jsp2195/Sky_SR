"""Build a 256x256 RGB patch dataset from images in data/raw/.

Reads GeoTIFF/TIFF/PNG/JPG, tiles each image with a configurable patch_size
and stride, drops blank/uniform/nodata-heavy patches, and writes:
  - PNG patches under --output_dir
  - JSONL manifest at --manifest

Resumable: skips manifest entries already present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Iterator, Optional

import numpy as np
from tqdm import tqdm

# Allow running as `python -m src.processing.build_patch_dataset` from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.image_io import (
    RasterioSource,
    is_geotiff,
    list_images,
    load_image_rgb,
    save_png,
)
from src.utils.geo_io import patch_transform, transform_to_bbox


def short_hash(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def iter_patches(
    img: np.ndarray,
    patch_size: int,
    stride: int,
    nodata_mask: Optional[np.ndarray] = None,
) -> Iterator[tuple[int, int, np.ndarray, Optional[np.ndarray]]]:
    h, w, _ = img.shape
    for r in range(0, h - patch_size + 1, stride):
        for c in range(0, w - patch_size + 1, stride):
            patch = img[r : r + patch_size, c : c + patch_size, :]
            mp = (
                nodata_mask[r : r + patch_size, c : c + patch_size]
                if nodata_mask is not None
                else None
            )
            yield r, c, patch, mp


def patch_is_good(
    patch: np.ndarray,
    mask: Optional[np.ndarray],
    *,
    skip_blank: bool,
    min_std: float,
    max_nodata_fraction: float,
) -> tuple[bool, str]:
    if mask is not None:
        frac = float(mask.mean())
        if frac > max_nodata_fraction:
            return False, f"nodata_fraction={frac:.3f}"

    if skip_blank:
        # Pure black or pure white patches.
        n = patch.size
        black_frac = float((patch == 0).sum()) / n
        white_frac = float((patch == 255).sum()) / n
        if black_frac > max_nodata_fraction:
            return False, f"black_fraction={black_frac:.3f}"
        if white_frac > max_nodata_fraction:
            return False, f"white_fraction={white_frac:.3f}"

    std = float(patch.astype(np.float32).std())
    if std < min_std:
        return False, f"std={std:.2f}"

    return True, "ok"


def load_existing_state(manifest_path: str) -> tuple[set[str], dict[str, int]]:
    """Return (existing patch_ids, per-source patch counts) from a manifest."""
    keys: set[str] = set()
    per_source: dict[str, int] = {}
    if not os.path.exists(manifest_path):
        return keys, per_source
    with open(manifest_path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                keys.add(rec["patch_id"])
                src = rec.get("source_path", "")
                per_source[src] = per_source.get(src, 0) + 1
            except Exception:
                continue
    return keys, per_source


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/raw")
    ap.add_argument("--output_dir", default="data/patches")
    ap.add_argument("--manifest", default="data/manifests/patches.jsonl")
    ap.add_argument("--patch_size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--max_patches", type=int, default=1000)
    ap.add_argument(
        "--max_patches_per_source",
        type=int,
        default=None,
        help="cap accepted patches per source image (default: unlimited)",
    )
    ap.add_argument("--rgb_only", action="store_true")
    ap.add_argument("--skip_blank", action="store_true")
    ap.add_argument("--min_std", type=float, default=5.0)
    ap.add_argument("--max_nodata_fraction", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)

    files = list_images(args.input_dir)
    if not files:
        print(f"[build_patch_dataset] no images found in {args.input_dir}")
        print("  place GeoTIFF/TIFF/PNG/JPG files there and re-run.")
        return 1

    if args.resume:
        existing, per_source_count = load_existing_state(args.manifest)
    else:
        existing, per_source_count = set(), {}
    if existing:
        print(f"[build_patch_dataset] resume: {len(existing)} patches already in manifest")

    written = len(existing)
    manifest_f = open(args.manifest, "a")

    pbar = tqdm(total=args.max_patches, initial=written, desc="patches")

    def write_patch_record(
        patch: np.ndarray,
        mp: Optional[np.ndarray],
        *,
        path: str,
        src_rel: str,
        src_key: str,
        base: str,
        r: int,
        c: int,
        crs: Optional[str],
        parent_transform: Optional[list],
    ) -> bool:
        """Filter, save PNG, append manifest record. Returns True if written."""
        nonlocal written
        patch_id = f"{src_key}_{r:06d}_{c:06d}"
        if patch_id in existing:
            return False

        ok, _reason = patch_is_good(
            patch,
            mp,
            skip_blank=args.skip_blank,
            min_std=args.min_std,
            max_nodata_fraction=args.max_nodata_fraction,
        )
        if not ok:
            return False

        out_name = f"{base}__{patch_id}.png"
        out_path = os.path.join(args.output_dir, out_name)
        save_png(out_path, patch)

        ptrans = patch_transform(parent_transform, c, r) if parent_transform else None
        bbox = (
            transform_to_bbox(ptrans, args.patch_size, args.patch_size)
            if ptrans
            else None
        )

        rec = {
            "patch_id": patch_id,
            "patch_path": os.path.relpath(out_path),
            "source_path": src_rel,
            "row": r,
            "col": c,
            "width": args.patch_size,
            "height": args.patch_size,
            "crs": crs,
            "transform": ptrans,
            "bbox": bbox,
        }
        manifest_f.write(json.dumps(rec) + "\n")
        manifest_f.flush()
        existing.add(patch_id)
        written += 1
        pbar.update(1)
        return True

    try:
        for path in files:
            if written >= args.max_patches:
                break

            src_rel = os.path.relpath(path)
            src_count = per_source_count.get(src_rel, 0)
            if (
                args.max_patches_per_source is not None
                and src_count >= args.max_patches_per_source
            ):
                continue

            src_key = short_hash(path)
            base = os.path.splitext(os.path.basename(path))[0]

            try:
                if is_geotiff(path):
                    # Windowed rasterio reads – never loads the full image.
                    with RasterioSource(path) as src:
                        if args.rgb_only and src.bands < 1:
                            continue
                        crs = src.crs
                        parent_transform = src.transform
                        for r, c in src.iter_windows(args.patch_size, args.stride):
                            if written >= args.max_patches:
                                break
                            if (
                                args.max_patches_per_source is not None
                                and src_count >= args.max_patches_per_source
                            ):
                                break
                            patch, mp = src.read_window(r, c, args.patch_size)
                            if write_patch_record(
                                patch,
                                mp,
                                path=path,
                                src_rel=src_rel,
                                src_key=src_key,
                                base=base,
                                r=r,
                                c=c,
                                crs=crs,
                                parent_transform=parent_transform,
                            ):
                                src_count += 1
                                per_source_count[src_rel] = src_count
                else:
                    # PNG / JPG / BMP: small, load whole.
                    img, meta = load_image_rgb(path)
                    if args.rgb_only and img.shape[-1] != 3:
                        continue
                    crs = meta.get("crs")
                    parent_transform = meta.get("transform")
                    nodata_mask = meta.get("nodata_mask")
                    for r, c, patch, mp in iter_patches(
                        img, args.patch_size, args.stride, nodata_mask
                    ):
                        if written >= args.max_patches:
                            break
                        if (
                            args.max_patches_per_source is not None
                            and src_count >= args.max_patches_per_source
                        ):
                            break
                        if write_patch_record(
                            patch,
                            mp,
                            path=path,
                            src_rel=src_rel,
                            src_key=src_key,
                            base=base,
                            r=r,
                            c=c,
                            crs=crs,
                            parent_transform=parent_transform,
                        ):
                            src_count += 1
                            per_source_count[src_rel] = src_count
            except Exception as e:
                print(f"[skip] {path}: {e}")
                continue

    finally:
        manifest_f.close()
        pbar.close()

    print(f"[build_patch_dataset] wrote {written} patches -> {args.output_dir}")
    print(f"[build_patch_dataset] manifest: {args.manifest}")

    used_sources = {s: n for s, n in per_source_count.items() if n > 0}
    print(
        f"[build_patch_dataset] sources used: {len(used_sources)} / {len(files)}"
    )
    for src, n in sorted(used_sources.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
