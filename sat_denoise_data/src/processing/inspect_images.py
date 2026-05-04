"""Inspect image files in data/raw/ and report metadata.

Reports for each file: type, dimensions, number of bands, dtype, min/max,
and CRS+transform when readable as a GeoTIFF.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.image_io import GEOTIFF_EXTS, list_images

try:
    import rasterio
    _HAS_RASTERIO = True
except Exception:
    _HAS_RASTERIO = False

from PIL import Image


def inspect_one(path: str) -> dict:
    info: dict = {"path": path, "ext": os.path.splitext(path)[1].lower()}
    is_gtiff = info["ext"] in GEOTIFF_EXTS and _HAS_RASTERIO
    info["reader"] = "rasterio" if is_gtiff else "PIL"

    if is_gtiff:
        try:
            with rasterio.open(path) as ds:
                info["width"] = ds.width
                info["height"] = ds.height
                info["bands"] = ds.count
                info["dtype"] = str(ds.dtypes[0])
                info["crs"] = str(ds.crs) if ds.crs else None
                info["transform"] = list(ds.transform)[:6] if ds.transform else None
                info["nodata"] = ds.nodata
                # Sample stats from first band only to keep this cheap.
                sample = ds.read(1, out_shape=(min(512, ds.height), min(512, ds.width)))
                finite = sample[np.isfinite(sample)] if sample.dtype.kind == "f" else sample
                if finite.size:
                    info["min_sample"] = float(finite.min())
                    info["max_sample"] = float(finite.max())
            return info
        except Exception as e:
            info["error"] = f"rasterio: {e}"

    try:
        with Image.open(path) as im:
            info["width"], info["height"] = im.size
            info["mode"] = im.mode
            info["bands"] = len(im.getbands())
            arr = np.array(im)
            info["dtype"] = str(arr.dtype)
            info["min_sample"] = float(arr.min())
            info["max_sample"] = float(arr.max())
    except Exception as e:
        info["error"] = f"PIL: {e}"

    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/raw")
    args = ap.parse_args()

    files = list_images(args.input_dir)
    if not files:
        print(f"[inspect] no images in {args.input_dir}")
        return 1

    print(f"[inspect] {len(files)} files in {args.input_dir}")
    print("-" * 80)
    for path in files:
        info = inspect_one(path)
        rel = os.path.relpath(path)
        print(rel)
        for k in (
            "reader",
            "width",
            "height",
            "bands",
            "dtype",
            "mode",
            "min_sample",
            "max_sample",
            "crs",
            "transform",
            "nodata",
            "error",
        ):
            if k in info:
                print(f"  {k}: {info[k]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
