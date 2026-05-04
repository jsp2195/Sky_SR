"""Image I/O helpers.

Two read paths:

  1) GeoTIFF / TIFF: opened with rasterio. Use `RasterioSource` for
     **windowed** reads – never loads the full image into memory.
     If rasterio is not installed, GeoTIFFs raise a clear error rather than
     falling back to PIL (PIL trips its decompression-bomb guard on large
     orthomosaics).

  2) PNG / JPG / BMP and tiny non-georeferenced TIFF: loaded as a whole image
     with PIL via `load_image_rgb`. These are assumed to be small.

Both paths return RGB uint8 arrays of shape (H, W, 3).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.windows import Window
    _HAS_RASTERIO = True
except Exception:
    _HAS_RASTERIO = False
    Window = None  # type: ignore


GEOTIFF_EXTS = {".tif", ".tiff", ".gtiff"}
PIL_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def is_geotiff(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in GEOTIFF_EXTS


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Safely scale to uint8.

    uint8 -> passthrough
    uint16 -> scale by 256 (assumes full 16-bit range)
    float -> percentile-stretch p2-p98 then to 0-255
    """
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr // 256).astype(np.uint8)
    arr = arr.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (2.0, 98.0))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# GeoTIFF: windowed reader
# ---------------------------------------------------------------------------


class RasterioSource:
    """Open a GeoTIFF and read patch windows on demand.

    Usage:
        with RasterioSource(path) as src:
            for r, c in src.iter_windows(patch_size, stride):
                patch, mask = src.read_window(r, c, patch_size)
    """

    def __init__(self, path: str):
        if not _HAS_RASTERIO:
            raise RuntimeError(
                f"rasterio is required to read GeoTIFFs without loading them "
                f"fully into memory: {path}. `pip install rasterio` and retry."
            )
        self.path = path
        self.ds = rasterio.open(path)
        self.width: int = self.ds.width
        self.height: int = self.ds.height
        self.bands: int = self.ds.count
        self.dtype: str = str(self.ds.dtypes[0])
        self.crs: Optional[str] = str(self.ds.crs) if self.ds.crs else None
        self.transform: Optional[list[float]] = (
            list(self.ds.transform) if self.ds.transform else None
        )
        self.nodata = self.ds.nodata

        if self.bands >= 3:
            self.indexes: list[int] = [1, 2, 3]
        elif self.bands == 1:
            self.indexes = [1, 1, 1]  # replicate to RGB
        else:
            # 2 bands is unusual; replicate band 1.
            self.indexes = [1, 1, 1]

    def close(self) -> None:
        try:
            self.ds.close()
        except Exception:
            pass

    def __enter__(self) -> "RasterioSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def iter_windows(
        self, patch_size: int, stride: int
    ):  # -> Iterator[tuple[int, int]]
        for r in range(0, self.height - patch_size + 1, stride):
            for c in range(0, self.width - patch_size + 1, stride):
                yield r, c

    def read_window(
        self, row: int, col: int, patch_size: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        win = Window(col, row, patch_size, patch_size)
        # boundless=False (default): we only iterate fully-inside windows.
        arr = self.ds.read(self.indexes, window=win)  # (3, H, W)
        arr = np.transpose(arr, (1, 2, 0))  # (H, W, 3)

        mask: Optional[np.ndarray] = None
        if self.nodata is not None:
            mask = np.any(arr == self.nodata, axis=2)

        arr_u8 = _to_uint8(arr)
        if mask is not None:
            arr_u8[mask] = 0
        return arr_u8, mask


# ---------------------------------------------------------------------------
# Whole-image PIL loader (for small PNG/JPG/BMP/non-geo TIFF)
# ---------------------------------------------------------------------------


def load_image_rgb(path: str) -> Tuple[np.ndarray, dict]:
    """Whole-image RGB uint8 load via PIL.

    Intended for PNG/JPG/BMP and small non-georeferenced TIFFs only.
    For large GeoTIFFs use `RasterioSource` to avoid loading the full image.
    """
    ext = os.path.splitext(path)[1].lower()
    meta: dict = {"path": path, "ext": ext}

    with Image.open(path) as im:
        meta["mode"] = im.mode
        meta["width"], meta["height"] = im.size
        if im.mode != "RGB":
            im = im.convert("RGB")
        arr = np.array(im)
        meta["bands"] = arr.shape[2] if arr.ndim == 3 else 1
        meta["dtype"] = str(arr.dtype)
        if arr.dtype != np.uint8:
            arr = _to_uint8(arr)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=2)
        meta["nodata_mask"] = None
        meta["transform"] = None
        meta["crs"] = None
        return arr, meta


def save_png(path: str, arr: np.ndarray) -> None:
    if arr.dtype != np.uint8:
        arr = _to_uint8(arr)
    Image.fromarray(arr).save(path, format="PNG", optimize=True)


def list_images(input_dir: str) -> list[str]:
    out = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in GEOTIFF_EXTS or ext in PIL_EXTS:
                out.append(os.path.join(root, f))
    out.sort()
    return out
