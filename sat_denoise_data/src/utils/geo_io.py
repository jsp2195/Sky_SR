"""Lightweight geospatial helpers.

Used for recording approximate per-patch georeferencing in the manifest.
Does not reproject. Falls back gracefully when rasterio is missing.
"""

from __future__ import annotations

from typing import Optional

try:
    import rasterio
    from rasterio.transform import Affine
    _HAS_RASTERIO = True
except Exception:
    _HAS_RASTERIO = False
    Affine = None  # type: ignore


def patch_transform(parent_transform, col_off: int, row_off: int):
    """Return an affine transform for a sub-window.

    parent_transform is a list/tuple of 6 affine coefficients (a, b, c, d, e, f)
    or a rasterio Affine. Returns the same kind back as a list.
    """
    if parent_transform is None:
        return None
    a, b, c, d, e, f = list(parent_transform)[:6]
    new_c = c + col_off * a + row_off * b
    new_f = f + col_off * d + row_off * e
    return [a, b, new_c, d, e, new_f]


def transform_to_bbox(transform, width: int, height: int) -> Optional[list[float]]:
    """Pixel-space affine -> (xmin, ymin, xmax, ymax) in source CRS units."""
    if transform is None:
        return None
    a, b, c, d, e, f = list(transform)[:6]

    def xy(col, row):
        return (a * col + b * row + c, d * col + e * row + f)

    xs, ys = zip(*[xy(0, 0), xy(width, 0), xy(0, height), xy(width, height)])
    return [min(xs), min(ys), max(xs), max(ys)]
