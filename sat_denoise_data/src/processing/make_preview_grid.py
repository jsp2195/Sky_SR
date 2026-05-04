"""Make a preview grid PNG of random patches from data/patches/."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_dir", default="data/patches")
    ap.add_argument("--output", default="data/previews/patch_grid.png")
    ap.add_argument("--num_images", type=int, default=64)
    ap.add_argument("--cell_size", type=int, default=128, help="resize each patch to this px")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = [
        os.path.join(args.patch_dir, f)
        for f in sorted(os.listdir(args.patch_dir))
        if f.lower().endswith(".png")
    ]
    if not files:
        print(f"[preview] no patches in {args.patch_dir}")
        return 1

    rng = random.Random(args.seed)
    n = min(args.num_images, len(files))
    chosen = rng.sample(files, n)

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    cs = args.cell_size

    grid = np.zeros((rows * cs, cols * cs, 3), dtype=np.uint8)
    for i, path in enumerate(chosen):
        r, c = divmod(i, cols)
        try:
            im = Image.open(path).convert("RGB").resize((cs, cs), Image.BILINEAR)
            grid[r * cs : (r + 1) * cs, c * cs : (c + 1) * cs, :] = np.array(im)
        except Exception as e:
            print(f"[preview] skip {path}: {e}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    Image.fromarray(grid).save(args.output)
    print(f"[preview] wrote {args.output}  grid={rows}x{cols}  patches={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
