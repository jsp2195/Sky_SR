"""Visual sanity check for synthetic degradations.

Loads a few clean patches and applies the same synthetic degradation code path
used by training. Saves a single grid PNG.

This is purely for visualization. No model is trained here.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.patch_degradation_dataset import degrade, resolve_profile


def pil_to_tensor(im: Image.Image) -> torch.Tensor:
    arr = np.array(im.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1.0


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    arr = ((t.clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    return arr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_dir", default="data/patches")
    ap.add_argument("--output", default="data/previews/degradation_grid.png")
    ap.add_argument("--num_rows", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--degradation_profile", choices=["balanced", "hard"], default="balanced")
    args = ap.parse_args()

    files = [
        os.path.join(args.patch_dir, f)
        for f in sorted(os.listdir(args.patch_dir))
        if f.lower().endswith(".png")
    ]
    if not files:
        print(f"[degradation] no patches in {args.patch_dir}")
        return 1

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    chosen = rng.sample(files, min(args.num_rows, len(files)))

    probs, strengths = resolve_profile(args.degradation_profile)
    cols = ["clean"] + [k for k, v in probs.items() if float(v) > 0.0]
    n_cols = len(cols)

    sample = Image.open(chosen[0]).convert("RGB")
    cs = sample.size[0]

    grid = np.zeros((len(chosen) * cs, n_cols * cs, 3), dtype=np.uint8)

    for r, path in enumerate(chosen):
        im = Image.open(path).convert("RGB")
        clean = pil_to_tensor(im)
        variants = [tensor_to_uint8(clean)]
        for c, kind in enumerate(cols[1:], start=1):
            torch.manual_seed(args.seed + r * 100 + c)
            degraded, _ = degrade(clean, kind, rng=random.Random(args.seed + r * 100 + c), strengths=strengths)
            variants.append(tensor_to_uint8(degraded))
        for c, v in enumerate(variants):
            grid[r * cs : (r + 1) * cs, c * cs : (c + 1) * cs, :] = v

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(grid).save(args.output)
    print(
        f"[degradation] wrote {args.output}  profile={args.degradation_profile}  "
        f"rows={len(chosen)}  cols={cols}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
