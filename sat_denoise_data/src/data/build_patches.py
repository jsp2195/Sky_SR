"""Thin wrapper around src.processing.build_patch_dataset.

Provides the command interface specified for the diverse dataset build:

    python -m src.data.build_patches \
        --input_dir data/raw_diverse \
        --output_dir data/patches_diverse \
        --manifest_out data/manifests/patches_diverse.jsonl \
        --patch_size 256 \
        --stride 128 \
        --max_patches 100000 \
        --max_patches_per_source 1000 \
        --rgb_only \
        --skip_blank

The only difference from src.processing.build_patch_dataset is the flag name
--manifest_out instead of --manifest. All other arguments are passed through.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build patch dataset from a directory of GeoTIFFs."
    )
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--manifest_out", required=True,
                    help="path to output JSONL manifest (alias for --manifest)")
    ap.add_argument("--patch_size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--max_patches", type=int, default=100000)
    ap.add_argument("--max_patches_per_source", type=int, default=None)
    ap.add_argument("--rgb_only", action="store_true")
    ap.add_argument("--skip_blank", action="store_true")
    ap.add_argument("--min_std", type=float, default=5.0)
    ap.add_argument("--max_nodata_fraction", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # Import the underlying builder and call it with remapped args.
    from src.processing.build_patch_dataset import main as _build_main
    import sys as _sys
    _sys.argv = [
        "build_patch_dataset",
        "--input_dir", args.input_dir,
        "--output_dir", args.output_dir,
        "--manifest", args.manifest_out,
        "--patch_size", str(args.patch_size),
        "--stride", str(args.stride),
        "--max_patches", str(args.max_patches),
        "--min_std", str(args.min_std),
        "--max_nodata_fraction", str(args.max_nodata_fraction),
    ]
    if args.max_patches_per_source is not None:
        _sys.argv += ["--max_patches_per_source", str(args.max_patches_per_source)]
    if args.rgb_only:
        _sys.argv.append("--rgb_only")
    if args.skip_blank:
        _sys.argv.append("--skip_blank")
    if args.resume:
        _sys.argv.append("--resume")

    return _build_main()


if __name__ == "__main__":
    raise SystemExit(main())
