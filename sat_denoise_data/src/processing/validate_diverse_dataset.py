"""Validate the diverse patch dataset and produce a diversity report.

Checks:
  - number of source GeoTIFFs in data/raw_diverse/
  - number of source scenes in manifest
  - patch count in data/patches_diverse/
  - patch manifest row count vs PNG count
  - per-source patch distribution (min/median/max/top-10)
  - overlap fraction from stride/patch_size
  - comparison against old 9-source dataset if available

Usage:
    python -m src.processing.validate_diverse_dataset
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_diverse_dir",    default="data/raw_diverse")
    ap.add_argument("--patches_diverse_dir",default="data/patches_diverse")
    ap.add_argument("--source_manifest",    default="data/manifests/raw_diverse_sources.jsonl")
    ap.add_argument("--patch_manifest",     default="data/manifests/patches_diverse.jsonl")
    ap.add_argument("--old_patch_manifest", default="data/manifests/patches.jsonl")
    ap.add_argument("--patch_size",  type=int, default=256)
    ap.add_argument("--stride",      type=int, default=128)
    return ap.parse_args()


def load_jsonl(path: str) -> list[dict]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    return records


def count_pngs(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0
    return sum(1 for f in os.listdir(directory) if f.lower().endswith(".png"))


def count_tifs(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0
    return sum(1 for f in os.listdir(directory)
               if f.lower().endswith((".tif", ".tiff")))


def source_counts_from_manifest(records: list[dict]) -> Counter:
    c: Counter = Counter()
    for r in records:
        src = r.get("source_path") or r.get("source_id") or "unknown"
        c[src] += 1
    return c


def median(xs: list) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("DIVERSE DATASET VALIDATION REPORT")
    print("=" * 60)

    # 1. Raw source files
    n_tifs = count_tifs(args.raw_diverse_dir)
    print(f"\n1. Raw source GeoTIFFs in {args.raw_diverse_dir}: {n_tifs}")

    # 2. Source manifest
    src_records = load_jsonl(args.source_manifest)
    print(f"2. Source manifest records:                  {len(src_records)}")
    if not os.path.exists(args.source_manifest):
        print(f"   MISSING: {args.source_manifest}")

    # 3. Patches
    n_pngs = count_pngs(args.patches_diverse_dir)
    patch_records = load_jsonl(args.patch_manifest)
    print(f"\n3. PNG patches in {args.patches_diverse_dir}: {n_pngs}")
    print(f"4. Patch manifest rows:                      {len(patch_records)}")
    if not os.path.exists(args.patch_manifest):
        print(f"   MISSING: {args.patch_manifest}")

    # 4. Manifest vs PNG count check
    match = n_pngs == len(patch_records) and len(patch_records) > 0
    print(f"\n5. Manifest rows == PNG count: {'YES' if match else 'NO'} "
          f"({len(patch_records)} vs {n_pngs})")

    # 5. Patch size / stride
    print(f"\n6. Patch size: {args.patch_size}")
    print(f"7. Stride:     {args.stride}")
    overlap = 1.0 - args.stride / args.patch_size
    print(f"8. Linear overlap fraction: {overlap:.2%} "
          f"(stride={args.stride} / patch_size={args.patch_size})")

    # 6. Per-source distribution
    if patch_records:
        src_counter = source_counts_from_manifest(patch_records)
        vals = list(src_counter.values())
        print(f"\n9. Sources used for patching: {len(src_counter)}")
        print(f"   Min patches/source:   {min(vals)}")
        print(f"   Median patches/source:{median(vals):.1f}")
        print(f"   Max patches/source:   {max(vals)}")
        print(f"\n10. Top 10 sources by patch count:")
        for src, cnt in sorted(src_counter.items(), key=lambda kv: -kv[1])[:10]:
            short = os.path.basename(src)[:60]
            print(f"    {cnt:6d}  {short}")
    else:
        print("\n9-10. No patch records to analyse.")

    # 7. Preview existence
    previews = [
        "data/previews/patch_grid_diverse.png",
        "data/previews/degradation_grid_diverse.png",
    ]
    print("\n11. Preview files:")
    for p in previews:
        exists = os.path.exists(p)
        print(f"    {'OK' if exists else 'MISSING'}  {p}")

    # 8. Compare against old dataset
    old_records = load_jsonl(args.old_patch_manifest)
    if old_records:
        old_sources = source_counts_from_manifest(old_records)
        print(f"\n12. OLD dataset comparison ({args.old_patch_manifest}):")
        print(f"    Old patches:        {len(old_records)}")
        print(f"    Old unique sources: {len(old_sources)}")
        new_n_sources = len(source_counts_from_manifest(patch_records)) if patch_records else 0
        print(f"    NEW unique sources: {new_n_sources}")
        ratio = new_n_sources / max(len(old_sources), 1)
        print(f"    Diversity ratio (new/old): {ratio:.1f}x")
        if ratio >= 5:
            print("    --> materially more diverse (>= 5x)")
        elif ratio >= 2:
            print("    --> more diverse (>= 2x) but not yet 100-source target")
        else:
            print("    --> similar diversity to old dataset")
    else:
        print("\n12. Old manifest not found; skipping comparison.")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
