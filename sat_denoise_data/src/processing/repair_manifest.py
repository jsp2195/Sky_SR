"""Repair the patch manifest.

Two safe operations:
    1. Drop rows whose `patch_path` does not resolve to an existing file.
    2. Drop duplicate rows (same `patch_path`); the first occurrence wins.

Read-only on PNG files: this script never deletes, modifies, or rebuilds
patches.

Usage:
    python -m src.processing.repair_manifest \
        --manifest data/manifests/patches.jsonl \
        --patch_dir data/patches
"""

from __future__ import annotations

import argparse
import json
import os
import shutil


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/patches.jsonl")
    ap.add_argument("--patch_dir", default="data/patches")
    ap.add_argument("--backup_suffix", default=".bak")
    return ap.parse_args()


def resolve(path: str, patch_dir: str) -> str:
    if os.path.exists(path):
        return path
    return os.path.join(patch_dir, os.path.basename(path))


def main() -> int:
    args = parse_args()
    manifest = args.manifest
    patch_dir = args.patch_dir
    backup = manifest + args.backup_suffix

    if not os.path.exists(manifest):
        print(f"[repair] manifest not found: {manifest}")
        return 1

    # Count PNGs in patch_dir.
    png_count = 0
    if os.path.isdir(patch_dir):
        for name in os.listdir(patch_dir):
            if name.lower().endswith(".png"):
                png_count += 1

    kept: list[str] = []
    seen_paths: set[str] = set()
    dropped_missing = 0
    dropped_duplicate = 0
    dropped_invalid = 0
    original = 0
    with open(manifest, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            original += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                dropped_invalid += 1
                continue
            path = rec.get("patch_path") or rec.get("path") or ""
            if not path:
                dropped_invalid += 1
                continue
            full = resolve(path, patch_dir)
            if not os.path.exists(full):
                dropped_missing += 1
                continue
            if path in seen_paths:
                dropped_duplicate += 1
                continue
            seen_paths.add(path)
            kept.append(line)
    dropped = dropped_missing + dropped_duplicate + dropped_invalid

    # Backup before overwriting.
    shutil.copy2(manifest, backup)
    tmp = manifest + ".tmp"
    with open(tmp, "w") as f:
        for line in kept:
            f.write(line + "\n")
    os.replace(tmp, manifest)

    print(f"[repair] manifest:           {manifest}")
    print(f"[repair] backup written:     {backup}")
    print(f"[repair] original rows:      {original}")
    print(f"[repair] repaired rows:      {len(kept)}")
    print(f"[repair] dropped (total):    {dropped}")
    print(f"[repair]   missing-file:     {dropped_missing}")
    print(f"[repair]   duplicate-path:   {dropped_duplicate}")
    print(f"[repair]   invalid/empty:    {dropped_invalid}")
    print(f"[repair] PNG files found:    {png_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
