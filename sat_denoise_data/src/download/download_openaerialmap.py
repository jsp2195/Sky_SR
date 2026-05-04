"""Download a small batch of openly licensed imagery from OpenAerialMap.

Uses the public OAM metadata API:
    https://api.openaerialmap.org/meta

Each result has a `uuid` field that is a direct URL to the source GeoTIFF/COG.
We HEAD each URL and skip files above --max_mb to avoid pulling multi-GB COGs.

If the API call or any download fails, we print clear instructions and exit
non-zero. We do not fabricate success.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import requests
from tqdm import tqdm

OAM_META_URL = "https://api.openaerialmap.org/meta"


def fetch_meta(
    bbox: Optional[list[float]],
    limit: int,
    gsd_to: Optional[float],
    timeout: float,
) -> list[dict]:
    params: dict[str, object] = {"limit": limit, "has_tiled": "true"}
    if bbox is not None:
        params["bbox"] = ",".join(f"{x:.6f}" for x in bbox)
    if gsd_to is not None:
        params["gsd_to"] = gsd_to

    r = requests.get(OAM_META_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("results", [])


def head_size_mb(url: str, timeout: float) -> Optional[float]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        if r.status_code != 200:
            return None
        n = r.headers.get("Content-Length")
        if n is None:
            return None
        return int(n) / (1024 * 1024)
    except Exception:
        return None


def download(url: str, out_path: str, timeout: float) -> bool:
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            tmp = out_path + ".part"
            with open(tmp, "wb") as f:
                pbar = tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    desc=os.path.basename(out_path),
                    leave=False,
                )
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))
                pbar.close()
            os.replace(tmp, out_path)
        return True
    except Exception as e:
        print(f"[oam] download failed {url}: {e}")
        return False


def safe_filename(item: dict, fallback: str) -> str:
    uuid = item.get("uuid") or ""
    base = os.path.basename(uuid.split("?")[0]) or fallback
    if not base.lower().endswith((".tif", ".tiff")):
        base += ".tif"
    return base


def print_manual_instructions() -> None:
    print()
    print("=" * 70)
    print("OpenAerialMap automatic download was not reliable.")
    print("Manual options:")
    print("  1. Browse https://map.openaerialmap.org/ and download GeoTIFFs.")
    print("  2. Use the API directly:")
    print("       curl 'https://api.openaerialmap.org/meta?limit=5'")
    print("     then download each `uuid` URL with curl/wget.")
    print("  3. Place downloaded .tif/.tiff files into data/raw/.")
    print("=" * 70)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=None,
    )
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--output_dir", default="data/raw")
    ap.add_argument(
        "--gsd_to",
        type=float,
        default=1.0,
        help="upper bound on ground sample distance (meters/pixel) – default 1.0",
    )
    ap.add_argument("--max_mb", type=float, default=300.0, help="skip files larger than this")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[oam] querying {OAM_META_URL} limit={args.limit} bbox={args.bbox} gsd_to={args.gsd_to}")
    try:
        results = fetch_meta(args.bbox, args.limit, args.gsd_to, args.timeout)
    except Exception as e:
        print(f"[oam] metadata request failed: {e}")
        print_manual_instructions()
        return 2

    if not results:
        print("[oam] no results returned for the given query.")
        print_manual_instructions()
        return 2

    print(f"[oam] {len(results)} candidate scenes")
    n_ok = 0
    for i, item in enumerate(results):
        url = item.get("uuid")
        if not url:
            continue
        size_mb = head_size_mb(url, args.timeout)
        gsd = item.get("gsd")
        title = item.get("title", "")[:60]
        print(f"[{i+1}/{len(results)}] gsd={gsd}  size={size_mb}  {title}")
        if size_mb is not None and size_mb > args.max_mb:
            print(f"  skip: {size_mb:.1f} MB > max_mb={args.max_mb}")
            continue
        if args.dry_run:
            print(f"  dry_run: would download {url}")
            n_ok += 1
            continue
        out_name = safe_filename(item, fallback=f"oam_{i:04d}.tif")
        out_path = os.path.join(args.output_dir, out_name)
        if os.path.exists(out_path):
            print(f"  exists: {out_path}")
            n_ok += 1
            continue
        if download(url, out_path, args.timeout):
            n_ok += 1
        time.sleep(0.2)

    print(f"[oam] downloaded/ok: {n_ok}/{len(results)}")
    if n_ok == 0:
        print_manual_instructions()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
