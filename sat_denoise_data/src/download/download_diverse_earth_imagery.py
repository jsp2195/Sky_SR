"""Download geographically diverse Earth imagery from OpenAerialMap.

Queries many distinct geographic bounding boxes to produce a source-diverse
dataset rather than clustering around a few locations.

Usage:
    python -m src.download.download_diverse_earth_imagery \
        --source oam \
        --output_dir data/raw_diverse \
        --target_sources 100 \
        --max_download_gb 30 \
        --seed 42 \
        --resume

NAIP is documented below but not automatically downloaded here (requires
Microsoft Planetary Computer credentials or an AWS token). See README for
NAIP instructions if OAM does not reach the target source count.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Optional

import requests
from tqdm import tqdm

OAM_META_URL = "https://api.openaerialmap.org/meta"

# ---------------------------------------------------------------------------
# 65+ geographically diverse bounding boxes for OAM queries.
# Each bbox is (west, south, east, north) in WGS84 degrees.
# Chosen to span all inhabited continents, avoiding clustering.
# ---------------------------------------------------------------------------
DIVERSE_BBOXES = [
    # North America
    {"region": "san_francisco_ca",    "bbox": [-122.5, 37.5, -122.0, 38.0]},
    {"region": "los_angeles_ca",      "bbox": [-118.5, 33.9, -118.0, 34.3]},
    {"region": "chicago_il",          "bbox": [-87.9,  41.8,  -87.5, 42.0]},
    {"region": "new_york_ny",         "bbox": [-74.1,  40.6,  -73.8, 40.8]},
    {"region": "miami_fl",            "bbox": [-80.5,  25.6,  -80.1, 25.9]},
    {"region": "seattle_wa",          "bbox": [-122.4, 47.5, -122.2, 47.7]},
    {"region": "denver_co",           "bbox": [-105.1, 39.6, -104.8, 39.9]},
    {"region": "houston_tx",          "bbox": [-95.5,  29.6,  -95.2, 29.9]},
    {"region": "new_orleans_la",      "bbox": [-90.1,  29.9,  -89.9, 30.1]},
    {"region": "portland_or",         "bbox": [-122.8, 45.4, -122.5, 45.6]},
    {"region": "mexico_city",         "bbox": [-99.3,  19.3,  -99.0, 19.5]},
    {"region": "toronto_canada",      "bbox": [-79.5,  43.6,  -79.3, 43.8]},
    # Caribbean / Central America
    {"region": "haiti_pap",           "bbox": [-72.5,  18.4,  -72.2, 18.6]},
    {"region": "dominican_republic",  "bbox": [-70.0,  18.4,  -69.8, 18.6]},
    {"region": "puerto_rico",         "bbox": [-66.3,  18.3,  -65.9, 18.5]},
    {"region": "guatemala_city",      "bbox": [-90.6,  14.6,  -90.4, 14.8]},
    {"region": "san_jose_cr",         "bbox": [-84.1,   9.9,  -83.9, 10.1]},
    # South America
    {"region": "bogota_colombia",     "bbox": [-74.2,   4.5,  -73.9,  4.8]},
    {"region": "lima_peru",           "bbox": [-77.2, -12.2,  -76.9,-12.0]},
    {"region": "sao_paulo_brazil",    "bbox": [-46.8, -23.7,  -46.5,-23.4]},
    {"region": "rio_de_janeiro",      "bbox": [-43.4, -23.0,  -43.1,-22.8]},
    {"region": "santiago_chile",      "bbox": [-70.7, -33.6,  -70.4,-33.3]},
    {"region": "buenos_aires",        "bbox": [-58.5, -34.7,  -58.3,-34.5]},
    {"region": "quito_ecuador",       "bbox": [-78.6,  -0.4,  -78.4, -0.2]},
    # Europe
    {"region": "london_uk",           "bbox": [-0.2,   51.4,   0.1,  51.6]},
    {"region": "paris_france",        "bbox": [2.2,    48.8,   2.5,  49.0]},
    {"region": "berlin_germany",      "bbox": [13.2,   52.4,  13.6,  52.6]},
    {"region": "amsterdam_nl",        "bbox": [4.7,    52.3,   5.0,  52.5]},
    {"region": "madrid_spain",        "bbox": [-3.8,   40.3,  -3.5,  40.5]},
    {"region": "rome_italy",          "bbox": [12.4,   41.8,  12.7,  42.0]},
    {"region": "warsaw_poland",       "bbox": [20.9,   52.1,  21.2,  52.3]},
    {"region": "kyiv_ukraine",        "bbox": [30.4,   50.3,  30.7,  50.5]},
    {"region": "istanbul_turkey",     "bbox": [28.9,   41.0,  29.2,  41.2]},
    {"region": "lisbon_portugal",     "bbox": [-9.2,   38.7,  -9.0,  38.8]},
    {"region": "athens_greece",       "bbox": [23.7,   37.9,  23.9,  38.1]},
    # Middle East / North Africa
    {"region": "cairo_egypt",         "bbox": [31.1,   30.0,  31.4,  30.2]},
    {"region": "amman_jordan",        "bbox": [35.9,   31.9,  36.0,  32.1]},
    {"region": "beirut_lebanon",      "bbox": [35.4,   33.8,  35.6,  33.9]},
    {"region": "baghdad_iraq",        "bbox": [44.2,   33.2,  44.5,  33.5]},
    {"region": "casablanca_morocco",  "bbox": [-7.7,   33.5,  -7.5,  33.7]},
    {"region": "tunis_tunisia",       "bbox": [10.0,   36.7,  10.3,  36.9]},
    {"region": "aleppo_syria",        "bbox": [37.1,   36.2,  37.3,  36.4]},
    # Sub-Saharan Africa
    {"region": "nairobi_kenya",       "bbox": [36.7,   -1.4,  37.1,  -1.1]},
    {"region": "dar_es_salaam_tz",    "bbox": [39.2,   -7.0,  39.4,  -6.7]},
    {"region": "kampala_uganda",      "bbox": [32.5,    0.3,  32.7,   0.5]},
    {"region": "accra_ghana",         "bbox": [-0.3,    5.5,   0.0,   5.7]},
    {"region": "lagos_nigeria",       "bbox": [3.3,     6.4,   3.6,   6.6]},
    {"region": "addis_ababa_et",      "bbox": [38.6,    9.0,  38.9,   9.2]},
    {"region": "johannesburg_sa",     "bbox": [28.0,  -26.3,  28.3, -26.1]},
    {"region": "kinshasa_drc",        "bbox": [15.2,   -4.5,  15.4,  -4.3]},
    {"region": "dakar_senegal",       "bbox": [-17.5,  14.6, -17.3,  14.8]},
    {"region": "maputo_mozambique",   "bbox": [32.5,  -26.0,  32.7, -25.8]},
    {"region": "kigali_rwanda",       "bbox": [30.0,   -1.9,  30.2,  -1.7]},
    # South / Southeast Asia
    {"region": "delhi_india",         "bbox": [77.0,   28.5,  77.3,  28.7]},
    {"region": "mumbai_india",        "bbox": [72.8,   18.9,  73.1,  19.1]},
    {"region": "chennai_india",       "bbox": [80.2,   13.0,  80.4,  13.2]},
    {"region": "dhaka_bangladesh",    "bbox": [90.3,   23.7,  90.5,  23.9]},
    {"region": "kathmandu_nepal",     "bbox": [85.2,   27.6,  85.4,  27.8]},
    {"region": "colombo_srilanka",    "bbox": [79.8,    6.8,  80.0,   7.0]},
    {"region": "yangon_myanmar",      "bbox": [96.1,   16.8,  96.3,  17.0]},
    {"region": "bangkok_thailand",    "bbox": [100.4,  13.7, 100.7,  13.9]},
    {"region": "ho_chi_minh_vn",      "bbox": [106.6,  10.7, 106.9,  10.9]},
    {"region": "jakarta_indonesia",   "bbox": [106.8,  -6.3, 107.0,  -6.1]},
    {"region": "manila_ph",           "bbox": [120.9,  14.5, 121.1,  14.7]},
    {"region": "singapore",           "bbox": [103.7,   1.2, 104.0,   1.4]},
    {"region": "kabul_afghanistan",   "bbox": [69.1,   34.5,  69.3,  34.7]},
    {"region": "karachi_pakistan",    "bbox": [67.0,   24.8,  67.2,  25.0]},
    # East Asia
    {"region": "tokyo_japan",         "bbox": [139.6,  35.6, 139.9,  35.8]},
    {"region": "seoul_south_korea",   "bbox": [126.9,  37.5, 127.2,  37.7]},
    # Oceania
    {"region": "sydney_australia",    "bbox": [151.1, -33.9, 151.3, -33.7]},
    {"region": "melbourne_australia", "bbox": [144.9, -37.9, 145.1, -37.7]},
    {"region": "auckland_nz",         "bbox": [174.7, -36.9, 174.9, -36.7]},
    # Pacific islands / humanitarian hotspots
    {"region": "vanuatu",             "bbox": [168.3, -17.7, 168.5, -17.5]},
    {"region": "tacloban_ph",         "bbox": [125.0,  11.2, 125.2,  11.4]},
]


# ---------------------------------------------------------------------------
# NAIP note (not auto-downloaded)
# ---------------------------------------------------------------------------
NAIP_NOTE = """
NAIP (National Agriculture Imagery Program) offers 1m aerial imagery for
the continental US. Access options:
  - Microsoft Planetary Computer STAC: requires account at planetarycomputer.microsoft.com
    pip install planetary-computer pystac-client
    Then query: https://planetarycomputer.microsoft.com/api/stac/v1
  - AWS open data (no credentials needed for us-east-2):
    s3://naip-source/  (requester-pays)
  - USDA EarthExplorer: https://earthexplorer.usgs.gov/
Place downloaded NAIP GeoTIFFs in data/raw_diverse/ and rerun the patch builder.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fetch_oam_meta(
    bbox: list[float],
    limit: int,
    gsd_to: Optional[float],
    timeout: float,
) -> list[dict]:
    params: dict[str, object] = {"limit": limit, "has_tiled": "true"}
    params["bbox"] = ",".join(f"{x:.6f}" for x in bbox)
    if gsd_to is not None:
        params["gsd_to"] = gsd_to
    r = requests.get(OAM_META_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json().get("results", [])


def head_size_mb(url: str, timeout: float) -> Optional[float]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        if r.status_code not in (200, 206):
            return None
        n = r.headers.get("Content-Length")
        return int(n) / (1024 * 1024) if n else None
    except Exception:
        return None


def download_file(url: str, out_path: str, timeout: float) -> Optional[float]:
    """Download url -> out_path. Returns MB written, or None on failure."""
    tmp = out_path + ".part"
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with open(tmp, "wb") as f:
                pbar = tqdm(total=total or None, unit="B", unit_scale=True,
                            desc=os.path.basename(out_path), leave=False)
                written = 0
                for chunk in r.iter_content(1 << 16):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)
                    pbar.update(len(chunk))
                pbar.close()
        os.replace(tmp, out_path)
        return written / (1024 * 1024)
    except Exception as e:
        print(f"[diverse] download error: {e}")
        if os.path.exists(tmp):
            os.unlink(tmp)
        return None


def safe_filename(item: dict, idx: int) -> str:
    url = item.get("uuid") or ""
    base = os.path.basename(url.split("?")[0]) or f"oam_{idx:04d}.tif"
    if not base.lower().endswith((".tif", ".tiff")):
        base += ".tif"
    return base


def load_source_manifest(path: str) -> dict[str, dict]:
    """Return {source_id -> record} for already-written sources."""
    existing: dict[str, dict] = {}
    if not os.path.exists(path):
        return existing
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    existing[rec["source_id"]] = rec
                except Exception:
                    continue
    return existing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download geographically diverse aerial imagery from OAM."
    )
    ap.add_argument("--source", choices=["oam", "naip", "both"], default="oam")
    ap.add_argument("--output_dir", default="data/raw_diverse")
    ap.add_argument("--manifest_out",
                    default="data/manifests/raw_diverse_sources.jsonl")
    ap.add_argument("--target_sources", type=int, default=100,
                    help="stop after this many source scenes are on disk")
    ap.add_argument("--max_download_gb", type=float, default=30.0,
                    help="stop before total downloaded exceeds this many GB")
    ap.add_argument("--max_mb_per_file", type=float, default=400.0,
                    help="skip individual files larger than this many MB")
    ap.add_argument("--limit_per_bbox", type=int, default=5,
                    help="OAM API results requested per bbox query")
    ap.add_argument("--gsd_to", type=float, default=None,
                    help="max GSD filter for OAM (None=no filter)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--resume", action="store_true",
                    help="skip sources already in the manifest")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.source in ("naip", "both"):
        print("[diverse] NAIP auto-download is not implemented.")
        print(NAIP_NOTE)
        if args.source == "naip":
            return 2

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest_out), exist_ok=True)

    existing = load_source_manifest(args.manifest_out) if args.resume else {}
    print(f"[diverse] resume={args.resume}  already in manifest: {len(existing)}")

    rng = random.Random(args.seed)
    bboxes = list(DIVERSE_BBOXES)
    rng.shuffle(bboxes)

    # Count sources already on disk.
    on_disk = len([f for f in os.listdir(args.output_dir)
                   if f.lower().endswith((".tif", ".tiff"))])
    total_gb_downloaded = 0.0
    n_downloaded = on_disk
    n_skipped_size = 0
    n_skipped_exists = 0
    n_api_errors = 0

    manifest_f = open(args.manifest_out, "a")

    def flush_record(rec: dict) -> None:
        manifest_f.write(json.dumps(rec) + "\n")
        manifest_f.flush()

    print(f"[diverse] target_sources={args.target_sources}  "
          f"max_gb={args.max_download_gb}  bboxes={len(bboxes)}")

    for bbox_info in bboxes:
        region = bbox_info["region"]
        bbox = bbox_info["bbox"]

        if n_downloaded >= args.target_sources:
            print(f"[diverse] reached target_sources={args.target_sources}, stopping")
            break
        if total_gb_downloaded >= args.max_download_gb:
            print(f"[diverse] reached max_download_gb={args.max_download_gb:.1f}, stopping")
            break

        print(f"[diverse] querying {region} {bbox}")
        try:
            results = fetch_oam_meta(bbox, args.limit_per_bbox, args.gsd_to,
                                     args.timeout)
        except Exception as e:
            print(f"[diverse] API error for {region}: {e}")
            n_api_errors += 1
            time.sleep(1.0)
            continue

        if not results:
            print(f"[diverse] no results for {region}")
            time.sleep(0.5)
            continue

        print(f"[diverse] {len(results)} candidates for {region}")
        for item in results:
            if n_downloaded >= args.target_sources:
                break
            if total_gb_downloaded >= args.max_download_gb:
                break

            url = item.get("uuid") or ""
            if not url:
                continue

            source_id = item.get("_id") or item.get("uuid") or url
            if source_id in existing:
                n_skipped_exists += 1
                continue

            fname = safe_filename(item, n_downloaded)
            out_path = os.path.join(args.output_dir, fname)
            if os.path.exists(out_path):
                n_skipped_exists += 1
                n_downloaded += 1
                # Add to manifest if missing
                if source_id not in existing:
                    rec = _make_record(item, source_id, out_path, fname, region)
                    flush_record(rec)
                    existing[source_id] = rec
                continue

            size_mb = head_size_mb(url, args.timeout)
            gsd = item.get("gsd")
            title = (item.get("title") or "")[:60]
            print(f"  [{region}] gsd={gsd}  size={size_mb}  {title}")

            if size_mb is not None and size_mb > args.max_mb_per_file:
                print(f"  skip: {size_mb:.1f} MB > max_mb={args.max_mb_per_file}")
                n_skipped_size += 1
                continue

            if args.dry_run:
                print(f"  dry_run: would download {url}")
                n_downloaded += 1
                continue

            mb = download_file(url, out_path, args.timeout)
            if mb is None:
                print(f"  download failed: {url}")
                continue

            total_gb_downloaded += mb / 1024.0
            n_downloaded += 1
            rec = _make_record(item, source_id, out_path, fname, region)
            flush_record(rec)
            existing[source_id] = rec
            print(f"  OK  {fname}  {mb:.1f} MB  "
                  f"total={total_gb_downloaded:.2f} GB  n={n_downloaded}")
            time.sleep(0.3)

        time.sleep(0.5)

    manifest_f.close()

    on_disk_final = len([f for f in os.listdir(args.output_dir)
                         if f.lower().endswith((".tif", ".tiff"))])
    manifest_count = len(load_source_manifest(args.manifest_out))

    print("\n[diverse] === SUMMARY ===")
    print(f"  source GeoTIFFs on disk:  {on_disk_final}")
    print(f"  manifest records:         {manifest_count}")
    print(f"  new downloads this run:   {n_downloaded - on_disk}")
    print(f"  skipped (exists):         {n_skipped_exists}")
    print(f"  skipped (too large):      {n_skipped_size}")
    print(f"  API errors:               {n_api_errors}")
    print(f"  total downloaded this run:{total_gb_downloaded:.2f} GB")
    print(f"  manifest:                 {args.manifest_out}")
    if on_disk_final < args.target_sources:
        print(f"\n[diverse] NOTE: only {on_disk_final} sources acquired vs "
              f"target {args.target_sources}.")
        print("  OAM coverage is sparse in some queried regions.")
        print("  Options to increase diversity:")
        print("  1. Re-run with --gsd_to 2.0 or no GSD filter.")
        print("  2. Download NAIP imagery (see README).")
        print(NAIP_NOTE)
    return 0


def _make_record(item: dict, source_id: str, out_path: str,
                 fname: str, region: str) -> dict:
    geojson = item.get("geojson") or {}
    bbox = None
    if geojson.get("type") == "Feature":
        coords = (geojson.get("geometry") or {}).get("coordinates") or []
        if coords:
            try:
                flat = [pt for ring in coords for pt in ring]
                xs = [p[0] for p in flat]
                ys = [p[1] for p in flat]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            except Exception:
                pass

    return {
        "source_id": source_id,
        "local_path": os.path.relpath(out_path),
        "data_source": "OAM",
        "url_or_asset_href": item.get("uuid"),
        "region": region,
        "bbox": bbox,
        "crs": "EPSG:4326",
        "width": item.get("properties", {}).get("width") if isinstance(item.get("properties"), dict) else None,
        "height": item.get("properties", {}).get("height") if isinstance(item.get("properties"), dict) else None,
        "acquisition_date": item.get("acquisition_start") or item.get("uploaded_at"),
        "gsd": item.get("gsd"),
        "title": item.get("title"),
        "license": item.get("license"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
