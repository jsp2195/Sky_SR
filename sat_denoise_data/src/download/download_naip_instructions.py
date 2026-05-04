"""Print exact manual instructions for acquiring NAIP imagery."""

from __future__ import annotations

INSTRUCTIONS = """
NAIP (National Agriculture Imagery Program) – manual acquisition guide
======================================================================

NAIP is ~1 m or 0.6 m GSD aerial imagery of the continental US, RGB or RGBNIR,
released into the public domain. It is excellent for proxy training of
denoising / super-resolution models.

1. Recommended access points (free):

   a) USDA Box DOQQ archive (per-state, per-year GeoTIFFs):
        https://nrcs.app.box.com/v/naip
      Browse to a state and year, download .tif tiles directly.

   b) AWS Open Data – Esri NAIP COG bucket (requester-pays-free public mirror):
        s3://naip-source/  (region us-west-2, "Requester Pays")
        s3://naip-analytic/
      Listing example (with AWS CLI):
        aws s3 ls s3://naip-source/ca/2022/60cm/ --request-payer requester

   c) Microsoft Planetary Computer STAC catalog:
        https://planetarycomputer.microsoft.com/dataset/naip
      Search via the STAC API; assets include `image` (COG) URLs that can be
      streamed or downloaded.

   d) USGS EarthExplorer (requires free login):
        https://earthexplorer.usgs.gov/
      Dataset: "NAIP" under "Aerial Imagery".

2. File types accepted by this pipeline:
     .tif / .tiff  (GeoTIFF or COG)  <- preferred, preserves CRS
     .png / .jpg                     <- works, but no geo metadata

3. Where to put the files:
     sat_denoise_data/data/raw/

   You can drop entire NAIP DOQQ tiles in there. They are typically
   ~7500x6500 pixels each, which yields hundreds of clean 256x256 patches.

4. Then build patches:
     python -m src.processing.build_patch_dataset \\
       --input_dir data/raw \\
       --output_dir data/patches \\
       --manifest data/manifests/patches.jsonl \\
       --patch_size 256 --stride 256 \\
       --max_patches 1000 --rgb_only --skip_blank --resume

5. Licensing:
   NAIP imagery is in the public domain (USDA-FSA-APFO). No attribution
   required, but recording the source year/state in your dataset card is
   good practice.
"""


def main() -> int:
    print(INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
