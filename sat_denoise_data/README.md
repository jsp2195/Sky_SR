# sat_denoise_data

Baseline open-source Earth imagery patch dataset for a future DDPM
denoising / restoration / super-resolution project. **Data only** – no model
code, no diffusion code, no InSAR code, no training loop.

The dataset built here is a **proxy** of clean Earth-surface patches. It is
intended to validate the data pipeline and the denoising/restoration training
stack before swapping in real InSAR data.

## Layout

```
sat_denoise_data/
  README.md
  requirements.txt
  data/
    raw/         <- drop GeoTIFF/TIFF/PNG/JPG sources here
    patches/     <- 256x256 PNG patches written by the builder
    manifests/   <- patches.jsonl
    previews/    <- patch_grid.png, degradation_grid.png
  src/
    download/
      download_openaerialmap.py
      download_naip_instructions.py
    processing/
      build_patch_dataset.py
      inspect_images.py
      make_preview_grid.py
      make_degradation_preview.py
    utils/
      image_io.py
      geo_io.py
```

## 1. Install

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`rasterio` is preferred for GeoTIFFs but optional – the pipeline falls back
to PIL when rasterio is unavailable.

## 2. Get imagery into `data/raw/`

Try one of the following, in order of convenience:

### a) OpenAerialMap (programmatic)

```
python -m src.download.download_openaerialmap \
  --limit 5 \
  --gsd_to 1.0 \
  --max_mb 300 \
  --output_dir data/raw
```

Optional bounding box (west south east north, WGS84):

```
python -m src.download.download_openaerialmap \
  --bbox -106.4 35.6 -106.2 35.9 \
  --limit 5 \
  --output_dir data/raw
```

The script hits `https://api.openaerialmap.org/meta`, HEADs each candidate
URL, skips files larger than `--max_mb`, and downloads the rest as GeoTIFFs.
If the API or downloads fail it prints manual instructions and exits non-zero
(no fake success).

### b) NAIP (manual, recommended for serious volume)

```
python -m src.download.download_naip_instructions
```

Prints exact instructions for downloading NAIP DOQQ GeoTIFFs from USDA Box,
the AWS open-data NAIP buckets, Microsoft Planetary Computer, or USGS
EarthExplorer. Drop the resulting `.tif` files into `data/raw/`.

### c) Anything else

Any openly licensed Earth imagery works – `.tif`, `.tiff`, `.png`, `.jpg`
under `data/raw/` (recursively) will be picked up.

## 3. Inspect raw imagery

```
python -m src.processing.inspect_images --input_dir data/raw
```

Reports type, dimensions, bands, dtype, sampled min/max, CRS, and transform.

## 4. Build a 1k-patch smoke dataset (source-balanced)

```
python -m src.processing.build_patch_dataset \
  --input_dir data/raw \
  --output_dir data/patches \
  --manifest data/manifests/patches.jsonl \
  --patch_size 256 \
  --stride 256 \
  --max_patches 1000 \
  --max_patches_per_source 125 \
  --rgb_only \
  --skip_blank \
  --resume
```

`--max_patches_per_source` caps how many valid patches are taken from any
single source image, so a multi-gigapixel scene cannot dominate the dataset.
Omit the flag for unlimited per-source patches.

Behaviour:

- Reads GeoTIFFs with rasterio, everything else with PIL.
- Forces RGB uint8; uint16 sources scaled by 256, float sources p2-p98 stretched.
- Drops nodata-heavy patches (`--max_nodata_fraction`, default 5%).
- Drops near-uniform patches (`--min_std`, default 5).
- With `--skip_blank`, also drops patches that are mostly pure black or pure white.
- `--resume` re-reads the manifest, skips `patch_id`s already present, and
  carries the per-source counts forward so the cap is honored across runs.
- Iteration continues across all source files until either `--max_patches`
  is reached or every source has either hit `--max_patches_per_source` or
  been exhausted.
- A summary at the end prints total patches, sources used / available, and
  patches per source.
- Patches are PNG. The manifest is JSONL with one record per patch:

```json
{"patch_id": "ab12cd34ef_000256_001024",
 "patch_path": "data/patches/scene__ab12cd34ef_000256_001024.png",
 "source_path": "data/raw/scene.tif",
 "row": 256, "col": 1024,
 "width": 256, "height": 256,
 "crs": "EPSG:26913",
 "transform": [1.0, 0.0, 332512.0, 0.0, -1.0, 3987456.0],
 "bbox": [332512.0, 3987200.0, 332768.0, 3987456.0]}
```

`crs`, `transform`, `bbox` are `null` for non-georeferenced inputs.

## 5. Make a preview grid

```
python -m src.processing.make_preview_grid \
  --patch_dir data/patches \
  --output data/previews/patch_grid.png \
  --num_images 64
```

## 6. Optional: visual sanity check on synthetic degradations

```
python -m src.processing.make_degradation_preview \
  --patch_dir data/patches \
  --output data/previews/degradation_grid.png
```

Columns: `clean | gaussian | speckle | blur+ds+us | blur+ds+noise+us`.
This is a visualization only – no model is trained.

## Scaling beyond the smoke set

The same builder scales to 100k–1M patches by:

- adding more sources to `data/raw/` (NAIP DOQQs are the cheapest path),
- raising `--max_patches`,
- optionally lowering `--stride` below `--patch_size` for overlap (only if
  you need more samples per scene; not recommended for evaluation patches).

Reading from COGs over HTTPS without local download is possible with
rasterio + GDAL `/vsicurl/`, but is intentionally not wired into the smoke
pipeline to keep this layer simple.

## Model Baselines

Three restoration modes on the structured RGB proxy:

1. Deterministic residual U-Net: `degraded -> residual`,
   `clean_pred = degraded + residual`.
2. Conditional DDPM (full-image): U-Net predicts epsilon from
   `concat(x_t, degraded)` (6 channels in, 3 channels out), trained with
   `MSE(pred_eps, eps)`. Diffusion target is the clean image.
3. Residual-DDPM refinement: a DDPM that *refines* a frozen residual U-Net.
   The diffusion target is `clean - restored_base` and the conditioning is
   `concat(degraded, restored_base)` with optional mask/reliability channels.
   Final restored image is `restored_base + sampled_residual`.
   Selected by `diffusion_target: residual` plus `--base_ckpt`.
4. Masked-completion DDPM: a DDPM used only where a degradation mask or
   low-reliability map marks the region as eligible. Conditioning is
   `concat(degraded, restored_base, degradation_mask, reliability_map)`.
   The target is either `clean - restored_base` or `clean`, selected by
   `masked_completion_target`. Final composition preserves the U-Net base
   outside the mask.

Train U-Net:

```
python -m src.training.train_unet_denoiser \
  --config configs/train_unet_denoiser.yaml
```

Evaluate U-Net:

```
python -m src.evaluation.eval_unet_denoiser \
  --ckpt outputs/unet_smoke/ckpt_best.pt \
  --manifest data/manifests/patches.jsonl \
  --patch_dir data/patches \
  --output_dir outputs/unet_eval \
  --max_samples 32
```

Train DDPM:

```
python -m src.training.train_ddpm_denoiser \
  --config configs/train_ddpm_denoiser.yaml
```

Evaluate DDPM:

```
python -m src.evaluation.eval_ddpm_denoiser \
  --ckpt outputs/ddpm_smoke/ckpt_best.pt \
  --manifest data/manifests/patches.jsonl \
  --patch_dir data/patches \
  --output_dir outputs/ddpm_eval \
  --max_samples 16 \
  --sampling_steps 100
```

Train residual-DDPM (refinement on top of a frozen residual U-Net):

```
python -m src.training.train_ddpm_denoiser \
  --config configs/train_ddpm_residual_denoiser.yaml \
  --base_ckpt outputs/unet_diverse_50k/ckpt_best.pt
```

Evaluate residual-DDPM. The mode is read from the saved DDPM config; the
base U-Net checkpoint must be supplied explicitly (or already present in
the saved config):

```
python -m src.evaluation.eval_ddpm_denoiser \
  --ckpt outputs/ddpm_residual_smoke/ckpt_best.pt \
  --base_ckpt outputs/unet_diverse_50k/ckpt_best.pt \
  --manifest data/manifests/patches_diverse.jsonl \
  --patch_dir data/patches_diverse \
  --output_dir outputs/ddpm_residual_eval \
  --max_samples 16 \
  --sampling_steps 100
```

Residual-mode evaluation reports three metric sets - degraded vs clean,
U-Net base vs clean, DDPM-refined vs clean - in `metrics.json` and prints
per-degradation-type summaries `in -> base -> out`. Diagnostic grids have
five columns: degraded | U-Net base | DDPM refined | clean | abs error.

The gated multiband RGB residual-DDPM proxy mirrors the future processed-InSAR
setup without implementing InSAR-specific physics:

- RGB degraded image corresponds to processed InSAR phase/displacement.
- For RGB, `degradation_mask` and `reliability_map` are synthetic proxies.
- For processed InSAR, `reliability_map` corresponds naturally to coherence
  or other confidence indicators, and `degradation_mask` corresponds to
  valid/invalid, low-confidence, or ambiguous regions.
- Diffusion should only modify low-confidence or missing regions through
  gated residual refinement or masked completion.
- DDPM should not be treated as the primary global denoiser/restorer for this
  RGB benchmark.

No InSAR-specific losses or physics are implemented in this step.

Train masked-completion DDPM:

```
python -m src.training.train_ddpm_denoiser \
  --config configs/train_ddpm_masked_completion_hard.yaml \
  --base_ckpt outputs/unet_hard_50k_b6w8/ckpt_best.pt
```

Train the localized masked-completion benchmark. This uses only localized
`mask_dropout` masks and rejects masks outside the configured area range:

```
python -m src.training.train_ddpm_denoiser \
  --config configs/train_ddpm_masked_completion_localized.yaml \
  --base_ckpt outputs/unet_hard_50k_b6w8/ckpt_best.pt
```

Preview localized masks:

```
python -m src.evaluation.preview_masked_completion_localized \
  --config configs/train_ddpm_masked_completion_localized.yaml \
  --output outputs/previews/masked_completion_localized_preview.png
```

Evaluate masked-completion DDPM with DDIM, mask metrics, and optional
uncertainty sampling:

```
python -m src.evaluation.eval_ddpm_denoiser \
  --ckpt outputs/ddpm_masked_completion_hard/ckpt_best.pt \
  --base_ckpt outputs/unet_hard_50k_b6w8/ckpt_best.pt \
  --manifest data/manifests/patches_diverse.jsonl \
  --patch_dir data/patches_diverse \
  --output_dir outputs/ddpm_masked_completion_hard_eval \
  --max_samples 16 \
  --sampling_mode ddim \
  --sampling_steps 50 \
  --num_samples_per_input 4
```

## Diagnostics

Each training run writes:

- `outputs/<run>/train_log.jsonl` – per-step train rows and per-epoch val rows
  (`epoch`, `train_loss`, `val_loss`, `degraded_psnr`, `restored_psnr`,
  `degraded_mae`, `restored_mae`, and `degradation_type_counts`).
- `outputs/<run>/samples/epoch_<NNNN>.png` – fixed validation samples shown
  every epoch as `degraded | restored | clean | abs-error`.
- `outputs/<run>/samples/step_<NNNNNN>.png` – rolling training samples.
- `outputs/<run>/plots/loss_curve.png`
- `outputs/<run>/plots/metric_curves.png` (U-Net only)
- `outputs/<run>/plots/degradation_type_distribution.png`
- `outputs/<run>/plots/degradation_param_histograms.png` (where parameters apply)

After training the U-Net, check:

- `outputs/unet_smoke/plots/loss_curve.png`
- `outputs/unet_smoke/plots/metric_curves.png`
- `outputs/unet_smoke/plots/degradation_type_distribution.png`
- `outputs/unet_smoke/samples/epoch_*.png`
- `outputs/unet_eval/metrics.json` and `grouped_metrics_by_degradation_type.json`

After training the DDPM, check:

- `outputs/ddpm_smoke/plots/loss_curve.png`
- `outputs/ddpm_smoke/samples/epoch_*.png` (skipped automatically if sampling
  fails or takes too long)
- `outputs/ddpm_eval/metrics.json` and `grouped_metrics_by_degradation_type.json`

## What is intentionally NOT here

- No real LR/HR pairing.
- No InSAR-, Sentinel-2-, or Landsat-specific code.
- No wrapped phase, coherence maps, complex tensors, circular losses,
  phase unwrapping, deformation inversion, or InSAR super-resolution.
- No web UI.
