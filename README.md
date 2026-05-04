# Sky_SR

**Satellite image restoration experiments for controlled degradation, deterministic recovery, and diffusion-based completion.**

Sky_SR is a research sandbox for building and stress-testing image restoration models on satellite-style RGB Earth imagery. The project takes clean image patches, applies controlled degradations that mimic common remote-sensing failure modes, trains restoration models, and evaluates whether the recovered imagery is visually plausible, quantitatively better, and spatially safe.

The central question is simple:

> When satellite imagery is corrupted by noise, blur, downsampling artifacts, haze-like bias, or missing local regions, how much can a deterministic restoration model recover, and where does conditional diffusion add value?

This repository answers that question with paired data, explicit degradation controls, reproducible training scripts, quantitative metrics, and labeled visual diagnostics.

---

## Why This Work Matters

Satellite and aerial imagery are rarely clean in practice. Resolution limits, atmospheric effects, sensor noise, compression artifacts, motion blur, resampling, and partial occlusion can all degrade downstream analysis. For many scientific and operational workflows, the problem is not just making images look better. The harder requirement is controlled restoration:

* recover useful structure without hallucinating globally,
* preserve trusted pixels where the input is already reliable,
* expose model failure modes visually,
* compare deterministic baselines against generative models honestly,
* and establish a clean benchmark before moving to more specialized remote-sensing modalities.

Sky_SR is designed around that standard. It is not a demo that only shows attractive samples. It is an experimental scaffold for asking which restoration method works, where it works, and whether it changes pixels it should not touch.

---

## Current Research Takeaway

The strongest result so far is the hard deterministic U-Net baseline. On the current paired RGB proxy benchmark, it substantially improves PSNR, MAE, and SSIM over the degraded input.

The diffusion path is most compelling as a constrained completion module rather than a global restoration replacement. In the masked-completion setting, the DDPM is allowed to modify only selected corrupted regions. The current evaluation verifies that the diffusion stage leaves pixels outside the allowed mask unchanged.

That distinction is the main value of the project:

* **U-Net:** strong global paired restoration baseline.
* **DDPM:** useful for constrained, masked, or gated generative completion.
* **Evaluation:** explicitly checks whether generated changes remain spatially safe.

---

## Latest Visual Outputs

The examples below come from the latest larger experiment outputs in this repository:

```text
sat_denoise_data/outputs/ddpm_masked_completion_hard_10k_eval_ddim
sat_denoise_data/outputs/unet_hard_50k_b6w8_eval
```

The masked-completion DDPM grids show the full restoration path:

```text
degraded | mask | reliability | U-Net base | diffusion final | clean | uncertainty
```

The white mask marks where DDPM completion is allowed. The evaluation verifies that the diffusion stage leaves pixels outside that region unchanged.

![Masked-completion DDPM eval grid 255](docs/assets/ddpm_masked_completion_hard_10k_eval_grid_255_labeled.png)

![Masked-completion DDPM eval grid 254](docs/assets/ddpm_masked_completion_hard_10k_eval_grid_254_labeled.png)

![Masked-completion DDPM eval grid 253](docs/assets/ddpm_masked_completion_hard_10k_eval_grid_253_labeled.png)

The U-Net evaluation grids show the deterministic restoration baseline:

```text
degraded | restored | clean | abs error
```

![Hard U-Net eval grid 000](docs/assets/unet_hard_50k_b6w8_eval_grid_000_labeled.png)

![Hard U-Net eval grid 001](docs/assets/unet_hard_50k_b6w8_eval_grid_001_labeled.png)

---

## Current Result Snapshot

### Hard U-Net Restoration

Evaluation run:

```text
sat_denoise_data/outputs/unet_hard_50k_b6w8_eval
```

| Metric | Degraded input | U-Net restored |  Change |
| ------ | -------------: | -------------: | ------: |
| PSNR   |          24.94 |          32.21 |   +7.26 |
| MAE    |         0.0487 |         0.0207 | -0.0280 |
| SSIM   |         0.6898 |         0.8948 | +0.2051 |

The U-Net result is the current best global reconstruction baseline. It removes much of the synthetic corruption while preserving the paired target structure.

### Masked-Completion DDPM

Evaluation run:

```text
sat_denoise_data/outputs/ddpm_masked_completion_hard_10k_eval_ddim
```

| Metric | Degraded input | U-Net base | Diffusion final |
| ------ | -------------: | ---------: | --------------: |
| PSNR   |          25.31 |      30.96 |           30.41 |
| MAE    |         0.0466 |     0.0223 |          0.0238 |
| SSIM   |         0.7785 |     0.9087 |          0.8891 |

Additional checks from the same run:

| Check                           |                                                                                      Value |
| ------------------------------- | -----------------------------------------------------------------------------------------: |
| Evaluation mode                 |                                                                        `masked_completion` |
| Degradation types               | `mixed_structured`, `blur_downsample_upsample`, `lowfreq_atmospheric_bias`, `mask_dropout` |
| Mean mask area                  |                                                                                   `89.14%` |
| Samples evaluated               |                                                                                      `512` |
| Sampler                         |                                                                                     `ddim` |
| Diffusion samples per input     |                                                                                        `4` |
| Outside-mask maximum difference |                                                                                      `0.0` |

The important result is not that DDPM beats the U-Net globally. It does not in this run. The important result is that the DDPM path can be constrained to operate only where completion is allowed. That makes it a useful experimental direction for restoration tasks where hallucination control matters more than unconstrained image enhancement.

---

## Method Overview

Sky_SR is organized around a four-stage experimental pipeline.

### 1. Build clean image patches

Open Earth-imagery sources are cut into 256 × 256 RGB patches. The manifest records patch path, source image, pixel window, and geospatial metadata when available.

Supported source formats include:

* GeoTIFF
* TIFF
* PNG
* JPG

The patch builder supports source balancing so the dataset does not rely entirely on dense overlapping crops from a small number of large scenes.

### 2. Apply controlled synthetic degradations

Training examples are generated from paired clean/degraded patches. Current degradation modes include:

* Gaussian noise,
* speckle-like noise,
* blur,
* downsample/upsample artifacts,
* low-frequency atmospheric-style bias,
* structured corruption,
* and localized mask dropout.

This controlled setup makes the benchmark interpretable: the clean target is known, the corruption process is explicit, and the restoration target is measurable.

### 3. Train restoration models

The main deterministic baseline is a residual U-Net. It predicts a correction that is added to the degraded input:

```text
clean_pred = degraded + residual
```

The diffusion variants are conditional DDPM models. They can predict either a clean image or a residual, and they can be conditioned on:

* degraded input,
* U-Net base restoration,
* degradation mask,
* reliability map,
* and optional gated residual controls.

### 4. Evaluate with metrics and labeled grids

Evaluation writes JSON metric summaries and labeled image grids. The grids are intentionally diagnostic. They are designed to make failure modes obvious rather than hide them behind aggregate scores.

Depending on the model, grids include:

* degraded input,
* restoration output,
* clean target,
* absolute error,
* mask,
* reliability map,
* U-Net base,
* diffusion final,
* and uncertainty.

---

## Model Families

This repository currently includes:

| Model / path                | Purpose                                                          |
| --------------------------- | ---------------------------------------------------------------- |
| `ResidualUNet`              | Deterministic residual restoration baseline                      |
| `DDPMUNet`                  | Conditional diffusion U-Net backbone                             |
| Full-image conditional DDPM | Predicts clean RGB images from degraded inputs                   |
| Residual-DDPM refinement    | Predicts a residual on top of a frozen U-Net base                |
| Gated residual-DDPM         | Restricts where sampled diffusion residuals can modify the image |
| Masked-completion DDPM      | Completes only regions selected by the degradation mask          |

The current evidence supports a conservative interpretation: deterministic restoration should remain the primary baseline, while diffusion should be evaluated as a constrained completion/refinement mechanism.

---

## Repository Layout

```text
sat_denoise_data/
  configs/                 Training configs for U-Net and DDPM variants
  data/                    Local imagery, patches, and manifests
  docs/                    Experiment notes and selected README assets
  outputs/                 Local training and evaluation artifacts
  src/
    data/                  Patch degradation dataset
    download/              Open imagery download helpers
    evaluation/            U-Net and DDPM evaluators
    models/                U-Net and diffusion model definitions
    processing/            Patch builders and preview tools
    training/              Training entrypoints
    utils/                 Metrics, checkpoints, grids, gates, and seeds
```

Large local artifacts are intentionally excluded from git:

* source imagery,
* generated patch datasets,
* training outputs,
* checkpoints,
* full evaluation directories.

Selected visual outputs are copied into `docs/assets/` so the README renders on GitHub without requiring the full local experiment directory.

---

## Setup

From the repository root:

```bash
cd sat_denoise_data
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision
```

`rasterio` is used for GeoTIFF handling. PNG and JPG paths can fall back to PIL-based loading for non-geospatial inputs.

---

## Build a Patch Dataset

Place source imagery under:

```text
sat_denoise_data/data/raw/
```

Inspect available imagery:

```bash
python -m src.processing.inspect_images --input_dir data/raw
```

Build a source-balanced patch dataset:

```bash
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

Create quick visual checks:

```bash
python -m src.processing.make_preview_grid \
  --patch_dir data/patches \
  --output data/previews/patch_grid.png \
  --num_images 64

python -m src.processing.make_degradation_preview \
  --patch_dir data/patches \
  --output data/previews/degradation_grid.png
```

---

## Train a U-Net Baseline

```bash
python -m src.training.train_unet_denoiser \
  --config configs/train_unet_denoiser.yaml
```

Evaluate it:

```bash
python -m src.evaluation.eval_unet_denoiser \
  --ckpt outputs/unet_smoke/ckpt_best.pt \
  --manifest data/manifests/patches.jsonl \
  --patch_dir data/patches \
  --output_dir outputs/unet_eval \
  --max_samples 32
```

---

## Train a Masked-Completion DDPM

The hard masked-completion configuration uses a frozen U-Net base. The DDPM is conditioned on the degraded image, base restoration, degradation mask, and reliability map. The final image is composed so that diffusion changes are applied only inside the allowed mask.

Train:

```bash
python -m src.training.train_ddpm_denoiser \
  --config configs/train_ddpm_masked_completion_hard.yaml
```

Evaluate:

```bash
python -m src.evaluation.eval_ddpm_denoiser \
  --ckpt outputs/ddpm_masked_completion_hard_10k/ckpt_best.pt \
  --base_ckpt outputs/unet_hard_50k_b6w8/ckpt_best.pt \
  --manifest data/manifests/patches_diverse.jsonl \
  --patch_dir data/patches_diverse \
  --output_dir outputs/ddpm_masked_completion_hard_10k_eval_ddim \
  --max_samples 512 \
  --sampling_mode ddim \
  --sampling_steps 50 \
  --num_samples_per_input 4
```

---

## What This Repository Is Good For

Sky_SR is useful for:

* testing satellite-style image restoration models before moving to specialized modalities,
* comparing deterministic and diffusion-based restoration under controlled corruption,
* measuring whether generative completion changes only the intended pixels,
* creating source-balanced RGB patch datasets from Earth imagery,
* producing visual diagnostics that expose failure modes quickly,
* and establishing a reproducible restoration baseline for later remote-sensing experiments.

---

## Current Limitations

This is an active research sandbox, not a final benchmark release.

Current limitations:

* The full local dataset is not included.
* Trained checkpoints are not included.
* Large generated outputs are not included.
* Metrics are from local smoke-scale and medium-scale experiments.
* The RGB dataset is a proxy benchmark, not a claim about all satellite modalities.
* Synthetic degradations are controlled approximations, not a complete physical sensor model.

These limitations are intentional at this stage. The goal is to create a reliable experimental harness before scaling to more specialized data, larger benchmarks, or sensor-specific restoration tasks.

---

## Public Use Notes

* Use imagery that you have permission to download, process, and redistribute.
* Keep large datasets, outputs, and checkpoints outside git.
* Treat the reported metrics as current experiment snapshots, not final benchmark claims.
* Use the visual grids alongside the JSON metrics; aggregate metrics alone are not enough to judge restoration quality.

---

## Bottom Line

Sky_SR shows that a strong residual U-Net is currently the best global restoration baseline for this paired RGB satellite-style benchmark. Conditional diffusion is not yet the strongest global restorer, but it becomes valuable when constrained to masked or gated completion, where the key requirement is not just image quality but controlled, spatially safe generation.
