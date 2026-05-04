# Next Task

## Task title
Audit Sky_SR baseline and split validity before new model work.

## Task objective
Create a read-only audit of the current strongest U-Net baseline and prior DDPM variants. Confirm exact configs, checkpoints, evaluation outputs, train/eval commands, metric summaries, and whether evaluation samples are source-disjoint or only patch-index deterministic.

## In scope
- Inspect configs:
  - `sat_denoise_data/configs/train_unet_denoiser_hard.yaml`
  - `sat_denoise_data/configs/train_ddpm_residual_denoiser_hard_gated_multiband.yaml`
  - `sat_denoise_data/configs/train_ddpm_masked_completion_hard.yaml`
- Inspect split logic in:
  - `sat_denoise_data/src/training/train_unet_denoiser.py`
  - `sat_denoise_data/src/training/train_ddpm_denoiser.py`
  - `sat_denoise_data/src/evaluation/eval_unet_denoiser.py`
  - `sat_denoise_data/src/evaluation/eval_ddpm_denoiser.py`
  - `sat_denoise_data/src/data/patch_degradation_dataset.py`
- Inspect docs:
  - `sat_denoise_data/docs/rgb_satellite_denoising_experiment_summary.md`
  - `sat_denoise_data/README.md`
- Inspect selected output summaries only:
  - `sat_denoise_data/outputs/unet_hard_50k_b6w8_eval/metrics.json` summary only
  - `sat_denoise_data/outputs/ddpm_residual_hard_gated_multiband_10k_eval_ddim_wideblend/metrics.json` summary only
  - `sat_denoise_data/outputs/ddpm_masked_completion_hard_10k_eval_ddim/metrics.json` summary only
  - `grouped_metrics_by_degradation_type.json` files if present
  - `train_log.jsonl` tails only

## Out of scope
- No implementation.
- No model architecture changes.
- No training.
- No data generation.
- No dependency installation.
- No full reads of huge metrics files.
- No edits outside `state/NEXT_TASK.md`.

## Likely files to inspect
- `sat_denoise_data/configs/train_unet_denoiser_hard.yaml`
- `sat_denoise_data/configs/train_ddpm_residual_denoiser_hard_gated_multiband.yaml`
- `sat_denoise_data/configs/train_ddpm_masked_completion_hard.yaml`
- `sat_denoise_data/src/training/train_unet_denoiser.py`
- `sat_denoise_data/src/training/train_ddpm_denoiser.py`
- `sat_denoise_data/src/evaluation/eval_unet_denoiser.py`
- `sat_denoise_data/src/evaluation/eval_ddpm_denoiser.py`
- `sat_denoise_data/src/data/patch_degradation_dataset.py`
- `sat_denoise_data/docs/rgb_satellite_denoising_experiment_summary.md`
- `sat_denoise_data/README.md`

## Likely files to modify
- `state/NEXT_TASK.md` only

## Required checks
- `python scripts/validate_state.py`
- `git diff -- state/NEXT_TASK.md`

## Exit criteria
- `NEXT_TASK.md` contains a bounded read-only audit task.
- State validation passes.
- Diff only touches `state/NEXT_TASK.md`.
