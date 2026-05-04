# RGB Satellite Denoising Experiment Summary

The completed RGB satellite restoration experiments show that the supervised U-Net restoration path is the winning baseline for this paired benchmark. The Hard U-Net produced the strongest measured restoration improvement, while residual-DDPM variants were made safer through gating but did not improve the supervised proxy task. In the final gated multiband residual-DDPM evaluation, the best blend was 0.0, meaning the optimal result ignored the sampled DDPM residual; increasing DDPM blend degraded PSNR, MAE, SSIM, and detail metrics. Therefore, DDPM should not be treated as the primary restoration model for this RGB benchmark.

## Experiment Summary

| Experiment | Role | Main Finding | Selection Outcome |
| --- | --- | --- | --- |
| Balanced U-Net | Supervised deterministic baseline | Established the paired RGB restoration path. | Useful baseline, but not the final winner. |
| Hard U-Net | Stronger supervised deterministic baseline | Produced the strongest restoration improvement among tested supervised baselines. | Selected RGB supervised restoration baseline. |
| Residual-DDPM | Generative residual refinement | Did not establish improvement over deterministic restoration. | Not selected as primary model. |
| Gated residual-DDPM | Safer generative residual refinement | Gating constrained where DDPM residuals could affect output. | Safer, but not beneficial on the proxy benchmark. |
| Gated multiband residual-DDPM | Gated residual refinement with frequency-band structure | Best blend was 0.0; higher DDPM blend worsened reconstruction and detail metrics. | Safe but not useful here. |

## Key Quantitative Results

### Hard U-Net Evaluation

| Metric | Degraded Input | Restored Output | Change |
| --- | ---: | ---: | ---: |
| PSNR | 24.94 | 32.21 | +7.27 |
| MAE | 0.0487 | 0.0207 | -0.0280 |
| SSIM | 0.6898 | 0.8948 | +0.2050 |

### Gated Multiband Residual-DDPM Evaluation

| DDPM Blend | PSNR | MAE | SSIM |
| ---: | ---: | ---: | ---: |
| Base / 0.0 | 30.96 | 0.0223 | 0.9087 |
| 0.1 | 30.95 | 0.0224 | 0.9085 |
| 0.25 | 30.88 | 0.0226 | 0.9063 |
| 0.5 | 30.66 | 0.0231 | 0.8976 |

- Best blend by PSNR and MAE: `0.0`.
- `outside_gate_max_diff = 0.0`, confirming that gating prevented changes outside the allowed region.
- Highpass and lowpass detail metrics worsened as DDPM blend increased.

## Interpretation By Model Family

The U-Net family is the strongest fit for the completed RGB supervised restoration benchmark. The task is paired, spatially aligned, and directly learnable from degraded-clean image pairs. The Hard U-Net made large deterministic corrections and produced the clearest restoration gain.

The residual-DDPM family became safer after gating, especially with `outside_gate_max_diff = 0.0`, but safety did not translate into benchmark improvement. The DDPM residual path did not improve metrics over the base prediction, and increasing the residual blend consistently degraded global reconstruction metrics and detail metrics.

## Why U-Net Wins

This benchmark is a paired supervised restoration problem. The degradation target is directly learnable from input-output pairs, and the desired output is a clean RGB image aligned to the degraded input. A U-Net can make deterministic, spatially aligned corrections that are rewarded by PSNR, MAE, and SSIM.

The DDPM introduces generative residual variation. That variation is not rewarded when the correct target is a paired clean RGB image rather than one of many plausible outputs. In this setup, accurate conditional reconstruction is more useful than generative diversity.

## Why DDPM Did Not Help

The DDPM residual path did not improve the supervised proxy benchmark. The best blend was `0.0`, meaning the optimal result ignored the sampled DDPM residual. Higher DDPM blending worsened PSNR, MAE, SSIM, and detail metrics.

Gating made the DDPM path safe, because it prevented residual changes outside the allowed gate. However, the gated residual still did not provide useful restoration signal for this paired RGB task. The result is safe but not beneficial.

## Implications For InSAR

InSAR should not start with DDPM as the primary architecture. The RGB experiments indicate that the first step should be a strong deterministic baseline, especially when the restoration target is supervised or partially supervised.

InSAR should begin with a coherence-aware, mask-aware U-Net restoration/completion model. Inputs should explicitly include observability, masks, coherence or confidence, and degraded signal channels. DDPM should be deferred until after the deterministic baseline is strong.

DDPM may later be useful for gated low-confidence completion, uncertainty sampling, or ambiguous missing-region synthesis. It should not be the first-line restoration model.

## Recommended Next Architecture

Build a coherence/mask-aware U-Net baseline first. Train and evaluate it as the primary deterministic restoration/completion model. The model should condition directly on degraded signal channels, observability masks, and coherence/confidence channels.

Add DDPM only later as a gated low-confidence residual or completion module. Keep it only if it proves measurable improvement over the deterministic baseline.

## Final Conclusion

U-Net wins this RGB supervised restoration benchmark. The Hard U-Net produced the strongest restoration improvement among the tested supervised baselines and should be treated as the final RGB restoration baseline.

Gated DDPM is safe but not useful here. Gating successfully constrained residual changes, but the sampled DDPM residual did not improve the supervised proxy benchmark, and larger DDPM blends degraded reconstruction and detail metrics.
