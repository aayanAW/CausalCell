# Calibration Comparison: Vanilla vs. Quantile-Calibrated Predictions

**Run date:** 2026-04-30
**Setup:** N=200 K562, partition size 20, seed 42
**Calibration method:** Quantile matching to empirical real K562 |log2FC| distribution at the same 200-gene scope (pseudocount 0.01, 1000 quantile knots)
**Reference:** Bolstad et al. 2003 (microarray normalization); Reinhard et al. 2001 (color transfer); same algorithm.

## Headline numbers

| Condition | Vanilla F1 | Calibrated F1 | Δ F1 | Vanilla TP | Calibrated TP | Δ TP |
|---|---:|---:|---:|---:|---:|---:|
| Real Data (ceiling) | 1.000 | 1.000 | — | 1,220 | 1,220 | — |
| ElasticNet (no calibration applied) | 0.4251 | 0.4251 | — | 519 | 519 | — |
| Overlay Real (UB, no calibration) | 0.4222 | 0.4222 | — | 514 | 514 | — |
| **CPA** | 0.3887 | **0.4051** | **+0.0164** | 476 | 497 | +21 |
| **GEARS** | 0.4055 | **0.4133** | **+0.0078** | 489 | 510 | +21 |
| **Geneformer** | 0.4324 | 0.4232 | **−0.0092** | 528 | 515 | −13 |

(ElasticNet and Overlay Real are not deep models and are not affected by calibration; they appear unchanged.)

## Effect-size distribution after calibration

| Source | Frac \|log2FC\| > 0.5 | Inflation ratio vs. real (0.114) |
|---|---:|---:|
| Real K562 (200-gene reference) | 11.4% | 1.0× |
| GEARS vanilla | 96.6% | 8.5× |
| GEARS calibrated | 11.2% | 1.0× |
| CPA vanilla | 97.7% | 8.6× |
| CPA calibrated | 11.4% | 1.0× |
| Geneformer vanilla | 97.7% | 8.6× |
| Geneformer calibrated | 11.4% | 1.0× |

(The 23× number from Phase 1 was computed against the full 8.5K-gene scope, where the real fraction is 2.7%. At the 200-gene perturbation-eligible scope used here, the real fraction is 11.4% — perturbation-eligible genes are inherently more responsive than the genome-wide average. The relative inflation is 8–9× at this scope; the calibration brings it to 1.0× exactly, as expected for quantile matching.)

## Reconstruction MSE (sanity check)

| Model | Vanilla MSE | Calibrated MSE | Change |
|---|---:|---:|---:|
| CPA | 3.96 | 4.21 | +6% |
| GEARS | 3.14 | 4.19 | +33% |
| Geneformer | 3.96 | 4.26 | +8% |

Reconstruction degrades modestly for CPA and Geneformer; more substantially for GEARS. This is the expected tradeoff: matching the real magnitude distribution slightly increases per-gene squared error because the model's predicted FCs were well-correlated with reality but at the wrong scale.

## Honest interpretation

**Calibration helps two of three models on causal recovery, hurts the third.**

- CPA: +1.6 F1 points. Closes 39% of the gap from vanilla CPA (0.389) to ElasticNet (0.425).
- GEARS: +0.8 F1 points. Closes 41% of the gap from vanilla GEARS (0.406) to ElasticNet (0.425).
- Geneformer: −0.9 F1 points. Calibration moves it *away* from ElasticNet.

**No calibrated deep model surpasses ElasticNet (0.425).** The gap-closing happens but stops at roughly the ElasticNet level for the two models calibration helps; the third model regresses. The cleanest reading: effect-size inflation is *one* failure mode that calibration corrects, but not the only one. CPA and GEARS still fall short of ElasticNet after correction, indicating residual failure in the joint distribution (covariance structure, sparsity pattern, direction of edges) that magnitude-only calibration cannot address.

**Geneformer's regression is informative.** Vanilla Geneformer has the best deep-model F1 (0.432). After calibration its F1 drops to 0.423. One plausible explanation: Geneformer's foundation-model pretraining captures something about the *distribution* of effects that flat quantile matching erases. Magnitude inflation is not the dominant failure mode for Geneformer; whatever signal it has lives in the correct relative ranking, and quantile matching is a no-op on rank-correct signal but a small loss when there is rank-correct calibration already implicit in the model.

## Implications for the proposal

- **Gate 2 partially passed.** Calibration moves causal F1 in the predicted direction for two of three models. The hypothesis that calibration alone closes the gap to ElasticNet is **not** supported.
- **The single-mode quantile matching is insufficient.** The covariance preservation and sparsity injection modes (Section 4.5 of v5 proposal) remain to be implemented and tested. Both are needed.
- **The mechanism story holds.** The ~9× inflation drops to 1× exactly — calibration does what it was designed to do at the distributional level. The fact that this only translates to modest causal F1 gains tells us causal recovery depends on more than the marginal distribution of effects. This is itself a publishable finding: it sharpens the claim from "models inflate effects" to "models inflate effects *and* misorient the joint structure, and fixing the marginals only fixes the marginal piece."

## Reproducibility

```bash
# Calibrate predictions
PYTHONPATH=. python3 scripts/calibrate_predictions.py \
  --diagnostics-out results/calibration_diagnostics.json

# Run GIES on calibrated predictions (writes eval_results_n200.json)
PYTHONPATH=. OMP_NUM_THREADS=1 python3 scripts/run_eval_local.py \
  --data-path data/k562_subset_n200_slim.h5ad \
  --pre-subset --n-genes 200 --partition-size 20 \
  --n-cells 100 --n-ctrl 200 --seed 42 \
  --model-outputs-dir data/model_outputs_real_n200_calibrated \
  --gies-cache-dir results/gies_cache_calibrated --skip-random
```
