# Pre-registration — C: cell-level vs mean+overlay causal evaluation

**Date:** 2026-07-05 · **Base:** branch `fix/camera-ready-corrections` ·
**Motivating finding:** the overlay sampler (`sample_perturbed_cells_overlay`,
`run_eval_local.py:615-651`) applies a prediction ONLY as a per-gene diagonal rescale of
control cells; GIES reads conditional-independence (precision support), which is invariant to
diagonal rescaling. Verified: two different mean predictions give correlation matrices
identical to max |Δ| = 1e-15. Consequence: the causal metric is provably blind to everything
in a prediction except the per-gene mean, and the overlay discards even the joint part of that.
This experiment tests whether feeding a model's **predicted cells** (its own joint samples)
directly to GIES — bypassing the overlay — lets the instrument register joint structure.

## Question

Does cell-level causal evaluation (predicted cells → GIES) separate a model's output from the
same output collapsed to mean+overlay? And does real-cell input (real joint structure) separate
from mean(real)+overlay? If yes, the overlay was the bottleneck and cell-level evaluation is the
fix. If no (both saturate), the overlay-invariance result stands as the finding and GIES at
N≈200 is genuinely resolution-limited.

## Design — 4 conditions, identical everything-else

On the RPE1 CPA predicted-cells file (`data/model_outputs_real_rpe1/cpa_predictions.h5ad`,
20,000 cells × 8,749 genes, output_type=cells, 200 perturbations, all 196 panel genes present)
and the real RPE1 data (`data/rpe1_subset_n200_slim.h5ad`, 29,695 cells; pert col `gene`).
Panel = the 196-gene `gene_subset` from `results/phase2_eval_rpe1.json`; partition_size=20,
n_cells_per_pert=100 (frozen to the existing RPE1 eval).

| id | condition | input to GIES | tests |
|----|-----------|---------------|-------|
| A | real cells | real perturbed cells (real joint) | the true achievable ceiling |
| B | overlay(real means) | mean(real cells) → overlay → cells | what the paper currently reports (~0.42 at K562) |
| C | predicted cells | CPA predicted cells (model joint) | the fixed instrument on a model |
| D | overlay(pred means) | mean(CPA cells) → overlay → cells | the current instrument on the same model |

All four share the same panel, partitions (per seed), edge budget, and ground truth, so F1 is
directly comparable. **B and D use the exact overlay path; A and C use the output_type=cells
path (`run_eval_local.py:718`) that routes cells straight to GIES.**

## Ground truth (two, both reported)

1. **GIES-on-real-cells** (condition A's own GIES edges) — the paper's existing circular GT;
   A scores ~1.0 against it by construction, so this GT is used to score B/C/D only, and to
   define the recoverable structure.
2. **Perturbation-anchored RPE1 GT** — Welch t-test per perturbed gene vs non-targeting
   control on log1p counts, BH-FDR, |log2FC| floor (same method as A1), primary operating
   point q=0.05, τ_fc=0.5. Non-circular. Reported for all four conditions.

## Primary tests (declared before running)

- **T1 (instrument sensitivity to model joint):** C > D, paired per seed. KEEP iff C beats D
  on ≥4/5 seeds AND mean(C−D) > run-to-run SD. This is the decisive test: if the overlay were
  not discarding structure, C and D would be equal.
- **T2 (input joint carries recoverable structure):** A > B, paired per seed, same gate. If
  real cells beat mean(real)+overlay, the overlay demonstrably discards recoverable joint
  structure regardless of any model.

## Outcomes — both pre-registered as publishable

- **Separation (T1 and/or T2 pass):** the mean+overlay instrument is blind to joint structure;
  cell-level evaluation is the fix. → reframed contribution: "mean-overlay causal benchmarks
  are structurally blind to the joint distribution; here is the cell-level evaluation that
  fixes it."
- **No separation (both fail):** the overlay-invariance result stands as a standalone finding,
  AND GIES at this N is resolution-limited even given real joint structure — a second,
  independent limitation of the current benchmark design. Either way the invariance proof
  (max |Δ|=1e-15) is the floor and does not depend on this outcome.

## Seeds / stopping

Seeds 42/123/456/789/1024. Fixed; no adaptive expansion. Test GT touched once per condition
per seed. Report all four conditions × both GTs × 5 seeds, gates T1/T2, and recovered-edge
Jaccard(C,D) and Jaccard(A,B) to quantify how much structure the overlay discards.

## Replication

Repeat the entire design on the Norman CPA cells file
(`data/model_outputs_real_norman/cpa_predictions.h5ad`, 10,200 × 8,028) with its own panel and
anchored GT, to test dataset-generality. Agreement → robust; divergence → reported honestly.

## Contingent GPU extensions (declared now, gated on the local result)

- **K562 CVAE cells (step 5):** ONLY if T1 passes on RPE1. Emit sampled cells from the trained
  CVAE decoder (recon + OT), score cell-level vs A1 anchored GT, compare to the mean+overlay
  numbers already on record (`b_gpu_definitive.json`).
- **OT objective re-test on the fixed instrument (step 6):** ONLY if the K562 cell-level
  instrument is sensitive. Re-run Sinkhorn-OT vs recon scored cell-level (not overlay), ≥5
  seeds, same locked gate as B2. This is the honest re-test of Track B on a metric that can
  register the covariance the objective shapes.