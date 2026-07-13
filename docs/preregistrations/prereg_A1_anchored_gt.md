# Pre-registration — A1 perturbation-anchored ground truth

**Date:** 2026-07-05 · **Base commit:** `fix/camera-ready-corrections` HEAD.
**FIT-GATE:** GREEN (large frozen Replogle K562 N=200 atlas where perturbed genes = measured genes; cheap automatable edge construction; deterministic truth).

## Question
Does the substitutability ranking (which data source best recovers the causal network) change when the ground truth is **perturbation-anchored** (built from real interventional responses) instead of **GIES-on-real** (the circular truth)? This is the #1 reviewer objection (S2): a circular GT means "causal recovery" = recovery of an algorithm's output. A1 replaces the truth with observed biology.

## Ground-truth construction (frozen protocol)
Data: `data/k562_subset_n200_slim.h5ad` (200 perturbed = 200 measured genes; `obs['gene']`, control = `non-targeting`).
For each perturbed gene A and each measured gene B≠A:
- responder test: **Welch's t-test** on `log1p(count)` between A-perturbed cells and control cells for gene B;
- multiple testing: **Benjamini–Hochberg FDR** across all B, per A;
- effect floor: **|log2 fold-change| ≥ τ_fc**.
Edge A→B iff `FDR ≤ q` AND `|log2FC| ≥ τ_fc`.
**Pre-registered grid (report all; pick primary before re-scoring):** `q ∈ {0.01, 0.05}`, `τ_fc ∈ {0.25, 0.5, 1.0}`. **Primary operating point (declared now, before seeing re-score F1): `q=0.05, τ_fc=0.5`** — a standard DE threshold. Self-edges A→A excluded.
Diagnostic (no bearing on the bet): edge count vs threshold; in/out-degree; Jaccard overlap with the GIES-on-real edge set (low overlap is expected and *is the point*).

## Re-score protocol
Conditions (all that have predicted means on disk): Real, ElasticNet, CPA, GEARS, Geneformer, STATE. For each, run the existing `sample_perturbed_cells_overlay → run_gies` pipeline (partition 20, N_cells 100, N_ctrl 200) and score recovered edges vs the anchored GT with `compute_set_metrics` → absolute **precision / recall / F1**.
**Seeds:** 42, 123, 456, 789, 1024 (5). Report per-seed + mean ± SD.

## Primary metric
Absolute F1 of each condition's recovered edges vs the perturbation-anchored GT (higher better). Secondary: precision, recall, AUPRC.

## What counts as an outcome (no keep/discard — this is a GT swap, not a bet)
Report straight, whatever it shows:
- If real-data F1 rises well above ElasticNet/VCM under the anchored GT while it was tied under GIES-on-real → the circular GT was masking a real gap (strong result, de-saturates S1+S2 at once).
- If the ranking is unchanged → the original finding is robust to a non-circular GT (also a strong, defensible result).
- If everything collapses to near-zero → the anchored GT is too sparse/indirect at N=200; report the coverage and treat as a scale limitation.
No result is reframed. The primary operating point is locked above; the other grid points are reported as sensitivity, not cherry-picked.

## Reporting
`results/hardening/a1_anchored_rescore.json` (per-seed, all conditions), `data/ground_truth/perturbation_anchored_k562_n200.json` (edges + build metadata), figures `a1_anchored_gt_diagnostics.png`, `a1_anchored_f1_by_condition.png`. Compare against the circular-GT numbers in `results/eval_results_n200_vanilla.json`.
