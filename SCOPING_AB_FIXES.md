# Scoping — Track A (hardening) + Track B (the fix)

**Date:** 2026-07-05 · **Branch:** `fix/camera-ready-corrections` · **Root:** `/Users/aayanalwani/virtual cells`
**Purpose:** cost, ground, and pre-register every documented A/B fix from `AUDIT_10of10_ROADMAP.md` §Bucket-2 and `HANDOFF_PerturbCausal_2026-07-03.md` §5 BEFORE running anything. Grounded against the actual files on disk (this session's reconnaissance), not against the roadmap prose alone.

---

## 0. What is actually on disk (verified 2026-07-05)

| Asset | Path | Fact |
|---|---|---|
| K562 N=200 slim panel | `data/k562_subset_n200_slim.h5ad` | 38,979 cells × **200 genes**; `obs['gene']` = knocked-out gene, **201 perturbations** incl. `non-targeting`. **The 200 perturbed genes ARE the 200 measured genes** → A1 anchored GT is directly constructable. |
| K562 full | `data/k562_real.h5ad` | 9.9 GB, 310k cells × 8,563 genes (used by J1). |
| Norman full | `data/norman2019.h5ad` | 111,445 cells × 33,694 genes. **131 unique doubles (41,759 cells)**, 105 singles, 11,855 ctrl (`nperts`∈{0,1,2}, labels like `SET_KLF1`). → A3 buildable. |
| Norman slim | `data/norman_subset_n200_slim.h5ad` | doubles already stripped (`nperts`∈{0,1}) — do **not** use for A3; use the full file. |
| Model predictions (K562 N=200) | `data/model_outputs_real_n200/{cpa,gears,geneformer}_predictions.h5ad`, `..._state/state_predictions_seed{42,123,456}.h5ad` | per-perturbation predicted means; drive the overlay→GIES pipeline. |
| External GRN on disk | `data/ground_truth/trrust_rawdata.human.tsv` | **only** TRRUST (already nulled by J1). No ENCODE-rE2G / Gasperini locally → A2 needs a fetch. |
| GIES-on-real baseline (the circular GT) | `results/eval_results_n200_vanilla.json` | F1: real 1.0, ElasticNet 0.425, overlay-real 0.422, CPA 0.389, GEARS 0.405, Geneformer 0.432, GRNBoost2 0.097. **These are the numbers A1 re-scores against a non-circular target.** |
| Eval pipeline | `scripts/run_eval_local.py` | `build_cb_input`, `run_gies(cb_input, seed, partition_size)`, `sample_perturbed_cells_overlay(ctrl_X, mean_pred, n_cells, seed)`, `compute_set_metrics(pred_edges, gt_edges, all_genes, seed)`. Reused verbatim. |
| No-GPU proxy pattern | `scripts/j2_lambda_sweep.py` | learned head `Linear200→ReLU→Linear200` trained on CPU as a CPA stand-in, driving overlay→GIES; pre-registered val/test discipline. **This is the template for Track B locally.** |
| Env | `.venv_gears/bin/python` (Py 3.9) | absolute path required. `gies`, `anndata`, `torch`, R+`inspre` installed. |

---

## 1. Feasibility — local vs GPU-gated

| Fix | Runs locally? | Compute (est.) | Odds of a clean positive | Notes |
|---|---|---|---|---|
| **A1** perturbation-anchored GT | ✅ fully local | GT build <2 min; re-score 6 conditions × 5 seeds ≈ 1–3 h (GIES) | n/a (it's a GT swap, not a bet) | Highest leverage. Breaks circularity. Correct target for B. |
| **A2** external K562 GRN | ⚠️ fetch-gated | fetch + score ≈ 1–2 h if network OK | Low–med (coverage-limited) | ENCODE-rE2G/Gasperini fetch may be blocked → fallback = coverage-optimized TRRUST re-subset. A low-overlap null is a valid, reportable outcome. |
| **A3** Norman doubles held-out | ✅ fully local | build + causal pipeline × seeds ≈ 2–4 h | Med (best place for a surprise VCM win) | Only ElasticNet + any model with predicted doubles; if no model predicts doubles on disk, restrict to ElasticNet-vs-real + document the VCM gap as GPU-gated. |
| **B** precision/OT objective | ⚠️ **proxy only** | loss+tests <1 h; proxy sweep 8λ×5 seeds ≈ 3–6 h | **~25–40%** (honest) | **Definitive CPA retrain is GPU-gated — CANNOT run on this Mac.** Local deliverable = the loss module + the pre-registered no-GPU proxy sweep, scored against the A1 hardened GT. |

**Hard blocker (explicit):** Track B's *definitive* test — retrain CPA (~45 min GPU, scvi-tools) end-to-end with the new objective — requires a GPU (Modal/HPC). It is **out of scope for local execution**. What ships locally: (1) the differentiable objective + unit tests, (2) the pre-registered no-GPU proxy sweep that stands in for the retrain exactly as J2 did. The proxy answers "does this loss term move the causal metric at all" — a green proxy justifies the GPU spend; a null proxy is itself informative (as J2's was).

---

## 2. Per-fix method (grounded)

### A1 — Perturbation-anchored ground truth  *(Step 2–3)*
Truth from interventions, not from an algorithm. For each perturbed gene A (200 of them, all measured): compute B's response = mean(perturbed_A) − mean(control), per measured gene B≠A, on the real K562 N=200 panel. Declare a **directed edge A→B** when B's response passes an effect-size + significance threshold (Welch t on log1p counts, BH-FDR; |log-FC| floor). Sweep the threshold to trace edges-vs-stringency. This edge set is causal-by-construction (A was actually knocked out). Then re-run overlay→GIES for every condition and score recovered edges against THIS set (absolute P/R/F1), 5 seeds. Compare to the circular-GT numbers in §0.
**Why non-circular:** GIES never touches the truth; truth = observed interventional response.
**Caveat to report:** anchored edges are *direct+indirect* interventional effects (A→B may be mediated); this is a known property of perturbation-response "networks" and will be stated, with the |log-FC|/FDR stringency as the direct-effect proxy knob.

### A2 — External K562 GRN  *(Step 4)*
Try to fetch a curated external K562 network (ENCODE-rE2G benchmark enhancer–gene, or Gasperini 2019 K562 CRISPRi). If network-blocked → `request_network_access` once; if denied → fallback to TRRUST but **re-pick the gene subset to maximize curated-edge overlap** (J1's null came from ~0 overlap on the essential-gene subset — a coverage problem, not necessarily a validity one). Score GIES-on-real and each model vs the external edges: absolute P/R + TF→target recall. Report coverage (n edges in subset) up front; a low-coverage null is honest and expected.

### A3 — Combination perturbations  *(Step 5)*
From `norman2019.h5ad`: 131 doubles held out. Train predictors on singles, form/predict the 131 unseen pairs, run the causal pipeline, compare recovery on the held-out doubles: ElasticNet (additive-of-singles baseline — provably cannot capture epistasis) vs any VCM prediction available. Combinations are the marquee "when do VCMs help?" case. Multi-seed. If no VCM double-predictions are on disk, the local result is ElasticNet-vs-real on doubles + an explicit note that the VCM double-prediction arm is GPU-gated.

### B — Precision-aware / OT objective  *(Step 6–7)*
The J2 covariance-Frobenius penalty is already **ruled out** (null). The untried, better-motivated objectives, added to `causalcellbench/training/`:
- **Precision-support penalty** — align the *sparse inverse-covariance (precision) support* of predicted vs real perturbation-response, graphical-lasso style. Rationale: interventional causal discovery (GIES/inspre) exploits conditional-independence = precision-matrix *zeros*, not covariance magnitude — so penalize support mismatch, not Frobenius distance. Ties directly to the J7 identifiability proposition.
- **Distribution-matching term** — MMD (RBF kernel) and/or Sinkhorn-OT between predicted and real joint perturbation-response, matching the whole joint, not moments.
Unit-tested (gradient + shape), paralleling `covariance_regularizer.py`. Then the **pre-registered no-GPU proxy sweep** (learned head, λ-grid, val-select, locked test ×1, 5 seeds) scored against the **A1 hardened GT**. Keep/discard gate applied honestly.

---

## 3. Pre-registration index
- `prereg_A1_anchored_gt.md` — A1 GT construction + re-score (locked comparison, 5 seeds).
- `prereg_A3_combinations.md` — A3 held-out doubles protocol (5 seeds, keep/discard).
- `prereg_B_precision_objective.md` — B objective proxy sweep (λ-grid, val-select, locked test, keep/discard) — mirrors `preregistration_j2_lambda.md`.
A2 is validation-only (no keep/discard bet), so its protocol lives in §2 above rather than a separate prereg.

## 4. Honesty commitments (carried from the project's discipline)
1. Every reported number traces to a `results/*.json`.
2. Multi-seed with per-seed values + CI; single-seed = "suggestive," never "significant."
3. Nulls reported straight — no null reframed as a win (the J1/J2 precedent).
4. Locked test splits touched exactly once; λ/threshold selection on validation only.
5. GPU-gated items labeled as such; the local proxy is never presented as the definitive test.
