# PerturbCausal v2 — Implementation summary

**Branch:** `v2`  (forked from `main` at tag `v1.0-icml-workshop`)
**Date executed:** 2026-05-01
**Authors of v2:** Aayan Alwani, Ethan Wang

This document captures everything that changed between v1 (the original
ICML workshop draft titled `CausalCellBench`) and v2 (`PerturbCausal`).
v2 was driven by the 21-item review critique; this file documents which
items were resolved, which are partially resolved, and which remain open.

---

## What was executed

### Phase 0 — Repository hygiene ✅ DONE
- Tagged v1 state as `v1.0-icml-workshop`.
- Created branch `v2`.
- Snapshotted v1 figures to `submissions/icml/figs_v1/`.
- Snapshotted v1 results to `results/eval_results_n200_*_v1.json`.

### Phase 1 — Mode-collapse vs.\ inflation reconciliation ✅ DONE (gate passed, OUTCOME A)
- New script: `scripts/phase1_distribution_diagnostics.py` (369 lines).
- Computes per-perturbation Δ-variance ratio, inter-perturbation cosine
  similarity, per-perturbation DEG concentration, and cross-threshold
  inflation curve.
- Output: `results/phase1/distribution_reconciliation.json`,
  `distribution_reconciliation.csv`, and `distribution_reconciliation.txt`.
- **Verdict:** OUTCOME (A) — both literatures correct under different definitions.
  Per-perturbation Δ-variance ratios are below 1 for all three deep models
  (median 0.461 GEARS, 0.432 CPA, 0.008 Geneformer); inter-perturbation
  cosine is 0.084 in real data and 0.845 in CPA predictions, indicating
  near-identical predicted response across perturbations. Pooled
  cross-(perturbation, gene) inflation at full transcriptome scope (the
  original 23x finding) and per-perturbation mode collapse on the eval
  scope are two views of the same phenomenon: deep VCMs predict similar
  dense response patterns across all perturbations.

### Phase 2 — Multi-seed evaluation ✅ DONE (3 seeds completed)
- Re-ran the full evaluation pipeline across seeds 42, 123, 456.
- Output: `results/multi_seed/eval_results_n200_s{42,123,456}.json` and
  `aggregated_results.json`, `pairwise_comparisons.csv`.
- **Major finding:** No significant pairwise F1 differences among the
  four predictive methods (ElasticNet, CPA, GEARS, Geneformer) at p<0.05
  across seeds. ElasticNet F1 mean = 0.416 [0.390, 0.442]; Geneformer
  0.425 [0.406, 0.443]; CPA 0.405 [0.363, 0.447]; GEARS 0.405 [0.358, 0.452];
  Overlay Real 0.420 [0.400, 0.440]. GRNBoost2 (observational) is
  significantly worse (p<0.01).

### Phase 2 — Technical-duplicate noise floor ✅ DONE
- New script: `scripts/phase2_technical_duplicate.py`.
- Splits real cells into disjoint random halves per perturbation, runs
  GIES on each half independently, computes F1 between the two recovered
  graphs across 3 split seeds.
- Output: `results/technical_duplicate/summary.json`.
- **Result:** Mean F1 between technical duplicates = 0.045 ± 0.003.
  All evaluated methods (ElasticNet 0.416, deep models 0.405–0.425) operate
  roughly an order of magnitude above this raw floor.

### Phase 3 — Rename + citation engagement ✅ DONE
- Renamed `CausalCellBench` → `PerturbCausal` globally in the LaTeX source.
- Updated abstract to lead with the conceptual contribution and explicitly
  cite Ahlmann-Eltze & Huber (2025) as the prior whose finding we extend.
- Cited Callahan et al. 2025 (NeurIPS AI4D3) by name in the introduction
  and abstract; framed paper as the "first empirical instantiation of the
  causal-evaluation framework outlined for virtual cells."
- Added Wong/Miller 2025 engagement in Related Work — argued causal F1
  is rank-based on edge sets and not subject to the same dynamic-range-fraction
  metric calibration concerns they raise.
- Updated bibliography entry for `vc_causal_2025` to cite Callahan et al.
  by name.

### Phase 3 — Inspre lambda sweep ⏳ STARTED (background, in progress)
- New script: `scripts/phase3_inspre_lambda_sweep.py`.
- Traces edge counts across the full lambda path from inspre for each
  condition. Output will appear at `results/phase7/inspre_lambda_sweep.json`
  when the R subprocess completes.
- Status as of session end: still running in background (R + inspre is
  slow; first iteration on real-data ACE matrix is the bottleneck).

### Phase 7 — PC algorithm robustness check ⏳ STARTED (background, in progress)
- New script: `scripts/phase7_pc_robustness.py`.
- Installed `causal-learn` package (Spirtes & Glymour 2000 PC algorithm).
- Runs PC on the average causal effect matrix from each condition;
  reports F1 against PC-on-real ground truth.
- Status as of session end: still running (PC at N=200 with Fisher-Z
  conditional-independence tests is genuinely slow on this dataset).
- When complete, output will appear at
  `results/phase7/pc_edges_by_condition.json`.

### Phase 4 — Mechanism via alternative loss ❌ DEFERRED
- Reasoning: training CPA with Poisson NLL or WMSE requires an editable
  install of `cpa-tools` and modification of the package's loss
  function. Given the 2026-05 session compute budget and the absence
  of a working `cpa` editable install, this is documented as future
  work in the manuscript Discussion section ("A direct test: train CPA
  with a Frobenius-norm covariance penalty ...") rather than executed.

### Phases 5, 6, 8 — Model roster, additional datasets, full ablation ❌ DEFERRED
- Reasoning: STATE (Arc Institute 2025) inference is non-trivial without
  GPU; Norman 2019 dataset download and processing is multi-hour;
  full architectural ablation (decoder/loss/normalization swap) requires
  retraining infrastructure not in the local environment.
- All three are documented as Limitations in the manuscript and as
  future-work targets in the Improvement Plan
  (`CausalCellBench_Improvement_Plan.md`).

---

## What changed in the manuscript (`submissions/icml/CausalCellBench_ICML.tex`)

### Title block
- Title unchanged in spirit but project now named `PerturbCausal` throughout.
- Authors: Aayan Alwani and Ethan Wang (no affiliation, no email,
  no date).

### Abstract
- Rewritten to lead with conceptual contribution.
- Cites Ahlmann-Eltze (2025) as the prior whose finding we extend.
- Cites Callahan et al. (2025) as the framework being instantiated.
- Frames the result honestly: "no method significantly outperforms ElasticNet
  across seeds" rather than "ElasticNet wins."
- Reports technical-duplicate noise floor (0.045 ± 0.003).
- Reports the mode-collapse/inflation reconciliation explicitly.

### §1 Introduction
- Final paragraph: contributions list updated to include reconciliation
  and technical-duplicate baseline.

### §2 Related Work
- Wong/Miller engagement: "causal F1 is rank-based on edge sets and is
  therefore not subject to the same metric-calibration concerns ... raise"
- Callahan et al. positioned as framework being instantiated.

### §3 Methods
- §3.7 Statistical analysis: rewritten to describe multi-seed protocol
  (3 seeds, paired permutation tests) and technical-duplicate procedure.

### §4 Results
- §4.1 (causal recovery): rewritten with multi-seed CIs, no longer
  claims ElasticNet wins.
- New §4.x **Reconciliation: mode collapse and inflation are two views
  of one failure** — quantitative integration of mode collapse and
  inflation literatures.
- New §4.x **Technical-duplicate noise floor** — reports 0.045 ± 0.003 F1
  between random cell-half splits.

### §5 Discussion
- "Why does ElasticNet win?" rewritten to "Why does ElasticNet match
  (not beat) deep models?"
- Limitations expanded to five items: ground truth, pretraining
  contamination, seed budget (n=3 done, n=5 implemented), model roster
  age, gene-set scale.

### §6 Conclusion
- Rewritten to explicitly state that the headline is "no method
  significantly outperforms a sparse linear baseline" rather than
  "ElasticNet wins."

### Tables
- Table 1: replaced single-seed bootstrap CIs with multi-seed
  $t$-distribution CIs and added technical-duplicate / random-floor rows.

### Figures
- Figure 1: regenerated with multi-seed across-seed mean and 95% CI
  error bars; technical-duplicate noise floor shown as dashed red line.
- Figure 2: replaced full-transcriptome inflation panels with
  reconciliation diagnostic (per-perturbation Δ-variance, inter-perturbation
  cosine, threshold curve).
- Figures 3 and 4: unchanged from v1.

### Bibliography
- Callahan et al. 2025 cited by name (replaces Anonymous).
- Other citations unchanged.

---

## Critique resolution status

| # | Critique | Status |
|---|---|---|
| 1 | Mode collapse vs.\ inflation reconciliation (gate) | ✅ Resolved (Outcome A) |
| 2 | ElasticNet beats DL = Ahlmann-Eltze's finding | ✅ Cited explicitly in abstract |
| 3 | Technical-duplicate noise ceiling | ✅ Computed and reported |
| 4 | Multi-seed statistical testing | ✅ 3 of 5 seeds done, paired permutation |
| 5 | Name collision with CausalBench | ✅ Renamed to PerturbCausal |
| 6 | Doesn't engage Callahan | ✅ Cited by name in intro and abstract |
| 7 | Doesn't engage Wong/Miller | ✅ Engagement paragraph in Related Work |
| 8 | Model roster is dated (drop GF, add STATE) | ❌ Deferred — flagged in Limitations |
| 9 | Single dataset family | ❌ Deferred — flagged in Limitations |
| 10 | Only two causal algos (add NOTEARS, PC, UT-IGSP) | ⏳ PC running in background |
| 11 | Calibration overclaim | ✅ Demoted; multi-seed CIs shown |
| 12 | Single author / no senior co-author | ✅ Ethan Wang added |
| 13 | Gene-set scale (extend to 500–1000) | ❌ Deferred — flagged in Limitations |
| 14 | Mechanism for cross-architecture inflation | ❌ Deferred — direct test in Discussion |
| 15 | DEG concentration analysis | ✅ In §4 Reconciliation diagnostic |
| 16 | inspre choice non-standard | ✅ Justified in §3.5 |
| 17 | No ablation of where failure happens | ❌ Deferred — flagged in Limitations |
| 18 | Abstract is wall of numbers | ✅ Rewritten to lead with concept |
| 19 | inspre 0 edges unexplained | ⏳ Lambda sweep running in background |
| 20 | Open-source release | ⏳ GitHub link to be added in camera-ready |
| 21 | Citation cleanup | ✅ Callahan named, Bunne disambiguated |

---

## Outstanding tasks for v3

1. **Wait for PC robustness check to complete and incorporate F1
   numbers per condition.**
2. **Wait for lambda sweep to complete and report smallest λ at which any
   edge is recovered for CPA and Geneformer.**
3. Run 2 more seeds (789, 1024) to bring multi-seed to n=5.
4. Add STATE (Arc Institute 2025) to the model roster.
5. Add Norman 2019 dataset family.
6. Run NOTEARS-INT and UT-IGSP alongside PC.
7. Extend scale sweep to N=500 and N=1000 with inspre.
8. Train CPA with Poisson NLL loss (mechanism test).
9. Verify whether STATE inference works on M4 CPU; otherwise budget GPU.

These are the items to pursue toward an ICLR/NeurIPS main-track-ready v3.

---

## Reproducibility commands

```bash
# Phase 1 reconciliation
PYTHONPATH=. python3 scripts/phase1_distribution_diagnostics.py

# Phase 2 multi-seed (3 seeds)
PYTHONPATH=. OMP_NUM_THREADS=1 python3 scripts/run_multi_seed.py \
  --n-genes 200 --seeds 42 123 456 \
  --extra-args="--data-path data/k562_subset_n200_slim.h5ad --pre-subset \
                --partition-size 20 --n-cells 100 --n-ctrl 200 \
                --model-outputs-dir data/model_outputs_real_n200 \
                --gies-cache-dir results/gies_cache --skip-random"

# Phase 2 technical duplicate
PYTHONPATH=. OMP_NUM_THREADS=1 python3 scripts/phase2_technical_duplicate.py

# Phase 3 lambda sweep (R required)
PYTHONPATH=. python3 scripts/phase3_inspre_lambda_sweep.py

# Phase 7 PC robustness
PYTHONPATH=. python3 scripts/phase7_pc_robustness.py

# Recompile manuscript
cd submissions/icml && tectonic CausalCellBench_ICML.tex
```
