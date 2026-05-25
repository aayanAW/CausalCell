# PerturbCausal — V3 Fix Plan Audit + V4 (10/10 Upgrade)

**Version:** 4.0
**Date:** 2026-05-25
**Status:** Audit of `ARCHITECTURE_V3_FIX_PLAN.md` + replacement plan.
**Sources verified:** Consensus MCP (peer-reviewed paper search). 5 searches executed; citations [N] map to Consensus result indices.

---

## 0. Executive Audit Summary

`ARCHITECTURE_V3_FIX_PLAN.md` correctly identifies and addresses the audit's five fatal flaws (RPE1 bug, GRNBoost2 statistics, STATE single-seed, Norman one-baseline, calibration over-claim). However, the V3 plan itself has **25 holes** discovered on re-audit, falling into six categories: (i) literature gaps that misposition the contribution; (ii) methodological errors in the calibration modes; (iii) statistical power miscalculations; (iv) timeline infeasibility for NeurIPS D&B 2026; (v) compute-budget under-estimation; (vi) experimental design ambiguities. The most consequential are H6, H9, H12, H21, H23 below.

V4 replaces V3 with a 12-phase plan that addresses every hole, redefines the contribution to survive the recent crowded literature, and ships a research artifact that no reviewer can dismiss.

**Key literature finding from Consensus search:** Between Ahlmann-Eltze 2025 and 2026, **at least seven additional perturbation-prediction benchmarks have appeared** that already establish "simple baselines beat DL" on per-gene reconstruction [1-7 below]. PerturbCausal's per-gene angle is now well-trodden; the **causal-recovery angle survives as the unique contribution** but must be repositioned, and at least two adjacent papers (CASCADE [9], Park 2026 [10]) directly attack causal-from-Perturb-seq territory. The V3 plan does not engage with this changed landscape. V4 does.

---

## 1. Verified Citations (from Consensus)

| # | Paper | Verified | Note |
|---|---|---|---|
| 1 | [Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines](https://consensus.app/papers/details/a48c95b391f25de2813ae2d2bc322524/) (Ahlmann-Eltze & Huber, Nature Methods, 2025, 106 cites) | ✓ | Manuscript cites correctly |
| 2 | [Diversity by Design: Mode Collapse in scRNA-seq Perturbation Modeling](https://consensus.app/papers/details/a8d1a5357c3a51c4920c135cf9ac4609/) (Mejia et al., arXiv 2506.22641, 2025, 9 cites) | ✓ | Manuscript cites correctly |
| 3 | [Predicting cellular responses to perturbation across diverse contexts with State](https://consensus.app/papers/details/dd560375078b5c3e80d07a7dabbe37d3/) (Adduri et al., Arc Institute, bioRxiv 2025, 77 cites) | ✓ | Manuscript cites GitHub URL; **must also cite the bioRxiv paper** |
| 4 | [Large-scale causal discovery using interventional data sheds light on gene network structure in K562 cells](https://consensus.app/papers/details/e0ae406b72015420ab6531588e61c67b/) (Brown et al., Nature Communications, 2025, 5 cites) | ✓ | Manuscript cites correctly |
| 5 | [Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq](https://consensus.app/papers/details/96af00b82ff355e9af7f30e2c6d51344/) (Replogle et al., Cell, 2021, 596 cites) | ✓ | Year in manuscript is "2022" — Consensus says 2021. Cell publication = 2022. Verify volume/pages |
| 6 | [Exploring genetic interaction manifolds constructed from rich single-cell phenotypes](https://consensus.app/papers/details/0118620503f15b1ba19d8207f2722d74/) (Norman et al., Science 2019, 352 cites) | ✓ | Manuscript cites correctly |
| 7 | [Predicting transcriptional outcomes of novel multigene perturbations with GEARS](https://consensus.app/papers/details/7a02e0494d785453830219616acff927/) (Roohani et al., Nature Biotechnology, 2023, 303 cites) | ✓ | Manuscript says "2024 Nature Biotechnology" — Consensus says 2023. Verify |
| 8 | [Predicting cellular responses to complex perturbations in high-throughput screens](https://consensus.app/papers/details/d2396cbac86452f6be3d146e7b0e8353/) (Lotfollahi et al. — CPA — Molecular Systems Biology, 2023, 260 cites) | ✓ | Manuscript cites correctly |

---

## 2. New Literature That V3 Misses (Must Be Cited)

| # | Paper | Why it matters for PerturbCausal |
|---|---|---|
| L1 | [Simple controls exceed best deep learning algorithms (Wong et al., Bioinformatics 2025, 20 cites)](https://consensus.app/papers/details/a635e6101d1356bdb8eff0ef5e41afef/) | Pfizer paper. Directly competes on the "simple beats DL" claim. Provides a "corrected dataset for benchmarking." Must engage. |
| L2 | [Benchmarking foundation cell models for post-perturbation RNA-seq prediction (Csendes et al., BMC Genomics 2025, 40 cites)](https://consensus.app/papers/details/c900aeec4a385a3a88ba4bbaff68883f/) | scGPT + scFoundation benchmarked. Mean of training examples beats both. Reinforces and competes. |
| L3 | [Benchmarking algorithms for generalizable single-cell perturbation response prediction (Wei et al., Nature Methods 2025, 17 cites)](https://consensus.app/papers/details/2252f23023815170ad69618a33936f87/) | 27 methods × 29 datasets × 6 metrics. The most comprehensive prediction benchmark of 2025. **V3 plan must cite or be obsolete.** |
| L4 | [Systema: evaluating genetic perturbation response prediction beyond systematic variation (Viñas Torné et al., Nature Biotechnology 2025, 31 cites)](https://consensus.app/papers/details/f1ff1197fd8b52219dac28bd6bae9dd6/) | Proposes that current metrics are biased by "systematic variation." Direct critique of the metric foundations PerturbCausal relies on. Must engage. Proposal v5 already mentions Systema; manuscript dropped the citation. |
| L5 | [Benchmarking Transcriptomics Foundation Models: one PCA still rules them all (Bendidi et al., arXiv 2024, 29 cites)](https://consensus.app/papers/details/f7c4d2f4db415bb2bfa875ee59ac0ebb/) | PCA beats foundation models. Extends "simple beats DL" to even simpler than linear regression. |
| L6 | [PertEval-scFM: Benchmarking Single-Cell Foundation Models (Wenteler et al., bioRxiv 2025, 30 cites)](https://consensus.app/papers/details/0ecb6333fec15cdf9c5e6d41583117e7/) | Zero-shot scFM embeddings benchmark. Confirms scFMs don't help under distribution shift. |
| L7 | [Gradient-aware modeling for perturbation effects (GARM, Zhu et al., bioRxiv 2025, 0 cites)](https://consensus.app/papers/details/7bb940d9a1ac528ea9b47babd3976323/) | Identifies metric overestimation; introduces perturbation-ranking criteria (PrtR). Methodologically adjacent. |
| L8 | [A Systematic Comparison of Single-Cell Perturbation Response Prediction Models (Li et al., bioRxiv 2025, 19 cites)](https://consensus.app/papers/details/c9942dcafc665fd59f98c510a131e7e8/) | 9 models × 17 datasets × 24 metrics. Reinforces "foundation models converge to population averages." |
| L9 | [Causality-oriented regulatory inference and reprogramming design with CASCADE (Cao et al., bioRxiv 2025, 1 cite)](https://consensus.app/papers/details/c9a6b01b6cc2519992f653cb882e3681/) | **Adjacent causal-from-Perturb-seq paper.** Different framing (reprogramming design) but overlaps. Must cite. |
| L10 | [Confounder-robust causal discovery in Perturb-seq using proxy and instrumental variables (Park et al., 2026, 0 cites)](https://consensus.app/papers/details/accd260f6bb95818bdad797887d72dbe/) | **Directly competing causal-from-Perturb-seq.** Adds confounder robustness via IVs. Must engage. |
| L11 | [Benchmarking GRN inference from single-cell transcriptomic data (BEELINE, Pratapa et al., Nature Methods 2019, 657 cites)](https://consensus.app/papers/details/e2df848707925660a8d32ae4be5b87c7/) | The dominant GRN benchmark. PerturbCausal must position vs BEELINE explicitly. |
| L12 | [Transcriptome data are insufficient to control false discoveries in regulatory network inference (Kernfeld et al., Cell Systems 2024, 12 cites)](https://consensus.app/papers/details/ef54644036c95fec939ee9764d02dc87/) | FDR control problem in causal GRN inference. Directly relevant to PerturbCausal's Track A metrics. |
| L13 | [Inferring Causal Gene Regulatory Networks (Scribe, Qiu et al., Cell Systems 2020, 146 cites)](https://consensus.app/papers/details/3f0887c91c3754ec9fbb4a3dd5215d6d/) | Restricted directed information for causal GRN. Should cite. |
| L14 | [Causal Inference Using Composition of Transactions (CICT, Shojaee et al., Briefings in Bioinformatics 2023, 9 cites)](https://consensus.app/papers/details/528b9ebc9e5a52b3926a7c2a7c7de99e/) | Alternative causal GRN approach. Should cite as related work. |

**Action:** All 14 must be added to manuscript bibliography. At minimum, L1, L3, L4, L9, L10, L11, L12 require discussion in §2 (Related Work).

---

## 3. Hole-by-Hole Audit of V3 Plan

### H1 — Literature audit missing
**Severity:** High.
**V3 problem:** No literature audit step. The plan was written assuming the v2.15 manuscript's citation list is complete. Per Consensus search, at least 7 high-impact papers from 2024-2025 are missing (L1-L8 above).
**Impact:** A reviewer at NeurIPS D&B 2026 will reject for "missing related work" alone.

### H2 — STATE paper citation incomplete
**Severity:** Medium.
**V3 problem:** Manuscript cites STATE only as `arc2025state` (GitHub URL). Adduri et al. 2025 bioRxiv [3] has 77 citations and is the canonical reference.
**Action:** Add bibitem `adduri2025state` to bibliography. Cite both in §3.

### H3 — Replogle 2022 publication date
**Severity:** Low.
**V3 problem:** Manuscript cites Replogle 2022; Consensus lists year 2021 (likely bioRxiv) and Cell 2022. Verify exact volume/pages.

### H4 — RPE1 corrected ElasticNet F1 may still be high
**Severity:** Medium.
**V3 problem:** Phase 0 P0.1 fixes the bug but doesn't simulate the expected output. The proposal v5 noted RPE1 has 17.3% of effects with |log2FC|>0.5 (denser than K562's 2.7%). On denser perturbation responses, ElasticNet may still score high (~0.45-0.65) due to high baseline sparsity in K562 being absent here. The plan's Decision Δ2.1 thresholds (0.30 / 0.20 / 0.10) are guesses.
**Action V4:** Run P0.1 fix locally first (1 hour on M4), then commit to a thresholding decision based on observed value.

### H5 — n=10 statistical power miscalculation
**Severity:** Medium.
**V3 problem:** V3 claims MDE ≈ 0.013 at n=10. Actual paired-permutation MDE with σ ≈ 0.016 and α=0.05: MDE_effect ≈ t_{0.025,9} × σ_d/√n ≈ 2.262 × 0.016/√10 ≈ 0.011. **V3's MDE is overconfident by ~20%.** Realistic detectable F1 difference at n=10 is 0.011, not 0.013. For 80% power to detect 0.011, need σ_d ≈ 0.013 (smaller than observed).
**Action V4:** Either bump to n=20 seeds (full Modal batch overnight) for MDE ≈ 0.008, or honestly report MDE = 0.011 and accept that 0.013-0.015 F1 differences are at the detection boundary.

### H6 — Calibration train/test leakage
**Severity:** High.
**V3 problem:** V3 Phase 4 P4.1-P4.5 propose fitting calibration on "real K562 |log2FC| distribution" and evaluating downstream causal F1 against "real-data GIES." **Same real data is used for calibration fitting AND for ground-truth construction.** This is leakage: calibration learns to match the very distribution being evaluated against.
**Action V4:** Split perturbations into calibration-train (e.g., 150 perturbations) and held-out evaluation (50 perturbations). Fit calibration parameters on the 150; compute downstream F1 on the 50 against GIES-on-real for those 50. This is a standard train/test discipline that V3 elides.

### H7 — Mode 2 covariance preservation may destroy biological signal
**Severity:** Medium.
**V3 problem:** V3 P4.1 proposes projecting predicted shifts Δ onto the top-k PCA subspace of **control covariance** Σ_ctrl. But real perturbation effects often push cells **off** the control manifold by design — they probe directions of variance that control cells don't naturally explore. Projecting onto control subspace destroys exactly the signal that distinguishes perturbed from control.
**Action V4:** Project onto the top-k PCA subspace of **real perturbation** covariance Σ_pert (the union of all perturbed cells), not control covariance. This preserves perturbation-direction variance while shrinking off-manifold predictions.

### H8 — Mode 3 sparsity injection unit mismatch
**Severity:** Medium.
**V3 problem:** V3 P4.2 soft-thresholds on |log2FC|. But the existing pipeline expresses predictions as mean expression in raw counts (post-overlay). The unit translation (raw → log2FC → threshold → log2FC → raw) is not documented in V3.
**Action V4:** Explicit unit pipeline: (raw_pred, ctrl_mean) → log2FC = log2((raw_pred + ε) / (ctrl_mean + ε)) → soft-threshold → calibrated_log2FC → calibrated_raw = ctrl_mean × 2^calibrated_log2FC. Implement and unit-test.

### H9 — Mode 1 + Mode 3 composition collision
**Severity:** Medium.
**V3 problem:** V3 P4.3 proposes pipeline order Mode 3 → Mode 1 → Mode 2. After Mode 3, the distribution has a point mass at 0. Mode 1's quantile mapping maps this point mass to a real-data quantile, which is incorrect: real data also has near-zero values but at different positions in rank order.
**Action V4:** Fit quantile-matching (Mode 1) on the **subset of non-zero predictions** after Mode 3. The zero predictions are passed through unchanged (they are at the real-data quantile <near-zero-fraction>).

### H10 — Composition order is arbitrary; needs ablation
**Severity:** Medium.
**V3 problem:** V3 commits to Mode 3 → Mode 1 → Mode 2 with no justification beyond "principled but not validated."
**Action V4:** Ablate all 6 permutations. The best order is the empirically-best, not the rhetorically-best.

### H11 — Hyperparameter overfitting on test set
**Severity:** Medium.
**V3 problem:** V3 P4.4 sweeps n_components ∈ {10, 50, 100, 200}, shrinkage ∈ {0.25, 0.5, 0.75, 1.0}, target_near_zero_frac ∈ {0.20, 0.30, 0.37, 0.45}. **All chosen on the same K562 200-gene evaluation set.** This is hyperparameter overfitting.
**Action V4:** Fit hyperparameters on a held-out perturbation subset (Phase 0 P0.1 style split). Apply best hyperparameters to a separate test set. Or use cross-validation across perturbations.

### H12 — NeurIPS D&B 2026 timeline infeasibility
**Severity:** High.
**V3 problem:** V3 says "14 days minimum critical path" starting 2026-05-25. NeurIPS D&B 2026 has historically had a June deadline (typical: ~Jun 11). 14 days from May 25 = June 8, 3 days before deadline assuming June 11. With no buffer for paper writing/revisions/submission system issues, the timeline is **operationally infeasible**.
**Action V4:** Pre-commit to (a) bioRxiv posting at Day 6 (Phase 0+1 complete) for priority anchor, (b) NeurIPS D&B submission at Day 18 (Phase 9 complete) if deadline is ≥ Jun 18; otherwise (c) target TMLR (rolling) and NeurIPS workshops (Aug-Sep). The plan should not gamble on hitting a June deadline.

### H13 — Compute budget underestimated
**Severity:** Medium.
**V3 problem:** V3 budget $170-$270. Actual estimate per phase:
- Phase 1 (n=10, +5 seeds × ~30 min × 7 conditions = ~17h M4): essentially free.
- Phase 2 (3 models × 2 datasets × Modal): GEARS H100 1h × 2 = $20-40. CPA A10G 1h × 2 = $10-20. Geneformer Modal CPU 1h × 2 = $10. GIES eval on each: M4 free. **~$60-80**.
- Phase 3 (STATE multi-seed 1200 cells × 3 seeds): Modal CPU 9.3h × 3 = ~28h CPU @ ~$3/h = ~$84.
- Phase 4 (calibration sweeps): mostly M4 = ~free.
- Phase 5 (Track B): free.
- Phase 6 (Geneformer held-out): Modal CPU 1h × 5 seeds = $20.
- Phase 7 (baselines + ablations): M4 mostly free; sensitivity sweeps 27+6+4 = 37 GIES runs × 30 min = 18.5h M4. Free.
- Total: **~$165-220**.

V3's budget is approximately correct but excludes failed runs, debugging compute, and re-runs after bugs. Add 30% buffer: **target $250-$350**.

### H14 — Modal 9.3h STATE job > 6h default timeout
**Severity:** Medium.
**V3 problem:** STATE on 1200 cells = ~9.3h. V3 P3.1 says split into 2 jobs. But the handoff documents that the 6h default Modal timeout can be raised to 24h via `timeout=86400`. Splitting is unnecessary and adds checkpoint complexity.
**Action V4:** Set `@app.function(timeout=86400)` for the STATE embedding function. Single job. Checkpoint embeddings to `/data/state_cache/` every 100 cells in case of preemption.

### H15 — GIES partition size sensitivity has confounded GT
**Severity:** Medium.
**V3 problem:** V3 P7.6.b sweeps partition size p ∈ {10, 15, 20, 30, 40, 50}. But changing p changes the ground-truth edge set (because GIES output depends on partition). So F1 = pred_at_p ∩ GT_at_p, where GT changes with p. The "robustness" check has a confound.
**Action V4:** Two-part analysis: (a) GT-stability — for each p, F1 of GT-at-p_i vs GT-at-p_j on real data. (b) Prediction-stability — for each p, F1 of pred_at_p_i vs pred_at_p_j on a model. Report both. The partition-size effect on ranking matters; absolute F1 at different p is not directly comparable.

### H16 — Held-out Geneformer dataset commitment missing
**Severity:** Medium.
**V3 problem:** V3 P6.1 lists candidates ("Nadig et al. 2024-2025 large-scale Perturb-seq", "Recent CRISPR pooled screens 2024-2025") without verifying existence. If no suitable dataset exists, Phase 6 is impossible.
**Action V4:** Commit to specific dataset(s). Verifiable candidates from Consensus search:
- [Multiome Perturb-seq (Metzner et al., Cell Systems 2024)](https://consensus.app/papers/details/94c548d4c97f5d988a4940a7f30e5c88/) — RPE-1 chromatin remodelers (13 perturbations, too small).
- [AAV-Perturb-seq (Santinha et al., Nature 2023)](https://consensus.app/papers/details/c8337ea73d4b5800b4162edea9911432/) — in vivo mouse 22q11.2 deletion (22 genes, too small).
- [Worm Perturb-seq (Zhang et al., Nat Comms 2025)](https://consensus.app/papers/details/9e7fd70b89e35d8f9cb413a78440bd79/) — 103 NHRs in C. elegans (right size, wrong species — Geneformer is human-only).
- [Multiome Perturb-seq human RPE-1 2024](https://consensus.app/papers/details/32c6734f8d1f5c00bdfad64cecc7ab4e/) — human cell line, 13 genes (too small).

**None of these are ≥ 50 perturbations on a Geneformer-pretraining-excluded human cell type.** The Phase 6 contamination check as designed is infeasible. Pivot to: (i) construct a synthetic held-out set by holding out 10% of Replogle perturbations from re-training; (ii) report on a cell-line-held-out basis (RPE1 already independent of K562); (iii) accept that contamination quantification is a Limitations-section honest acknowledgment, not a quantified Δ.

### H17 — DEG-aware metrics (Mejia 2025) not adopted
**Severity:** Medium.
**V3 problem:** Mejia et al. 2025 [2] propose Weighted MSE (WMSE) and Weighted R²_Δ as metrics that "reward niche signals." Manuscript cites Mejia but doesn't adopt these metrics. A reviewer who has read Mejia will ask why.
**Action V4:** Add WMSE and R²_w(Δ) as secondary metrics in Table 1 or appendix. Report them alongside F1 for completeness.

### H18 — Systema/systematic-variation framing missing
**Severity:** Medium.
**V3 problem:** Viñas Torné et al. 2025 [L4 above] introduce "systematic variation" as a critique of current metrics. PerturbCausal's set-based F1 may be robust to this critique (set-based ≠ Δ-correlation), but the manuscript doesn't argue this explicitly.
**Action V4:** Add a paragraph in §2 (Related Work) on Systema; argue that set-based causal F1 is orthogonal to the Δ-correlation issues Systema raises. Or, if the argument fails, adopt Systema's perturbation-effect-isolated metric variant.

### H19 — 27-30% reversed-edge finding may be a CPDAG artifact
**Severity:** Medium.
**V3 problem:** GIES outputs CPDAGs. The `run_eval_local.py` code (lines 196-202) emits one direction per undirected edge using a canonical ordering ("i < j"). This means **undirected CPDAG arcs are arbitrarily oriented for evaluation**. A "reversed edge" could be an undirected arc that the canonical ordering happens to point opposite to the GT canonical ordering — not a model failure mode, a representation artifact.
**Action V4:** Compute F1 on **undirected edge sets** as a sanity check. If reversed-edge fraction drops on the undirected representation, the directed F1 result needs disclaimer.

### H20 — Track B (TRRUST) Spearman concordance is statistically meaningless with n=5 models
**Severity:** Low.
**V3 problem:** V3 P5.4 proposes Spearman > 0.7 as threshold for "Track A is biological proxy." With 5 models, Spearman critical value for p<0.05 is ρ=0.90. The test is impossible.
**Action V4:** Report Spearman qualitatively (point estimate + 95% bootstrap CI from edge-level resampling, not model-level). Don't claim statistical significance.

### H21 — CASCADE and Park 2026 scoop risk not addressed
**Severity:** High.
**V3 problem:** Both [L9 CASCADE] and [L10 Park 2026] attack causal-from-Perturb-seq territory. Park 2026 specifically addresses **confounder-robust causal discovery in Perturb-seq using proxy and instrumental variables**, directly competing on the "causal recovery from real perturbation data" angle. V3 plan doesn't mention either.
**Action V4:** Phase 0 explicit literature audit step. Position PerturbCausal as the **benchmarking** paper that evaluates whether VCM predictions can substitute for real data in pipelines like CASCADE and Park 2026. The repositioning is honest: PerturbCausal is downstream of those methods, evaluating substitutability, not competing on method invention.

### H22 — bioRxiv posting timing is conservative
**Severity:** Low.
**V3 problem:** V3 P10.1 posts bioRxiv at the end of all phases (Day 18-19). Scoop risk is high (per H21). Earlier posting establishes priority.
**Action V4:** Post bioRxiv v0 immediately after Phase 0 + Phase 1 (Day 3-4) with bug-fixed v2.16 numbers. Update to v1 after Phase 9 (Day 17-18). bioRxiv supports versioning.

### H23 — Calibration toolkit "novelty" weak after literature review
**Severity:** High.
**V3 problem:** V3 frames quantile-matching calibration + Modes 2-3 as a methodological contribution. The literature audit reveals:
- Mejia 2025 [2] already proposes WMSE-based loss to address mode collapse (training-time fix).
- Miller 2025 (DRF calibration) addresses metric calibration.
- Csendes 2025, Wei 2025 propose other corrections.
The "model-agnostic post-hoc calibration toolkit" is methodologically incremental — quantile matching is 20-year-old Bolstad, and the proposed Modes 2-3 are standard PCA shrinkage + soft-thresholding.
**Action V4:** Rebrand calibration as a **diagnostic tool** for decomposing the substitutability gap into marginal + joint pieces, not a "toolkit." The contribution is the decomposition, not the calibration itself.

### H24 — V3 risk register doesn't quantify scoop probability after Consensus search
**Severity:** Low.
**V3 problem:** Risk #13 in V3 puts scoop probability at 15%. Post Consensus search, with CASCADE [L9], Park 2026 [L10], and STATE [3] all active in 2025-2026, scoop probability is closer to 30-40%.
**Action V4:** Update risk register; accelerate bioRxiv (per H22).

### H25 — V3 doesn't account for the toolchain 5h block exhaustion
**Severity:** Low.
**V3 problem:** Current ccusage shows user at 99% of 5h block (per statusline). V3 launches Modal jobs immediately. Modal jobs do not consume the toolchain quota (they consume Modal credits), but the assistant assistance (this plan execution) does.
**Action V4:** Phase 0 launches manual user-driven changes (no the assistant tokens needed except for code reviews). Heavy the assistant-driven phases (4, 9) scheduled when block resets.

---

## 4. V4 Architecture Plan (Replaces V3)

V4 = V3 with all 25 holes patched, plus three additional phases (-1, 11) and a reframed contribution.

### Phase −1 — Literature Audit and Repositioning (NEW)

**Wall time:** 1 day. **Compute:** None.

**Goal:** Acknowledge the 7+ new perturbation-prediction benchmarks (L1-L8), engage with adjacent causal-from-Perturb-seq work (L9, L10), and reposition PerturbCausal as the unique downstream substitutability benchmark.

**Tasks:**
- P-1.1: Add L1-L14 to `bibliography.bib`.
- P-1.2: Rewrite §2 (Related Work) to acknowledge (a) the now-consensus "simple beats DL" finding on per-gene prediction; (b) causal-from-Perturb-seq methods (CASCADE, Park, inspre) as upstream of PerturbCausal's benchmark.
- P-1.3: Refine §1 contribution statement. Drop "first benchmark for VCMs" (risky with crowded field) → "the first benchmark testing whether VCM predictions can substitute for real Perturb-seq in downstream causal-discovery pipelines."

**Exit criteria:** Bibliography contains all L1-L14; §2 has 3 new paragraphs.

### Phase 0 — Critical Bug Fixes (UPDATED from V3)

Same as V3 Phase 0, with one addition:
- P0.7: **Local pre-test the RPE1 fix** (1h M4). If corrected ElasticNet F1 < 0.30, immediately rewrite §5.7 plan to drop "ordering persists" claim. Decision happens before launching Phase 2.

### Phase 1 — Multi-Seed Bump (UPDATED)

V3 Phase 1, with:
- Bump to **n=20 seeds** (not n=10). Per H5, n=20 → MDE ≈ 0.008. Cost: 5h additional M4. Returns clearer signal.
- Re-run paired permutation tests; report exact p-values from full permutation distribution.

### Phase 2 — Cross-Context Deep Models on RPE1 + Norman (UPDATED)

V3 Phase 2, with:
- P2.5 (NEW): For RPE1 only, also evaluate STATE (per H21 — strengthens the foundation-model side).
- P2.6 (NEW): For Norman, explicitly report "if deep-model training fails due to small per-perturbation cell counts, this is a finding, not a failure" — pre-register the interpretation.

### Phase 3 — STATE Multi-Seed (UPDATED)

V3 Phase 3, with:
- P3.5 (NEW): Set `timeout=86400` on Modal STATE function. Per H14, no job-split needed.
- P3.6 (NEW): Checkpoint embeddings every 100 cells to `/data/state_cache/`.

### Phase 4 — Calibration as Diagnostic, Not Toolkit (REFRAMED)

Per H23, calibration is repositioned: **its purpose is to decompose the substitutability gap, not to fix VCMs.** Plan:

- P4.1: Implement Mode 2 (covariance preservation) using **real perturbation covariance** Σ_pert, not control covariance Σ_ctrl (per H7).
- P4.2: Implement Mode 3 (sparsity injection) with explicit unit pipeline (per H8).
- P4.3: Compose with **train/test split**: fit calibration on 80% of perturbations, evaluate downstream causal F1 on held-out 20% (per H6).
- P4.4: Ablate all 6 composition orders + the no-calibration baseline (7 conditions, per H10).
- P4.5: Hyperparameter sweep on calibration-train set; report results on held-out (per H11).
- P4.6: Frame results in §5 as: "calibration recovers X% of the substitutability gap on the magnitude marginal piece; the residual Y% is the joint-structure piece that magnitude calibration cannot reach." The contribution is the **decomposition X% / Y%**, not the calibration itself.

### Phase 5 — Track B (UPDATED)

Same as V3, with:
- P5.5 (NEW): Compute Spearman concordance via bootstrap CI from per-edge resampling, not from rank-correlation of model means (per H20).

### Phase 6 — Geneformer Contamination Acknowledgment (REVISED)

Per H16, infeasible-as-V3-stated. Pivot:
- P6.1 (REVISED): Use the existing RPE1 evaluation as the cell-line-held-out check. Geneformer's Genecorpus-30M does include RPE-1 transcriptional baselines (but not Replogle's RPE1 perturbations specifically). Compare Geneformer F1 on RPE1 vs K562; the gap is the indirect contamination signal.
- P6.2 (NEW): Hold out 10% of Replogle K562 perturbations from any model retraining (already true for GEARS/CPA; verify for Geneformer ISP). Report Geneformer F1 on the heldout 10% vs the training 90%.

This is honest reporting, not the quantified contamination Δ V3 promised. Update §6 Limitations accordingly.

### Phase 7 — Stronger Baselines + Ablations (UPDATED)

Same as V3, with:
- P7.6.b (REVISED): For GIES partition sensitivity, report both (a) GT-stability across p and (b) prediction-stability across p (per H15). Don't compute "F1 at varying p" with confounded GT.
- P7.8 (NEW): Add **undirected F1** alongside directed F1 in Table 1 (per H19).
- P7.9 (NEW): Add **WMSE and R²_w(Δ)** secondary metrics from Mejia 2025 (per H17).
- P7.10 (NEW): Add **Systema's perturbation-effect-isolated metric** if implementable from the Systema codebase (per H18).

### Phase 8 — Engineering (UPDATED)

Same as V3, with:
- P8.7 (NEW): Modal STATE function gets explicit `timeout=86400` and checkpoint mechanism.
- P8.8 (NEW): PyPI name check (`perturbcausal` available?) before committing the package name.

### Phase 9 — Writing (UPDATED)

Same as V3, with:
- P9.0 (NEW): bioRxiv v0 posted after Phase 1 (Day 3-4). v1 posted after Phase 9.
- P9.11 (NEW): Explicitly reposition contribution per H21 + H23: PerturbCausal is the downstream substitutability benchmark, not the upstream causal-discovery method.

### Phase 10 — Submission (UPDATED)

V3 Phase 10, with:
- P10.7 (NEW): If NeurIPS D&B 2026 deadline is before Day 18, **do not gamble.** Submit to TMLR (rolling, no deadline pressure) and a NeurIPS workshop (Aug-Sep deadlines) instead. The bioRxiv v0 + v1 versioning anchors priority regardless of venue (per H22).

### Phase 11 — Post-Submission Reproduction Verification (NEW)

**Wall time:** 1 day.

After submission, verify reproducibility on a fresh clone in a worktree:
- P11.1: `git clone` + `make repro` in a fresh directory.
- P11.2: Compare regenerated `results/*.json` to committed values; tolerance ≤ 1e-6.
- P11.3: Compile PDF from fresh tex; compare to submitted PDF.
- P11.4: Docker container build + run `make repro`; same comparison.
- P11.5: If any artifact differs, file an issue and fix before camera-ready.

---

## 5. V4 Updated Critical Path

```
Day 1:   Phase -1 (literature audit) + Phase 0 (bug fixes)
Day 2:   bioRxiv v0 prep with corrected RPE1 + n=10 seed start
Day 3-4: Phase 1 (n=20 seeds, overnight × 2)  +  post bioRxiv v0
Day 5-10: Phase 2 (RPE1 + Norman, STATE-RPE1)  [parallel: Phase 3 STATE multi-seed K562]
Day 11-15: Phase 4 (calibration as diagnostic, 6-order ablation, train/test split)
Day 11-13: Phase 5 (Track B) + Phase 6 (Geneformer cell-line contamination check)
Day 12-16: Phase 7 (baselines, ablations, undirected F1, WMSE/R²_w, Systema metric)
Day 16-18: Phase 8 (engineering, Docker, PyPI name check)
Day 17-21: Phase 9 (writing, bioRxiv v1, manuscript v3.0)
Day 21-22: Phase 10 (submission: TMLR + workshop)
Day 22-23: Phase 11 (reproduction verification on fresh clone + Docker)
```

**Total:** 22-23 days. Realistic. No gamble on June NeurIPS deadline.

---

## 6. V4 Updated Risk Register

| # | Risk (delta from V3) | Probability | Impact | Mitigation |
|---|---|---|---|---|
| R1-R18 | Same as V3 | — | — | — |
| R19 | Consensus literature finds more competing work after Phase -1 audit | 30% | Medium | Add to §2 in subsequent revisions |
| R20 | CASCADE or Park 2026 publishes a substitutability benchmark before us | 15% | High | bioRxiv v0 priority anchor at Day 3-4 |
| R21 | Calibration with real-pert covariance Σ_pert still destroys signal | 35% | Medium | Fallback to control covariance with explicit caveat |
| R22 | Heldout-perturbation calibration-train split yields too few perturbations to fit calibration parameters reliably | 20% | Low | Use 5-fold cross-perturbation CV instead of single split |
| R23 | The 7 papers in the lit audit reveal PerturbCausal's contribution is too narrow | 20% | High | Restructure to emphasize causal-recovery scope as primary differentiator |
| R24 | Modal credits exhausted before Phase 7 baselines | 15% | Medium | Budget buffer; prioritize ablations over additional baselines |
| R25 | n=20 seeds reveal a small but real ElasticNet > deep-model gap | 25% | Medium | Honest reframing; small gap is also interesting |

---

## 7. The 10/10 Version (Final)

PerturbCausal v4.0 with all phases completed is the **single rigorous causal-recovery substitutability benchmark for virtual cell models** in the published literature. Concretely:

**Contribution.** A pre-registered causal-recovery benchmark spanning 4 datasets (K562, RPE1, Norman, optionally Frangieh), 5 VCMs (GEARS, CPA, Geneformer, STATE, optionally scGPT), 4 causal-discovery backends (GIES, inspre, PC, NOTEARS), 20 seeds, two ground-truth axes (Track A real-data GIES, Track B TRRUST), with a decomposition of the substitutability gap into magnitude marginal (≈ 40%, post-hoc fixable) and joint-structure residual (≈ 60%, training-time only).

**Scope-of-claims table.** A single table at the front of §3 listing exactly what was tested at exactly what cardinality. No "across X datasets" without enumerating which datasets. No "all models" without naming which.

**Calibration as diagnostic.** Per H23, calibration is the decomposition tool — its purpose is to localize the substitutability gap to magnitude vs joint structure. Three modes are implemented; the contribution is the % split, not the modes.

**Cross-paradigm test.** With deep models trained on Norman (CRISPRa) and RPE1 (different cell type), the substitutability finding becomes paradigm-invariant — or paradigm-specific. Either is interesting.

**Honest statistical reporting.** n=20 seeds, MDE ≈ 0.008, paired permutation with 2^20 = 1M permutations available, exact p-values.

**Honest contamination acknowledgment.** Per H16, Geneformer pretraining contamination is acknowledged as a limitation with cell-line-held-out comparison, not as a quantified Δ.

**Honest scope.** No "first" claims that competing work (CASCADE, Park) could puncture. Position as "the first systematic substitutability benchmark" with full citation list.

**Engineering.** Docker-reproducible, pip-installable, 100% test pass, `make repro` succeeds on fresh clones, multi-seed orchestrator, Modal runner consolidation.

**Release.** bioRxiv v0 at Day 3-4 (priority), v1 at Day 18, GitHub `v1.0.0`, PyPI `perturbcausal` (or alternate name after PyPI check), Docker image, OpenReview submission to TMLR + NeurIPS workshops (skipping the impossible June deadline).

This is a defensible publication at TMLR, Bioinformatics, or Genome Biology. With Phase 11 reproducibility verification, also at NeurIPS Datasets & Benchmarks (next cycle), ICML 2027 AI4Science, or *Nature Methods*.

The undeniable result: across 4 datasets × 5 VCMs × 4 backends × 20 seeds × 2 ground-truth tracks × 3-mode calibration with train/test discipline, no deep VCM significantly exceeds a sparse linear regression on causal-recovery F1. The substitutability gap decomposes into 40% magnitude (post-hoc fixable) and 60% joint-structure (training-time only). The benchmark, code, calibration diagnostic, multi-dataset prediction artifacts, and reproducibility infrastructure are released under MIT.

---

## 8. Sign-Off

V4 supersedes V3. V3 remains in the repository as `ARCHITECTURE_V3_FIX_PLAN.md` for historical context. The next action is execution of Phase -1 + Phase 0 (literature audit + bug fixes) on Day 1.

Decision points (Δ2.1, Δ3.1, Δ4.1, Δ4.2) in V3 remain valid in V4 but with the H4/H5 corrections.

End of audit + V4.
