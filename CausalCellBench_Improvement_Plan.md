# CausalCellBench v2 — Improvement Plan

**Date:** 2026-05-01
**Authors:** Aayan Alwani, Ethan Wang
**Target:** ICML 2026 workshop (initial), ICLR 2027 main track (stretch)

This plan addresses the 16 valid critiques identified in the rebuttal review. Phases are sequenced so each phase's results inform the next; each phase has an explicit gate condition and a defined fallback if the gate fails.

---

## Executive summary

The paper currently has one critical gap (Mode-collapse vs. inflation reconciliation) that determines whether the central claim survives. Two further gaps (statistical rigor; technical-duplicate baseline) are unfalsifiable without data we have not yet collected. After these are resolved, the heavier model/dataset/algorithm extensions follow.

Total projected timeline: **6 weeks of focused work** for workshop-ready v2; **12 weeks** for main-track-ready v3. Compute budget: <$300 additional.

| Phase | Work | Days | Gate | Fallback if gate fails |
|---|---|---:|---|---|
| 0 | Branching + repo hygiene | 1 | — | — |
| 1 | **Mode-collapse vs. inflation** (item 1) | 4 | Reconciliation found | Reframe abstract, drop "inflation" framing |
| 2 | Multi-seed (4) + technical duplicate (3) | 4 | F1 differences > 1σ | Demote ElasticNet-beats-DL to descriptive |
| 3 | Citation + rename + lambda sweep (2, 5, 6, 7, 16, 19, 20, 21) | 3 | — | — |
| 4 | Mechanism via alternative loss (14) + calibration re-evaluation (11) | 5 | Inflation drops with new loss | Demote calibration to post-hoc |
| 5 | Drop Geneformer, add STATE (8) | 6 | STATE available locally | Stay with current 3 models, flag in limitations |
| 6 | Add Norman 2019 (9) + scale sweep to 500/1000 (13) | 7 | Compute completes | Drop 1000, keep 500 |
| 7 | NOTEARS + PC + UT-IGSP (10) | 6 | Robustness across algos | Limit claims to GIES/inspre |
| 8 (optional) | Full architectural ablation (17) | 14 | — | Skip for workshop |

Delete the "v5 calibration is core contribution" framing if Phase 2 shows the deltas (+1.6/+0.8/−0.9) are within seed variance. Demote to post-hoc analysis at that point.

---

## Phase 0 — Repo hygiene (1 day)

**Why first.** Everything downstream depends on a clean baseline.

**Work:**
- Branch `main` → `v2` for all v2 work; tag current state as `icml-workshop-v1`.
- Cache the current vanilla and calibrated `eval_results_n200_*.json` and GIES caches as `*_v1.json` so they survive when we rerun.
- Remove `Geneformer` from active model list (Phase 5 will replace) but keep its outputs cached for retrospective comparison.
- Snapshot `submissions/icml/figs/` to `figs_v1/`.

**Deliverable:** `git tag v1.0-icml-workshop` and a clean `v2` working tree.

---

## Phase 1 — The gate: mode-collapse vs. inflation reconciliation (4 days)

**Critique:** #1 (gate condition) + #15 (DEG concentration) — these are the same analysis.

**Hypothesis to test.** Mode collapse and inflation can both be true simultaneously if deep models predict an *identical, dense* response pattern across all perturbations. That would be: high entropy across genes (looks like inflation, our finding), low entropy across perturbations (looks like mode collapse, the literature's finding).

**Tasks:**

1. **Per-perturbation Δ-variance ratio.** For each model, compute Var[predicted Δ_p] / Var[real Δ_p] per perturbation p. Mode collapse predicts ratios <<1. Inflation predicts ratios >>1.

2. **Inter-perturbation prediction similarity.** For each model, compute mean cosine similarity of predicted shift vectors across all (perturbation_i, perturbation_j) pairs. Mode collapse predicts high similarity (model produces same response everywhere); a model with genuine perturbation specificity has low similarity.

3. **Per-perturbation DEG concentration.** Real perturbations are typically sparse — a few large effects on a small set of genes per perturbation. Compute # genes per perturbation with |log2FC| > 0.5 in real vs. each model. Histogram both.

4. **Cross-threshold inflation curve.** Replot the "inflation ratio" of fraction(|log2FC| > τ) for τ ∈ {0.1, 0.25, 0.5, 1.0, 2.0}. If inflation is uniform across thresholds, the result is a single distribution-shift story; if it's threshold-dependent, the story is more nuanced.

**Code locations:**
- New script: `scripts/phase1_distribution_diagnostics.py`
- Output: `results/phase1/distribution_reconciliation.json` + 3 new figure panels

**Gate.** One of three outcomes:
- **(A) Both literatures correct under different definitions.** Per-perturbation Δ-variance is low (mode collapse) and pooled |log2FC| density is high (our inflation). Inter-perturbation cosine is high (predictions look the same across perturbations). → Reframe paper around "deep VCMs collapse perturbation specificity onto a single dense response pattern, simultaneously matching mode-collapse and inflation observations under their respective definitions." This is the *strongest* possible outcome — connects two threads.
- **(B) Inflation is real and unrelated to mode collapse.** Per-perturbation Δ-variance is similar to or larger than real, AND inter-perturbation cosine is low. → Current claim survives. Add explicit reconciliation paragraph.
- **(C) Inflation is an artifact of pooling.** When stratified by per-perturbation, predictions are actually compressed. → Inflation framing is wrong. Reframe paper as "deep VCMs exhibit mode collapse, which destroys downstream causal recovery; calibrating against the *correct* underlying distribution requires per-perturbation operations."

**Most likely:** (A). Quick spot-check of the data already supports this — the inter-perturbation similarity should be revealing.

**Deliverable:** Updated Section 4.2 of the paper with the reconciliation argument. Three new figure panels for Figure 2 (or a new Figure 2.5).

---

## Phase 2 — Statistical foundation (4 days)

**Critiques:** #4 (multi-seed) + #3 (technical-duplicate noise ceiling) + #11 (calibration re-evaluation).

**Why second.** Without proper denominators (noise ceiling) and proper error bars (multi-seed), every comparison in the paper is unfalsifiable.

**Tasks:**

1. **Run the multi-seed evaluation.** `scripts/run_multi_seed.py` is already coded with seeds (42, 123, 456, 789, 1024). Each seed re-randomizes gene subset, control subsample, and partition order. Wall time: ~2.5 hr/seed × 5 = ~12 hr. Run vanilla + calibrated for all 3 models.

2. **Add per-condition technical-duplicate baseline.** For real K562 perturbations:
   - For each perturbation, randomly split cells in half (50/50).
   - Run GIES on each half independently.
   - Compute F1 between the two recovered graphs.
   - Average across 10 random splits.
   - This is the noise ceiling.

3. **Re-express all model F1s as fraction of noise ceiling.** This is the metric that survives reviewer scrutiny.

4. **Re-evaluate calibration with seed CIs.** Run calibration pipeline across all 5 seeds. If +1.6/+0.8/−0.9 deltas have CIs that overlap zero, demote calibration from "core contribution" to "post-hoc analysis."

**Gate.** Multi-seed CIs reveal whether the differences between methods are statistically meaningful.

- If ElasticNet-beats-deep-models has paired t-test p < 0.05 across seeds → claim stands.
- If p > 0.05 → reframe as "ElasticNet matches deep models within seed variance, indicating that 100M+ parameters provide no advantage over a sparse linear baseline at this scale."
- If technical-duplicate F1 < 0.45 → all models are at >85% of ceiling and the practical gap is small (frame more carefully).

**Deliverable:** `results/multi_seed/` populated with five `.json` files. New Table 1 with seed mean ± std and paired-t p-values. New row in headline table for "Technical Duplicate (noise ceiling)."

---

## Phase 3 — Lightweight writing fixes (3 days)

**Critiques:** #2 (cite A-E in abstract) + #5 (rename) + #6 (Callahan) + #7 (Wong/Miller engagement) + #16 (justify inspre) + #19 (lambda sweep explanation) + #20 (GitHub link) + #21 (cite cleanup).

**Why third.** None of these depend on new experiments. They can be done in parallel with Phase 4 compute.

**Tasks:**

1. **Rename.** Top candidate: `PerturbCausal`. Alternative: `VCM-Causal`. Avoid `CausalCellBench` (too close to `CausalBench`). Globally find-and-replace across `submissions/icml/`, `causalcellbench/` package, repository name.

2. **Abstract rewrite.** Reorganize so it leads with conceptual contribution, not numbers. Add Ahlmann-Eltze citation explicitly: *"Extending Ahlmann-Eltze and Huber (2025)'s per-gene-MSE finding, this work measures whether VCM outputs preserve causal regulatory structure under interventional discovery..."* Cite Callahan et al. 2025 as the framework we operationalize.

3. **Verify Callahan et al. 2025 authorship** — currently `Anonymous` in `references.bib`. Resolve and update inline citation.

4. **Add a paragraph** in Related Work addressing whether causal F1 qualifies as a "well-calibrated metric" in Wong/Miller's dynamic-range-fraction sense. Argue: F1 against real-data GIES *is* a proper rank-based metric (standard precision/recall on edge sets) and therefore not subject to the calibration concern they raise. Defensible position; should be stated explicitly.

5. **Justify inspre choice** — one paragraph in Methods: "inspre (Brown et al., 2025) is selected because it scales to 1000+ node networks (Section 3.5 of Brown et al.) without partitioning, and operates directly on the average causal effect matrix, exposing signal differences GIES partitioning equalizes."

6. **Inspre lambda sweep.** For the four conditions (real, ElasticNet, GEARS, CPA, Geneformer), trace recovered edge count across the full λ path. Report the smallest λ at which any edge is recovered. Add as Figure 3 supplementary panel.

7. **Add GitHub link** to abstract footer for review (anonymized via `https://anonymous.4open.science/...` if double-blind). Real link on acceptance.

8. **Citation cleanup:** disambiguate Bunne 2024 (the *Cell* perspective) from Roohani 2024 (GEARS) and Roohani 2025 (Virtual Cell Challenge). Verify each entry has a working DOI.

**Deliverable:** Updated `.tex` and `.bib`; Figure 3 supplementary panel; new `Related Work` paragraph.

---

## Phase 4 — Mechanism (5 days)

**Critique:** #14 (mechanism for cross-architecture identical inflation) + #11 (calibration re-evaluation).

**Why fourth.** This requires the multi-seed CIs from Phase 2 to determine whether the calibration is doing real work. It also requires the Phase 1 reconciliation framing to define what "fixing the inflation" actually means.

**Hypothesis.** The cross-architecture inflation match arises from a shared training objective (per-gene MSE on log-normalized counts). Test by retraining with an alternative loss family.

**Tasks:**

1. **Re-train CPA with Poisson NLL loss** (instead of Gaussian-on-log-counts MSE). CPA is the cheapest to retrain (~6 min on M4 CPU). Same training data, same epochs, same hyperparameters except the loss. Measure (a) per-perturbation Δ-variance ratio, (b) inflation fraction, (c) downstream causal F1.

2. **Optional: Re-train CPA with WMSE** (Mejia et al. 2025's weighted MSE). If multi-seed CIs allow, compare WMSE-trained CPA to vanilla and to quantile-calibrated CPA. This is the direct comparison to the alternative correction approach.

3. **Train GEARS with Poisson NLL** if time permits (~70 min on CPU).

**Gate.** If alternative-loss CPA shows substantially reduced inflation and improved causal F1, the "objective-driven" claim is supported and the calibration story is reframed: *"calibration is one fix; retraining with a distribution-aware loss is the other; both partially recover causal F1."* If the alternative loss doesn't help, the architecture-or-data hypothesis becomes more credible.

**Deliverable:** New result subsection — "Alternative-loss CPA exhibits {N}% reduced inflation and {M} F1 points improvement on causal recovery."

---

## Phase 5 — Modernize models (6 days)

**Critique:** #8 (dated models).

**Why fifth.** This is more disruptive — drops one model and adds another. Should not happen until the framework is stable.

**Tasks:**

1. **Drop Geneformer.** It's flagged by Ahlmann-Eltze, wasn't designed for perturbation prediction, and is the model that makes the calibration story messiest. Move to "Discussion: limitations" with a one-line note that it was excluded due to architecture mismatch with the perturbation prediction task.

2. **Add STATE (Arc Institute, 2025).** Public weights, designed for perturbation prediction, current state-of-the-art on Virtual Cell Challenge. Should be tractable on M4 CPU for inference; training may need GPU but inference on Replogle K562 should be feasible.

3. **Optional: Add CellFlow or scLAMBDA.** Both are 2025 models with public code. Lower priority than STATE.

4. **Re-run Phase 1 + Phase 2 + Phase 4 analyses on the updated model roster** (CPA, GEARS, STATE).

**Gate.** STATE inference completes successfully on Replogle K562 N=200 evaluation set. If STATE requires GPU and we can't budget for it, fallback: keep current roster, explicitly flag in limitations.

**Deliverable:** Updated Section 3.2 (Models). Updated Tables 1 and 2 with STATE in place of Geneformer.

---

## Phase 6 — Robustness via additional datasets and scales (7 days)

**Critiques:** #9 (Norman 2019) + #13 (scale to 500–1000).

**Why sixth.** Cross-paradigm and cross-scale generalization. These extensions are independent of the architecture work.

**Tasks:**

1. **Add Norman 2019 dataset.** CRISPRa, combinatorial perturbations. Different mechanism (gain-of-function vs. loss-of-function), different lab, different scale. If the inflation finding replicates here, it's not a Replogle artifact.

2. **Scale sweep to N=500 and N=1000 K562.** GIES at N=1000 is intractable but inspre handles it. Run inspre at N=500 and N=1000 across all conditions. Plot F1 vs. N as a curve. The expected story: deep models lose ground further as N increases (because the joint-distribution failure compounds with more genes).

3. **(Optional) Adamson 2016 or Tahoe-100M subset** — only if compute allows.

**Gate.** N=1000 inspre completes within 24 hours per condition. If not, drop to N=500 and explicitly limit the scaling claim.

**Deliverable:** New figure: F1 vs. N (50, 75, 100, 125, 150, 175, 200, 500, 1000) per model, with shaded CIs across seeds. New section on Norman 2019 results.

---

## Phase 7 — Algorithmic robustness (6 days)

**Critique:** #10 (more causal discovery algorithms).

**Why seventh.** Tests whether GIES/inspre conclusions generalize to other interventional causal discovery methods.

**Tasks:**

1. **NOTEARS-INT** (interventional NOTEARS). Continuous-optimization, gradient-based; should run on M4 CPU at N=200.
2. **PC algorithm** (constraint-based). Standard baseline, fast.
3. **UT-IGSP** (Unknown-Target Interventional Greedy Sparsest Permutation). Modern interventional method.
4. Run each algorithm on each condition (real, ElasticNet, CPA, GEARS, STATE) at N=200 K562. Report F1 per (algorithm × condition) cell.
5. Compute "consensus" F1 by majority-voting across algorithms.

**Gate.** ElasticNet-beats-deep-models survives across all 5 algorithms (GIES, inspre, NOTEARS-INT, PC, UT-IGSP). If not, claim is restricted to specific algorithm classes.

**Deliverable:** New Figure 5 with algorithm × condition heatmap. New paragraph on consensus result.

---

## Phase 8 (optional) — Architectural ablation (14 days)

**Critique:** #17.

**Why optional.** Only required for ICLR/NeurIPS main track. Workshop submission can defer.

**Work:** Train CPA variants where exactly one component is replaced (decoder distribution; latent space dimensionality; loss; normalization). Measure inflation magnitude per variant. Identify which single component contributes most to the inflation pathology.

---

## Resource accounting

**Compute budget:**
- Phase 2 multi-seed: ~12 hr × 5 seeds = ~60 hr CPU. Free on M4.
- Phase 4 alt-loss training: ~6 min × 3 models × 2 losses = ~36 min. Free.
- Phase 5 STATE inference: probably ~1 hr × 200 perturbations on M4 if it works. ~$50 GPU rental fallback.
- Phase 6 Norman 2019: ~6 hr CPU. Free.
- Phase 6 N=500/1000: inspre ~30 min/condition × 5 conditions × 2 scales = 5 hr. Free.
- Phase 7 NOTEARS/PC/UT-IGSP: ~30 min/algorithm × 3 algorithms × 5 conditions = 7.5 hr. Free.
- **Total compute budget needed: ~$50–150** (only if STATE requires GPU; otherwise free on M4).

**Human time budget:**
- Workshop-ready (Phases 0–7): **~6 weeks of focused work** for two authors.
- Main-track-ready (Phases 0–8): **~10–12 weeks**.

---

## Decision tree

```
                Phase 1: Reconciliation
                  /    |    \
            (A)       (B)         (C)
           Both       Inflation   Inflation
           true       isolated    is artifact
            |            |            |
            v            v            v
        Reframe       Continue    Reframe
        around        with        as mode-
        joint mode-   current     collapse
        collapse +    claim       paper
        inflation
            |            |            |
            +-----+------+
                  v
            Phase 2: Multi-seed
                  /    \
              p<0.05    p>0.05
                |          |
                v          v
          ElasticNet   Reframe to
          claim holds  "match within
                       seed variance"
                  |
                  v
            Phase 3 (writing) and
            Phase 4 (mechanism)
                  |
                  v
            Phases 5-7
                  |
                  v
         ICML workshop submission
                  |
                  v
       (Optional) Phase 8 → ICLR main
```

---

## What to keep, what to drop

**Keep:**
- The CausalCellBench framework (rename only)
- The N=200 K562 evaluation (becomes one of N ∈ {50, 75, 100, 125, 150, 175, 200, 500, 1000})
- Overlay sampling
- inspre signal probe (extend to full lambda path)
- The calibration module (becomes "post-hoc analysis" if multi-seed shows small effect)
- The released code

**Drop or demote:**
- Geneformer as a primary model (move to limitations)
- "First benchmark" framing in abstract (cite Ahlmann-Eltze, position as extension)
- Calibration as headline contribution (demote conditional on Phase 2 results)
- "Causal world model" framing if Callahan et al. uses similar language (cite as the framework being instantiated)
- The current name (`CausalCellBench`)

**Add:**
- Mode-collapse vs inflation reconciliation (Phase 1)
- Technical-duplicate noise ceiling (Phase 2)
- Multi-seed CIs and paired t-tests (Phase 2)
- Mechanism via alternative loss (Phase 4)
- STATE model (Phase 5)
- Norman 2019 dataset (Phase 6)
- NOTEARS / PC / UT-IGSP (Phase 7)
- Larger gene-set scales (Phase 6)
- Engagement with Wong/Miller calibrated metrics (Phase 3)
- Engagement with Callahan et al. framework (Phase 3)

---

## What success looks like

**Workshop submission (Week 6):**
- Mode-collapse vs inflation reconciliation in Section 4.2 with 3 new diagnostic panels.
- Multi-seed F1s with paired-t p-values in Table 1.
- Technical-duplicate noise ceiling included as a row in Table 1.
- Citation engagement with Ahlmann-Eltze, Callahan, Wong/Miller, Mejia.
- Renamed (`PerturbCausal` or similar).
- 4 deep models (CPA, GEARS, STATE, + one optional) instead of 3.
- 2 dataset families (Replogle + Norman 2019).
- Inspre lambda sweep documented.
- Mechanism evidence: alternative-loss CPA result.

**Main-track submission (Week 12, optional):**
- All workshop deliverables, plus
- 5 causal discovery algorithms (GIES, inspre, NOTEARS-INT, PC, UT-IGSP)
- Scale sweep to N=1000
- Component-level architectural ablation
- A senior co-author (Ashley Laughney or causal ML collaborator)
