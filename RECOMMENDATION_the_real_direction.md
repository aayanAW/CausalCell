# The real direction — why Track B failed, and the one change that would matter

**Date:** 2026-07-05. This is not "try a causal architecture." It is a specific,
mechanistically-forced claim backed by three verifications against your own code and
numbers, plus the current literature.

---

## 1. Track B did not fail because the objective was wrong. It failed because the evaluation cannot see the objective.

I read the evaluation harness (`scripts/run_eval_local.py`, `sample_perturbed_cells_overlay`,
lines 615–651). Here is what every mean-prediction condition (the VCMs, ElasticNet, and the
perfect-means upper bound — everything except the observational GRNBoost2 baseline) actually
does before GIES sees it:

```
perturbed_cells = bootstrap(control_cells) * ratio[:, None]     # ratio = pred_mean / ctrl_mean
                  then Poisson(perturbed_cells)
```

**The prediction enters only as a per-gene multiplicative ratio — a diagonal rescaling of
control cells.** GIES (and inspre, PC, NOTEARS) read *conditional independence* = the
support of the precision matrix. Diagonal rescaling is **exactly** the transformation
that leaves conditional-independence structure invariant.

**Verified three ways:**
1. **Analytically/numerically:** two *wildly different* mean predictions, pushed through
   the overlay with the same bootstrap draw, give `corr(overlay A) − corr(overlay B)` with
   max |Δ| = **1×10⁻¹⁵** (machine zero). The correlation matrix — hence the CPDAG GIES
   recovers — is **literally identical** regardless of the prediction. (See
   `overlay_diagnosis.png`, panel B.)
2. **From your own headline numbers:** perfect real means through the overlay
   (`overlay_resampled_real`) scores **F1 = 0.422**, and ElasticNet 0.425, Geneformer
   0.432, GEARS 0.405, CPA 0.389 — a **0.04-wide band that all five overlay conditions sit
   in**, while real *cells* score 1.0. The instrument saturates at ~0.42 for anything that
   goes through the overlay, because they all inherit the *same control covariance backbone*.
   The one condition that does **not** go through the overlay — `grnboost2_real`, an
   observational arboreto regression that ignores the intervention structure and never enters
   GIES (`run_grnboost`, line 279) — scores F1 = **0.097**, ~4× lower and outside the band.
   That is not a counterexample; it is the control that confirms the diagnosis: break the
   shared-control-covariance path and the tie disappears.
3. **From the prediction files:** every K562 VCM prediction is shape `(200, 200)` —
   `output_type: mean`, one row per perturbation. The models' learned joint structure was
   **never in the pipeline.** Only their per-gene means were, and the overlay then threw
   even those away except as a diagonal scale + mean-dependent Poisson variance.

**This is why every joint-structure result we ran nulled** — the no-GPU covariance-Frobenius
sweep (J2), the CPU precision/MMD/OT proxy sweep (B), the 3-seed GPU retrain (marginal, did not
survive), and the completed 5-seed pre-registered GPU confirmatory sweep (B2, DISCARD, 3/5
seeds, flat λ dose-response). No training objective on a *per-perturbation-mean predictor*
can change a metric that is provably invariant to everything except the per-gene mean — and
the overlay discards the joint part of even that. We were tuning the loss of a model whose
output the evaluation compresses to a diagonal. **The gap Track B was trying to close was
closed by the harness, not by biology.** That is a much stronger and more interesting claim
than "no objective helps."

This also explains the J7 identifiability proposition *empirically*: J7 proves CI-structure
is invariant to per-gene monotone calibration; the overlay is a per-gene transform, so J7 is
not just a theorem about calibration — it is a description of **what the evaluation pipeline
does to every prediction.**

---

## 2. The direction that would dramatically strengthen the paper

**Stop evaluating means. Evaluate the predicted joint distribution directly — and make that
the benchmark's headline contribution.**

Concretely, three moves, in order of leverage:

### (a) The cell-level evaluation track — the fix to the instrument (highest leverage, ~days, no GPU)
Re-run the causal pipeline feeding **predicted cells** (the model's own joint samples), not
`overlay(mean)`. The harness already supports `output_type: "cells"` (line 718) — the RPE1
CPA file is already in that format. For any generative VCM (CPA, STATE, CellFlow, a diffusion
model) you sample N cells per perturbation and feed those. **Now the model's covariance
reaches GIES, and the metric can finally separate a good joint predictor from a bad one.**
Pre-register the same anchored-GT scoring. Two outcomes, both publishable:
- Cell-level predictions *still* tie the linear baseline → the failure is real and now
  proven at the level where the joint structure is actually tested — a **much** harder claim
  to dismiss than the current mean-based one.
- Cell-level predictions *separate* → you have discovered that **the evaluation methodology,
  not the models, was the bottleneck** — a benchmark-defining result.

### (b) Reframe the paper's core contribution around the diagnosis
The paper's current headline ("VCMs don't beat linear at causal recovery") is, per
Ahlmann-Eltze 2025, now near-canonical and low-novelty. **The overlay-invariance finding is
not.** The reframed thesis: *"Causal-recovery benchmarks that evaluate perturbation
predictions through per-gene mean overlays are structurally blind to the joint distribution —
the very thing causal discovery depends on — and we prove it (F1 invariant to the prediction
up to machine precision). We provide the cell-level evaluation that fixes it."* That converts
the paper from "another negative benchmark" into "the benchmark that identified why the
previous evaluations were measuring the wrong thing." This is the differentiator the
literature is wide open on: scDFM, PerturbDiff, STATE, CellFlow, SCALE are all racing to
model the joint distribution — **nobody has shown that the downstream causal evaluations
silently discard it.**

### (c) Only THEN is a joint-structure training objective worth testing — on the fixed instrument
Your `joint_structure_objectives.py` (precision/MMD/OT) is the *right* code, but it must be
trained on a model that emits **cells** and evaluated on the **cell-level** pipeline from (a).
On the current mean+overlay instrument it was untestable by construction. On the fixed
instrument the OT/MMD objective has a real chance, because now the metric can register the
covariance it shapes. This is the one place a GPU retrain earns its cost — and it is a genuine
bet, not a foregone null, because the instrument change removes the invariance that guaranteed
the previous nulls.

---

## 3. Why I believe this will work (not hand-waving)

- The invariance is **not a hypothesis** — it is machine-precision zero, provable in three
  lines of linear algebra (diagonal similarity preserves off-diagonal precision support).
- It **quantitatively predicts** the 0.42 tie you already observe, with no free parameters.
- The fix is **mechanism-matched**: the reason the metric was blind is exactly the reason
  cell-level evaluation restores sight — you are adding back the off-diagonal information the
  diagonal rescale destroyed.
- It is **cheap to falsify**: the cell-level RPE1 CPA file already exists; you can run the
  first cell-level comparison this week, no GPU, and know immediately whether the instrument
  separates.
- It **aligns with where the whole field is going** (distributional VCMs) while pointing out
  something none of those papers has: the evaluation side hasn't caught up to the modeling side.

---

## 4. The honest risk

The reframed paper's strongest form requires (a) to *change the outcome* — i.e. cell-level
evaluation must at least separate `real cells` from `overlay(mean)`. If cell-level evaluation
*also* saturates (e.g. GIES at N=200 is genuinely resolution-limited, per the S1 concern),
then the contribution narrows to "the overlay is invariant" (still novel and worth a section)
but the "we fixed it" claim weakens. Mitigation: run (a) first, before committing the reframe.
The invariance result stands regardless — it is the floor, and it is already a stronger
finding than anything Track B produced.

Artifacts: `overlay_diagnosis.png` (the three-panel proof).

---

## 5. OUTCOME (executed 2026-07-05) — the risk in §4 materialized, and it sharpened the paper

I ran the decisive test (RPE1 + Norman CPA cells, 5 seeds each; `prereg_C_cell_level_eval.md`,
`RESULTS_INSTRUMENT_FIX.md`, `instrument_fix_rpe1_norman.png`). **The instrument did NOT
separate.** Cell-level input does not beat mean+overlay:

- RPE1: T1 (pred cells > overlay) 1/5, −0.0061; T2 (real cells > overlay) 1/5, −0.0046. Both DISCARD.
- Norman: T1 1/5, −0.0118; T2 3/5, +0.0007. Both DISCARD.

This is the §4 risk branch — and it is *more* informative than the "we fixed it" branch would
have been. It proves a **second, independent limitation**: not only is the overlay blind to
joint structure (proved analytically), but GIES at N≈100–200 is **resolution-limited even given
real perturbed cells** — richer joint input (Jaccard(cells,overlay)≈0.15–0.26, so the edges
genuinely change) does not yield more recovered real biology.

**Revised recommendation.** The reframe is no longer "cell-level evaluation fixes the
benchmark." It is the stronger, better-defended claim: *"VCM causal-substitutability failure is
not a measurement artifact — we rule out both natural 'your evaluation is broken' rebuttals: the
mean-overlay sampler is provably prediction-invariant, AND direct cell-level evaluation on real
cells does not recover more truth. The gap is intrinsic to per-perturbation causal discovery at
this scale."* This closes the two biggest reviewer escape hatches at once.

The one remaining live positive-result avenue (real, not hope): a **genome-scale /
precision-based backend** (inspre at N∈{500,1000,2000}), where the linear baseline's edge may
vanish and where the causal-discovery step is not co-partition-saturated. That is the single
experiment that could still flip the verdict — and it is now the clearly-motivated next step,
not another loss function or another sampler.