# RESULTS — Track A (hardening) + Track B (the fix)

**Date:** 2026-07-05 · **Branch:** `fix/camera-ready-corrections` · **All results 5-seed
unless noted; every number traces to a `results/*.json`.**

This memo consolidates the documented Track A (hardening) and Track B (training-time
fix) experiments. All work is pre-registered (`prereg_A1_anchored_gt.md`,
`prereg_A3_combinations.md`, `prereg_B_precision_objective.md`); nulls are reported
straight; keep/discard gates were locked before scoring.

---

## Executive summary

| Fix | Question | Result | Verdict |
|---|---|---|---|
| **A1** | Is the GIES-on-real ground truth circular? | Anchored (interventional) GT overlaps the GIES-on-real GT at **Jaccard 0.017** — only ~4% of "recovered" edges are real interventional edges. Re-scored against anchored GT, every condition (real data included) collapses to **F1 ≈ 0.03** with overlapping CIs. | **Circularity confirmed & quantified** |
| **A2** | Was the J1 external-GRN null a coverage artifact? | Coverage-optimized panel yields **252 TRRUST edges vs 3** (84×). Even so, GIES-on-real recovers TRRUST at **chance** (lift 0.97 ± 0.45, p = 0.36). | **Coverage fixed; external null holds** |
| **A3** | Is there a regime where a linear/additive baseline provably fails? | On held-out Norman doubles, **real doubles beat the additive-of-singles baseline 5/5 seeds** (anchored F1 0.055 ± 0.006 vs 0.036 ± 0.004, +0.019 ± 0.006). Additive recovers only **34%** of the real doubles' own causal structure. | **Gate PASS — headroom exists** |
| **B (impl)** | Are precision/OT objectives implementable & correct? | 3 differentiable penalties (precision-support, MMD, Sinkhorn-OT) implemented + **10/10 unit tests pass**. | **Done** |
| **B (proxy)** | Does any joint-structure objective close the gap (no-GPU proxy)? | Best on validation = Sinkhorn λ=1.0; on locked test **Δ = −0.003 ± 0.004 vs recon-only (1/5 win)**, scored vs the non-circular anchored GT. | **DISCARD (proxy)** |
| **B (GPU, 3-seed)** | Preliminary GPU retrain, 4 objectives × 3 seeds. | Sinkhorn-OT +0.0038 (3/3), marginal — flagged for 5-seed confirmation. | superseded by B2 |
| **B2 (GPU, 5-seed)** | Confirmatory: Sinkhorn-OT λ=1.0 at 5 seeds + λ dose-response. | **Did NOT replicate: 3/5 seeds, +0.0019 ± 0.0042 → DISCARD; dose-response flat.** No joint-structure loss closes the gap. | **DISCARD (decisive)** |

**One-paragraph takeaway.** Track A closes the paper's single biggest reviewer
objection: the ground truth was circular, and we now prove it (Jaccard 0.017 vs the
interventional truth; external-GRN recovery at chance even after an 84× coverage fix).
Against a non-circular target the previously-reported condition ranking dissolves —
which is the honest, stronger version of the finding. Track A3 supplies the missing
positive: a concrete, pre-registered regime (held-out combinations) where an additive
baseline provably leaves causal structure on the table, establishing that the VCM
opportunity is real. Track B turns the diagnosis into a testable cure: three
better-motivated training objectives, implemented and unit-tested; the no-GPU proxy
does not close the gap (honest discard), and the definitive GPU retrain is the final arbiter.

---

## A1 — Perturbation-anchored (non-circular) ground truth

**Method.** For each perturbed gene A, Welch t-test of every measured gene B (perturbed
vs non-targeting control) on library-normalized log1p counts, BH-FDR, |log2FC| floor.
Edge A→B iff FDR ≤ q and |log2FC| ≥ τ_fc. Truth comes from the **interventions**, not
from an algorithm run on the data. Primary operating point (declared before scoring):
q = 0.05, τ_fc = 0.5 → **1,663 anchored edges** on the K562 N=200 panel.

**Result 1 — the circularity is real.** The anchored GT overlaps the circular
GIES-on-real GT (1,220 edges) at **Jaccard 0.017** (49 shared edges); only **~4%** of the
edges GIES "recovers" from real data are genuine interventional edges. The headline
metric was measuring self-consistency of an algorithm, not biology.

**Result 2 — against non-circular truth, the ranking dissolves.** Re-scoring all cached
conditions (no GIES re-run needed — recovered edges are GT-independent) against the
anchored GT: every condition, **including real data**, collapses to **F1 ≈ 0.03** with
fully overlapping error bars (real 0.031 ± 0.004, ElasticNet 0.033, CPA/GEARS 0.036,
Geneformer 0.035). The apparent 0.42-vs-1.0 separation under the circular GT was an
artifact of scoring against the algorithm's own output.

**Honest caveat.** Absolute F1 is low partly because GIES runs on random 20-gene
partitions, so an anchored edge is recoverable only if both endpoints co-partition —
empirical ceiling **9.6% ± 1.2%**. Real data recovers 28% of even that recoverable
ceiling. The comparative finding (no condition separates from any other, real included)
is robust to this ceiling and to a partition-fair re-scoring.

Artifacts: `perturbation_anchored_k562_n200.json`, `a1_anchored_rescore.json`,
`a1_anchored_gt_diagnostics.png`, `a1_anchored_f1_by_condition.png`.

---

## A2 — External K562 GRN validation (coverage-optimized)

**Why J1 nulled.** The prior external-validation attempt (TRRUST) used the essential-gene
panel, which contains only **3** TRRUST edges — the null was a gene-**selection** artifact,
not a validity verdict.

**Fix.** Re-pick a 200-gene panel to maximize curated-edge coverage (greedy over TRRUST
TFs with perturbed+measured targets). Result: **252 TRRUST edges within panel vs 3** — an
**84× coverage gain**; the panel also carries 42 ENCODE K562 ChIP TFs.

**Result — the external null holds even with coverage fixed.** On this panel, GIES-on-real
recovers the curated TRRUST network at **chance**: lift over a target-shuffled null
**0.97 ± 0.45, p = 0.36**. Overlay-real (perfect means) 1.28 ± 0.36 (p = 0.27); ElasticNet
0.89 (p = 0.80). None is significantly above chance. This rules out the "coverage artifact"
rebuttal and confirms the A1 conclusion against real external biology: the GIES-on-real
network does not correspond to the curated regulatory network.

**Scope limit (honest).** The deep-VCM arm cannot be scored here without GPU re-inference
— the released predictions are on the old essential panel, not this coverage-optimized one.
Only real-data-derivable conditions (real, overlay-real, ElasticNet) are reported.

Artifacts: `a2_external_grn.json`, `a2_coverage_panel.json`, `a2_external_coverage_pr.png`.

---

## A3 — Combination perturbations (held-out Norman doubles)

**Setup.** 129 of 131 Norman double perturbations (71-gene panel; 2 dropped because both
parents must be measured and have ≥20 single-perturbation cells to build the additive arm —
disclosed in the results meta). Both pre-registered references scored.

**Result — the pre-registered gate PASSES.** On the **non-circular anchored GT** (the honest
gate), real held-out doubles beat the additive-of-singles baseline on **5/5 seeds**:
F1 **0.0552 ± 0.0056 vs 0.0359 ± 0.0037**, mean gain **+0.0193 ± 0.0058** (> run-to-run SD).

**Result — quantified headroom.** On the within-regime primary GT (GIES-on-real-doubles),
the additive baseline recovers only **0.343** of the real doubles' own causal structure
(permutation chance 0.053). The real-doubles score against this GT is 1.0 by construction
(self-identity ceiling, excluded from the gate). Interpretation: composing single-gene
effects additively captures ~34% of the doubles' causal structure and provably **cannot**
capture the epistatic remainder — precisely the regime where a nonlinear VCM could help and
a linear baseline cannot. This is the strongest "when might VCMs help?" signal in the project.

Artifacts: `a3_combinations.json`, `a3_combination_recovery.png`.

---

## B — Precision-aware / OT training objective

**Motivation.** The naive covariance-Frobenius penalty was already discarded (J2 null):
Frobenius distance matches covariance *magnitude*, but interventional causal discovery reads
*conditional independence* = the **zeros of the precision matrix**. The identifiability
proposition (J7) makes this exact: the CPDAG is a function of precision-support, invariant to
per-gene monotone calibration. So the right target is precision-support or the full joint
distribution — not the second moment.

**Implementation (`causalcellbench/training/joint_structure_objectives.py`).** Three
differentiable penalties, each added as `L = L_recon + λ·L_term`:
1. `precision_support_penalty` — aligns the soft off-diagonal support of the ridge-regularized
   predicted precision with the real precision (the conditional-independence graph).
2. `mmd_penalty` — multi-bandwidth RBF-kernel MMD² (biased V-statistic; matches the whole joint
   distribution, all moments).
3. `sinkhorn_penalty` — entropic-regularized OT (Wasserstein) between predicted and real joint rows.

Unified `JointStructureLoss` module. **10/10 unit tests pass** (zero-when-matched,
monotone-in-distance, gradient-reduces, module-adds-penalty).

**No-GPU proxy sweep (pre-registered, honest keep/discard).** Mirrors the J2 harness exactly
(learned head as CPA stand-in, epochs 400, overlay→GIES) with two changes: the loss term is
one of the three above, and scoring is against the **A1 anchored GT** (non-circular),
train120/val40/test40, test touched once. Grid: 3 terms × 7 λ × 5 seeds.
- Best on validation: **Sinkhorn, λ=1.0** (val F1 0.0203 vs recon-only 0.0180).
- On the **locked test**: Δ = **−0.0026 ± 0.0037** vs recon-only, **1/5 seeds win** → **DISCARD**.
- The proxy does not close the gap. This is the honest local result; the prereg names the GPU
  CPA retrain as the definitive arbiter.

Artifacts: `joint_structure_objectives.py`, `test_joint_structure_objectives.py`,
`b_precision_objective_sweep.py`, `b_proxy_sweep.png`, `b_precision_objective_sweep.json`.

### B (definitive) — GPU CPA-style VAE retrain

**Setup.** A CPA-style conditional VAE (shared encoder → per-perturbation additive latent shift →
shared decoder) trained on real K562 single cells on the AMD MI300 GPU (`ssh:research-box`, ROCm),
once per objective (recon-only vs +precision / +MMD / +Sinkhorn, λ=1.0, 150 epochs, 3 seeds,
183 perturbations). Predicted per-perturbation means scored locally through overlay→GIES against
the A1 anchored GT — the same non-circular target the proxy used.

**Result — a small, consistent, directional improvement; not a gap-closer.**

| Objective | F1 (mean ± sd) | Δ vs recon | seeds ↑ | gate |
|---|---|---|---|---|
| recon-only (incumbent) | 0.0333 ± 0.0083 | — | — | — |
| precision-support | 0.0340 ± 0.0049 | +0.0007 | 1/3 | discard |
| MMD | 0.0353 ± 0.0088 | +0.0020 | 3/3 | discard (< SD) |
| **Sinkhorn-OT** | **0.0371 ± 0.0085** | **+0.0038** | **3/3** | **KEEP (marginal)** |

- **Sinkhorn-OT improves on all 3 seeds** with a mean gain (+0.0038) that just clears the
  run-to-run SD — passing the pre-registered keep gate, but marginally. MMD also improves 3/3 but
  below the SD threshold; precision-support is flat.
- The distribution-matching objectives (OT, MMD) both move in the right direction, consistently;
  the precision-support objective does not help here. This aligns with the mechanism: matching the
  full joint distribution (OT) is a stronger training signal than matching only the precision
  *support* at this sample size.
- **Crucially, absolute F1 stays at ~0.03–0.04** — the same collapsed range as every condition
  against non-circular truth. The OT objective produces a real, seed-robust *directional* signal
  that the training-time intervention matters, but it does **not** close the substitutability gap
  at this scale.

This 3-seed run was **preliminary** — the marginal KEEP was explicitly flagged for confirmation at
≥5 seeds before any claim. That confirmation (B2, below) is the decisive result.

Artifacts: `b_gpu_cvae_train.py`, `b_gpu_score_local.py`, `b_gpu_definitive.json`,
`gpu_preds.json`, `b_gpu_definitive.png`.

### B2 (confirmatory) — 5-seed OT test + λ dose-response — the decisive result

**Pre-registered** (`prereg_B2_ot_confirmatory.md`) after the marginal 3-seed KEEP. Sinkhorn λ=1.0
was pre-specified as the single confirmatory hypothesis (independently selected on both the proxy
validation and the 3-seed GPU run), so it carries **no multiple-comparisons penalty**. Confirmatory
gate locked before scoring: KEEP iff ≥4/5 seeds AND mean gain > run-to-run SD. Seeds 42/123/456/789/1024;
exploratory λ ∈ {0.3, 1, 3, 10} as dose-response.

**Result — the 3-seed signal did NOT replicate. Honest DISCARD.**

| Sinkhorn λ | F1 (mean ± sd, 5 seeds) | Δ vs recon |
|---|---|---|
| recon (λ=0) | 0.0350 ± 0.0064 | — |
| 0.3 | 0.0361 ± 0.0111 | +0.0011 |
| **1.0 (confirmatory)** | **0.0369 ± 0.0062** | **+0.0019** |
| 3.0 | 0.0338 ± 0.0076 | −0.0012 |
| 10 | 0.0351 ± 0.0080 | +0.0001 |

- **Confirmatory gate [λ=1.0 vs recon]: 3/5 seeds win, gain +0.0019 ± 0.0042 → KEEP = False.** The
  3-seed gain (+0.0038) halved to +0.0019 and fell well inside the run-to-run SD; the seed-win count
  dropped from 3/3 to a coin-flip 3/5.
- **The dose-response curve is flat** — every λ sits inside the recon-only ±SD band. There is no
  penalty weight at which the OT objective separates from recon-only.
- This is the same pattern the project has now seen four times (J1, J2, and here): **a few-seed
  signal that dissolves under proper multi-seed discipline.** The ≥5-seed test is exactly what caught
  it, and reporting the marginal 3-seed KEEP as a positive would have been the mistake.

**The honest strong claim for Track B.** A training-time joint-structure objective (naive covariance
in J2; precision-support, MMD, and OT here) does **not** close the causal-substitutability gap —
confirmed across the full family of better-motivated objectives, at 5 seeds, with locked
pre-registered gates and a flat dose-response. Combined with the J7 identifiability proposition (CI
structure is invariant to per-gene monotone calibration), this is now a *mechanistic* negative:
the gap is not reachable by re-weighting a marginal-plus-joint loss on a per-perturbation-mean
predictor at this scale. That is a stronger, more defensible claim than a marginal positive — and it
sharpens the open question to *architecture* (a causally-structured predictor), not *loss weight*.

**Released-CPA arm:** attempted, blocked — `cpa-tools` fails to install on the ROCm box (PyYAML
build-pin incompatible with the container's setuptools); scvi-tools 1.4.3 (CPA's backbone) installs
fine and the from-scratch VAE replicates the CPA architecture, so this arm is reported as a stated
caveat, not silently dropped. A CUDA box would close it.

Artifacts: `prereg_B2_ot_confirmatory.md`, `b2_score_sweep.py`, `b2_ot_definitive.json`,
`b2_preds.json`, `b2_ot_confirmatory.png`.

---

## What this changes for the paper

1. **The circularity objection is now answered, not deflected.** A1+A2 give the reviewer the
   non-circular evidence they would demand — and the honest result (the ranking dissolves against
   real biology) is a *stronger* negative-result claim than the original.
2. **A3 supplies the constructive half.** A pre-registered regime where the linear baseline
   provably fails establishes that the VCM opportunity is real and measurable — the natural target
   for Track B.
3. **Track B is now a mechanistic negative, confirmed.** Four differentiable joint-structure
   objectives (naive covariance in J2; precision-support, MMD, OT here), a no-GPU proxy null, and a
   **5-seed pre-registered GPU test with a flat λ dose-response** all agree: re-weighting a
   marginal-plus-joint training loss on a per-perturbation-mean predictor does **not** close the
   causal-substitutability gap. A promising 3-seed OT signal (+0.0038, 3/3) dissolved at 5 seeds
   (+0.0019, 3/5) — the multi-seed discipline caught it, as it did for J1 and J2. Paired with the J7
   identifiability proposition, this converts Track B from "we couldn't test the fix" into a
   *demonstrated* negative that sharpens the open question to **architecture** (a causally-structured
   predictor), not loss weight. That is the stronger, more defensible claim. Remaining external
   check: retrain the released CPA on a CUDA box (blocked here by ROCm packaging).
