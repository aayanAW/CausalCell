# Phase B — Mechanism + Novelty Kill Gate

**How this gate is run.** Per the user's standing instruction, the Phase-B gate operates here as a
**mechanism filter first, idea-novelty second**: a change passes only if there is a causal mechanism
by which it can move (or correctly interpret) the reported result. This is a hardening pass on an
already-accepted paper (ICML 2026 FAGEN), so "novelty ≥ 7 or die" is applied to the *contribution of
each change*, not to a brand-new seed.

**Tool honesty.** The `council` skill and Codex (GPT-5.5 cross-model) named in the workflow are not
available in this environment. Fallbacks actually used: (a) literature retrieval via `web_search` +
OpenAlex/CrossRef; (b) a 3-voice adversarial council via independent-persona `host.llm` passes
(methodologist / ML-area-chair / perturbation biologist). This is a **Claude-only** adversarial pass,
NOT a true second-model-family check — stated plainly so the gate is not over-credited.

## Step 1 — Novelty verification (literature)

Two of the changes are the paper's actual candidate contributions; scored honestly:

| Contribution | Nearest prior art | Novelty of the change | Score |
|---|---|---|---|
| **Wall-1 invariance lemma** (mean-overlay causal eval is blind to a prediction's joint structure) | Precision-support-based causal discovery is textbook (Loh–Bühlmann-type results; inverse-covariance / sparsest-permutation discovery). The linear algebra (positive-diagonal congruence preserves precision support) is standard. | The *math* is not novel. The **application is**: identifying that a mean-overlay evaluation of perturbation-prediction models is a diagonal rescale, and therefore provably cannot reward the joint structure such models are trained to produce. No retrieved work states this about perturbation-prediction benchmarks. | **7** (as a benchmark-methodology result, not a math theorem) |
| **A3 non-additivity regime** | Norman 2019 double-perturbation genetic-interaction map; recent single-cell models explicitly use Norman doubles to probe **non-additive** genetic interactions. Non-additivity of Norman doubles is established biology. | Re-deriving that Norman doubles are non-additive is **not novel**. The only possibly-novel wrapper is "a causal-recovery regime where an additive baseline provably underperforms" — and even that is at risk of being Wall-1 restated (see gate). | **4** as a standalone claim → must be reframed as marginal-mean non-additivity, not a VCM-joint-structure result |

**Where the paper's real novelty lives** (unchanged from prior assessment): the *causal-substitutability
question* + the **combined honest characterization** of two distinct walls (overlay-blindness AND
task-under-determination). Neither retrieved benchmark (CausalBench-family, network-inference
benchmarks, the 2026 "when does GRN inference break" diagnostic) makes the diagonal-rescale
invariance argument or reports real-data recovery sinking into the permutation band at genome scale.

**Nearest scoop:** a 2026 diagnostic study showing causal GRN methods underperform correlational
baselines under confounding, echoing the "prediction ≠ causal recovery" framing. **What survives it:**
that work diagnoses discovery *methods*; it does not prove the *evaluation channel* is mean-only
invariant, nor does it run the non-circular anchored-GT + genome-scale under-determination axes. The
Wall-1 lemma + Wall-2 empirical package is not scooped.

## Step 2 — Multi-perspective adversarial audit (council fallback)

Three independent adversarial voices (full text in `handoff/council.json`). Convergent, load-bearing
findings — all three flagged that verdicts were being **over-stated**, in different ways:

1. **Lemma is over-claimed as "exactly invariant."** The overlay adds Poisson noise *after* the
   rescale; Poisson variance is mean-dependent, so a diagonal rescale D perturbs off-diagonal
   correlations gene-by-gene and vanishes only as counts→∞. The clip at [0.01,10] can also cluster
   ratios and inject spurious cross-gene collinearity. → **Lemma restated:** exact for the
   deterministic rescale; approximate (with an empirically-bounded, count-dependent residual) once
   Poisson noise is added. Verification must include an **edge-flip sensitivity sweep at realistic
   UMI depths**, not just a symbolic identity.
2. **"KILL Track-B by proof" over-reaches.** Wall-1 is a property of *this overlay harness*, not of
   causal recovery in general. → **Verdicts #4/#5 softened:** the primary kill reason is the existing
   **5-seed empirical DISCARD**; Wall-1 is the *mechanistic explanation, scoped to the overlay eval*.
   Do not present the proof as if it forbids all conceivable evaluations.
3. **A3's decisive control is a mean-matched null.** `real_doubles` are fed as real *cells* (keeping
   covariance); `additive_singles` is a *mean* through the overlay. So real's win conflates joint
   structure with mean differences. → **A3 redesigned:** add condition **real-doubles-means-through-
   overlay** (joint structure destroyed, means preserved). If it reproduces ~0.055, the advantage is
   marginal-mean non-additivity (honest, data-level claim). If it collapses toward additive's 0.036,
   the advantage was joint structure the cell-path reads — a separate, surprising finding.

## Verdict per change (the gate)

| # | Change | Gate verdict | Reason |
|---|--------|-------------|--------|
| 1 | Wall-1 lemma | **PASS → BUILD** (restated) | Novel as benchmark methodology (7); must state exact-deterministic + asymptotic-under-noise and verify edge-flips empirically |
| 2 | A3 mechanism discriminator | **PASS → BUILD as feasibility gate** | Not a contribution yet; the mean-matched-null decides whether A3 is a defensible marginal-mean-non-additivity result or Wall-1 restated |
| 3 | Manuscript reframe | **PASS → BUILD** | Corrects claims to match evidence; Wall-1 proven, Wall-2 empirical |
| 4 | Track-B retrain | **KILL** | 5-seed empirical DISCARD (primary); Wall-1 explains why, scoped to overlay eval. No mechanism to move the metric. |
| 5 | Causal architecture | **KILL under current eval** | Collapses to a per-gene mean at overlay; the only bypass (instrument-fix on real cells) is already null. Revivable only with a joint-structure-reading eval that does not yet exist. |
| 6 | Wall-2 inspre-ADMM confirmation | **DEFER** | Confirmation of a negative; needs R + free CPU; off critical path |

**Realistic oral odds:** the reframed methodology-first paper clears a journal-workshop / mid-tier
venue bar (~7.5); a top-venue oral is not in reach on a negative-result benchmark even hardened. The
lemma + two-wall framing is the defensible core; A3's fate is decided by the feasibility experiment.

**Gate outcome:** proceed to Phase D with changes 1, 2, 3 (2 as the feasibility gate); 4 and 5 killed
on record; 6 deferred.
