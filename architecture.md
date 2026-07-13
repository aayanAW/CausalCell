# Architecture — PerturbCausal hardening (Phase C)

**Scope.** This is a benchmark/analysis project, not a model-training project, so "architecture"
here means *the design space of candidate changes* to the paper's evidence, each scored by a single
hard criterion: **does it have a mechanism to change the reported result?** The reported result is a
causal-recovery F1 produced by a fixed discovery algorithm (GIES/inspre/PC) reading data whose only
model-dependent content is a per-gene mean vector.

## The master filter (why most candidate "fixes" are dead on arrival)

The evaluation samples model-predicted perturbed cells via `sample_perturbed_cells_overlay`
(`scripts/run_eval_local.py`). Verified from source, the operative line is:

```python
cells *= ratios[np.newaxis, :]          # ratios = clip(mean_pred / ctrl_mean, 0.01, 10)
cells = rng.poisson(lam=cells)          # per-gene, independent
```

i.e. the prediction enters as a **diagonal (per-gene) rescale** `D = diag(r)` of bootstrapped
control cells, plus independent per-gene Poisson noise. A diagonal rescale sends the control
covariance `Σ → D Σ D`; the **correlation matrix and the support of the precision matrix `Σ⁻¹` are
exactly invariant** under `D` (a positive diagonal congruence). Any conditional-independence-based
discovery algorithm reads only that support. Therefore:

> **The causal metric depends on a model's prediction ONLY through its per-gene mean, and only up to
> the ratio clip [0.01, 10]. It is blind to the prediction's joint (covariance) structure.**

This is Wall 1. Every candidate below is scored against it first.

**Precise form (corrected after the Phase-B council — do not overclaim "exactly invariant"):**
the invariance is **exact for the deterministic rescale** `cells *= r` (correlation matrix unchanged
to machine precision; precision-support Jaccard = 1). The pipeline then adds **per-gene Poisson noise
whose variance equals the (rescaled) mean**, so the rescale does perturb off-diagonal sample
correlations gene-by-gene; this residual **vanishes as counts→∞** and must be bounded *empirically* at
realistic UMI depth (an edge-flip sweep), not asserted as zero. Two second-order channels are also
acknowledged: (i) the ratio clip at [0.01,10] can saturate many genes to the same boundary, and (ii)
Poisson shot noise. Both are quantified in the verification, not hand-waved. The lemma's claim is thus
"**mean-only up to a count-dependent, empirically-negligible noise residual**," which is the honest and
still-decisive statement.

## Candidate change set (from the handoff's open next-steps)

| # | Candidate | Mechanism it would need | Passes Wall-1 filter? | Design verdict |
|---|-----------|------------------------|----------------------|----------------|
| 1 | **Wall-1 invariance lemma** (formalize + verify) | It's a proof *about* the filter, not a change to the pipeline | N/A (it IS the filter) | **BUILD** — converts the strongest empirical finding into a stated result |
| 2 | **A3 mechanism discriminator** | Explain *why* real doubles beat additive-of-singles: epistasis (data non-additivity) vs marginal-mean artifact. Both act through means, so it is *scoreable* under Wall-1 | Yes — operates on the mean channel the metric can see | **BUILD** — cheapest decisive experiment (feasibility.md) |
| 3 | **Manuscript reframe** | Writing; fold lemma + honest Wall-2 status | N/A | **BUILD** (Phase-D writing step) |
| 4 | **Track-B retrain** (precision/MMD/OT joint-structure loss) | Would reshape joint structure of predictions | **NO** under this overlay eval | **KILL** — primary reason is the existing **5-seed empirical DISCARD**; Wall-1 is the *scoped mechanistic explanation* (why it discarded), NOT a claim that no eval could ever reward it |
| 5 | **Causally-structured predictor** (new architecture, CUDA-gated) | A better latent still collapses to a per-gene mean at overlay time | **NO** under current eval | **KILL under current eval**; only revivable if paired with an eval that bypasses overlay (the instrument-fix — already null, Wall 2) |

> **Scope note (from the Phase-B council).** Wall-1 is a property of the *overlay evaluation harness*,
> not of causal recovery in general. The honest kill statement for #4/#5 is therefore: "under the
> mean-overlay evaluation used by this benchmark (and the already-tested real-cell bypass), these
> directions have no mechanism to move the metric, and Track-B was empirically nulled at 5 seeds." We
> do not claim the proof forbids all conceivable evaluations.
| 6 | **Wall-2 inspre-ADMM confirmation** | Robustness of the "real data doesn't recover its own interventional graph above chance at scale" result to the full ADMM estimator (local screen used closed-form) | Yes (different estimator, same target) | **DEFER** — confirmation of a negative; needs R + free CPU; low leverage, not on the critical path |

## Two walls, kept distinct (this is the architectural spine of the reframe)

- **Wall 1 — overlay blindness.** The metric cannot see a prediction's joint structure (diagonal-rescale
  invariance). Kills candidates 4 and 5. *Provable, trivial, exact.* → Lemma (candidate 1).
- **Wall 2 — task under-determination.** Even feeding **real** cells (bypassing the overlay), causal
  discovery does not recover the anchored interventional graph above chance — on RPE1+Norman
  (instrument-fix null) and at genome scale N=500/1000 (real sinks to 0.33× chance). *Empirical,
  load-bearing, NOT yet a theorem.* → honest empirical status in the reframe; candidate 6 is its
  only open robustness check.

Conflating these two is the single biggest risk to the paper's honesty. The reframe must state Wall 1
as proven and Wall 2 as empirical.

## Winning design per active change

- **Candidate 1 (lemma):** state as a proposition over positive-diagonal congruence of a Gaussian
  covariance; corollary that GIES/PC/inspre CPDAG output is a function of `mean_pred` only through
  `sign/support`, invariant to any joint-structure reshaping. **Verify numerically against the ACTUAL
  `sample_perturbed_cells_overlay` function** (import it, don't reimplement): two very different mean
  predictions with identical ratio-support → identical correlation and precision support; quantify the
  Poisson-noise residual separately (it is diagonal, injects no cross-gene structure).
- **Candidate 2 (A3 discriminator):** see feasibility.md. Decompose anchored-GT edges into
  additive-predictable vs epistasis-only; test whether real's advantage concentrates on epistasis-only
  edges (→ genuine data non-additivity, a claim about biology) or on additive-predictable edges (→
  marginal-mean/noise artifact, and A3 must be reframed).

## Named metrics

Causal-recovery **F1 / precision / recall** of edge sets vs a declared ground truth (anchored
interventional GT primary; GIES-on-real secondary). Invariance verified by **max|Δ corr|** and
**precision-support Jaccard = 1.0**. All under the pre-registered multi-seed keep/discard gate
(seeds 42/123/456/789/1024; KEEP iff ≥4/5 and mean gain > run-to-run SD).
