# Feasibility — the single cheapest decisive experiment (Phase C)

Per the workflow: before any infrastructure, the one smallest experiment that proves or kills the
core open question. There is exactly one open change with an uncertain outcome (the lemma is a proof;
the reframe is writing; Tracks B/5 are killed by Wall 1). That change is:

## The A3 mechanism discriminator

**The question.** A3 is the project's only positive: on held-out Norman doubles, real doubles beat
additive-of-singles on **5/5 seeds** (anchored-GT F1 **0.0552 ± 0.0056 vs 0.0359 ± 0.0037**, gain
**+0.0193 ± 0.0058**). The handoff proposes to headline A3 as "when DO VCMs help." Before that claim
is made, we must know **why real wins** — because of Wall 1, both predictors reach the metric only
through per-gene means, so the win is a *marginal-mean* phenomenon by construction. The only
scientifically defensible interpretation of a marginal-mean win is **data non-additivity (epistasis)**:
the real double's per-gene response to the (A,B) perturbation is not the sum of the single responses,
so an additive-of-singles predictor structurally cannot reproduce it. The discriminator tests whether
the win actually comes from that, or from ordinary mean/noise differences that would not support an
epistasis claim.

**Design (revised after the Phase-B council).** The decisive control is a **mean-matched null**, not
an edge decomposition. The subtlety the council surfaced: in the current A3 code, `real_doubles` are
fed to discovery as real **cells** (they keep their covariance — the `output_type=="cells"` path
bypasses the overlay), while `additive_singles` is a **mean** pushed through the overlay's diagonal
rescale. So real's win over additive currently **conflates two things**: (i) real cells' joint
structure, and (ii) real doubles' per-gene means differing from additive-of-singles means. We must
separate them. Reuse `scripts/a3_combinations.py` machinery, `.venv_gears`. Score three conditions
against the same anchored GT and the GIES-on-real secondary GT:

- **A. real doubles (cells)** — real covariance + real means. (the existing 0.055)
- **B. real-doubles-means-through-overlay** — take the real doubles' per-gene mean vector and inject it
  through the SAME `sample_perturbed_cells_overlay` onto control cells. Joint structure **destroyed**;
  real non-additive **means preserved**. (the new mean-matched null)
- **C. additive-of-singles-means-through-overlay** — the existing additive baseline (0.036).

The contrast B vs C isolates the marginal-mean channel (both go through the identical overlay, so any
difference is purely the mean vector). The contrast A vs B isolates whatever the cell-path reads beyond
means.

**Exact numbers.**
- Dataset/split: Norman 2019 held-out doubles, 71-gene panel, 129 doubles (both parents measured with
  ≥20 single cells), same as locked A3. Seeds 42/123/456/789/1024. Anchored GT primary (q=0.05,
  τ_fc=0.5); GIES-on-real secondary.
- **CONFIRM — A3 is a defensible marginal-mean-non-additivity result:** condition **B beats C**
  (real means-through-overlay > additive means-through-overlay) on ≥4/5 seeds, gain > run-to-run SD.
  This means real's advantage lives in the non-additive **means** — a claim about the *data* (real
  double means are not the sum of single means), aligned with Norman 2019, and honestly *within* Wall 1
  (the metric sees means, real's means are simply non-additive). A3 becomes: "an additive predictor
  systematically mis-predicts double-perturbation means, and that mis-prediction is visible even
  through a joint-structure-blind causal evaluation."
- **KILL — A3 is not a mean effect:** B ≈ C (≤ run-to-run SD). Then real's original advantage (A vs C)
  came from the **cell-path joint structure**, not means. That would be surprising given the
  instrument-fix null and would itself need investigation before any claim; A3 is removed from the
  positive column pending that.
- **Secondary read (edge color, non-gating):** among anchored edges `(A,B)→C`, fraction that are
  "epistasis-only" (real crosses threshold, additive-of-singles mean does not) — reported as supporting
  detail for a CONFIRM, not as the gate.

**Compute cost.** CPU only, `.venv_gears`, no GPU. The anchored-GT construction and additive means are
already computed inside `a3_combinations.py`; the discriminator is a re-scoring/decomposition pass over
existing per-gene mean shifts — minutes, well inside local resources. (192 GB VRAM ceiling from the CV
workflow is not binding; there is no training here.)

**Why this is decisive.** It is the one experiment whose outcome changes what we can honestly claim. A
CONFIRM lets the reframe present A3 as evidence of a real biological non-additivity regime (aligned with
Norman 2019's genetic-interaction map) — while being explicit that it is a claim about the *data*, not
about VCM joint structure (Wall 1 forbids the latter). A KILL removes A3 from the positive column and
makes the paper a cleaner all-negative result. Either way the manuscript claim is corrected before it
is written — which is the entire point of running feasibility first.
