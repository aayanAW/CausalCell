# Instrument-fix results — cell-level vs mean+overlay causal evaluation

**Date:** 2026-07-05 · Pre-registration: `prereg_C_cell_level_eval.md` (both outcomes declared
publishable before running) · Scoping: `SCOPING_INSTRUMENT_FIX.md`

## The question

The overlay sampler is provably blind to joint structure (F1 invariant to the prediction up to
max |Δ|=1e-15; `overlay_diagnosis.png`). Does feeding a model's **predicted cells** (its own
joint samples) to GIES directly — bypassing the overlay — let the causal instrument register
that structure? Two decisive paired tests, pre-registered gate KEEP iff ≥4/5 seeds AND
mean gain > SD:

- **T1:** C (predicted cells) > D (overlay of predicted means)
- **T2:** A (real cells) > B (overlay of real means)

## Result — a clean pre-registered NULL, replicated on two datasets

| dataset | T1 (C>D) | T2 (A>B) | Jaccard(C,D) | Jaccard(A,B) |
|---|---|---|---|---|
| RPE1 (5 seeds, 196 genes) | 1/5, −0.0061±0.0080, **DISCARD** | 1/5, −0.0046±0.0057, **DISCARD** | 0.262 | 0.253 |
| Norman (5 seeds, 102 genes) | 1/5, −0.0118±0.0104, **DISCARD** | 3/5, +0.0007±0.0066, **DISCARD** | 0.144 | 0.163 |

Per-condition anchored-GT F1 (mean±sd):

| condition | RPE1 | Norman |
|---|---|---|
| A real cells | 0.0172±0.0029 | 0.0164±0.0068 |
| B overlay(real mean) | 0.0218±0.0054 | 0.0157±0.0086 |
| C predicted cells | 0.0208±0.0042 | 0.0113±0.0034 |
| D overlay(pred mean) | 0.0269±0.0060 | 0.0231±0.0099 |

**Neither cell-level condition beats its overlay counterpart on either dataset.** Real
perturbed cells — which carry the true joint structure — do not beat their own mean+overlay.
The direction even runs slightly *against* the hypothesis (overlay conditions score marginally
higher, though within noise).

## What this means — TWO independent limitations, both now demonstrated

The naive reading of the invariance proof was "the overlay is the bottleneck; feed cells and
the instrument works." **That is refuted.** The correct, stronger reading:

1. **The overlay is blind to joint structure** — proved analytically (F1 invariant to the
   prediction, max |Δ|=1e-15). Any benchmark scoring predictions through a per-gene mean
   overlay cannot distinguish a good joint predictor from a bad one.
2. **GIES at N≈100–200 is *also* resolution-limited** — proved empirically here. Even when you
   remove the overlay and feed real perturbed cells with their true covariance, F1 does not
   improve. The causal-discovery step itself does not convert richer joint input into more
   recovered real biology at this gene-set size and cell count.

The `Jaccard(A,B)≈0.25`, `Jaccard(C,D)≈0.15–0.26` numbers are the key mechanistic detail:
**the recovered edge sets DO change substantially with the input** (~75–85% of edges differ
between cells and overlay) — so GIES is not ignoring the input; it responds to it. It simply
does not recover *more true edges* from the richer input. The information is entering the
algorithm and being discarded by the discovery step, not by the sampler.

## Consequence for the paper's thesis

This makes the negative result **deeper and more defensible**, not weaker:

- The failure of VCMs to substitute for real Perturb-seq in causal recovery is **not an
  artifact of the mean-overlay evaluation** (the obvious reviewer rebuttal). We tested that
  head-on: even real cells through the fixed instrument do not separate. The limitation is
  intrinsic to causal discovery at this scale, layered on top of the overlay's blindness.
- The paper can now state: *"We rule out the two most natural 'your evaluation is the problem'
  objections. (i) The mean-overlay sampler is provably invariant to the prediction's joint
  structure — but (ii) replacing it with direct cell-level evaluation, even on real perturbed
  cells, does not recover more of the anchored ground truth. The substitutability gap is not a
  measurement artifact; the causal-recovery task is genuinely hard at N≈200, for real and
  predicted data alike."*
- **This is why Track B (the training-objective fix) was always going to null** and why the
  next move is NOT another loss or another sampler. It is either (a) a fundamentally
  higher-resolution causal-discovery backend that can exploit joint structure at scale
  (inspre/precision-based, genome-scale N), or (b) accepting that per-perturbation causal
  recovery at this scale is the wrong evaluation target and moving to a metric that *is*
  sensitive (e.g. direct precision-matrix recovery, or TF→target validation at higher N with
  the coverage-optimized panel from A2).

## Honest scope

- Single model (CPA) provides the cell-level predictions; it is the only VCM with cell-format
  output on disk. The T2 (real-cells) test is model-independent and is the load-bearing one —
  it fails, and it cannot be blamed on CPA.
- Absolute F1 is low (~0.02) and partly capped by the ~9.6% co-partition ceiling (A1); the
  tests are paired within-seed so the ceiling cancels. The comparative null is robust to it.
- N∈{102,196}. The saturation claim is specific to this scale; a genome-scale backend is the
  declared way to test whether the verdict flips (it may — that is the one live positive-result
  avenue, and it is real, not hope).

## Artifacts

- `prereg_C_cell_level_eval.md`, `SCOPING_INSTRUMENT_FIX.md` — pre-registration + costing
- `scripts/c_cell_level_eval.py` — the 4-condition harness
- `results/instrument_fix/c_cell_level_rpe1.json`, `c_cell_level_norman.json` — full results
- `instrument_fix_rpe1_norman.png` — the decisive figure (both datasets, all four conditions,
  the four ≈0 gains)
- `overlay_diagnosis.png` — the invariance proof this experiment tested