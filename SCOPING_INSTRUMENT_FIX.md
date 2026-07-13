# Scoping — the instrument-fix execution

## The one-line thesis being tested

The causal-recovery benchmark scores predictions through a mean+overlay sampler that is
mathematically blind to joint structure (proved: F1 invariant to the prediction up to
max |Δ|=1e-15). The fix is to feed models' **predicted cells** to the causal algorithm
directly. This scoping doc costs the experiment before running it.

## What is local / no-GPU (the decisive core — steps 1–4, 7)

| item | status |
|------|--------|
| RPE1 CPA predicted **cells** | EXISTS `data/model_outputs_real_rpe1/cpa_predictions.h5ad` (20k×8749, output_type=cells, 200 perts, 196/196 panel genes) |
| Norman CPA predicted **cells** | EXISTS `data/model_outputs_real_norman/cpa_predictions.h5ad` (10.2k×8028) |
| RPE1 real cells + panel | EXISTS `data/rpe1_subset_n200_slim.h5ad` (29,695 cells), panel from `phase2_eval_rpe1.json` (196 genes, part=20, ncells=100) |
| harness cells→GIES path | EXISTS `run_eval_local.py:718` (`output_type=="cells"` routes cells straight to GIES, bypassing overlay) |
| eval env | `.venv_gears` (Py3.9, gies/sklearn/torch) |

**The decisive test — C(predicted cells) vs D(overlay(predicted means)), and A(real cells) vs
B(overlay(real means)) — runs entirely locally on existing data.** No new model training is
required to answer the central question.

## What is GPU-gated and contingent (steps 5–6)

Only reached if the local RPE1 test shows cell-level separates (T1 pass). Both use the
`gpu-box-callcommand-relay` protocol on ssh:research-box (namespaced workdir, VRAM check,
base64-chunk transfer, detached docker exec). If T1 fails locally, these are skipped and the
paper's contribution is the invariance finding + the resolution-limit finding — recorded, not
forced.

## Failure modes and honesty guards

- **The circular GT scores A≈1.0 by construction** — so A-vs-B and C-vs-D are scored on the
  circular GT for B/C/D and on the non-circular anchored GT for all four. The gate verdicts
  use the paired within-seed differences, not absolute levels.
- **Do not rescue a null.** If C≈D and A≈B (no separation), that is reported straight: the
  overlay-invariance finding stands and GIES is additionally resolution-limited. This is the
  same multi-seed discipline that correctly nulled J1/J2/B/B2; it applies here too.
- **Partition ceiling** (A1's ~9.6% co-partition recall) still caps absolute anchored F1; the
  comparative (paired) test is what carries the claim.

## Deliverables

Preregs (`prereg_C_cell_level_eval.md`, this doc); `c_cell_level_eval.py`; per-dataset result
JSONs + figures (RPE1, Norman; K562 if reached); `RESULTS_INSTRUMENT_FIX.md`; updated
`RECOMMENDATION_the_real_direction.md` and handoff.