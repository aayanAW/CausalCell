# Pre-registration — A3 combination perturbations (held-out Norman doubles)

**Date:** 2026-07-05 · **Base commit:** `fix/camera-ready-corrections` HEAD.
**FIT-GATE:** GREEN (frozen Norman 2019 K562 combinatorial atlas on disk; deterministic held-out split; cheap scalar causal-recovery eval).

## Question
On the **held-out combination perturbations** — where an additive/linear baseline provably cannot capture epistasis — does any predictor recover the causal structure of the unseen doubles better than the linear baseline? This is the marquee "when do VCMs help?" test: combinations are the one regime where a VCM has a structural reason to beat ElasticNet.

## Data + split (frozen)
`data/norman2019.h5ad`: **131 unique doubles (41,759 cells)**, 105 singles (57,831 cells), 11,855 control (`nperts`∈{0,1,2}; double labels `GENEA_GENEB`).
- **Held-out set = all 131 doubles.** Predictors see only singles + control at fit time.
- Gene panel: the measured-gene subset used by the existing N=200 pipeline (intersect with genes present in Norman); report the final panel size.

## Predictors
- **ElasticNet additive baseline** — predict double(A,B) response = combination of single-A and single-B fits (the explicit no-epistasis reference).
- **Any VCM double-prediction on disk** (`data/model_outputs_real_norman/`) — include if present.
- **Ceiling:** real held-out double cells (F1 = 1.0 by construction).
If no VCM predicts doubles on disk, the local result is ElasticNet-vs-real on the held-out doubles; the VCM double-prediction arm is explicitly labeled **GPU-gated** (requires retrain/inference) and deferred.

## Protocol
For each predictor, form predicted means for the 131 held-out doubles, run `sample_perturbed_cells_overlay → run_gies` (same params as A1), score recovered edges vs the **doubles' own real-data causal structure** (GIES-on-real-doubles as the within-regime reference) AND, as a secondary target, the A1-style anchored edges restricted to the double genes. Absolute P/R/F1.
**Seeds:** 42, 123, 456, 789, 1024.

## Keep/discard (Guard 1 — multi-seed)
A predictor "helps on combinations" only if it beats the ElasticNet additive baseline on held-out-double F1 across **≥3 of 5 seeds** AND mean improvement > run-to-run SD. Otherwise: the linear additive baseline is not beaten even where it structurally should be — reported straight.

## Reporting
`results/hardening/a3_combinations.json` (per-seed, all predictors), figure `a3_combination_recovery.png` (per-seed strip + mean by predictor). Panel size, held-out count, and any GPU-gated arm stated explicitly.
