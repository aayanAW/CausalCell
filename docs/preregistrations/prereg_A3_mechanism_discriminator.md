# Pre-registration — A3 mechanism discriminator (mean-matched null)

**Registered before scoring.** Locks the question, conditions, metric, and keep/discard thresholds.

## Question
A3's positive (real Norman doubles beat additive-of-singles, anchored F1 0.055 vs 0.036) currently
conflates two channels, because `real_doubles` enter causal discovery as real **cells** (covariance
preserved) while `additive_singles` enters as a **mean through the overlay** (diagonal rescale). Does
real's advantage come from (i) the non-additive **means** of real doubles, or (ii) **cell-level joint
structure** the overlay eval is supposed to be blind to?

## Conditions (all scored against identical GTs, same seeds)
- **A. real_doubles** — real double cells (existing; covariance + non-additive means).
- **B. real_means_overlay** — real doubles' per-gene mean vector pushed through the SAME
  `sample_perturbed_cells_overlay` onto control cells. Joint structure destroyed; real non-additive
  means preserved. (NEW — the mean-matched null.)
- **C. additive_singles** — additive-of-singles mean through the overlay (existing baseline).

## Ground truths
- PRIMARY (gate): anchored interventional GT from real doubles (A1 method, q=0.05, τ_fc=0.5).
  Non-circular, independent of any GIES output.
- SECONDARY (report only): GIES-on-real-doubles within-regime GT.

## Seeds
42, 123, 456, 789, 1024 (paired within-seed, as in A3).

## Pre-registered gate (locked)
Primary contrast = **B vs C** on the anchored GT:
- **CONFIRM** (A3 = marginal-mean non-additivity, honest & within Wall-1): B beats C on **≥4/5 seeds**
  AND mean(B−C) > SD(B−C). Interpretation: real's advantage is carried by non-additive means the
  overlay CAN see. A3 is reframed as "additive prediction mis-estimates double-perturbation means, and
  this is detectable even through a joint-structure-blind causal metric."
- **KILL** (A3 not a mean effect): B ≈ C (mean(B−C) ≤ SD, or <3/5 wins). Then real's original
  advantage (A vs C) came from the cell-path joint structure — surprising given the instrument-fix
  null; A3 leaves the positive column pending investigation.

Secondary contrast = **A vs B** (report, non-gating): isolates cell-path joint structure. Expected
A ≈ B if Wall-1/instrument-fix picture holds (means carry the signal, covariance adds little).

## Sanity assertions (the gate script's test suite)
1. All three conditions produce non-empty edge sets on ≥1 seed (harness wired correctly).
2. Condition C's anchored F1 reproduces the locked A3 additive number (0.036 ± 0.004) within 2 SD
   — proves the reused machinery is unchanged.
3. Condition A's anchored F1 reproduces the locked A3 real number (0.055 ± 0.006) within 2 SD.
Any failure = infrastructure bug, not a result; stop and fix.
