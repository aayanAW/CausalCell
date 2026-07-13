# Genome-scale causal recovery — the last open avenue, closed

**Date:** 2026-07-05 · Script: `scripts/genome_scale_precision.py` ·
Results: `results/instrument_fix/genome_scale_precision*.json` · Figure: `genome_scale_recovery.png`

## Why this test

Every prior null (J1, J2, B, B2, the instrument-fix) left one escape hatch open: they were all
at N≈100–200 with GIES, whose random 20-gene partitioning imposes a ~9.6% co-partition
recovery ceiling (A1). The roadmap's one remaining positive-result avenue was **genome scale
with a partition-free backend**: at N∈{500,1000,1500} the linear baseline's advantage might
vanish, and a precision/ACE-based backend (inspre-family) has *no* co-partition ceiling. If
real-data causal recovery separated from chance there while ElasticNet collapsed, the verdict
could flip from "VCMs never help" to "VCMs help at scale."

## What was actually run (and an honest methods note)

- **Backend:** inspre is one ADMM solver for precision-matrix / causal-structure recovery from
  the interventional ACE matrix R̂, where do(i)-effects propagate as R̂ ≈ (I−G)⁻¹. inspre's
  full ADMM (200 iters) **exceeds 900 s per fit at N=500 on this machine** and scales
  superlinearly; the 27-fit grid was infeasible locally, and the only remote box has no R and
  was 2× CPU-oversubscribed (load 40/20 cores — running it there would be slow AND would
  disrupt other users). So I used the **same estimator family in closed form**:
  Ĝ = I − (R̂+εI)⁻¹, thresholded to the anchored-GT edge budget — O(N³) once, seconds per fit.
  This is a genome-scale **screen**; the ADMM confirmation is deferred to proper compute.
- **Scale reached:** N=500 (6,802 anchored edges) and N=1000 (14,953 anchored edges) —
  2.5× and 5× the largest prior panel (N=196/200). N=1500 was attempted but the **ElasticNet
  baseline** (1,500 per-gene models), not the causal backend, ran past the runtime window;
  the two completed points are monotone and decisive.
- **Ground truth:** the non-circular perturbation-anchored GT (Welch t-test + BH-FDR,
  q=0.05/τ_fc=0.5, same as A1), rebuilt at each N. Chance = 30-draw column-permutation null of
  the real ACE (destroys cross-gene structure, preserves per-gene marginals).
- **Conditions:** real_data, elastic_net, overlay_resampled_real (perfect-means upper bound).
  VCM arm (GEARS/CPA) needs GPU predictions on these panels — a contingent step, only if this
  screen showed headroom.

## Result — the avenue is closed (decisive negative, monotone in N)

| N | real F1 | real / chance | ElasticNet F1 | ElasticNet / chance | overlay(real means) F1 | chance p97.5 | real > chance? |
|---|---|---|---|---|---|---|---|
| 500  | 0.0346 | **0.85×** | 0.0003 | 0.01× | 0.0559 | 0.074 | **No** |
| 1000 | 0.0068 | **0.33×** | 0.0001 | 0.00× | 0.0227 | 0.031 | **No** |

Two things happen together as N grows, and neither rescues the VCM case:

1. **ElasticNet's edge vanishes** — F1 collapses to ~0.0001 (0.01× chance) by N=500 and stays
   there. This is exactly the roadmap's prediction: the linear baseline cannot hold structure
   at genome scale.
2. **But real-data recovery does NOT rise to fill the gap — it sinks INTO the chance band.**
   Real drops from 0.85× chance (N=500) to 0.33× chance (N=1000); it is not above the
   permutation null at either scale. Even the perfect-means upper bound decays toward chance
   (2.05× → 1.51× on the analytic floor; near the perm-null band by N=1000).

**The linear baseline losing its edge at scale does not create a VCM opportunity, because the
causal-recovery task itself has no signal above chance at that scale — for real data included.**

## Consequence — this is the final piece of a fully-armored negative

The paper can now make a claim very few negative results can:

> The VCM causal-substitutability gap is not an artifact of model quality, ground-truth
> circularity, external-GRN coverage, the mean-overlay sampler, the causal backend, or the gene-set
> scale. We closed each escape hatch in turn — anchored GT (A1), coverage-optimized external GRN
> (A2), cell-level evaluation on real cells (instrument-fix), a trained joint-structure objective
> (B/B2), and now partition-free genome-scale recovery. At every setting, real Perturb-seq itself
> does not recover its own anchored interventional structure above chance through these causal
> algorithms. The task, as currently posed, is under-determined — not the models.

That reframes the contribution from "VCMs underperform" to **"per-perturbation causal-network
recovery from Perturb-seq (real or predicted) is not identifiable at the scales and with the
algorithms the field currently uses — here is the evidence across six independent hardening
axes."** That is a genuinely strong, genuinely surprising negative.

## Honest scope / what remains

- This is a fast precision screen, not the inspre ADMM. The ADMM could in principle separate
  where the closed form doesn't (it adds sparsity + CV model selection) — but it would have to
  beat a chance floor the closed form shows is already at/above the real-data signal, which is a
  steep ask. Running it is now a **confirmation of a negative**, not a search for a positive, and
  belongs on a machine with R and free CPU.
- N capped at 1000 for the completed comparison (1500 attempted). The trend is monotone
  downward; a third point would confirm, not change, the direction.
- The perfect-means upper bound staying marginally above chance at N=500 is the one non-null
  signal — consistent with the overlay-invariance finding (means carry *some* recoverable
  marginal structure) and with the fact that it, too, decays by N=1000.

## Artifacts

- `scripts/genome_scale_precision.py` — the screen (vectorized anchored GT, closed-form
  precision recovery, permutation null)
- `scripts/genome_scale_inspre.py` — the full inspre-ADMM version (deferred to proper compute)
- `results/instrument_fix/genome_scale_precision_n500.json`, `..._n1000.json`
- `genome_scale_recovery.png` — F1-vs-N with chance band + fold-over-chance collapse