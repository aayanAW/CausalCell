# Pre-registration — J2 covariance-regularizer λ sweep

**Date:** 2026-06-23 · **FIT-GATE:** GREEN (causal-method benchmark, large frozen
Replogle K562 atlas, cheap automatable scalar eval, lockable test split).
**Base commit:** branch `fix/camera-ready-corrections` HEAD.

## Question

Does any weight λ on the covariance-Frobenius training penalty make the learned
joint transform (J2 proxy) beat the recon-only head (λ=0) on held-out causal F1?
The single-λ run (λ=1) showed no gain (−0.004); this sweep tests the grid before
concluding the term doesn't help.

## Search space (only λ is mutated; everything else FROZEN)

- **λ ∈ {0, 0.1, 0.3, 1, 3, 10, 30, 100}** (0 = recon-only incumbent).
- FROZEN: head architecture (Linear200→ReLU→Linear200), epochs=400, lr=1e-3,
  optimizer Adam, overlay/GIES pipeline, partition size 20, N_cells=100,
  N_ctrl=200, the normalized Frobenius penalty.

## Splits (Guard 2 — locked test)

Per seed, the ~200 perturbations are split **train 120 / val 40 / test 40**.

- λ is selected on the **validation** split only.
- The **test** split is sealed and touched **exactly once**, at the end, for the
  selected λ\*, for λ=0, and for vanilla CPA. No λ selection uses test F1.

## Primary metric

Held-out edge-set causal F1 vs real-data GIES ground truth (higher better),
within-backend comparable (same GT, edge budget, partitioning).

## Stopping rule

Fixed grid: 8 λ × 5 seeds (42, 123, 456, 789, 1024). No early stop, no adaptive
expansion of the grid.

## Keep/discard rule (Guard 1 — multi-seed significance)

λ\* (best mean **val** F1) is reported as an improvement over recon-only ONLY if,
on the **test** split, it beats λ=0 across ≥3 of 5 seeds AND the mean test-F1
improvement exceeds the run-to-run SD. Otherwise the honest conclusion stands:
the covariance penalty does not help, and the gain in J2 is the learned mapping.

## Reporting

Report: λ*, the val trajectory (mean F1 per λ), the SINGLE locked-test numbers
(λ*, recon, vanilla), per-seed values, and the trial count (8×5=40 val trials)
so multiple-comparison exposure is visible. Negative result reported straight.
Decision ledger: `results/phase4/j2_lambda_decision_ledger.jsonl`.
