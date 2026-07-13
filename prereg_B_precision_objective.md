# Pre-registration — B precision-aware / OT training objective (no-GPU proxy sweep)

**Date:** 2026-07-05 · **Base commit:** `fix/camera-ready-corrections` HEAD.
**FIT-GATE:** GREEN (frozen K562 N=200 atlas, cheap automatable scalar eval, lockable test split). **Directly mirrors `preregistration_j2_lambda.md`.**

## Context (what is already ruled out)
The naive **covariance-Frobenius** penalty `‖Cov(pred)−Cov(real)‖_F` is **DISCARD** (J2 null: best-val λ gained +0.003 ± 0.035, fails the multi-seed gate). This prereg tests the *better-motivated* untried objectives.

## Question
Does adding a **conditional-independence-aware** term — precision-matrix support alignment (graphical-lasso style) and/or a distribution-matching term (MMD / Sinkhorn-OT) — to the reconstruction loss make the learned joint transform beat recon-only on **held-out causal F1 measured against the A1 perturbation-anchored ground truth** (the correct, non-circular target)?

Rationale: interventional causal discovery exploits conditional independence = **precision-matrix zeros**, not covariance magnitude (ties to the J7 identifiability proposition). The J2 penalty matched the wrong statistic; this matches the right one.

## Objectives under test (each swept independently; no mixing until a single term shows signal)
1. **Precision-support** `L_prec = ‖ soft_support(Θ_pred) − soft_support(Θ_real) ‖` where Θ = graphical-lasso precision of the perturbation-response shift; soft-support via a differentiable |·|/(|·|+ε) gate.
2. **MMD** RBF-kernel maximum mean discrepancy between predicted and real joint shift rows.
3. **Sinkhorn-OT** entropic-regularized Wasserstein between predicted and real shift distributions.
Loss: `L = L_recon + λ · L_term`. **One term at a time.**

## Search space (only λ mutated; everything else FROZEN — identical to J2)
- **λ ∈ {0, 0.1, 0.3, 1, 3, 10, 30, 100}** (0 = recon-only incumbent).
- FROZEN: head `Linear200→ReLU→Linear200`, epochs 400, lr 1e-3, Adam, overlay/GIES pipeline, partition 20, N_cells 100, N_ctrl 200.

## Splits (Guard 2 — locked test)
Per seed, ~200 perturbations split **train 120 / val 40 / test 40**. λ (and which of the 3 terms) selected on **validation** only. **Test sealed, touched exactly once** at the end for: recon-only (λ=0), the selected (term, λ\*), and vanilla CPA.

## Primary metric
Held-out edge-set causal **F1 vs the A1 perturbation-anchored GT** (not GIES-on-real). Within-comparable (same GT, budget, partitioning).

## Stopping rule
Fixed grid: ≤3 terms × 8 λ × 5 seeds (42, 123, 456, 789, 1024) on validation. No adaptive expansion. Report the full val-trial count so multiple-comparison exposure is visible.

## Keep/discard (Guard 1 — multi-seed significance)
The selected (term, λ\*) is reported as an improvement over recon-only ONLY if, on the **test** split, it beats λ=0 across **≥3 of 5 seeds** AND mean test-F1 improvement exceeds the run-to-run SD. Otherwise: the honest conclusion is that the training-time joint-structure term does not close the gap in the proxy, and the definitive answer awaits the GPU CPA retrain.

## GPU caveat (explicit, pre-committed)
This proxy is a **CPU stand-in** for retraining CPA end-to-end with the objective (the definitive test, ~45 min GPU, out of scope on this Mac). A green proxy justifies the GPU spend; a null proxy is informative but not final. The proxy is never reported as the definitive result.

## Reporting
`results/phaseB/b_precision_objective_sweep.json` (per-seed, all terms/λ), decision ledger `results/phaseB/b_decision_ledger.jsonl`, figure `b_objective_sweep.png`. Negative result reported straight.
