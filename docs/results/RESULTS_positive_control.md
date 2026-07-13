# Results — Working Positive Control + Falsification Battery

**Date:** 2026-07-08/09 · **Branch:** `fix/validity-crisis-positive-control`
**Pre-registration:** `prereg_positive_control.md` (+ Amendments 1, 2). **FIT-GATE:** GREEN.
**Verdict: Branch B (refined).** The substitutability headline survives; its _mechanism_ is corrected and the causal claim is honestly scoped.

---

## 1. What was tested

The audit (`CausalCellBench_AUDIT_2026-07-08.json`) found the benchmark had **no working positive control**: on the non-circular perturbation-anchored ground truth, REAL K562 Perturb-seq scored F1 ≈ 0.03, at/below the permutation null, so the substitutability comparison had no operand. We pushed hardest on restoring a positive control (matched estimand + strata + full-power cells + cell-split), then adversarially tried to refute whatever we found.

Ground truth (frozen): perturbation-anchored, **total-effect / reachability** — edge A→B iff BH-FDR(Welch-t of B under A-KO vs control) ≤ 0.05 and |log2FC| ≥ 0.5. Estimand (matched): threshold |ACE_ij| (ACE = E[X|do i] − E[X|ctrl]) = a total-effect ranking. Cross-split: GT built on held-out cell-half H1, ACE on disjoint H2. 10 fresh seeds; permutation-null via column-shuffle; BH-FDR across gates.

---

## 2. The apparent positive control (and why it is not causal discovery)

On the total-effect GT, cross-split, REAL clears the null 2–3× at every stratum (10 seeds):

| stratum               | panel | GT edges | real F1 (total)      | null p97.5 | in-sample F1 |
| --------------------- | ----- | -------- | -------------------- | ---------- | ------------ |
| S0 baseline           | 200   | 1920     | 0.433 [0.430, 0.436] | 0.173      | 0.440        |
| S1 strong-pert        | 200   | 2076     | 0.374                | 0.105      | 0.386        |
| S2 high-cov           | 200   | 2647     | 0.406                | 0.148      | 0.411        |
| S4 strong∩cov         | 117   | 999      | 0.355                | 0.124      | 0.362        |
| SVCM (VCM-pert panel) | 176   | 711      | 0.303                | 0.135      | —            |

Naively this "passes" (real ≫ null, real > overlay, real ≫ corr). **The falsification battery shows it is a reliability coefficient, not recovery:**

- **F1 — a plain DE t-test beats the "causal estimand."** Ranking (A,B) by |Welch-t of B under A-KO| (the statistic that _defines_ the GT) scores **0.60–0.75**, far above the matched |ACE| (0.30–0.43), at every gate. Both are estimates of the same marginal shift Δ(A,B); real≫null just certifies that the perturbation-effect matrix is **reproducible across disjoint cell halves** — split-half reliability, which holds for any heterogeneous reproducible measurement, causal or not. (in-sample 0.44 ≈ cross-split 0.43: the split cost ~0.01, so it broke nothing.)

| gate | real \|ACE\| | DE t-test | cond5 (real>de)  |
| ---- | ------------ | --------- | ---------------- |
| S0   | 0.433        | 0.754     | FALSE (Δ −0.321) |
| S1   | 0.374        | 0.661     | FALSE (Δ −0.287) |
| S4   | 0.355        | 0.703     | FALSE (Δ −0.349) |
| SVCM | 0.303        | 0.601     | FALSE (Δ −0.298) |

- **F2 — direct causal structure is near-unidentifiable.** Scored against a direct-edge GT (transitive reduction of the total-effect set), REAL sits at **F1 0.03–0.12** — ~10× below the reachability signal, marginally above a ~0.02–0.05 null. Every VCM is **at or below the direct-edge null** at every gate. The 0.43 "recovery" is entirely the total-effect/reachability object; direct causal edges are recovered barely above chance for real data and below chance for predicted data.

---

## 3. Substitutability: VCMs fail even the trivial (mean/reachability) channel

The decisive substitutability numbers (total-effect GT, 10 seeds):

| gate | real  | overlay (perfect-mean proxy) | CPA   | GEARS | Geneformer |
| ---- | ----- | ---------------------------- | ----- | ----- | ---------- |
| S0   | 0.433 | 0.422                        | 0.094 | 0.094 | 0.082      |
| S1   | 0.374 | 0.355                        | 0.027 | 0.030 | 0.029      |
| S4   | 0.355 | 0.338                        | 0.041 | 0.042 | 0.039      |
| SVCM | 0.303 | 0.287                        | 0.131 | 0.185 | 0.094      |

- **overlay ≈ real** (0.42 ≈ 0.43): a model that emits the _correct per-gene means_ recovers everything real does — the recoverable signal **is the marginal means**, not hidden joint structure.
- **real VCMs ≪ real** (0.08–0.18 vs 0.30–0.43, a 3–5× deficit): CPA/GEARS/Geneformer do **not** emit correct means (mode collapse / 23× effect-size inflation), so they fail even this trivial reachability channel. This is _not_ an inversion (predicted does not tie real) and _not_ a joint-structure-blindness artifact — it is a direct consequence of wrong marginal predictions.

---

## 4. Conclusion — Branch B (refined); implications for the manuscript

The substitutability headline **survives and is strengthened** (no VCM reaches real quality — they fail even the generous total-effect channel), but the paper's _explanation_ must change:

1. **The "positive control" is DE test-retest reliability, not causal discovery** — a t-test recovers the total-effect GT better than the matched estimand. What real Perturb-seq reproducibly recovers is _reachability / total effects_, i.e. a differential-expression map.
2. **Direct causal-structure recovery is near-unidentifiable at achievable SNR** — real F1 0.03–0.12 (~10× below reachability, marginally above chance); VCMs at/below chance. This is the honest scope of the causal claim and matches the audit's original 0.03 floor.
3. **VCMs are non-substitutable because their marginal means are wrong (mode collapse), not because causal discovery needs joint structure the mean-overlay hides.** The overlay (perfect means) ties real; the VCMs (wrong means) do not. → The Wall-1 lemma / "signal lives in cell-level joint structure" framing is **not the operative mechanism** and must be demoted (the lemma stays only as a corollary of arXiv:2603.17041).

### Manuscript actions

- Replace the "restored positive control / joint-structure residual" narrative with the reachability-vs-direct decomposition + the mean-fidelity mechanism above.
- Report the total-effect positive control honestly as a **reliability floor** (with the DE-baseline caveat), not as causal recovery.
- Keep the negative substitutability verdict; re-explain it via mode-collapsed means failing the reachability channel + direct-causal unidentifiability.
- Demote the overlay-invariance lemma to a corollary; drop the A3 "positive control localizes signal to joint structure" claim (contradicted: signal is in the means).
- Cite PerturbHD (Bereket & Leskovec, bioRxiv 2026.04.23.719015) and fix the CausalBench attribution (interventional-doesn't-beat-observational = Comms Biol 2025 benchmark paper, not the CLeaR-2025 challenge report).

### Data

Grid: `results/positive_control/pc_grid.json`, `pc_summary.md`. Falsification: `results/positive_control/pc_falsify.json`. Figure: `figs/pc_real_vs_null_by_stratum.png`. Ledger: `results/positive_control/decision_ledger.jsonl`. Reproduce: `.venv_gears/bin/python scripts/positive_control.py` and `scripts/pc_falsify.py`.

### Caveats (honest)

- Direct-edge GT via transitive reduction is imperfect on a cyclic network (audit); it is one of several imperfect direct-edge proxies, reported as such. TRRUST-curated in-panel edges were too sparse (≈0 on the essential-gene panels) to serve as an independent direct-edge GT.
- VCM predictions are single-fixed-mean per perturbation (N=200 K562); the VCM arm varies only through the GT split.
- Absolute F1 magnitudes are not identical to the committed `genome_scale_precision.py` (this harness uses correct step-up BH — the reference had a naive-BH bug — and capped cells); the positive control's validity is internal (real vs its own matched null on the same data).
