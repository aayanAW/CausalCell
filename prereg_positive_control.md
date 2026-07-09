# Pre-registration — Working Positive Control (SNR-restricted recoverable regime)

**Date:** 2026-07-08 · **Branch:** `fix/validity-crisis-positive-control` · **Base commit:** cut from `fix/camera-ready-corrections` HEAD.
**Author decision (this session):** push hardest on **Branch A** before accepting a reframe; correctness-first, no deadline; all compute available.
**Supersedes for the reframe:** the substitutability thesis in `CausalCellBench_ICML_reframed.tex` is **on hold** pending this result — it decides which paper we honestly have.

---

## 0. FIT-GATE (autoresearch-loop Step 0)

**GREEN — proceed with guards.** Causal/method benchmark on a large frozen real Perturb-seq atlas (Replogle K562, `data/k562_real.h5ad`, 310k cells); eval is cheap, automatable, a single comparable scalar (edge-set F1 vs a permutation null), hard to overfit given a locked null and fresh multi-seed.

**Sharp hazard (declared):** the search for "a regime where REAL clears the null" is a **garden-of-forking-paths engine** (audit verdict, `CausalCellBench_AUDIT_2026-07-08.json` → positive-control verdict). Mitigations, all locked below BEFORE looking: (1) the strata grid is fully enumerated here; (2) only the **matched-estimand** arm may declare Branch A (other estimands are diagnostics, not extra chances to win); (3) the permutation null is frozen; (4) fresh seeds, not the reused `[42,123,456,789,1024]`; (5) BH-FDR across the gate family; (6) all outcome branches (A / B / C-inversion) are written now.

---

## 1. Question & the three pre-committed branches

Is there a **restricted, biologically motivated regime** in which REAL interventional K562 Perturb-seq, scored through an **estimand-matched** recovery, provably beats a permutation null against the non-circular perturbation-anchored ground truth — establishing a _working positive control_ the substitutability comparison can stand on?

The whole benchmark's meaning turns on this. Pre-committed outcomes (report whichever the data selects; none is reframed post-hoc):

- **Branch A — a matched-estimand stratum clears the null** (margin > MDE, multi-seed, real > overlay in-window, real ≥ correlational baseline) → a **real substitutability benchmark inside the identifiable window**; VCMs are then compared inside that window (Tier 1).
- **Branch B — no stratum clears even with the matched estimand + strong-perturbation + high-coverage + TF-anchored strata** (the audit's modal outcome) → **honest reframe: causal GRN recovery from Perturb-seq is unidentifiable at achievable SNR on this instrument / VCMs collapse below the recoverable floor** — a mechanism/diagnostic paper filling the gap arXiv 2605.04930 concedes.
- **Branch C — a stratum clears AND overlay (now) or a VCM (Tier 1) clears it equally/higher** → the headline **inverts** (predicted ≥ real inside the noise band). Already visible at N=200 (cpa 0.0359, gears 0.0358 > real 0.0314). Must be reported as inversion, never suppressed.

---

## 2. Frozen definitions (locked before any run)

### Data

`data/k562_real.h5ad` (Replogle K562, `obs['gene']` perturbation label, `non-targeting` control, `var['gene_name']` symbols). Panel = the N perturbed=measured genes with ≥ `MIN_CELLS=50` cells, selected per the stratum rule in §3.

### Ground truth (non-circular, total-effect / reachability) — FROZEN

Rebuilt per panel exactly as `scripts/genome_scale_precision.py::anchored_gt`, **primary operating point `q=0.05, τ_fc=0.5`** (locked; matches `prereg_A1_anchored_gt.md`): edge A→B iff BH-FDR(Welch-t of B under A-KO vs control) ≤ 0.05 AND |log2FC| ≥ 0.5, on library-normalized log1p counts, perts with ≥20 cells, self-edges excluded. This GT is a **total-effect / ancestral-reachability** object — the estimand must match it (§2, estimands).

### Permutation null — FROZEN

Per panel: `n_perm ≥ 50` independent **column-shuffles** of the real-data ACE (`col_permute`, RNG seeds `1000+k`, fixed), each scored through the SAME estimand pipeline against the anchored GT. Gives null mean, sd, and **p97.5**. The null is never re-selected after seeing results.

### Estimands scored against the anchored (total-effect) GT

- **E-match (PRIMARY, the only arm that may declare Branch A):** **total-effect ACE ranking** — threshold |ACE_ij| (ACE = E[X_j | do(X_i)] − E[X_j | ctrl]) to the GT edge budget. Same graph object as the reachability GT → the audit's single most plausible route to lifting real above null (path (a), "match the graph object").
- **E-prec (CONTRAST, diagnostic only):** direct-edge precision-support `G = I − (R+ridge)⁻¹` thresholded to budget (the current `precision_edges`). The deliberate estimand↔GT _mismatch_; reported to quantify how much of the floor it explains.
- **E-corr (BASELINE, always reported):** control-cell co-expression |Pearson r| thresholded to budget (a GRNBoost2-family correlational stand-in). If E-corr ≥ E-match on real data in a "passing" stratum, the GT is rewarding correlation → the stratum does **not** count as a causal positive control (guard against the GRNBoost2-beats-causal artifact).

### Data conditions (per estimand)

`real_data`, `elastic_net` (control-only sparse-linear ACE), `overlay_resampled_real` (mean-through-diagonal-overlay upper bound). Deep-VCM conditions are **out of scope for the positive control** (they belong to Tier 1, only meaningful once a positive control exists).

---

## 3. Strata search space — ENUMERATED BEFORE LOOKING (Branch-A levers)

Panel sizes: **N ∈ {200, 500}** (bounded; 200 = densest per-gene GT, 500 = scale check).

Strata (the perturbation/gene-selection rule for the panel), all pre-declared:

| id                            | rule                                                                                                                                                                  |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **S0** baseline               | top-N perts by cell count (current regime)                                                                                                                            |
| **S1** strong-perturbation    | perts with on-target \|log2FC\| ≥ 1.0 (KO demonstrably worked), then top-N by cell count                                                                              |
| **S2** high-coverage          | perts with ≥ 200 cells, then top-N by \|on-target log2FC\|                                                                                                            |
| **S3** TF-anchored            | panel restricted to TFs present in TRRUST (`data/ground_truth/trrust_rawdata.human.tsv`) with ≥1 in-panel curated target — attacks essential-gene pleiotropy directly |
| **S4** strong ∩ high-coverage | S1 ∧ S2 — the maximal-SNR window                                                                                                                                      |

**Gate family for Branch A = {S0..S4} × {N∈{200,500}} × {E-match only}** ≤ 10 gates. E-prec and E-corr are computed on every gate but are **diagnostics/baselines**, not additional win-chances.

---

## 4. Multi-seed protocol (Guard 1)

- **Seeds (fresh, locked): `[2027, 4099, 6113, 8117, 10133, 12157, 14173, 16183, 18191, 20201]` (10).** None overlap the reused `[42,123,456,789,1024]`; disjoint per-axis to avoid correlated errors (audit gotcha).
- Per seed, the ACE-building cells for each gene and the control pool are **bootstrap-resampled** (this is what gives the real condition a genuine SD — the current script's `--seeds` is vestigial because the pooled mean is deterministic; the new harness must resample cells per seed).
- Report per-seed values + across-seed **mean ± SD + 95% t-CI** for every condition/estimand/stratum. Never report a single-seed number as the result.

---

## 5. Primary metric, decision rule, MDE, multiplicity (Guard 2 + 3)

**Primary metric:** across-seed mean edge-set **F1** of `real_data` under **E-match** vs the anchored GT, per (stratum, N).

**MDE (pre-specified negligibility margin):** `MDE_pre = 0.010` F1 — comfortably above the audit's measured n=5 MDE (0.0047) so a "pass" is not noise.

**Gate PASS(stratum, N)** requires ALL of:

1. real F1 95% CI **lower bound > null p97.5** (across-seed), AND
2. (real mean − null mean) **> MDE_pre**, AND
3. real F1 **> overlay F1** (paired across seeds) — else the win is the resampler confound, AND
4. real F1 **≥ E-corr F1** on real data in that stratum — else the GT rewards correlation, not causation.

**Multiplicity:** Benjamini–Hochberg **FDR at q=0.05** across the full Branch-A gate family (§3). A stratum "clears" only if its across-seed paired real-vs-null test survives FDR.

**Stopping rule:** run the full pre-registered grid **once** (all strata × N × 10 seeds). No adaptive expansion, no post-hoc stratum addition. If Branch A does not clear, that is the result — do not keep searching for a passing window.

**Branch decision:**

- **A** iff ≥1 gate PASSes (all four conditions) and survives FDR, and no inversion flag.
- **C (inversion)** iff a gate PASSes conditions 1–2 but overlay ≥ real (fails 3) — report as inversion; the "signal" is the resampler/mean channel, not real biology.
- **B** iff no gate PASSes across the entire locked grid.

---

## 6. What each branch means for the manuscript (pre-committed)

- **A** → keep the substitutability framing but scope every claim to the identifiable window; add the working positive control as a headline; run VCMs inside the window (Tier 1).
- **B** → rewrite: the contribution is the _diagnosis_ (no identifiable window at achievable SNR + the estimand/overlay/retest instrument failures) plus the honest negative; the invariance lemma stays only as a corollary; PerturbHD/CausalBench-Challenge cited as the framing/empirical priors.
- **C** → the paper leads with the inversion (predicted ties/beats real inside the noise band) as the finding, with the null-band caveat central.

---

## 7. Reporting (fixed outputs)

- `results/positive_control/pc_grid.json` — per (stratum, N, seed, estimand, condition): F1/precision/recall/edge-count, plus per-gate null band, real_above_null, FDR-adjusted p, overlay/corr comparisons, PASS flags, and the selected Branch.
- `results/positive_control/pc_summary.md` — the human-readable scorecard + the Branch verdict.
- Decision ledger `results/positive_control/decision_ledger.jsonl` — one line per gate with git SHA + config hash.
- Figures: `figs/pc_real_vs_null_by_stratum.png`.

**Honesty clause:** if the modal Branch B obtains, it is reported as the finding, not hidden. "10 seeds × 10 gates, no matched-estimand stratum cleared the null after FDR" is a valid, publishable outcome.

---

## AMENDMENT 1 — 2026-07-08 (before the confirmatory run; all changes STRENGTHEN rigor)

Filed after a 4-agent self-check (logic / requirements / integration / leakage) of `scripts/positive_control.py` and a 1-seed in-sample smoke. The smoke showed the **estimand match** lifts real F1 from ~0.03 (below null, mismatched precision estimand) to ~0.52 (3× above null, matched total-effect estimand) — but the self-check flagged that in-sample `real > null` is **near-tautological** (the anchored GT is a thresholded copy of the same do(A)→B shift the estimand ranks, on the same cells). These amendments make the positive control defensible. None loosens a criterion; the cross-split is strictly harder to pass.

**A1 — CELL-SPLIT is now the PRIMARY Branch-A test (decisive).** Per seed, each perturbation's cells and the control pool are randomly partitioned into two disjoint halves H1, H2 (`rng(seed)`). The anchored GT is built on **H1**; every ACE (real / overlay / elastic / corr) is estimated on **H2**; edges from H2 are scored against the H1 GT. This breaks the same-data alignment, so **cond1 (real > null) is load-bearing again** and measures whether the total-effect structure is *reproducible across disjoint cell samples* (the honest test-retest, analogous to the paper's technical-duplicate F1=0.045 for the direct-edge path). **Branch A requires the CROSS-SPLIT gate to pass all four conditions + FDR.** The in-sample version (GT+ACE on the same cells) is retained only as a labeled UPPER-BOUND reference and can never declare a pass.

**A2 — Full-power cells (removes the Branch-B confound).** `cap_cells` raised 400→2000 (rarely binds; high-coverage perts keep their SNR) and the GT/ACE use **all** non-targeting controls (~10,691), not a 3,000 cap — so a Branch-B ("no identifiable window at *achievable* SNR") verdict cannot be an artifact of a memory cap. Splits/bootstraps are re-drawn per seed, so the across-seed CI includes split/selection variance.

**A3 — Stricter correlational guard.** cond4 now requires real ≥ **both** control-only co-expression (the pre-registered GRNBoost2-family baseline) **and** all-cell co-expression (the stronger, GRNBoost2-exact analog). A stratum is disqualified if *either* correlational baseline meets/beats the matched causal estimand on real data.

**A4 — Estimand/budget/BH clarifications (documentation of what the code does).**
- Edge budget = **len(GT)** exactly (the off-by-one in top-budget thresholding is fixed so the emitted edge count equals the budget); the 200-edge floor is dropped. A gate with GT < 50 edges is reported but flagged low-power and cannot declare A.
- The permutation null is the pooled set of `n_perm × n_seed` column-shuffles of each seed's real ACE (≥ 50 × 10), each scored through the identical matched estimand and budget. This is a strengthening (more null draws), not a change of null object.
- BH-FDR uses correct step-up (this fixes a latent naive-BH bug present in the committed `genome_scale_precision.py`; GT edge sets are therefore a superset and absolute magnitudes are **not** claimed identical to the paper's genome-scale numbers — the positive control's validity is internal: real vs its own matched null on the same data).

**A5 — S3 (TF-anchored) definition.** Panel = perturbed TRRUST TFs that have ≥1 TRRUST target also perturbed-in-pool, unioned with those in-pool targets, top-N by cell count. This realizes the prereg's "in-panel curated target" intent and attacks essential-gene pleiotropy with genuine curated regulatory edges. If < 20 genes qualify, S3 is skipped (reported, not silent).

**A6 — Reporting.** `pc_grid.json` now stores per-seed arrays, precision/recall, and predicted-edge counts for every (stratum, N, seed, estimand, condition, {in_sample, cross_split}); `pc_summary.md` (scorecard + branch verdict) and `figs/pc_real_vs_null_by_stratum.png` are written.

**Decision rule (restated, cross-split):** Branch **A** iff ≥1 (stratum, N) CROSS-SPLIT matched-estimand gate passes cond1–4 and survives BH-FDR q=0.05; **C_inversion** iff a cross-split gate clears cond1+cond2 but real ≤ overlay; **B** (modal) iff no cross-split gate passes across the full locked grid. Report whichever obtains, honestly.
