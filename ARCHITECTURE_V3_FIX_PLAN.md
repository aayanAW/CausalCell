# PerturbCausal — Architecture V3 Fix Plan

**Version:** 3.0 (fix plan, post-audit)
**Date:** 2026-05-25
**Status:** Pre-implementation. Predecessor: ARCHITECTURE_V2.md (2026-04-09).
**Driver doc:** This plan replaces the "pending tasks" section of `HANDOFF_PerturbCausal_2026-05-25.md` and supersedes the open Gates 2-3 in `CausalCellBench_Proposal_v5.md`.

---

## 0. Executive Summary

The 2026-05-25 audit identified five fatal flaws, three overstated claims, and ~ten engineering / framing gaps. This plan converts the project from a 6/10 manuscript to a defensible 8/10 manuscript suitable for NeurIPS 2026 Datasets & Benchmarks Track (Jun 11), with a fallback to TMLR or NeurIPS workshops.

**Total estimated wall time:** 18-25 days of focused work, parallelizable to 12-15 days with overnight Modal jobs.
**Total estimated compute spend:** $300-500 Modal (Max 20x plan extras already running; net new cash spend ≤ $300).
**Hard blockers solved in Phase 0** (Day 1-2): RPE1 bug, GRNBoost2 claim, STATE caveat, calibration framing.
**Strongest possible artifact:** PerturbCausal v3 — multi-dataset, multi-model, multi-backend, multi-seed, multi-calibration-mode, with bioRxiv preprint + GitHub release.

The plan is structured as **11 phases**, each with: goals, scope, file touchpoints, code skeletons where load-bearing, exit criteria, compute, wall-time, risk, and dependencies. Phases are ordered by the critical path; parallelization opportunities are noted explicitly.

Decision points (kill / pivot triggers) are embedded throughout. If a decision point fails (e.g., corrected RPE1 ElasticNet F1 < 0.10), the plan branches to a defined fallback rather than continuing blind.

---

## 1. Pre-Conditions and State Freeze

Before any phase begins, freeze the current state:

```bash
cd "/Users/aayanalwani/virtual cells"
git checkout -b v3-fixplan
git add -A
git commit -m "v3-baseline: snapshot before audit-driven fixes"
git tag v2.15-final  # mark the submission package state
```

This isolates the fix-plan branch from the v2.15 submission package. If the plan misfires, `git checkout v2 && git reset --hard v2.15-final` restores the submission.

**Lock the following before any phase touches code:**

1. `results/multi_seed/aggregated_results.json` — the verified-correct headline numbers.
2. `results/state/state_f1_summary.json` — the STATE single-seed result.
3. `results/norman2019_summary.json` — Norman ElasticNet baseline.
4. `data/k562_subset_n200_slim.h5ad` — the eval data.
5. `results/gies_cache/n200_s42_p20/real_data_edges.json` — ground truth.
6. `submissions/icml/CausalCellBench_ICML.tex` (v2.15) — manuscript baseline.

These are the artifacts whose values must not regress without a documented reason.

**Compute pre-conditions.** Modal account `alwaniaayan6` active. Local M4 16GB. R 4.5.2 + inspre 0.0.1. `.venv_gears` Python 3.9. `project config` autonomous mode guarantees no mid-task prompts.

**Process pre-conditions.**
- Every code change passes pre-flight (AST + import + smoke) before Modal launch (`feedback_preflight_checks.md`).
- Every non-trivial change spawned through self-check agents (`feedback_self_check_agents.md`).
- Every Modal failure mode tagged in the Risk Register (§4).

---

## 2. Phase Plan

### Phase 0 — Critical-Path Bug Fixes and Framing Corrections

**Wall time:** 1-2 days (Day 1-2).
**Compute:** ~$5 Modal + ~2h M4.
**Goal:** Eliminate the load-bearing falsifiable claims that audit identified. Nothing in subsequent phases requires new experiments — Phase 0 is text + small reruns only.

**Tasks:**

P0.1 — **Fix `scripts/phase9_rpe1.py` RPE1 pipeline bug.**
Bug location: lines 269 (real condition) and 286-287 (claimed "ElasticNet" condition) both call `build_overlay_data(ctrl_X, ctrl_mean, pert_means, ...)` with the **same `pert_means`** argument. The ElasticNet condition therefore uses real perturbation means — not ElasticNet predictions. Fix:

```python
# scripts/phase9_rpe1.py — replace lines ~285-293

# Train ElasticNet on RPE1 ctrl cells (per-target-gene model)
logger.info("\nStep 3b: Fitting ElasticNet on RPE1 ctrl cells")
n_g = len(gene_subset)
enet_models = {}
for g_idx in range(n_g):
    feature_idx = [i for i in range(n_g) if i != g_idx]
    m = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=0)
    m.fit(ctrl_X[:, feature_idx], ctrl_X[:, g_idx])
    enet_models[g_idx] = m

# Predict per-perturbation means with ENet propagation
enet_pert_means = {}
ctrl_means_local = ctrl_X.mean(axis=0)
for g_idx, gene in enumerate(gene_subset):
    input_state = ctrl_means_local.copy()
    input_state[g_idx] *= 0.1  # 90% CRISPRi knockdown
    mean_pred = input_state.copy()
    for t_idx, model in enet_models.items():
        if t_idx == g_idx:
            continue
        feature_idx = [i for i in range(n_g) if i != t_idx]
        mean_pred[t_idx] = max(
            float(model.predict(input_state[feature_idx].reshape(1, -1))[0]), 0.0
        )
    enet_pert_means[gene] = mean_pred

# Now build overlay data using ENet predictions
expr_enet, interv_enet = build_overlay_data(
    ctrl_X, ctrl_mean, enet_pert_means, gene_subset, N_CTRL, N_CELLS_PER_PERT, SEED
)
```

After fix, **invalidate the cache** at `results/phase9_rpe1/elastic_net_edges.json` and re-run GIES on the corrected ENet overlay.

P0.2 — **Re-frame GRNBoost2 significance.**
Edit `submissions/icml/CausalCellBench_ICML.tex` Table 1 caption and §5.1/§5.2 text. Replace "p<0.01 vs. all interventional methods" with the actually-achievable phrasing:

> GRNBoost2 (observational baseline) sits at F1 0.098 ± 0.004, non-overlapping with the predictive methods (95% CIs disjoint by ≥0.27 F1). With n=5 paired-sign-flip permutation, the minimum two-sided p-value is 2/2⁵=0.0625; the n=5 budget cannot certify p<0.05, but the magnitude of the gap (≈30 standard errors of the deep models' CI) leaves no statistical ambiguity about the direction.

Optionally bump multi-seed to n=10 in Phase 1 to retire this caveat entirely.

P0.3 — **STATE caveat in abstract.**
Edit abstract to read:

> ... STATE is the first model in the roster to materially exceed the linear baseline (F1=0.462, single seed, 320-cell embedding subset), but its absolute causal F1 remains below 0.5 ...

Phase 3 will re-run STATE at n≥3 seeds; if successful, this caveat is removed at that point.

P0.4 — **Calibration framing.**
Edit abstract + §1 contributions + §10 deliverables to use "quantile-matching calibration" (singular) rather than "calibration toolkit (three modes)." This is honest about what is actually implemented. Phase 4 will decide whether to implement Modes 2-3.

P0.5 — **RPE1 §5.7 rewrite, pending P0.1 result.**
After P0.1 re-runs, if corrected ENet F1 ≥ 0.30, rewrite §5.7 with the real number and a single sentence acknowledging the prior result was a pipeline bug. If F1 < 0.30, restructure §5.7 to "RPE1 baseline behavior" and drop the "cross-context replication" framing entirely.

P0.6 — **Norman §3.5 scope narrowing.**
Rewrite to read:

> Norman 2019 substitutability is tested for the ElasticNet baseline only; deep-model training on Norman is reported in Phase 2 of this plan (Section 5.X). The ElasticNet result on Norman (F1=0.261) confirms the baseline survives on a CRISPRa paradigm and different lab, but is silent on deep-model substitutability in that setting.

**Exit criteria for Phase 0:**
- `scripts/phase9_rpe1.py` patched and re-run; `results/phase9_rpe1/phase9_results.json` regenerated with a non-1.0 precision.
- Manuscript v2.16 commits with §3.5, §5.1/§5.2 GRNBoost2 phrasing, abstract STATE caveat, calibration framing, and §5.7 RPE1 rewrite.
- All changes pass a `git diff` self-check agent review.

**Dependencies:** None. Phase 0 is the unblocker for all subsequent phases.

**Risks:** RPE1 corrected ENet F1 could be < 0.10, killing the cross-context wing. Mitigation: Phase 2 introduces deep models on RPE1, which would re-establish the wing on different evidence. If both fail, drop RPE1 from the paper and lean on Norman.

---

### Phase 1 — Multi-Seed Bump (n=5 → n=10)

**Wall time:** 1 day (5h compute on M4, can run overnight).
**Compute:** ~$10-30 Modal (optional acceleration) or zero (M4 only).
**Goal:** Tighten CIs, retire the n=5 paired-permutation floor (p=0.0625), and provide statistical power for direction-aware claims.

**Tasks:**

P1.1 — Extend the seed list in `scripts/aggregate_multi_seed.py` from `[42, 123, 456, 789, 1024]` to `[42, 123, 456, 789, 1024, 2048, 4096, 7919, 13337, 31337]`. The 10 seeds give a t-distribution 95% CI half-width ≈ 1.83 × σ/√10 ≈ 0.6σ, vs n=5 half-width 2.78 × σ/√5 ≈ 1.24σ — roughly halves the CI width.

P1.2 — Run the full K562 evaluation on the 5 new seeds. For each seed: ElasticNet, CPA, GEARS, Geneformer, overlay-real, GRNBoost2 conditions through GIES. Per-seed wall ≈ 30 min on M4 → ~2.5h total for new seeds.

P1.3 — Re-run paired permutation tests with n=10. Minimum two-sided p = 2/2¹⁰ = 0.00195. The "GRNBoost2 significantly worse, p<0.01" claim becomes achievable.

P1.4 — Regenerate Table 1 in the manuscript with n=10 CIs, replace n=5 numbers, update GRNBoost2 significance language.

P1.5 — Statistical-power calculation. Compute minimum detectable F1 difference at n=10, std≈0.016: MDE ≈ 1.83 × 0.016 × √(2/10) ≈ 0.013 F1. Add this to manuscript §4 (statistical analysis section).

**Exit criteria:**
- `results/multi_seed/aggregated_results.json` contains 10 seeds × 7 conditions.
- Pairwise permutation table includes p-values down to 0.002 floor.
- Manuscript v2.17 commits with updated Table 1.

**Dependencies:** Phase 0 complete (no manuscript text races).

**Risks:** The 0.014 F1 difference between ElasticNet and the deep models might become significant at n=10. If so, the headline finding shifts from "tied" to "small but real gap." This is an honest improvement; rewrite §5.1.

---

### Phase 2 — Cross-Context (RPE1) and Cross-Paradigm (Norman) Deep-Model Training

**Wall time:** 5-7 days, mostly Modal queue.
**Compute:** ~$100-150 Modal (GEARS + CPA + Geneformer × 2 datasets).
**Goal:** Convert "cross-context" and "cross-paradigm" from one-baseline claims to multi-model claims.

**Tasks:**

P2.1 — **Train GEARS, CPA, Geneformer on RPE1.**
Adapt the existing `scripts/modal_run_gears_real.py`, `modal_run_cpa_real.py`, `modal_run_geneformer_real.py` to consume `data/rpe1_real.h5ad`. The model hyperparameters should match the K562 runs (Appendix §A.1 of manuscript). The eval gene panel: 200 RPE1 perturbation-eligible genes selected with seed 42 from the eligible set, deterministic across all three model runs.

Each model produces a `.h5ad` of predicted means → push through `scripts/state_eval_gies.py` (refactored to be dataset-agnostic) + the GIES pipeline.

Expected wall:
- GEARS RPE1: ~60 min Modal H100.
- CPA RPE1: ~6 min Modal CPU (or A10G).
- Geneformer RPE1: ~10 min Modal CPU.
- GIES on each: 30 min M4 per condition × 3 conditions = 1.5h.

P2.2 — **Train GEARS, CPA, Geneformer on Norman 2019.**
Similar but on `data/norman2019.h5ad`. Norman has fewer cells per perturbation than Replogle (≤ 30 vs ≤ 300), so hyperparameters may need adjustment:
- Increase `max_epochs` 50% (more passes through fewer cells).
- Drop CPA's `n_epochs_adv_warmup` to 30 (smaller dataset).
- Geneformer in-silico perturbation is data-independent — should work as-is.

Same per-model evaluation.

P2.3 — **Multi-seed (n=5 minimum) on RPE1 + Norman.**
For the headline-quality result, each model's predictions are evaluated across 5 seeds of GIES (varying gene-subset selection, ctrl-subsample, partition order). 5 seeds × 3 conditions × 2 datasets × 30 min = ~15h M4 compute, run overnight in 2-3 batches.

P2.4 — **Update §5.7 and §3.5 with the real results.**
The "ElasticNet ties deep models" claim either replicates on RPE1 + Norman (strengthens the paper substantially) or fails on one or both (interesting finding by itself: substitutability depends on dataset).

**Exit criteria:**
- `data/model_outputs_rpe1_n200/{gears,cpa,geneformer}_predictions.h5ad` exist.
- `data/model_outputs_norman_n100/{gears,cpa,geneformer}_predictions.h5ad` exist.
- `results/multi_seed/aggregated_rpe1.json` and `aggregated_norman.json` written.
- Manuscript v2.18 commits with cross-context table + cross-paradigm table.

**Dependencies:** Phase 0 (P0.5, P0.6 framing).

**Risks:**
- Deep models on Norman may diverge or underperform substantially due to small per-perturbation cell counts. Mitigation: report failure mode honestly; reframe as "deep models require ≥50 cells/perturbation to train stably."
- Modal GPU exposure may flake again (handoff documents 5 failed CUDA attempts for STATE). Mitigation: train on CPU if needed, use longer Modal timeouts.

**Decision point Δ2.1:** After P2.1, inspect corrected RPE1 ElasticNet F1.
- If F1 ≥ 0.30 and deep models also ≥ 0.30: cross-context replication holds — keep §5.7 expanded.
- If F1 < 0.20: ElasticNet baseline doesn't transfer — rewrite §5.7 as "structural differences between K562 and RPE1 mean the substitutability test is dataset-specific" and reduce RPE1's prominence in the paper.

---

### Phase 3 — STATE Multi-Seed + Larger Cell Budget

**Wall time:** 2-3 days, mostly Modal queue.
**Compute:** ~$50-100 Modal CPU (Modal GPU exposure broken — handoff §"Known Issues").
**Goal:** Convert the headline STATE result from single-seed-320-cell to multi-seed-≥1000-cell.

**Tasks:**

P3.1 — **Larger cell-budget STATE embedding.**
Current: 320 cells (1 perturbed per gene + 120 controls). Target: 5 cells per perturbation + 200 controls = 1200 cells. SE-600M at 28s/cell CPU = ~9h. Fits inside Modal's 6h timeout if split into 2 jobs (600 cells each, 4.7h each).

Alternative: GPU debug attempt. The handoff documents that `gpu="A10G"` returns `CUDA: False`. Spend 4h diagnosing with `nvidia-smi` in a minimal Modal function before falling back to CPU. If GPU works, embedding wall drops to ~15 min.

P3.2 — **STATE inference on multiple seeds.**
The STATE model itself is deterministic given inputs; "seed" variation comes from the downstream GIES partitioning + ctrl-subsample. Run 3 seeds × 1 STATE embedding (deterministic) = 3 GIES evaluations × 30 min = 1.5h M4.

P3.3 — **Update §5 STATE paragraph with mean ± std.**
Report F1 mean across 3 seeds, range, and explicit per-seed values. If mean is below 0.43 with the larger cell budget, the "STATE materially exceeds ElasticNet" claim weakens — restructure as "STATE narrows but does not close the gap."

P3.4 — **Optional: STATE on RPE1 + Norman.**
If Phase 2 succeeds and the project has time, run STATE on the other two datasets. Each: 1 embedding (Modal CPU 5-10h) + 3 GIES seeds. Total ~30h Modal + ~10h M4. Defer to post-submission v2 if time-constrained.

**Exit criteria:**
- `results/state/state_multiseed_f1.json` contains 3 seeds × 1 dataset (K562) with mean ± std + range.
- Optionally state_multiseed_rpe1.json, state_multiseed_norman.json.
- Manuscript STATE paragraph updated.

**Dependencies:** None for P3.1 / P3.2. P3.4 depends on Phase 2.

**Risks:**
- Modal CPU spend doubles vs current ($30 → $60 per dataset). Acceptable.
- STATE solver might fail on RPE1 (different gene panel, may hit a non-K562-specific version of the `ALL_PARALLEL_STYLES is None` bug). Mitigation: same source-patch trick worked for K562; expect to work for RPE1.

**Decision point Δ3.1:** STATE F1 after multi-seed at 1200 cells.
- F1 ≥ 0.45 with σ ≤ 0.02: keep "first model to exceed ElasticNet" headline.
- F1 between 0.40 and 0.45 with overlapping CIs: rewrite to "STATE narrows the gap to within CI overlap of ElasticNet."
- F1 < 0.40: the original headline collapses; restructure paper around "the 2025 foundation-model wave has not yet closed the substitutability gap."

---

### Phase 4 — Calibration Modes 2 & 3 (Decision: Implement or Drop)

**Wall time:** 3-5 days if implemented; 0 days if dropped.
**Compute:** ~$10 M4 (calibration is fast).
**Goal:** Either deliver the "three-mode calibration toolkit" the proposal promised, or honestly drop the "toolkit" framing.

**Decision point Δ4.1 (made at start of phase):**
- If Phase 2 + Phase 3 results indicate the substitutability gap is dominated by joint-structure failure (likely): Modes 2-3 become the methodological pitch of the paper. **Implement.**
- If results indicate the gap is largely magnitude (less likely given current data): Mode 1 is sufficient. **Drop Modes 2-3, rebrand as "Bolstad calibration applied to VCM outputs."**

Default branch: **Implement** (matches the audit's recommended path forward).

**Tasks (Implement branch):**

P4.1 — **Mode 2: Covariance preservation.**

Math: given predicted shift matrix Δ ∈ R^(N_pert × N_gene) and control covariance Σ_ctrl, project Δ onto the top-k principal subspace of Σ_ctrl and shrink off-axis components.

```python
# causalcellbench/calibration/covariance_preservation.py
class CovariancePreservingCalibrator:
    def __init__(self, n_components: int = 50, shrinkage: float = 0.5):
        self.n_components = n_components
        self.shrinkage = shrinkage  # 0 = no shrinkage, 1 = project entirely to subspace
        self.basis = None  # (n_genes, n_components)

    def fit(self, ctrl_X: np.ndarray) -> "CovariancePreservingCalibrator":
        # ctrl_X: (n_ctrl_cells, n_genes)
        ctrl_centered = ctrl_X - ctrl_X.mean(axis=0, keepdims=True)
        # PCA via SVD
        U, S, Vt = np.linalg.svd(ctrl_centered, full_matrices=False)
        self.basis = Vt[: self.n_components].T  # (n_genes, n_components)
        return self

    def transform(self, delta: np.ndarray) -> np.ndarray:
        # delta: (n_perturbations, n_genes) — predicted shifts
        # Project onto basis and reconstruct; mix with original by shrinkage factor
        coords = delta @ self.basis  # (n_perts, n_components)
        on_basis = coords @ self.basis.T  # (n_perts, n_genes)
        return self.shrinkage * on_basis + (1 - self.shrinkage) * delta
```

Sweep `n_components ∈ {10, 50, 100, 200}` and `shrinkage ∈ {0.25, 0.5, 0.75, 1.0}`. The "right" setting is the one that maximizes downstream causal F1 on a held-out perturbation set.

P4.2 — **Mode 3: Sparsity injection.**

Math: soft-threshold predicted log-FC values to drive the near-zero fraction to match empirical (~37% in real data at the 200-gene scope).

```python
# causalcellbench/calibration/sparsity_injection.py
class SparsityInjectionCalibrator:
    def __init__(self, target_near_zero_frac: float = 0.37, tau_eps: float = 1e-3):
        self.target_near_zero_frac = target_near_zero_frac
        self.tau_eps = tau_eps
        self.threshold = None  # learned soft-threshold tau

    def fit(self, log_fc_pred: np.ndarray) -> "SparsityInjectionCalibrator":
        # Binary search for tau s.t. fraction(|FC| < tau) ≈ target
        abs_fc = np.abs(log_fc_pred.flatten())
        lo, hi = 0.0, abs_fc.max() + self.tau_eps
        while hi - lo > self.tau_eps:
            mid = (lo + hi) / 2
            frac = (abs_fc < mid).mean()
            if frac < self.target_near_zero_frac:
                lo = mid
            else:
                hi = mid
        self.threshold = (lo + hi) / 2
        return self

    def transform(self, log_fc: np.ndarray) -> np.ndarray:
        # Soft-threshold: keep sign, shrink magnitude near zero
        sign = np.sign(log_fc)
        abs_fc = np.abs(log_fc)
        # Smooth shrinkage (not hard cutoff)
        shrunk = np.maximum(abs_fc - self.threshold, 0.0) / (
            1 + self.threshold / (abs_fc + 1e-8)
        )
        return sign * shrunk
```

Sweep `target_near_zero_frac ∈ {0.20, 0.30, 0.37, 0.45}`.

P4.3 — **Composability and pipeline integration.**

The three modes are applied in sequence: Mode 3 (sparsity) → Mode 1 (quantile match magnitude) → Mode 2 (covariance projection). Order matters: sparsity removes noise first, magnitude matching corrects the remaining distribution, covariance projection aligns to control structure last.

```python
# causalcellbench/calibration/__init__.py
class CalibrationPipeline:
    def __init__(self, modes: list[str] = ["sparsity", "quantile", "covariance"]):
        self.modes = modes
        self.calibrators = {}

    def fit(self, real_data, ctrl_X, pred_log_fc):
        if "sparsity" in self.modes:
            target = (np.abs(real_data) < 0.05).mean()
            self.calibrators["sparsity"] = (
                SparsityInjectionCalibrator(target_near_zero_frac=target).fit(pred_log_fc)
            )
        if "quantile" in self.modes:
            self.calibrators["quantile"] = QuantileCalibrator().fit(
                real_data.flatten(), pred_log_fc.flatten()
            )
        if "covariance" in self.modes:
            self.calibrators["covariance"] = CovariancePreservingCalibrator().fit(ctrl_X)
        return self

    def transform(self, pred_log_fc):
        out = pred_log_fc.copy()
        for mode in self.modes:
            if mode in self.calibrators:
                out = self.calibrators[mode].transform(out)
        return out
```

P4.4 — **Evaluate full pipeline on K562 + RPE1 + Norman.**

For each (dataset × model × mode-subset) → causal F1. ~24 combinations × ~30 min each = ~12h M4, run overnight.

P4.5 — **Tests.**

```python
# tests/test_calibration_modes.py
def test_sparsity_injection_matches_target_frac():
    rng = np.random.default_rng(0)
    log_fc = rng.normal(0, 1, size=(100, 100))
    cal = SparsityInjectionCalibrator(target_near_zero_frac=0.30).fit(log_fc)
    out = cal.transform(log_fc)
    actual = (np.abs(out) < 0.05).mean()
    assert abs(actual - 0.30) < 0.05

def test_covariance_preservation_preserves_top_subspace():
    rng = np.random.default_rng(0)
    ctrl = rng.normal(0, 1, size=(200, 100))
    ctrl[:, :10] *= 5  # high-variance subspace
    cal = CovariancePreservingCalibrator(n_components=10, shrinkage=1.0).fit(ctrl)
    delta = rng.normal(0, 1, size=(50, 100))
    out = cal.transform(delta)
    # Output should live in the 10-dim subspace -> rank ≤ 10
    assert np.linalg.matrix_rank(out, tol=1e-6) <= 11
```

**Exit criteria:**
- `causalcellbench/calibration/__init__.py` exports `QuantileCalibrator`, `CovariancePreservingCalibrator`, `SparsityInjectionCalibrator`, `CalibrationPipeline`.
- All three modes tested (3+ unit tests per mode).
- `results/calibration_modes/` contains per-mode F1 numbers for at least the K562 + RPE1 datasets.
- Manuscript v2.19 commits with three-mode table and discussion of which mode contributes most.

**Tasks (Drop branch):**

P4.D1 — Edit manuscript abstract, §1 contribution 4, §4.5 (Methods/Calibration), and §10 deliverables. Replace all "toolkit" / "three modes" / "covariance preservation, sparsity injection" mentions with "quantile-matching calibration."

P4.D2 — Edit `causalcellbench/calibration/` to remove the unimplemented mode stubs.

**Dependencies:** Phase 1, Phase 2 (need multi-dataset multi-seed baselines to evaluate calibration modes against).

**Risks:**
- Mode 2 (covariance preservation) may collapse predictions onto a trivial subspace and hurt all models uniformly. Mitigation: shrinkage parameter sweep; fall back to `shrinkage=0` (no-op).
- Mode 3 (sparsity injection) is a non-linear transformation that can destabilize downstream GIES if too aggressive. Mitigation: target sparsity must match real data; bound by hyperparameter sweep.
- Composability of three modes is unproven. The order chosen above is principled but not validated. Mitigation: ablation on order (3! = 6 permutations × small subset = ~3h compute).

**Decision point Δ4.2:** After P4.4 + sweeps.
- If Mode 1+2+3 pipeline closes ≥ 70% of the substitutability gap on K562: strong calibration result; promote to a co-headline alongside the foundation-model evaluation.
- If pipeline closes 40-70%: keep as a contribution but soften "toolkit" language.
- If pipeline closes < 40% (essentially same as Mode 1 alone): publish honestly, frame as "magnitude calibration is necessary but joint-structure correction requires training-time intervention."

---

### Phase 5 — Track B (Biological Ground Truth via TRRUST)

**Wall time:** 1-2 days.
**Compute:** Negligible (M4 only).
**Goal:** Convert the "Track B implemented in released code" promise into an actual headline number.

**Tasks:**

P5.1 — **Run `scripts/track_b_biological_ground_truth.py`** on the K562 predictions from each model (vanilla + calibrated). Need:
- TRRUST v2 download (`https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv`) — present at `causalcellbench/data/trrust.py`.
- For each predicted edge set, compute (a) F1 against TRRUST-curated K562-relevant edges, (b) Spearman concordance of per-model ranking against Track A.

P5.2 — **Extend Track B to RPE1 and Norman** if Phase 2 delivers multi-model results.

P5.3 — **Headline number to manuscript.**
A single sentence in §5 (Results) plus a row in Table 1: "Track B F1 (TRRUST) and Spearman concordance with Track A."

P5.4 — **Discussion of concordance.**
If Track A and Track B rankings agree (Spearman > 0.7): "Track A is a faithful proxy for biological signal." If they disagree (Spearman < 0.4): "Track A measures pipeline reproducibility, Track B measures biological recovery; the substitutability question requires both axes."

**Exit criteria:**
- `results/track_b/trrust_per_model_f1.json` exists.
- `results/track_b/spearman_concordance.json` exists.
- Manuscript v2.20 commits with Track B row in Table 1 + 2-sentence Discussion paragraph.

**Dependencies:** None for K562 (predictions already exist). RPE1+Norman extension depends on Phase 2.

**Risks:**
- TRRUST has limited K562-relevant edge coverage (~8400 TF→target edges across all human cell types; few are K562-specific). Track B F1 will be low in absolute terms (<0.05 expected). Frame as "Track B is a directionality check, not an absolute performance metric."
- ChIP-seq could be a stronger Track B alternative (more cell-type-specific). Defer to v2 of the paper.

---

### Phase 6 — Geneformer Memorization Quantification

**Wall time:** 2-3 days.
**Compute:** ~$30 Modal CPU.
**Goal:** Disentangle Geneformer's F1 = 0.425 into (memorization) + (zero-shot generalization).

**Tasks:**

P6.1 — **Identify post-Geneformer-V2-pretraining Perturb-seq datasets.**
Geneformer V2's Genecorpus-30M was finalized circa 2023. Candidates:
- Norman et al. 2019 — pre-Geneformer, contaminated.
- Replogle et al. 2022 — pre-Geneformer, **contaminated**.
- Frangieh et al. 2021 — pre-Geneformer.
- **Nadig et al. 2024-2025 large-scale Perturb-seq** — post-Geneformer, candidate.
- **Recent CRISPR pooled screens 2024-2025 on novel cell lines** — candidate.

Pick one with public availability and ≥ 50 perturbation-eligible genes.

P6.2 — **Re-run Geneformer in-silico perturbation on the held-out dataset.**
~10 min Modal CPU per condition × 5 seeds = 50 min.

P6.3 — **Compare per-seed F1.** If Geneformer F1 drops substantially on the held-out dataset (e.g., from 0.425 to 0.32), then memorization accounts for ~25% of its Replogle performance. Report transparently.

P6.4 — **Manuscript update.**
A new §6 Limitations subsection: "Pretraining contamination is quantified by comparing Geneformer V2 on Replogle (in-corpus) and [held-out dataset] (out-of-corpus): F1 0.425 → F1 [held-out value]. The Δ provides a lower bound on memorization-driven performance."

**Exit criteria:**
- A new dataset evaluated (h5ad in `data/heldout_postpretraining/`).
- Per-seed Geneformer F1 reported in `results/geneformer_memorization.json`.
- Manuscript v2.21 commits with memorization quantification.

**Dependencies:** None (parallel to Phase 2-4).

**Risks:**
- The held-out dataset may have different ground-truth GIES edge counts due to different cell-line biology. Mitigation: report F1 absolute and Δ relative to ElasticNet on the same dataset.
- Geneformer may simply fail to perturb genes outside its training distribution. Mitigation: report failure mode; the failure is itself evidence of memorization-driven Replogle performance.

---

### Phase 7 — Stronger Baselines and Additional Ablations

**Wall time:** 3-4 days.
**Compute:** ~$30 M4 + ~$50 Modal.
**Goal:** Add baselines that block reviewer "what about X?" objections.

**Tasks:**

P7.1 — **Trivial baseline #1: Predict-zero-shift.**
For each perturbation, return the control mean unchanged. Should produce F1 = 0 (zero predicted edges from overlay sampling). Confirms the overlay sampling pipeline's null behavior.

P7.2 — **Trivial baseline #2: Predict-ctrl-mean.**
Same as #1 but with a small Poisson noise floor. Confirms the random-bootstrap baseline.

P7.3 — **Random forest baseline.**
Per-target-gene scikit-learn `RandomForestRegressor` trained on ctrl cells. Same training data as ElasticNet. Hypothesis: nonlinearity doesn't help.

```python
# scripts/baseline_rf.py — uses the existing run_eval_local infrastructure
from sklearn.ensemble import RandomForestRegressor

rf_models = {}
for g_idx in range(n_g):
    feature_idx = [i for i in range(n_g) if i != g_idx]
    m = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0, n_jobs=-1)
    m.fit(ctrl_X[:, feature_idx], ctrl_X[:, g_idx])
    rf_models[g_idx] = m
# Predict in same propagation manner as ElasticNet
```

P7.4 — **Kernel regression baseline.**
A second non-linear: Gaussian-kernel ridge regression. Tests whether the linearity of ElasticNet is what matters or just its regularization.

P7.5 — **scGPT.**
Either re-attempt scGPT (per the existing `scripts/modal_run_scgpt_real.py`) and report results, or explicitly state in the paper's "Limitations" that scGPT was excluded due to in-silico-perturbation API failures.

P7.6 — **Ablations beyond CPA architecture:**

P7.6.a — **Overlay-sampling sensitivity.**
Sweep:
- Clip range: [0.001, 100], [0.01, 10] (current), [0.1, 5].
- Pseudocount: 0.001, 0.01 (current), 0.1.
- Noise model: Poisson (current), NB, no-noise.
3 × 3 × 3 = 27 combinations × ~10 min M4 each = 4.5h.

P7.6.b — **GIES partition-size sensitivity.**
p ∈ {10, 15, 20 (current), 30, 40, 50}. Per-condition F1 at each partition size. 6 settings × ~20 min = 2h.

P7.6.c — **Cells-per-perturbation sensitivity.**
n ∈ {25, 50, 100 (current), 200}. 4 settings × ~30 min = 2h.

P7.7 — **Add a third causal-discovery backend (optional): DCDI or NOTEARS-MLP.**
The handoff and proposal mention DCDI as "implemented in package as robustness check; not yet run." Run it. Provides a fifth backend that complements GIES (score), inspre (sparse inverse), PC (constraint), NOTEARS (continuous-linear).

**Exit criteria:**
- All baselines have entries in `results/baselines_extra/` with per-seed F1 for K562.
- Sensitivity sweep summaries in `results/ablation_overlay/`, `results/ablation_partition/`, `results/ablation_cells/`.
- Manuscript v2.22 commits with extended Table 1 (or new appendix table) covering all baselines.

**Dependencies:** Phase 1 (multi-seed) for fair CIs.

**Risks:**
- A new baseline could outperform deep models too. If RF F1 > 0.45 on K562, the headline shifts from "linear matches deep" to "any sklearn baseline matches deep" — strengthens the paper's core argument but requires text restructuring.

---

### Phase 8 — Engineering: Reproducibility, Environment, Tests

**Wall time:** 2-3 days.
**Compute:** None.
**Goal:** A fresh clone runs the appendix §C.4 commands end-to-end without surprises.

**Tasks:**

P8.1 — **Lock environment.**
- Write `pyproject.toml` with pinned dependencies (Python 3.9, gies 0.0.1, anndata 0.10.9, scanpy 1.10.3, scikit-learn 1.x, transformers 4.52.3, numpy 1.26.4, scipy 1.13.1, networkx 3.x, causal-learn 0.1.x).
- Provide `requirements-modal.txt` for Modal images (Python 3.11, same pins).
- Document the transformers source-patch step for STATE in `INSTALL.md`.

P8.2 — **Unify multiprocessing.**
Replace `fork` context in `scripts/run_eval_local.py` with `spawn` (matches Norman script). Validate locally on M4 + on Modal.

P8.3 — **One-command reproducibility.**
A `make repro` target (or `scripts/reproduce_paper.sh`) that:
1. Downloads K562 subset from Modal volume.
2. Runs vanilla evaluation across 10 seeds.
3. Runs calibrated evaluation across 10 seeds.
4. Runs STATE (skipped if no Modal account).
5. Runs Norman (skipped if data unavailable).
6. Runs Track B.
7. Generates all figures.
8. Compiles PDF (skipped if `tectonic` unavailable).

Estimated end-to-end wall: ~12h M4.

P8.4 — **Test suite expansion.**
Add tests for:
- `causalcellbench/calibration/*` — all three modes (already in Phase 4).
- `scripts/run_norman_local.py` — pipeline smoke test (jitter + sequential GIES).
- `scripts/phase9_rpe1.py` — after P0.1, smoke test that ElasticNet ≠ real means.
- Overlay sampling — covariance preservation property test.
- Multi-seed aggregation — CI calculation matches `scipy.stats.t.interval`.

Run `pytest tests/` end-to-end; target wall < 10 min on M4.

P8.5 — **Modal job consolidation.**
Replace `modal_run_gears.py`, `modal_run_cpa.py`, `modal_run_geneformer.py`, `modal_run_state_v3.py`, `modal_run_scgpt.py` with one configurable `modal_runner.py --model <name> --dataset <name> --seed <int>`. Reduces 5 one-off scripts to 1.

P8.6 — **Docker image for absolute reproducibility.**
A Dockerfile that pins the entire environment + reproducibility script. Useful for archival; submit alongside the bioRxiv preprint.

**Exit criteria:**
- `pyproject.toml`, `requirements-modal.txt`, `INSTALL.md`, `Dockerfile`, `Makefile` all present.
- `make repro` succeeds on a fresh clone (verified via worktree).
- `pytest tests/` passes 100%.
- Consolidated `modal_runner.py` replaces ≥3 one-off scripts.

**Dependencies:** Phase 0-7 (all code changes need to be in place before locking the env).

**Risks:**
- Modal image rebuild times balloon (~5-15 min per image). Mitigation: aggressive image caching, declared up-front.
- M4 vs Modal float reproducibility may differ in the 4th decimal place. Mitigation: document expected ranges, not exact equality.

---

### Phase 9 — Writing and Framing Revision

**Wall time:** 3-4 days.
**Compute:** None.
**Goal:** Restructure the manuscript to match the strengthened evidence base.

**Tasks:**

P9.1 — **Restructure intro.**
Open with the negative result: "Across 3+1 datasets × 5 VCMs × 4+ causal-discovery backends × 10 seeds, no deep model significantly exceeds a sparse linear regression on causal-recovery F1." This is the strong frame; STATE result becomes "the closest model to closing the gap but does not."

P9.2 — **Scope-of-claims table** in §3 (Methods).
A table listing what was tested: dataset × model × seed-count × backend. This pre-empts reviewer "what about X?" objections by being explicit.

P9.3 — **Reframe Track A vs Track B.**
Track A = "substitutability against GIES-on-real" → measures pipeline-level reproducibility of real-data signal.
Track B = "biological recovery against TRRUST" → measures biological signal recovery.
The two are not redundant; both must hold for a strong substitutability claim. Add a §3.3 explaining this.

P9.4 — **Reversed-edge as primary finding.**
Currently §7-style discussion. Promote to a §5 result subsection: 27-30% of deep-model false positives are reversed-direction true edges (per multi-seed JSON .reversed values). This is the clearest mechanistic finding and deserves prominence.

P9.5 — **STATE caveats and framing.**
After Phase 3, STATE is reported with mean ± std across 3+ seeds. Re-frame the abstract: STATE narrows but does not close the gap; the gap survives the 2025 foundation-model wave.

P9.6 — **Calibration framing per Δ4.2 outcome.**
Either three-mode toolkit (if Phase 4 implement) or single-mode (if drop).

P9.7 — **Limitations section reorganization.**
Six explicit limitations:
1. Seed budget (n=10 ≥ minimum credible but not n=100).
2. Gene-set scale (N=200 for GIES tractability, N=500-1000 via inspre).
3. Model roster (5 VCMs, not the entire 2024-2026 wave).
4. Track B coverage (TRRUST is broad-net, not K562-specific).
5. Geneformer pretraining contamination (quantified via held-out dataset, but not zero).
6. STATE cell budget (1200 cells, not full 38K).

P9.8 — **Conclusion rewrite.**
Three sentences:
1. The substitutability gap survives multi-seed, multi-dataset, multi-backend, multi-model evaluation including 2025 foundation models.
2. The gap decomposes into magnitude marginal (~40%, post-hoc fixable) and joint-structure residual (~60%, requires training-time intervention).
3. The benchmark is released as PerturbCausal, a pip-installable Python package.

P9.9 — **bioRxiv preprint preparation.**
Format the manuscript per bioRxiv's preferred LaTeX template (or convert to PDF-only submission). Cover letter naming the contribution.

P9.10 — **Update appendix.**
After all phases:
- §A.1 (Methods): updated hyperparameter table including RF, kernel, optional scGPT.
- §A.2 (Scope of claims): cross-reference Phase 0-7 outcomes.
- §A.3 (Multi-seed): n=10 update.
- §A.4 (Ablations): overlay, partition, cells sensitivity tables.
- §A.5 (Calibration modes 1-3): if Phase 4 implements.
- §A.6 (Geneformer memorization): held-out result.
- §A.7 (Reproducibility): updated commands.

**Exit criteria:**
- Manuscript v3.0 (camera-ready candidate for submission) commits with all phases' results integrated.
- bioRxiv-ready PDF.
- Cover letter for chosen venue.

**Dependencies:** All prior phases.

**Risks:**
- Manuscript expands beyond ICML 8-page limit. Mitigation: appendix is unlimited; bioRxiv has no page limit.
- Writing might oversell calibration despite Phase 4 outcome. Mitigation: self-check agent on §1, §4, §10 prose.

---

### Phase 10 — Submission and Release

**Wall time:** 1-2 days.
**Compute:** None.
**Goal:** bioRxiv preprint posted, code released, target venue submitted.

**Tasks:**

P10.1 — **bioRxiv preprint.**
Upload `submissions/biorxiv/PerturbCausal_v3.pdf` with cover letter. License: CC-BY 4.0. Subject: Genomics + Systems Biology.

P10.2 — **Code release.**
Push to `github.com/[user]/perturbcausal` (de-anonymized post-camera-ready):
- README with installation, quick-start, full reproduction.
- LICENSE (MIT).
- CITATION.cff.
- All scripts + tests + Dockerfile.

P10.3 — **PyPI package.**
`pip install perturbcausal` should install the calibration module + benchmark CLI. Standard `setup.cfg` or `pyproject.toml` packaging.

P10.4 — **Target-venue submission.**
Per the audit's recommendation: NeurIPS 2026 Datasets & Benchmarks (Jun 11 deadline). Submit via OpenReview. Wait for paired-anonymous reviews.

P10.5 — **Workshop fallback.**
If main-track NeurIPS rejects: submit to NeurIPS workshops (LMRL, GenBio, AI4D3) (Aug-Sep deadlines). All accept benchmark papers.

P10.6 — **Journal fallback.**
TMLR (rolling), Bioinformatics, or Genome Biology if NeurIPS path fails.

**Exit criteria:**
- bioRxiv DOI assigned.
- GitHub release tagged `v1.0.0`.
- OpenReview submission ID confirmed.

---

## 3. Critical Path and Gantt

```
Day 1-2:  Phase 0 (bug fixes, framing)             [BLOCKING]
Day 3:    Phase 1 (n=10 seeds, overnight)          [BLOCKING for stats claims]
Day 4-10: Phase 2 (RPE1 + Norman deep models)      [PARALLELIZABLE with 4, 5, 6, 7]
Day 4-7:  Phase 3 (STATE multi-seed)               [PARALLEL with 2]
Day 5-12: Phase 4 (Calibration modes 2-3)          [PARALLEL with 2, 3]
Day 5-7:  Phase 5 (Track B)                        [PARALLEL with 2, 3, 4]
Day 6-9:  Phase 6 (Geneformer memorization)        [PARALLEL with 2, 3, 4]
Day 8-12: Phase 7 (baselines + ablations)          [PARALLEL with 4]
Day 13-15:Phase 8 (engineering / repro)            [SOFT-BLOCKING for 9]
Day 13-17:Phase 9 (writing / framing)              [REQUIRES 2-7]
Day 18-19:Phase 10 (submission / release)          [REQUIRES 9]
```

**Critical path:** Phase 0 → Phase 1 → Phase 2 → Phase 9 → Phase 10 = 14 days minimum.
**Optimistic (full parallelization):** 12 days.
**Pessimistic (single-thread Modal queue):** 25 days.

**Suggested cadence:** Day 1-2 = bug fixes (Phase 0). Day 3 = launch all parallel phases overnight (Phase 1, 2, 3, 5, 6 on Modal). Days 4-7 = monitor + iterate. Days 8-12 = Phase 4 (Modes 2-3) + Phase 7 (baselines). Days 13-17 = Phase 8 + 9 (engineering + writing). Days 18-19 = Phase 10 (submission).

---

## 4. Risk Register

| # | Risk | Probability | Impact | Phase | Mitigation |
|---|---|---|---|---|---|
| 1 | RPE1 ElasticNet F1 after bug fix is < 0.10 | 30% | High | 0 | Cross-context drops to Norman + held-out only; restructure §5.7 |
| 2 | n=10 seeds shifts headline from "tied" to "small gap" | 25% | Medium | 1 | Rewrite §5.1 honestly; still a paper |
| 3 | Modal GPU exposure fails for STATE again | 70% | Medium | 3 | CPU fallback at 1200 cells (3-job split); documented |
| 4 | Calibration Mode 2 collapses predictions to trivial subspace | 30% | Medium | 4 | Shrinkage sweep; fallback to shrinkage=0 |
| 5 | Calibration Mode 3 destabilizes downstream GIES | 25% | Medium | 4 | Target near-zero frac matched to real; bounded sweep |
| 6 | Track B TRRUST F1 is uniformly < 0.05 | 60% | Low | 5 | Frame as directionality check; Spearman concordance is the real metric |
| 7 | Held-out dataset for Geneformer doesn't exist publicly | 40% | Medium | 6 | Use one of the recent Nat Methods 2024-2025 perturbation papers; if none, drop the contamination quantification |
| 8 | RF or kernel baseline outperforms deep models | 35% | Low | 7 | Strengthens the paper; rewrite |
| 9 | Deep model training on Norman fails (low cell-per-pert) | 50% | Medium | 2 | Report failure mode honestly; restrict Norman to ElasticNet result + Geneformer ISP (no retraining) |
| 10 | Multi-seed compute exceeds Modal credit | 20% | Low | 1, 2 | Have $300 budget; tracked via ccusage |
| 11 | Manuscript exceeds 8 pages even with appendix overflow | 50% | Low | 9 | Convert to D&B Track format (allows 10+ pages); bioRxiv is unlimited |
| 12 | NeurIPS D&B rejects | 40% | Medium | 10 | TMLR / workshop fallback per Phase 10.5/10.6 |
| 13 | Scoop (Arc Institute publishes their own substitutability eval) | 15% | High | 10 | bioRxiv first thing; if scooped, pivot to "first independent benchmark" |
| 14 | Reviewer demands ChIP-seq Track B | 30% | Low | (review) | Append in revision; TRRUST + Spearman defensible for first round |
| 15 | RF/RandomForest baseline regression chain hits memory limit | 15% | Low | 7 | Reduce n_estimators; M4 has 16GB |
| 16 | Modal `transformers/modeling_utils.py` source-patch breaks on a transformers point release | 30% | Medium | 8 | Pin transformers exactly at 4.52.3; document the patch step |
| 17 | Multi-seed Norman variance is huge due to small dataset | 50% | Low | 2 | Report wider CIs honestly; the qualitative ordering is what matters |
| 18 | Track B requires gene-symbol vs Ensembl ID harmonization | 70% | Low | 5 | `causalcellbench/data/trrust.py` already handles per existing implementation |

---

## 5. Decision Tree

```
Phase 0 → P0.1: corrected RPE1 ElasticNet F1
├── F1 ≥ 0.30: Phase 2 strong path
├── F1 0.10-0.30: Phase 2 weak path; RPE1 retains "different but baseline survives"
└── F1 < 0.10: Drop RPE1 from paper; lean on Norman + held-out

Phase 1 → P1.3: n=10 pairwise permutation
├── No new significance: keep "tied" headline (same as v2.15)
├── ElasticNet-vs-deep p < 0.05: rewrite to "small but real gap"
└── GRNBoost2 p < 0.001: retire all p-value caveats

Phase 2 → P2.1/P2.2: deep models on RPE1 + Norman
├── ElasticNet ties deep on both: "the tie is paradigm-independent"
├── ElasticNet ties deep on one: "the tie is paradigm-dependent" + describe which
└── ElasticNet loses on both: pivot to "the substitutability gap is real, but deep models can close it on RPE1/Norman" — major restructure

Phase 3 → Δ3.1: STATE multi-seed
├── F1 ≥ 0.45 with σ ≤ 0.02: STATE narrows the gap (keeps STATE as a co-headline)
├── F1 0.40-0.45 overlapping CIs: STATE ties ElasticNet (the gap survives 2025 foundation models)
└── F1 < 0.40: STATE is in the GEARS/CPA cluster (the gap is robust to scale)

Phase 4 → Δ4.1: implement vs drop Modes 2-3
├── Implement, Δ4.2 ≥ 70% gap closure: calibration is the methodological contribution
├── Implement, Δ4.2 40-70%: calibration is partial fix
└── Drop: paper is purely a benchmark + diagnostic paper; calibration is a single Mode 1 baseline

Phase 5 → Track B Spearman with Track A
├── Spearman > 0.7: Track A is a faithful biological proxy
├── Spearman 0.3-0.7: Track A and B measure overlapping but distinct things
└── Spearman < 0.3: Track A is pipeline reproducibility; Track B is biological; the substitutability question is more nuanced

Phase 6 → Geneformer contamination Δ
├── Held-out F1 > 0.40: Geneformer's pretraining helps but doesn't dominate
├── Held-out F1 0.30-0.40: ~25% of Replogle F1 is memorization
└── Held-out F1 < 0.30: Geneformer collapses on out-of-corpus data; treat as evidence that VCMs require in-distribution training

Phase 9 → Manuscript length
├── ≤ 8 pages: ICML / AI4Science 2027 format compatible
├── 9-12 pages: NeurIPS D&B format
└── > 12 pages: TMLR / journal only
```

---

## 6. Compute and Cost Budget

| Phase | Compute | Estimated $ | Notes |
|---|---|---|---|
| 0 | M4 only (~2h) | $0 | Bug fix + 1 RPE1 re-run |
| 1 | M4 ~5h + optional Modal | $0-$30 | Multi-seed bump |
| 2 | Modal ~30h | $100-$150 | GEARS+CPA+GF × RPE1+Norman |
| 3 | Modal CPU ~15h | $30-$50 | STATE multi-seed at 1200 cells |
| 4 | M4 ~15h | $0 | Calibration sweeps |
| 5 | M4 ~5h | $0 | Track B (TRRUST) |
| 6 | Modal CPU ~5h | $20 | Held-out Geneformer |
| 7 | M4 ~15h + Modal ~5h | $20 | Baselines + ablations |
| 8 | None | $0 | Engineering |
| 9 | None | $0 | Writing |
| 10 | None | $0 | Submission |
| **Total** | ~75h M4 + ~55h Modal | **$170-$270** | Conservative estimate |

ccusage current state (per statusline 2026-05-25): **5h block at 99%, 7d at 32%**, extra credit at 90% of $12,000 monthly. The fix plan's estimated additional spend ($170-$270) sits well inside the remaining monthly extra credit (~$1,259 remaining). No additional cash spend should be needed.

**Compute pacing.** Modal jobs should be batched overnight to avoid hitting the 5h block while interactive work is happening on the same day. Use the orchestrator pattern from `scripts/overnight_orchestrator.sh` (already present) with the addition of integration scripts per phase.

---

## 7. Acceptance Criteria per Phase

### Phase 0
- [ ] `git diff scripts/phase9_rpe1.py` shows ElasticNet model fit + prediction.
- [ ] `results/phase9_rpe1/phase9_results.json` shows `elastic_net_precision != 1.0`.
- [ ] Manuscript v2.16 commits with §3.5, §5.1, §5.2, abstract changes.

### Phase 1
- [ ] `results/multi_seed/aggregated_results.json` contains 10 seeds.
- [ ] Pairwise permutation p-values include values below 0.01.
- [ ] Table 1 caption no longer says "p<0.01" unless achievable.

### Phase 2
- [ ] 6 new prediction h5ad files: `{gears,cpa,geneformer}_predictions.h5ad` × {RPE1, Norman}.
- [ ] `results/multi_seed/aggregated_rpe1.json` and `aggregated_norman.json` exist.
- [ ] Manuscript §5.7 + §3.5 rewritten with real numbers.

### Phase 3
- [ ] `results/state/state_multiseed_f1.json` contains ≥ 3 seeds.
- [ ] STATE F1 reported as mean ± std.
- [ ] Abstract STATE phrasing updated.

### Phase 4 (Implement branch)
- [ ] `causalcellbench/calibration/covariance_preservation.py` exists with unit tests.
- [ ] `causalcellbench/calibration/sparsity_injection.py` exists with unit tests.
- [ ] `causalcellbench/calibration/__init__.py` exports `CalibrationPipeline`.
- [ ] `results/calibration_modes/` contains per-mode F1 numbers.
- [ ] Manuscript v2.19 with three-mode results table.

### Phase 4 (Drop branch)
- [ ] Manuscript abstract, §1, §4.5, §10 say "quantile-matching calibration" (singular).
- [ ] No "toolkit" or "three modes" language remains.

### Phase 5
- [ ] `results/track_b/trrust_per_model_f1.json` and `spearman_concordance.json` exist.
- [ ] Manuscript v2.20 with Track B row in Table 1.

### Phase 6
- [ ] A held-out post-Geneformer-pretraining dataset evaluated.
- [ ] `results/geneformer_memorization.json` reports Δ F1.

### Phase 7
- [ ] `results/baselines_extra/` contains RF, kernel, predict-zero, predict-ctrl-mean.
- [ ] `results/ablation_overlay/`, `ablation_partition/`, `ablation_cells/` exist.
- [ ] Extended Table 1 or appendix table with all baselines.

### Phase 8
- [ ] `pyproject.toml` pins all major dependencies.
- [ ] `make repro` succeeds on fresh git worktree.
- [ ] `pytest tests/` passes 100%.
- [ ] `modal_runner.py` replaces 5 one-off Modal scripts.
- [ ] `Dockerfile` builds and runs `make repro` successfully.

### Phase 9
- [ ] Manuscript v3.0 with restructured intro, scope-of-claims table, reversed-edge primary finding, calibration framing per Δ4.2, six explicit limitations, three-sentence conclusion.
- [ ] bioRxiv-ready PDF.
- [ ] Cover letter drafted.

### Phase 10
- [ ] bioRxiv DOI assigned.
- [ ] GitHub release `v1.0.0` tagged.
- [ ] NeurIPS D&B OpenReview submission ID confirmed.

---

## 8. File-by-File Change Manifest

This list enumerates every file expected to change. Files marked `[CREATE]` are new; `[MODIFY]` are edits; `[DELETE]` are removed.

### Manuscripts
- `[MODIFY] submissions/icml/CausalCellBench_ICML.tex` — Phase 0, 1, 2, 3, 4, 5, 6, 7, 9 incremental updates → v3.0.
- `[MODIFY] submissions/icml/CausalCellBench_ICML_Short4pp.tex` — Phase 9 condensed update for workshop-format option.
- `[MODIFY] submissions/icml/CausalCellBench_ICML_4pp_Supplementary.tex` — Phase 9 expanded with all new ablations.
- `[CREATE] submissions/biorxiv/PerturbCausal_v3.tex` — Phase 10, bioRxiv format.
- `[CREATE] submissions/cover_letter_NeurIPS_DB.md` — Phase 10.

### Scripts
- `[MODIFY] scripts/phase9_rpe1.py` — Phase 0 P0.1.
- `[MODIFY] scripts/run_eval_local.py` — Phase 8 P8.2 (fork → spawn).
- `[CREATE] scripts/baseline_rf.py` — Phase 7 P7.3.
- `[CREATE] scripts/baseline_kernel.py` — Phase 7 P7.4.
- `[CREATE] scripts/baseline_trivial.py` — Phase 7 P7.1, P7.2.
- `[CREATE] scripts/sensitivity_overlay.py` — Phase 7 P7.6.a.
- `[CREATE] scripts/sensitivity_partition.py` — Phase 7 P7.6.b.
- `[CREATE] scripts/sensitivity_cells.py` — Phase 7 P7.6.c.
- `[CREATE] scripts/modal_runner.py` — Phase 8 P8.5 (consolidates 5 scripts).
- `[DELETE] scripts/modal_run_gears_real.py` — replaced by modal_runner.
- `[DELETE] scripts/modal_run_cpa_real.py` — replaced by modal_runner.
- `[DELETE] scripts/modal_run_geneformer_real.py` — replaced by modal_runner.
- `[DELETE] scripts/modal_run_scgpt_real.py` and `.bak` — replaced by modal_runner.
- `[CREATE] scripts/reproduce_paper.sh` — Phase 8 P8.3.
- `[MODIFY] scripts/aggregate_multi_seed.py` — Phase 1 extend to 10 seeds.
- `[CREATE] scripts/track_b_run.py` — Phase 5 (wrapper around existing `track_b_biological_ground_truth.py`).
- `[CREATE] scripts/geneformer_heldout.py` — Phase 6.

### causalcellbench package
- `[CREATE] causalcellbench/calibration/covariance_preservation.py` — Phase 4 P4.1 (Implement branch).
- `[CREATE] causalcellbench/calibration/sparsity_injection.py` — Phase 4 P4.2 (Implement branch).
- `[MODIFY] causalcellbench/calibration/__init__.py` — Phase 4 P4.3 (Implement branch) — add `CalibrationPipeline`.

### Tests
- `[MODIFY] tests/test_calibration.py` — Phase 4 P4.5 (Implement branch).
- `[CREATE] tests/test_calibration_modes.py` — Phase 4 P4.5.
- `[CREATE] tests/test_phase9_rpe1.py` — Phase 0 + 8 P8.4 smoke test.
- `[CREATE] tests/test_overlay_covariance.py` — Phase 8 P8.4.
- `[CREATE] tests/test_multi_seed_aggregation.py` — Phase 8 P8.4.

### Environment / reproducibility
- `[CREATE] pyproject.toml` — Phase 8 P8.1 (or modify if exists).
- `[CREATE] requirements-modal.txt` — Phase 8 P8.1.
- `[CREATE] INSTALL.md` — Phase 8 P8.1 (includes transformers source-patch instructions).
- `[CREATE] Makefile` — Phase 8 P8.3 (`make repro`, `make test`, `make figures`).
- `[CREATE] Dockerfile` — Phase 8 P8.6.

### Results JSONs
- `[MODIFY] results/phase9_rpe1/phase9_results.json` — Phase 0 P0.1.
- `[MODIFY] results/phase9_rpe1/elastic_net_edges.json` — Phase 0 P0.1.
- `[MODIFY] results/multi_seed/aggregated_results.json` — Phase 1 P1.2.
- `[CREATE] results/multi_seed/aggregated_rpe1.json` — Phase 2 P2.3.
- `[CREATE] results/multi_seed/aggregated_norman.json` — Phase 2 P2.3.
- `[CREATE] results/state/state_multiseed_f1.json` — Phase 3 P3.2.
- `[CREATE] results/calibration_modes/{mode2,mode3,pipeline}_*.json` — Phase 4.
- `[CREATE] results/track_b/trrust_per_model_f1.json` — Phase 5.
- `[CREATE] results/track_b/spearman_concordance.json` — Phase 5.
- `[CREATE] results/geneformer_memorization.json` — Phase 6.
- `[CREATE] results/baselines_extra/{rf,kernel,trivial_zero,trivial_ctrl}_f1.json` — Phase 7.
- `[CREATE] results/ablation_overlay/sensitivity_summary.json` — Phase 7.
- `[CREATE] results/ablation_partition/sensitivity_summary.json` — Phase 7.
- `[CREATE] results/ablation_cells/sensitivity_summary.json` — Phase 7.

### Documentation
- `[CREATE] ARCHITECTURE_V3_FIX_PLAN.md` — this document.
- `[MODIFY] HANDOFF_PerturbCausal_2026-05-25.md` — Phase 10 update with completion status.
- `[MODIFY] CausalCellBench_Proposal_v5.md` — Phase 9, update with new findings.
- `[CREATE] CHANGELOG_v3.md` — Phase 10, version log.

### Figures
- `[MODIFY] figs/fig1.pdf` — Phase 1 (10-seed update).
- `[MODIFY] figs/fig2.pdf` — Phase 2 (cross-context overlay).
- `[MODIFY] figs/fig3.pdf` — Phase 5 (Track B addition).
- `[MODIFY] figs/fig4.pdf` — Phase 4 (3-mode calibration if implemented).
- `[CREATE] figs/fig5.pdf` — Phase 2 (RPE1 + Norman deep-model results).
- `[CREATE] figs/fig6.pdf` — Phase 7 (sensitivity ablations).

---

## 9. Verification and Self-Check Protocol

After each phase, run the following before committing:

1. **Self-check agents** per `~/config/projects/-Users-aayanalwani-virtual-cells/memory/feedback_self_check_agents.md`:
   - Logic review agent (off-by-one, shape mismatch, sign errors).
   - Requirements agent (does the change match what the manuscript claims).
   - Integration agent (does it break existing callers).
   - Data-distribution agent (no leakage, no implicit selection bias).

2. **Pre-flight checks** per `feedback_preflight_checks.md`:
   - AST parse all changed Python files.
   - Import-only smoke test.
   - Dry-run on a 5-gene test case before launching the full pipeline.

3. **JSON consistency check**: every numeric value reported in the manuscript must trace to a JSON in `results/`. A scripted `python3 scripts/verify_manuscript_numbers.py` checks this (created in Phase 8 P8.4).

4. **Git hygiene**: each phase is a single commit with a descriptive message; the v2-baseline tag is never modified.

5. **Reproducibility spot-check**: at the end of each phase, run `bash scripts/reproduce_paper.sh --quick` (a fast subset of `make repro`) to confirm nothing has regressed.

---

## 10. The "10/10 Version" — What Success Looks Like

After all 10 phases complete, PerturbCausal v3 has the following properties:

- **3+1 datasets evaluated**: Replogle K562 (multi-seed n=10), Replogle RPE1 (multi-seed n=5, deep models trained), Norman 2019 (multi-seed n=5, deep models trained), one held-out post-Geneformer-pretraining dataset.
- **5 VCMs**: GEARS, CPA, Geneformer V2, STATE, plus either scGPT (if successful) or explicit exclusion note.
- **4+ causal-discovery backends**: GIES, inspre, PC, NOTEARS, optionally DCDI.
- **2 ground-truth axes**: Track A (GIES-on-real) + Track B (TRRUST + Spearman concordance).
- **3-mode calibration pipeline** (if Phase 4 Implement branch): quantile + sparsity + covariance, with documented composability.
- **6+ baselines**: ElasticNet (current), RF, kernel, predict-zero, predict-ctrl-mean, GRNBoost2, technical-duplicate noise floor.
- **3+ ablation axes**: overlay sampling (clip, pseudocount, noise), GIES partition size, cells-per-perturbation.
- **Reproducibility**: pyproject.toml, Dockerfile, Makefile, `pytest tests/` 100%, end-to-end Reproducibility Script.
- **Honest stats**: n=10 seeds; minimum detectable F1 ≈ 0.013; paired permutation p down to 0.002.
- **Honest framing**: scope-of-claims table; no overstated significance; explicit limitations; reversed-edge as primary finding.
- **Release**: bioRxiv preprint, GitHub release v1.0.0, pip-installable PyPI package.

This is a NeurIPS D&B Track accept. It is also a defensible Bioinformatics or Genome Biology submission.

The undeniable result from the strongest version of this paper:

> Across 3+1 datasets, 5 VCMs (including the 2025 foundation models STATE and scGPT), 4 causal-discovery backends, and 10 seeds, no deep VCM significantly exceeds a sparse linear regression on causal-recovery F1 against either Track A (GIES-on-real) or Track B (TRRUST). The substitutability gap decomposes into a magnitude marginal (~40%, fixable post-hoc via quantile + sparsity calibration) and a joint-structure residual (~60%, fixable only via training-time covariance loss). The benchmark, multi-dataset multi-model artifact, calibration pipeline, and reproducibility infrastructure are released as the open-source PerturbCausal Python package.

---

## 11. Open Questions Logged for Resolution During Execution

1. **Held-out Geneformer dataset choice**: which 2024-2025 Perturb-seq paper is best for the contamination check? Candidates listed in Phase 6 P6.1; final choice during execution.

2. **scGPT**: re-attempt or formally exclude? Defer to Phase 7 P7.5.

3. **DCDI as fifth backend**: nice-to-have or required? Defer to Phase 7 P7.7.

4. **STATE on RPE1 + Norman**: ship in v3 or defer to v3.1? Defer to Phase 3 P3.4.

5. **bioRxiv timing**: post immediately after Phase 0 + Phase 1 (v2.17 with bug fix + n=10), or wait for Phase 9 v3.0? Recommended: immediately after Phase 1, then update at v3.0 to establish earliest possible priority.

6. **PyPI package name**: `perturbcausal` vs `causalcellbench`. Defer to Phase 10. Recommendation: `perturbcausal` (the rebranded name; matches manuscript title).

7. **Venue submission decision**: NeurIPS D&B (Jun 11) vs TMLR (rolling)? Recommendation: NeurIPS D&B first; TMLR fallback if rejected.

---

## 12. Sign-Off

This plan was generated 2026-05-25 by automated tooling in autonomous mode following the audit of the PerturbCausal v2.15 submission package. Approval to begin Phase 0 is the next user action. Phases 1-10 proceed autonomously per the user's standing instruction in `project config`.

Decision points and risk register entries above are designed to be self-resolving — the assistant makes the most reasonable choice and documents the assumption, per autonomous mode. The human reads the manuscript v3.0 + bioRxiv link when Phase 10 completes.

End of plan.
