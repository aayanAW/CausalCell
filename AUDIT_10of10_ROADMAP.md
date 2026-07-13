# PerturbCausal → 10/10: Audit + Roadmap

## Update — experiments run 2026-06-23 (Phase E, autoresearch-loop guards)

Two pre-registered/multi-seed experiments run locally; both gave **honest nulls that corrected earlier single-seed claims** (the multi-seed discipline did its job):

- **J2 covariance-regularizer λ-sweep (DISCARD).** Pre-registered grid λ∈{0…100}, val/test split, locked test touched once. Best-validation λ=100 gains only **+0.003 ± 0.035** test F1 over recon-only (3/5 seeds) — fails the multi-seed gate. Confirms the earlier single-λ finding rigorously: the +0.11 J2 gain is the _learned joint map_, not the covariance penalty. (`results/phase4/j2_lambda_sweep.json`, `preregistration_j2_lambda.md`.)
- **J1 TRRUST multi-seed (NULL — overturns the single-seed claim).** Across 5 seeds the GIES-on-real ground truth recovers TRRUST at **0.83 ± 0.70× chance** (per-seed 0.46/0.69/1.14/1.84/0.00) — _not_ robustly above chance. The seed-42 "2.4× lift" did **not** replicate. So the circularity is **not** closed; it remains an honest open limitation (cell-sampling-dominated at this scale, per the 0.045 tech-dup floor). Paper corrected accordingly. (`results/track_b/j1_trrust_multiseed.json`.)

## Workflow file read + methods employed (`~/Desktop/claude ultimate workflow/`)

This audit followed the user's "claude ultimate workflow" prompt files, read in full at session start:

| File                  | What it prescribes                                                                                                                   | How it was employed                                                                                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `README.md`           | A→H phase map; cross-model verification; audit-prompt index                                                                          | Used as the orchestration map; ran the two audit prompts below                                                                                                                  |
| `audit.md`            | Area-chair (ICML/NeurIPS/ICLR) "is the idea a 10/10?" brutal critique                                                                | Applied as the novelty/claim-validity/reviewer-2 lenses in the multi-agent audit                                                                                                |
| `AUDIT_CrossModel.md` | Whole-repo 6-dimension audit, "built to run as a deterministic multi-agent Workflow" with adversarial cross-verification → synthesis | Ran as a **9-dimension Workflow (18 agents, adversarial verify)**: novelty, claim-validity, stats, repro/code, number-honesty, figures, writing, reviewer-2, computational-gaps |
| `H_Paper_Writing.md`  | Submission bar: `scientific-writing`, `humanizer`, `peer-review`/`santa-method` self-check agents, citation verification             | Self-check agents (adversarial verify stage); humanizer prose pass (P4); citations web-verified before adding                                                                   |
| `paper-cowriter.md`   | Model-paper templating; 7 self-check agents; claim–evidence honesty; figure routing                                                  | Number-honesty + figure + claim-evidence dimensions; figures regenerated to camera-ready vector standard                                                                        |

Cross-model (Codex) leg of `AUDIT_CrossModel.md`: not run (codex plugin not invoked this session); the Claude-side multi-agent + adversarial-verify path was used and is noted as the fallback the file permits.

## Execution status (branch `fix/camera-ready-corrections`, 11 commits, all builds green, all tests pass)

**DONE — camera-ready (moves ~5 → ~7):**

- P0 data-integrity: Table 1 Edges column regenerated; GRNBoost2 3,119→2,327 (fabricated); RPE1 deflated bootstrap means → true point estimates (GEARS 0.414/CPA 0.450/ENet 0.437); GRNBoost2 significance un-misattributed (n=20 p<1e-4 vs n=5 floor); RPE1 810-vs-1,137 reconciled; invalid pairwise test deprecated; cosine direction fixed; CPA cov 0.997→0.996; GT range corrected.
- P1 stats: invalid STATE n=3 parametric CI removed (report raw per-seed); MDE corrected to include n=5 deep (0.022–0.027); 0.32-vs-0.42 ElasticNet inconsistency fixed.
- P2 framing: "first" claim tightened; "invariant"→"consistent across conditions tested"; 2 uncited competitors added (fernandez2026grnbreak arXiv 2605.04930, zinati2024groundgan Nat Commun 15:4055 — both web-verified) + GRN-simulator distinction paragraph.
- P3 figures: **Figure 1 regenerated** (Wong colorblind palette, correct seed counts, rotated labels, de-clipped) — the CRITICAL fig issue.
- P4 prose: abstract cut ~330→~210 words (mechanism-led, de-AI, removed invalid \S\ref); Conclusion split 1→3 paragraphs; "Six limitations" signpost removed; Track B zero-coverage disclosed honestly.
- P5 repro: Python version reconciled (Methods↔appendix).

**REMAINING — camera-ready polish (gets cleaner, not score-critical):** pipeline overview schematic (under-figured); per-seed strip/box plot; TOST equivalence test; fig3 legend overlap + F1 typography unify; em-dash/repetition audit; K≥1000 permutation band; Zenodo data hosting (needs an account — user action).

**DONE — journal-tier, fully local (moves ~7 → ~8):**

- **J7 identifiability theorem** — Proposition + proof that conditional independence is invariant under per-gene monotone calibration, so marginal quantile matching provably cannot close the gap (Corollary). Explains the empirical calibration null as an identifiability fact; adds the paper's first formal result (the declared theorem environments were unused). `\S sec:identifiability`.
- **TOST equivalence test** — deep models are statistically _equivalent_ to ElasticNet at Δ=0.045 (90% CIs within band: CPA +0.014, GEARS +0.020, Geneformer +0.0002). Converts "no significant difference" → demonstrated equivalence.
- **Per-seed strip plot** (Fig. `fig:seeds`) + permutation-count fix (500→10,000) + dropped unsupported Bonferroni claim.

**REMAINING — journal bucket (GPU-gated; a hard external blocker — not runnable in this session):** J2 covariance regularizer (trained model+tool), J3 multi-seed cross-dataset, J5 combinations, J6 genome-scale — all need Modal/HPC training. **J1** (external GRN gold standard — the #1 reject-reason fix) is local but a multi-step experiment (fetch GRN, coverage-optimized re-subset, ~45-min GIES re-runs) best done in a fresh full-context session. See below.

---

**Date:** 2026-06-23 · **Audited SHA:** `e89cb02` (main) · **Manuscript:** `submissions/icml/CausalCellBench_ICML.tex`
**Method:** 9-dimension multi-agent area-chair + hostile-reviewer audit (18 agents, adversarial cross-verification), per the `audit.md` / `AUDIT_CrossModel.md` / `H_Paper_Writing.md` workflow. Every finding below was verified against the manuscript, the result JSONs, or the code; refuted findings are dropped.

---

## Verdict

**Current honest acceptance trajectory: ~5/10** (top ML venue: reject-leaning; top methods journal: major revision, borderline desk-reject). Not the 7 I quoted before the deep audit — the audit surfaced data-integrity errors and a metric-saturation problem I had not caught.

Per-dimension: novelty 6.5 · claim-validity 6 · stats ~5 · repro 5 · number-honesty 7 · figures 5 · writing ~5 · **reviewer-2 4** (= the live reject risk) · computational-gaps 5.

- **Camera-ready bucket (no new compute) lifts it to ~7** — fixes errors, kills the credibility holes, sharpens framing.
- **Journal bucket (3 experiments + 1 artifact) lifts it to ~9** — breaks the circularity, adds a trained-model deliverable, generalizes the claim.
- **10/10 is reachable** only with the journal bucket _and_ the covariance-regularizer model (turns benchmark → method+tool, your stated preference).

---

## The spine: 3 issues that actually gate the score

Everything else is downstream of these.

### S1 — The headline GIES metric is resolution-saturated (CRITICAL, validity)

`overlay_resampled_real` = perfect real perturbation means pushed through the pipeline = the paper's stated upper bound. It scores **F1 = 0.422**, which ElasticNet (0.425) and Geneformer (0.432) **tie or beat** (`results/eval_results_n200_vanilla.json`). When perfect real means and a control-only linear baseline are indistinguishable, the GIES-at-N=200 instrument is **not resolving predictor quality** — it is dominated by the shared real-control covariance backbone the overlay sampler injects into _every_ condition and the ground truth alike (`run_eval_local.py:545`, clip `[0.01,10]` discards the 23× inflation before GIES sees it). The lead table therefore measures a metric ceiling, not a model ranking.

- **Fix:** demote the GIES/overlay table from the lead. Promote **inspre** (the one backend that _does_ separate: 12,280 vs 6,234 vs 0 edges) and the confound-free **ACE** pipeline as primary evidence. Add (a) the **zero-shift overlay null** (ratio≡1) at N=200 across 5 seeds as the operative chance reference, and (b) a **positive control** (a deliberately-degraded predictor must fall below the null) to prove the instrument can separate good from bad. Report the GIES table explicitly as a saturated-instrument result.

### S2 — Circular ground truth with ZERO biological validation (CRITICAL, reviewer-2)

Ground truth = the same backend (GIES/PC/NOTEARS/inspre) run on real data. The one defense — Track B (TRRUST) — is **vacuous**: `results/track_b/track_b_results.json` has `edges_in_gene_subset = 0`, `n_tfs = 0`, every Track-B F1 = 0.0. So "causal recovery" is recovery of an _unvalidated algorithm's output_, and a hostile reviewer reads the entire headline as circular.

- **Fix (highest leverage in the whole project):** add an **external K562 GRN gold standard** — ENCODE-rE2G, Gasperini 2019 enhancer-gene, or a 2024+ K562 CROP-seq curated network. Re-pick the gene subset to maximize curated-edge coverage (the current essential-gene subset is _why_ overlap is 0). Report absolute precision/recall + a TF→target metric against real biology, not just self-consistency. This single change converts the paper from "consistency benchmark" to "validated causal benchmark."

### S3 — Mechanism-light + wrong numbers + invalid stats (HIGH, multiple)

The negative result, stripped of framing, currently teaches little mechanism, and several load-bearing numbers don't survive contact with the JSONs (see Bucket 1, P0). The mechanism that _does_ exist — inflation ≡ mode-collapse identity, joint-structure residual, the 6-variant ablation, the held-out calibration null — is under-led and partly single-seed.

- **Fix:** lead with the mechanism; fix the numbers; build the **covariance-preserving training regularizer** the paper already proposes verbatim (Discussion l.378) — it does not exist in code yet (`covariance_preservation.py` is a 35-line post-hoc stub). Building+training it is the move from diagnosis → solution.

---

## Bucket 1 — Camera-ready (text / number / figure only; no new compute)

### P0 — Data-integrity errors (MUST fix before any submission; a reviewer/checker catches these)

| #   | Issue                                                                                                                                                                                                                         | Location                                                      | Fix                                                                                                                                                               | Verified |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| 1   | **RPE1 F1 are deflated bootstrap means, not point estimates; ordering wrong.** Paper: GEARS 0.321 / CPA 0.361 / ENet 0.340 / rand 0.347. True single-seed: GEARS **0.414** / CPA **0.450** / ENet **0.437** / rand **0.436**. | `tex:268`; `results/phase2_eval_rpe1.json` `conditions` block | Report the `conditions` point estimates; mark single-seed; fix the bootstrap (resample cells/perts, don't dedupe the edge list) or drop the CI.                   | ✅ JSON  |
| 2   | **GRNBoost2 edge count 3,119 is fabricated** — outside the entire observed range [2,212–2,460].                                                                                                                               | `tex:235` Table 1                                             | Replace with n=20 mean (~2,327) or seed-42 (2,440).                                                                                                               | ✅ JSON  |
| 3   | **Entire "Edges" column of Table 1 is stale** — matches no current JSON (Overlay 1,076 vs actual ~1,224, etc.).                                                                                                               | `tex:229-238`                                                 | Regenerate from `aggregated_results.json` at the row's n, or drop the column.                                                                                     | ✅ JSON  |
| 4   | **Invalid pairwise permutation test** — pools A∪B edges and repartitions; tests the wrong null, yields p≈1.0.                                                                                                                 | `run_eval_local.py:980-1016`                                  | Remove it, or replace with a bootstrap-over-cells/perturbations test. Cross-dataset cells are single-seed → run no pairwise test (already conceded "suggestive"). | ✅ code  |
| 5   | **GRNBoost2 "p=0.0625 not certifiable at n=5" contradicts the released n=20 result** (p<1e-4).                                                                                                                                | `tex:217,220`                                                 | Delete the n=5 caveat; state the certified n=20 result (p<1e-4).                                                                                                  | ✅ CSV   |
| 6   | **Figure 1 says "5 seeds" but Table 1 is n=20/n=5; not reproducible from repo code** (`make_figure1_multiseed.py` hard-codes 3 seeds, different palette).                                                                     | `figs/fig1.pdf`; generator                                    | Regenerate fig1 from one committed script reading `aggregated_results.json`; seed label must match the aggregation.                                               | ✅ image |
| 7   | **Two contradictory RPE1 GT edge counts** (1,137 main vs 810 appendix Table 8).                                                                                                                                               | `tex:268` vs `tex:696`                                        | Use one panel's count everywhere, or state the two panels differ and why.                                                                                         | ✅ JSON  |

### P1 — Statistical validity

- **STATE n=3 CI [-0.415, 0.783] crosses the F1≥0 support boundary and is bimodal** — parametric CI invalid. Drop the CI; report the three raw values [0.462, 0.041, 0.048] and make the non-replication point directly. (`tex:104,422`)
- **Headline "tie" is at n=5, MDE ≈ 0.022–0.027** (not the quoted 0.011, which is the n=20 baseline-only figure). State the deep-model MDE explicitly. (`tex:217,426`)
- **Add a TOST equivalence test** with a pre-registered margin (e.g. the 0.045 noise floor): convert "failed to reject" → "demonstrated equivalence." This is the single cleanest upgrade to the central positive claim. (`tex:217`)
- **K=30 permutation band** makes the 97.5-percentile unstable; near-band "above-chance" verdicts (GEARS·inspre·Norman) hinge on it. Bump to K≥1000 (cached, cheap) and widen near-band calls to "inside." (`phase7_multibackend_crossdataset.py:421`)
- **Bonferroni "over 21 pairs" absent from the artifact**, and 21≠15 for 6 conditions. Add the corrected column or drop the claim; fix the count. (`tex:203`)
- **Permutation count mismatch**: text says 500, code uses 10,000 / 1,000. (`tex:203`)
- **ElasticNet quoted as 4 different F1** (0.426/0.425/0.418/0.416) without signposting n. Standardize on n=20 = 0.418; annotate every context-specific value with its n. (`tex:217,310,384,418`)

### P2 — Framing / positioning (defends against the obvious rejections)

- **Cite the two uncited works that weaken originality** (HIGH): "When Does GRN Inference Break?" (arXiv 2605.04930, May 2026 — same causal<correlational puzzle) and **GRouNdGAN-Toolkit** (bioRxiv 2025.08.14.670294 — the "isn't this already done?" simulator-for-GRN-benchmarking objection). One sentence each, drawing the distinction. (`tex:130-141`)
- **Tighten the "first benchmark" claim** to the precise scope: "first to evaluate whether VCM-predicted _interventional_ data, through a _fixed_ causal algorithm, recovers the network that algorithm recovers from _real_ Perturb-seq." Footnote the adjacent 2026 VCM-utility benchmarks (bioRxiv 2026.04.23.719015; arXiv 2604.27646). (`tex:432`)
- **Re-center novelty on the MECHANISM**, not "linear beats DL downstream" (which reads as incremental over Ahlmann-Eltze, and the single-seed ElasticNet-beats-deep claim already didn't survive multi-seed). Lead with: prediction-accuracy (marginal) and causal-substitutability (joint) are _distinct measurable axes_; the gap lives in the joint structure; STATE (2025 foundation model) confirms it and its single-seed win fails to replicate. (`tex:100-127`)
- **Downgrade "backend- and dataset-invariant" → "consistent across the conditions tested"** — the cross-dataset cells are n=1; do not claim invariance the data can't support. Move the single-seed caveat into the abstract. (`tex:105,432`)
- **Flag CASCADE (Cao 2025) scoop risk** in the roster limitation: a causally-structured predictor is the natural candidate to close the gap. (`tex:426`)
- **Address the below-chance NOTEARS cells** (deep models below their own permutation floor) — a reviewer assumes a bug; state the thresholding/edge-budget mechanism or fix the cell. (`tex:288-293`)

### P3 — Figures (camera-ready vector standard)

- Regenerate **fig1** (P0 #6) + Wong colorblind-safe palette, consistent model→color across all figures.
- **Add the missing pipeline/overview schematic** (Figure 0): VCM preds + real → overlay/ACE → 4 backends → per-backend real-data E\* → F1 + chance band, with the shared-control confound drawn honestly. A 21-page benchmark paper with 5 figures and no overview is under-figured.
- **Add a per-seed strip/box plot** of the multi-seed F1 distribution — the "no significant difference" headline currently rests on CI numbers with no visual.
- fig3 legend overlaps the Real-Data bar; fig2(c) per-perturbation vs pooled scope needs an on-panel label; fig_backend needs an "n/c" marker for the omitted RPE1·PC ElasticNet point; unify F1 typography ($F_1$).

### P4 — Prose / anti-AI (the `humanizer` pass)

- **Abstract is ~400 words, caveat-dominated, contains a literal `\S\ref` cross-ref.** Cut to ~180–200: question → headline negative result → mechanism → calibration null → release. (`tex:100-108`)
- **Conclusion is one ~250-word wall** carrying 5 claims. Split to lead sentence + 2–3 short paras. (`tex:432`)
- Strip AI tells: em-dash overuse (cap 1/sentence), "not X but Y" negative parallelism, rule-of-three triads, intensifiers ("striking", "the point is", "in the sense that matters"), count-then-enumerate signposts ("Six limitations are noted"), code identifiers in body prose, the 10-citation single-bracket dump (`tex:135`).
- De-duplicate the cross-dataset two-exception accounting (restated ~4× verbatim). State once in Results.
- **Fix cosine claim** (number-honesty): paper says all three deep models have inter-perturbation cosine "elevated above" real, but GEARS (0.075) and Geneformer (0.059) are _below_ real (0.084); only CPA (0.845) is elevated. (`tex:377`)
- Minor: CPA covariance 0.997→0.996; K562 GT-edge range "1,080–1,249" → actual [1,106–1,230]. (`tex:247,217`)

### P5 — Reproducibility

- **Host the data artifacts** (slim h5ads, 4 prediction h5ads, calibrated preds, gene lists, gies caches) on Zenodo/HF with a DOI; add the download command before the verbatim repro block. Currently every repro command depends on gitignored, untracked files → reproducibility is broken for a stranger. (`tex:941-972`)
- Reconcile Python/lib versions between Methods (3.10) and appendix (3.9/3.11); ship a lockfile.
- Drop/rename the shuffle-artifact "legacy AUPRC"; make the package `gies_runner` the single source of truth (the eval script inlines a divergent jittered copy); regenerate or rename the stray seed-2147483647 `eval_results_n200.json`.

---

## Bucket 2 — Journal (new compute / new artifacts; the path to 9–10)

| #   | Experiment / artifact                                                                                                                                                                                                                               | What it buys                                                                                                                                  | Rough compute            | Track                |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | -------------------- |
| J1  | **External K562 GRN gold standard** (ENCODE-rE2G / Gasperini / curated), subset re-picked for coverage; absolute P/R + TF-target metric                                                                                                             | **Breaks the circularity** (S2) — the single highest-leverage item; kills the #1 reject reason                                                | <1 GPU-hr, no retraining | both                 |
| J2  | **Covariance-preserving training regularizer**: `L_recon + λ‖Cov(pred)−Cov(real)‖_F` (+ optional precision-sparsity) on CPA; sweep λ, retrain, run overlay→GIES/inspre, report vs the 0.414 alt-loss bar and ElasticNet; ship checkpoint on HF/PyPI | **Benchmark → method+tool** (your stated preference); directly tests the paper's own central hypothesis                                       | ~1–2 GPU-days            | journal              |
| J3  | **Multi-seed cross-dataset** (RPE1, Norman) — currently all n=1; the "invariance" claim is unsupported without it                                                                                                                                   | Lets the generality claim stand; removes the "single-seed" fatal absence                                                                      | ~hours/condition         | journal              |
| J4  | **2025-26 model roster**: scGPT (markets GRN inference), scFoundation, a diffusion VCM (scPPDM); re-embed STATE to ≥5 seeds                                                                                                                         | Pre-empts "you tested obsolete models"; STATE alone (n=3) is uninformative                                                                    | ~hours/model GPU         | both (1–2 cam-ready) |
| J5  | **Combination perturbations** (re-include the ~135 dropped Norman doubles as held-out): train on singles, predict unseen pairs, run causal pipeline vs ElasticNet                                                                                   | The marquee use case justifying VCMs over a linear baseline that _cannot_ extrapolate to unseen pairs — strongest "when do VCMs help?" result | ~1–2 GPU-days            | both                 |
| J6  | **Genome-scale head-to-head** N∈{500,1000,2000+} via inspre/precision backends (fix the N=500 ACE collapse with a regularized precision estimator first)                                                                                            | Verdict could _flip_ at scale where the linear edge vanishes; tests generality                                                                | ~10–30 CPU-hr/condition  | journal              |
| J7  | **Identifiability theory**: prove for linear-Gaussian SEMs that a predictor preserves the CPDAG iff it preserves precision-matrix support (not the marginal); corollary bounds edge error by off-diagonal support norm                              | Converts an empirical observation into a _theorem_; the preamble already declares theorem environments but the paper has zero theorems        | ~0 compute               | journal              |

**Priority order for the journal version:** J1 (circularity) → J2 (the model/tool) → J3 (multi-seed) → J5 (combinations) → J4/J6/J7.

---

## What is NOT a problem (refuted or already-handled — don't waste effort)

- **Leakage:** genuinely closed — real perturbation values never enter prediction/training (verified in code).
- **Headline numbers:** Table 3 (backend robustness), STATE per-seed, calibration sweep, inflation diagnostics, inspre probe all reproduce **exactly** from the JSONs.
- **Calibration honesty:** the held-out 160/40 null result is the paper's cleanest finding and is reported straight (no null reframed as a win).
- **The single-seed "ElasticNet beats deep" overclaim** was already retracted to "statistically tied" — honest.
- The calibration-module-as-novelty overclaim exists only in `CausalCellBench_Explanation.md`, not the paper (the paper already says "the calibration itself is a standard technique"). Fix the explainer, not the manuscript.

---

## Recommended sequencing

1. **Now (mandatory):** Bucket 1 P0 + P1 — fix the errors and invalid stats. Non-negotiable before any submission at any quality bar.
2. **Camera-ready submission (→7):** + P2 framing, P3 fig1+overview, P4 humanizer, P5 data hosting. Submit to TMLR / NeurIPS D&B / a workshop with honest scope.
3. **Journal version (→9–10):** J1 first (circularity), then J2 (the trained covariance-regularized model), J3, J5. This is the version that clears a top methods journal.
