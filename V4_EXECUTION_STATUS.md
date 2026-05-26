# V4 Execution Status

**Live status:** check `logs/v4_state.json`
**Multi-seed progress:** `ls results/multi_seed/eval_results_n200_s*.json | wc -l` (target 20)
**Orchestrator:** `tmux attach -t perturbcausal-v4` (Ctrl-b d to detach)

## Completed (committed to v4-execution branch)

- [x] **Phase -1 Lit audit + bibliography**: Added 15 missing bibitems (STATE Adduri 2025, Wong 2025, Csendes 2025, Wei 2025 Nature Methods, Systema Viñas-Torné 2025 Nat Biotech, Bendidi 2024, PertEval-scFM, GARM, Li 2025, CASCADE Cao 2025, Park 2026, BEELINE Pratapa 2019, Kernfeld 2024, Scribe Qiu 2020). §2 Related Work rewritten with two new paragraphs covering recent prediction benchmarks + causal-from-Perturb-seq methods.

- [x] **Phase 0.1 RPE1 pipeline bug**: `scripts/phase9_rpe1.py` lines 286-287 fixed. ElasticNet now actually fitted on RPE1 ctrl cells and propagated. Buggy cache moved to `.bug_v0`. Local re-run blocked on missing `data/rpe1_real.h5ad`.

- [x] **Phase 0.2 framing fixes**: GRNBoost2 "p<0.01" → achievable wording. STATE 320-cell caveat added to abstract. "Calibration toolkit" → "quantile-matching calibration" + repositioned as gap-decomposition diagnostic.

- [x] **Phase 4 calibration code**: `causalcellbench/calibration/` now exports `QuantileCalibrator` + `CovariancePreservingCalibrator` (Mode 2, Σ_pert per V4 H7) + `SparsityInjectionCalibrator` (Mode 3, textbook soft-threshold per V4 H8) + `CalibrationPipeline` (6-permutation composer). Smoke-tested.

- [x] **Phase 4 sweep**: 27 calibrated h5ad files generated for {gears, cpa, geneformer} × {q, c, s, sqc, scq, qsc, qcs, csq, cqs} with 80/20 perturbation train/test split (V4 H6 leakage fix). Manifest in `results/calibration_modes/sweep_manifest.json`. Real finding: composition order matters dramatically — `q`-late orderings approach real |FC|>0.5 fraction (9.3%), `q`-early collapse to zero.

- [x] **Phase 5 Track B (TRRUST)**: executed. Spearman concordance Track A ↔ Track B = **0.30** (n=5 models). Only 1 TRRUST edge overlaps with the 200-gene K562 subset (low TF coverage). `results/track_b/track_b_results.json`.

- [x] **Phase 8 engineering**: `pyproject.toml`, `requirements-modal.txt`, `Makefile`, `Dockerfile`, `INSTALL.md`, `README.md` (PerturbCausal-branded), `CITATION.cff`, fork→spawn fix in `run_eval_local.py`. PyPI name `perturbcausal` AVAILABLE.

- [x] **Phase 9 PDF + bioRxiv stage**: manuscript v3.0 PDF built (~233 KiB). `submissions/biorxiv/PerturbCausal_v0_20260525.pdf` staged. Cover letter drafted.

- [x] **Phase 10 USER_ACTIONS.md**: written. Manual steps documented (bioRxiv upload, GitHub release, PyPI publish, TMLR submission). PyPI name check confirmed `perturbcausal` available.

## In progress

- [ ] **Phase 1 multi-seed n=20**: background `run_multi_seed.py` running. 16/20 per-seed JSONs done. ETA ~30-40 min.
- [ ] **Phase 7 baselines**: RF + trivial baselines running in background. ~20 min.
- [ ] **Phase 4 GIES eval**: orchestrator will run this after multi-seed completes (~2.5h for all 27 conditions).

## Deferred (need Modal engineering)

- [ ] **Phase 2 deep models on RPE1 + Norman**: existing `modal_run_{gears,cpa,geneformer}_real.py` are K562-hardcoded. Building a unified `modal_runner.py --model X --dataset Y` is ~1-2 days of work.
- [ ] **Phase 3 STATE multi-seed at 1200 cells**: `modal_run_state_v3.py` needs `--n-cells-per-pert` flag.
- [ ] **Phase 6 Geneformer contamination**: V4 H16 already pivoted to "cell-line holdout" via RPE1 comparison. Needs Phase 2 RPE1 deep-model results.

## Pending

- [ ] **Phase 11 reproducibility check**: scheduled at orchestrator tail.

## Key claims & where they live now

| Claim | Location | Status |
|---|---|---|
| ElasticNet ties deep VCMs on K562 N=200 | `results/multi_seed/aggregated_results.json` | n=5 done; n=20 in progress |
| GRNBoost2 dramatically worse but n=5 p-floor 0.0625 | manuscript Table 1 caption | ✓ committed |
| 27-30% reversed-edge fraction | manuscript §5 Reversed-direction failure mode | ✓ promoted to §5 subsection |
| STATE F1=0.462 (single-seed, 320 cells) | `results/state/state_f1_summary.json` + abstract | ✓ caveat added |
| Substitutability gap decomposes magnitude/joint | manuscript §3 contributions + §5 Discussion | ✓ committed |
| Bibliography current as of 2026-05-25 | manuscript bibliography | ✓ 15 new bibitems |
| Scope-of-claims explicit | manuscript Table 2 sec:scope | ✓ committed |
| Limitations expanded to 6 items | manuscript §6 sec:limitations | ✓ committed |
| Conclusion compacted to single paragraph | manuscript §7 | ✓ committed |

## How to take this from here

1. Wait for multi-seed n=20 to finish (~30-40 min).
2. Orchestrator auto-runs Phase 4 GIES (~2.5h).
3. Orchestrator auto-runs Phase 11 repro check.
4. Upload `submissions/biorxiv/PerturbCausal_v0_20260525.pdf` to bioRxiv (manual, ~10 min).
5. Tag GitHub release `v1.0.0` (manual one-liner; see `submissions/USER_ACTIONS.md`).
6. Decide venue: TMLR (rolling) or NeurIPS workshops (Aug-Sep).
7. Optional: build `modal_runner.py` to un-defer Phase 2/3/6 for a follow-up bioRxiv v1.
