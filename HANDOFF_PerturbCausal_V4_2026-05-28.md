# Handoff: PerturbCausal — V4 Audit-Driven Overhaul

**Date:** 2026-05-28 (autonomous execution session)
**Status:** active (~80% complete; 3 phases deferred on Modal engineering)
**Last session model:** automated
**Branch:** `v4-execution` (35 commits ahead of `main`, NOT pushed)

## Current State (1-paragraph summary)

PerturbCausal benchmarks whether virtual cell model (VCM) predicted Perturb-seq can substitute for real Perturb-seq in downstream causal GRN recovery. This session executed the V4 plan — an audit-driven overhaul that found and addressed 25 holes in the prior V3 plan. The headline result changed: with a proper 160/40 perturbation train/test split, post-hoc calibration is shown to NOT close the substitutability gap (it hurts every model on held-out data), overturning the v2.15 "calibration recovers ~40%" claim. Multi-seed was bumped to n=20 (GRNBoost2 now significantly worse p<0.0001; new finding ElasticNet < Overlay-Real p=0.036). 15 missing citations added via Consensus MCP literature audit. The manuscript PDF is rebuilt and submittable at its current honest scope. Three phases (deep models on RPE1/Norman, STATE multi-seed, Geneformer contamination) are deferred because they need Modal engineering and/or missing data. No jobs running; working tree clean.

## What's Running / In Progress

Nothing. No tmux sessions, no background processes, no Modal jobs. `pgrep -f 'orchestrate|run_multi|phase_4'` returns nothing. Working tree clean.

## Completed Tasks

- [x] **Phase -1** Literature audit + 15 bibitems (Consensus-verified). §2 Related Work rewritten.
  - Key files: `submissions/icml/CausalCellBench_ICML.tex` (bibliography + §2)
- [x] **Phase 0.1** RPE1 phase9 pipeline bug fixed in code.
  - Key files: `scripts/phase9_rpe1.py` (lines ~279-330 now fit real ElasticNet)
  - Decision: prior code fed real pert_means into both conditions → precision=1.0 artifact
- [x] **Phase 0.2** GRNBoost2 stat phrasing, STATE caveat, calibration-toolkit→singular.
  - Key files: `submissions/icml/CausalCellBench_ICML.tex` (abstract, Table 1 caption, §5.1)
- [x] **Phase 1** Multi-seed n=20. ElasticNet/Overlay/GRNBoost2 at n=20; deep models n=5.
  - Key files: `results/multi_seed/aggregated_results.json`, `scripts/run_multi_seed.py`
- [x] **Phase 4** Calibration as diagnostic — 3 modes, 6 orderings, train/test split, GIES eval.
  - Key files: `causalcellbench/calibration/{pipeline,covariance_preservation,sparsity_injection}.py`, `scripts/v4/phase_4_calibration_{sweep,gies,baseline_uncal}.py`, `results/calibration_modes/{f1_per_ordering,vanilla_baseline_f1,sweep_manifest}.json`
- [x] **Phase 5** Track B (TRRUST). Spearman A↔B = 0.30.
  - Key files: `scripts/track_b_biological_ground_truth.py`, `results/track_b/track_b_results.json`
- [x] **Phase 7** RF + ctrl-mean + zero baselines.
  - Key files: `scripts/v4/phase_7_baselines_ablations.sh`, `results/baselines_extra/*.json`
- [x] **Phase 8** Engineering: pyproject, Dockerfile, Makefile, INSTALL, README, CITATION, fork→spawn.
- [x] **Phase 9** Manuscript v3.0: scope-of-claims table, reversed-edge §5 promotion, 6 limitations, n=20 Table 1, calibration-sweep paragraph. PDF rebuilt (235 KiB).
- [x] **Phase 10** USER_ACTIONS.md + PyPI name check (`perturbcausal` AVAILABLE).
- [x] **Phase 11** Reproduction worktree script (stub; branch-conflict, trivial fix needed).

## Pending Tasks (priority order)

1. **Restore `data/rpe1_real.h5ad`** — blocks Phase 0 RPE1 rerun + Phase 2 RPE1 deep models.
   - Source: `/Volumes/STORAGE/virtual-cells-archive/` or GEO GSE221321.
2. **Phase 2: deep models on RPE1 + Norman** — un-defer. Norman data IS on Modal volume (`ccbench-data:/raw/norman2019.h5ad`); RPE1 missing.
   - Blocked by: no unified `modal_runner.py`. Existing `scripts/modal_run_{gears,cpa,geneformer}_real.py` are K562-hardcoded (input path `/data/raw/k562.h5ad` at line ~105 of each).
   - Work: build `modal_runner.py --model X --dataset Y` handling per-dataset perturbation column (Replogle `gene`, Norman `perturbation_name`). ~1-2 days.
3. **Phase 3: STATE multi-seed 1200 cells** — extend `scripts/modal_run_state_v3.py` with `--n-cells-per-pert`, add `--seed` to `scripts/state_eval_gies.py`. ~$84 Modal CPU.
4. **Phase 6: Geneformer contamination** — depends on Phase 2 RPE1 deep models (cell-line holdout). No post-Genecorpus-V2 public dataset exists (V4 H16).
5. **User: `git push origin v4-execution`** (35 commits ahead) + bioRxiv upload + venue submission.

## Key Files & Locations

| Purpose | Path | Notes |
|---|---|---|
| V4 audit (the real plan) | `ARCHITECTURE_V4_AUDIT_AND_UPGRADE.md` | 25 holes + 12-phase V4 plan |
| V3 plan (superseded) | `ARCHITECTURE_V3_FIX_PLAN.md` | historical |
| Final status | `V4_EXECUTION_STATUS.md` | what's done / deferred |
| Manuscript | `submissions/icml/CausalCellBench_ICML.tex` | 235 KiB PDF |
| bioRxiv stage | `submissions/biorxiv/PerturbCausal_v2_*.pdf` | latest |
| User actions | `submissions/USER_ACTIONS.md` | manual steps |
| Calibration modules | `causalcellbench/calibration/` | pipeline + 3 modes |
| Phase 4 sweep | `scripts/v4/phase_4_calibration_sweep.py` | 27 calibrated h5ad |
| Phase 4 GIES eval | `scripts/v4/phase_4_calibration_gies.py` | F1 per ordering |
| Phase 4 vanilla baseline | `scripts/v4/phase_4_baseline_uncal.py` | head-to-head |
| Multi-seed n=20 | `results/multi_seed/aggregated_results.json` | 20 seeds |
| Calibration F1 | `results/calibration_modes/f1_per_ordering.json` | 27 conditions |
| Vanilla baseline F1 | `results/calibration_modes/vanilla_baseline_f1.json` | test split |
| V4 orchestrator | `scripts/v4/orchestrate.sh` + `phase_*.sh` | resumable |
| Eval pipeline | `scripts/run_eval_local.py` | fork→spawn applied |
| RPE1 pipeline (fixed) | `scripts/phase9_rpe1.py` | needs rpe1 data |

## Critical Decisions & Why

- **Calibration reframed from "fix" to "negative diagnostic".** With train/test discipline (V4 H6), calibration hurts every model on held-out data (GEARS −0.014, CPA −0.007, Geneformer −0.029). The v2.15 claim that calibration recovers ~40% was an in-domain single-seed artifact. The honest finding is stronger: the gap is in joint structure, requiring training-time intervention. Alternative (keep the positive claim) rejected — it doesn't survive a held-out test.
- **n=20 not n=10.** V4 audit H5 showed n=10 MDE ≈ 0.011 was overconfident in V3. n=20 → MDE ≈ 0.008, and it surfaced the ElasticNet < Overlay-Real gap (p=0.036) that smaller budgets miss.
- **Phase 2/3/6 deferred, not faked.** A unified `modal_runner.py` is real engineering; faking it with smoke stubs would pollute results. The Scope-of-claims table in the manuscript is explicit that RPE1/Norman rest on the ElasticNet baseline alone for now.
- **Deep models stuck at n=5.** Cached prediction h5ad files (`data/model_outputs_real_n200/{gears,cpa,geneformer}_predictions.h5ad`) are bound to the seed-42 gene subset. New seeds use different gene subsets, so model predictions can't be reused. Un-deferring requires per-seed model retraining (Modal).

## Known Issues / Gotchas

- **`data/rpe1_real.h5ad` is missing** locally and on the Modal volume. `data/rpe1_raw/` is empty. Phase 0 + Phase 2-RPE1 are blocked on this.
- **GEARS predictions contain small negative values** → `log2` produces NaN. `phase_4_calibration_sweep.py` clips pred ≥ 0 + `np.nan_to_num`. Do NOT remove the clip.
- **`sample_perturbed_cells_overlay(ctrl_X, mean_pred, n_cells, seed=0)`** takes 4 args, NOT 5. Earlier V4 code passed ctrl_mean + rng object → TypeError. Fixed.
- **`run_multi_seed.py --extra-args` needs `=` syntax**: `--extra-args="--skip-random --partition-size 20"`. Space form is eaten by argparse.
- **`run_multi_seed.py` default `--model-outputs-dir` is None** → new seeds skip deep models. Only ElasticNet/Overlay/GRNBoost2 ran for the 15 new seeds.
- **Orchestrator heredoc clobbers calibration modules.** `phase_4_calibration.sh` writes the module files via heredoc; it now skips overwrite if they exist. The committed `causalcellbench/calibration/pipeline.py` is the V4 version (with `_Log2FCQuantileMatcher`); do NOT let the V3 stub (imports `quantile_calibration`, wrong name) overwrite it.
- **DO NOT use `multiprocessing.Pool(fork)` in Modal** (deadlocks). `run_eval_local.py` now uses spawn.
- **Modal `gpu="A10G"` does not reliably expose CUDA** on this account (5 failed attempts historically). STATE runs CPU.

## Environment

- Python: 3.9 via `.venv_gears` (local), 3.11 (Modal)
- Key pins: transformers==4.52.3 (STATE), numpy==1.26.4, scipy==1.13.1, anndata==0.10.9, gies==0.0.1, scanpy==1.10.3
- Modal: account `alwaniaayan6`, volume `ccbench-data` (has raw/k562*, raw/norman2019.h5ad, state_cache; NO rpe1)
- Remote: `origin https://github.com/alwaniaayan6-png/CausalCell.git` (NOT pushed)
- Tectonic for PDF builds

## Second Brain

`~/Vault/` updated this session — log entry appended for 2026-05-28 with the V4 calibration-negative-result, n=20 finding, deferred phases.

- `~/Vault/config.md` — wiki schema
- `~/Vault/index.md` — wiki state
- `~/Vault/log.md` — last entry = this V4 session
- `~/Vault/wiki/projects/CausalCellBench.md` — project page (read fully)

## Reference Materials — MUST READ FULLY

### User's Global Instructions (READ FIRST)
- `project config` — autonomous mode, don't-compromise, research-grade standards, novelty-verification protocol.
- `~/config/projects/-Users-aayanalwani-virtual-cells/memory/MEMORY.md` — pre-flight checks rule, self-check agents rule, lab independence.

### Architecture & Design Docs
- `ARCHITECTURE_V4_AUDIT_AND_UPGRADE.md` — **READ FULLY**. The 25-hole audit + 12-phase V4 plan. This is the authoritative plan.
- `V4_EXECUTION_STATUS.md` — **READ FULLY**. What's done vs deferred.
- `ARCHITECTURE_V2.md` — original v2 pipeline + statistical framework.
- `HANDOFF_PerturbCausal_2026-05-25.md` — prior handoff (v2.15 submission package).

### Literature (fetch + read; verified via Consensus this session)
- Ahlmann-Eltze & Huber, *Nature Methods* 2025 (106 cites) — linear baselines beat DL on per-gene; PerturbCausal extends to causal recovery.
- Adduri et al. (STATE), bioRxiv 2025 (77 cites) — the 2025 foundation model.
- Brown et al. (inspre), *Nature Communications* 2025 — sparse-inverse causal backend.
- Mejia et al., arXiv 2506.22641 2025 — mode collapse; reconciled with inflation.
- CASCADE (Cao 2025) + Park 2026 — adjacent causal-from-Perturb-seq; scoop risk.
- Wei 2025 *Nat Methods*, Wong 2025 *Bioinformatics*, Systema (Viñas-Torné 2025 *Nat Biotech*), Csendes 2025 BMC Genomics — recent prediction benchmarks.
- BEELINE (Pratapa 2019 *Nat Methods*) — GRN inference benchmark.
- Replogle 2022 *Cell*, Norman 2019 *Science* — datasets.

### Existing Codebase (read before editing)
| Purpose | Path | Read fully? |
|---|---|---|
| Eval pipeline | `scripts/run_eval_local.py` | YES |
| Multi-seed runner | `scripts/run_multi_seed.py` | YES |
| Calibration pipeline | `causalcellbench/calibration/pipeline.py` | YES (V4 version) |
| Phase 4 sweep | `scripts/v4/phase_4_calibration_sweep.py` | YES |
| Phase 4 GIES | `scripts/v4/phase_4_calibration_gies.py` | YES |
| RPE1 (fixed) | `scripts/phase9_rpe1.py` | YES |
| Manuscript | `submissions/icml/CausalCellBench_ICML.tex` | YES |

### Data & Artifacts
- K562 slim: `data/k562_subset_n200_slim.h5ad` (33 MB)
- Model predictions: `data/model_outputs_real_n200/{gears,cpa,geneformer}_predictions.h5ad`
- Calibrated: `data/model_outputs_phase4/*.h5ad` (27 files)
- Real-data GIES GT: `results/gies_cache/n200_s42_p20/real_data_edges.json`
- Multi-seed n=20: `results/multi_seed/`
- RPE1: **MISSING** (`data/rpe1_real.h5ad` not present)

## How to Resume

1. Read `project config` FULLY.
2. Read `~/Vault/` (config.md, index.md, last log entry, project page).
3. Read this handoff fully.
4. Read `ARCHITECTURE_V4_AUDIT_AND_UPGRADE.md` + `V4_EXECUTION_STATUS.md` fully.
5. Read cited papers not known cold (Ahlmann-Eltze, STATE Adduri, CASCADE, Park 2026, Systema).
6. Explore codebase files marked YES.
7. Verify no jobs: `pgrep -f orchestrate; tmux ls; modal app list`.
8. Confirm git: `git -C "/Users/aayanalwani/virtual cells" log --oneline v4-execution ^main | wc -l` should be ≥35.
9. THEN start next pending task (RPE1 data restore → Phase 2 modal_runner).

**Do NOT start coding before steps 1-8.**

## Self-Check Agents — MANDATORY

After ANY non-trivial change: spawn Opus self-check agents (logic review, requirements vs V4 plan, integration, data-distribution/leakage). Before any Modal GPU job: run all four. Modal spend per attempt is non-trivial.

## Open Questions for User

1. Restore RPE1 data, or drop RPE1 from the paper entirely and lean on Norman?
2. Build unified `modal_runner.py` to un-defer Phase 2/3/6, or submit at current scope (paper is honest as-is)?
3. Venue: TMLR (rolling), NeurIPS workshop (Aug-Sep), or journal (Bioinformatics/Genome Biology)?
4. Push `v4-execution` to remote + merge to main, or keep iterating on the branch?
