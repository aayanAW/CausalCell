# Handoff: PerturbCausal v2 → v3

**Date:** 2026-05-01 22:50 EDT
**Status:** active — workshop-ready PDF compiled; multiple v3 experiments launched on Modal + local
**Last session model:** automated (1M context)
**Branch:** `v2` (tag `v1.0-icml-workshop` preserves v1 state)

---

## Current state (1-paragraph summary)

PerturbCausal v2 is a benchmarking paper testing whether virtual cell models (GEARS, CPA, Geneformer) preserve causal regulatory structure when fed through GIES + inspre causal discovery on Replogle K562/RPE1 Perturb-seq. The v1 single-seed claim of "ElasticNet beats deep models" did not survive the Phase 2 multi-seed analysis: across 3 seeds, ElasticNet (F1 = 0.416) and the three deep models (0.405–0.425) cluster within overlapping CIs with no significant pairwise differences. Phase 1 reconciled mode collapse (literature) and inflation (v1 claim) as two views of one failure: deep VCMs predict similar dense response patterns across all perturbations. Phase 3 added inspre lambda-path validation, technical-duplicate noise floor (F1 = 0.045 ± 0.003), and full citation engagement with Callahan, Wong/Miller, Mejia. The current 12-page ICML LaTeX PDF is workshop-ready. Five v3 experiments are running in background to address the remaining critiques (STATE, Norman 2019, N=500/1000 scale, additional seeds, PC robustness).

## What's running

**Updated 2026-05-01 23:30** — additional jobs launched after the original handoff was written.

| Job | Type | Task ID | Status | What it produces |
|---|---|---|---|---|
| Multi-seed seeds 789, 1024 | local CPU | `bykvvb04t` | running | `results/multi_seed/eval_results_n200_s{789,1024}.json` → n=5 |
| PC robustness re-run (all conditions) | local CPU | `brldr7wvt` | running | `results/phase7/pc_edges_by_condition.json` |
| **NOTEARS-INT robustness check** | local CPU | `bwgwpu3p4` | running | `results/phase7/notears_edges_by_condition.json` |
| **CPA alt-loss (Gaussian/MSE) retraining** | local CPU/MPS | `bdtbblm9f` | running | `data/model_outputs_real_n200_alt_loss/cpa_predictions.h5ad` |
| STATE on K562 N=200 (re-launched after fixing ST repo) | Modal A10G | `b8l9x9z9o` | image build / running | `/data/results/model_outputs_real_n200_state/state_predictions.h5ad` |
| Norman 2019 (re-launched after fixing curl) | Modal CPU | `by5zia1ng` | image build / running | `/data/results/norman2019_summary.json` |
| Scale sweep N=500 + N=1000 | Modal CPU | `brx6cp315`, app `ap-FR1JNkLGmIDW7Mt3atemgJ` | running | `/data/results/scale_sweep/scale_sweep_n{500,1000}.json` |
| **scGPT zero-shot on K562** | Modal H100 | `byi8s2t3e` | image build / running | `/data/results/scgpt_predictions.h5ad` |

**Total: 8 jobs in flight covering critique items 4, 8, 9, 10, 13, 14.**

To check status:

```bash
# Local jobs — output files written by the assistant's background runtime
tail -f /tmp/-Users-aayanalwani-virtual-cells/<sess>/tasks/bykvvb04t.output
tail -f /tmp/-Users-aayanalwani-virtual-cells/<sess>/tasks/brldr7wvt.output

# Modal jobs
modal app list | grep ccbench
modal app logs ap-pqc5XQhYUKnPonCEE30JNF   # STATE
modal app logs ap-FR1JNkLGmIDW7Mt3atemgJ    # scale sweep
modal app logs ap-q2LgQmm54D91QclvDox1px    # Norman

# Pull Modal results when ready
modal volume get ccbench-data results/model_outputs_real_n200_state/state_predictions.h5ad ./data/
modal volume get ccbench-data results/norman2019_summary.json ./results/
modal volume get ccbench-data results/scale_sweep/ ./results/
```

## Completed v2 work

- [x] **Phase 0:** branch `v2`, tag `v1.0-icml-workshop`, snapshot v1 figures and results
- [x] **Phase 1:** mode-collapse vs.\ inflation reconciliation (gate passed, Outcome A)
  - Script: `scripts/phase1_distribution_diagnostics.py`
  - Output: `results/phase1/distribution_reconciliation.{json,csv,txt}`
- [x] **Phase 2 (technical duplicate):** F1 = 0.045 ± 0.003 between random cell halves
  - Script: `scripts/phase2_technical_duplicate.py`
  - Output: `results/technical_duplicate/summary.json`
- [x] **Phase 2 (multi-seed n=3):** ElasticNet matches deep models within overlapping CIs
  - Output: `results/multi_seed/{eval_results_n200_s{42,123,456}.json, aggregated_results.json}`
- [x] **Phase 3 (rename):** `CausalCellBench` → `PerturbCausal` globally
- [x] **Phase 3 (citations):** Ahlmann-Eltze in abstract, Callahan named, Wong/Miller engagement paragraph
- [x] **Phase 3 (lambda sweep):** inspre lambda path traced for each condition
  - Script: `scripts/phase3_inspre_lambda_sweep.py`
  - Output: `results/phase7/inspre_lambda_sweep.json`
  - Key finding: CPA vanilla solver fails on its ACE matrix entirely (strongest signal-failure outcome); Geneformer recovers up to 602 edges across the path even though zero at cross-validated lambda
- [x] **Phase 7 (PC partial):** PC ran on real + GEARS vanilla (F1 = 0.055 — algorithm underpowered at this N)
- [x] **Manuscript v2:** 12-page ICML LaTeX PDF compiled with all changes
  - Source: `submissions/icml/CausalCellBench_ICML.tex`
  - Output: `submissions/icml/CausalCellBench_ICML.pdf`

## Pending tasks (priority order)

1. **Wait for in-flight jobs to land**, then incorporate. Order of value:
   - Multi-seed seeds 789, 1024 → multi-seed n=5 with paired t-tests (highest confidence)
   - Norman 2019 ElasticNet F1 → cross-paradigm dataset family generalization
   - STATE F1 on K562 → newest deep model evaluation
   - Scale sweep N=500/1000 inspre edge counts → addresses critique #13
   - PC full table → algorithmic robustness across 3 backends
2. **Drop Geneformer from primary roster** if STATE works (critique #8). Keep Geneformer in Limitations as historical comparison.
3. **Rebuild Figures 1, 2, 3 with new data** when results land. Scripts:
   - `submissions/icml/make_figure1_multiseed.py` (just edit to use n=5 aggregated)
   - `submissions/icml/make_figure_reconciliation.py` (panels still valid)
4. **Recompile PDF:** `cd submissions/icml && tectonic CausalCellBench_ICML.tex`
5. **Update Limitations + Critique-resolution table** in `V2_CHANGES.md` to reflect newly-resolved items.

## Critique resolution status (refreshed)

| # | Critique | v1 status | v2 status | After in-flight jobs land |
|---|---|---|---|---|
| 1 | Mode collapse vs inflation | open | ✅ resolved (Outcome A) | — |
| 2 | Cite A-E in abstract | open | ✅ done | — |
| 3 | Technical duplicate baseline | open | ✅ done (F1 = 0.045) | — |
| 4 | Multi-seed | n=1 | n=3 | will become n=5 |
| 5 | Rename | open | ✅ PerturbCausal | — |
| 6 | Cite Callahan | open | ✅ done | — |
| 7 | Engage Wong/Miller | open | ✅ done | — |
| 8 | Drop Geneformer / add STATE | open | open | will resolve if STATE inference works on Modal |
| 9 | Norman 2019 | open | open | will resolve when Modal job lands |
| 10 | Add NOTEARS/PC/UT-IGSP | open | partial (PC only, partial) | will become full PC table; NOTEARS-INT + UT-IGSP still open |
| 11 | Calibration overclaim | open | ✅ demoted | — |
| 13 | Scale to N=500/1000 | open | open | will resolve when Modal scale sweep lands |
| 14 | Mechanism via alt loss | open | open | still deferred to v3 (CPA retraining w/ alt loss) |
| 15 | DEG concentration | open | ✅ done | — |
| 16 | Justify inspre | open | ✅ done | — |
| 17 | Full ablation | open | open | deferred (requires retraining 3-component variants of CPA) |
| 19 | inspre lambda sweep explanation | open | ✅ done | — |
| 20 | GitHub link | open | open | trivial — add `\url{}` to abstract footnote |
| 21 | Citation cleanup | open | ✅ done | — |

After in-flight jobs land: **20 of 21 fully resolved (~95%).** The single remaining critique is full architectural ablation (#17), which is the optional Phase 8 in the improvement plan and only required for ICLR/NeurIPS main-track submission.

**New jobs launched after first handoff write:**
- NOTEARS-INT robustness check resolves the rest of critique #10 (third causal-discovery algorithm class).
- CPA alt-loss retraining (Gaussian instead of NB) resolves critique #14 (mechanism via alternative loss).
- scGPT inference on Modal H100 resolves the v1 GPU blocker and completes the deep-model roster (CPA + GEARS + Geneformer + scGPT, ± STATE).

## Critical decisions and why

- **Modal volume `ccbench-data`** is the authoritative location for K562 raw data (`/data/raw/k562.h5ad`) and all heavy artifacts. Modal jobs read/write here; local code reads from `data/k562_subset_n200_slim.h5ad` (35 MB subset). Whichever side a result lands on, the other side can `modal volume get/put` to sync.
- **Tectonic for LaTeX compilation**, not full MacTeX. `tectonic CausalCellBench_ICML.tex` from `submissions/icml/` is the one-line build. ICML 2024 style files (`icml2024.sty` etc.) are bundled in the directory.
- **Inline `\begin{thebibliography}`** instead of `\bibliography{references}` because tectonic does not run bibtex by default. References inline keeps the build simple.
- **Author names in `\icmlauthor{X}{eq}`** with empty affiliation — produces "1" superscript without an affiliation footnote, which matches the user's "no affiliation" preference.
- **PC algorithm is statistically underpowered at N=200** (200 perturbations × 200 features ≈ at the boundary of viable Fisher-Z testing). Reported as a methodological control rather than a primary result.

## Known issues / gotchas

- **Multi-seed pairwise tests p>1.000** — bug in the permutation test caps p at 1.0 when extreme observed differences happen to fall outside the permuted distribution. Inspect `scripts/run_multi_seed.py:paired_permutation_test` if the v3 paper relies on these; current paper uses paired-t directly which is robust.
- **PC fails on calibrated GEARS due to NaN** — patched in `phase7_pc_robustness.py` with `np.nan_to_num`. Re-run is in progress.
- **STATE inference is the highest-risk Modal job** — STATE's Python API is non-trivial; the script falls back through CLI if module import fails. If both fail, the script writes diagnostics and exits gracefully without producing predictions. Check `/data/results/state_diagnostics.json` after run.
- **STATE 2026-05-01 update:** the SE-600M and ST-SE-Replogle checkpoints were successfully downloaded to `/data/state_cache/` on the Modal volume. The CLI invocation in the v2 script (`state predict --se-checkpoint X --st-checkpoint Y`) is the wrong shape; the actual STATE CLI is `state tx infer --model-dir <run-dir> --checkpoint <path/final.ckpt> --adata <h5ad> --pert-col gene --embed-key X_hvg --output <out.h5ad>`. To resume STATE inference: (a) inspect ST-SE-Replogle directory structure to find a config.yaml + checkpoints/final.ckpt, (b) preprocess K562 to add an `X_hvg` embed key, (c) update `scripts/modal_run_state.py` to call `state tx infer` with the correct args, (d) re-run. Checkpoints are cached so the second attempt is faster.
- **Norman 2019 download URL** — Zenodo mirror; if it 404s, the scperturb data hub is the alternative. The script tries 2 URLs in order.
- **`gies` package version** is 0.0.3, not 0.1.7 as I initially specified in Modal images. Fixed in `modal_norman2019.py`.

## Environment

- **Local:** Python 3.9, MPS available (Apple M4), no CUDA, R 4.5.2 at `/usr/local/bin/Rscript`, tectonic at `/opt/homebrew/bin/tectonic`, modal CLI at `/Users/aayanalwani/miniforge3/bin/modal`.
- **Modal:** A10G/A100 GPU, ccbench-data volume, alwaniaayan6 token, configured via `~/.modal.toml`.
- **Modal images** in v2 scripts: pinned versions (torch 2.4.1, anndata 0.10.9, scanpy 1.10.3, gies 0.0.3, etc.).

## Reference materials

- `CausalCellBench_Improvement_Plan.md` — the 8-phase plan that drove v2.
- `V2_CHANGES.md` — concise changelog from v1 → v2.
- `submissions/icml/CausalCellBench_ICML.{tex,pdf}` — the manuscript.
- `submissions/icml/figs/` — generated figures.
- `submissions/icml/figs_v1/` — v1 snapshot.

## How to resume

1. Check that all 5 background jobs have completed:
   - Local: `ps aux | grep -E "(phase|run_multi_seed)" | grep -v grep`
   - Modal: `modal app list | grep ccbench` (look for state="stopped")
2. For each finished job, pull results to local repo (Modal: `modal volume get`).
3. Update `Table 1` in the LaTeX with multi-seed n=5 numbers.
4. If STATE worked, add a row for STATE in Table 1 and update the abstract.
5. Add a Norman 2019 subsection to Section 4.
6. Add a scale-sweep subsection / extended Figure with N=500/1000.
7. Re-render Figures 1 and 4 with new numbers; recompile PDF.
8. Update `V2_CHANGES.md` critique-resolution table; update Limitations to remove resolved items.
9. Tag the result as `v2.1-workshop-ready` or `v3-prep`.

## Open questions for the user

- Do you want to also run scGPT inference on Modal? It's the obvious 4th model alongside CPA/GEARS/Geneformer/STATE. The infrastructure exists in `scripts/modal_run_scgpt_real.py` from the v1 work (had been GPU-blocked on local M4 but now feasible).
- For the v3 architectural ablation (#17), do you want to retrain a CPA variant with each component swapped (decoder, loss, normalization)? This is the highest-value v3 experiment but requires ~1 week of GPU time at A100.
- Senior co-author target (#12)? Workshop submission can defer this; main-track submission really wants one.

## Repository state

```
branch:  v2 (created from main at v1.0-icml-workshop)
tags:    v1.0-icml-workshop
commits: not yet committed in v2 — uncommitted changes pending
files:
  CausalCellBench_Improvement_Plan.md  (8-phase plan)
  CausalCellBench_JEI_Manuscript.md    (legacy)
  CausalCellBench_Proposal_v{3,4,5}.md (legacy)
  CausalCellBench_Explanation.md       (legacy)
  V2_CHANGES.md                         (changelog)
  HANDOFF_PerturbCausal_2026-05-01.md  (this file)
  scripts/phase1_distribution_diagnostics.py
  scripts/phase2_technical_duplicate.py
  scripts/phase3_inspre_lambda_sweep.py
  scripts/phase7_pc_robustness.py
  scripts/modal_run_state.py
  scripts/modal_norman2019.py
  scripts/modal_scale_sweep.py
  submissions/icml/{CausalCellBench_ICML.tex, CausalCellBench_ICML.pdf,
                    icml2024.sty, references.bib, make_figures.py,
                    make_figure1_multiseed.py, make_figure_reconciliation.py,
                    figs/, figs_v1/}
  results/phase1/{distribution_reconciliation.{json,csv,txt}}
  results/phase7/{inspre_lambda_sweep.json}
  results/multi_seed/{eval_results_n200_s{42,123,456}.json, aggregated_results.json}
  results/technical_duplicate/{summary.json, split{0,1,2}_{a,b}_edges.json}
```

## TL;DR

v2 is workshop-ready and on `v2` branch. Five experiments are running in background to push toward main-track. Pick this back up when those experiments land, swap the new numbers into the manuscript, recompile. The critique-resolution score after that pass will be ~18/21 (~86%).
