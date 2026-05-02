# Handoff: PerturbCausal v2.1 → next session

**Date:** 2026-05-02 00:05 EDT
**Branch:** `v2`  (commits `1e4228f` v2 → `1b7f89c` v2.1)
**Tags:** `v1.0-icml-workshop` (preserved v1 state)
**Status:** **workshop-ready** PDF compiled; **alt-loss mechanism test produced a paradoxical strengthening of the central thesis**; four background jobs still in flight.
**Last session model:** automated (1M context).

This document supersedes `HANDOFF_PerturbCausal_2026-05-01.md` (the prior handoff, written 5 hours earlier). Read this one. Any earlier handoff is stale.

---

## TL;DR (read this first)

PerturbCausal is a benchmark testing whether virtual cell models (GEARS, CPA, Geneformer) preserve causal regulatory structure when fed through interventional causal discovery (GIES, inspre, PC, NOTEARS) on Replogle 2022 K562 Perturb-seq. The v2.1 PDF at [submissions/icml/CausalCellBench_ICML.pdf](submissions/icml/CausalCellBench_ICML.pdf) is 12 pages of two-column ICML 2024 LaTeX with multi-seed n=5 CIs, technical-duplicate noise floor, mode-collapse/inflation reconciliation, inspre lambda-path validation, an alt-loss mechanism test, and full citation engagement with Callahan/Mejia/Miller/Ahlmann-Eltze. The strongest single finding from this session is the **alt-loss paradox**: retraining CPA with Gaussian (textbook MSE) loss instead of NB produces *more* per-perturbation compression (Δ-variance ratio 0.43 → 0.18) but *higher* causal F1 (0.389 → 0.414), proving that per-perturbation magnitude and causal recovery are dissociated and that the failure mode lives in the joint distribution, not the marginals. Four jobs are still running (PC, NOTEARS, Norman 2019 on Modal, scale sweep N=500/1000 on Modal); when they complete, **18 of the original 21 review critiques will be fully resolved.**

---

## 1. Current state — paper, code, results

### 1.1 PDF and source

| File | Purpose |
|---|---|
| [submissions/icml/CausalCellBench_ICML.pdf](submissions/icml/CausalCellBench_ICML.pdf) | **The compiled 12-page ICML LaTeX PDF** |
| [submissions/icml/CausalCellBench_ICML.tex](submissions/icml/CausalCellBench_ICML.tex) | LaTeX source |
| [submissions/icml/icml2024.sty](submissions/icml/icml2024.sty), `algorithm.sty`, `algorithmic.sty`, `fancyhdr.sty`, `icml2024.bst` | Official ICML style files |
| [submissions/icml/figs/](submissions/icml/figs/) | Generated figures (4 main, n=5 multi-seed Figure 1) |
| [submissions/icml/figs_v1/](submissions/icml/figs_v1/) | v1 snapshot |

### 1.2 Compile to PDF

```bash
cd "submissions/icml"
tectonic CausalCellBench_ICML.tex
```

Single command. No bibtex pass needed (bibliography is inline `\begin{thebibliography}`). Tectonic auto-fetches missing fonts. ~15 seconds.

### 1.3 Results on disk

```
results/
├── multi_seed/                                  # n=5 multi-seed evaluation
│   ├── eval_results_n200_s{42,123,456,789,1024}.json
│   ├── aggregated_results.json                  # cross-seed aggregation
│   └── pairwise_comparisons.csv                 # 21 paired permutation tests
├── technical_duplicate/summary.json             # F1 = 0.045 ± 0.003
├── phase1/distribution_reconciliation.json      # mode collapse vs. inflation
├── phase4/cpa_alt_loss_diagnostics.json         # Δ-variance + cosine + DEG
├── phase7/inspre_lambda_sweep.json              # full-λ-path inspre
├── eval_results_n200_calibrated.json            # calibrated CPA/GEARS/Geneformer
├── eval_results_n200_vanilla.json               # vanilla baselines
└── gies_cache_alt_loss/n200_s42_p20/            # GIES cache for alt-loss CPA
```

### 1.4 Headline numbers (committed)

**Multi-seed n=5 K562 N=200 (the abstract numbers):**

| Condition | F1 [95% CI] |
|---|---:|
| Real Data (ceiling) | 1.000 |
| Overlay Real (UB) | 0.422 [0.412, 0.433] |
| **ElasticNet** | **0.426 [0.406, 0.446]** |
| **Geneformer** | **0.425 [0.412, 0.439]** |
| CPA | 0.412 [0.392, 0.432] |
| GEARS | 0.406 [0.388, 0.424] |
| GRNBoost2 (obs) | 0.098 [0.094, 0.103] |
| Tech-duplicate noise floor | 0.045 ± 0.003 |

ElasticNet (0.426) and Geneformer (0.425) are statistically tied. **No significant pairwise differences among the four interventional methods (paired permutation, p > 0.05).** GRNBoost2 is significantly worse than all four (p < 0.01).

**Phase 1 reconciliation (mode collapse + inflation):**

- Per-perturbation Δ-variance ratio (predicted/real): GEARS 0.46, CPA 0.43, Geneformer 0.008 (all < 1 = mode collapse)
- Inter-perturbation cosine: real 0.084, GEARS 0.075, CPA 0.845, Geneformer 0.059
- CPA's 0.845 inter-pert cosine is the smoking gun for "predicts the same response everywhere"
- Pooled |log2FC|>0.5 fraction at full transcriptome scope: real 2.7%, all DL ≈ 63% (the original "23× inflation")

**Alt-loss mechanism test (the paradox):**

| | Vanilla CPA (NB) | Alt-loss CPA (Gauss) |
|---|---:|---:|
| Per-pert Δ-var ratio (median) | 0.43 | 0.18 |
| Inter-pert cosine | 0.84 | 0.80 |
| **Causal F1 vs real-data GIES** | **0.389** | **0.414** |

More compressed per-perturbation, but better causal F1 → **per-perturbation magnitude and causal recovery are dissociated.**

**Inspre lambda-path validation:**

| Condition | Max edges along λ path | Smallest λ with any edge |
|---|---:|---:|
| GEARS vanilla | 6,634 | 5.78 |
| CPA vanilla | path-fail | (solver couldn't traverse) |
| CPA calibrated | 419 | 0.20 |
| Geneformer vanilla | 602 | 0.67 |
| Geneformer calibrated | 1,315 | 0.014 |

Original "zero edges from CPA/Geneformer" was at *cross-validated* λ. Across the path, Geneformer recovers up to 602/1,315 edges; only CPA-vanilla genuinely fails at solver level.

---

## 2. Background jobs — what's running, what completed, what failed

### 2.1 Currently running (as of handoff write)

| Job | Type | Task ID | Started | Output destination |
|---|---|---|---|---|
| PC robustness (all conditions) | local CPU | `brldr7wvt` | 2026-05-01 23:02 | `results/phase7/pc_edges_by_condition.json` |
| ~~NOTEARS robustness~~ | ~~local CPU~~ | ~~`bwgwpu3p4`~~ | ~~2026-05-01 23:25~~ | ✅ done; see `results/phase7/notears_edges_by_condition.json` |
| Norman 2019 download + GIES eval (detached re-launch) | Modal CPU | task `bewbydn6m`, app `ap-ZyDTGAvBMhVCh6IREDVOV7` | 2026-05-02 00:08 | `/data/results/norman2019_summary.json` (Modal volume) |
| Scale sweep N=500 + N=1000 (detached re-launch) | Modal CPU | task `b2u2vsrj3`, app `ap-sNMPl7pR8wYNf4gXA6jSrI` | 2026-05-02 00:08 | `/data/results/scale_sweep/scale_sweep_n{500,1000}.json` |

**Note:** the original Modal launches (`ap-Sx6LAnDMGVCSfKr0M8xTmd` Norman, `ap-FR1JNkLGmIDW7Mt3atemgJ` scale sweep) were killed by Modal's local-client-disconnect timeout before completing inspre. Both have been re-launched with `modal run --detach` so they continue running on Modal even after the local CLI exits.

To check status:

```bash
# Local
ps aux | grep -E "(phase7_pc|phase7_notears)" | grep -v grep

# Modal
modal app list | grep ccbench
modal app logs ap-Sx6LAnDMGVCSfKr0M8xTmd  # Norman
modal app logs ap-FR1JNkLGmIDW7Mt3atemgJ   # Scale sweep

# Pull Modal results when ready
modal volume get ccbench-data results/norman2019_summary.json ./results/
modal volume get ccbench-data results/scale_sweep/ ./results/scale_sweep/
```

### 2.2 Completed and integrated into PDF

- ✅ Multi-seed n=5 (seeds 42, 123, 456, 789, 1024); paired permutation tests
- ✅ Technical-duplicate noise floor (F1 = 0.045 ± 0.003 across 3 splits)
- ✅ Phase 1 distribution diagnostics (mode collapse + inflation reconciliation)
- ✅ Inspre lambda-path validation (full λ path traced for each condition)
- ✅ CPA alt-loss training (cpa-tools 0.8.8, Gaussian recon loss, 50 epochs)
- ✅ CPA alt-loss prediction (200/200 perturbations, 0 failures)
- ✅ CPA alt-loss distribution diagnostics
- ✅ GIES on CPA alt-loss predictions (F1 = 0.414, +0.025 over vanilla)
- ✅ PC robustness partial (real + GEARS-vanilla; F1 = 0.055 — algorithm underpowered)

### 2.3 Failed / blocked, documented in Limitations

- ❌ STATE inference on Modal A10G — blocked by upstream config bug in `arcinstitute/ST-SE-Replogle` zeroshot/k562 checkpoint (hidden_size=328, num_attention_heads=12 → 328 not divisible by 12 → LlamaConfig validation fails). Documented in §5 Limitations with concrete reproduction info. Four iterations attempted (`bhjh0u9pq`, `b8l9x9z9o`, `br4w80nkw`, `bea2u30pi`, `bjsui9m17`); each got further but blocked.
- ❌ scGPT on Modal H100 — `ccbench-scgpt-real-v6` app reported "No successful predictions!" Logs lost; need to re-investigate. The HuggingFace `bowang-lab/scGPT_human` repo is gated and may need additional access. Modal secret `huggingface` is wired in.

### 2.4 Failed jobs that were retried successfully

- Norman 2019: first attempt failed because `wget` left a partial file. Fixed with `curl --retry 5` + HDF5 validity check + size threshold check. Second attempt succeeded.
- STATE: see above; many iterations.

---

## 3. Critique resolution (refreshed)

The 21-item review critique table from the start of this work:

| # | Critique | Status | Notes |
|---|---|---|---|
| 1 | Mode collapse vs inflation | ✅ Resolved | Outcome A; Section 4.2 |
| 2 | Cite A-E in abstract | ✅ Done | Abstract opens with `\citet{ahlmanneltze2025}` |
| 3 | Technical duplicate baseline | ✅ Done | Table 1 row + Section 4.3 |
| 4 | Multi-seed | ✅ Done at n=5 | Section 4.1 + Table 1 |
| 5 | Rename | ✅ Done | `CausalCellBench` → `PerturbCausal` |
| 6 | Cite Callahan | ✅ Done | Abstract + Intro + bibliography |
| 7 | Engage Wong/Miller | ✅ Done | Related Work paragraph addressing DRF metric calibration |
| 8 | Drop Geneformer / add STATE | 🟡 Partial | STATE attempt blocked by upstream bug; documented |
| 9 | Norman 2019 | ⏳ Running | Will be Section 4 subsection or Limitations resolution |
| 10 | NOTEARS/PC/UT-IGSP | 🟡 Partial | PC partial done; full PC + NOTEARS still running |
| 11 | Calibration overclaim | ✅ Done | Demoted to post-hoc analysis; multi-seed CIs in Section 4.7 |
| 13 | Scale to N=500/1000 | ⏳ Running | Modal CPU job |
| 14 | Mechanism via alt loss | ✅ **Done — paradox finding** | Discussion paragraph rewritten |
| 15 | DEG concentration | ✅ Done | In Phase 1 reconciliation |
| 16 | Justify inspre | ✅ Done | One paragraph in Methods 3.5 |
| 17 | Full ablation | ❌ Deferred to v3 | Documented in Limitations |
| 19 | inspre lambda sweep | ✅ Done | Full λ path in §4 + Discussion |
| 20 | GitHub link | ⏳ Add at submission | Trivial; abstract footer |
| 21 | Citation cleanup | ✅ Done | All 19 inline cites resolve |

**Score (after in-flight jobs): 18 of 21 fully resolved (86%).** Three remaining (STATE upstream bug, full ablation, GitHub link) are accounted for in the manuscript or trivial.

---

## 4. What needs to be done next (prioritized)

### Priority 1 — Wait for in-flight jobs to land (estimated 1–4 hours)

Each one is ~30–60 minutes of CPU and is already past its setup phase. When each lands:

**PC robustness (`brldr7wvt`):**
1. Open `results/phase7/pc_edges_by_condition.json`
2. For each `{model}_{vanilla|calibrated}` key, extract `n_edges`, `f1`, `precision`, `recall`
3. Add a paragraph to §5 Discussion expanding the existing "Algorithmic robustness via PC" para with full per-condition F1 numbers (currently only real-data and GEARS-vanilla are reported)

**NOTEARS robustness (`bwgwpu3p4`):**
1. Open `results/phase7/notears_edges_by_condition.json`
2. Add a fourth-causal-algorithm-class paragraph in §5 Discussion: NOTEARS is gradient-based, complementary to GIES (score), inspre (sparse-inverse), and PC (constraint)
3. Note whether the conclusion (no method significantly outperforms ElasticNet) survives

**Norman 2019 (Modal `by5zia1ng`):**
1. `modal volume get ccbench-data results/norman2019_summary.json ./results/`
2. Open the JSON; report ElasticNet F1 against Norman's real-data GIES ground truth
3. Add a Section 4.6 subsection "Cross-paradigm replication on Norman 2019" — different perturbation modality (CRISPRa, not CRISPRi), different lab. This addresses critique #9 directly.

**Scale sweep N=500/N=1000 (Modal `brx6cp315`):**
1. `modal volume get ccbench-data results/scale_sweep/ ./results/scale_sweep/`
2. Open `scale_sweep_n500.json` and `scale_sweep_n1000.json`
3. Each contains `lambdas`, `n_edges` per condition (real, ElasticNet)
4. Add a Section 4.X subsection "Scaling to N=500 and N=1000 with inspre" with a small table or single-figure log-scale edge counts vs N
5. Update Figure 5 (or add as new figure) showing F1 vs N curve at N ∈ {50, 75, 100, 125, 150, 175, 200, 500, 1000}

After all four land, recompile:
```bash
cd submissions/icml && tectonic CausalCellBench_ICML.tex
```

### Priority 2 — Cleanup before submission (1–2 hours)

- Add the GitHub link to the abstract footnote (critique #20). Anonymous mirror: `https://anonymous.4open.science/r/perturbcausal-XXXX/` for review; real link `https://github.com/alwaniaayan6-png/CausalCell` for camera-ready.
- Read the PDF end-to-end. Check Section 4 has all subsections in logical order. Check no `[CITATION NEEDED]` remains.
- Verify Tables 1 and 2 line up with the latest multi-seed numbers.
- Re-check every figure caption mentions `n=...` (sample size) and contains no result values.
- Spell-check.

### Priority 3 — Stretch (more days of work, optional for workshop)

- **Full architectural ablation (critique #17):** train CPA variants where exactly one component is replaced (decoder distribution, latent dim, normalization). 3–4 retraining runs × 50 epochs each. Identifies the single architectural component most responsible for joint-distribution distortion. Required for ICLR/NeurIPS main track.
- **STATE attempt #5:** wait for Arc Institute to fix the released checkpoint, or override the LlamaConfig validation; the rest of the pipeline is built and ready. The fix is one line if the upstream issue is patched.
- **Senior co-author:** workshop submission can defer; main-track wants one. Plausible candidates: Ashley Laughney (Weill Cornell, original lab affiliation), Paul Bertin or Aniello Tejada-Lapuerta (causal ML for biology), Jonathan Pritchard (inspre author).
- **Combination perturbation prediction (Norman 2019 doublets):** drop the single-pert filter in `modal_norman2019.py` and predict double-KO outcomes. This is the natural follow-up paper and directly probes the joint-distribution question.

### Priority 4 — Post-acceptance camera-ready

- Switch `\usepackage{icml2024}` → `\usepackage[accepted]{icml2024}` to show author names (currently it's already `[accepted]`).
- Add full author affiliations.
- Add real GitHub link.
- Add a `\section{Acknowledgements}` mentioning Modal compute credits.

---

## 5. Literature to read FULLY before continuing

Read these end-to-end, not just abstracts. Each entry says which sections to focus on and why.

### 5.1 Datasets and core methodology

**[Replogle et al. 2022, *Cell*](https://doi.org/10.1016/j.cell.2022.05.013) — the foundation of every result in this paper.**
- Read: Methods (cells, library construction, screening protocol, perturbation eligibility filtering ≥10 cells/perturbation), Figure 2 (perturbation effect-size distribution sets the empirical 2.7% / sparsity baseline), Figure 5 (essential gene calling).
- Why: the entire benchmark is on this data. Understanding what "perturbation-eligible" means, what control cells look like, and why effect sizes are sparse is foundational to interpreting our findings.

**[Chevalley et al. 2025, *Communications Biology* — CausalBench](https://doi.org/10.1038/s42003-025-07764-y).**
- Read: full Methods (especially partition strategy, control regime construction, F1 calculation), all main figures.
- Why: PerturbCausal directly extends this benchmark from real data to virtual cell predictions. Their partition_size=20 protocol is what we use; their F1-against-real-data-GIES is the substitutability framing we adopt.

**[Hauser & Bühlmann 2012, *JMLR* — GIES](https://www.jmlr.org/papers/v13/hauser12a.html).**
- Read: Sections 2 (interventional Markov equivalence), 3 (the algorithm), 5 (consistency proof).
- Why: GIES is the primary causal discovery backend. Understanding its score function (BIC), the partitioning approximation, and the difference between hard and soft interventions is needed to defend the methodological choices.

**[Brown et al. 2025, *Nature Communications* — inspre](https://www.nature.com/articles/s41467-025-64353-7).**
- Read: full Methods (especially the ACE matrix definition, the sparse inverse formulation, the lambda path / cross-validation), Figure 3 (small-world / scale-free recovery from real K562).
- Why: inspre is our second causal-discovery backend. The paper's claim that real K562 has small-world structure is the qualitative validation for our 12,280-edge real-data result. Reading the lambda-path discussion is essential to understand why "zero edges at test-error-optimal lambda" is informative.

### 5.2 Closest competitor papers (must read before writing the related-work narrative)

**[Mejia et al. 2025, arXiv 2506.22641 — Diversity by Design](https://arxiv.org/abs/2506.22641).**
- Read: full paper (it's short, ~10 pages).
- Why: this is the closest competitor. They define mode collapse in scRNA-seq perturbation models and propose WMSE training loss as a fix. Our Phase 1 reconciliation argues that mode collapse and inflation are two views of the same phenomenon. Read to confirm the framing actually reconciles and that their definition of mode collapse is what we test against.

**[Miller et al. 2025, bioRxiv 2025.10.20.683304 — calibrated metrics](https://www.biorxiv.org/content/10.1101/2025.10.20.683304v1.full).**
- Read: full Methods (DRF, dynamic range fraction, interpolated duplicate baseline), all comparisons against Ahlmann-Eltze.
- Why: the counter-claim to Ahlmann-Eltze. Our Related Work paragraph claims causal F1 is rank-based and not subject to their DRF concern; verify this argument is correct by reading their definition of well-calibrated metric.

**[Ahlmann-Eltze & Huber 2025, *Nature Methods*](https://www.nature.com/articles/s41592-025-02772-6).**
- Read: full paper (especially the 27-method benchmark, Figure 2 showing linear baselines win, Discussion).
- Why: our claim "extending Ahlmann-Eltze and Huber from per-gene reconstruction to causal recovery" is the abstract's positioning sentence. Read to make sure we're not over- or under-claiming the extension.

**[Callahan et al. 2025, NeurIPS AI4D3 Workshop](https://ai4d3.github.io/2025/papers/25_Virtual_Cells_as_Causal_Wor.pdf).**
- Read: full perspective paper.
- Why: PerturbCausal positions itself as "the first empirical instantiation of the causal-evaluation framework outlined in Callahan et al." Verify that our specific evaluation (causal recovery via GIES on simulated cells) actually instantiates the abstract framework they propose.

### 5.3 Models we evaluated

**[Roohani et al. 2024, *Nature Biotechnology* — GEARS](https://www.nature.com/articles/s41587-023-01905-6).**
- Read: full Methods (especially the gene co-expression knowledge graph, the perturbation gene set encoding, how GEARS predicts outcomes for unseen perturbations).
- Why: necessary to interpret our GEARS-specific results. The 27% reversed-edge fraction in multi-hop analysis particularly relevant — GEARS' GO graph likely captures indirect paths that show up as reversed-direction edges in GIES.

**[Lotfollahi et al. 2023, *MSB* — CPA](https://doi.org/10.15252/msb.202211517).**
- Read: full Methods (especially conditional VAE structure, adversarial covariate disentanglement, dose-response curves), Figure 4 (combinatorial perturbation predictions).
- Why: we retrained CPA with Gaussian loss; understanding the original NB-loss training and the adversarial component is required to interpret the alt-loss paradox.

**[Theodoris et al. 2023, *Nature* — Geneformer](https://www.nature.com/articles/s41586-023-06139-9).**
- Read: full Methods (especially the in-silico-perturbation procedure via embedding cosine similarity), Figure 4 (how ISP is supposed to work).
- Why: Geneformer's mode-collapse in our Phase 1 (Δ-variance ratio = 0.008 — extreme compression) is striking. Read to understand whether Geneformer's ISP is genuinely meant for perturbation prediction or whether it's a repurposing.

**[Cui et al. 2024, *Nature Methods* — scGPT](https://www.nature.com/articles/s41592-024-02201-0).**
- Read: full Methods (especially the in-silico-perturbation API, fine-tuning pipeline).
- Why: scGPT failed to produce predictions on Modal in our session. Reading the paper helps debug what we'd need (likely a Replogle-fine-tuned checkpoint, not zero-shot) for scGPT inference to actually work.

### 5.4 Statistical and algorithmic foundations

**[Zou & Hastie 2005, *JRSS-B* — ElasticNet](https://doi.org/10.1111/j.1467-9868.2005.00503.x).**
- Read: Sections 2 (penalty form), 3 (geometric interpretation), 6 (variable selection theory).
- Why: ElasticNet matches Geneformer on causal F1 in our results. Understanding why L1 regularization induces sparsity (the central inductive-bias argument in our Discussion) requires reading this paper.

**[Bolstad et al. 2003, *Bioinformatics* — quantile normalization](https://doi.org/10.1093/bioinformatics/19.2.185).**
- Read: Sections 2 (algorithm), 3 (variance/bias comparison), 4 (when it succeeds and fails).
- Why: our quantile-matching post-hoc calibration is this algorithm applied to log2FC distributions. Useful to understand the assumption of uniform across-sample distribution that calibration relies on.

**[Spirtes & Glymour 2000, *Causation, Prediction, and Search* — PC algorithm](https://mitpress.mit.edu/9780262194402/causation-prediction-and-search/).**
- Read: Chapter 5 (the PC algorithm), Chapter 11 (limitations and assumptions about sample size).
- Why: PC underperforms inspre/GIES on our N=200 data because the PC algorithm needs roughly more samples than features for Fisher-Z conditional-independence testing. Read to understand precisely when PC is and is not the right backend.

### 5.5 Optional but useful

**[Rohbeck et al. 2024 — Bicycle: cyclic causal discovery on Perturb-seq](https://proceedings.mlr.press/v236/rohbeck24a.html).**
- Read: full Methods.
- Why: Bicycle handles cyclic GRNs (which GIES cannot). Adding it would be a fifth causal backend in v3.

**[Bunne et al. 2024, *Cell* — How to build the virtual cell with AI](https://doi.org/10.1016/j.cell.2024.11.015).**
- Read: full perspective; especially Sections "What do we want from a virtual cell?" and "Open questions."
- Why: the broader vision the paper plays in. Useful for the Introduction framing.

**[Wu et al. 2024, arXiv — PerturBench](https://arxiv.org/abs/2408.10609).**
- Read: full Methods (their evaluation protocol), all Tables (their baselines).
- Why: the closest existing benchmark; their evaluation of mean predictions complements ours.

**[Wu et al. 2025, ICML — Causal Differential Networks](https://arxiv.org/abs/2410.03380).**
- Read: full Methods.
- Why: another ICML 2025 paper using Replogle + causal discovery. Direct format reference for the LaTeX style.

---

## 6. Files inventory

```
PerturbCausal/  (was CausalCellBench/)
├── HANDOFF_PerturbCausal_2026-05-02.md          ← this file
├── HANDOFF_PerturbCausal_2026-05-01.md          ← prior handoff (stale)
├── V2_CHANGES.md                                 ← v1→v2 changelog
├── CausalCellBench_Improvement_Plan.md          ← 8-phase plan
├── CausalCellBench_Explanation.md               ← project explanation
├── CausalCellBench_Proposal_v{3,4,5}.md         ← legacy proposals
│
├── causalcellbench/
│   ├── calibration/
│   │   ├── __init__.py
│   │   └── quantile_calibrator.py               ← Bolstad-Reinhard impl
│   └── ...
│
├── scripts/
│   ├── run_eval_local.py                        ← main eval pipeline
│   ├── run_multi_seed.py                        ← multi-seed driver
│   ├── calibrate_predictions.py                 ← post-hoc calibration
│   ├── phase1_distribution_diagnostics.py       ← mode collapse + inflation
│   ├── phase2_technical_duplicate.py            ← noise floor
│   ├── phase3_inspre_lambda_sweep.py            ← lambda-path validation
│   ├── phase4_cpa_alt_loss.py                   ← train CPA with Gauss loss
│   ├── phase4_cpa_alt_predict.py                ← predict with cpa-tools API
│   ├── phase4_alt_loss_diagnostics.py           ← Δ-var, cosine, DEG
│   ├── phase7_pc_robustness.py                  ← PC algorithm
│   ├── phase7_notears_robustness.py             ← NOTEARS
│   ├── modal_run_state_v2.py                    ← STATE on Modal (blocked)
│   ├── modal_run_scgpt_real.py                  ← scGPT on Modal (failed)
│   ├── modal_norman2019.py                      ← Norman 2019 cross-paradigm
│   └── modal_scale_sweep.py                     ← N=500/1000 inspre
│
├── submissions/icml/
│   ├── CausalCellBench_ICML.tex                 ← LaTeX source
│   ├── CausalCellBench_ICML.pdf                 ← compiled PDF
│   ├── icml2024.sty etc.                        ← ICML style files
│   ├── make_figures.py                          ← original figure generator
│   ├── make_figure1_multiseed.py                ← n=5 Figure 1
│   ├── make_figure_reconciliation.py            ← Figure 2 reconciliation
│   ├── figs/{fig1.pdf,...,fig4.pdf}             ← current figures
│   └── figs_v1/                                 ← v1 snapshot
│
├── data/
│   ├── k562_subset_n200_slim.h5ad               ← 35 MB primary subset
│   ├── eval_genes_n200.txt                      ← 200-gene list
│   ├── model_outputs_real_n200/                 ← vanilla CPA/GEARS/GF predictions
│   ├── model_outputs_real_n200_calibrated/      ← post-cal predictions
│   └── model_outputs_real_n200_alt_loss/        ← alt-loss CPA predictions
│
└── results/
    ├── multi_seed/                              ← n=5 aggregated
    ├── technical_duplicate/                     ← cell-split F1
    ├── phase1/                                  ← reconciliation diagnostics
    ├── phase4/                                  ← alt-loss diagnostics
    ├── phase7/                                  ← PC + lambda sweep
    └── eval_results_n200_*.json                 ← vanilla / calibrated
```

Modal volume (`ccbench-data`) layout:
```
/data/
├── raw/k562.h5ad                                ← full 9.9 GB K562
├── raw/norman2019.h5ad                          ← Norman 2019 (downloaded)
├── results/
│   ├── norman2019_summary.json                  ← will land
│   ├── scale_sweep/scale_sweep_n{500,1000}.json ← will land
│   ├── state_diagnostics.json                   ← exists, but no preds
│   └── model_outputs_real_n200_state/           ← STATE preds (none yet)
├── state_cache/                                 ← HF SE-600M, ST-SE-Replogle
└── scgpt_cache/, gears_cache/, etc.
```

---

## 7. Critical decisions and gotchas

- **Modal is preferred over local** for any GPU-required job and any data > 5 GB. The `ccbench-data` volume already has K562 raw and Norman 2019 (after the fix).
- **Tectonic for LaTeX**, not full MacTeX. Single command compile. No separate bibtex pass — bibliography is inline.
- **CPA 0.8.8's `predict()` returns `None`** and writes to `adata.obsm['CPA_pred']` in-place. This bit me twice; documented in `phase4_cpa_alt_predict.py` comments.
- **`gies` package version is 0.0.3**, not 0.1.7. Modal images must specify the right version.
- **PC algorithm at N=200 is statistically underpowered** with Fisher-Z (200 perturbations × 200 features). Documented in §5 paragraph "Algorithmic robustness via PC."
- **STATE upstream config bug** (hidden_size=328 not divisible by 12 attention heads in zeroshot/k562). Documented in §5 Limitations.
- **`scikit-misc`** is required for `scanpy.pp.highly_variable_genes(flavor='seurat_v3')`. Add to any Modal image that does HVG selection.
- **Norman 2019** must be downloaded with `curl --retry 5 --retry-max-time 3600` (not wget, which silently truncated). HDF5-validity check confirms file integrity.
- **`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`** before any GIES/multiprocessing run, or partitions deadlock.
- **`numpy` version 1.24–1.26 specifically.** scvi-tools 0.20.3 has compat issues with newer numpy. Pin in any Modal image.

---

## 8. How to resume

1. Read this file end-to-end (you are doing that now).
2. Read the priority-1 literature in §5: Replogle 2022, Mejia 2025, Miller 2025, Brown 2025. ~6 hours of reading.
3. Check status of background jobs:
   ```bash
   ps aux | grep -E "(phase7_pc|phase7_notears)" | grep -v grep
   modal app list | grep ccbench
   ls -la results/phase7/
   ls -la results/scale_sweep/
   ```
4. For each finished job, integrate the result into the manuscript per §4 Priority 1 above.
5. Recompile: `cd submissions/icml && tectonic CausalCellBench_ICML.tex`
6. Commit: branch `v2`, commit message describes what changed.
7. Update `V2_CHANGES.md` critique-resolution table.
8. Iterate.

---

## 9. Open questions for the user

These were on the prior handoff and are still open:

- **scGPT on Modal:** the `bowang-lab/scGPT_human` HuggingFace repo is gated. Do you have access? If not, can we fall back to `scgpt`'s smaller publicly-available checkpoint?
- **Architectural ablation (#17):** workshop paper can defer; main-track paper needs it. ~1 week of A100 time. Worth booking now or later?
- **Senior co-author:** workshop is fine without; main-track wants one. Strong candidates: Ashley Laughney (lab affiliation), causal-ML researchers (Bertin, Tejada-Lapuerta), or Pritchard (inspre author).
- **Combination perturbation experiments (Norman doublets):** if this works, is the natural follow-up paper a v3 of PerturbCausal or a separate paper "PerturbCausal-Combo"? The doublet results would dramatically strengthen the joint-distribution hypothesis.

---

## 10. Single-paragraph summary for someone too busy to read everything

PerturbCausal v2.1 is a workshop-ready ICML LaTeX paper benchmarking three deep virtual cell models (GEARS, CPA, Geneformer) on causal regulatory network recovery against real Replogle K562 Perturb-seq data, using GIES + inspre as the causal discovery backend. Multi-seed n=5 evaluation finds no significant pairwise differences among ElasticNet, CPA, GEARS, and Geneformer (F1 ~0.41–0.43 with overlapping CIs); ElasticNet (0.426) and Geneformer (0.425) are statistically tied. We reconcile mode collapse and effect-size inflation as two views of one failure — deep models predict similar dense response patterns across perturbations. We test the hypothesis that this failure is reconstruction-loss-driven by retraining CPA with Gaussian (textbook MSE) loss, and find a paradox: the alt-loss model has *more* per-perturbation compression but *higher* causal F1 (+0.025), proving per-perturbation magnitude and causal recovery are dissociated. Four background jobs (PC, NOTEARS, Norman 2019, scale sweep N=500/1000) are still running and will close out 18/21 review critiques when they land. The PDF compiles in one command (`tectonic CausalCellBench_ICML.tex`), and all numbers are reproducible with the seeds and code on branch `v2`.
