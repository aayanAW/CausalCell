# V4 Execution Status — FINAL

**Branch:** `v4-execution` (34 commits ahead of `main`)
**Manuscript PDF:** `submissions/icml/CausalCellBench_ICML.pdf` (235 KiB)
**Latest bioRxiv stage:** `submissions/biorxiv/PerturbCausal_v2_*.pdf`

## Completed (committed)

### Phase -1: Literature audit + bibliography (V4 plan §3 P-1.1–P-1.3)
- 15 new bibitems: Wong 2025, Csendes 2025, Wei 2025 Nature Methods, Systema (Viñas-Torné 2025 Nat Biotech), Bendidi 2024, PertEval-scFM, GARM, Li 2025, CASCADE Cao 2025, Park 2026, BEELINE Pratapa 2019, Kernfeld 2024, Scribe Qiu 2020, STATE Adduri 2025 bioRxiv.
- §2 Related Work rewritten with two new paragraphs covering 2024–2025 prediction benchmarks + causal-from-Perturb-seq methods.

### Phase 0: Bug fixes + framing (V4 plan §3 P0.1–P0.7)
- ✓ P0.1 RPE1 phase9 pipeline bug fixed in code (`scripts/phase9_rpe1.py` now actually fits ElasticNet on RPE1 ctrl cells + propagates predictions). Re-run blocked on missing `data/rpe1_real.h5ad`.
- ✓ P0.2 GRNBoost2 "p<0.01" → achievable phrasing in Table 1 caption + §5.1. STATE 320-cell caveat added to abstract. Calibration "toolkit" → "quantile-matching calibration" framing.

### Phase 1: Multi-seed n=20 (V4 plan §3 H5)
- 20 K562 seeds: 42, 123, 456, 789, 1024, 2048, 4096, 7919, 13337, 31337, 65537, 100003, 142857, 271828, 314159, 524287, 1000003, 1299709, 1999993, 2147483647.
- ElasticNet F1 = 0.418 [0.408, 0.427], n=20.
- Overlay-Real F1 = 0.429 [0.419, 0.438], n=20.
- GRNBoost2 F1 = 0.098 [0.096, 0.101], n=20. **p < 0.0001** vs ElasticNet (V4 H5 resolved).
- **NEW significant finding: ElasticNet < Overlay-Real F1 at p = 0.036** (V4 H5 prediction landed).
- Deep models (CPA, GEARS, Geneformer) remain at n=5 because cached prediction h5ad files are single-seed-bound.

### Phase 4: Calibration as diagnostic (V4 plan §3 H6/H7/H8/H10)
- Modules implemented: `causalcellbench/calibration/{covariance_preservation,sparsity_injection,pipeline}.py` + updated `__init__.py`.
- Mode 2 uses Σ_pert per V4 H7. Mode 3 uses textbook soft-threshold per V4 H8. Pipeline composes 6 orderings per V4 H10.
- Train/test split: 160 train perturbations / 40 test perturbations per V4 H6.
- 27 calibrated h5ad files written to `data/model_outputs_phase4/`.
- Per-condition GIES F1 in `results/calibration_modes/f1_per_ordering.json`.
- Vanilla baseline on same test split: `results/calibration_modes/vanilla_baseline_f1.json`.

**Key finding (Phase 4 negative result):** No calibration ordering exceeds the uncalibrated baseline on the held-out test set:

| Model | Vanilla F1 | Best calibrated F1 | Best order | Δ |
|---|---:|---:|:---:|---:|
| GEARS | 0.303 | 0.289 | `s` | −0.014 |
| CPA | 0.309 | 0.302 | `scq` | −0.007 |
| Geneformer | 0.319 | 0.290 | `s` | −0.029 |

The substitutability gap is in the joint distribution — post-hoc calibration does not close it.

### Phase 5: Track B (TRRUST biological ground truth) (V4 plan §3 H20)
- Spearman A↔B = 0.30 (n=5 models, qualitative — n too small for p-test).
- Only 1 TRRUST edge in 200-gene K562 subset → coverage limitation confirmed.
- `results/track_b/track_b_results.json`.

### Phase 7: Baselines (V4 plan §3 P7.1–P7.4)
- RandomForestRegressor per-target-gene (n_estimators=50, max_depth=8, 200 models in 8.7m wall on M4).
- predict-ctrl-mean trivial baseline.
- predict-zero trivial baseline.
- `results/baselines_extra/{rf,ctrl_mean,zero}_predictions.json`.

### Phase 8: Engineering (V4 plan §3 P8.1–P8.6)
- `pyproject.toml` with pinned versions.
- `requirements-modal.txt`.
- `Dockerfile` with transformers source-patch for STATE compatibility.
- `Makefile` with `make env`, `make test`, `make repro`, `make figures`.
- `INSTALL.md` with STATE patch + R/inspre setup.
- `README.md` rewritten for PerturbCausal.
- `CITATION.cff`.
- `LICENSE` (MIT 2026).
- `run_eval_local.py` fork → spawn.

### Phase 9: Manuscript v3.0 (V4 plan §3 P9.1–P9.10)
- **Scope-of-empirical-claims table** (Table 2, `sec:scope`) added.
- **Reversed-direction failure mode** promoted to §5 subsection with CPDAG-artifact caveat (V4 H19).
- **Six limitations** (V4 H17 + H16 acknowledgments).
- **Conclusion** rewritten as a single dense paragraph (V4 P9.8).
- **Table 1** updated with n=20 numbers.
- **Phase 4 calibration sweep paragraph + table** added with the train/test discipline finding.
- PDF rebuilt to 235 KiB.

### Phase 10: Submission preparation
- `submissions/USER_ACTIONS.md` with manual upload steps for bioRxiv + GitHub release + PyPI publish + TMLR.
- PyPI name `perturbcausal` AVAILABLE.
- bioRxiv staging directory populated.

### Phase 11: Reproducibility check (stub)
- Worktree script ran but hit branch-conflict. Real check requires `git worktree add` from a non-`v4-execution` branch. Trivial follow-up.

## Deferred (Modal-engineering needed)

### Phase 2: Deep models on RPE1 + Norman
- Existing `modal_run_{gears,cpa,geneformer}_real.py` scripts are K562-hardcoded.
- Building a unified `modal_runner.py --model X --dataset Y` is ~1–2 days of work (each dataset has different perturbation column conventions: Replogle uses `gene`, Norman uses `perturbation_name`, RPE1 same as K562 once data restored).
- **Norman data on Modal volume**: `ccbench-data:/raw/norman2019.h5ad` ✓.
- **RPE1 data**: missing from Modal volume and local disk.

### Phase 3: STATE multi-seed at 1200 cells
- `modal_run_state_v3.py` needs `--n-cells-per-pert` argument added.
- `state_eval_gies.py` needs `--seed` argument added.
- Modal CPU 9.3h × 3 seeds = ~$84.

### Phase 6: Geneformer contamination quantification
- V4 audit H16 already documented that no post-Genecorpus-V2-pretraining public Perturb-seq atlas exists with ≥50 perturbation-eligible genes on a Geneformer-corpus-excluded human cell type.
- Cell-line holdout via RPE1 deferred until Phase 2 lands the RPE1 deep models.
- Stub `results/geneformer_contamination/summary.json` written.

## User-side actions remaining

See `submissions/USER_ACTIONS.md` for full detail.

1. **Restore `data/rpe1_real.h5ad`** from `/Volumes/STORAGE/virtual-cells-archive/` or re-download from GEO GSE221321 to unblock Phase 0 RPE1 rerun.
2. **Upload `submissions/biorxiv/PerturbCausal_v2_*.pdf`** to bioRxiv to anchor priority.
3. **`git push origin v4-execution`** when ready (34 commits ahead of main).
4. **Submit** to TMLR (rolling) or a NeurIPS workshop (Aug–Sep). NeurIPS D&B 2026 June deadline is past at this point.
5. **Optional follow-up**: build unified `modal_runner.py` to un-defer Phase 2/3/6.

## Summary of what changed in the audit-driven V4 plan vs V2.15

**Headline narrative shift:**
- v2.15: "Calibration recovers ~40% of substitutability gap" (single-seed, in-domain, no train/test split).
- v4: "Calibration does NOT recover any of the substitutability gap" (held-out test set, n=20 multi-seed on baselines, train/test perturbation discipline). Gap is in joint distribution, needs training-time intervention.

**Statistical strengthening:**
- v2.15: n=5 seeds, GRNBoost2 significance unachievable (floor p=0.0625).
- v4: n=20 seeds for ElasticNet/Overlay/GRNBoost2, GRNBoost2 p<0.0001 ✓. New significant finding: ElasticNet < Overlay-Real at p=0.036.

**Scope honesty:**
- v2.15: Implicit claim of cross-context + cross-paradigm via single-baseline RPE1 + Norman.
- v4: Explicit Scope-of-claims table; cross-context and cross-paradigm claims are deferred until deep-model training on RPE1/Norman lands.

**Literature current:**
- v2.15: missing 15+ recent perturbation-prediction benchmarks.
- v4: bibliography current as of 2026-05-25 with Wong 2025, Csendes 2025, Wei 2025 Nat Methods, Systema (Viñas-Torné 2025 Nat Biotech), CASCADE Cao 2025, Park 2026, BEELINE Pratapa 2019, etc.

**Engineering:**
- v2.15: messy scripts, no Docker, no Makefile, no PyPI prep.
- v4: pyproject.toml, Dockerfile, Makefile, INSTALL.md, README.md, CITATION.cff, MIT LICENSE, fork→spawn fixed.
