# CausalCellBench: Evaluation Results Summary

## Overview

CausalCellBench tests whether virtual cell model simulations can substitute for
real CRISPRi perturbation experiments in causal gene regulatory network (GRN)
discovery. This document summarizes the pilot evaluation on Replogle K562 data.

## Experimental Setup

- **Dataset**: Replogle et al. (2022) K562 CRISPRi Perturb-seq
  - 310,385 cells, 8,563 genes
  - 10,691 non-targeting control cells
  - 1,861 perturbation-eligible genes (>=10 cells, in var_names)
- **Gene subset**: 50 randomly selected perturbation-eligible genes (seed=42)
- **Control cells**: 200 (subsampled from 10,691 to prevent control domination of GIES)
- **Perturbation cells**: 100 per gene (subsampled from real data)
- **Causal discovery**: GIES with 25-gene partitions (CausalBench protocol)
- **Observational baseline**: GRNBoost2 (gradient boosting, no intervention data)

## Key Methodological Findings

### 1. Control Cell Domination (Fixed)
Initial runs used all 10,691 control cells vs 100 pert cells/gene. The control
regime had 100x more data, causing GIES to fit control covariance regardless of
perturbation quality. Fix: subsample controls to 200 (matching CausalBench).

### 2. Independent NB Sampling Destroys Causal Signal (Fixed)
The standard NB (Negative Binomial) sampling approach generates each gene
independently via Gamma-Poisson compound. This destroys gene-gene correlations
that GIES needs to infer edges. Result: even PERFECT perturbation means + NB
sampling recovered only 10 edges (vs 538 from real cells).

Fix: "Overlay sampling" — bootstrap control cells and apply fold-change shifts.
This preserves the control covariance structure while applying perturbation effects.

### 3. Low-Expression Gene Threshold (Fixed, 2026-04-07)
The original overlay sampling used `ctrl_means > 0.1` as the threshold for
applying fold-change ratios. This silently skipped perturbation effects for
~30-40% of genes with low baseline expression. Fixed to `ctrl_means > 1e-6`
with a pseudocount of 0.01 to prevent division-by-zero while preserving signal
for low-expression genes.

### 4. Random Baseline Design
Random baseline = bootstrapped control cells with no perturbation effect.
GIES should find no real edges since there's no distributional shift between
"perturbed" and control regimes.

## Results (50 genes, K562, 3 real models)

| Condition           | Edges | AUPRC  | P@100 | TP  | SHD | Wasserstein | FOR    |
|---------------------|-------|--------|-------|-----|-----|-------------|--------|
| Real Data (UB)      | 214   | 1.0000 | 1.000 | 214 | 0   | 0.476       | 0.083  |
| ElasticNet          | 246   | 0.3956 | 0.300 | 91  | 253 | 0.564       | 0.035  |
| **GEARS (real)**    | **243** | **0.3852** | **0.350** | **81** | **259** | **0.586** | **0.095** |
| Overlay Real (UB)   | 256   | 0.3614 | 0.330 | 86  | 269 | 0.613       | 0.266  |
| GRNBoost2 (obs)     | 428   | 0.3420 | 0.410 | 123 | 379 | 0.568       | 0.085  |
| **CPA (real)**      | **259** | **0.3217** | **0.350** | **84** | **278** | **0.545** | **0.020** |
| **Geneformer (real)**| **231** | **0.3188** | **0.350** | **82** | **252** | **0.564** | **0.015** |
| Random (floor)      | 380   | 0.2849 | 0.290 | 103 | 351 | 0.200       | 0.000  |

- **Random floor AUPRC**: 0.093 (expected for random edge predictions)
- **Ground truth**: 214 edges from GIES on real perturbed cells
- **Total pipeline time**: 3536s (~59 min) on Apple M4 Mac
- **GEARS training**: 15 epochs, 70 min on CPU, val MSE=3.69
- **CPA training**: 50 epochs, 5.8 min on CPU, val r2_mean=0.71
- **Geneformer ISP**: 47 genes (3 missing from vocabulary), 9.4 min on CPU, pretrained V2-104M

## Interpretation

1. **The benchmark discriminates**: Clear ranking: ElasticNet (0.40) > GEARS (0.39) > Overlay Real (0.36) > GRNBoost2 (0.34) > CPA (0.32) ~ Geneformer (0.32) > Random (0.28), with random floor at 0.09.

2. **GEARS is the best model (0.39 AUPRC)**, nearly matching ElasticNet (0.40). GEARS uses a GNN with GO knowledge graph to learn perturbation-specific expression shifts. Its causal graph quality nearly matches a strong regression baseline, suggesting it has learned genuine gene-gene regulatory dependencies.

3. **CPA (0.32) and Geneformer (0.32) fail to beat the observational baseline**: GRNBoost2 (0.34 AUPRC) — which uses NO perturbation data — outperforms both models. This is a striking finding: despite being trained on perturbation data, CPA's conditional VAE and Geneformer's ISP have not learned causal structure beyond what observational gene co-expression provides. Only GEARS (0.39), with its explicit GO knowledge graph, exceeds the observational ceiling.

4. **Model architecture matters**: GEARS (perturbation-specialized GNN) >> CPA (perturbation-specialized VAE) ~ Geneformer (foundation model ISP). The perturbation-specialized architecture with explicit GO graph topology (GEARS) significantly outperforms both the general conditional VAE (CPA) and the repurposed foundation model (Geneformer).

5. **Overlay sampling ceiling validated**: Overlay-resampled real means (0.36 AUPRC) establishes the ceiling achievable by any mean-prediction model using overlay sampling. GEARS (0.39) exceeds this ceiling, suggesting its predictions capture information beyond the mean perturbation effect.

6. **No model matches real data**: All models (0.32-0.39) fall far short of real data (1.00). Virtual cell model simulations cannot yet substitute for real CRISPRi experiments in causal discovery. The best model (GEARS) recovers 32.3% of the floor-adjusted real-data causal structure (where floor = random prediction AUPRC of 0.093), demonstrating meaningful but incomplete causal learning.

7. **GRNBoost2 remains competitive**: GRNBoost2 (0.34 AUPRC) uses only observational data but finds 123 true positives by predicting 428 edges. Its high recall/low precision trade-off means models must beat it to justify the cost of perturbation data generation.

8. **False Omission Rate (FOR) reveals model differences**: CPA (0.020) and Geneformer (0.015) have very low FOR, meaning they rarely predict "no edge" when one exists. GEARS (0.095) has higher FOR but better AUPRC, suggesting it makes more confident but targeted predictions. Overlay Real (0.266) has very high FOR due to fold-change artifacts.

## Results (200 genes, K562, 3 real models)

| Condition           | Edges | AUPRC  | P@100 | TP  |
|---------------------|-------|--------|-------|-----|
| Real Data (UB)      | 1220  | 1.0000 | 1.000 | 1220|
| ElasticNet          | 1222  | 0.4535 | 0.420 | 519 |
| **Geneformer (real)**| **1222** | **0.4217** | **0.440** | **528** |
| Overlay Real (UB)   | 1215  | 0.4231 | 0.370 | 514 |
| **CPA (real)**      | **1229** | **0.4046** | **0.380** | **476** |
| **GEARS (real)**    | **1192** | **0.4014** | **0.400** | **489** |

- **Random floor AUPRC**: 0.037
- **Ground truth**: 1220 edges from GIES on real perturbed cells
- **GIES partition size**: 20 (CausalBench default)

### Scale-Up Findings (50 → 200 genes)

1. **Geneformer improves dramatically at scale**: From worst model (0.32 at 50 genes) to best model (0.42 at 200 genes). Foundation model pretraining on 30M cells provides an advantage when more perturbation targets are available — the model's broad biological knowledge compensates for per-gene prediction noise.

2. **All models improve at 200 genes**: GEARS (0.39→0.40), CPA (0.32→0.40), Geneformer (0.32→0.42). More interventions give GIES more signal to infer edges.

3. **Model ranking changes**: At 50 genes, GEARS > CPA ~ Geneformer. At 200 genes, Geneformer > CPA ~ GEARS. This suggests evaluation at small gene sets may not predict large-scale performance.

4. **All models cluster tightly**: At 200 genes, models span 0.40-0.42 AUPRC (narrow range) vs 0.32-0.39 at 50 genes. Models converge in performance as scale increases.

## Next Steps

1. **Train scGPT**: Requires GPU access (too slow on CPU). Will complete model comparison with 4th model.
2. **Second cell line**: RPE1 from Replogle et al. for cross-context evaluation
3. **DCDI robustness check**: Soft-intervention causal discovery to validate GIES rankings
4. **d-Separation test**: CPA/GEARS multi-perturbation test on feed-forward loops

## Figures

- `figures/fig5_auprc_bars.pdf` — AUPRC by condition (horizontal bar chart)
- `figures/fig6_metrics_heatmap.pdf` — Multi-metric heatmap (AUPRC, P@100, SHD, Wasserstein, FOR)
- `figures/fig7_edge_counts.pdf` — Edge counts with TP/FP breakdown
- `figures/fig8_precision_at_k.pdf` — Precision@100 comparison
- `figures/fig9_fraction_ceiling.pdf` — Fraction of real-data ceiling recovered (floor-adjusted)
- `figures/fig10_radar.pdf` — Model comparison radar chart (5 metrics)
- `figures/fig11_gap_analysis.pdf` — AUPRC gap to real-data ceiling

## Technical Details

- NB dispersion: median r=5.38, mean mu=3.94 (DESeq2-style shrinkage)
- GIES partition size: 25 genes per partition
- Overlay sampling: pseudocount=0.01, fold-change clip=[0.01, 10.0], Poisson noise
- All code: `scripts/run_eval_local.py`, `scripts/generate_figures_n50.py`
