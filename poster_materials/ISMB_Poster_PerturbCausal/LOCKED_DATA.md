# LOCKED DATA — verify the poster against this

Every value below is verbatim from the FM4LS camera-ready (`PerturbCausal_paper_FM4LS_cameraready.pdf`). The poster must reproduce these exactly — no re-rounding, no invention.

## Identity

- **Title:** PerturbCausal: Virtual Cell Model Predictions Do Not Substitute for Real Perturb-seq in Causal Network Recovery
- **Author:** Aayan Alwani · William A. Shine Great Neck South High School
- **Poster venue:** ISMB 2026, Washington, D.C.

## Novel contribution (one line)

First benchmark testing whether VCM **predicted** Perturb-seq can substitute for **real** Perturb-seq one step **downstream** — in causal GRN recovery — instead of per-gene prediction. Method: hold the causal-discovery algorithm **fixed**, vary **only** the data source, compare recovered edge sets. Plus a joint-structure mechanism and an identifiability result.

## Methods

- Predictors: GEARS (graph NN), CPA (conditional VAE), Geneformer (104M transformer), STATE (Arc 2025); comparator = sparse-linear ElasticNet.
- Backends (held fixed): GIES (primary), inspre, PC, NOTEARS.
- Datasets: Replogle K562 (200-gene subset), RPE1 (cross-context), Norman 2019 (cross-paradigm CRISPRa). CRISPRi + CRISPRa.
- Metric: edge-set F1 vs the same algorithm's real-data network. Chance: per-backend permutation null [2.5, 97.5].

## Results — K562, N=200, 5 seeds (42, 123, 456, 789, 1024)

| Condition                       | Edge-set F1 |
| ------------------------------- | ----------- |
| Real Data (ceiling)             | 1.000       |
| Overlay-Real (upper bound)      | 0.429       |
| ElasticNet                      | 0.426       |
| Geneformer                      | 0.425       |
| CPA                             | 0.412       |
| GEARS                           | 0.406       |
| GRNBoost2 (observational)       | 0.098       |
| Technical-duplicate noise floor | 0.045       |

- No pairwise difference among predictive methods significant at p<0.05.
- TOST-equivalent at Δ=0.045 F1: CPA +0.014, GEARS +0.020, **Geneformer +0.0002** (print exactly, do NOT round to 0.001).

## Mechanism

- Real K562: only **2.7%** of (perturbation, gene) effects exceed |log2FC|=0.5.
- Deep models inflate to **~63%** (GEARS 63.2, CPA 63.4, Geneformer 63.5) = **23× inflation**, identical across GNN/VAE/transformer to within **0.3 percentage points** → objective-driven (per-gene MSE), not architecture-driven.
- CPA inter-perturbation cosine **0.84** vs real **0.08**.

## Backend + dataset invariance

- 6 cross-dataset cells: {RPE1, Norman} × {PC, NOTEARS, inspre}.
- Largest deep-model F1 = **0.440** (CPA under inspre on RPE1) — inside that backend's chance band.
- ElasticNet strongest in **4 of 6** cells. No deep VCM exceeds ElasticNet on any backend/dataset.
- inspre signal probe: CPA & Geneformer ACE fall below the detection threshold → recover **ZERO** edges where real data and ElasticNet recover **~12,000**.

## STATE (2025 foundation model)

- Per-seed F1 = **[0.462, 0.041, 0.048]**. Single-seed 0.462 does NOT replicate.

## Calibration (held-out 160/40 perturbation split)

- Quantile-matching calibration corrects the magnitude marginal exactly; magnitude drops onto the real = **0.114** line.
- Causal F1 does not close: GEARS **0.303→0.290**, CPA **0.309→0.289**, Geneformer **0.319→0.287** (Geneformer ends below ElasticNet = 0.425).

## Identifiability result (Proposition)

Conditional independence is invariant under per-gene monotone (rank-preserving) calibration → marginal rescaling provably cannot change the recovered graph. Residual = off-diagonal joint (precision-matrix support) structure → closing the gap needs a **training-time** intervention.

## Preliminary (gap is reachable)

- Learned joint-transform proxy (no-GPU): held-out CPA causal F1 **0.424 → 0.535 (+0.111)**.
- Naive covariance-Frobenius penalty alone fails a pre-registered multi-seed gate (**+0.003**).

## Why ElasticNet ties a 104M transformer

Inductive-bias matching — L1 sparsity matches real-response sparsity (**56.8%** of variance in the leading singular vector).

## Honest limitations

Ground truth internally consistent but not yet externally biologically validated; cross-dataset cells single-seed; N=200 scale.
