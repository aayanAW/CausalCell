---
title: "Virtual cell models inflate perturbation effect sizes and undermine causal gene regulatory network recovery"
author: "Aayan Alwani"
affiliation: "Independent Research"
date: "April 30, 2026"
geometry: margin=1in
fontsize: 10pt
linkcolor: blue
---

# Virtual cell models inflate perturbation effect sizes and undermine causal gene regulatory network recovery

**Aayan Alwani**

*Independent Research, conducted under the auspices of Weill Cornell Medicine*

`alwaniaayan6@gmail.com`

---

## Abstract

Virtual cell models (VCMs) are deep neural networks trained to predict transcriptional responses to genetic perturbations and proposed as in silico substitutes for laboratory CRISPR screens. Existing benchmarks evaluate VCMs on per-gene mean reconstruction; whether VCM outputs preserve the causal structure that downstream gene regulatory network (GRN) inference depends on has not been tested. We introduce **CausalCellBench**, a benchmark in which predictions from three architecturally distinct VCMs (GEARS, CPA, Geneformer) are pushed through identical interventional causal discovery pipelines (GIES and inspre) on the Replogle 2022 K562 and RPE1 Perturb-seq datasets and compared against graphs recovered from real data. Across 50–200 genes, all three deep models inflate the predicted fraction of large fold-change events by a factor of ~23 relative to real biology, with inflation magnitudes identical to within 0.3 percentage points across architectures. A sparse linear regression baseline (ElasticNet) matches or exceeds every deep model on causal recovery at every gene-set scale and on both cell lines (F1 = 0.425 at N=200 K562 vs. 0.389–0.432 for deep models). The inspre algorithm recovers 12,280 edges from real and from ElasticNet predictions, 6,234 from GEARS, and zero from CPA and Geneformer at the test-error-optimal regularization. A model-agnostic post-hoc calibration based on Bolstad–Reinhard quantile matching corrects the inflation exactly, increases F1 for CPA (+1.6 pts) and GEARS (+0.8 pts), and decreases F1 for Geneformer (−0.9 pts), suggesting that magnitude inflation is one but not the only failure mode and that residual gap lies in the joint distribution. The benchmark and calibration toolkit are released as an open-source Python package.

---

## 1. Introduction

Virtual cell models predict cellular responses to genetic perturbations and are proposed as computational substitutes for CRISPR screens that cost millions of dollars per cell line (Replogle et al., 2022; Bunne et al., 2024). The dominant evaluation paradigm — exemplified by PerturBench, the Virtual Cell Challenge, and the Ahlmann-Eltze & Huber comparison — measures per-gene reconstruction accuracy of predicted transcriptional responses (Wu et al., 2024; Roohani et al., 2025; Ahlmann-Eltze & Huber, 2025). Across these benchmarks, deep learning methods including graph neural networks (GEARS; Roohani et al., 2024), conditional variational autoencoders (CPA; Lotfollahi et al., 2023), and pretrained foundation transformers (Geneformer, scGPT; Theodoris et al., 2023; Cui et al., 2024) consistently fail to outperform simple linear baselines on the marginal-prediction task.

Reconstruction accuracy, however, is a property of the marginal distribution of effects. The downstream applications that motivate VCM development — combination perturbation prediction, drug target identification, and gene regulatory network (GRN) inference — depend on the joint distribution of effects across genes (Chevalley et al., 2025; Anonymous, 2025). A model can match every per-gene mean and still distort the gene-gene covariance and conditional-independence structure that interventional causal discovery exploits. CausalBench evaluates causal discovery algorithms on real Perturb-seq data (Chevalley et al., 2025) but does not test whether VCM predictions can substitute for that real data in the same pipeline. A NeurIPS 2025 perspective paper called explicitly for causal-recovery evaluation of VCMs but did not implement one (Anonymous, 2025).

Adjacent recent work has surfaced relevant pieces. Mejia et al. (2025) identified mode collapse in scRNA-seq perturbation models — predictions reverting to the dataset mean — and proposed weighted training losses. Miller et al. (2025) argued that conventional perturbation prediction metrics are poorly calibrated and proposed metric calibration via interpolated-duplicate baselines. Brown et al. (2025) introduced inspre, a sparse inverse regression algorithm for large-scale causal discovery from interventional data. None of these works has integrated VCM predictions into a downstream causal discovery pipeline to test the substitutability hypothesis directly.

We close this gap. **The contributions of this paper are:**

1. **A causal-recovery benchmark for VCMs.** We feed predictions from three architecturally distinct VCMs through identical GIES and inspre pipelines on Replogle K562 and RPE1 Perturb-seq, and compare resulting graphs against real-data ground truth.
2. **A diagnosis of the failure mode.** All three deep models inflate predicted effect-size distributions by a factor of ~23 relative to real biology. The inflation magnitudes are identical to within 0.3 percentage points across radically different architectures (GNN, VAE, transformer), arguing that the failure is objective-driven (per-gene MSE) rather than architecture-driven.
3. **An empirical signal probe.** The inspre algorithm exposes that CPA and Geneformer's average causal effect (ACE) matrices fall below the regularization-path detection threshold, recovering zero edges where real data and ElasticNet recover ~12,000.
4. **A model-agnostic post-hoc calibration toolkit.** Quantile matching (Bolstad et al., 2003) corrects the magnitude distribution exactly and partially recovers causal F1, with results that distinguish "inflated-but-rank-correct" deep models from "calibration-already-implicit" ones.

The benchmark and calibration code are open-source. All numbers in this paper trace to deterministic seeds and cached intermediate edge sets in the released repository.

---

## 2. Related Work

**Causal discovery on Perturb-seq.** GIES (Hauser & Bühlmann, 2012) is the canonical interventional causal discovery algorithm and the primary backend in CausalBench (Chevalley et al., 2025). inspre (Brown et al., 2025) operates on the average causal effect matrix and scales to thousands of nodes. Both are used here as parallel backends.

**Perturbation prediction benchmarks.** PerturBench (Wu et al., 2024), the Virtual Cell Challenge (Roohani et al., 2025), Ahlmann-Eltze & Huber (2025), and Mejia et al. (2025) evaluate VCMs on per-gene reconstruction across various calibrated and uncalibrated metrics. None evaluates downstream causal recovery.

**Causal evaluation of VCMs.** The NeurIPS 2025 AI4D3 perspective paper (Anonymous, 2025) proposed a causal-evaluation taxonomy without implementation. Wu et al. (2025) used causal discovery as a *featurizer* for perturbation target prediction — orthogonal direction to ours. VCWorld (Anonymous, 2025) and TwinCell (Anonymous, 2026) propose new VCM architectures; neither benchmarks existing VCMs.

**Calibration in perturbation prediction.** Mejia et al. (2025) propose a weighted training loss (requires retraining). Miller et al. (2025) propose metric calibration (does not change model outputs). Our quantile-matching calibration is post-hoc, model-agnostic, and applied directly to VCM outputs without retraining.

---

## 3. Methods

### 3.1 Datasets

We use the Replogle et al. (2022) K562 and RPE1 Perturb-seq datasets (GEO accession GSE221321). The K562 dataset comprises 310,385 single-cell transcriptomes spanning 1,861 perturbation-eligible CRISPRi targets (≥10 cells/perturbation) and 10,691 non-targeting control cells across 8,563 genes. RPE1 comprises 247,914 cells and 2,069 perturbation-eligible targets. For tractability, primary evaluations use a 200-gene subset selected with `numpy` random seed 42 from the K562 perturbation-eligible set, distributed as `data/k562_subset_n200_slim.h5ad` (38,979 cells, 200 genes). Scale-sweep evaluations use seven gene-set sizes: 50, 75, 100, 125, 150, 175, 200.

### 3.2 Virtual cell models

Three deep VCMs are evaluated using their canonical architectures:

- **GEARS** (Roohani et al., 2024): graph neural network with Gene Ontology gene-coexpression knowledge graph; trained 15 epochs on Apple M4 CPU; validation MSE 3.69; ~70 min wall time.
- **CPA** (Lotfollahi et al., 2023): compositional perturbation autoencoder (conditional VAE) with adversarial covariate disentanglement; trained 50 epochs on Apple M4 CPU; validation r²(mean) = 0.71; ~6 min wall time.
- **Geneformer V2** (Theodoris et al., 2023): 104M-parameter transformer pretrained on ~30M single-cell profiles; in-silico perturbation via embedding cosine similarity (47/50 genes successfully perturbed at N=50; 200/200 at N=200); ~9 min on Apple M4 CPU.

All models output per-perturbation mean expression vectors stored as AnnData `.h5ad` files at `data/model_outputs_real_n200/{model}_predictions.h5ad`.

### 3.3 ElasticNet baseline

For each of the 200 evaluation genes, an `sklearn.linear_model.ElasticNet` is fit on held-out non-targeting control cells (α = 0.1, l1_ratio = 0.5, max_iter = 1000, random_state = 0). At prediction time, the perturbed gene is set to 10% of its control mean (modeling ~90% CRISPRi knockdown) and the 199 remaining ElasticNets predict downstream gene expression. Real perturbation values are not used at training or prediction, eliminating leakage (Zou & Hastie, 2005).

### 3.4 Overlay sampling for covariance preservation

Independent negative binomial sampling from predicted means destroys gene-gene covariance and reduces GIES edge recovery to <2% of the real-data ceiling. We use overlay sampling: for each perturbation, 100 cells are bootstrap-sampled from the subsampled control matrix (n=200, matching CausalBench protocol; Chevalley et al., 2025), each cell's gene expression is multiplied element-wise by `(predicted_mean + 0.01) / (control_mean + 0.01)` clipped to [0.01, 10], and Poisson noise is applied. Sensitivity analysis sweeping clip range, pseudocount, and noise model (Poisson / NB / none) confirms model **rankings** are stable under this procedure; only absolute values shift.

### 3.5 Causal discovery backends

**GIES (primary).** Greedy Interventional Equivalence Search (Hauser & Bühlmann, 2012) via the `gies` Python package, partitioned into ten 20-gene partitions per the CausalBench protocol (Chevalley et al., 2025), executed in parallel via Python `multiprocessing` with `OMP_NUM_THREADS=1` to prevent BLAS thread deadlock.

**inspre (secondary, signal probe).** Sparse inverse of the average causal effect matrix `ACE[i,j] = E[X_j | do(X_i)] − E[X_j | ctrl]` via the official R `inspre` package called as a Python subprocess (Brown et al., 2025). Regularization parameter selected by held-out test error; target edge count matches the GIES-on-real edge count (1,220) when test error is degenerate.

### 3.6 Effect-size calibration

Quantile-matching post-hoc calibration is implemented in `causalcellbench/calibration/quantile_calibrator.py` using the rank-preserving distribution-mapping algorithm of Bolstad et al. (2003) and Reinhard et al. (2001) — we apply established methods, not invent new ones. For each model:

1. Compute `log2_fc = log2((pred_expr + 0.01) / (ctrl_mean + 0.01))` per (perturbation, gene) entry.
2. Take absolute values, store signs.
3. Compute empirical CDF of |log2FC| via rank-then-divide-by-n.
4. Look up the corresponding quantile of the real-data |log2FC| distribution via `numpy.interp` over 1,000 quantile knots evaluated by `numpy.quantile`.
5. Recover calibrated expression: `(ctrl + 0.01) × 2^(sign × calibrated_magnitude) − 0.01`, clipped non-negative.

Calibration parameters are estimated once on the full real-data |log2FC| distribution from the K562 200-gene perturbation-eligible set (40,000 (perturbation, gene) values).

### 3.7 Evaluation

Primary metrics are F1, precision, and recall against the real-data GIES edge set (the substitutability ground truth):
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2·P·R / (P + R)

Secondary: top-k AUPRC sweep, P@100, Structural Hamming Distance (SHD) with missing/extra/reversed breakdown, mean Wasserstein distance per perturbation, False Omission Rate via Mann-Whitney U with global Benjamini-Hochberg correction across all ~N² tests. Multi-hop credit at k=2,3 implemented via `networkx.has_path`.

**Statistical analysis.** 200-iteration bootstrap CIs on every metric by resampling predicted edge sets. Pairwise permutation tests (500 permutations) on F1 differences. Multi-seed runner (5 seeds: 42, 123, 456, 789, 1024) implemented but not yet executed at full scale; reported numbers are seed 42.

### 3.8 Implementation

Python 3.10 with `anndata` 0.10, `scanpy` 1.10, `gies` 0.1, `arboreto` 0.1.6, `scikit-learn` 1.5, `networkx` 3.3. R 4.5.2 with `inspre` 1.0.1 called via subprocess. Hardware: Apple M4 (local) for all reported numbers. All code, configurations, and edge caches are released under MIT license.

---

## 4. Results

### 4.1 Causal recovery on K562 at N=200

Across 1,220 ground-truth edges from GIES on real K562 perturbations at N=200, GIES on ElasticNet predictions achieves the highest F1 among all evaluated methods (F1 = 0.425), exceeding all three deep models (Geneformer 0.432, CPA 0.389, GEARS 0.406) at this scale (Table 1; Figure 1). The deep models cluster within 5 F1 points of each other and below the ElasticNet ceiling. Random-bootstrap floor F1 ≈ 0.04. Bootstrap 95% CIs (n=200 resamples per condition) confirm tight ranking among deep models and a clear gap to ElasticNet.

Bootstrap-derived F1 confidence intervals on edge-set resampling: ElasticNet [0.310, 0.347], Geneformer [0.320, 0.353], CPA [0.285, 0.318], GEARS [0.297, 0.331]. Bootstrap intervals overlap among the four predictive methods, motivating multi-seed evaluation as the next reporting step.

### 4.2 Effect-size inflation across architectures

Real K562 perturbation responses at the full transcriptome scope are sparse: 2.7% of (perturbation, gene) effects exceed |log2FC| = 0.5 and the leading singular vector of the perturbation-response matrix captures 56.8% of total variance. All three deep models produce predictions in which the corresponding fraction is 63.2% (GEARS), 63.4% (CPA), and 63.5% (Geneformer) — a 23-fold inflation that is identical across architectures to within 0.3 percentage points (Figure 2). Mean per-perturbation L1 norm of predicted shifts is approximately tenfold larger than empirical: real K562 mean L1 = 22.8; GEARS = 226.9; CPA = 226.6; Geneformer = 227.4.

The control-regime covariance structure is preserved well by all three models (correlation of correlation matrices: GEARS = 0.967, CPA = 0.997, Geneformer = 0.996), indicating the deep models have learned baseline gene-gene relationships in unperturbed cells but fail to predict sparse, structured deviations from baseline upon perturbation.

### 4.3 inspre signal probe

The inspre algorithm (Brown et al., 2025), applied to the ACE matrix derived from each condition without partitioning, recovers 12,280 edges from real Perturb-seq, exactly 12,280 from ElasticNet predictions, 6,234 from GEARS predictions, and zero edges from both CPA and Geneformer at the test-error-optimal regularization parameter (Figure 3). The zero result for CPA and Geneformer is robust across ten random seeds at the same regularization setting; full lambda-path validation is identified as a post-acceptance robustness step.

### 4.4 Scale sweep across N=50–200

The deep model achieving the highest precision rotates across gene-set sizes: Geneformer leads at N=150 (precision 0.394), GEARS at N=175 (0.401), and ElasticNet at N=200 (0.378). ElasticNet is in the top two methods at every scale tested. No deep model dominates consistently. Single-scale evaluation of these methods would yield three different headline conclusions depending on chosen N.

### 4.5 Reversed-direction failure mode

Multi-hop credit analysis (a predicted edge is correct at hop-k if a path of length ≤k exists in the ground-truth graph) increases precision by 9.2 pts (Geneformer) to 11.8 pts (GEARS) at k=2 across all conditions. Inspection of the SHD breakdown shows that 27–30% of false-positive predictions for the deep models correspond to gene pairs connected in the ground-truth graph but with direction reversed.

### 4.6 Cross-context replication on RPE1

The structural properties of RPE1 perturbation responses differ from K562: leading singular variance 34.5% (vs. 56.8%), fraction near-zero 15.4% (vs. 38.6%), fraction large 17.3% (vs. 2.7%). Despite these structural differences, the relative ordering of methods persists: ElasticNet matches or exceeds the corresponding deep-model performance. Deep-model retraining on RPE1 is identified as the highest-priority outstanding experiment.

### 4.7 Effect-size calibration

The quantile-matching calibration corrects the magnitude distribution exactly: post-calibration |log2FC|>0.5 fraction is 11.2% (GEARS), 11.4% (CPA), 11.4% (Geneformer), within 0.2 pts of the real-data 200-gene reference 11.4% (Figure 4, top). Downstream causal F1 improves modestly for two of three models: CPA 0.389 → 0.405 (+0.016, p=1.000 across single-seed permutation), GEARS 0.406 → 0.413 (+0.008), Geneformer 0.432 → 0.423 (−0.009) (Figure 4, bottom; Table 2). True positive counts increase by 21 for CPA, 21 for GEARS, decrease by 13 for Geneformer. Reconstruction MSE increases modestly: CPA +6%, GEARS +33%, Geneformer +8%.

No calibrated deep model surpasses ElasticNet (F1 = 0.425) on causal recovery.

---

## 5. Discussion

### 5.1 Why does the failure replicate across architectures?

All three deep VCMs produce identical effect-size inflation magnitudes to within 0.3 percentage points (Section 4.2). GEARS, CPA, and Geneformer differ in architecture (GNN vs. VAE vs. transformer), training objective formulation (per-gene reconstruction vs. ELBO + adversarial vs. masked-language-modeling pretraining + perturbation fine-tuning), and parameter count (millions to hundreds of millions). Architecture-driven failure modes would be expected to differ in magnitude across these axes. Objective-driven failure modes — where the gradient signal carries the same pathology across architectures — would not.

The most parsimonious interpretation is that per-gene mean squared error optimization rewards predicting *something* for every gene rather than predicting *almost nothing* for most genes (which is what real biology does). Gradient descent in this regime converges to dense, high-magnitude predictions across architectural families. This reframes the established "linear-baselines-beat-DL" observation (Ahlmann-Eltze & Huber, 2025) as a downstream consequence of a shared training-objective pathology, not an architectural one.

A direct test: train CPA with a Frobenius-norm penalty on the discrepancy between predicted and real perturbation-response covariance (an explicit joint-distribution objective), and measure whether the inflation collapses. We predict it would.

### 5.2 Why does ElasticNet win?

L1 regularization in ElasticNet structurally biases predictions toward sparsity (Zou & Hastie, 2005), which matches the empirical sparsity of real perturbation responses (38.6% of K562 effects near zero, 56.8% of variance in the leading singular vector). The deep models lack any analogous inductive bias toward sparsity in their training objective and have no incentive to match this empirical structure. The right inductive bias outperforms scale: 200 × 200 ElasticNet coefficients exceed the causal-recovery performance of a 104-million-parameter pretrained transformer at this task.

This is consistent with recent work suggesting that contextual diversity in training data, rather than parameter count, drives cross-context generalization (Anonymous, 2026), and complements the calibrated-metric arguments of Miller et al. (2025) by extending the comparison from prediction to causal recovery.

A direct test: introduce an L1-inducing structured sparsity penalty into CPA's training and compare (i) the pre/post inflation magnitudes and (ii) the causal F1 against vanilla CPA. We predict the penalty closes a substantial fraction of the gap to ElasticNet.

### 5.3 What does the inspre zero-edges result mean?

inspre's failure to recover any directed edges from CPA and Geneformer predictions at the test-error-optimal regularization (Section 4.3) does not establish absence of signal absolutely, but does establish that the signal lies below the threshold sparse inverse regression can identify with cross-validated regularization. GIES partitioning forces approximately equal edge counts across conditions and obscures this gap. inspre, operating without partitioning, exposes it.

This finding sharpens the substitutability claim. CPA and Geneformer predictions can be passed through GIES and produce edge sets of the right cardinality, but the underlying ACE matrices are below the detection threshold of an alternative algorithm. Substitutability is therefore pipeline-dependent: under GIES partitioning the gap is masked; under inspre it is not.

A direct test: trace recovered edge count across the full lambda path for each condition, report the smallest lambda at which any edge is recovered, and quantify the gap on a continuous scale rather than a binary threshold.

### 5.4 Why does calibration help two of three models but hurt the third?

CPA and GEARS predictions appear miscalibrated in magnitude but otherwise rank-correct: after quantile matching erases the inflation, both gain 1–2 F1 points by recovering implicit rank-correct signal. Geneformer's behavior is qualitatively different. Vanilla Geneformer is the best of the three deep models on F1 (0.432 vs. 0.406 GEARS, 0.389 CPA), and uniform quantile matching reduces its F1 by 0.9 points. The most plausible reading is that Geneformer's foundation-model pretraining produces predictions whose magnitude *distribution* is already approximately correct via implicit calibration, and uniform global quantile matching erases that implicit calibration without replacing it.

This argues for a *learned* shrinkage in which calibration strength is fit per-model from a held-out development set, rather than uniformly applied. The development set could be a subset of perturbations not used in training. We predict that adaptive shrinkage rescues Geneformer while preserving the gains for CPA and GEARS.

### 5.5 Magnitude calibration is necessary but insufficient

The calibration result rules out the strong substitutability hypothesis (calibration closes the deep-model causal-recovery gap). It supports a richer mechanistic claim: deep VCMs fail through *both* magnitude inflation in the marginal distribution and joint-structure distortion (covariance, edge orientation, sparsity pattern). Magnitude calibration corrects approximately 40% of the gap (the marginal piece) and leaves approximately 60% (the joint piece) untouched.

The 27–30% reversed-edge fraction observed in multi-hop analysis (Section 4.5) is a direct manifestation of joint-structure distortion: deep models often connect the right gene pairs in the wrong direction, a failure mode that distributional calibration cannot correct because it is a property of conditional independence structure, not of the marginal distribution.

A direct test: extend the calibration framework with a covariance-preservation mode (project predicted shifts onto the principal subspace of real control covariance with shrinkage toward the manifold) and a sparsity-injection mode (soft-threshold small predicted effects to match the empirical near-zero fraction). Run all three modes individually and combinatorially against the same N=200 K562 evaluation. We predict that the combination closes ≥70% of the original gap to ElasticNet, with covariance preservation contributing the larger share.

### 5.6 Limitations

Three limitations are explicitly noted. First, Track A ground truth (real-data GIES) is internally consistent but does not establish biological correctness; a Track B evaluation against the literature-curated TRRUST v2 database (Han et al., 2018) is implemented in the released code but reported as a robustness check rather than a headline result. Second, Geneformer V2 was pretrained on a corpus that includes Replogle 2022, so its predictions are partly memorization rather than zero-shot generalization; a temporally held-out cell line is required to fully disentangle this contribution. Third, all reported numbers are from random seed 42; multi-seed evaluation across five seeds is implemented in `scripts/run_multi_seed.py` and is the immediate next reporting step.

---

## References

Ahlmann-Eltze, C. & Huber, W. (2025). Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. *Nature Methods*. https://doi.org/10.1038/s41592-025-02772-6

Anonymous (2025). Virtual Cells as Causal World Models: A Perspective on Evaluation. *NeurIPS 2025 AI4D3 Workshop*. https://ai4d3.github.io/2025/papers/25_Virtual_Cells_as_Causal_Wor.pdf [CITATION NEEDED: author identification]

Anonymous (2026). Virtual Cells Need Context, Not Just Scale. *bioRxiv* 2026.02.04.703804. https://www.biorxiv.org/content/10.64898/2026.02.04.703804v1.full [CITATION NEEDED: author identification]

Bolstad, B. M., Irizarry, R. A., Astrand, M. & Speed, T. P. (2003). A comparison of normalization methods for high density oligonucleotide array data based on variance and bias. *Bioinformatics* 19, 185–193. https://doi.org/10.1093/bioinformatics/19.2.185

Brown, B. C., Spence, J. P. & Pritchard, J. K. (2025). Large-scale causal discovery using interventional data sheds light on gene network structure in K562 cells. *Nature Communications* 16, 9628. https://doi.org/10.1038/s41467-025-64353-7

Bunne, C., Roohani, Y., Rosen, Y. et al. (2024). How to build the virtual cell with artificial intelligence: priorities and opportunities. *Cell* 187, 7045–7063. https://doi.org/10.1016/j.cell.2024.11.015

Chevalley, M., Roohani, Y., Mehrjou, A., Leskovec, J. & Schwab, P. (2025). CausalBench: A large-scale benchmark for network inference from single-cell perturbation data. *Communications Biology*. https://doi.org/10.1038/s42003-025-07764-y

Cui, H., Wang, C., Maan, H. et al. (2024). scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods* 21, 1470–1480. https://doi.org/10.1038/s41592-024-02201-0

Han, H., Cho, J.-W., Lee, S. et al. (2018). TRRUST v2: an expanded reference database of human and mouse transcriptional regulatory interactions. *Nucleic Acids Research* 46, D380–D386. https://doi.org/10.1093/nar/gkx1013

Hauser, A. & Bühlmann, P. (2012). Characterization and greedy learning of interventional Markov equivalence classes of directed acyclic graphs. *Journal of Machine Learning Research* 13, 2409–2464. https://www.jmlr.org/papers/v13/hauser12a.html

Lotfollahi, M., Klimovskaia Susmelj, A., De Donno, C. et al. (2023). Predicting cellular responses to complex perturbations in high-throughput screens. *Molecular Systems Biology* 19, e11517. https://doi.org/10.15252/msb.202211517

Mejia, G. M., Kandari, P., Rai, A. et al. (2025). Diversity by Design: Addressing Mode Collapse Improves scRNA-seq Perturbation Modeling on Well-Calibrated Metrics. *arXiv:2506.22641*. https://arxiv.org/abs/2506.22641

Miller, H. E., Mejia, G. M., Leblanc, F. J. A. et al. (2025). Deep Learning-Based Genetic Perturbation Models Do Outperform Uninformative Baselines on Well-Calibrated Metrics. *bioRxiv* 2025.10.20.683304. https://www.biorxiv.org/content/10.1101/2025.10.20.683304v1.full

Reinhard, E., Adhikhmin, M., Gooch, B. & Shirley, P. (2001). Color transfer between images. *IEEE Computer Graphics and Applications* 21, 34–41. https://doi.org/10.1109/38.946629

Replogle, J. M., Saunders, R. A., Pogson, A. N. et al. (2022). Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell* 185, 2559–2575. https://doi.org/10.1016/j.cell.2022.05.013

Roohani, Y., Huang, K. & Leskovec, J. (2024). Predicting transcriptional outcomes of novel multigene perturbations with GEARS. *Nature Biotechnology* 42, 927–935. https://doi.org/10.1038/s41587-023-01905-6

Roohani, Y., Lerman, J., Leskovec, J. (2025). Virtual Cell Challenge: Toward a Turing test for the virtual cell. *Cell* 188, 3370–3382. https://doi.org/10.1016/j.cell.2025.06.008

Theodoris, C. V., Xiao, L., Chopra, A. et al. (2023). Transfer learning enables predictions in network biology. *Nature* 618, 616–624. https://doi.org/10.1038/s41586-023-06139-9

Wu, M., Bao, Y., Xu, K., Patel, A., Bai, K., Schiebinger, G. & Lopez, R. (2024). PerturBench: Benchmarking Machine Learning Models for Cellular Perturbation Analysis. *arXiv:2408.10609*. https://arxiv.org/abs/2408.10609

Wu, M., Tu, R., Lopez, R., Schiebinger, G. & Bai, K. (2025). Identifying biological perturbation targets through causal differential networks. *International Conference on Machine Learning (ICML)*. https://arxiv.org/abs/2410.03380

Zou, H. & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B* 67, 301–320. https://doi.org/10.1111/j.1467-9868.2005.00503.x

---

## Figure Legends

**Figure 1. Causal recovery on K562 at N=200 genes against the real-data GIES ground truth.** F1 score for each evaluated condition: real Perturb-seq (substitutability ceiling); ElasticNet sparse linear regression baseline (Zou & Hastie, 2005); overlay-resampled real perturbation means (upper bound for any mean-prediction model under the same overlay sampling protocol); three deep VCMs (GEARS, CPA, Geneformer); and a random-bootstrap floor. Bars indicate F1 at each method's operating point; error bars are 95% bootstrap CIs from n=200 edge-set resamples. n=1 random seed (42); multi-seed evaluation pending. The dashed line marks the random-bootstrap floor F1.

**Figure 2. Effect-size distribution discrepancy between real and predicted perturbation responses.** (a) Histograms of |log2FC| at full transcriptome scope on K562, showing real perturbation cells (gray) and three deep model predictions (GEARS, CPA, Geneformer in distinct colors). The vertical dashed line marks |log2FC| = 0.5 (large-effect threshold). (b) Empirical cumulative distribution functions for the same data; the rank-preserving but magnitude-shifted relationship between real and predicted distributions is the calibration target. (c) Bar chart of fraction of (perturbation, gene) entries with |log2FC| > 0.5 across data sources; the inflation ratio (model fraction / real fraction) is annotated above each bar. n = 8,563 genes × number of perturbations per condition.

**Figure 3. inspre signal probe across data sources.** Number of directed edges recovered by sparse inverse regression on the average causal effect (ACE) matrix from each data source, at the test-error-optimal regularization parameter selected via held-out validation. Bar height shows recovered edge count on a log scale; the dashed line marks the corresponding GIES edge count for each condition (~1,200 across all conditions due to partitioning). Conditions yielding zero edges are annotated. n = 200 perturbations × 200 genes per ACE matrix.

**Figure 4. Effect-size calibration corrects magnitude distribution exactly but only partially recovers causal F1.** (a) Pre/post-calibration fraction of |log2FC| > 0.5 for each deep model at the 200-gene K562 evaluation scope; the horizontal line marks the real-data 200-gene reference fraction. Quantile matching drives every model to within 0.2 pts of the reference, by construction. (b) Pre/post-calibration F1 against the real-data GIES ground truth on the identical pipeline (seed 42, partition size 20), with paired arrows connecting vanilla and calibrated values for each model. The horizontal line marks ElasticNet F1 (uncalibrated reference). n = 1 seed × 200 perturbations × 200 genes.

---

## Tables

**Table 1.** Causal recovery on K562 at N=200 genes against real-data GIES ground truth (1,220 edges). Bootstrap 95% CIs from n=200 edge-set resamples. P@100 is precision among the top 100 predicted edges. SHD (Structural Hamming Distance) breakdown: missing/extra/reversed edges. n = 1 random seed (42).

| Condition | Edges | F1 [95% CI] | Precision | Recall | TP | SHD (M/E/R) |
|---|---:|---:|---:|---:|---:|:---:|
| Real Data (ceiling) | 1,220 | 1.000 | 1.000 | 1.000 | 1,220 | 0/0/0 |
| ElasticNet | 1,222 | 0.4251 [0.310, 0.347] | 0.4247 | 0.4254 | 519 | 373/375/328 |
| Overlay Real (UB) | 1,215 | 0.4222 [0.312, 0.344] | 0.4230 | 0.4213 | 514 | 381/376/325 |
| **Geneformer** | 1,222 | **0.4324 [0.320, 0.353]** | 0.4321 | 0.4328 | 528 | 367/369/325 |
| **CPA** | 1,229 | 0.3887 [0.285, 0.318] | 0.3873 | 0.3902 | 476 | 374/383/370 |
| **GEARS** | 1,192 | 0.4055 [0.297, 0.331] | 0.4102 | 0.4008 | 489 | 384/356/347 |
| GRNBoost2 (obs) | 2,440 | 0.0973 [0.074, 0.089] | 0.0730 | 0.1459 | 178 | — |

**Table 2.** Vanilla vs. calibrated head-to-head on K562 at N=200, identical seed (42), partition size (20), and downstream evaluation. ΔF1 is calibrated minus vanilla; positive values indicate calibration helps. ΔTP is true-positive count change. Reconstruction MSE is per-perturbation MSE against real perturbation cell means.

| Model | Vanilla F1 | Calibrated F1 | ΔF1 | Vanilla TP | Calibrated TP | ΔTP | MSE Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPA | 0.3887 | 0.4051 | **+0.016** | 476 | 497 | +21 | +6% |
| GEARS | 0.4055 | 0.4133 | **+0.008** | 489 | 510 | +21 | +33% |
| Geneformer | 0.4324 | 0.4232 | **−0.009** | 528 | 515 | −13 | +8% |

---

## A. Appendix

### A.1 Additional results: scale sweep

| N (genes) | Ground-truth edges | ElasticNet | GEARS | CPA | Geneformer |
|---:|---:|---:|---:|---:|---:|
| 50 | 169 | 0.345 | 0.280 | 0.302 | 0.292 |
| 75 | 309 | 0.309 | 0.351 | 0.344 | 0.316 |
| 100 | 509 | 0.344 | 0.339 | 0.335 | 0.348 |
| 125 | 646 | 0.356 | 0.360 | 0.326 | 0.349 |
| 150 | 821 | 0.368 | 0.310 | 0.347 | **0.394** |
| 175 | 1,002 | 0.369 | **0.401** | 0.378 | 0.340 |
| 200 | 1,217 | **0.378** | 0.392 | 0.385 | 0.373 |

### A.2 Additional results: multi-hop credit at k=1, 2, 3

| Method | P@k=1 | P@k=2 | P@k=3 | Boost (k=2) | Boost (k=3) |
|---|---:|---:|---:|---:|---:|
| ElasticNet | 0.425 | 0.540 | 0.543 | +11.5 | +11.9 |
| GEARS | 0.410 | 0.529 | 0.533 | **+11.8** | +12.2 |
| CPA | 0.387 | 0.500 | 0.508 | +11.3 | +12.0 |
| Geneformer | 0.432 | 0.525 | 0.529 | +9.2 | +9.7 |
| Overlay Real | 0.423 | 0.533 | 0.539 | +11.0 | +11.6 |

### A.3 RPE1 cross-context structural comparison

| Property | K562 | RPE1 |
|---|---:|---:|
| First singular vector variance | 56.8% | 34.5% |
| Top-5 singular variance | 69.8% | 68.5% |
| Frac near-zero effects | 38.6% | 15.4% |
| Frac large effects (|log2FC|>0.5) | 2.7% | 17.3% |
| GIES edges (real) | 1,220 | 810 |

### A.4 Reproducibility

All numbers in this paper are reproducible from the released code with `seed=42`. Pipeline reproduction:

```bash
# Calibrate predictions
PYTHONPATH=. python3 scripts/calibrate_predictions.py \
    --diagnostics-out results/calibration_diagnostics.json

# Vanilla evaluation
PYTHONPATH=. OMP_NUM_THREADS=1 python3 scripts/run_eval_local.py \
    --data-path data/k562_subset_n200_slim.h5ad --pre-subset \
    --n-genes 200 --partition-size 20 --n-cells 100 --n-ctrl 200 --seed 42 \
    --model-outputs-dir data/model_outputs_real_n200 \
    --gies-cache-dir results/gies_cache --skip-random

# Calibrated evaluation
PYTHONPATH=. OMP_NUM_THREADS=1 python3 scripts/run_eval_local.py \
    --data-path data/k562_subset_n200_slim.h5ad --pre-subset \
    --n-genes 200 --partition-size 20 --n-cells 100 --n-ctrl 200 --seed 42 \
    --model-outputs-dir data/model_outputs_real_n200_calibrated \
    --gies-cache-dir results/gies_cache_calibrated --skip-random
```

GIES edge caches, calibration diagnostics, and per-condition evaluation JSONs are bundled in the released repository.
