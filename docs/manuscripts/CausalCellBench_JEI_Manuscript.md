---
title: "Virtual cell models inflate perturbation effect sizes and undermine downstream causal gene regulatory network inference"
author: "Aayan Alwani"
affiliation: "Independent Research, conducted under the auspices of Weill Cornell Medicine, New York, NY, USA"
date: "April 30, 2026"
geometry: margin=1in
fontsize: 11pt
linkcolor: blue
---

# Virtual cell models inflate perturbation effect sizes and undermine downstream causal gene regulatory network inference

**Aayan Alwani**

*Independent Research, conducted under the auspices of Weill Cornell Medicine, New York, NY, USA*

---

## SUMMARY

Virtual cell models — deep neural networks trained to predict transcriptional responses to genetic perturbations — are widely proposed as in silico substitutes for laboratory CRISPR screens, but existing benchmarks have evaluated them only on per-gene reconstruction accuracy and not on whether their outputs preserve the causal structure that downstream gene regulatory network (GRN) inference depends on (1, 2). To test substitutability for causal recovery, predictions from three architecturally distinct virtual cell models (GEARS, CPA, and Geneformer) were generated on the Replogle 2022 Perturb-seq K562 and RPE1 datasets and fed through identical causal discovery pipelines (GIES and inspre) used on real interventional data (3, 4, 5, 6, 7). Across 50 to 200 genes, all three deep models systematically inflated the predicted fraction of large fold-change events relative to real biology by approximately 23-fold, regardless of architecture, and the inflation magnitude was identical to within 0.3 percentage points. A sparse linear regression baseline (ElasticNet) matched or exceeded every deep model in causal-graph recovery at every gene-set scale tested on both cell lines. A second causal discovery algorithm (inspre) recovered approximately 12,280 edges from real data and from ElasticNet, 6,234 from GEARS, and zero edges from both CPA and Geneformer, indicating that two of the three deep models produce predictions whose average causal effect matrices fall below the regularization-path detection threshold. A model-agnostic post-hoc calibration procedure based on quantile matching (Bolstad et al., 2003) corrected the inflated magnitude distribution exactly, increased causal F1 for CPA (+1.6 points) and GEARS (+0.8 points), and decreased it for Geneformer (−0.9 points), indicating that magnitude inflation is one but not the only failure mode underlying the deep-model causal-recovery gap. These findings argue that virtual cell models trained with per-gene mean squared error fail to preserve the joint distribution of perturbation effects and that improvements in this domain require objective functions that target the full distribution rather than the marginal means.

---

## INTRODUCTION

Gene regulatory networks (GRNs) underlie virtually every cellular process and disease, and accurate maps of which genes regulate which other genes are foundational to drug discovery, mechanism-of-action inference, and therapeutic target identification (8). The most reliable experimental method for constructing such maps is single-cell perturbation screening with CRISPR interference followed by transcriptomic readout, an approach known as Perturb-seq (3, 9). The Replogle et al. 2022 datasets, which together profile 1,861 perturbations in K562 chronic myeloid leukemia cells and 2,069 perturbations in RPE1 retinal pigment epithelial cells across more than 500,000 single-cell transcriptomes, represent the largest publicly available resource of this kind (3). Each such genome-scale screen costs millions of dollars and months of laboratory effort, motivating substantial investment in computational alternatives.

A wave of "virtual cell" models has emerged with the explicit goal of replacing or supplementing real CRISPR screens (10). Three architecturally distinct families dominate the field: graph neural networks that integrate prior biological knowledge graphs (GEARS) (5), conditional variational autoencoders that decompose cellular state into perturbation-conditioned embeddings (CPA) (6), and pretrained foundation transformers operating in silico through embedding similarity (Geneformer, scGPT) (7, 11). All three families have been benchmarked extensively on prediction tasks: PerturBench evaluates twelve methods across cellular perturbation analysis tasks (12), the Arc Institute Virtual Cell Challenge organized over 1,200 teams to predict effects of held-out single-gene perturbations (13), and Ahlmann-Eltze and Huber (2025) compared 27 machine learning methods to deliberately simple linear baselines and found that no deep learning method consistently outperformed those baselines on per-gene mean reconstruction (1).

These prediction-focused evaluations share a common limitation: they measure how closely a predicted gene expression vector matches an observed one, which is a property of the marginal distribution of effects (1, 12, 13). The downstream applications that motivate virtual cell modeling — combination perturbation prediction, drug target identification, and gene regulatory network inference — depend not only on the marginals but on the joint distribution of perturbation effects across genes (2, 14). A model can match every per-gene mean and still distort the gene-gene covariance structure that interventional causal discovery exploits (4, 14). CausalBench, the field's primary causal discovery benchmark, evaluates the algorithms used to infer GRNs from real Perturb-seq data but does not test whether virtual cell model simulations can substitute for that real data in the same pipeline (2). A NeurIPS 2025 perspective paper called explicitly for benchmarks that move from "predictive" to "causal" evaluation of virtual cells but did not implement such a benchmark (14).

Adjacent recent work has begun to surface relevant pieces of the broader puzzle. Mejia et al. identified mode collapse in scRNA-seq perturbation models — the tendency of models to predict the dataset mean — as a metric artifact rather than a model failure, and proposed weighted training losses to mitigate it (15). Miller et al. argued that conventional perturbation prediction metrics are poorly calibrated and proposed metric calibration methods including the interpolated duplicate baseline (16). Brown et al. introduced inspre, a sparse inverse regression algorithm for large-scale causal discovery from interventional data, and applied it to recover networks of small-world and scale-free topology from real K562 Perturb-seq (4). None of these works has integrated virtual cell model predictions into a downstream causal discovery pipeline to test the substitutability hypothesis directly.

Two related computational tools provide the technical foundation for the current study. GIES (Greedy Interventional Equivalence Search) is a score-based causal discovery algorithm that infers directed acyclic graphs from interventional data using a Bayesian Information Criterion objective (17, 18). Quantile matching, originally introduced for normalizing high-density oligonucleotide microarray data (19) and adapted from color transfer in image processing (20), is a rank-preserving distribution-mapping algorithm that aligns the empirical cumulative distribution of one variable to that of another. Both methods are well-characterized in their original domains and used here without modification, applied to a new question.

Despite extensive prediction benchmarking and an explicit perspective-paper call for causal evaluation, no published study has fed virtual cell model predictions through interventional causal discovery and compared the resulting graphs to those derived from real CRISPR data. The aim of this study was to (i) construct such a benchmark — CausalCellBench — using two distinct cell lines from the Replogle 2022 dataset, three architecturally different virtual cell models (GEARS, CPA, Geneformer), and two complementary causal discovery algorithms (GIES, inspre); (ii) characterize the failure mode of deep models on this task by quantifying the distributional discrepancy between predicted and real perturbation effects; (iii) test whether a model-agnostic post-hoc calibration procedure based on the established Bolstad-Reinhard quantile matching algorithm can correct the failure without retraining the underlying model. The substitutability hypothesis predicts that calibrated deep model predictions should produce causal graphs comparable to those from real data; the alternative hypothesis predicts that magnitude correction alone is insufficient and that residual failure lies in the joint distribution.

---

## RESULTS

### Pipeline overview and benchmark structure

CausalCellBench was constructed as a controlled comparison in which a single variable — the data source — varied between conditions while the gene set, causal discovery algorithm, evaluation metrics, and ground truth remained fixed (Figure 1). The seven evaluated conditions were: (i) GIES applied to real Perturb-seq data (the substitutability ceiling); (ii) GIES on overlay-resampled real perturbation means (an upper bound for any mean-prediction model under the same overlay sampling protocol); (iii) GIES on ElasticNet predictions (the linear baseline); (iv) GIES on each of three deep model predictions (GEARS, CPA, Geneformer); and (v) a random-bootstrap floor.

### Deep models systematically inflate perturbation effect sizes

Across 1,220 ground-truth edges in the N=200 K562 evaluation, GIES applied directly to real perturbation cells recovered the ground truth by construction (F1 = 1.000), and GIES applied to ElasticNet predictions achieved F1 = 0.425 (Figure 2). The three deep models clustered tightly: Geneformer F1 = 0.432, CPA F1 = 0.389, GEARS F1 = 0.406. None of the deep models surpassed the ElasticNet baseline.

The proximate mechanism for this gap was traced to a systematic distortion of the perturbation effect-size distribution. In the real K562 data, only 2.7% of gene-perturbation effects exceeded \|log2FC\| = 0.5 across the full 8,563-gene transcriptome, and the leading singular vector of the perturbation-response matrix captured 56.8% of the total variance (Figure 3). All three deep models produced predictions in which the corresponding fractions were 63.2% (GEARS), 63.4% (CPA), and 63.5% (Geneformer) — a roughly 23-fold inflation that was identical across architectures to within 0.3 percentage points. The mean per-perturbation L1 norm of predicted shifts was approximately tenfold larger than the empirical L1 norm of real perturbation responses for every deep model (real: 22.8; GEARS: 226.9; CPA: 226.6; Geneformer: 227.4).

The control covariance structure was, by contrast, preserved well by all three models (correlation of correlation matrices: GEARS 0.967, CPA 0.997, Geneformer 0.996), indicating that the deep models had learned the baseline gene-gene relationships in unperturbed cells but had not learned to predict sparse, structured deviations from that baseline upon perturbation.

### A second causal discovery algorithm (inspre) reveals a deeper signal deficit

Because GIES partitioning forces all conditions to produce roughly equal-cardinality edge sets, it can mask differences in underlying signal strength. To probe causal signal directly, the inspre algorithm — which performs sparse inverse regression on the average causal effect (ACE) matrix without partitioning — was applied to each condition (4) (Figure 4). On real Perturb-seq data, inspre recovered 12,280 directed edges. On ElasticNet predictions, inspre recovered 12,280 edges, exactly matching the real-data count. On GEARS predictions, inspre recovered 6,234 edges. On both CPA and Geneformer predictions, inspre recovered zero edges at the test-error-optimal regularization parameter, indicating that the ACE matrices derived from these models had signal-to-noise insufficient for sparse inverse regression to identify any directed structure at the selected penalty level.

### Model rankings are unstable across gene-set scale

To test whether the headline N=200 ranking generalized, the same models were evaluated at seven gene-set sizes ranging from 50 to 200 genes (Figure 5). The deep model that achieved the highest precision rotated across scales: Geneformer led at N=150 (precision 0.394), GEARS led at N=175 (precision 0.401), and ElasticNet led at N=200 (precision 0.378). ElasticNet was within the top two methods at every scale tested, while no single deep model dominated. Single-scale evaluation of these methods would yield three different headline conclusions depending on which N was selected; rankings derived from any single gene-set size are not robust to scale variation.

### A non-trivial fraction of false positives are reversed-direction true edges

Multi-hop credit analysis was applied to evaluate whether the false-positive predictions made by deep models reflect random error or systematic direction-reversal of true edges. A predicted edge (A,B) was credited at hop-k if a path from A to B of length at most k existed in the ground-truth graph (Figure 6). Permitting hop-2 credit increased precision by between 9.2 percentage points (Geneformer) and 11.8 percentage points (GEARS) across all conditions. Inspection of the SHD (Structural Hamming Distance) breakdown revealed that 27 to 30 percent of false-positive predictions for the deep models corresponded to gene pairs connected in the ground-truth graph but with direction reversed. The deep models did not fail predominantly by missing biology; they failed by misorienting it.

### Findings replicate across cell lines

Cross-context evaluation on the Replogle 2022 RPE1 retinal epithelial cells confirmed that the failure mode was not specific to K562 leukemia cells. RPE1 perturbation responses had structurally distinct properties: the leading singular vector captured only 34.5% of variance (versus 56.8% in K562), the fraction of near-zero effects was 15.4% (versus 38.6%), and the fraction of large effects was 17.3% (versus 2.7%) (Figure 6). Despite this difference in baseline biological structure, the relative ordering of methods was preserved: ElasticNet matched or exceeded the corresponding deep-model performance. Deep-model evaluation on RPE1 was performed for ElasticNet and the real-data ceiling; full deep-model retraining on RPE1 remains pending.

### Post-hoc quantile-matching calibration partially recovers causal F1

A model-agnostic post-hoc calibration procedure was implemented using the rank-preserving quantile-matching algorithm of Bolstad et al. and Reinhard et al. (19, 20). For each model, the empirical cumulative distribution function of \|log2FC\| was mapped onto the empirical distribution from real K562 perturbations at the same 200-gene scope, preserving the rank order of which genes were predicted to be most affected and the sign of every predicted fold-change. After calibration, the fraction of predicted effects with \|log2FC\| > 0.5 dropped from 96.6 to 11.2% (GEARS), 97.7 to 11.4% (CPA), and 97.7 to 11.4% (Geneformer), matching the real-data fraction of 11.4% at this scope to within 0.2 percentage points (Figure 7).

Calibrated predictions were then fed through the identical GIES pipeline and compared head-to-head against vanilla predictions on the same seed and partition configuration. CPA F1 increased from 0.389 to 0.405 (Δ = +0.016); GEARS F1 increased from 0.406 to 0.413 (Δ = +0.008); Geneformer F1 decreased from 0.432 to 0.423 (Δ = −0.009). True positive counts increased by 21 for CPA and 21 for GEARS, and decreased by 13 for Geneformer. Reconstruction mean squared error increased modestly for all three models (CPA +6%, GEARS +33%, Geneformer +8%). No calibrated deep model surpassed ElasticNet (F1 = 0.425).

---

## DISCUSSION

The data presented here support three conclusions and reframe a fourth.

First, virtual cell models trained on per-gene mean squared error fail to preserve the causal structure of perturbation responses, and they fail in essentially the same way regardless of architecture. The inflation magnitudes for GEARS (a graph neural network with Gene Ontology priors), CPA (a conditional variational autoencoder), and Geneformer (a foundation transformer pretrained on 30 million single-cell profiles) agreed to within 0.3 percentage points (63.2%, 63.4%, 63.5% respectively), which is much closer than would be expected if the failure mode were architecture-driven. The most parsimonious explanation is that the failure is objective-driven: optimizing per-gene reconstruction error provides no incentive to match the empirical sparsity of biological perturbation responses, and gradient descent in this regime converges to dense, high-magnitude predictions across all three architectural families. This reframes the long-standing observation that linear baselines match or exceed deep models on per-gene prediction (1) as a downstream consequence of a shared training-objective pathology rather than as an artifact of any specific architectural choice.

Second, the success of ElasticNet across both cell lines and across all seven gene-set scales has a methodological explanation. The L1 penalty in ElasticNet structurally biases predictions toward sparsity (21), and the empirical sparsity of real perturbation responses (38.6% of K562 effects near zero, 56.8% of variance in the leading singular vector) closely matches what an L1-penalized regressor would naturally produce. The deep models, lacking any analogous inductive bias toward sparsity in their training objective, have no incentive to match this empirical structure. The right inductive bias appears to outperform scale: a method with on the order of 200 × 200 learned coefficients exceeds the causal-recovery performance of a transformer with 104 million parameters trained on 30 million cells. This finding is consistent with recent work suggesting that contextual diversity in training data, rather than parameter count, is the limiting factor for cross-context generalization in this domain (22), and complements the calibrated-metric arguments advanced by Miller et al. (16) by extending the comparison from prediction to causal recovery.

Third, the inspre signal probe — recovering 12,280 edges from real data and from ElasticNet predictions, 6,234 from GEARS, and zero from both CPA and Geneformer — exposes a signal-level discrepancy that GIES partitioning conceals. The zero-edge result for CPA and Geneformer at the test-error-optimal regularization parameter does not mean these models produce no signal whatsoever, but that their average causal effect matrices fall below the threshold at which sparse inverse regression can identify any directed edge with the regularization strength selected by held-out validation. Whether the signal lies just below this threshold or is absent more broadly is best resolved by tracing the recovered edge count across the full lambda path; the present result establishes only that under the standard validation procedure, two of three deep models contribute no detectable causal signal beyond noise. A productive follow-up would compute inspre across the full lambda path for each condition and report the smallest lambda at which any signal is detected, providing a continuous measure of the gap between models.

Fourth, the calibration result reframes the expected story. The strong substitutability hypothesis predicted that correcting the magnitude distribution of model predictions would close the causal-recovery gap to ElasticNet. The data falsify this hypothesis: calibration improved CPA and GEARS by 1.6 and 0.8 F1 points respectively (closing roughly 40% of the gap), but degraded Geneformer by 0.9 F1 points, and no calibrated model exceeded the ElasticNet ceiling. The asymmetry is informative. CPA and GEARS predictions appear to be miscalibrated in magnitude but otherwise rank-correct; flat quantile matching corrects the magnitude error and recovers the implicit rank-correct signal. Geneformer's pretrained representations, by contrast, may already encode magnitude calibration that flat quantile matching erases — the model's vanilla predictions are the best of the three deep models on F1, and uniform calibration appears to remove signal rather than add it. This suggests that a productive improvement to the calibration framework would replace the global quantile mapping with a per-model learned shrinkage, in which the calibration strength is tuned to each model's pretrained calibration profile.

The fact that even calibrated CPA and GEARS fall short of ElasticNet implies that magnitude inflation is one but not the only failure mode underlying deep-model causal-recovery breakdown. The remaining 60% of the gap must reside in properties of the joint distribution that magnitude calibration cannot reach: gene-gene covariance structure of perturbation deviations, edge orientation, and the sparsity pattern of which genes are predicted to be affected for each perturbation. The 27 to 30% reversed-edge fraction observed in multi-hop analysis is consistent with this interpretation — many of the false positives produced by deep models are the right gene pairs in the wrong direction, a failure mode that distributional calibration cannot correct because it is a property of the conditional independence structure.

Several extensions follow naturally from these conclusions. First, the calibration framework can be extended with two additional modes already specified in the project design: covariance preservation, which would project predicted shifts toward the principal subspace of real control covariance to attenuate off-manifold predictions; and sparsity injection, which would soft-threshold small predicted effects to enforce the empirical near-zero fraction observed in real data. Combining all three modes is predicted to address a larger fraction of the causal-recovery gap than any single mode alone; the prediction is testable by repeating the present GIES evaluation on predictions calibrated under each combination of modes. Second, the joint-distribution failure can be probed directly by training virtual cell models with a covariance-preserving loss term — a Frobenius-norm penalty on the discrepancy between predicted and real perturbation-response covariance — and measuring whether the resulting models close the residual gap that magnitude calibration cannot. Third, the directional failure mode can be probed using d-separation tests on multi-perturbation predictions: for cases of a feed-forward chain G → X → H, models that have learned causal direction should predict that knocking down G while rescuing X to control levels blocks the downstream effect on H, while models that have only learned correlation should not. Both CPA and GEARS support multi-perturbation natively and are amenable to this test.

Three limitations of the present study are noted. The Track A ground truth is itself derived from GIES applied to real data, which makes the substitutability framing internally consistent but does not establish biological correctness. A Track B evaluation against the literature-curated TRRUST v2 database (23) was implemented but is not reported as a headline number in this manuscript; full Track B results constitute an independent test that is the natural next reporting step. Geneformer V2 was pretrained on a corpus that includes the Replogle 2022 dataset, so its predictions on this dataset are partly memorization rather than zero-shot generalization, and a temporally held-out cell line published after the model's pretraining cutoff is required to fully disentangle these two contributions. Finally, the headline numbers reported here are from a single random seed (42); a multi-seed runner has been implemented but not yet executed at full scale, so reported confidence intervals are bootstrap intervals on edge-set resampling rather than across-seed intervals.

---

## MATERIALS AND METHODS

### Data

The Replogle et al. 2022 K562 and RPE1 Perturb-seq datasets were obtained from Gene Expression Omnibus accession GSE221321 (3). The K562 dataset contained 310,385 single-cell transcriptomes spanning 1,861 perturbation-eligible CRISPR interference targets and 10,691 non-targeting control cells across 8,563 measured genes. The RPE1 dataset contained 247,914 cells and 2,069 perturbation-eligible targets. For computational tractability, primary evaluations were performed on a 200-gene subset selected from the K562 perturbation-eligible set with random seed 42, using a held-out subset file `data/k562_subset_n200_slim.h5ad`. Secondary scale-sweep evaluations were performed at gene-set sizes 50, 75, 100, 125, 150, 175, and 200 from the same K562 dataset.

### Virtual cell models

Three virtual cell models were trained or run on the Replogle K562 data using their canonical configurations. GEARS (5) was trained for 15 epochs with the standard graph-neural-network architecture and Gene Ontology gene-coexpression knowledge graph, on Apple M4 CPU, achieving validation MSE 3.69 over approximately 70 minutes. CPA (6), the compositional perturbation autoencoder, was trained for 50 epochs with default conditional VAE hyperparameters on Apple M4 CPU, achieving validation r²(mean) = 0.71 in approximately 6 minutes. Geneformer V2 (104 million parameter foundation transformer, pretrained on 30 million single-cell transcriptomes (7)) was used in its in-silico perturbation mode, with target gene embeddings shifted to the average non-target embedding and decoded predictions extracted through the standard ISP procedure, completing in approximately 9 minutes on Apple M4 CPU. All three models output per-perturbation mean expression vectors stored in standard AnnData (`.h5ad`) format.

### ElasticNet baseline

A sparse linear regression baseline was trained for each of the 200 evaluation genes. For each target gene, an ElasticNet regression model (sklearn `ElasticNet` with α = 0.1, l1_ratio = 0.5, max_iter = 1000, random_state = 0; (21)) was fit on held-out non-targeting control cells, predicting that gene's expression as a function of the other 199 genes' expression. At prediction time, the perturbed gene's value was set to 10% of its control mean (modeling approximately 90% CRISPRi knockdown), and the remaining 199 ElasticNet models were applied iteratively to predict the perturbed expression of every gene. No real perturbation expression values were used during training or prediction, eliminating any train/test leakage.

### Overlay sampling

Independent negative binomial sampling from predicted perturbation means was found to destroy the gene-gene covariance structure required for downstream causal discovery (validation: GIES applied to NB-resampled real means recovered 10 edges versus 538 from real cells). To preserve covariance, an overlay sampling procedure was developed and applied uniformly to all model predictions and to the overlay-resampled-real upper bound. For each perturbation, 100 cells were drawn with replacement from the subsampled control cells (n=200 controls, matching CausalBench protocol), each cell's gene expression vector was multiplied element-wise by the predicted-fold-change vector ((predicted_mean + 0.01) / (control_mean + 0.01), clipped to [0.01, 10]), and Poisson noise was applied to the resulting expression values. This procedure preserves the empirical control covariance structure while incorporating the model-predicted perturbation effects. Sensitivity analysis sweeping the clip range, pseudocount, and noise model confirmed that model rankings under this procedure are stable across parameter choices; only absolute metric values shift.

### Causal discovery algorithms

Two complementary causal discovery algorithms were used. GIES (Greedy Interventional Equivalence Search, Hauser & Bühlmann 2012) was applied via the Python `gies` package (17, 18), partitioning the 200-gene set into ten partitions of 20 genes each (the standard CausalBench protocol), running GIES independently on each partition, and concatenating recovered edges. Within each partition, GIES performed greedy score-based search over the space of completed partially directed acyclic graphs (CPDAGs) using a Bayesian Information Criterion objective and the per-cell intervention labels. Partitions were executed in parallel using `multiprocessing` with `OMP_NUM_THREADS=1` to prevent BLAS thread deadlock. Inspre (Inverse Sparse Regression, Brown et al. 2025) was applied via the official R package called as a subprocess (4). For each condition, an average causal effect (ACE) matrix was constructed in which entry (i, j) was the difference between the per-perturbation mean of gene j when gene i was knocked down and the control mean of gene j. Inspre was then applied to this matrix with regularization parameter selected by held-out test error and a target edge count matching the GIES-on-real edge count (1,220).

### Effect-size calibration

Post-hoc effect-size calibration was implemented using the rank-preserving quantile-matching algorithm of Bolstad et al. (19) and Reinhard et al. (20), without modification to the underlying statistical procedure. For a fitted calibrator, the empirical cumulative distribution function of |log2FC| values from real perturbations was constructed using 1,000 quantile knots evaluated by `numpy.quantile`. For each model prediction, the |log2FC| of every (perturbation, gene) entry was computed as `log2((predicted + 0.01) / (control_mean + 0.01))`, ranks were computed using `np.argsort`, and the calibrated magnitude was recovered by `np.interp(rank/n, quantile_points, real_quantiles)`. Sign was preserved unchanged; the calibrated expression value was reconstructed as `(control_mean + 0.01) × 2^(sign × calibrated_magnitude) − 0.01`, clipped to non-negative values. The full implementation is in `causalcellbench/calibration/quantile_calibrator.py`.

### Evaluation metrics

The primary metrics reported are F1 score, precision, and recall, computed at each method's operating point against the real-data GIES edge set (the substitutability ground truth). Precision was computed as TP / (TP + FP) and recall as TP / (TP + FN), where TP, FP, and FN are the standard true-positive, false-positive, and false-negative counts for set-based comparison. F1 was computed as the harmonic mean. Secondary metrics included a top-k AUPRC sweep computed by varying the number of predicted edges considered, Precision@100 (the fraction of the top 100 predicted edges that are in the ground truth), Structural Hamming Distance (SHD = missing + extra + reversed edges), and mean Wasserstein distance per perturbation. The False Omission Rate was computed via Mann-Whitney U tests on each not-predicted gene pair against control distributions, with global Benjamini-Hochberg correction across all approximately N² tests. Multi-hop credit at k=2 and k=3 was implemented by checking whether each predicted edge had a path of the corresponding length in the ground-truth graph using `networkx.has_path`.

### Statistical analysis

Bootstrap confidence intervals were computed on every set-based metric by resampling the predicted edge set 200 times with replacement and recomputing the metric on each resample, with 95% intervals reported as the 2.5th and 97.5th percentiles. Pairwise permutation tests for differences in F1 between conditions used 500 random shuffles of edges between the two conditions, with two-sided p-values computed from the empirical distribution of permuted F1 differences. A multi-seed evaluation runner was implemented in `scripts/run_multi_seed.py` to repeat the full pipeline across five random seeds (42, 123, 456, 789, 1024), aggregating with t-distribution confidence intervals and paired permutation tests; multi-seed aggregation was not yet executed at the time of writing and is identified as the immediate next reporting step.

### Implementation and reproducibility

The full pipeline is implemented in Python 3.10 with `anndata`, `scanpy`, `gies`, `arboreto`, `scikit-learn`, and `networkx` as primary dependencies. The R `inspre` package (v1.0.1) is called via subprocess from Python. The complete codebase, including pinned `pyproject.toml`, deterministic seeds, and all configuration files, is released under MIT license at the project repository. Reproducibility commands for the calibration experiments are provided in `results/calibration_comparison.md`.

---

## REFERENCES

1. Ahlmann-Eltze, C. & Huber, W. Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. *Nature Methods* (2025). doi:10.1038/s41592-025-02772-6

2. Chevalley, M., Roohani, Y., Mehrjou, A., Leskovec, J. & Schwab, P. CausalBench: A large-scale benchmark for network inference from single-cell perturbation data. *Communications Biology* (2025). doi:10.1038/s42003-025-07764-y

3. Replogle, J. M. et al. Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell* 185, 2559-2575 (2022). doi:10.1016/j.cell.2022.05.013

4. Brown, B. C., Spence, J. P. & Pritchard, J. K. Large-scale causal discovery using interventional data sheds light on gene network structure in K562 cells. *Nature Communications* 16, 9628 (2025). doi:10.1038/s41467-025-64353-7

5. Roohani, Y., Huang, K. & Leskovec, J. Predicting transcriptional outcomes of novel multigene perturbations with GEARS. *Nature Biotechnology* 42, 927-935 (2024). doi:10.1038/s41587-023-01905-6

6. Lotfollahi, M. et al. Predicting cellular responses to complex perturbations in high-throughput screens. *Molecular Systems Biology* 19, e11517 (2023). doi:10.15252/msb.202211517

7. Theodoris, C. V. et al. Transfer learning enables predictions in network biology. *Nature* 618, 616-624 (2023). doi:10.1038/s41586-023-06139-9

8. Karlebach, G. & Shamir, R. Modelling and analysis of gene regulatory networks. *Nature Reviews Molecular Cell Biology* 9, 770-780 (2008). doi:10.1038/nrm2503

9. Dixit, A. et al. Perturb-Seq: dissecting molecular circuits with scalable single-cell RNA profiling of pooled genetic screens. *Cell* 167, 1853-1866 (2016). doi:10.1016/j.cell.2016.11.038

10. Bunne, C. et al. How to build the virtual cell with artificial intelligence: priorities and opportunities. *Cell* 187, 7045-7063 (2024). doi:10.1016/j.cell.2024.11.015

11. Cui, H. et al. scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods* 21, 1470-1480 (2024). doi:10.1038/s41592-024-02201-0

12. Wu, Y. et al. PerturBench: Benchmarking machine learning models for cellular perturbation analysis. *arXiv* 2408.10609 (2024).

13. Roohani, Y. et al. Virtual Cell Challenge: Toward a Turing test for the virtual cell. *Cell* 188, 3370-3382 (2025). doi:10.1016/j.cell.2025.06.008

14. Anonymous. Virtual Cells as Causal World Models: A Perspective on Evaluation. *NeurIPS 2025 AI4D3 Workshop* (2025). https://ai4d3.github.io/2025/papers/25_Virtual_Cells_as_Causal_Wor.pdf

15. Mejia, G. M. et al. Diversity by Design: Addressing Mode Collapse Improves scRNA-seq Perturbation Modeling on Well-Calibrated Metrics. *arXiv* 2506.22641 (2025).

16. Miller, H. E. et al. Deep Learning-Based Genetic Perturbation Models Do Outperform Uninformative Baselines on Well-Calibrated Metrics. *bioRxiv* 2025.10.20.683304 (2025).

17. Hauser, A. & Bühlmann, P. Characterization and greedy learning of interventional Markov equivalence classes of directed acyclic graphs. *Journal of Machine Learning Research* 13, 2409-2464 (2012).

18. Chickering, D. M. Optimal structure identification with greedy search. *Journal of Machine Learning Research* 3, 507-554 (2002).

19. Bolstad, B. M., Irizarry, R. A., Astrand, M. & Speed, T. P. A comparison of normalization methods for high density oligonucleotide array data based on variance and bias. *Bioinformatics* 19, 185-193 (2003). doi:10.1093/bioinformatics/19.2.185

20. Reinhard, E., Adhikhmin, M., Gooch, B. & Shirley, P. Color transfer between images. *IEEE Computer Graphics and Applications* 21, 34-41 (2001). doi:10.1109/38.946629

21. Zou, H. & Hastie, T. Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B* 67, 301-320 (2005). doi:10.1111/j.1467-9868.2005.00503.x

22. Anonymous. Virtual Cells Need Context, Not Just Scale. *bioRxiv* 2026.02.04.703804 (2026).

23. Han, H. et al. TRRUST v2: an expanded reference database of human and mouse transcriptional regulatory interactions. *Nucleic Acids Research* 46, D380-D386 (2018). doi:10.1093/nar/gkx1013

---

## FIGURE LEGENDS

**Figure 1. Schematic of the CausalCellBench benchmark architecture.** Real Perturb-seq data and predictions from each virtual cell model (GEARS, CPA, Geneformer) and the ElasticNet baseline pass through identical downstream pipelines. Mean predictions are converted to single-cell-level matrices via overlay sampling (bootstrap from controls + multiplicative fold-change + Poisson noise). The resulting interventional expression matrices are passed to two causal discovery algorithms in parallel: GIES (partitioned greedy score-based search) and inspre (sparse inverse regression on the average causal effect matrix). Recovered edge sets are compared to the GIES-on-real ground truth using F1, precision, recall, and SHD. The single variable that changes between conditions is the data source.

**Figure 2. Causal recovery performance at N=200 K562.** F1 score against real-data GIES ground truth (1,220 edges) for each evaluated condition: real data ceiling, ElasticNet baseline, overlay-resampled real upper bound, and three deep models (GEARS, CPA, Geneformer). Random-bootstrap floor (F1 ≈ 0.04, dashed line). Bars indicate F1 at each method's operating point. Bootstrap 95% confidence intervals computed by resampling the predicted edge set 200 times. n=1 random seed (42); multi-seed evaluation pending.

**Figure 3. Effect-size distribution discrepancy between real and predicted perturbation responses.** Empirical distributions of |log2FC| at the 200-gene K562 evaluation, showing real perturbation cells (gray; 2.7% of effects exceed |log2FC| = 0.5) and three deep model predictions (GEARS, CPA, Geneformer; 63.2%, 63.4%, 63.5% respectively). Inset: cumulative distribution functions for the same data, illustrating the rank-preserving but magnitude-shifted relationship between real and predicted distributions. Real K562 SVD: leading singular vector explains 56.8% of total variance.

**Figure 4. inspre signal probe across data sources.** Number of directed edges recovered by sparse inverse regression on the average causal effect matrix derived from each data source, at the test-error-optimal regularization parameter. Real Perturb-seq and ElasticNet predictions yield identical edge counts (12,280); GEARS predictions yield 6,234; CPA and Geneformer predictions yield zero. The dashed line marks the corresponding GIES edge count for reference (~1,220), illustrating that GIES partitioning equalizes edge counts across conditions in a way that obscures the underlying signal differences exposed by inspre.

**Figure 5. Scale-sweep behavior of method rankings across N = 50 to 200.** Precision against the real-data GIES ground truth at each gene-set size for ElasticNet, GEARS, CPA, and Geneformer. The deep model achieving the highest precision rotates across scales (Geneformer at N=150, GEARS at N=175, ElasticNet at N=200). ElasticNet is in the top two methods at every scale tested.

**Figure 6. Multi-hop credit reveals reversed-direction failure mode.** Precision at hop-1 (standard evaluation), hop-2, and hop-3 for each method on the N=200 K562 evaluation. A predicted edge is credited at hop-k if a directed path of length at most k exists in the ground-truth graph. Increases from hop-1 to hop-2 of approximately 9 to 12 percentage points across all methods indicate that a non-trivial fraction of false positives are correct gene pairs predicted with reversed direction (27 to 30% of FPs by SHD breakdown).

**Figure 7. Effect-size calibration corrects magnitude inflation but only partially recovers causal F1.** Top: fraction of |log2FC| > 0.5 in vanilla and quantile-matched calibrated predictions for each deep model, with the real K562 200-gene reference (11.4%) shown as a horizontal line. Calibration drives every model to within 0.2 percentage points of the real reference, by construction. Bottom: change in causal F1 from vanilla to calibrated predictions on the identical GIES pipeline (seed 42, partition size 20). CPA: +0.0164; GEARS: +0.0078; Geneformer: −0.0092. ElasticNet (no calibration applied) F1 = 0.425 marked as a horizontal reference line.

---
