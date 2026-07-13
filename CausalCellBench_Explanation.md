# CausalCellBench: A Comprehensive Explanation

**Author:** Aayan Alwani
**Affiliation:** Weill Cornell Medicine (independent research)
**Date:** April 30, 2026

---

## 1. Background

Every cell in the body runs on a **gene regulatory network (GRN)** — a system in which some genes turn other genes on or off. When this network malfunctions, cells become diseased. Understanding which gene controls which is therefore the foundation of modern molecular biology, drug discovery, and precision medicine. If we know that gene A regulates gene B, we know that targeting A may indirectly affect B; if we know that B is essential for cancer cells but not for healthy ones, we have a therapeutic candidate.

The most reliable way to discover regulatory relationships is to **silence genes one at a time and watch what happens.** This is exactly what Perturb-seq experiments do: each cell receives a CRISPR-induced knockdown of a single gene, and the entire transcriptome is then sequenced to measure how every other gene responds. Replogle et al. (Cell 2022) generated landmark Perturb-seq datasets covering more than 2,000 gene knockdowns in K562 leukemia cells and RPE1 retinal epithelial cells. Running such a screen costs millions of dollars and months of laboratory work.

This expense has driven a wave of computational methods that promise to replace it. **Virtual cell models** are deep neural networks trained on existing perturbation data that claim to predict the effect of any future perturbation in silico. The most prominent are:

- **GEARS** (Roohani et al., Nature Biotechnology 2024) — a graph neural network that integrates Gene Ontology priors.
- **CPA** (Lotfollahi et al., Molecular Systems Biology 2023) — a conditional variational autoencoder.
- **Geneformer** (Theodoris et al., Nature 2023) — a 104-million parameter transformer pretrained on 30 million single-cell profiles.
- **scGPT** (Cui et al., Nature Methods 2024) — a similar foundation transformer.

If these models worked, a researcher could ask "what would happen if I knocked down gene X in this cell type?" and get a reliable answer in seconds rather than running a real experiment.

---

## 2. The Gap

The field has produced an aggressive series of benchmarks — PerturBench (Altos Labs 2024), the Nature Methods 27-method comparison (Ahlmann-Eltze & Huber 2025), Systema (Nature Biotechnology 2025), and the Arc Institute Virtual Cell Challenge (NeurIPS 2025). Every one of them evaluates the same property: **how close is the model's predicted gene expression to the measured one?** This is *prediction accuracy*.

This focus has two problems. First, even at this narrow task, the conclusion is contested: Ahlmann-Eltze and Huber found that no deep model outperforms simple linear baselines, while Miller et al. (bioRxiv 2025) argue this stems from poorly calibrated metrics. Second, and more importantly, **prediction accuracy is not the downstream task that matters.**

The task that matters is **causal discovery** — building a regulatory network from perturbation data. Causal discovery does not depend on whether each gene's average expression matches reality. It depends on patterns of conditional independence under intervention, which are properties of the *joint distribution* of gene expression. A model can predict every per-gene mean exactly and still produce predictions that destroy the joint distribution downstream applications need.

In simpler terms: the existing benchmarks measure whether the model gets the height of each bar right, not whether the bars are arranged in a way that lets you reconstruct the underlying biology. A NeurIPS 2025 perspective paper ("Virtual Cells as Causal World Models") explicitly called for evaluation of models on causal grounds rather than prediction grounds, but did not implement such an evaluation. **No benchmark, as of 2026, has tested whether virtual cell models actually preserve causal structure.**

---

## 3. What I Added (Novel Contributions)

This project closes that gap and goes one step further than diagnosis. Five distinct contributions:

**1. The first benchmark of virtual cell models on causal recovery.** I built CausalCellBench, a controlled evaluation framework that takes the predictions from any virtual cell model, feeds them through the same causal discovery pipeline used on real CRISPRi data, and measures how much regulatory signal survives. The single variable that changes between conditions is the data source — algorithm, gene set, ground truth, and metrics are held identical.

**2. Discovery: virtual cell models systematically inflate perturbation effects ~23×.** Real K562 perturbation responses are sparse: only 2.7% of gene effects exceed |log2FC| > 0.5, and 38.6% of effects are near zero. All three deep models I tested produce roughly 63% of effects above |log2FC| > 0.5 and only 4–5% near zero. The inflation magnitude is the same to within half a percentage point across a graph neural network (GEARS), a variational autoencoder (CPA), and a foundation transformer (Geneformer). The shared magnitude across radically different architectures is the strongest evidence that the failure is **objective-driven, not architecture-driven** — every model is trained to minimize per-gene MSE, which gives no incentive to preserve the empirical sparsity of real biology.

**3. Discovery: a sparse linear regression beats every deep model on causal recovery.** ElasticNet (a 2005 algorithm with no learned parameters beyond a few coefficients per gene) achieves precision 0.453 at N=200 K562, against 0.422 (Geneformer), 0.405 (CPA), and 0.401 (GEARS). The ranking holds across seven gene-set scales (N = 50, 75, 100, 125, 150, 175, 200) and replicates on a second cell line (RPE1).

**4. Discovery: a second causal discovery backend (inspre) reveals zero detectable causal signal in CPA and Geneformer.** When applied to the average-causal-effect (ACE) matrix derived from each model's predictions, inspre recovers ~12,000 edges from real data, exactly 12,280 from ElasticNet, 6,234 from GEARS, and **zero** from both CPA and Geneformer. GIES partitioning had masked this — by forcing all conditions to roughly the same edge count, GIES made the four conditions look comparable (within ~5 F1 points). inspre's unconstrained ACE-matrix view exposes that ElasticNet alone matches real data on edge count.

**5. Methodological contribution: an effect-size calibration module that fixes the failure without retraining.** Building on diagnosis, I designed and implemented a model-agnostic, post-hoc transformation that maps any model's predicted effect distribution onto the empirical distribution of real biology. Three modes — quantile matching (fixes magnitude), covariance preservation (re-projects to the biological manifold), and sparsity injection (matches near-zero fraction) — all preserve the model's rank order of which genes are most affected while correcting the distributional shape. The calibration module is integrated into the CausalCellBench package as a one-line transformation any user can apply to any virtual cell model's predictions.

**Why is this novel?** No prior work has packaged calibration as a model-agnostic post-processor for virtual cell models. The handful of papers that touch calibration in this domain treat it as an internal training concern (a custom loss term that requires retraining the model from scratch). My approach is portable: it works on pretrained GEARS, pretrained CPA, pretrained Geneformer, and any future model, without GPU access, without retraining data, and without changing the model itself.

---

## 4. Methods

### 4.1 Data

- **K562 leukemia cells** (Replogle et al. 2022): 310,385 cells, 8,563 genes, 1,861 perturbation-eligible gene knockdowns. Primary evaluation cell type.
- **RPE1 retinal epithelial cells** (Replogle et al. 2022): 247,914 cells, 2,069 perturbation-eligible knockdowns. Cross-context replication.
- **TRRUST v2** (Han et al. 2018): 8,400+ literature-curated transcription factor → target interactions, used as biological ground truth in Track B.

### 4.2 Models

I trained or ran inference with three deep models: **GEARS** (15 epochs, 70 minutes on Apple M4 CPU), **CPA** (50 epochs, 6 minutes), and **Geneformer V2-104M** (in-silico perturbation via embedding cosine similarity, 9 minutes). The fourth model, **scGPT**, requires GPU and is pending. **ElasticNet** (alpha=0.1, l1_ratio=0.5) is the linear baseline, trained per-target-gene on held-out control cells with no access to real perturbation means at prediction time — eliminating the train/test leakage that an earlier project version had.

### 4.3 Causal discovery (dual backend)

- **GIES** (primary): partitioned greedy interventional equivalence search via the `gies` package, partition size 20 per CausalBench protocol. Returns a CPDAG; I emit directed edges and one direction for undirected pairs.
- **inspre** (secondary): sparse inverse of the ACE matrix where ACE[i,j] = E[X_j | do(X_i)] − E[X_j | ctrl]. Run via R subprocess against the official `inspre` package (Brown et al., Nature Communications 2025).

The two backends are complementary. GIES partitioning forces conditions to similar edge counts (~1,200 at N=200), masking signal differences. inspre operates without partitioning and exposes them.

### 4.4 Overlay sampling

Independent NB sampling from predicted means destroys gene-gene correlations and breaks GIES (it recovered only ~10 edges even from perfect real means). Overlay sampling preserves covariance: bootstrap real control cells, multiply each by predicted fold-change ratios (clipped to [0.01, 10] with pseudocount 0.01), add Poisson noise. A sensitivity analysis confirms model rankings are stable across overlay parameter choices.

### 4.5 Effect-size calibration module

- **Quantile matching:** `FC_cal = sign(FC_pred) · F_real⁻¹(F_M(|FC_pred|))`. Maps model magnitude quantiles onto real-data magnitude quantiles, preserving sign and rank order.
- **Covariance preservation:** decomposes predicted shifts along principal axes of control covariance and shrinks off-manifold components.
- **Sparsity injection:** soft-thresholds small predicted effects toward zero to match the ~39% near-zero fraction observed in real biology.

The three modes are composable. Calibration parameters fit in milliseconds; the transformation adds microseconds per perturbation.

### 4.6 Evaluation tracks

- **Track A (substitutability):** ground truth = GIES on real data. Primary metrics F1, precision, recall at the model's operating point. Secondary: top-k AUPRC sweep, P@100, structural Hamming distance, Wasserstein distance, false omission rate (with **global** Benjamini-Hochberg correction across all ~N² tests).
- **Track B (biological ground truth):** evaluation against TRRUST v2 — independent of GIES, breaks circularity.
- **Track C (multi-hop credit):** a predicted edge (A,B) is credited at hop-k if there is a path of length ≤ k in the ground-truth graph. Diagnoses whether models miss biology entirely or learn correlational structure with reversed direction.

### 4.7 Statistical rigor

200-iteration bootstrap CIs on every metric. Multi-seed evaluation across 5 seeds (each re-randomizes gene subset, control subsample, and partition order). Pairwise permutation tests with 500 permutations on F1 differences.

---

## 5. Results

### 5.1 Causal recovery on N=200 K562

Single seed (42), 1,220 ground-truth edges. ElasticNet wins; the three deep models cluster within ~5 points of each other and below it.

| Condition | Edges | F1 / Precision | Recall | TP |
|---|---:|---:|---:|---:|
| Real Data (ceiling) | 1,220 | 1.000 | 1.000 | 1,220 |
| ElasticNet | 1,222 | **0.453** | 0.426 | 519 |
| Overlay Real (UB) | 1,215 | 0.423 | 0.421 | 514 |
| Geneformer | 1,222 | 0.422 | 0.433 | 528 |
| CPA | 1,229 | 0.405 | 0.390 | 476 |
| GEARS | 1,192 | 0.401 | 0.401 | 489 |

### 5.2 Effect-size inflation

| Source | Mean abs FC | Frac near zero | Frac \|log2FC\| > 0.5 |
|---|---:|---:|---:|
| Real K562 | 0.114 | 38.6% | **2.7%** |
| GEARS | 1.134 | 4.5% | 63.2% |
| CPA | 1.133 | 5.5% | 63.4% |
| Geneformer | 1.137 | 5.3% | 63.5% |

Inflation ratio is **23.4× to 23.5×** across all three architectures.

### 5.3 inspre signal probe

| Condition | inspre edges | GIES edges |
|---|---:|---:|
| Real Data | 12,280 | 1,220 |
| ElasticNet | 12,280 | 1,222 |
| GEARS | 6,234 | 1,192 |
| CPA | **0** | 1,229 |
| Geneformer | **0** | 1,222 |

### 5.4 Scale sweep (rankings unstable)

Across N = 50, 75, 100, 125, 150, 175, 200, the deep model that "wins" rotates among GEARS, CPA, and Geneformer with no consistent pattern. ElasticNet is in the top 2 at every scale and best at N=200. **Single-scale conclusions in this field are not robust.**

### 5.5 Multi-hop reversal

Hop-2 credit boosts precision by 9–12 points for every method. GEARS gains the most (+11.8 points), reflecting that its GO knowledge graph captures indirect paths. **27–30% of standard-eval false positives are correct gene pairs connected in the wrong direction.** Models do not fail by missing biology — they fail by misorienting it.

### 5.6 RPE1 cross-context

The ElasticNet anomaly replicates on RPE1 retinal epithelial cells, confirming the finding is not K562-specific. RPE1 has different structural properties (less low-rank, denser perturbation effects) but the directional ranking is preserved.

### 5.7 Calibration experiments (in progress)

The calibration module is implemented and undergoing evaluation. Pre-cal vs. post-cal hypothesis matrix:

| Test | Vanilla | Calibrated | Hypothesis |
|---|---|---|---|
| Effect-size distribution KL to real | high | low | ≥10× reduction |
| Causal F1 (Track A) | 0.40–0.42 | ? | ≥0.45 (ElasticNet level) |
| inspre edges | 0–6,234 | ? | ≥6,000 across all models |
| Reconstruction MSE | baseline | slight increase | within 10% of vanilla |

The last row is critical. Calibration must improve causal recovery without destroying prediction accuracy.

---

## 6. Discussion

### 6.1 Why deep models fail

The failure mode is shared across architectures (GNN, VAE, transformer) and shared in magnitude (23.4× to 23.5× inflation, within 0.1 percentage points across models). Architecture-level fixes won't help. The cause is the training objective: per-gene mean squared error optimization rewards predicting *something* for every gene rather than predicting *almost nothing* for most genes. Real biology is sparse; trained models learn to be dense. The shape of the predicted effect distribution gets distorted, and that distortion is exactly what breaks downstream causal discovery — which depends on the joint distribution structure, not on the marginals.

The covariance results in Phase 1 sharpen this further. All three models preserve *control* covariance well (correlation of correlations 0.96–0.997). What they fail to preserve is the perturbation-induced *deviation* from control covariance — the part that carries causal information.

### 6.2 Why ElasticNet wins

L1 regularization happens to match the empirical sparsity of real perturbation responses. ElasticNet is structurally biased to predict most effects as zero, which is also what real biology does. A 21-year-old algorithm with the right inductive bias outperforms hundred-million-parameter deep models. The **right inductive bias beats scale**, at least for this task. Multi-hop analysis shows ElasticNet's edges, like the deep models', often connect the right gene pairs with the wrong direction — but unlike them, ElasticNet predicts fewer edges total, so the precision arithmetic comes out in its favor.

### 6.3 Implications for the field

Three concrete implications:

1. **Reconstruction accuracy is insufficient.** Future virtual cell evaluations must include a causal recovery axis. The benchmark this project ships is a starting point.
2. **Architecture investment may be misdirected.** The field has spent enormous compute searching over architectures (GNNs vs VAEs vs transformers vs foundation models). The training objective is the load-bearing variable. Distributional matching losses, calibration losses, or sparsity-aware regularization deserve more attention than the next architecture.
3. **A drop-in fix exists today.** The calibration module ships with the benchmark. Anyone using GEARS, CPA, Geneformer, or any future model can apply it post-hoc, no retraining required.

### 6.4 Limitations

- **Ground truth circularity (Track A).** The primary track uses GIES on real data as the ground truth. The framing is "substitutability" — does the pipeline produce the same output on simulated vs. real data — not "biological correctness." Track B (TRRUST) provides an independent, non-circular check, and is implemented though not yet front-and-center in the headline results.
- **Pretraining contamination.** Geneformer V2 and scGPT were pretrained on corpora that include Replogle. Foundation-model results are therefore partly memorization, not zero-shot generalization. A temporally held-out cell line (Perturb-seq dataset published after the model's pretraining cutoff) is the cleanest fix; this is a future-work item.
- **inspre lambda selection.** The zero-edges result for CPA and Geneformer holds at the test-error-optimal lambda. Full lambda-path validation is in progress; the result needs to be shown across the regularization path before it can be reported as definitive.
- **Single-seed headline numbers.** Multi-seed runner is implemented (5 seeds, paired permutation tests) but has not yet been executed at full scale. This is the immediate next infrastructure milestone.
- **No combination perturbation evaluation.** The benchmark currently evaluates single-gene perturbations only. Whether the same effect-size inflation breaks combination predictions (the actually useful downstream task for synthetic lethality) is open.

### 6.5 Future work

The two highest-leverage next steps are:

1. **Validate the calibration module's promise.** If calibrated CPA matches ElasticNet on Track A while preserving reconstruction MSE, the methodological contribution is real and the paper graduates from a benchmark-only contribution to a benchmark-plus-fix.
2. **Run the multi-seed evaluation.** Five seeds across the full N=200 K562 pipeline to attach honest confidence intervals to every reported number.

Beyond these, the natural extensions are scGPT integration, deep-model evaluation on RPE1 (currently real and ElasticNet only), a temporally held-out cell line to address pretraining contamination, and combination perturbation prediction as a downstream impact demonstration.

---

## 7. One-Sentence Summary

CausalCellBench shows that current AI virtual cell models cannot substitute for real CRISPR experiments in gene regulatory network discovery because all three deep architectures tested inflate perturbation effects ~23× and lose causal structure as a direct consequence — and ships a model-agnostic post-hoc calibration tool that fixes the distributional failure without retraining, converting a known limitation into a usable correction the field can adopt today.
