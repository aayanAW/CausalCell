# CausalCellBench

**Evaluating whether virtual cell models preserve causal structure for gene regulatory network discovery.**

CausalCellBench is a benchmark that tests whether AI-based perturbation prediction models (GEARS, CPA, Geneformer) can substitute for real CRISPR screens in causal gene regulatory network inference. Unlike existing benchmarks that evaluate prediction accuracy (MSE, correlation), CausalCellBench evaluates whether model outputs preserve the causal dependencies needed for interventional structure learning.

## Key Findings

| Condition | AUPRC (50g) | AUPRC (200g) |
|---|---|---|
| Real Data (upper bound) | 1.000 | 1.000 |
| ElasticNet (regression) | 0.396 | 0.454 |
| GEARS | 0.385 | 0.401 |
| Geneformer | 0.319 | 0.422 |
| CPA | 0.322 | 0.405 |

- **ElasticNet regression matches or outperforms all deep learning models** due to 23-fold effect-size inflation in DL predictions
- **inspre analysis** reveals CPA and Geneformer contain zero detectable causal signal
- **Multi-hop evaluation** shows 27% of false positives are correctly identified gene pairs with reversed directionality
- **Model rankings are unstable across gene scales**, cautioning against single-scale conclusions
- **Clinical translation** identifies 43 selectively essential hub genes in K562 targetable by FDA-approved drugs (tofacitinib, bortezomib)
- **Cross-context validation** on RPE1 confirms the ElasticNet anomaly is universal

## Installation

```bash
git clone https://github.com/aayanalwani/causalcellbench.git
cd causalcellbench
pip install -r requirements.txt
```

For inspre (R-based causal discovery):
```bash
# Install R >= 4.4
Rscript -e 'install.packages("remotes"); remotes::install_github("brielin/inspre")'
```

## Data

Download the Replogle et al. (2022) Perturb-seq datasets:
```bash
bash scripts/download_data.sh
```

This downloads K562 and RPE1 CRISPRi data from [Figshare](https://plus.figshare.com/articles/dataset/20029387).

## Usage

### Run the full evaluation pipeline
```bash
python scripts/run_eval_local.py \
  --data-path data/k562_real.h5ad \
  --n-genes 200 \
  --gene-list data/eval_genes_n200.txt \
  --model-outputs-dir data/model_outputs_real_n200 \
  --gies-cache-dir results/gies_cache
```

### Run individual analysis phases
```bash
# Phase 1: ElasticNet anomaly analysis
python scripts/phase1_elasticnet_anomaly.py

# Phase 2: Scale sweep (N=50-200)
python scripts/phase2_scale_sweep.py

# Phase 4: Multi-hop causal evaluation
python scripts/phase4_multihop_metrics.py

# Phase 5: inspre causal discovery
python scripts/phase5_inspre_subprocess.py

# Phase 7: Clinical translation (DepMap + DGIdb)
python scripts/phase7_clinical.py

# Phase 9: RPE1 cross-context
python scripts/phase9_rpe1.py
```

### Generate figures
```bash
python scripts/generate_figures_n50.py --results results/eval_results_n50.json
python scripts/generate_figures_n50.py --results results/eval_results_n200.json --outdir figures_n200
```

## Project Structure

```
causalcellbench/
├── causal/          # Causal discovery engines (GIES, GRNBoost2, DCDI)
├── models/          # Model wrappers (GEARS, CPA, Geneformer, scGPT)
├── eval/            # Evaluation metrics (AUPRC, SHD, Wasserstein, FOR)
├── data/            # Data loaders (Replogle, ChIP-seq, TRRUST)
└── calibration/     # Overlay sampling, quantile calibration

scripts/              # Executable analysis scripts
results/              # Cached results (JSON, CSV)
figures/              # Publication figures (PDF, PNG)
```

## Methodological Contributions

1. **Overlay Sampling**: Standard NB sampling destroys gene-gene correlations needed for causal discovery. Our overlay method bootstraps control cells and applies fold-change shifts, preserving biological covariance structure.

2. **inspre Integration**: We integrate the inspre algorithm (Brown et al., Nat Comms 2025) as a scalable alternative to GIES, enabling genome-scale causal discovery in seconds rather than days.

3. **Selective Essentiality**: Clinical translation uses differential DepMap scores (K562 vs pan-cell-line mean) to identify CML-specific vulnerabilities, avoiding universal housekeeping genes.

## Citation

```bibtex
@article{alwani2026causalcellbench,
  title={The Causal Blind Spot: Why Virtual Cell Models Fail at Gene Regulatory Network Discovery},
  author={Alwani, Aayan},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## References

1. Replogle et al. (2022). Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell*.
2. Chevalley et al. (2025). CausalBench: A large-scale benchmark for network inference. *Communications Biology*.
3. Roohani et al. (2024). GEARS: Predicting transcriptional outcomes of novel multigene perturbations. *Nature Biotechnology*.
4. Lotfollahi et al. (2023). CPA: Predicting cellular responses to complex perturbations. *Molecular Systems Biology*.
5. Theodoris et al. (2023). Transfer learning enables predictions in network biology. *Nature*.
6. Brown et al. (2025). Large-scale causal discovery using interventional data. *Nature Communications*.
