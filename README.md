# PerturbCausal

A benchmark for testing whether virtual cell model (VCM) predicted Perturb-seq
data can substitute for real Perturb-seq in downstream causal gene regulatory
network recovery.

## TL;DR

Across multi-seed evaluation on Replogle K562, no deep VCM
(GEARS, CPA, Geneformer V2, STATE) significantly exceeds a sparse linear
regression baseline (ElasticNet) on causal-recovery F1. The substitutability
failure decomposes into a magnitude marginal (~40%, post-hoc fixable via
quantile matching) and a joint-structure residual (~60%, requiring
training-time intervention).

This extends the linear-baselines-beat-DL finding of
[Ahlmann-Eltze & Huber 2025 *Nature Methods*](https://doi.org/10.1038/s41592-025-02772-6)
from per-gene reconstruction to causal recovery.

## Install

```bash
git clone https://github.com/<repo>/perturbcausal.git
cd perturbcausal
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

For STATE inference (Arc Institute 2025 foundation model):

```bash
pip install -e .[state]
# Apply transformers source patch documented in INSTALL.md
```

## Docker (recommended for reproducibility)

```bash
docker build -t perturbcausal .
docker run --rm -v $(pwd)/results:/app/results perturbcausal
```

## Quick start

```python
import anndata as ad
from causalcellbench.calibration import CalibrationPipeline

# Load a model's predicted perturbation means (any VCM)
predictions = ad.read_h5ad("path/to/model_predictions.h5ad")

# Fit the three-mode calibration pipeline on real-data control + perturbation
# statistics, then apply to the predictions:
pipeline = CalibrationPipeline(order=["sparsity", "quantile", "covariance"])
pipeline.fit(
    real_log_fc=real_log_fc_train,           # (n_train_perts, n_genes)
    real_pert_shifts=real_pert_shifts_train, # (n_train_perts, n_genes)
    pred_log_fc=predicted_log_fc_train,
)
calibrated = pipeline.transform(predicted_log_fc_test)
```

## Reproducing paper results

```bash
make repro    # runs the seed=42 K562 N=200 evaluation
make figures  # generates the manuscript figures from cached results
```

Full multi-seed reproduction (~7-15h on a 16-core machine):

```bash
python scripts/run_multi_seed.py \
    --n-genes 200 \
    --seeds 42 123 456 789 1024 2048 4096 7919 13337 31337 65537 100003 \
            142857 271828 314159 524287 1000003 1299709 1999993 2147483647 \
    --extra-args="--skip-random --partition-size 20"
```

## Repository structure

```
causalcellbench/    # Python package: data loaders, calibration, causal engines
scripts/            # CLI scripts: training, evaluation, integration, plotting
scripts/v4/         # V4-plan orchestrator + per-phase scripts
submissions/icml/   # ICML / AI4Science LaTeX manuscript
submissions/biorxiv # bioRxiv staging
results/            # JSON outputs from all phases
data/               # Local data caches (gitignored)
tests/              # pytest suite
```

## Citation

If you use PerturbCausal, please cite:

```bibtex
@article{perturbcausal2026,
  author    = {Anonymous},
  title     = {Virtual Cell Models Inflate Perturbation Effect Sizes and
               Undermine Causal Gene Regulatory Network Recovery},
  journal   = {bioRxiv},
  year      = {2026},
  note      = {Submitted to NeurIPS Datasets and Benchmarks Track 2026}
}
```

## License

MIT — see `LICENSE`.

## Acknowledgements

This work builds directly on the K562/RPE1 Perturb-seq atlas from Replogle
et al. 2022, the inspre algorithm of Brown et al. 2025, the CausalBench
framework of Chevalley et al. 2025, and the STATE foundation model released
by Arc Institute (Adduri et al. 2025).
