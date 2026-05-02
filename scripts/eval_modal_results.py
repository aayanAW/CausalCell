"""Download and evaluate real model predictions from Modal volume.

Downloads .h5ad prediction files from the Modal volume, validates they're
real (not stubs), and runs them through GIES causal discovery.

Usage:
    python3 scripts/eval_modal_results.py
    python3 scripts/eval_modal_results.py --n-genes 200 --n-ctrl 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causalcellbench.causal.causalbench_bridge import (
    CausalBenchInput,
    build_from_disk,
)
from causalcellbench.causal.gies_runner import run_gies
from causalcellbench.eval.metrics import (
    compute_auprc,
    compute_precision_at_k,
    compute_shd,
    compute_wasserstein,
    compute_false_omission_rate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = ["scgpt", "gears", "cpa", "geneformer"]
VOLUME_NAME = "ccbench-data"
LOCAL_DIR = Path("data/model_outputs")


def download_from_volume(model: str) -> Path | None:
    """Download model prediction .h5ad from Modal volume."""
    remote_path = f"results/{model}_predictions.h5ad"
    local_path = LOCAL_DIR / f"{model}_predictions.h5ad"

    os.makedirs(LOCAL_DIR, exist_ok=True)

    try:
        result = subprocess.run(
            ["python3", "-m", "modal", "volume", "get", VOLUME_NAME, remote_path, str(local_path)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            logger.warning("Failed to download %s: %s", model, result.stderr.strip())
            return None
        return local_path
    except subprocess.TimeoutExpired:
        logger.warning("Timeout downloading %s", model)
        return None


def validate_prediction(path: Path, min_size_kb: int = 300) -> bool:
    """Check if prediction file is real (not a stub)."""
    size_kb = path.stat().st_size / 1024
    if size_kb < min_size_kb:
        logger.warning("%s is only %.1f KB — likely a stub", path.name, size_kb)
        return False

    pred = ad.read_h5ad(path)
    logger.info(
        "%s: %d perturbations, %d genes, output_type=%s",
        path.name, pred.n_obs, pred.n_vars,
        pred.uns.get("output_type", "unknown"),
    )
    return pred.n_obs > 0 and pred.n_vars > 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-genes", type=int, default=200)
    parser.add_argument("--n-ctrl", type=int, default=200)
    parser.add_argument("--n-cells", type=int, default=100)
    parser.add_argument("--partition-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", type=str, default="data/k562_real.h5ad")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    # Step 1: Download predictions
    if not args.skip_download:
        logger.info("Downloading predictions from Modal volume...")
        for model in MODELS:
            path = download_from_volume(model)
            if path:
                logger.info("  Downloaded %s (%.1f KB)", path.name, path.stat().st_size / 1024)

    # Step 2: Validate predictions
    valid_models = {}
    for model in MODELS:
        path = LOCAL_DIR / f"{model}_predictions.h5ad"
        if path.exists() and validate_prediction(path):
            valid_models[model] = path
        else:
            logger.info("  Skipping %s (not available or stub)", model)

    if not valid_models:
        logger.error("No valid model predictions found!")
        logger.info("Available files in %s:", LOCAL_DIR)
        if LOCAL_DIR.exists():
            for f in LOCAL_DIR.iterdir():
                logger.info("  %s (%.1f KB)", f.name, f.stat().st_size / 1024)
        return

    logger.info("Valid models: %s", list(valid_models.keys()))

    # Step 3: Load K562 data
    logger.info("Loading K562 data from %s...", args.data_path)
    adata = ad.read_h5ad(args.data_path)
    logger.info("Loaded: %d cells, %d genes", adata.n_obs, adata.n_vars)

    # Identify gene subset
    rng = np.random.default_rng(args.seed)
    control_mask = adata.obs["gene"] == "non-targeting"
    pert_genes = [g for g in adata.obs["gene"].unique() if g != "non-targeting"]
    eligible = [g for g in pert_genes if (adata.obs["gene"] == g).sum() >= 10]
    eligible = [g for g in eligible if g in adata.var_names or
                ("gene_name" in adata.var.columns and g in adata.var["gene_name"].values)]
    eligible.sort()

    gene_subset = sorted(rng.choice(eligible, size=min(args.n_genes, len(eligible)), replace=False))
    logger.info("Gene subset: %d genes", len(gene_subset))

    # Get gene names for alignment
    if any(str(v).startswith("ENSG") for v in adata.var_names[:5]) and "gene_name" in adata.var.columns:
        gene_names = list(adata.var["gene_name"].astype(str).values)
    else:
        gene_names = list(adata.var_names)

    # Filter to gene subset
    gene_idx = [i for i, g in enumerate(gene_names) if g in set(gene_subset)]
    gene_names_subset = [gene_names[i] for i in gene_idx]

    # Step 4: Build ground truth from real data
    logger.info("Building ground truth from real perturbation data...")
    control_adata = adata[control_mask].copy()

    # Step 5: For each valid model, build CausalBench input and run GIES
    results = {}
    for model, path in valid_models.items():
        logger.info("Processing %s...", model)
        t0 = time.time()

        try:
            # Build CausalBenchInput using build_from_disk with a temporary dir
            # containing just this model's predictions
            import tempfile
            import shutil
            with tempfile.TemporaryDirectory() as tmpdir:
                shutil.copy2(path, os.path.join(tmpdir, path.name))
                cb_input = build_from_disk(
                    tmpdir,
                    control_adata,
                    gene_names_subset,
                    n_cells_per_pert=args.n_cells,
                    max_control_cells=args.n_ctrl,
                    seed=args.seed,
                )

            prep_time = time.time() - t0
            logger.info("  %s: %d total cells (%.1fs prep)", model, cb_input.total_cells, prep_time)

            # Run GIES
            t0 = time.time()
            edges = run_gies(cb_input, seed=args.seed, partition_size=args.partition_size)
            gies_time = time.time() - t0
            logger.info("  %s: %d edges found (%.1fs GIES)", model, len(edges), gies_time)

            results[model] = {
                "n_edges": len(edges),
                "edges": sorted(edges),
                "prep_time": prep_time,
                "gies_time": gies_time,
            }

        except Exception as e:
            logger.error("  %s failed: %s", model, e)
            import traceback
            traceback.print_exc()

    # Step 6: Save results
    output_path = Path("results/modal_eval_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "models_evaluated": list(results.keys()),
            "n_genes": len(gene_subset),
            "gene_subset": gene_subset,
            "seed": args.seed,
            "results": {
                k: {kk: vv for kk, vv in v.items() if kk != "edges"}
                for k, v in results.items()
            },
        }, f, indent=2)

    logger.info("Results saved to %s", output_path)

    # Print summary
    print("\n=== Modal Model Evaluation Results ===")
    print(f"Gene subset: {len(gene_subset)} genes")
    for model, r in results.items():
        print(f"  {model}: {r['n_edges']} edges ({r['gies_time']:.1f}s)")


if __name__ == "__main__":
    main()
