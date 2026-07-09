"""CausalCellBench orchestration script.

Runs all model wrappers in isolated conda environments, then
evaluates their predictions through causal discovery algorithms.

Usage:
    python scripts/benchmark.py --cell-line k562 --n-genes 500 --seed 0
    python scripts/benchmark.py --cell-line k562 --n-genes 200 --pilot  # Gate 0
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Model configurations: environment name, script path, extra args
MODELS = {
    "scgpt": {
        "env": "env_scgpt",
        "script": "causalcellbench/models/scgpt_wrapper.py",
        "extra_args": ["--checkpoint", "checkpoints/scgpt"],
    },
    "geneformer": {
        "env": "env_geneformer",
        "script": "causalcellbench/models/geneformer_wrapper.py",
        "extra_args": [],
    },
    "cpa": {
        "env": "env_cpa",
        "script": "causalcellbench/models/cpa_wrapper.py",
        "extra_args": ["--checkpoint", "checkpoints/cpa"],
    },
    "gears": {
        "env": "env_gears",
        "script": "causalcellbench/models/gears_wrapper.py",
        "extra_args": [],
    },
}

# Lightweight models run in eval env (no torch conflicts)
EVAL_ENV = "env_eval"


def run_model(
    model_name: str,
    config: dict,
    input_path: str,
    output_path: str,
    genes: list[str],
) -> bool:
    """Run a single model in its isolated conda environment."""
    cmd = [
        "conda",
        "run",
        "-n",
        config["env"],
        "python",
        config["script"],
        "--input",
        input_path,
        "--output",
        output_path,
        "--genes",
        ",".join(genes),
    ] + config.get("extra_args", [])

    logger.info("Running %s: %s", model_name, " ".join(cmd[:6]) + " ...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            logger.error(
                "%s FAILED (exit %d):\n%s",
                model_name,
                result.returncode,
                result.stderr[-500:],
            )
            return False
        logger.info("%s completed successfully", model_name)
        return True
    except subprocess.TimeoutExpired:
        logger.error("%s TIMED OUT (1 hour limit)", model_name)
        return False
    except FileNotFoundError:
        logger.error("conda not found. Is conda installed and on PATH?")
        return False


def run_baselines(
    input_path: str,
    results_dir: str,
    genes: list[str],
    n_cells: int,
    seed: int,
) -> None:
    """Run lightweight baselines (Elastic Net, Random) in eval env."""
    import anndata as ad
    from causalcellbench.models.baselines import ElasticNetBaseline, RandomBaseline
    import pandas as pd

    control = ad.read_h5ad(input_path)

    # Elastic Net
    logger.info("Running Elastic Net baseline...")
    enet = ElasticNetBaseline()
    enet.load_model(adata_train=control)

    enet_preds = []
    enet_labels = []
    for gene in genes:
        try:
            result = enet.predict_perturbation(control, gene, n_cells, genes, seed)
            enet_preds.append(result.expression_matrix.mean(axis=0))
            enet_labels.append(gene)
        except Exception as e:
            logger.warning("ElasticNet failed on %s: %s", gene, e)

    if enet_preds:
        enet_adata = ad.AnnData(
            X=np.stack(enet_preds).astype(np.float32),
            obs=pd.DataFrame({"perturbation": enet_labels}),
            var=pd.DataFrame(index=genes),
            uns={"model": "elastic_net", "output_type": "mean"},
        )
        enet_adata.write_h5ad(f"{results_dir}/elastic_net_predictions.h5ad")

    # Random baseline
    logger.info("Running Random baseline...")
    rand = RandomBaseline()
    rand.load_model()

    rand_preds = []
    rand_labels = []
    for gene in genes:
        result = rand.predict_perturbation(control, gene, n_cells, genes, seed)
        rand_preds.append(result.expression_matrix.mean(axis=0))
        rand_labels.append(gene)

    rand_adata = ad.AnnData(
        X=np.stack(rand_preds).astype(np.float32),
        obs=pd.DataFrame({"perturbation": rand_labels}),
        var=pd.DataFrame(index=genes),
        uns={"model": "random", "output_type": "mean"},
    )
    rand_adata.write_h5ad(f"{results_dir}/random_predictions.h5ad")


def run_evaluation(
    results_dir: str,
    control_path: str,
    gene_names: list[str],
    n_cells_per_pert: int,
    seed: int,
) -> dict:
    """Run causal discovery + evaluation on all model outputs."""
    import anndata as ad
    from causalcellbench.causal.causalbench_bridge import build_from_disk
    from causalcellbench.causal.gies_runner import run_gies

    control = ad.read_h5ad(control_path)
    all_results = {}

    # Process each model's output
    for h5ad_path in sorted(Path(results_dir).glob("*_predictions.h5ad")):
        model_name = h5ad_path.stem.replace("_predictions", "")
        logger.info("Evaluating %s...", model_name)

        # Create a temp dir with just this model's output
        tmp_dir = Path(results_dir) / f"_tmp_{model_name}"
        tmp_dir.mkdir(exist_ok=True)
        import shutil

        shutil.copy(h5ad_path, tmp_dir / h5ad_path.name)

        try:
            # Bridge: disk → CausalBenchInput with NB sampling
            cb_input = build_from_disk(
                results_dir=tmp_dir,
                control_adata=control,
                gene_names=gene_names,
                n_cells_per_pert=n_cells_per_pert,
                seed=seed,
            )

            # GIES causal discovery
            edges = run_gies(cb_input, seed=seed, partition_size=30)

            # Metrics (using random baseline AUPRC as reference)
            # TODO: Load actual ground truth from ChIP-seq/TRRUST
            metrics = {
                "n_edges": len(edges),
                "n_cells": cb_input.total_cells,
                "n_control": cb_input.n_control_cells,
                "n_pert": cb_input.n_perturbation_cells,
            }
            all_results[model_name] = metrics
            logger.info("  %s: %d edges discovered", model_name, len(edges))
        except Exception as e:
            logger.error("  %s evaluation failed: %s", model_name, e)
            all_results[model_name] = {"error": str(e)}
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="CausalCellBench benchmark")
    parser.add_argument("--cell-line", default="k562", choices=["k562", "rpe1"])
    parser.add_argument("--n-genes", type=int, default=500)
    parser.add_argument("--n-cells", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--pilot", action="store_true", help="Gate 0 pilot (200 genes, 3 models)"
    )
    parser.add_argument("--models", nargs="*", help="Specific models to run")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    if args.pilot:
        args.n_genes = 200
        logger.info("PILOT MODE: 200 genes, Gate 0 assessment")

    # Step 1: Prepare control data
    logger.info("Step 1: Loading %s data...", args.cell_line)
    from causalcellbench.data.replogle import (
        load_replogle_via_pertpy,
        extract_control_cells,
        get_perturbed_genes,
        select_gene_subset,
    )

    adata = load_replogle_via_pertpy(args.cell_line)
    control = extract_control_cells(adata)
    perturbed_genes = get_perturbed_genes(adata)
    gene_subset = select_gene_subset(perturbed_genes, args.n_genes, args.seed)

    control_path = str(results_dir / "control.h5ad")
    control.write_h5ad(control_path)
    logger.info(
        "Control: %d cells. Gene subset: %d genes.", control.n_obs, len(gene_subset)
    )

    # Step 2: Run models
    logger.info("Step 2: Running models...")
    models_to_run = MODELS
    if args.models:
        models_to_run = {k: v for k, v in MODELS.items() if k in args.models}

    model_results = {}
    for model_name, config in models_to_run.items():
        output_path = str(results_dir / f"{model_name}_predictions.h5ad")
        success = run_model(model_name, config, control_path, output_path, gene_subset)
        model_results[model_name] = "success" if success else "failed"

    # Step 3: Run baselines
    logger.info("Step 3: Running baselines...")
    run_baselines(control_path, str(results_dir), gene_subset, args.n_cells, args.seed)

    # Step 4: Evaluate
    logger.info("Step 4: Evaluating...")
    eval_results = run_evaluation(
        str(results_dir),
        control_path,
        gene_subset,
        args.n_cells,
        args.seed,
    )

    # Step 5: Save summary
    summary = {
        "cell_line": args.cell_line,
        "n_genes": args.n_genes,
        "n_cells_per_pert": args.n_cells,
        "seed": args.seed,
        "model_results": model_results,
        "evaluation": eval_results,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Benchmark complete. Results in %s", results_dir)
    logger.info("Summary: %s", json.dumps(model_results, indent=2))


if __name__ == "__main__":
    main()
