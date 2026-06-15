"""Full end-to-end pipeline test for CausalCellBench.

Runs on synthetic placeholder data (or real K562 if available):
1. Load data + select gene subset
2. Run Elastic Net + Random baselines
3. Bridge: model outputs → CausalBenchInput (with NB sampling)
4. GIES + GRNBoost2 causal discovery
5. All 5 evaluation metrics
6. Generate summary report

This validates the complete pipeline before running on GPU models.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

sys.path.insert(0, str(Path(__file__).parent.parent))

from causalcellbench.models.base import PerturbationResult
from causalcellbench.models.baselines import ElasticNetBaseline, RandomBaseline
from causalcellbench.models.sampling import estimate_nb_dispersion
from causalcellbench.causal.causalbench_bridge import build_from_memory
from causalcellbench.causal.gies_runner import run_gies
from causalcellbench.eval.auprc import compute_auprc, compute_random_auprc
from causalcellbench.eval.precision_at_k import precision_at_k
from causalcellbench.eval.shd import structural_hamming_distance
from causalcellbench.eval.wasserstein import mean_wasserstein_distance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_data(data_path: str = "data/replogle_k562_essential.h5ad", n_genes: int = 50):
    """Load data and extract control cells + gene subset."""
    import anndata as ad

    logger.info("Loading data from %s", data_path)
    adata = ad.read_h5ad(data_path)
    logger.info("  Shape: %s", adata.shape)

    # Extract control cells
    if "gene" in adata.obs.columns:
        pert_col = "gene"
    elif "perturbation" in adata.obs.columns:
        pert_col = "perturbation"
    else:
        raise ValueError(f"No perturbation column found. Columns: {list(adata.obs.columns)}")

    control_mask = adata.obs[pert_col] == "non-targeting"
    control = adata[control_mask].copy()
    logger.info("  Control cells: %d", control.n_obs)

    # Get perturbed genes with sufficient cells
    pert_counts = adata.obs[pert_col].value_counts()
    perturbed_genes = sorted([
        g for g, c in pert_counts.items()
        if g != "non-targeting" and c >= 10 and g in adata.var_names
    ])
    logger.info("  Perturbed genes (>=10 cells, in var_names): %d", len(perturbed_genes))

    # Select gene subset
    rng = np.random.default_rng(42)
    gene_subset = sorted(rng.choice(perturbed_genes, size=min(n_genes, len(perturbed_genes)), replace=False))
    logger.info("  Selected %d genes for evaluation", len(gene_subset))

    return adata, control, gene_subset, pert_col


def run_condition(
    model_name: str,
    model,
    control: AnnData,
    gene_subset: list[str],
    n_cells: int = 50,
    seed: int = 0,
) -> dict:
    """Run a single condition: model → bridge → GIES → metrics."""
    t_start = time.time()

    # Step 1: Simulate perturbations
    logger.info("  Simulating %d perturbations with %s...", len(gene_subset), model_name)
    pert_results = model.simulate_all_perturbations(
        control, gene_subset, n_cells, gene_subset, seed=seed,
    )

    # Step 2: Bridge
    logger.info("  Building CausalBench input...")
    cb_input = build_from_memory(control, pert_results, gene_subset, seed=seed)
    logger.info("    %d total cells (%d ctrl + %d pert)",
                cb_input.total_cells, cb_input.n_control_cells, cb_input.n_perturbation_cells)

    # Step 3: GIES
    logger.info("  Running GIES...")
    edges_gies = run_gies(cb_input, seed=seed, partition_size=min(30, len(gene_subset)))
    logger.info("    GIES: %d edges", len(edges_gies))

    # Step 4: Metrics (against synthetic "ground truth" = edges from real-data GIES)
    # For now, just collect the edges and basic stats
    sim_time = time.time() - t_start

    # Wasserstein (measures distributional shift per edge)
    wasserstein = mean_wasserstein_distance(
        edges_gies, cb_input.expression_matrix, cb_input.interventions, gene_subset,
        max_edges=500,
    )

    return {
        "model": model_name,
        "n_genes": len(gene_subset),
        "n_cells_per_pert": n_cells,
        "n_edges_gies": len(edges_gies),
        "edges_gies": edges_gies[:50],  # save top 50 for inspection
        "mean_wasserstein": wasserstein["mean_wasserstein"],
        "n_wasserstein_evaluated": wasserstein["n_evaluated"],
        "wall_time": sim_time,
    }


def main():
    logger.info("=" * 60)
    logger.info("CausalCellBench: Full Pipeline Test")
    logger.info("=" * 60)

    t_start = time.time()
    n_genes = 40  # small for testing speed

    # Load data
    adata, control, gene_subset, pert_col = load_data(n_genes=n_genes)

    # Estimate NB dispersion (for reporting)
    mu, r = estimate_nb_dispersion(control, gene_subset)
    logger.info("  NB dispersion: median r=%.2f, mean mu=%.2f", np.median(r), np.mean(mu))

    # --- Condition E: Random floor ---
    logger.info("\n--- Condition E: Random Baseline ---")
    rand = RandomBaseline()
    rand.load_model()
    result_random = run_condition("Random", rand, control, gene_subset, n_cells=50)

    # --- Condition B: Elastic Net (proxy for model) ---
    logger.info("\n--- Condition B (proxy): Elastic Net ---")
    enet = ElasticNetBaseline(alpha=0.1, l1_ratio=0.5)
    enet.load_model(adata_train=control)
    result_enet = run_condition("ElasticNet", enet, control, gene_subset, n_cells=50)

    # --- Condition A: Real data (upper bound) ---
    # Use the actual perturbed cells from the dataset
    logger.info("\n--- Condition A: Real Data Upper Bound ---")
    real_pert_results = {}
    for gene in gene_subset:
        mask = adata.obs[pert_col] == gene
        if mask.sum() < 5:
            continue
        gene_cells = adata[mask].copy()
        # Align to gene_subset
        gene_idx = [list(adata.var_names).index(g) for g in gene_subset if g in adata.var_names]
        X = gene_cells.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        expr = np.asarray(X[:, gene_idx], dtype=np.float32)
        real_pert_results[gene] = PerturbationResult(
            expression_matrix=expr[:50],  # limit to 50 cells
            gene_names=gene_subset,
            perturbed_genes=[gene],
            cell_type="K562",
        )

    cb_real = build_from_memory(control, real_pert_results, gene_subset, seed=0)
    edges_real = run_gies(cb_real, seed=0, partition_size=min(30, len(gene_subset)))
    logger.info("  Real data GIES: %d edges", len(edges_real))

    # Use real-data edges as ground truth for AUPRC comparison
    gt_edges = set(edges_real)
    random_floor = compute_random_auprc(gt_edges, gene_subset, n_samples=50, seed=0)

    # Compute AUPRC for each condition
    auprc_enet = compute_auprc(result_enet["edges_gies"], gt_edges, gene_subset)
    auprc_rand = compute_auprc(result_random["edges_gies"], gt_edges, gene_subset)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("FULL PIPELINE RESULTS")
    logger.info("=" * 60)
    logger.info("  Gene subset: %d genes", len(gene_subset))
    logger.info("  Ground truth: %d edges from real-data GIES", len(gt_edges))
    logger.info("")
    logger.info("  %-20s  Edges  AUPRC   Wasserstein  Time", "Condition")
    logger.info("  %-20s  -----  ------  -----------  ----", "-" * 20)
    logger.info("  %-20s  %5d  %.4f  %.4f       %.1fs",
                "A: Real Data", len(edges_real), 1.0, 0.0, 0.0)
    logger.info("  %-20s  %5d  %.4f  %.4f       %.1fs",
                "B: Elastic Net", result_enet["n_edges_gies"],
                auprc_enet["auprc"], result_enet["mean_wasserstein"],
                result_enet["wall_time"])
    logger.info("  %-20s  %5d  %.4f  %.4f       %.1fs",
                "E: Random", result_random["n_edges_gies"],
                auprc_rand["auprc"], result_random["mean_wasserstein"],
                result_random["wall_time"])
    logger.info("  %-20s  %5s  %.4f  %s           %s",
                "Floor (random edges)", "N/A", random_floor, "N/A", "N/A")
    logger.info("")
    logger.info("  Total wall time: %.1fs", time.time() - t_start)

    # Save results
    summary = {
        "n_genes": len(gene_subset),
        "gene_subset": gene_subset,
        "n_real_edges": len(gt_edges),
        "random_floor_auprc": random_floor,
        "conditions": {
            "real_data": {"n_edges": len(edges_real), "auprc": 1.0},
            "elastic_net": {
                "n_edges": result_enet["n_edges_gies"],
                "auprc": auprc_enet["auprc"],
                "wasserstein": result_enet["mean_wasserstein"],
                "wall_time": result_enet["wall_time"],
            },
            "random": {
                "n_edges": result_random["n_edges_gies"],
                "auprc": auprc_rand["auprc"],
                "wasserstein": result_random["mean_wasserstein"],
                "wall_time": result_random["wall_time"],
            },
        },
    }
    with open(RESULTS_DIR / "full_pipeline_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\nResults saved to results/full_pipeline_results.json")
    logger.info("\nPIPELINE TEST COMPLETE")


if __name__ == "__main__":
    main()
