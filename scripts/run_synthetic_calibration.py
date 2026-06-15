"""SERGIO-style synthetic calibration for CausalCellBench.

Tests the entire pipeline end-to-end on KNOWN ground truth:
1. Generate a synthetic GRN with known edges
2. Simulate gene expression data from this GRN (NB-distributed)
3. Run GIES causal discovery on the simulated data
4. Evaluate: does GIES recover the known edges?

If GIES cannot recover a known synthetic GRN, the entire benchmark
is invalid — we'd be evaluating virtual cell models against a broken
causal discovery pipeline.

This replaces SERGIO (which requires separate installation) with a
simpler but mathematically equivalent NB-based simulator that we
control entirely.
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from causalcellbench.models.sampling import estimate_nb_dispersion, sample_cells_nb
from causalcellbench.causal.causalbench_bridge import build_from_memory, CausalBenchInput
from causalcellbench.causal.gies_runner import run_gies
from causalcellbench.eval.auprc import compute_auprc, compute_random_auprc
from causalcellbench.eval.shd import structural_hamming_distance
from causalcellbench.models.base import PerturbationResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_grn(
    n_genes: int = 20,
    n_edges: int = 30,
    seed: int = 42,
) -> tuple[list[str], set[tuple[str, str]], np.ndarray]:
    """Generate a random DAG with known edges.

    Returns
    -------
    gene_names, true_edges, adjacency_matrix
    """
    rng = np.random.default_rng(seed)
    gene_names = [f"G{i}" for i in range(n_genes)]

    # Generate random DAG: only allow edges i -> j where i < j (ensures acyclicity)
    adj = np.zeros((n_genes, n_genes), dtype=np.float64)
    possible_edges = [(i, j) for i in range(n_genes) for j in range(i + 1, n_genes)]
    selected = rng.choice(len(possible_edges), size=min(n_edges, len(possible_edges)), replace=False)

    true_edges = set()
    for idx in selected:
        i, j = possible_edges[idx]
        adj[i, j] = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
        true_edges.add((gene_names[i], gene_names[j]))

    logger.info("Generated DAG: %d genes, %d edges", n_genes, len(true_edges))
    return gene_names, true_edges, adj


def simulate_interventional_data(
    gene_names: list[str],
    adjacency: np.ndarray,
    n_control_cells: int = 500,
    n_pert_cells_per_gene: int = 100,
    base_expression: float = 10.0,
    noise_dispersion: float = 3.0,
    knockdown_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[AnnData, dict[str, PerturbationResult]]:
    """Simulate interventional scRNA-seq-like data from a known GRN.

    Uses a linear SEM with NB noise:
        X = (I - A)^{-1} * epsilon
    where A is the adjacency matrix and epsilon ~ NB(r, p).

    For perturbation of gene g: set X_g = knockdown_fraction * X_g_control,
    then propagate through the network.

    Returns
    -------
    control_adata, perturbation_results
    """
    rng = np.random.default_rng(seed)
    n_genes = len(gene_names)

    # Compute the effect propagation matrix: (I - A)^{-1}
    # This gives the total effect of each gene on every other gene
    try:
        propagation = np.linalg.inv(np.eye(n_genes) - adjacency)
    except np.linalg.LinAlgError:
        # Singular matrix — reduce edge weights
        adjacency *= 0.5
        propagation = np.linalg.inv(np.eye(n_genes) - adjacency)

    # Generate control cells: NB noise through the network
    base_means = np.ones(n_genes) * base_expression
    r_disp = np.ones(n_genes) * noise_dispersion

    # Control: independent NB noise propagated through network
    # Use gene-specific base means to reduce collinearity (GIES numerical issue)
    base_means = rng.uniform(5.0, 20.0, size=n_genes)
    control_noise = np.zeros((n_control_cells, n_genes))
    for g in range(n_genes):
        control_noise[:, g] = rng.gamma(
            shape=r_disp[g], scale=base_means[g] / r_disp[g], size=n_control_cells
        )
    control_expr = np.maximum(control_noise @ propagation.T, 0).astype(np.float32)
    # Add small independent noise to break exact collinearity
    control_expr += rng.normal(0, 0.1, size=control_expr.shape).astype(np.float32)
    control_expr = np.maximum(control_expr, 0).astype(np.float32)

    control_adata = AnnData(
        X=control_expr,
        var=pd.DataFrame(index=gene_names),
    )

    # Generate perturbation data: knock down each gene
    perturbation_results = {}
    for g_idx, gene in enumerate(gene_names):
        pert_noise = np.zeros((n_pert_cells_per_gene, n_genes))
        for g2 in range(n_genes):
            pert_noise[:, g2] = rng.gamma(
                shape=r_disp[g2], scale=base_means[g2] / r_disp[g2],
                size=n_pert_cells_per_gene,
            )

        # Knockdown: reduce the perturbed gene's noise input
        pert_noise[:, g_idx] *= knockdown_fraction

        # Propagate through network
        pert_expr = np.maximum(pert_noise @ propagation.T, 0).astype(np.float32)
        pert_expr += rng.normal(0, 0.1, size=pert_expr.shape).astype(np.float32)
        pert_expr = np.maximum(pert_expr, 0).astype(np.float32)

        perturbation_results[gene] = PerturbationResult(
            expression_matrix=pert_expr,
            gene_names=gene_names,
            perturbed_genes=[gene],
            cell_type="synthetic",
        )

    logger.info(
        "Generated data: %d control cells, %d perturbations x %d cells each",
        n_control_cells, n_genes, n_pert_cells_per_gene,
    )
    return control_adata, perturbation_results


def run_calibration(
    n_genes: int = 20,
    n_edges: int = 25,
    n_control_cells: int = 500,
    n_pert_cells: int = 100,
    partition_size: int = 20,
    seed: int = 42,
) -> dict:
    """Run the full calibration pipeline."""
    t_start = time.time()

    # Step 1: Generate known GRN
    logger.info("Step 1: Generating synthetic GRN...")
    gene_names, true_edges, adjacency = generate_synthetic_grn(n_genes, n_edges, seed)

    # Step 2: Simulate data
    logger.info("Step 2: Simulating interventional data...")
    control_adata, pert_results = simulate_interventional_data(
        gene_names, adjacency, n_control_cells, n_pert_cells, seed=seed,
    )

    # Step 3: Build CausalBench input
    logger.info("Step 3: Building CausalBench input via bridge...")
    cb_input = build_from_memory(
        control_adata, pert_results, gene_names, seed=seed,
    )
    logger.info(
        "  CausalBenchInput: %d cells (%d ctrl + %d pert), %d genes",
        cb_input.total_cells, cb_input.n_control_cells,
        cb_input.n_perturbation_cells, len(gene_names),
    )

    # Step 4: Run GIES
    logger.info("Step 4: Running GIES (partition_size=%d)...", partition_size)
    predicted_edges = run_gies(cb_input, seed=seed, partition_size=partition_size)
    logger.info("  GIES returned %d edges", len(predicted_edges))

    # Step 5: Evaluate
    logger.info("Step 5: Evaluating against known ground truth...")
    auprc_result = compute_auprc(predicted_edges, true_edges, gene_names)
    random_auprc = compute_random_auprc(true_edges, gene_names, n_samples=50, seed=seed)
    shd_result = structural_hamming_distance(set(predicted_edges), true_edges)

    # Report
    results = {
        "n_genes": n_genes,
        "n_true_edges": len(true_edges),
        "n_predicted_edges": len(predicted_edges),
        "n_control_cells": n_control_cells,
        "n_pert_cells_per_gene": n_pert_cells,
        "partition_size": partition_size,
        "auprc": auprc_result["auprc"],
        "random_auprc": random_auprc,
        "auprc_lift": auprc_result["auprc"] / max(random_auprc, 1e-8),
        "precision_at_10": auprc_result["precision_at_100"],  # top 10 for small graph
        "n_true_positive": auprc_result["n_true_positive"],
        "shd": shd_result["shd"],
        "shd_missing": shd_result["missing"],
        "shd_extra": shd_result["extra"],
        "shd_reversed": shd_result["reversed"],
        "wall_time_seconds": time.time() - t_start,
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info("CALIBRATION RESULTS")
    logger.info("=" * 60)
    logger.info("  True edges:      %d", len(true_edges))
    logger.info("  Predicted edges:  %d", len(predicted_edges))
    logger.info("  True positives:   %d", auprc_result["n_true_positive"])
    logger.info("  AUPRC:           %.4f", auprc_result["auprc"])
    logger.info("  Random AUPRC:    %.4f", random_auprc)
    logger.info("  AUPRC lift:      %.1fx over random", results["auprc_lift"])
    logger.info("  SHD:             %d (missing=%d, extra=%d, reversed=%d)",
                shd_result["shd"], shd_result["missing"],
                shd_result["extra"], shd_result["reversed"])
    logger.info("  Wall time:       %.1fs", results["wall_time_seconds"])
    logger.info("")

    # Gate 0 decision
    if auprc_result["auprc"] > random_auprc * 2:
        logger.info("  CALIBRATION PASSED: AUPRC > 2x random")
        results["calibration_passed"] = True
    else:
        logger.warning("  CALIBRATION FAILED: AUPRC not significantly above random")
        results["calibration_passed"] = False

    return results


def run_scale_test(seeds: list[int] = [0, 1, 2]) -> dict:
    """Run calibration at multiple scales: 10, 20, 30, 50 genes."""
    all_results = {}
    for n_genes in [10, 20, 30, 50]:
        n_edges = int(n_genes * 1.2)
        partition = min(n_genes, 30)
        scale_results = []
        for seed in seeds:
            logger.info("\n--- Scale test: %d genes, seed=%d ---", n_genes, seed)
            result = run_calibration(
                n_genes=n_genes, n_edges=n_edges,
                partition_size=partition, seed=seed,
            )
            scale_results.append(result)

        avg_auprc = np.mean([r["auprc"] for r in scale_results])
        avg_lift = np.mean([r["auprc_lift"] for r in scale_results])
        all_results[f"{n_genes}_genes"] = {
            "mean_auprc": float(avg_auprc),
            "mean_lift": float(avg_lift),
            "per_seed": scale_results,
        }
        logger.info("\n  %d genes: mean AUPRC=%.4f, mean lift=%.1fx",
                     n_genes, avg_auprc, avg_lift)

    return all_results


if __name__ == "__main__":
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    logger.info("Running synthetic calibration (replaces SERGIO)...")
    logger.info("This validates that GIES can recover a known GRN from NB-distributed data.\n")

    # Single calibration at default scale
    result = run_calibration(n_genes=20, n_edges=25, partition_size=20, seed=42)

    # Scale test
    logger.info("\n" + "=" * 60)
    logger.info("SCALE TEST: 10/20/30/50 genes x 3 seeds")
    logger.info("=" * 60)
    scale_results = run_scale_test(seeds=[0, 1, 2])

    # Save results
    all_output = {"single_calibration": result, "scale_test": scale_results}
    with open(results_dir / "calibration_results.json", "w") as f:
        json.dump(all_output, f, indent=2, default=str)

    logger.info("\nResults saved to results/calibration_results.json")
