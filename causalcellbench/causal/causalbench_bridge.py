"""Bridge between model outputs (disk-based .h5ad) and CausalBench input format.

Architecture: Each model runs in an isolated conda environment and writes
predictions to disk as .h5ad files. This bridge reads those files, applies
overlay sampling for mean-only models (preserving gene-gene correlations),
and produces a unified CausalBenchInput ready for causal discovery algorithms.

CausalBench algorithms expect:
    - expression_matrix: np.ndarray of shape (total_cells, n_genes)
    - interventions: list[str] — "non-targeting" for controls, gene name for perturbed
    - gene_names: list[str]
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from anndata import AnnData

from causalcellbench.models.sampling import (
    sample_perturbed_cells_overlay,
)

logger = logging.getLogger(__name__)


@dataclass
class CausalBenchInput:
    """Formatted input ready for CausalBench algorithms.

    Attributes
    ----------
    expression_matrix
        Shape ``(total_cells, n_genes)`` — control + all perturbation cells.
    interventions
        Per-cell intervention label: ``"non-targeting"`` or gene name.
    gene_names
        Gene names matching columns of expression_matrix.
    n_control_cells
        Number of control cells at the start of the matrix.
    n_perturbation_cells
        Total number of perturbation cells.
    perturbed_genes
        Unique genes that were perturbed (sorted).
    """

    expression_matrix: np.ndarray
    interventions: list[str]
    gene_names: list[str]
    n_control_cells: int
    n_perturbation_cells: int
    perturbed_genes: list[str]

    @property
    def total_cells(self) -> int:
        return self.n_control_cells + self.n_perturbation_cells

    def validate(self) -> None:
        """Check internal consistency. Raises ValueError on any mismatch."""
        n_cells, n_genes = self.expression_matrix.shape
        if n_cells != len(self.interventions):
            raise ValueError(
                f"Cell count mismatch: matrix has {n_cells} rows "
                f"but interventions has {len(self.interventions)} entries"
            )
        if n_genes != len(self.gene_names):
            raise ValueError(
                f"Gene count mismatch: matrix has {n_genes} cols "
                f"but gene_names has {len(self.gene_names)} entries"
            )
        if n_cells != self.total_cells:
            raise ValueError(
                f"Cell count mismatch: {n_cells} rows but "
                f"{self.n_control_cells} control + "
                f"{self.n_perturbation_cells} pert = {self.total_cells}"
            )
        n_ctrl = sum(1 for i in self.interventions if i == "non-targeting")
        if n_ctrl != self.n_control_cells:
            raise ValueError(
                f"Expected {self.n_control_cells} 'non-targeting' labels, "
                f"found {n_ctrl}"
            )
        # Check for NaN/Inf in expression matrix
        if not np.isfinite(self.expression_matrix).all():
            n_nan = np.isnan(self.expression_matrix).sum()
            n_inf = np.isinf(self.expression_matrix).sum()
            raise ValueError(
                f"Expression matrix contains non-finite values: "
                f"{n_nan} NaN, {n_inf} Inf"
            )
        # Check perturbed_genes matches actual non-control labels
        actual_pert = sorted(set(
            i for i in self.interventions if i != "non-targeting"
        ))
        if actual_pert != sorted(self.perturbed_genes):
            raise ValueError(
                f"perturbed_genes mismatch: declared {sorted(self.perturbed_genes)[:5]}... "
                f"but interventions contain {actual_pert[:5]}..."
            )


def build_from_disk(
    results_dir: str | Path,
    control_adata: AnnData,
    gene_names: list[str],
    n_cells_per_pert: int = 100,
    max_control_cells: int | None = None,
    seed: int = 0,
) -> CausalBenchInput:
    """Read model .h5ad outputs from disk and build CausalBenchInput.

    For models that output means: apply overlay sampling (bootstrap controls +
    fold-change) to preserve gene-gene correlations for GIES.
    For models that output cell-level predictions (CPA): use directly.

    Parameters
    ----------
    results_dir
        Directory containing model output .h5ad files.
    control_adata
        AnnData of unperturbed control cells.
    gene_names
        Gene names to include (defines column ordering).
    n_cells_per_pert
        Number of cells to sample per perturbation (for mean-only models).
    max_control_cells
        If set, subsample control cells.
    seed
        Random seed.

    Returns
    -------
    CausalBenchInput
    """
    results_dir = Path(results_dir)
    rng = np.random.default_rng(seed)

    # Extract and align control expression matrix
    ctrl_X = _extract_aligned_matrix(control_adata, gene_names)
    if max_control_cells and ctrl_X.shape[0] > max_control_cells:
        idx = rng.choice(ctrl_X.shape[0], size=max_control_cells, replace=False)
        ctrl_X = ctrl_X[idx]

    n_control = ctrl_X.shape[0]

    # Collect all perturbation cells
    all_pert_matrices = []
    all_pert_labels = []

    h5ad_files = sorted(results_dir.glob("*.h5ad"))
    if not h5ad_files:
        raise FileNotFoundError(f"No .h5ad files found in {results_dir}")

    for h5ad_path in h5ad_files:
        logger.info("Reading %s", h5ad_path.name)
        import anndata as ad
        pred = ad.read_h5ad(h5ad_path)

        output_type = pred.uns.get("output_type", "mean")
        model_name = pred.uns.get("model", h5ad_path.stem)

        if output_type == "cells":
            # CPA or similar: already has cell-level predictions
            # Group by perturbation and align genes
            for gene in pred.obs["perturbation"].unique():
                mask = pred.obs["perturbation"] == gene
                gene_cells = pred[mask]
                aligned = _extract_aligned_matrix(gene_cells, gene_names)
                all_pert_matrices.append(aligned)
                all_pert_labels.extend([gene] * aligned.shape[0])
        else:
            # Mean-only models: each row is a mean prediction for one perturbation
            for idx_row in range(pred.n_obs):
                gene = str(pred.obs["perturbation"].iloc[idx_row])
                mean_expr = pred.X[idx_row]
                if hasattr(mean_expr, "toarray"):
                    mean_expr = mean_expr.toarray().flatten()
                mean_expr = np.asarray(mean_expr, dtype=np.float64).flatten()

                # Align genes if needed
                mean_aligned = _align_mean_prediction(
                    mean_expr, list(pred.var_names), gene_names
                )

                # Overlay sampling: preserves gene-gene correlations for GIES
                sampled = sample_perturbed_cells_overlay(
                    ctrl_X, mean_aligned, n_cells_per_pert,
                    seed=seed + int(hashlib.md5(gene.encode()).hexdigest()[:8], 16),
                )
                all_pert_matrices.append(sampled)
                all_pert_labels.extend([gene] * n_cells_per_pert)

    # Stack everything
    if all_pert_matrices:
        pert_X = np.concatenate(all_pert_matrices, axis=0).astype(np.float32)
    else:
        pert_X = np.empty((0, len(gene_names)), dtype=np.float32)

    expression_matrix = np.concatenate(
        [ctrl_X.astype(np.float32), pert_X], axis=0
    )
    interventions = ["non-targeting"] * n_control + all_pert_labels
    perturbed_genes = sorted(set(all_pert_labels))

    result = CausalBenchInput(
        expression_matrix=expression_matrix,
        interventions=interventions,
        gene_names=gene_names,
        n_control_cells=n_control,
        n_perturbation_cells=pert_X.shape[0],
        perturbed_genes=perturbed_genes,
    )
    result.validate()

    logger.info(
        "CausalBenchInput: %d control + %d pert cells, %d genes, %d perturbations",
        n_control, pert_X.shape[0], len(gene_names), len(perturbed_genes),
    )
    return result


def build_from_memory(
    control_adata: AnnData,
    perturbation_results: dict[str, "PerturbationResult"],
    gene_names: list[str],
    max_control_cells: int | None = None,
    seed: int = 0,
) -> CausalBenchInput:
    """Build CausalBenchInput from in-memory PerturbationResults.

    Alternative to disk-based loading for models that run in the same
    process (e.g., Elastic Net baseline, unit tests).
    """
    from causalcellbench.models.base import PerturbationResult

    rng = np.random.default_rng(seed)

    ctrl_X = _extract_aligned_matrix(control_adata, gene_names)
    if max_control_cells and ctrl_X.shape[0] > max_control_cells:
        idx = rng.choice(ctrl_X.shape[0], size=max_control_cells, replace=False)
        ctrl_X = ctrl_X[idx]

    n_control = ctrl_X.shape[0]
    pert_matrices = []
    pert_labels = []

    for gene_name in sorted(perturbation_results.keys()):
        result = perturbation_results[gene_name]
        result.validate()
        result_gene_idx = _get_gene_indices(result.gene_names, gene_names)
        expr = result.expression_matrix[:, result_gene_idx]
        pert_matrices.append(expr)
        pert_labels.extend([gene_name] * expr.shape[0])

    if pert_matrices:
        pert_X = np.concatenate(pert_matrices, axis=0).astype(np.float32)
    else:
        pert_X = np.empty((0, len(gene_names)), dtype=np.float32)

    expression_matrix = np.concatenate(
        [ctrl_X.astype(np.float32), pert_X], axis=0
    )
    interventions = ["non-targeting"] * n_control + pert_labels

    result = CausalBenchInput(
        expression_matrix=expression_matrix,
        interventions=interventions,
        gene_names=gene_names,
        n_control_cells=n_control,
        n_perturbation_cells=pert_X.shape[0],
        perturbed_genes=sorted(set(pert_labels)),
    )
    result.validate()
    return result


def _extract_aligned_matrix(
    adata: AnnData, target_genes: list[str]
) -> np.ndarray:
    """Extract expression matrix with columns aligned to target gene order."""
    gene_idx = _get_gene_indices(list(adata.var_names), target_genes)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X[:, gene_idx], dtype=np.float32)


def _align_mean_prediction(
    mean_expr: np.ndarray,
    source_genes: list[str],
    target_genes: list[str],
) -> np.ndarray:
    """Reorder a mean prediction vector to match target gene ordering."""
    idx = _get_gene_indices(source_genes, target_genes)
    return mean_expr[idx]


def _get_gene_indices(
    source_names: list[str], target_names: list[str]
) -> list[int]:
    """Get indices to reorder source genes to match target ordering."""
    source_to_idx = {name: i for i, name in enumerate(source_names)}
    indices = []
    missing = []
    for name in target_names:
        if name in source_to_idx:
            indices.append(source_to_idx[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(
            f"{len(missing)} target genes not found in source: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    return indices
