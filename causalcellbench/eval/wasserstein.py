"""Wasserstein distance metric — CausalBench's primary statistical metric.

For each predicted edge (A -> B), measures the distributional shift in B
when A is intervened vs observational. Higher Wasserstein distance means
the predicted edge corresponds to a real functional effect.

Reference: Chevalley et al., Communications Biology 2025.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance as _wasserstein_1d


def mean_wasserstein_distance(
    predicted_edges: list[tuple[str, str]],
    expression_matrix: np.ndarray,
    interventions: list[str],
    gene_names: list[str],
    max_edges: int | None = None,
) -> dict[str, float]:
    """Compute mean Wasserstein distance across predicted edges.

    For each edge (A -> B):
        1. Get expression of B in cells where A was intervened
        2. Get expression of B in control (non-targeting) cells
        3. Compute 1D Wasserstein distance between distributions

    Parameters
    ----------
    predicted_edges
        List of (source, target) edge tuples.
    expression_matrix
        Shape (n_cells, n_genes).
    interventions
        Per-cell intervention label.
    gene_names
        Gene names matching columns.
    max_edges
        If set, only evaluate top N edges.

    Returns
    -------
    dict with ``mean_wasserstein``, ``median_wasserstein``, ``n_evaluated``.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Group cell indices by intervention
    ctrl_idx = [i for i, v in enumerate(interventions) if v == "non-targeting"]
    pert_groups: dict[str, list[int]] = {}
    for i, v in enumerate(interventions):
        if v != "non-targeting":
            pert_groups.setdefault(v, []).append(i)

    edges_to_eval = predicted_edges
    if max_edges is not None:
        edges_to_eval = edges_to_eval[:max_edges]

    distances = []
    for src, tgt in edges_to_eval:
        if src not in pert_groups or tgt not in gene_to_idx:
            continue
        tgt_idx = gene_to_idx[tgt]
        pert_vals = expression_matrix[pert_groups[src], tgt_idx]
        ctrl_vals = expression_matrix[ctrl_idx, tgt_idx]

        if len(pert_vals) < 5 or len(ctrl_vals) < 5:
            continue

        wd = _wasserstein_1d(ctrl_vals, pert_vals)
        distances.append(wd)

    if not distances:
        return {"mean_wasserstein": 0.0, "median_wasserstein": 0.0, "n_evaluated": 0}

    return {
        "mean_wasserstein": float(np.mean(distances)),
        "median_wasserstein": float(np.median(distances)),
        "n_evaluated": len(distances),
    }
