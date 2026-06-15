"""GRNBoost2 runner — observational causal discovery baseline.

GRNBoost2 is CausalBench's best-performing algorithm on real data.
It uses gradient boosting to predict each gene's expression from all
other genes, then ranks edges by feature importance.

Key properties:
    - Observational only (does NOT use intervention information)
    - No acyclicity constraint (tolerates cyclic GRNs)
    - Scales to genome-wide (thousands of genes)
    - Produces a ranked edge list with confidence scores

If model-simulated data through GRNBoost2 beats model-simulated data
through GIES, it means the interventional structure of the simulations
doesn't help causal discovery — an important finding.

Reference: Moerman et al., Bioinformatics 2019 (arboreto package).
"""

from __future__ import annotations

import logging

import numpy as np

from causalcellbench.causal.causalbench_bridge import CausalBenchInput

logger = logging.getLogger(__name__)


def run_grnboost2(
    cb_input: CausalBenchInput,
    seed: int = 0,
    n_top_edges: int = 5000,
    use_control_only: bool = False,
) -> list[tuple[str, str]]:
    """Run GRNBoost2 on CausalBench-formatted input.

    GRNBoost2 is observational — it ignores intervention labels and
    infers regulatory relationships purely from expression covariation.

    Parameters
    ----------
    cb_input
        Formatted input from bridge.
    seed
        Random seed.
    n_top_edges
        Number of top edges to return (ranked by importance).
    use_control_only
        If True, only use control (non-targeting) cells.
        If False, use all cells (control + perturbed).

    Returns
    -------
    list of (source_gene, target_gene) edge tuples, ranked by confidence.
    """
    try:
        from arboreto.algo import grnboost2
    except ImportError:
        raise ImportError(
            "arboreto not installed. Run: pip install arboreto"
        )

    import pandas as pd

    # Select cells
    if use_control_only:
        mask = [i == "non-targeting" for i in cb_input.interventions]
        expr = cb_input.expression_matrix[mask]
    else:
        expr = cb_input.expression_matrix

    logger.info(
        "Running GRNBoost2 on %d cells x %d genes...",
        expr.shape[0], expr.shape[1],
    )

    # GRNBoost2 expects a DataFrame with gene names as columns
    df = pd.DataFrame(expr, columns=cb_input.gene_names)

    # Run GRNBoost2
    # tf_names = None means all genes are potential regulators
    network = grnboost2(
        expression_data=df,
        tf_names=None,
        seed=seed,
        verbose=False,
    )

    # network is a DataFrame with columns: TF, target, importance
    # Sort by importance (descending) and take top edges
    network = network.sort_values("importance", ascending=False)
    top_edges = [
        (str(row["TF"]), str(row["target"]))
        for _, row in network.head(n_top_edges).iterrows()
    ]

    logger.info("GRNBoost2 returned %d edges", len(top_edges))
    return top_edges


def run_grnboost2_with_weights(
    cb_input: CausalBenchInput,
    seed: int = 0,
    n_top_edges: int = 5000,
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], float]]:
    """Run GRNBoost2 and return edges with importance weights.

    Returns
    -------
    (edges, weights) where weights maps each edge to its importance score.
    Useful for weighted AUPRC computation.
    """
    try:
        from arboreto.algo import grnboost2
    except ImportError:
        raise ImportError("arboreto not installed. Run: pip install arboreto")

    import pandas as pd

    df = pd.DataFrame(
        cb_input.expression_matrix, columns=cb_input.gene_names
    )

    network = grnboost2(
        expression_data=df, tf_names=None, seed=seed, verbose=False,
    )
    network = network.sort_values("importance", ascending=False)

    edges = []
    weights = {}
    for _, row in network.head(n_top_edges).iterrows():
        edge = (str(row["TF"]), str(row["target"]))
        edges.append(edge)
        weights[edge] = float(row["importance"])

    return edges, weights
