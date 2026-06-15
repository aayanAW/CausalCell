"""AUPRC (Area Under Precision-Recall Curve) for causal graph evaluation.

AUPRC is the primary metric for CausalCellBench (Track A and B).
It handles sparse, incomplete ground truth properly — unlike SHD, it
does not penalize predictions absent from incomplete databases.

Ground truth sources:
    - Track A: Replogle perturbation-effect structure (via CausalBench)
    - Track B: ENCODE K562 ChIP-seq + TRRUST v2 curated TF-target pairs
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
)


def compute_auprc(
    predicted_edges: list[tuple[str, str]],
    ground_truth_edges: set[tuple[str, str]],
    all_genes: list[str],
    edge_weights: dict[tuple[str, str], float] | None = None,
) -> dict[str, float]:
    """Compute AUPRC of predicted edges against ground truth.

    Parameters
    ----------
    predicted_edges
        List of (source, target) edges from causal discovery.
        Order matters: earlier edges are higher confidence.
    ground_truth_edges
        Set of known true (source, target) edges.
    all_genes
        All gene names (for computing the total possible edge space).
    edge_weights
        Optional confidence weights per edge. If None, uses rank-based
        weights (earlier = higher confidence).

    Returns
    -------
    dict with keys:
        - ``auprc``: Area Under Precision-Recall Curve
        - ``precision_at_100``: Precision at top 100 predictions
        - ``precision_at_500``: Precision at top 500 predictions
        - ``n_predicted``: Number of predicted edges
        - ``n_ground_truth``: Number of ground truth edges
        - ``n_true_positive``: Number of correct predictions
    """
    if not predicted_edges:
        return {
            "auprc": 0.0,
            "precision_at_100": 0.0,
            "precision_at_500": 0.0,
            "n_predicted": 0,
            "n_ground_truth": len(ground_truth_edges),
            "n_true_positive": 0,
        }

    # Build binary labels and scores
    n_pred = len(predicted_edges)
    y_true = np.array(
        [1 if e in ground_truth_edges else 0 for e in predicted_edges],
        dtype=np.int32,
    )

    if edge_weights is not None:
        y_score = np.array(
            [edge_weights.get(e, 0.0) for e in predicted_edges],
            dtype=np.float64,
        )
    else:
        # Rank-based: earlier edges get higher scores
        y_score = np.linspace(1.0, 0.0, n_pred)

    # AUPRC
    auprc = float(average_precision_score(y_true, y_score))

    # Precision@K
    p_at_100 = _precision_at_k(y_true, 100)
    p_at_500 = _precision_at_k(y_true, 500)

    n_tp = int(y_true.sum())

    return {
        "auprc": auprc,
        "precision_at_100": p_at_100,
        "precision_at_500": p_at_500,
        "n_predicted": n_pred,
        "n_ground_truth": len(ground_truth_edges),
        "n_true_positive": n_tp,
    }


def _precision_at_k(y_true: np.ndarray, k: int) -> float:
    """Precision at top K predictions (assumes y_true is in ranked order)."""
    if len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    return float(y_true[:k].sum() / k)


def compute_random_auprc(
    ground_truth_edges: set[tuple[str, str]],
    all_genes: list[str],
    n_samples: int = 10,
    n_edges: int = 1000,
    seed: int = 0,
) -> float:
    """Compute expected AUPRC for random edge predictions (floor baseline).

    Parameters
    ----------
    ground_truth_edges
        Set of true edges.
    all_genes
        All gene names.
    n_samples
        Number of random draws to average over.
    n_edges
        Number of random edges per sample.
    seed
        Random seed.

    Returns
    -------
    Mean AUPRC across random samples.
    """
    rng = np.random.default_rng(seed)
    n_genes = len(all_genes)
    auprcs = []

    for s in range(n_samples):
        # Random directed edges (no self-loops)
        sources = rng.integers(0, n_genes, size=n_edges)
        targets = rng.integers(0, n_genes, size=n_edges)
        edges = [
            (all_genes[s_], all_genes[t_])
            for s_, t_ in zip(sources, targets)
            if s_ != t_
        ]
        result = compute_auprc(edges, ground_truth_edges, all_genes)
        auprcs.append(result["auprc"])

    return float(np.mean(auprcs))
