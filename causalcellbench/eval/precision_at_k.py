"""Precision@K metric for causal graph evaluation."""

from __future__ import annotations

import numpy as np


def precision_at_k(
    predicted_edges: list[tuple[str, str]],
    ground_truth_edges: set[tuple[str, str]],
    k: int,
) -> float:
    """Precision at top K predicted edges.

    Parameters
    ----------
    predicted_edges
        Ranked list of (source, target) edges (highest confidence first).
    ground_truth_edges
        Set of known true edges.
    k
        Number of top predictions to evaluate.

    Returns
    -------
    Fraction of top-K predictions that are in ground truth.
    """
    if not predicted_edges or k <= 0:
        return 0.0
    top_k = predicted_edges[:k]
    n_correct = sum(1 for e in top_k if e in ground_truth_edges)
    return n_correct / min(k, len(top_k))
