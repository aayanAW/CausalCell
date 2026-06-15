"""Structural Hamming Distance (SHD) — secondary metric.

SHD counts edge additions, deletions, and reversals needed to transform
the predicted graph into the ground truth. It is SECONDARY because it
penalizes correct predictions that are missing from incomplete databases
(TRRUST, ChIP-seq).

Use AUPRC as primary metric; SHD as supplementary.
"""

from __future__ import annotations


def structural_hamming_distance(
    predicted_edges: set[tuple[str, str]],
    ground_truth_edges: set[tuple[str, str]],
) -> dict[str, int]:
    """Compute SHD between predicted and ground truth edge sets.

    Parameters
    ----------
    predicted_edges
        Set of (source, target) predicted edges.
    ground_truth_edges
        Set of (source, target) ground truth edges.

    Returns
    -------
    dict with keys:
        - ``shd``: Total SHD
        - ``missing``: Edges in GT but not in predicted (false negatives)
        - ``extra``: Edges in predicted but not in GT (false positives)
        - ``reversed``: Edges present but with wrong direction
    """
    pred_set = set(predicted_edges)
    gt_set = set(ground_truth_edges)

    # Reversed: (a,b) in predicted and (b,a) in GT
    reversed_edges = set()
    for src, tgt in pred_set:
        if (tgt, src) in gt_set and (src, tgt) not in gt_set:
            reversed_edges.add((src, tgt))

    # Missing: in GT but not predicted (in either direction)
    missing = gt_set - pred_set - {(t, s) for s, t in reversed_edges}

    # Extra: in predicted but not GT (and not a reversal of a GT edge)
    extra = pred_set - gt_set - reversed_edges

    shd = len(missing) + len(extra) + len(reversed_edges)

    return {
        "shd": shd,
        "missing": len(missing),
        "extra": len(extra),
        "reversed": len(reversed_edges),
    }
