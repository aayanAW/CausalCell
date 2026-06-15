"""False Omission Rate (FOR) — CausalBench's primary biological metric.

FOR measures the rate at which existing causal interactions are missed.
Computed via Mann-Whitney U tests on hold-out test set: tests whether
perturbing gene A significantly affects gene B's expression distribution.

Multiple testing correction (Benjamini-Hochberg) is applied to control
the false discovery rate across all tested pairs.

Reference: Chevalley et al., Communications Biology 2025.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def false_omission_rate(
    predicted_edges: set[tuple[str, str]],
    expression_matrix: np.ndarray,
    interventions: list[str],
    gene_names: list[str],
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute False Omission Rate for predicted edge set.

    For each gene pair (A, B) NOT in predicted edges:
        1. Test if perturbing A significantly shifts B's distribution
           (Mann-Whitney U, two-sided)
        2. Apply Benjamini-Hochberg correction across all tests
        3. If significant after correction: false omission (missed true edge)

    FOR = n_false_omissions / n_non_predicted_pairs_tested

    Parameters
    ----------
    predicted_edges
        Set of (source, target) predicted edges.
    expression_matrix
        Shape (n_cells, n_genes).
    interventions
        Per-cell intervention labels.
    gene_names
        Gene names matching columns.
    alpha
        Significance threshold after BH correction.

    Returns
    -------
    dict with ``for_rate``, ``n_false_omissions``, ``n_tested``,
    ``n_significant_before_correction``.
    """
    # Convert to set for O(1) lookup
    predicted_set = set(predicted_edges)

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    ctrl_idx = np.array(
        [i for i, v in enumerate(interventions) if v == "non-targeting"]
    )
    pert_groups: dict[str, np.ndarray] = {}
    for i, v in enumerate(interventions):
        if v != "non-targeting":
            pert_groups.setdefault(v, []).append(i)
    pert_groups = {k: np.array(v) for k, v in pert_groups.items()}

    # Collect all p-values first (for BH correction)
    p_values = []
    pair_keys = []

    perturbed_genes = sorted(pert_groups.keys())
    for src in perturbed_genes:
        if src not in gene_to_idx:
            continue
        src_cells = pert_groups[src]
        if len(src_cells) < 5:
            continue

        for tgt in gene_names:
            if tgt == src:
                continue
            if (src, tgt) in predicted_set:
                continue

            tgt_idx = gene_to_idx[tgt]
            pert_vals = expression_matrix[src_cells, tgt_idx]
            ctrl_vals = expression_matrix[ctrl_idx, tgt_idx]

            if len(ctrl_vals) < 5:
                continue

            try:
                _, p_value = mannwhitneyu(
                    pert_vals, ctrl_vals, alternative="two-sided"
                )
            except ValueError:
                continue

            p_values.append(p_value)
            pair_keys.append((src, tgt))

    n_tested = len(p_values)
    if n_tested == 0:
        return {
            "for_rate": 0.0,
            "n_false_omissions": 0,
            "n_tested": 0,
            "n_significant_before_correction": 0,
        }

    # Benjamini-Hochberg correction
    p_arr = np.array(p_values)
    n_significant_raw = int((p_arr < alpha).sum())
    rejected = _benjamini_hochberg(p_arr, alpha)
    n_false_omissions = int(rejected.sum())

    for_rate = n_false_omissions / n_tested
    return {
        "for_rate": for_rate,
        "n_false_omissions": n_false_omissions,
        "n_tested": n_tested,
        "n_significant_before_correction": n_significant_raw,
    }


def _benjamini_hochberg(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg procedure for FDR control.

    Parameters
    ----------
    p_values
        Array of p-values.
    alpha
        Target FDR level.

    Returns
    -------
    Boolean array: True where null hypothesis is rejected.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH thresholds: alpha * rank / n
    thresholds = alpha * np.arange(1, n + 1) / n

    # Find largest k where p_(k) <= threshold_(k)
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool)

    max_k = np.max(np.where(below)[0])
    rejected = np.zeros(n, dtype=bool)
    rejected[sorted_idx[:max_k + 1]] = True
    return rejected
