"""GIES (Greedy Interventional Equivalence Search) runner.

Thin wrapper around the ``gies`` package that converts CausalBenchInput
into the format expected by ``gies.fit_bic``.

GIES assumptions (known limitations):
    - Acyclic graphs only (DAGs) — GRNs can be cyclic
    - Hard interventions — CRISPRi is soft (~80-90% knockdown)
    - Known intervention targets

Despite these mismatches, CausalBench uses GIES on real CRISPRi data
as a primary algorithm, so results are directly comparable to their baselines.

Reference: Hauser & Buhlmann, JMLR 2012.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from causalcellbench.causal.causalbench_bridge import CausalBenchInput


def run_gies(
    cb_input: CausalBenchInput,
    seed: int = 0,
    partition_size: int = 30,
) -> list[tuple[str, str]]:
    """Run GIES on CausalBench-formatted input.

    Uses CausalBench's partitioning strategy: splits genes into partitions
    of ``partition_size`` and runs GIES on each partition independently.

    Parameters
    ----------
    cb_input
        Formatted input from ``build_causalbench_input()``.
    seed
        Random seed.
    partition_size
        Number of genes per partition. CausalBench uses 30 for GIES.

    Returns
    -------
    list of (source_gene, target_gene) edge tuples.
    """
    import gies

    gene_names = cb_input.gene_names
    n_genes = len(gene_names)

    # Build per-gene intervention index
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Group cells by intervention regime
    # GIES expects: list of np.arrays, one per regime
    # I = list of lists: intervention targets per regime
    regimes: dict[str, list[int]] = {}  # intervention_label -> cell indices
    for cell_idx, label in enumerate(cb_input.interventions):
        regimes.setdefault(label, []).append(cell_idx)

    all_edges: list[tuple[str, str]] = []

    # Partition genes for scalability
    rng = np.random.default_rng(seed)
    gene_order = rng.permutation(n_genes)
    partitions = [
        gene_order[i : i + partition_size]
        for i in range(0, n_genes, partition_size)
    ]

    for partition_idx in partitions:
        partition_genes = [gene_names[i] for i in partition_idx]
        partition_gene_set = set(partition_genes)
        n_part = len(partition_idx)

        # Build data matrices and intervention targets per regime
        data_list = []
        intervention_targets_list = []

        for label, cell_indices in regimes.items():
            X_regime = cb_input.expression_matrix[cell_indices][:, partition_idx]
            data_list.append(X_regime.astype(np.float64))

            if label == "non-targeting":
                intervention_targets_list.append([])
            elif label in partition_gene_set:
                # This gene is in our partition — mark as intervened
                local_idx = list(partition_genes).index(label)
                intervention_targets_list.append([local_idx])
            else:
                # Gene not in this partition — treat as observational
                intervention_targets_list.append([])

        # Run GIES
        try:
            adjacency, score = gies.fit_bic(
                data_list, intervention_targets_list
            )
        except Exception as e:
            # Skip partition on failure (e.g., singular matrix)
            continue

        # Convert adjacency to edges.
        # GIES returns a CPDAG: adjacency[i,j]=1 AND adjacency[j,i]=1
        # means an undirected edge. We emit only the (i,j) direction
        # where i<j to avoid double-counting undirected edges.
        for i in range(n_part):
            for j in range(n_part):
                if i == j:
                    continue
                if adjacency[i, j] != 0:
                    if adjacency[j, i] != 0:
                        # Undirected edge in CPDAG — emit only once (i<j)
                        if i < j:
                            src = partition_genes[i]
                            tgt = partition_genes[j]
                            all_edges.append((src, tgt))
                    else:
                        # Directed edge i -> j
                        src = partition_genes[i]
                        tgt = partition_genes[j]
                        all_edges.append((src, tgt))

    return all_edges


def run_gies_full(
    expression_matrix: np.ndarray,
    interventions: list[str],
    gene_names: list[str],
    seed: int = 0,
    partition_size: int = 30,
) -> list[tuple[str, str]]:
    """Convenience wrapper taking raw arrays instead of CausalBenchInput."""
    cb_input = CausalBenchInput(
        expression_matrix=expression_matrix,
        interventions=interventions,
        gene_names=gene_names,
        n_control_cells=sum(1 for i in interventions if i == "non-targeting"),
        n_perturbation_cells=sum(1 for i in interventions if i != "non-targeting"),
        perturbed_genes=sorted(set(i for i in interventions if i != "non-targeting")),
    )
    return run_gies(cb_input, seed=seed, partition_size=partition_size)
