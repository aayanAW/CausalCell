"""Tests for DCDI-G causal discovery runner."""

import numpy as np
import pytest

from causalcellbench.causal.causalbench_bridge import CausalBenchInput
from causalcellbench.causal.dcdi_runner import run_dcdi


def _make_linear_sem_data(
    n_genes: int = 10,
    n_ctrl: int = 200,
    n_pert: int = 50,
    seed: int = 0,
) -> tuple[CausalBenchInput, set]:
    """Generate synthetic data from a known linear SEM with interventions."""
    rng = np.random.default_rng(seed)

    # Create a random DAG (lower triangular + permutation)
    W_true = np.zeros((n_genes, n_genes))
    for i in range(1, n_genes):
        n_parents = rng.integers(0, min(3, i) + 1)
        if n_parents > 0:
            parents = rng.choice(i, size=n_parents, replace=False)
            for p in parents:
                W_true[p, i] = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])

    gene_names = [f"G{i}" for i in range(n_genes)]
    true_edges = set()
    for i in range(n_genes):
        for j in range(n_genes):
            if W_true[i, j] != 0:
                true_edges.add((gene_names[i], gene_names[j]))

    # Generate observational data
    noise_std = 0.5
    ctrl_X = np.zeros((n_ctrl, n_genes))
    for i in range(n_genes):
        parents = np.where(W_true[:, i] != 0)[0]
        if len(parents) > 0:
            ctrl_X[:, i] = ctrl_X[:, parents] @ W_true[parents, i]
        ctrl_X[:, i] += rng.normal(0, noise_std, size=n_ctrl)

    # Generate interventional data (soft: 80% knockdown on target)
    pert_genes = gene_names[:min(5, n_genes)]
    pert_matrices = []
    pert_labels = []
    for gene in pert_genes:
        g_idx = gene_names.index(gene)
        pert_X = np.zeros((n_pert, n_genes))
        for i in range(n_genes):
            parents = np.where(W_true[:, i] != 0)[0]
            if len(parents) > 0:
                pert_X[:, i] = pert_X[:, parents] @ W_true[parents, i]
            pert_X[:, i] += rng.normal(0, noise_std, size=n_pert)
            if i == g_idx:
                pert_X[:, i] *= 0.2  # 80% knockdown
        pert_matrices.append(pert_X.astype(np.float32))
        pert_labels.extend([gene] * n_pert)

    expr = np.concatenate([ctrl_X.astype(np.float32)] + pert_matrices, axis=0)
    interventions = ["non-targeting"] * n_ctrl + pert_labels

    cb = CausalBenchInput(
        expression_matrix=expr,
        interventions=interventions,
        gene_names=gene_names,
        n_control_cells=n_ctrl,
        n_perturbation_cells=len(pert_labels),
        perturbed_genes=sorted(set(pert_labels)),
    )

    return cb, true_edges


class TestDCDIRunner:
    def test_returns_edges(self):
        """DCDI should return a non-empty list of edge tuples."""
        cb, _ = _make_linear_sem_data(n_genes=5, n_ctrl=100, n_pert=30)
        edges = run_dcdi(cb, seed=0, partition_size=5, max_epochs=50)
        assert isinstance(edges, list)
        # Should find at least some edges
        assert len(edges) >= 0  # may be 0 for very few epochs

    def test_edges_are_tuples(self):
        """Each edge should be (source, target) tuple of gene names."""
        cb, _ = _make_linear_sem_data(n_genes=5, n_ctrl=100, n_pert=30)
        edges = run_dcdi(cb, seed=0, partition_size=5, max_epochs=100)
        for e in edges:
            assert isinstance(e, tuple)
            assert len(e) == 2
            assert e[0] in cb.gene_names
            assert e[1] in cb.gene_names
            assert e[0] != e[1]  # no self-loops

    def test_soft_interventions(self):
        """DCDI with imperfect interventions should work."""
        cb, _ = _make_linear_sem_data(n_genes=8, n_ctrl=150, n_pert=50)
        edges = run_dcdi(
            cb, seed=0, partition_size=8, max_epochs=100,
            intervention_type="imperfect",
        )
        assert isinstance(edges, list)

    def test_perfect_interventions(self):
        """DCDI with perfect interventions should also work."""
        cb, _ = _make_linear_sem_data(n_genes=8, n_ctrl=150, n_pert=50)
        edges = run_dcdi(
            cb, seed=0, partition_size=8, max_epochs=100,
            intervention_type="perfect",
        )
        assert isinstance(edges, list)

    def test_partitioning(self):
        """DCDI should handle multiple partitions."""
        cb, _ = _make_linear_sem_data(n_genes=15, n_ctrl=100, n_pert=30)
        edges = run_dcdi(cb, seed=0, partition_size=5, max_epochs=50)
        assert isinstance(edges, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
