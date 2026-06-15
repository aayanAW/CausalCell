"""Tests for evaluation metrics."""

import numpy as np
import pytest

from causalcellbench.eval.auprc import compute_auprc, compute_random_auprc
from causalcellbench.eval.precision_at_k import precision_at_k
from causalcellbench.eval.shd import structural_hamming_distance
from causalcellbench.eval.wasserstein import mean_wasserstein_distance


class TestAUPRC:
    def test_perfect_predictions(self):
        gt = {("A", "B"), ("C", "D"), ("E", "F")}
        pred = [("A", "B"), ("C", "D"), ("E", "F")]
        result = compute_auprc(pred, gt, ["A", "B", "C", "D", "E", "F"])
        assert result["auprc"] == 1.0
        assert result["n_true_positive"] == 3

    def test_no_predictions(self):
        gt = {("A", "B")}
        result = compute_auprc([], gt, ["A", "B"])
        assert result["auprc"] == 0.0
        assert result["n_predicted"] == 0

    def test_all_wrong(self):
        gt = {("A", "B")}
        pred = [("C", "D"), ("E", "F")]
        result = compute_auprc(pred, gt, ["A", "B", "C", "D", "E", "F"])
        assert result["auprc"] == 0.0
        assert result["n_true_positive"] == 0

    def test_precision_at_k_values(self):
        gt = {("A", "B"), ("C", "D")}
        pred = [("A", "B"), ("X", "Y"), ("C", "D"), ("Z", "W")]
        result = compute_auprc(pred, gt, ["A", "B", "C", "D", "X", "Y", "Z", "W"])
        assert result["precision_at_100"] == 0.5  # 2 correct out of 4

    def test_random_auprc_near_chance(self):
        genes = [f"G{i}" for i in range(50)]
        gt = {(f"G{i}", f"G{j}") for i in range(5) for j in range(5, 10)}
        random_auprc = compute_random_auprc(gt, genes, n_samples=20, seed=0)
        # Random should be near |GT| / |possible edges|
        expected_approx = len(gt) / (50 * 49)
        assert random_auprc < 0.2  # should be quite low


class TestPrecisionAtK:
    def test_perfect_top_k(self):
        gt = {("A", "B"), ("C", "D")}
        pred = [("A", "B"), ("C", "D"), ("X", "Y")]
        assert precision_at_k(pred, gt, 2) == 1.0

    def test_zero_precision(self):
        gt = {("A", "B")}
        pred = [("X", "Y"), ("Z", "W")]
        assert precision_at_k(pred, gt, 2) == 0.0

    def test_k_larger_than_predictions(self):
        gt = {("A", "B")}
        pred = [("A", "B")]
        assert precision_at_k(pred, gt, 100) == 1.0


class TestSHD:
    def test_identical_graphs(self):
        edges = {("A", "B"), ("C", "D")}
        result = structural_hamming_distance(edges, edges)
        assert result["shd"] == 0

    def test_extra_edges(self):
        pred = {("A", "B"), ("C", "D"), ("E", "F")}
        gt = {("A", "B")}
        result = structural_hamming_distance(pred, gt)
        assert result["extra"] == 2

    def test_missing_edges(self):
        pred = {("A", "B")}
        gt = {("A", "B"), ("C", "D"), ("E", "F")}
        result = structural_hamming_distance(pred, gt)
        assert result["missing"] == 2

    def test_reversed_edges(self):
        pred = {("B", "A")}
        gt = {("A", "B")}
        result = structural_hamming_distance(pred, gt)
        assert result["reversed"] == 1


class TestWasserstein:
    def test_with_real_shift(self):
        """Perturbed cells with shifted expression should have high Wasserstein."""
        n_ctrl = 100
        n_pert = 50
        n_genes = 5
        rng = np.random.default_rng(0)

        ctrl_expr = rng.normal(5.0, 1.0, size=(n_ctrl, n_genes))
        pert_expr = rng.normal(10.0, 1.0, size=(n_pert, n_genes))  # shifted by 5

        expr = np.vstack([ctrl_expr, pert_expr]).astype(np.float32)
        interventions = ["non-targeting"] * n_ctrl + ["G0"] * n_pert
        genes = [f"G{i}" for i in range(n_genes)]

        edges = [("G0", f"G{i}") for i in range(n_genes)]
        result = mean_wasserstein_distance(edges, expr, interventions, genes)
        assert result["mean_wasserstein"] > 3.0  # large shift expected
        assert result["n_evaluated"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
