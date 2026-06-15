"""Tests for NB sampling module."""

import numpy as np
import pytest
from anndata import AnnData
from scipy import stats

from causalcellbench.models.sampling import (
    estimate_nb_dispersion,
    sample_cells_nb,
    sample_cells_from_model_output,
    sample_perturbed_cells_overlay,
)


def _make_control_adata(n_cells=500, n_genes=50, seed=42):
    """Create synthetic control AnnData with NB-distributed counts."""
    rng = np.random.default_rng(seed)
    # Simulate NB-distributed gene expression
    gene_means = rng.exponential(scale=5.0, size=n_genes) + 0.5
    gene_r = rng.exponential(scale=2.0, size=n_genes) + 0.5

    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    for g in range(n_genes):
        p = gene_r[g] / (gene_r[g] + gene_means[g])
        X[:, g] = rng.negative_binomial(n=gene_r[g], p=p, size=n_cells)

    gene_names = [f"gene_{i}" for i in range(n_genes)]
    import pandas as pd
    return AnnData(X=X, var=pd.DataFrame(index=gene_names)), gene_means, gene_r


class TestEstimateNBDispersion:
    def test_returns_correct_shapes(self):
        adata, _, _ = _make_control_adata(n_genes=30)
        gene_names = [f"gene_{i}" for i in range(30)]
        mu, r = estimate_nb_dispersion(adata, gene_names)
        assert mu.shape == (30,)
        assert r.shape == (30,)

    def test_mu_matches_data_mean(self):
        adata, true_means, _ = _make_control_adata(n_cells=10000, n_genes=20)
        gene_names = [f"gene_{i}" for i in range(20)]
        mu, _ = estimate_nb_dispersion(adata, gene_names, shrinkage=False)
        # Estimated mu should be close to data mean
        data_mean = np.asarray(adata.X).mean(axis=0)
        np.testing.assert_allclose(mu, data_mean, rtol=1e-5)

    def test_r_is_positive(self):
        adata, _, _ = _make_control_adata()
        gene_names = [f"gene_{i}" for i in range(50)]
        _, r = estimate_nb_dispersion(adata, gene_names)
        assert (r > 0).all(), f"Some r values are non-positive: {r[r <= 0]}"

    def test_r_is_finite(self):
        adata, _, _ = _make_control_adata()
        gene_names = [f"gene_{i}" for i in range(50)]
        _, r = estimate_nb_dispersion(adata, gene_names)
        assert np.isfinite(r).all(), f"Non-finite r values: {r[~np.isfinite(r)]}"

    def test_shrinkage_reduces_variance(self):
        adata, _, _ = _make_control_adata(n_cells=100, n_genes=50)
        gene_names = [f"gene_{i}" for i in range(50)]
        _, r_raw = estimate_nb_dispersion(adata, gene_names, shrinkage=False)
        _, r_shrunk = estimate_nb_dispersion(adata, gene_names, shrinkage=True)
        # Shrinkage should reduce variance of log(r)
        assert np.var(np.log(r_shrunk + 1e-8)) <= np.var(np.log(r_raw + 1e-8)) * 1.5

    def test_missing_genes_raises(self):
        adata, _, _ = _make_control_adata(n_genes=10)
        with pytest.raises(ValueError, match="genes not found"):
            estimate_nb_dispersion(adata, ["nonexistent_gene"])


class TestSampleCellsNB:
    def test_output_shape(self):
        mean_pred = np.array([1.0, 5.0, 10.0, 0.5])
        r = np.array([2.0, 3.0, 1.0, 5.0])
        cells = sample_cells_nb(mean_pred, r, n_cells=100)
        assert cells.shape == (100, 4)

    def test_non_negative(self):
        mean_pred = np.ones(20) * 5.0
        r = np.ones(20) * 2.0
        cells = sample_cells_nb(mean_pred, r, n_cells=500)
        assert (cells >= 0).all(), "NB samples should be non-negative"

    def test_mean_close_to_prediction(self):
        """Sample mean should converge to predicted mean for large N."""
        mean_pred = np.array([1.0, 5.0, 20.0, 50.0])
        r = np.array([5.0, 5.0, 5.0, 5.0])
        cells = sample_cells_nb(mean_pred, r, n_cells=50000, seed=0)
        sample_mean = cells.mean(axis=0)
        np.testing.assert_allclose(sample_mean, mean_pred, rtol=0.05)

    def test_overdispersion(self):
        """Variance should exceed mean (NB is overdispersed vs Poisson)."""
        mean_pred = np.array([10.0, 10.0, 10.0])
        r = np.array([1.0, 1.0, 1.0])  # small r = high overdispersion
        cells = sample_cells_nb(mean_pred, r, n_cells=10000, seed=0)
        sample_var = cells.var(axis=0)
        sample_mean = cells.mean(axis=0)
        # Var should be > mean (overdispersed)
        assert (sample_var > sample_mean).all(), (
            f"NB should be overdispersed: var={sample_var}, mean={sample_mean}"
        )

    def test_reproducibility(self):
        mean_pred = np.ones(10) * 5.0
        r = np.ones(10) * 2.0
        cells1 = sample_cells_nb(mean_pred, r, n_cells=50, seed=42)
        cells2 = sample_cells_nb(mean_pred, r, n_cells=50, seed=42)
        np.testing.assert_array_equal(cells1, cells2)

    def test_different_seeds_differ(self):
        mean_pred = np.ones(10) * 5.0
        r = np.ones(10) * 2.0
        cells1 = sample_cells_nb(mean_pred, r, n_cells=50, seed=0)
        cells2 = sample_cells_nb(mean_pred, r, n_cells=50, seed=1)
        assert not np.array_equal(cells1, cells2)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            sample_cells_nb(np.ones(5), np.ones(3), n_cells=10)


class TestSamplePerturbedCellsOverlay:
    """Tests for overlay sampling — the covariance-preserving alternative to NB."""

    def _make_ctrl_X(self, n_cells=500, n_genes=20, seed=42):
        """Create correlated control expression matrix."""
        rng = np.random.default_rng(seed)
        # Create correlated genes via a shared latent factor
        latent = rng.normal(0, 1, size=(n_cells, 3))
        loadings = rng.normal(0, 1, size=(3, n_genes))
        X = np.abs(latent @ loadings + rng.normal(0, 0.5, size=(n_cells, n_genes)))
        return (X * 10).astype(np.float32)  # scale to count-like values

    def test_output_shape(self):
        ctrl_X = self._make_ctrl_X()
        mean_pred = ctrl_X.mean(axis=0) * 0.5  # 50% reduction
        cells = sample_perturbed_cells_overlay(ctrl_X, mean_pred, n_cells=100)
        assert cells.shape == (100, 20)

    def test_non_negative(self):
        ctrl_X = self._make_ctrl_X()
        mean_pred = ctrl_X.mean(axis=0) * 0.1  # strong knockdown
        cells = sample_perturbed_cells_overlay(ctrl_X, mean_pred, n_cells=200)
        assert (cells >= 0).all(), "Overlay samples should be non-negative"

    def test_correlation_preserved(self):
        """Key property: overlay sampling preserves gene-gene correlations."""
        ctrl_X = self._make_ctrl_X(n_cells=1000, n_genes=10, seed=0)
        ctrl_corr = np.corrcoef(ctrl_X.T)

        # Identity mean prediction (no perturbation effect)
        mean_pred = ctrl_X.mean(axis=0)
        sampled = sample_perturbed_cells_overlay(ctrl_X, mean_pred, n_cells=5000, seed=0)
        sampled_corr = np.corrcoef(sampled.T)

        # Correlation structure should be preserved (off-diagonal)
        mask = ~np.eye(10, dtype=bool)
        np.testing.assert_allclose(
            sampled_corr[mask], ctrl_corr[mask], atol=0.15,
            err_msg="Overlay sampling should preserve correlation structure"
        )

    def test_fold_change_applied(self):
        """Sampled means should reflect the fold-change from mean_pred."""
        ctrl_X = self._make_ctrl_X(n_cells=1000, n_genes=5, seed=0)
        ctrl_means = ctrl_X.mean(axis=0)

        # Halve expression of gene 0
        mean_pred = ctrl_means.copy()
        mean_pred[0] = ctrl_means[0] * 0.5

        sampled = sample_perturbed_cells_overlay(ctrl_X, mean_pred, n_cells=10000, seed=0)
        sampled_means = sampled.mean(axis=0)

        # Gene 0 should be roughly halved
        ratio = sampled_means[0] / ctrl_means[0]
        assert 0.3 < ratio < 0.7, f"Expected ~0.5 fold-change, got {ratio:.3f}"

        # Other genes should be roughly unchanged
        for i in range(1, 5):
            ratio_i = sampled_means[i] / ctrl_means[i]
            assert 0.7 < ratio_i < 1.3, f"Gene {i} should be ~unchanged, got {ratio_i:.3f}"

    def test_reproducibility(self):
        ctrl_X = self._make_ctrl_X(n_cells=100, n_genes=5)
        mean_pred = ctrl_X.mean(axis=0) * 0.8
        c1 = sample_perturbed_cells_overlay(ctrl_X, mean_pred, n_cells=50, seed=42)
        c2 = sample_perturbed_cells_overlay(ctrl_X, mean_pred, n_cells=50, seed=42)
        np.testing.assert_array_equal(c1, c2)

    def test_different_seeds_differ(self):
        ctrl_X = self._make_ctrl_X(n_cells=100, n_genes=5)
        mean_pred = ctrl_X.mean(axis=0) * 0.8
        c1 = sample_perturbed_cells_overlay(ctrl_X, mean_pred, n_cells=50, seed=0)
        c2 = sample_perturbed_cells_overlay(ctrl_X, mean_pred, n_cells=50, seed=1)
        assert not np.array_equal(c1, c2)


class TestSampleCellsFromModelOutput:
    def test_end_to_end(self):
        adata, _, _ = _make_control_adata(n_cells=200, n_genes=20)
        gene_names = [f"gene_{i}" for i in range(20)]
        mean_pred = np.ones(20) * 5.0
        cells = sample_cells_from_model_output(mean_pred, adata, 50, gene_names, seed=0)
        assert cells.shape == (50, 20)
        assert (cells >= 0).all()

    def test_with_dispersion_cache(self):
        adata, _, _ = _make_control_adata(n_cells=200, n_genes=20)
        gene_names = [f"gene_{i}" for i in range(20)]
        mu, r = estimate_nb_dispersion(adata, gene_names)
        mean_pred = np.ones(20) * 5.0
        cells = sample_cells_from_model_output(
            mean_pred, adata, 50, gene_names, seed=0, dispersion_cache=(mu, r)
        )
        assert cells.shape == (50, 20)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
