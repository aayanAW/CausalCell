"""Tests for the GEARS model wrapper.

Tests are split into two categories:
1. Unit tests (no GEARS dependency) -- always run
2. Integration tests (require cell-gears) -- skipped if not installed

Run unit tests only:
    pytest tests/test_gears_wrapper.py -v -k "not integration"

Run all (requires env_gears):
    conda run -n env_gears pytest tests/test_gears_wrapper.py -v
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_adata(n_ctrl=200, n_pert_genes=10, n_cells_per_pert=30,
                     n_total_genes=100, seed=42):
    """Create a synthetic perturbation AnnData mimicking CRISPRi data.

    Returns AnnData with:
        obs['gene'] -- perturbation labels ('non-targeting' or gene name)
        var_names   -- gene_0, gene_1, ...
        X           -- NB-like count matrix
    """
    rng = np.random.default_rng(seed)
    gene_names = [f"gene_{i}" for i in range(n_total_genes)]
    pert_genes = gene_names[:n_pert_genes]

    total_cells = n_ctrl + n_pert_genes * n_cells_per_pert

    # Generate NB-like expression
    gene_means = rng.exponential(scale=5.0, size=n_total_genes) + 0.5
    X = np.zeros((total_cells, n_total_genes), dtype=np.float32)
    labels = []

    # Control cells
    for c in range(n_ctrl):
        for g in range(n_total_genes):
            X[c, g] = rng.poisson(gene_means[g])
        labels.append("non-targeting")

    # Perturbed cells (knockdown target gene by ~90%)
    idx = n_ctrl
    for pg in pert_genes:
        g_idx = gene_names.index(pg)
        for c in range(n_cells_per_pert):
            for g in range(n_total_genes):
                X[idx, g] = rng.poisson(gene_means[g])
            X[idx, g_idx] *= 0.1  # knockdown
            labels.append(pg)
            idx += 1

    obs = pd.DataFrame({"gene": labels})
    var = pd.DataFrame(index=gene_names)

    return AnnData(X=X, obs=obs, var=var)


def _make_control_adata(n_cells=200, n_genes=50, seed=42):
    """Create synthetic control-only AnnData."""
    rng = np.random.default_rng(seed)
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    for g in range(n_genes):
        X[:, g] = rng.poisson(rng.exponential(5.0) + 0.5, size=n_cells)
    return AnnData(X=X, var=pd.DataFrame(index=gene_names))


# ---------------------------------------------------------------------------
# Unit tests (no GEARS dependency)
# ---------------------------------------------------------------------------


class TestPrepareAdataForGears:
    """Test the data preparation helper."""

    def test_imports_without_gears(self):
        """The prepare function only needs anndata, not gears."""
        from causalcellbench.models.gears_wrapper import prepare_adata_for_gears
        assert callable(prepare_adata_for_gears)

    def test_condition_column_created(self):
        from causalcellbench.models.gears_wrapper import prepare_adata_for_gears
        adata = _make_mock_adata(n_ctrl=10, n_pert_genes=3, n_cells_per_pert=5)
        result = prepare_adata_for_gears(adata)

        assert "condition" in result.obs.columns
        assert "cell_type" in result.obs.columns
        assert "gene_name" in result.var.columns

    def test_control_labeled_as_ctrl(self):
        from causalcellbench.models.gears_wrapper import prepare_adata_for_gears
        adata = _make_mock_adata(n_ctrl=10, n_pert_genes=3, n_cells_per_pert=5)
        result = prepare_adata_for_gears(adata)

        ctrl_mask = result.obs["condition"] == "ctrl"
        assert ctrl_mask.sum() == 10

    def test_perturbation_format(self):
        """Single perturbations should be formatted as 'GENE+ctrl'."""
        from causalcellbench.models.gears_wrapper import prepare_adata_for_gears
        adata = _make_mock_adata(n_ctrl=5, n_pert_genes=2, n_cells_per_pert=3)
        result = prepare_adata_for_gears(adata)

        non_ctrl = result.obs[result.obs["condition"] != "ctrl"]["condition"].unique()
        for cond in non_ctrl:
            assert cond.endswith("+ctrl"), f"Expected GENE+ctrl format, got '{cond}'"

    def test_cell_type_assigned(self):
        from causalcellbench.models.gears_wrapper import prepare_adata_for_gears
        adata = _make_mock_adata(n_ctrl=5, n_pert_genes=2, n_cells_per_pert=3)
        result = prepare_adata_for_gears(adata, cell_type_label="RPE1")
        assert (result.obs["cell_type"] == "RPE1").all()

    def test_custom_perturbation_key(self):
        from causalcellbench.models.gears_wrapper import prepare_adata_for_gears
        adata = _make_mock_adata()
        adata.obs["my_pert_col"] = adata.obs["gene"]
        del adata.obs["gene"]

        result = prepare_adata_for_gears(adata, perturbation_key="my_pert_col")
        assert "condition" in result.obs.columns

    def test_does_not_modify_original(self):
        from causalcellbench.models.gears_wrapper import prepare_adata_for_gears
        adata = _make_mock_adata(n_ctrl=5, n_pert_genes=2, n_cells_per_pert=3)
        orig_cols = set(adata.obs.columns)
        _ = prepare_adata_for_gears(adata)
        assert set(adata.obs.columns) == orig_cols, "Original adata should not be modified"


class TestGEARSModelInit:
    """Test GEARSModel class without loading a real model."""

    def test_name(self):
        from causalcellbench.models.gears_wrapper import GEARSModel
        model = GEARSModel()
        assert model.name == "GEARS"

    def test_supports_multi_perturbation(self):
        from causalcellbench.models.gears_wrapper import GEARSModel
        model = GEARSModel()
        assert model.supports_multi_perturbation is True

    def test_predict_before_load_raises(self):
        from causalcellbench.models.gears_wrapper import GEARSModel
        model = GEARSModel()
        control = _make_control_adata(n_cells=10, n_genes=5)
        with pytest.raises(RuntimeError, match="load_model"):
            model.predict_perturbation(control, "gene_0", 10, ["gene_0"])

    def test_load_without_args_raises(self):
        from causalcellbench.models.gears_wrapper import GEARSModel
        model = GEARSModel()
        with pytest.raises(ValueError, match="checkpoint_path.*adata_train"):
            model.load_model()


class TestPredictPerturbationFunction:
    """Test the standalone predict_perturbation function with mock model."""

    def test_dict_return(self):
        """predict_perturbation should handle dict return from GEARS."""
        from causalcellbench.models.gears_wrapper import predict_perturbation

        class MockGEARS:
            def predict(self, pert_list):
                gene = pert_list[0][0]
                return {gene: np.ones(100, dtype=np.float32) * 5.0}

        result = predict_perturbation(MockGEARS(), "TP53")
        assert result.shape == (100,)
        assert np.allclose(result, 5.0)

    def test_tuple_return_uncertainty(self):
        """Handle uncertainty mode (tuple return)."""
        from causalcellbench.models.gears_wrapper import predict_perturbation

        class MockGEARS:
            def predict(self, pert_list):
                gene = pert_list[0][0]
                pred = {gene: np.ones(50) * 3.0}
                logvar = {gene: np.zeros(50)}
                return (pred, logvar)

        result = predict_perturbation(MockGEARS(), "BRCA1")
        assert result.shape == (50,)
        assert np.allclose(result, 3.0)

    def test_gene_plus_ctrl_key(self):
        """GEARS may return 'gene+ctrl' as key."""
        from causalcellbench.models.gears_wrapper import predict_perturbation

        class MockGEARS:
            def predict(self, pert_list):
                gene = pert_list[0][0]
                return {f"{gene}+ctrl": np.ones(80) * 7.0}

        result = predict_perturbation(MockGEARS(), "MDM2")
        assert result.shape == (80,)
        assert np.allclose(result, 7.0)


class TestPredictMultiPerturbation:
    """Test combinatorial perturbation prediction."""

    def test_combo_dict_return(self):
        from causalcellbench.models.gears_wrapper import predict_multi_perturbation

        class MockGEARS:
            def predict(self, pert_list):
                key = "+".join(pert_list[0])
                return {key: np.ones(60) * 2.5}

        result = predict_multi_perturbation(MockGEARS(), ["TP53", "MDM2"])
        assert result.shape == (60,)
        assert np.allclose(result, 2.5)

    def test_underscore_key(self):
        from causalcellbench.models.gears_wrapper import predict_multi_perturbation

        class MockGEARS:
            def predict(self, pert_list):
                key = "_".join(pert_list[0])
                return {key: np.ones(40) * 1.0}

        result = predict_multi_perturbation(MockGEARS(), ["GENE_A", "GENE_B"])
        assert result.shape == (40,)


class TestGEARSModelWithMock:
    """Test GEARSModel predict methods with a mock GEARS backend."""

    def _setup_mock_model(self):
        """Create a GEARSModel with mock internals."""
        from causalcellbench.models.gears_wrapper import GEARSModel

        n_genes = 20
        gene_names = [f"gene_{i}" for i in range(n_genes)]

        class MockGEARS:
            def predict(self, pert_list):
                gene = pert_list[0][0]
                # Return mean expression = 5.0 for all genes
                return {gene: np.ones(n_genes, dtype=np.float32) * 5.0}

        model = GEARSModel()
        model._model = MockGEARS()
        model._gene_list = gene_names
        return model, gene_names

    def test_predict_mean(self):
        model, gene_names = self._setup_mock_model()
        result = model.predict_perturbation_mean("gene_0")
        assert result.shape == (20,)

    def test_predict_perturbation_with_sampling(self):
        model, gene_names = self._setup_mock_model()
        control = _make_control_adata(n_cells=100, n_genes=20)

        result = model.predict_perturbation(
            control_adata=control,
            perturbed_gene="gene_0",
            n_cells=50,
            gene_names=gene_names,
            seed=42,
        )

        assert result.expression_matrix.shape == (50, 20)
        assert result.gene_names == gene_names
        assert result.perturbed_genes == ["gene_0"]
        assert (result.expression_matrix >= 0).all()

    def test_gene_alignment(self):
        """Test that GEARS output is aligned to requested gene_names."""
        model, full_genes = self._setup_mock_model()
        control = _make_control_adata(n_cells=100, n_genes=20)

        # Request only a subset of genes
        subset = ["gene_5", "gene_10", "gene_15"]
        result = model.predict_perturbation(
            control_adata=control,
            perturbed_gene="gene_0",
            n_cells=30,
            gene_names=subset,
            seed=0,
        )

        assert result.expression_matrix.shape == (30, 3)
        assert result.gene_names == subset

    def test_dispersion_cache_reused(self):
        """Second call should reuse cached dispersion."""
        model, gene_names = self._setup_mock_model()
        control = _make_control_adata(n_cells=100, n_genes=20)

        _ = model.predict_perturbation(control, "gene_0", 10, gene_names, seed=0)
        assert model._dispersion_cache is not None

        cache_before = model._dispersion_cache
        _ = model.predict_perturbation(control, "gene_1", 10, gene_names, seed=1)
        assert model._dispersion_cache is cache_before


# ---------------------------------------------------------------------------
# Integration tests (require cell-gears installed)
# ---------------------------------------------------------------------------

try:
    from gears import PertData, GEARS
    GEARS_AVAILABLE = True
except ImportError:
    GEARS_AVAILABLE = False


@pytest.mark.skipif(not GEARS_AVAILABLE, reason="cell-gears not installed")
class TestGEARSIntegration:
    """Integration tests that use the real GEARS package.

    These tests download data on first run (GO annotations from Harvard Dataverse).
    Run in env_gears conda environment.
    """

    def test_pertdata_init(self, tmp_path):
        """PertData can be initialized with a temp directory."""
        pert_data = PertData(str(tmp_path))
        assert pert_data is not None

    def test_prepare_and_process(self, tmp_path):
        """Full pipeline: prepare AnnData -> PertData.new_data_process."""
        from causalcellbench.models.gears_wrapper import prepare_adata_for_gears

        adata = _make_mock_adata(
            n_ctrl=50, n_pert_genes=5, n_cells_per_pert=20, n_total_genes=50
        )
        gears_adata = prepare_adata_for_gears(adata)

        pert_data = PertData(str(tmp_path))
        pert_data.new_data_process(
            dataset_name="test_data",
            adata=gears_adata,
        )

        # Check processed data exists
        processed_path = tmp_path / "test_data" / "perturb_processed.h5ad"
        assert processed_path.exists(), f"Expected {processed_path} to exist"

    @pytest.mark.slow
    def test_train_small(self, tmp_path):
        """Train GEARS on tiny synthetic data (smoke test)."""
        from causalcellbench.models.gears_wrapper import train_gears

        adata = _make_mock_adata(
            n_ctrl=50, n_pert_genes=5, n_cells_per_pert=20, n_total_genes=50
        )

        model, pert_data = train_gears(
            adata=adata,
            data_path=str(tmp_path),
            dataset_name="test_train",
            epochs=2,
            hidden_size=16,
            batch_size=8,
            device="cpu",
        )

        assert model is not None
        assert pert_data is not None

    @pytest.mark.slow
    def test_train_and_predict(self, tmp_path):
        """Train GEARS and make predictions."""
        from causalcellbench.models.gears_wrapper import (
            train_gears, predict_perturbation
        )

        adata = _make_mock_adata(
            n_ctrl=50, n_pert_genes=5, n_cells_per_pert=20, n_total_genes=50
        )

        model, pert_data = train_gears(
            adata=adata,
            data_path=str(tmp_path),
            dataset_name="test_predict",
            epochs=2,
            hidden_size=16,
            batch_size=8,
            device="cpu",
        )

        # Predict for a gene that was in the training data
        result = predict_perturbation(model, "gene_0")
        assert result.ndim == 1
        assert len(result) == len(pert_data.adata.var_names)
        assert np.isfinite(result).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
