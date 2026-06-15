"""Tests for CPA wrapper.

Tests are structured in tiers:
1. Unit tests that work WITHOUT cpa-tools installed (data prep, gene mapping)
2. Integration tests that REQUIRE cpa-tools (marked with pytest.mark.skipif)
3. A smoke test on tiny synthetic data that validates the full pipeline

Run:
    pytest tests/test_cpa_wrapper.py -v
    pytest tests/test_cpa_wrapper.py -v -k "not requires_cpa"  # skip CPA tests
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

# ---------------------------------------------------------------------------
# Check if cpa-tools is available
# ---------------------------------------------------------------------------

try:
    import cpa

    CPA_AVAILABLE = True
except ImportError:
    CPA_AVAILABLE = False

requires_cpa = pytest.mark.skipif(
    not CPA_AVAILABLE, reason="cpa-tools not installed"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_perturbation_adata(
    n_control: int = 200,
    n_perts: int = 5,
    n_cells_per_pert: int = 40,
    n_genes: int = 100,
    seed: int = 42,
) -> AnnData:
    """Create synthetic Perturb-seq-like AnnData.

    Returns an AnnData with:
    - NB-like count data in .X
    - 'gene' column in .obs with perturbation names
    - 'non-targeting' as control label
    - gene symbols as var_names
    """
    rng = np.random.default_rng(seed)

    total_cells = n_control + n_perts * n_cells_per_pert

    # Simulate NB-distributed counts
    gene_means = rng.exponential(scale=5.0, size=n_genes) + 0.5
    gene_r = rng.exponential(scale=2.0, size=n_genes) + 1.0

    X = np.zeros((total_cells, n_genes), dtype=np.float32)
    for g in range(n_genes):
        p = gene_r[g] / (gene_r[g] + gene_means[g])
        X[:, g] = rng.negative_binomial(n=max(1, int(gene_r[g])), p=p, size=total_cells)

    # Create perturbation labels
    pert_names = [f"GENE{i}" for i in range(n_perts)]
    labels = (
        ["non-targeting"] * n_control
        + [p for p in pert_names for _ in range(n_cells_per_pert)]
    )

    # Introduce perturbation effects: knock down the target gene for perturbed cells
    for pi, pname in enumerate(pert_names):
        start = n_control + pi * n_cells_per_pert
        end = start + n_cells_per_pert
        target_gene_idx = pi % n_genes
        X[start:end, target_gene_idx] *= 0.1  # CRISPRi knockdown
        # Also add downstream effects on a few other genes
        for offset in [1, 2, 3]:
            neighbor = (target_gene_idx + offset) % n_genes
            X[start:end, neighbor] *= rng.uniform(0.5, 1.5)

    obs = pd.DataFrame({
        "gene": labels,
        "cell_type": ["K562"] * total_cells,
    })

    gene_names = [f"TP53" if i == 0 else f"GeneSymbol{i}" for i in range(n_genes)]
    var = pd.DataFrame(index=gene_names)

    return AnnData(X=X, obs=obs, var=var)


def _make_ensembl_adata(n_genes: int = 50, seed: int = 42) -> AnnData:
    """Create AnnData with Ensembl IDs as var_names and gene_name column."""
    rng = np.random.default_rng(seed)
    n_cells = 100
    X = rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32)

    ensembl_ids = [f"ENSG{i:011d}" for i in range(n_genes)]
    symbols = [f"Gene{i}" for i in range(n_genes)]

    obs = pd.DataFrame({"gene": ["non-targeting"] * n_cells})
    var = pd.DataFrame({"gene_name": symbols}, index=ensembl_ids)

    return AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# Unit tests (no CPA required)
# ---------------------------------------------------------------------------


class TestMapEnsemblToSymbols:
    """Test Ensembl ID to gene symbol mapping."""

    def test_converts_ensembl_to_symbols(self):
        from causalcellbench.models.cpa_wrapper import map_ensembl_to_symbols

        adata = _make_ensembl_adata()
        assert adata.var_names[0].startswith("ENSG")

        result = map_ensembl_to_symbols(adata)
        assert result.var_names[0] == "Gene0"
        assert "ensembl_id" in result.var.columns

    def test_preserves_symbol_var_names(self):
        from causalcellbench.models.cpa_wrapper import map_ensembl_to_symbols

        adata = _make_perturbation_adata(n_control=10, n_perts=0, n_genes=10)
        original_names = list(adata.var_names)

        result = map_ensembl_to_symbols(adata)
        assert list(result.var_names) == original_names

    def test_handles_duplicate_symbols(self):
        from causalcellbench.models.cpa_wrapper import map_ensembl_to_symbols

        adata = _make_ensembl_adata(n_genes=5)
        # Create duplicate gene names
        adata.var["gene_name"] = ["ACTB", "ACTB", "TP53", "TP53", "MYC"]

        result = map_ensembl_to_symbols(adata)
        # Should have unique var_names
        assert len(set(result.var_names)) == len(result.var_names)

    def test_no_gene_name_column(self):
        from causalcellbench.models.cpa_wrapper import map_ensembl_to_symbols

        adata = _make_ensembl_adata()
        del adata.var["gene_name"]

        result = map_ensembl_to_symbols(adata)
        # Should return unchanged (still Ensembl IDs)
        assert result.var_names[0].startswith("ENSG")


class TestCheckCpaAvailable:
    """Test the lazy import check."""

    def test_returns_bool(self):
        from causalcellbench.models.cpa_wrapper import _check_cpa_available

        result = _check_cpa_available()
        assert isinstance(result, bool)
        assert result == CPA_AVAILABLE


class TestCPAModelInit:
    """Test CPAModel class instantiation (no CPA required)."""

    def test_init_defaults(self):
        from causalcellbench.models.cpa_wrapper import CPAModel

        wrapper = CPAModel()
        assert wrapper.name == "CPA"
        assert wrapper.supports_multi_perturbation is True
        assert wrapper.model is None
        assert wrapper.perturbation_key == "gene"
        assert wrapper.control_label == "non-targeting"

    def test_predict_before_load_raises(self):
        from causalcellbench.models.cpa_wrapper import CPAModel

        wrapper = CPAModel()
        adata = _make_perturbation_adata(n_control=10, n_perts=1, n_genes=10)

        with pytest.raises(RuntimeError, match="No model loaded"):
            wrapper.predict_perturbation(adata, "GENE0", n_cells=5)


# ---------------------------------------------------------------------------
# Integration tests (require cpa-tools)
# ---------------------------------------------------------------------------


@requires_cpa
class TestPrepareAdataForCPA:
    """Test data preparation for CPA."""

    def test_basic_setup(self):
        from causalcellbench.models.cpa_wrapper import prepare_adata_for_cpa

        adata = _make_perturbation_adata()
        result = prepare_adata_for_cpa(
            adata,
            perturbation_key="gene",
            control_label="non-targeting",
        )
        assert "cpa_dosage" in result.obs.columns
        assert result.n_obs == adata.n_obs

    def test_creates_dosage_column(self):
        from causalcellbench.models.cpa_wrapper import prepare_adata_for_cpa

        adata = _make_perturbation_adata()
        result = prepare_adata_for_cpa(adata, perturbation_key="gene")

        # Control cells should have dosage 0, perturbed should have 1
        ctrl_mask = result.obs["gene"] == "non-targeting"
        assert (result.obs.loc[ctrl_mask, "cpa_dosage"] == 0.0).all()
        assert (result.obs.loc[~ctrl_mask, "cpa_dosage"] == 1.0).all()

    def test_invalid_perturbation_key_raises(self):
        from causalcellbench.models.cpa_wrapper import prepare_adata_for_cpa

        adata = _make_perturbation_adata()
        with pytest.raises(ValueError, match="not found in adata.obs"):
            prepare_adata_for_cpa(adata, perturbation_key="nonexistent")

    def test_invalid_control_label_raises(self):
        from causalcellbench.models.cpa_wrapper import prepare_adata_for_cpa

        adata = _make_perturbation_adata()
        with pytest.raises(ValueError, match="not found"):
            prepare_adata_for_cpa(
                adata, perturbation_key="gene", control_label="WRONG_LABEL"
            )

    def test_copy_flag(self):
        from causalcellbench.models.cpa_wrapper import prepare_adata_for_cpa

        adata = _make_perturbation_adata()
        result = prepare_adata_for_cpa(adata, copy=True)
        # Original should not have the dosage column
        assert "cpa_dosage" not in adata.obs.columns
        assert "cpa_dosage" in result.obs.columns


@requires_cpa
class TestCPATrainSmoke:
    """Smoke test: train CPA on tiny synthetic data.

    This validates the full pipeline end-to-end with minimal data.
    Not a real training run -- just checks that nothing crashes.
    """

    def test_train_and_predict_smoke(self):
        """Full pipeline: train CPA for 3 epochs, predict one perturbation."""
        from causalcellbench.models.cpa_wrapper import train_cpa, predict_perturbation

        adata = _make_perturbation_adata(
            n_control=100,
            n_perts=3,
            n_cells_per_pert=20,
            n_genes=50,
        )

        # Train for a tiny number of epochs
        model = train_cpa(
            adata,
            perturbation_key="gene",
            control_label="non-targeting",
            n_latent=16,
            recon_loss="gauss",  # Gauss is faster and works with any data
            variational=False,  # Faster for smoke test
            max_epochs=3,
            batch_size=64,
            early_stopping_patience=100,  # Don't early stop in smoke test
            check_val_every_n_epoch=1,
            lr=1e-3,
            seed=42,
        )

        assert model is not None

        # Predict a perturbation
        expr = predict_perturbation(
            model=model,
            adata=adata,
            target_perturbation="GENE0",
            perturbation_key="gene",
            control_label="non-targeting",
            n_samples=1,  # Minimal for speed
            batch_size=64,
        )

        assert isinstance(expr, np.ndarray)
        assert expr.ndim == 2
        # Should have same number of genes as training data
        assert expr.shape[1] == adata.n_vars
        # Should have non-negative expression values
        # (reconstruction loss is Gaussian so values can be negative,
        # but they should be mostly reasonable)
        assert np.isfinite(expr).all()

    def test_cpa_model_wrapper_smoke(self):
        """Test the CPAModel wrapper class end-to-end."""
        from causalcellbench.models.cpa_wrapper import CPAModel

        adata = _make_perturbation_adata(
            n_control=80,
            n_perts=2,
            n_cells_per_pert=20,
            n_genes=30,
        )

        wrapper = CPAModel(
            perturbation_key="gene",
            control_label="non-targeting",
        )

        # Train
        wrapper.train(
            adata,
            n_latent=8,
            recon_loss="gauss",
            variational=False,
            max_epochs=2,
            batch_size=32,
            lr=1e-3,
        )

        assert wrapper.model is not None

        # Predict single perturbation
        expr = wrapper.predict_perturbation(
            control_adata=adata,
            perturbed_gene="GENE0",
            n_cells=10,
            n_samples=1,
        )

        assert expr.shape == (10, 30)
        assert np.isfinite(expr).all()

    def test_predict_batch_smoke(self):
        """Test batch prediction for multiple perturbations."""
        from causalcellbench.models.cpa_wrapper import (
            train_cpa,
            predict_perturbation_batch,
        )

        adata = _make_perturbation_adata(
            n_control=80,
            n_perts=3,
            n_cells_per_pert=20,
            n_genes=30,
        )

        model = train_cpa(
            adata,
            perturbation_key="gene",
            control_label="non-targeting",
            n_latent=8,
            recon_loss="gauss",
            variational=False,
            max_epochs=2,
            batch_size=32,
            lr=1e-3,
        )

        results = predict_perturbation_batch(
            model=model,
            adata=adata,
            target_perturbations=["GENE0", "GENE1"],
            perturbation_key="gene",
            control_label="non-targeting",
            n_cells=10,
            n_samples=1,
            batch_size=32,
        )

        assert isinstance(results, dict)
        # At least some predictions should succeed
        assert len(results) > 0
        for pert_name, expr_matrix in results.items():
            assert expr_matrix.shape == (10, 30)
            assert np.isfinite(expr_matrix).all()


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
