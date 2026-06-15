"""Tests for the Geneformer model wrapper.

Tests are split into two categories:
1. Unit tests (no Geneformer dependency) -- always run
2. Integration tests (require geneformer) -- skipped if not installed

Run unit tests only:
    pytest tests/test_geneformer_wrapper.py -v -k "not integration"

Run all (requires env_geneformer):
    conda run -n env_geneformer pytest tests/test_geneformer_wrapper.py -v
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_control_adata(
    n_cells: int = 200,
    n_genes: int = 50,
    seed: int = 42,
    with_ensembl: bool = False,
) -> AnnData:
    """Create synthetic control-only AnnData.

    Parameters
    ----------
    with_ensembl
        If True, add fake Ensembl IDs to var['ensembl_id'].
    """
    rng = np.random.default_rng(seed)
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    for g in range(n_genes):
        X[:, g] = rng.poisson(rng.exponential(5.0) + 0.5, size=n_cells)

    var_data = {"gene_name": gene_names}
    if with_ensembl:
        var_data["ensembl_id"] = [f"ENSG{i:011d}" for i in range(n_genes)]

    obs_data = {"n_counts": X.sum(axis=1).astype(np.float32)}

    return AnnData(
        X=X,
        obs=pd.DataFrame(obs_data),
        var=pd.DataFrame(var_data, index=gene_names),
    )


def _make_adata_with_ensembl_varnames(n_cells: int = 100, n_genes: int = 30) -> AnnData:
    """AnnData where var_names ARE Ensembl IDs (some datasets use this)."""
    rng = np.random.default_rng(123)
    ensembl_ids = [f"ENSG{i:011d}" for i in range(n_genes)]
    X = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float32)
    return AnnData(
        X=X,
        obs=pd.DataFrame({"n_counts": X.sum(axis=1)}),
        var=pd.DataFrame(index=ensembl_ids),
    )


# ---------------------------------------------------------------------------
# Unit tests: Gene ID mapping
# ---------------------------------------------------------------------------


class TestGeneSymbolToEnsemblMap:
    """Test the gene symbol to Ensembl ID mapping builder."""

    def test_from_ensembl_col(self):
        from causalcellbench.models.geneformer_wrapper import (
            build_gene_symbol_to_ensembl_map,
        )
        adata = _make_control_adata(n_cells=10, n_genes=5, with_ensembl=True)
        mapping = build_gene_symbol_to_ensembl_map(adata)
        assert len(mapping) == 5
        assert mapping["gene_0"] == "ENSG00000000000"
        assert mapping["gene_4"] == "ENSG00000000004"

    def test_from_varnames_ensembl(self):
        from causalcellbench.models.geneformer_wrapper import (
            build_gene_symbol_to_ensembl_map,
        )
        adata = _make_adata_with_ensembl_varnames(n_cells=5, n_genes=3)
        mapping = build_gene_symbol_to_ensembl_map(adata)
        assert len(mapping) == 3
        # When varnames are Ensembl IDs, maps to itself
        assert mapping["ENSG00000000000"] == "ENSG00000000000"

    def test_no_ensembl_warns(self):
        from causalcellbench.models.geneformer_wrapper import (
            build_gene_symbol_to_ensembl_map,
        )
        adata = _make_control_adata(n_cells=5, n_genes=3, with_ensembl=False)
        mapping = build_gene_symbol_to_ensembl_map(adata)
        assert len(mapping) == 0  # No mapping possible


# ---------------------------------------------------------------------------
# Unit tests: AnnData preparation
# ---------------------------------------------------------------------------


class TestPrepareAdataForGeneformer:
    """Test AnnData preparation for Geneformer tokenization."""

    def test_with_ensembl_col(self):
        from causalcellbench.models.geneformer_wrapper import (
            prepare_adata_for_geneformer,
        )
        adata = _make_control_adata(n_cells=20, n_genes=10, with_ensembl=True)
        result = prepare_adata_for_geneformer(adata)
        assert "ensembl_id" in result.var.columns
        assert "n_counts" in result.obs.columns
        assert result.n_vars == 10

    def test_with_ensembl_varnames(self):
        from causalcellbench.models.geneformer_wrapper import (
            prepare_adata_for_geneformer,
        )
        adata = _make_adata_with_ensembl_varnames()
        result = prepare_adata_for_geneformer(adata)
        assert "ensembl_id" in result.var.columns
        assert result.n_vars == 30

    def test_computes_n_counts(self):
        from causalcellbench.models.geneformer_wrapper import (
            prepare_adata_for_geneformer,
        )
        adata = _make_control_adata(n_cells=10, n_genes=5, with_ensembl=True)
        del adata.obs["n_counts"]
        result = prepare_adata_for_geneformer(adata)
        assert "n_counts" in result.obs.columns
        assert (result.obs["n_counts"] > 0).all()

    def test_no_ensembl_raises(self):
        from causalcellbench.models.geneformer_wrapper import (
            prepare_adata_for_geneformer,
        )
        adata = _make_control_adata(n_cells=10, n_genes=5, with_ensembl=False)
        with pytest.raises(ValueError, match="Ensembl"):
            prepare_adata_for_geneformer(adata)

    def test_filters_invalid_ensembl(self):
        from causalcellbench.models.geneformer_wrapper import (
            prepare_adata_for_geneformer,
        )
        adata = _make_control_adata(n_cells=10, n_genes=5, with_ensembl=True)
        # Corrupt two Ensembl IDs
        adata.var.loc["gene_0", "ensembl_id"] = "INVALID"
        adata.var.loc["gene_1", "ensembl_id"] = ""
        result = prepare_adata_for_geneformer(adata)
        assert result.n_vars == 3  # Only 3 valid remain

    def test_does_not_modify_original(self):
        from causalcellbench.models.geneformer_wrapper import (
            prepare_adata_for_geneformer,
        )
        adata = _make_control_adata(n_cells=10, n_genes=5, with_ensembl=True)
        orig_shape = adata.shape
        _ = prepare_adata_for_geneformer(adata)
        assert adata.shape == orig_shape


# ---------------------------------------------------------------------------
# Unit tests: Cosine similarity to fold change
# ---------------------------------------------------------------------------


class TestCosineSimToFoldChange:
    """Test the cosine similarity to expression fold change mapping."""

    def test_perfect_similarity(self):
        from causalcellbench.models.geneformer_wrapper import (
            cosine_sim_to_fold_change,
        )
        assert cosine_sim_to_fold_change(1.0) == pytest.approx(1.0)

    def test_zero_similarity(self):
        from causalcellbench.models.geneformer_wrapper import (
            cosine_sim_to_fold_change,
        )
        fc = cosine_sim_to_fold_change(0.0, max_fold_change=10.0)
        assert fc == pytest.approx(10.0)

    def test_negative_similarity(self):
        from causalcellbench.models.geneformer_wrapper import (
            cosine_sim_to_fold_change,
        )
        fc = cosine_sim_to_fold_change(-0.5, max_fold_change=10.0)
        assert fc == 10.0  # Capped at max

    def test_monotonicity(self):
        """Lower cosine similarity should give higher fold change."""
        from causalcellbench.models.geneformer_wrapper import (
            cosine_sim_to_fold_change,
        )
        fc_high = cosine_sim_to_fold_change(0.95)
        fc_mid = cosine_sim_to_fold_change(0.7)
        fc_low = cosine_sim_to_fold_change(0.3)
        assert fc_low > fc_mid > fc_high > 1.0

    def test_custom_max(self):
        from causalcellbench.models.geneformer_wrapper import (
            cosine_sim_to_fold_change,
        )
        fc = cosine_sim_to_fold_change(0.0, max_fold_change=5.0)
        assert fc == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Unit tests: Expression prediction from embeddings
# ---------------------------------------------------------------------------


class TestPredictExpressionFromEmbeddings:
    """Test expression prediction using per-gene cosine similarities."""

    def _make_gene_embs_dict(
        self,
        pert_token: int,
        affected_tokens: dict[int, float],
    ) -> dict:
        """Helper to create a mock gene_embs_dict.

        Parameters
        ----------
        pert_token
            Token ID of perturbed gene.
        affected_tokens
            {affected_token: mean_cosine_similarity}
        """
        d = {}
        for at, cos_sim in affected_tokens.items():
            d[(pert_token, at)] = [cos_sim]
        return d

    def test_perturbed_gene_knockdown(self):
        from causalcellbench.models.geneformer_wrapper import (
            predict_expression_from_embeddings,
        )
        control_mean = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        gene_names = ["gene_A", "gene_B", "gene_C"]

        pred = predict_expression_from_embeddings(
            control_mean=control_mean,
            gene_names=gene_names,
            perturbed_gene="gene_A",
            gene_embs_dict=None,
            perturbed_gene_token=100,
            token_to_ensembl={},
            ensembl_to_symbol={},
            knockdown_residual=0.05,
        )

        # Perturbed gene should be knocked down
        assert pred[0] == pytest.approx(10.0 * 0.05)
        # Other genes unchanged (no gene_embs_dict)
        assert pred[1] == pytest.approx(20.0)
        assert pred[2] == pytest.approx(30.0)

    def test_affected_genes_scaled(self):
        from causalcellbench.models.geneformer_wrapper import (
            predict_expression_from_embeddings,
        )
        control_mean = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        gene_names = ["gene_A", "gene_B", "gene_C"]

        # gene_B has low cosine sim (large effect)
        # gene_C has high cosine sim (small effect)
        gene_embs = {
            (100, 201): [0.5],   # gene_B: low cos_sim -> large change
            (100, 202): [0.95],  # gene_C: high cos_sim -> small change
        }

        token_to_ensembl = {201: "ENSG_B", 202: "ENSG_C"}
        ensembl_to_symbol = {"ENSG_B": "gene_B", "ENSG_C": "gene_C"}

        pred = predict_expression_from_embeddings(
            control_mean=control_mean,
            gene_names=gene_names,
            perturbed_gene="gene_A",
            gene_embs_dict=gene_embs,
            perturbed_gene_token=100,
            token_to_ensembl=token_to_ensembl,
            ensembl_to_symbol=ensembl_to_symbol,
        )

        # gene_B should be more affected than gene_C
        assert pred[1] < 20.0  # gene_B expression decreased
        assert pred[2] < 30.0  # gene_C expression decreased
        # gene_B should be more affected (lower cos_sim)
        assert (20.0 - pred[1]) / 20.0 > (30.0 - pred[2]) / 30.0

    def test_output_shape(self):
        from causalcellbench.models.geneformer_wrapper import (
            predict_expression_from_embeddings,
        )
        n_genes = 50
        control_mean = np.ones(n_genes, dtype=np.float32) * 5.0
        gene_names = [f"g_{i}" for i in range(n_genes)]

        pred = predict_expression_from_embeddings(
            control_mean=control_mean,
            gene_names=gene_names,
            perturbed_gene="g_0",
            gene_embs_dict=None,
            perturbed_gene_token=0,
            token_to_ensembl={},
            ensembl_to_symbol={},
        )

        assert pred.shape == (n_genes,)
        assert pred.dtype == np.float32


class TestPredictExpressionDirect:
    """Test simplified expression prediction (cell-level only)."""

    def test_knockdown(self):
        from causalcellbench.models.geneformer_wrapper import (
            predict_expression_direct,
        )
        control_mean = np.array([10.0, 20.0, 30.0])
        pred = predict_expression_direct(
            control_mean=control_mean,
            gene_names=["gene_A", "gene_B", "gene_C"],
            perturbed_gene="gene_A",
            knockdown_residual=0.1,
        )
        assert pred[0] == pytest.approx(1.0)

    def test_no_cell_emb(self):
        """Without cell embedding info, non-target genes should be close to control."""
        from causalcellbench.models.geneformer_wrapper import (
            predict_expression_direct,
        )
        control_mean = np.array([10.0, 20.0, 30.0])
        pred = predict_expression_direct(
            control_mean=control_mean,
            gene_names=["gene_A", "gene_B", "gene_C"],
            perturbed_gene="gene_A",
            cell_emb_cos_sim=None,
        )
        assert pred[1] == pytest.approx(20.0)
        assert pred[2] == pytest.approx(30.0)

    def test_non_negative(self):
        from causalcellbench.models.geneformer_wrapper import (
            predict_expression_direct,
        )
        control_mean = np.array([0.1, 0.01, 0.001])
        pred = predict_expression_direct(
            control_mean=control_mean,
            gene_names=["g0", "g1", "g2"],
            perturbed_gene="g0",
            cell_emb_cos_sim=0.5,
        )
        assert (pred >= 0).all()


# ---------------------------------------------------------------------------
# Unit tests: GeneformerModel class
# ---------------------------------------------------------------------------


class TestGeneformerModelInit:
    """Test GeneformerModel class without loading a real model."""

    def test_name(self):
        from causalcellbench.models.geneformer_wrapper import GeneformerModel
        model = GeneformerModel()
        assert model.name == "Geneformer"

    def test_no_multi_perturbation(self):
        from causalcellbench.models.geneformer_wrapper import GeneformerModel
        model = GeneformerModel()
        assert model.supports_multi_perturbation is False

    def test_predict_before_load_raises(self):
        from causalcellbench.models.geneformer_wrapper import GeneformerModel
        model = GeneformerModel()
        control = _make_control_adata(n_cells=10, n_genes=5)
        with pytest.raises(RuntimeError, match="load_model"):
            model.predict_perturbation(control, "gene_0", 10, ["gene_0"])


class TestGeneformerModelWithMock:
    """Test GeneformerModel predict methods with mock ISP results."""

    def _setup_mock_model(self, n_genes: int = 20):
        """Create a GeneformerModel with mock internals."""
        from causalcellbench.models.geneformer_wrapper import GeneformerModel

        gene_names = [f"gene_{i}" for i in range(n_genes)]
        ensembl_ids = [f"ENSG{i:011d}" for i in range(n_genes)]

        model = GeneformerModel()
        model._model_directory = "mock_model"
        model._token_dict = {ens: i + 100 for i, ens in enumerate(ensembl_ids)}
        model._token_dict["<cls>"] = 0
        model._token_dict["<eos>"] = 1
        model._token_dict["<pad>"] = 2
        model._token_to_ensembl = {i + 100: ens for i, ens in enumerate(ensembl_ids)}
        model._symbol_to_ensembl = dict(zip(gene_names, ensembl_ids))
        model._ensembl_to_symbol = dict(zip(ensembl_ids, gene_names))

        # Pre-cache predictions for gene_0
        control = _make_control_adata(n_cells=100, n_genes=n_genes)
        X = control.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        control_mean = X.mean(axis=0)

        # Simulate ISP results: knock down gene_0, affect gene_1 and gene_2
        pred = control_mean.copy()
        pred[0] *= 0.05  # knockdown
        pred[1] *= 0.7   # affected
        pred[2] *= 0.9   # slightly affected
        model._isp_results_cache["gene_0"] = pred.astype(np.float32)

        return model, gene_names

    def test_predict_from_cache(self):
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

    def test_predict_uncached_gene_fallback(self):
        """Gene not in ISP cache should use direct prediction fallback."""
        model, gene_names = self._setup_mock_model()
        control = _make_control_adata(n_cells=100, n_genes=20)

        # gene_5 is not in cache -- should fall back to direct prediction
        result = model.predict_perturbation(
            control_adata=control,
            perturbed_gene="gene_5",
            n_cells=30,
            gene_names=gene_names,
            seed=0,
        )

        assert result.expression_matrix.shape == (30, 20)
        assert result.perturbed_genes == ["gene_5"]

    def test_dispersion_cache_reused(self):
        model, gene_names = self._setup_mock_model()
        control = _make_control_adata(n_cells=100, n_genes=20)

        _ = model.predict_perturbation(control, "gene_0", 10, gene_names, seed=0)
        assert model._dispersion_cache is not None

        cache_before = model._dispersion_cache
        _ = model.predict_perturbation(control, "gene_0", 10, gene_names, seed=1)
        assert model._dispersion_cache is cache_before

    def test_gene_subset_alignment(self):
        """Requesting a subset of genes should produce correct shape."""
        model, _ = self._setup_mock_model(n_genes=20)
        control = _make_control_adata(n_cells=100, n_genes=20)

        subset = ["gene_5", "gene_10", "gene_15"]
        result = model.predict_perturbation(
            control_adata=control,
            perturbed_gene="gene_5",
            n_cells=20,
            gene_names=subset,
            seed=0,
        )

        assert result.expression_matrix.shape == (20, 3)
        assert result.gene_names == subset


# ---------------------------------------------------------------------------
# Unit tests: Load perturbation results
# ---------------------------------------------------------------------------


class TestLoadPerturbationResults:
    """Test loading ISP output pickle files."""

    def test_empty_directory(self, tmp_path):
        from causalcellbench.models.geneformer_wrapper import (
            load_perturbation_results,
        )
        cell_embs, gene_embs = load_perturbation_results(str(tmp_path), "test")
        assert cell_embs == {}
        assert gene_embs is None

    def test_load_cell_embs(self, tmp_path):
        import pickle
        from causalcellbench.models.geneformer_wrapper import (
            load_perturbation_results,
        )

        # Create mock cell embedding pickle
        mock_dict = {
            (100, "cell_emb"): [0.95, 0.93, 0.97],
            (101, "cell_emb"): [0.88, 0.90, 0.85],
        }
        with open(tmp_path / "test_cell_embs_dict_batch0.pickle", "wb") as f:
            pickle.dump(mock_dict, f)

        cell_embs, gene_embs = load_perturbation_results(str(tmp_path), "test")
        assert len(cell_embs) == 2
        assert len(cell_embs[(100, "cell_emb")]) == 3

    def test_load_gene_embs(self, tmp_path):
        import pickle
        from causalcellbench.models.geneformer_wrapper import (
            load_perturbation_results,
        )

        # Create mock gene embedding pickle
        mock_cell = {(100, "cell_emb"): [0.95]}
        mock_gene = {
            (100, 201): [0.7, 0.65, 0.72],
            (100, 202): [0.95, 0.94, 0.96],
        }
        with open(tmp_path / "test_cell_embs_dict_batch0.pickle", "wb") as f:
            pickle.dump(mock_cell, f)
        with open(tmp_path / "test_gene_embs_dict_batch0.pickle", "wb") as f:
            pickle.dump(mock_gene, f)

        cell_embs, gene_embs = load_perturbation_results(str(tmp_path), "test")
        assert gene_embs is not None
        assert len(gene_embs) == 2
        assert len(gene_embs[(100, 201)]) == 3

    def test_merge_multiple_batches(self, tmp_path):
        import pickle
        from causalcellbench.models.geneformer_wrapper import (
            load_perturbation_results,
        )

        # Two batch files for same key
        batch0 = {(100, "cell_emb"): [0.95, 0.93]}
        batch1 = {(100, "cell_emb"): [0.97, 0.91]}

        with open(tmp_path / "test_cell_embs_dict_batch0.pickle", "wb") as f:
            pickle.dump(batch0, f)
        with open(tmp_path / "test_cell_embs_dict_batch1.pickle", "wb") as f:
            pickle.dump(batch1, f)

        cell_embs, _ = load_perturbation_results(str(tmp_path), "test")
        assert len(cell_embs[(100, "cell_emb")]) == 4


# ---------------------------------------------------------------------------
# Integration tests (require geneformer installed)
# ---------------------------------------------------------------------------

try:
    from geneformer import InSilicoPerturber, TranscriptomeTokenizer, EmbExtractor
    GENEFORMER_AVAILABLE = True
except ImportError:
    GENEFORMER_AVAILABLE = False


@pytest.mark.skipif(not GENEFORMER_AVAILABLE, reason="geneformer not installed")
class TestGeneformerIntegration:
    """Integration tests that use the real Geneformer package.

    These tests require the env_geneformer conda environment.
    First run may download model weights from HuggingFace.
    """

    def test_token_dictionary_loads(self):
        from causalcellbench.models.geneformer_wrapper import (
            load_geneformer_token_dict,
        )
        token_dict = load_geneformer_token_dict()
        assert len(token_dict) > 1000  # Should have many genes
        assert "<cls>" in token_dict
        assert "<eos>" in token_dict

    def test_tokenizer_init(self):
        tokenizer = TranscriptomeTokenizer(
            custom_attr_name_dict={},
            nproc=1,
            model_input_size=4096,
            special_token=True,
            model_version="V2",
        )
        assert tokenizer is not None

    def test_prepare_and_tokenize(self, tmp_path):
        """Tokenize a small synthetic dataset with real Ensembl IDs."""
        from causalcellbench.models.geneformer_wrapper import (
            load_geneformer_token_dict,
            tokenize_adata,
        )

        # Get real Ensembl IDs from token dict
        token_dict = load_geneformer_token_dict()
        real_ensembl = [k for k in token_dict.keys()
                        if isinstance(k, str) and k.startswith("ENSG")][:50]

        # Create AnnData with real Ensembl IDs
        rng = np.random.default_rng(42)
        n_cells = 20
        n_genes = len(real_ensembl)
        X = rng.poisson(5.0, size=(n_cells, n_genes)).astype(np.float32)

        adata = AnnData(
            X=X,
            obs=pd.DataFrame({"n_counts": X.sum(axis=1)}),
            var=pd.DataFrame(
                {"ensembl_id": real_ensembl},
                index=real_ensembl,
            ),
        )

        tokenized_path = tokenize_adata(
            adata,
            output_dir=str(tmp_path),
            output_prefix="test",
            nproc=1,
        )

        assert tokenized_path.exists()

    @pytest.mark.slow
    def test_isp_small(self, tmp_path):
        """Run ISP on tiny synthetic data (smoke test)."""
        from causalcellbench.models.geneformer_wrapper import (
            load_geneformer_token_dict,
            tokenize_adata,
            run_in_silico_perturbation,
        )

        token_dict = load_geneformer_token_dict()
        real_ensembl = [k for k in token_dict.keys()
                        if isinstance(k, str) and k.startswith("ENSG")][:30]

        rng = np.random.default_rng(42)
        X = rng.poisson(5.0, size=(10, len(real_ensembl))).astype(np.float32)

        adata = AnnData(
            X=X,
            obs=pd.DataFrame({"n_counts": X.sum(axis=1)}),
            var=pd.DataFrame(
                {"ensembl_id": real_ensembl},
                index=real_ensembl,
            ),
        )

        # Tokenize
        tokenized_path = tokenize_adata(
            adata,
            output_dir=str(tmp_path / "tokenized"),
            output_prefix="test",
            nproc=1,
        )

        # Run ISP with one gene
        isp_out = run_in_silico_perturbation(
            model_directory="ctheodoris/Geneformer",
            tokenized_data_path=str(tokenized_path),
            output_directory=str(tmp_path / "isp"),
            output_prefix="test",
            genes_to_perturb=[real_ensembl[0]],
            emb_mode="cell",  # Faster than cell_and_gene
            forward_batch_size=10,
            max_ncells=5,
            nproc=1,
        )

        assert isp_out.exists()
        # Should have at least one output file
        output_files = list(isp_out.iterdir())
        assert len(output_files) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
