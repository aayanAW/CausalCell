"""Tests for the scGPT model wrapper.

Tests are split into two categories:
1. Unit tests (no scGPT/GEARS dependency) -- always run
2. Integration tests (require scGPT + GEARS) -- skipped if not installed

Run unit tests only:
    pytest tests/test_scgpt_wrapper.py -v -k "not integration"

Run all (requires env_scgpt):
    conda run -n env_scgpt pytest tests/test_scgpt_wrapper.py -v
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_control_adata(n_cells=200, n_genes=50, seed=42):
    """Create synthetic control-only AnnData."""
    rng = np.random.default_rng(seed)
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    for g in range(n_genes):
        X[:, g] = rng.poisson(rng.exponential(5.0) + 0.5, size=n_cells)
    return AnnData(X=X, var=pd.DataFrame(index=gene_names))


def _make_mock_adata(n_ctrl=200, n_pert_genes=10, n_cells_per_pert=30,
                     n_total_genes=100, seed=42):
    """Create a synthetic perturbation AnnData mimicking CRISPRi data."""
    rng = np.random.default_rng(seed)
    gene_names = [f"gene_{i}" for i in range(n_total_genes)]
    pert_genes = gene_names[:n_pert_genes]

    total_cells = n_ctrl + n_pert_genes * n_cells_per_pert
    gene_means = rng.exponential(scale=5.0, size=n_total_genes) + 0.5
    X = np.zeros((total_cells, n_total_genes), dtype=np.float32)
    labels = []

    for c in range(n_ctrl):
        for g in range(n_total_genes):
            X[c, g] = rng.poisson(gene_means[g])
        labels.append("non-targeting")

    idx = n_ctrl
    for pg in pert_genes:
        g_idx = gene_names.index(pg)
        for c in range(n_cells_per_pert):
            for g in range(n_total_genes):
                X[idx, g] = rng.poisson(gene_means[g])
            X[idx, g_idx] *= 0.1
            labels.append(pg)
            idx += 1

    obs = pd.DataFrame({"gene": labels})
    var = pd.DataFrame(index=gene_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_mock_checkpoint(tmpdir, n_vocab=100):
    """Create a fake scGPT checkpoint directory for testing load logic."""
    ckpt_dir = os.path.join(tmpdir, "mock_scgpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    # vocab.json
    token2idx = {f"gene_{i}": i + 3 for i in range(n_vocab)}
    token2idx["<pad>"] = 0
    token2idx["<cls>"] = 1
    token2idx["<eoc>"] = 2
    with open(os.path.join(ckpt_dir, "vocab.json"), "w") as f:
        json.dump(token2idx, f)

    # args.json
    config = {
        "embsize": 64,
        "nheads": 4,
        "nlayers": 2,
        "d_hid": 64,
        "n_layers_cls": 1,
    }
    with open(os.path.join(ckpt_dir, "args.json"), "w") as f:
        json.dump(config, f)

    # best_model.pt (empty tensor dict -- just for file existence check)
    import pickle
    with open(os.path.join(ckpt_dir, "best_model.pt"), "wb") as f:
        pickle.dump({}, f)

    return ckpt_dir


# ---------------------------------------------------------------------------
# Unit tests (no scGPT/GEARS dependency)
# ---------------------------------------------------------------------------


class TestConstants:
    """Test module-level constants and configuration."""

    def test_special_tokens(self):
        from causalcellbench.models.scgpt_wrapper import (
            PAD_TOKEN, CLS_TOKEN, EOC_TOKEN, SPECIAL_TOKENS,
        )
        assert PAD_TOKEN == "<pad>"
        assert CLS_TOKEN == "<cls>"
        assert EOC_TOKEN == "<eoc>"
        assert len(SPECIAL_TOKENS) == 3

    def test_default_hyperparams(self):
        from causalcellbench.models.scgpt_wrapper import (
            DEFAULT_EMBSIZE, DEFAULT_NHEADS, DEFAULT_NLAYERS,
            DEFAULT_D_HID, DEFAULT_N_LAYERS_CLS,
        )
        assert DEFAULT_EMBSIZE == 512
        assert DEFAULT_NHEADS == 8
        assert DEFAULT_NLAYERS == 12
        assert DEFAULT_D_HID == 512
        assert DEFAULT_N_LAYERS_CLS == 3

    def test_finetune_prefixes(self):
        from causalcellbench.models.scgpt_wrapper import FINETUNE_PARAM_PREFIXES
        assert "encoder" in FINETUNE_PARAM_PREFIXES
        assert "value_encoder" in FINETUNE_PARAM_PREFIXES
        assert "transformer_encoder" in FINETUNE_PARAM_PREFIXES


class TestScGPTModelInit:
    """Test ScGPTModel class without loading a real model."""

    def test_name(self):
        from causalcellbench.models.scgpt_wrapper import ScGPTModel
        model = ScGPTModel(device="cpu")
        assert model.name == "scGPT"

    def test_supports_multi_perturbation_false(self):
        from causalcellbench.models.scgpt_wrapper import ScGPTModel
        model = ScGPTModel(device="cpu")
        assert model.supports_multi_perturbation is False

    def test_predict_before_load_raises(self):
        from causalcellbench.models.scgpt_wrapper import ScGPTModel
        model = ScGPTModel(device="cpu")
        control = _make_control_adata(n_cells=10, n_genes=5)
        with pytest.raises(RuntimeError, match="load_model"):
            model.predict_perturbation(control, "gene_0", 10, ["gene_0"])

    def test_predict_mean_before_load_raises(self):
        from causalcellbench.models.scgpt_wrapper import ScGPTModel
        model = ScGPTModel(device="cpu")
        with pytest.raises(RuntimeError, match="load_model"):
            model.predict_perturbation_mean("gene_0")

    def test_device_auto_detect(self):
        from causalcellbench.models.scgpt_wrapper import ScGPTModel
        model = ScGPTModel()
        # Should be either 'cuda' or 'cpu' depending on environment
        assert model._device in ("cuda", "cpu")

    def test_device_explicit(self):
        from causalcellbench.models.scgpt_wrapper import ScGPTModel
        model = ScGPTModel(device="cpu")
        assert model._device == "cpu"

    def test_initial_state_is_none(self):
        from causalcellbench.models.scgpt_wrapper import ScGPTModel
        model = ScGPTModel(device="cpu")
        assert model._model is None
        assert model._vocab is None
        assert model._gene_ids is None
        assert model._ctrl_adata is None
        assert model._dispersion_cache is None


class TestBuildGeneIdMapping:
    """Test gene vocabulary mapping (without real GeneVocab)."""

    def test_mapping_with_mock_vocab(self):
        from causalcellbench.models.scgpt_wrapper import build_gene_id_mapping

        class MockVocab:
            def __init__(self):
                self._d = {"<pad>": 0, "TP53": 1, "MDM2": 2, "BRCA1": 3}

            def __contains__(self, item):
                return item in self._d

            def __getitem__(self, item):
                return self._d.get(item, 0)

        vocab = MockVocab()
        gene_names = ["TP53", "MDM2", "UNKNOWN_GENE", "BRCA1"]
        gene_ids = build_gene_id_mapping(vocab, gene_names)

        assert gene_ids.shape == (4,)
        assert gene_ids[0] == 1  # TP53
        assert gene_ids[1] == 2  # MDM2
        assert gene_ids[2] == 0  # UNKNOWN -> pad
        assert gene_ids[3] == 3  # BRCA1

    def test_all_unknown_genes(self):
        from causalcellbench.models.scgpt_wrapper import build_gene_id_mapping

        class MockVocab:
            def __init__(self):
                self._d = {"<pad>": 0}

            def __contains__(self, item):
                return item in self._d

            def __getitem__(self, item):
                return self._d.get(item, 0)

        vocab = MockVocab()
        gene_ids = build_gene_id_mapping(vocab, ["X", "Y", "Z"])
        assert (gene_ids == 0).all()


class TestMapGeneSymbols:
    """Test gene symbol to Ensembl ID mapping."""

    def test_fallback_identity(self):
        """Without gget, should return identity mapping."""
        from causalcellbench.models.scgpt_wrapper import map_gene_symbols_to_ensembl
        result = map_gene_symbols_to_ensembl(["TP53", "MDM2"])
        assert result["TP53"] == "TP53"
        assert result["MDM2"] == "MDM2"


class TestPredictPerturbation:
    """Test the predict_perturbation function with mock model."""

    def test_returns_dict_with_gene_key(self):
        """predict_perturbation should return dict with gene keys."""
        import torch
        from causalcellbench.models.scgpt_wrapper import predict_perturbation

        n_genes = 50
        gene_names = [f"gene_{i}" for i in range(n_genes)]
        gene_ids = np.arange(n_genes, dtype=int)

        class MockModel:
            def eval(self):
                pass

            def pred_perturb(self, batch_data, include_zero_gene, gene_ids, amp):
                bsz = len(batch_data.y)
                return torch.ones(bsz, n_genes) * 5.0

            def parameters(self):
                return [torch.tensor([1.0])]

        # Create mock control adata
        ctrl = _make_control_adata(n_cells=50, n_genes=n_genes)
        ctrl.obs["condition"] = "ctrl"

        # We need GEARS for cell graph creation, so this test will be
        # skipped if GEARS is not available
        try:
            from gears.utils import create_cell_graph_dataset_for_prediction
        except ImportError:
            pytest.skip("cell-gears not installed")

        # This would need actual GEARS data pipeline, so skip for unit test
        pytest.skip("Requires GEARS data pipeline for cell graph creation")


class TestScGPTModelWithMock:
    """Test ScGPTModel predict methods with a mock backend."""

    def _setup_mock_model(self, n_genes=20):
        """Create a ScGPTModel with mock internals."""
        from causalcellbench.models.scgpt_wrapper import ScGPTModel

        gene_names = [f"gene_{i}" for i in range(n_genes)]

        model = ScGPTModel(device="cpu")
        model._gene_list = gene_names
        model._gene_ids = np.arange(n_genes, dtype=int)
        model._model_config = {"embsize": 64}

        # Mock the predict_perturbation_mean to return a fixed vector
        # (bypasses actual model inference)
        mock_mean = np.ones(n_genes, dtype=np.float32) * 5.0

        class MockSelf:
            pass

        model._model = MockSelf()  # just needs to not be None

        # Patch predict_perturbation_mean
        model._original_predict_mean = model.predict_perturbation_mean
        model.predict_perturbation_mean = lambda gene: mock_mean.copy()

        return model, gene_names

    def test_predict_perturbation_with_sampling(self):
        model, gene_names = self._setup_mock_model(n_genes=20)
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
        """Test that scGPT output is aligned to requested gene_names."""
        model, full_genes = self._setup_mock_model(n_genes=20)
        control = _make_control_adata(n_cells=100, n_genes=20)

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
        model, gene_names = self._setup_mock_model(n_genes=20)
        control = _make_control_adata(n_cells=100, n_genes=20)

        _ = model.predict_perturbation(control, "gene_0", 10, gene_names, seed=0)
        assert model._dispersion_cache is not None

        cache_before = model._dispersion_cache
        _ = model.predict_perturbation(control, "gene_1", 10, gene_names, seed=1)
        assert model._dispersion_cache is cache_before

    def test_result_metadata(self):
        model, gene_names = self._setup_mock_model(n_genes=20)
        control = _make_control_adata(n_cells=100, n_genes=20)

        result = model.predict_perturbation(
            control, "gene_5", 10, gene_names, seed=0
        )

        assert "model_config" in result.metadata

    def test_missing_gene_uses_control_mean(self):
        """Genes not in scGPT vocab should fall back to control mean."""
        from causalcellbench.models.scgpt_wrapper import ScGPTModel

        n_genes = 20
        gene_names_model = [f"gene_{i}" for i in range(n_genes)]
        gene_names_request = gene_names_model + ["EXTRA_GENE"]

        model = ScGPTModel(device="cpu")
        model._gene_list = gene_names_model
        model._gene_ids = np.arange(n_genes, dtype=int)
        model._model_config = {}
        model._model = object()  # not None
        model.predict_perturbation_mean = lambda g: np.ones(n_genes) * 5.0

        # Create control with 21 genes (including EXTRA_GENE)
        rng = np.random.default_rng(42)
        X = np.zeros((100, 21), dtype=np.float32)
        for g in range(21):
            X[:, g] = rng.poisson(3.0, size=100)
        control = AnnData(
            X=X,
            var=pd.DataFrame(index=gene_names_request),
        )

        result = model.predict_perturbation(
            control, "gene_0", 10, gene_names_request, seed=0
        )

        # Should have 21 genes in output
        assert result.expression_matrix.shape == (10, 21)


# ---------------------------------------------------------------------------
# Availability checks (needed for skipif decorators below)
# ---------------------------------------------------------------------------

try:
    from scgpt.model import TransformerGenerator
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    SCGPT_AVAILABLE = True
except ImportError:
    SCGPT_AVAILABLE = False

try:
    from gears import PertData
    GEARS_AVAILABLE = True
except ImportError:
    GEARS_AVAILABLE = False


class TestDownloadCheckpoint:
    """Test checkpoint download logic."""

    def test_cached_checkpoint_returned(self):
        """If checkpoint exists, should return path without downloading."""
        from causalcellbench.models.scgpt_wrapper import download_scgpt_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake cached checkpoint
            ckpt_dir = os.path.join(tmpdir, "scGPT_human")
            os.makedirs(ckpt_dir)
            for f in ["vocab.json", "args.json", "best_model.pt"]:
                with open(os.path.join(ckpt_dir, f), "w") as fh:
                    fh.write("{}")

            result = download_scgpt_checkpoint(
                "scGPT_human", cache_dir=tmpdir
            )
            assert result == ckpt_dir


class TestLoadCheckpoint:
    """Test checkpoint loading validation.

    These tests require scGPT to be installed because
    load_scgpt_checkpoint imports scGPT internally.
    """

    @pytest.mark.skipif(not SCGPT_AVAILABLE, reason="scGPT not installed")
    def test_missing_files_raises(self):
        """Should raise FileNotFoundError for missing checkpoint files."""
        from causalcellbench.models.scgpt_wrapper import load_scgpt_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty directory
            with pytest.raises(FileNotFoundError, match="vocab.json"):
                load_scgpt_checkpoint(tmpdir, device="cpu")

    @pytest.mark.skipif(not SCGPT_AVAILABLE, reason="scGPT not installed")
    def test_missing_weights_raises(self):
        """Should raise if best_model.pt is missing."""
        from causalcellbench.models.scgpt_wrapper import load_scgpt_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create vocab.json and args.json but not best_model.pt
            with open(os.path.join(tmpdir, "vocab.json"), "w") as f:
                json.dump({"<pad>": 0}, f)
            with open(os.path.join(tmpdir, "args.json"), "w") as f:
                json.dump({"embsize": 64}, f)

            with pytest.raises(FileNotFoundError, match="best_model.pt"):
                load_scgpt_checkpoint(tmpdir, device="cpu")


class TestHasCuda:
    """Test CUDA detection helper."""

    def test_returns_bool(self):
        from causalcellbench.models.scgpt_wrapper import _has_cuda
        result = _has_cuda()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Integration tests (require scGPT + GEARS installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SCGPT_AVAILABLE, reason="scGPT not installed")
class TestScGPTIntegrationImports:
    """Integration tests that verify scGPT components load correctly."""

    def test_transformer_generator_init(self):
        """TransformerGenerator can be instantiated."""
        import torch

        vocab_dict = {f"gene_{i}": i + 3 for i in range(100)}
        vocab_dict["<pad>"] = 0
        vocab_dict["<cls>"] = 1
        vocab_dict["<eoc>"] = 2
        vocab = GeneVocab.from_dict(vocab_dict)

        model = TransformerGenerator(
            ntoken=len(vocab),
            d_model=64,
            nhead=4,
            d_hid=64,
            nlayers=2,
            nlayers_cls=1,
            n_cls=1,
            vocab=vocab,
            dropout=0.0,
            pad_token="<pad>",
            pad_value=0,
            pert_pad_id=0,
            use_fast_transformer=False,
        )
        assert model is not None
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_gene_vocab_roundtrip(self, tmp_path):
        """GeneVocab can be saved and loaded."""
        genes = [f"gene_{i}" for i in range(50)]
        vocab = GeneVocab(genes, specials=["<pad>", "<cls>", "<eoc>"])
        vocab.set_default_index(vocab["<pad>"])

        vocab_file = tmp_path / "vocab.json"
        vocab.save_json(str(vocab_file))

        vocab2 = GeneVocab.from_file(str(vocab_file))
        assert len(vocab2) == len(vocab)
        assert vocab2["gene_0"] == vocab["gene_0"]

    def test_load_checkpoint_function(self, tmp_path):
        """load_scgpt_checkpoint works with a small model."""
        import torch
        from causalcellbench.models.scgpt_wrapper import load_scgpt_checkpoint

        # Create minimal checkpoint
        n_genes = 50
        genes = [f"gene_{i}" for i in range(n_genes)]
        vocab = GeneVocab(genes, specials=["<pad>", "<cls>", "<eoc>"])
        vocab.set_default_index(vocab["<pad>"])
        vocab.save_json(str(tmp_path / "vocab.json"))

        config = {
            "embsize": 64,
            "nheads": 4,
            "nlayers": 2,
            "d_hid": 64,
            "n_layers_cls": 1,
        }
        with open(tmp_path / "args.json", "w") as f:
            json.dump(config, f)

        # Create a small model and save its state dict
        model = TransformerGenerator(
            ntoken=len(vocab),
            d_model=64,
            nhead=4,
            d_hid=64,
            nlayers=2,
            nlayers_cls=1,
            n_cls=1,
            vocab=vocab,
            dropout=0.0,
            pad_token="<pad>",
            pad_value=0,
            pert_pad_id=0,
            use_fast_transformer=False,
        )
        torch.save(model.state_dict(), str(tmp_path / "best_model.pt"))

        # Load it back
        loaded_model, loaded_vocab, loaded_config, _ = load_scgpt_checkpoint(
            str(tmp_path), device="cpu", use_fast_transformer=False,
        )

        assert loaded_model is not None
        assert len(loaded_vocab) == len(vocab)
        assert loaded_config["embsize"] == 64

    def test_build_gene_id_mapping_with_real_vocab(self, tmp_path):
        """build_gene_id_mapping works with real GeneVocab."""
        from causalcellbench.models.scgpt_wrapper import build_gene_id_mapping

        genes = ["TP53", "MDM2", "BRCA1", "EGFR"]
        vocab = GeneVocab(genes, specials=["<pad>", "<cls>", "<eoc>"])
        vocab.set_default_index(vocab["<pad>"])

        gene_ids = build_gene_id_mapping(vocab, ["TP53", "UNKNOWN", "BRCA1"])
        assert gene_ids.shape == (3,)
        assert gene_ids[0] == vocab["TP53"]
        assert gene_ids[1] == vocab["<pad>"]  # unknown -> pad
        assert gene_ids[2] == vocab["BRCA1"]


@pytest.mark.skipif(
    not (SCGPT_AVAILABLE and GEARS_AVAILABLE),
    reason="scGPT or GEARS not installed",
)
class TestScGPTIntegrationFull:
    """Full integration tests with scGPT + GEARS.

    These tests may download data on first run.
    Run in env_scgpt conda environment.
    """

    @pytest.mark.slow
    def test_predict_single_gene(self, tmp_path):
        """End-to-end: load small model, predict one gene perturbation."""
        import torch
        from causalcellbench.models.scgpt_wrapper import (
            predict_single_perturbation, build_gene_id_mapping,
        )

        # Load a small GEARS dataset
        pert_data = PertData(str(tmp_path / "gears_data"))
        pert_data.load(data_name="adamson")
        pert_data.prepare_split(split="simulation", seed=1)
        pert_data.get_dataloader(batch_size=32, test_batch_size=32)

        gene_list = pert_data.gene_names.values.tolist()
        ctrl_mask = pert_data.adata.obs["condition"] == "ctrl"
        ctrl_adata = pert_data.adata[ctrl_mask].copy()

        # Create small model with matching vocab
        genes = gene_list[:100]  # subset for speed
        vocab = GeneVocab(genes, specials=["<pad>", "<cls>", "<eoc>"])
        vocab.set_default_index(vocab["<pad>"])

        gene_ids = build_gene_id_mapping(vocab, gene_list)

        model = TransformerGenerator(
            ntoken=len(vocab),
            d_model=64,
            nhead=4,
            d_hid=64,
            nlayers=2,
            nlayers_cls=1,
            n_cls=1,
            vocab=vocab,
            dropout=0.0,
            pad_token="<pad>",
            pad_value=0,
            pert_pad_id=0,
            use_fast_transformer=False,
        )
        model.eval()
        device = "cpu"

        # Pick a gene that exists in the GEARS data
        test_genes = [
            g for g in gene_list
            if g in pert_data.adata.var["gene_name"].values
        ]
        if not test_genes:
            pytest.skip("No test genes found")

        gene = test_genes[0]
        result = predict_single_perturbation(
            model=model,
            gene=gene,
            ctrl_adata=ctrl_adata,
            gene_names=gene_list,
            gene_ids=gene_ids,
            device=device,
            pool_size=10,
            batch_size=8,
        )

        assert result.ndim == 1
        assert len(result) == len(gene_list)
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
