"""Tests for CausalBench bridge (disk-based and in-memory)."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from causalcellbench.causal.causalbench_bridge import (
    CausalBenchInput,
    build_from_disk,
    build_from_memory,
)
from causalcellbench.models.base import PerturbationResult


def _make_control(n_cells=100, n_genes=10, seed=0):
    rng = np.random.default_rng(seed)
    gene_names = [f"G{i}" for i in range(n_genes)]
    X = rng.negative_binomial(n=5, p=0.5, size=(n_cells, n_genes)).astype(np.float32)
    return AnnData(X=X, var=pd.DataFrame(index=gene_names))


def _make_perturbation_result(gene, n_cells=20, n_genes=10, seed=0):
    rng = np.random.default_rng(seed)
    gene_names = [f"G{i}" for i in range(n_genes)]
    X = rng.negative_binomial(n=5, p=0.5, size=(n_cells, n_genes)).astype(np.float32)
    return PerturbationResult(
        expression_matrix=X,
        gene_names=gene_names,
        perturbed_genes=[gene],
        cell_type="K562",
    )


class TestCausalBenchInput:
    def test_validate_passes_for_correct_input(self):
        cbi = CausalBenchInput(
            expression_matrix=np.zeros((30, 5)),
            interventions=["non-targeting"] * 10 + ["G0"] * 10 + ["G1"] * 10,
            gene_names=[f"G{i}" for i in range(5)],
            n_control_cells=10,
            n_perturbation_cells=20,
            perturbed_genes=["G0", "G1"],
        )
        cbi.validate()  # should not raise

    def test_validate_fails_on_cell_count_mismatch(self):
        cbi = CausalBenchInput(
            expression_matrix=np.zeros((25, 5)),  # wrong: 25 != 10+20
            interventions=["non-targeting"] * 10 + ["G0"] * 10 + ["G1"] * 10,
            gene_names=[f"G{i}" for i in range(5)],
            n_control_cells=10,
            n_perturbation_cells=20,
            perturbed_genes=["G0", "G1"],
        )
        with pytest.raises(ValueError):
            cbi.validate()

    def test_total_cells(self):
        cbi = CausalBenchInput(
            expression_matrix=np.zeros((30, 5)),
            interventions=["non-targeting"] * 10 + ["G0"] * 20,
            gene_names=[f"G{i}" for i in range(5)],
            n_control_cells=10,
            n_perturbation_cells=20,
            perturbed_genes=["G0"],
        )
        assert cbi.total_cells == 30


class TestBuildFromMemory:
    def test_basic_build(self):
        control = _make_control(n_cells=50, n_genes=10)
        gene_names = [f"G{i}" for i in range(10)]
        results = {
            "G0": _make_perturbation_result("G0", n_cells=20, n_genes=10, seed=1),
            "G1": _make_perturbation_result("G1", n_cells=20, n_genes=10, seed=2),
        }
        cbi = build_from_memory(control, results, gene_names)
        cbi.validate()
        assert cbi.n_control_cells == 50
        assert cbi.n_perturbation_cells == 40
        assert cbi.total_cells == 90
        assert len(cbi.perturbed_genes) == 2

    def test_intervention_labels_correct(self):
        control = _make_control(n_cells=30, n_genes=5)
        gene_names = [f"G{i}" for i in range(5)]
        results = {
            "G0": _make_perturbation_result("G0", n_cells=10, n_genes=5),
        }
        cbi = build_from_memory(control, results, gene_names)
        assert cbi.interventions[:30] == ["non-targeting"] * 30
        assert cbi.interventions[30:] == ["G0"] * 10

    def test_max_control_cells(self):
        control = _make_control(n_cells=1000, n_genes=5)
        gene_names = [f"G{i}" for i in range(5)]
        results = {"G0": _make_perturbation_result("G0", n_cells=10, n_genes=5)}
        cbi = build_from_memory(control, results, gene_names, max_control_cells=100)
        assert cbi.n_control_cells == 100


class TestBuildFromDisk:
    def test_reads_mean_h5ad(self):
        """Test that disk-based loading works for mean-only model outputs."""
        import anndata as ad

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create control
            control = _make_control(n_cells=50, n_genes=5)
            gene_names = [f"G{i}" for i in range(5)]

            # Create a mock model output (mean predictions)
            mean_preds = np.random.default_rng(0).random((3, 5)).astype(np.float32) * 10
            pred_adata = ad.AnnData(
                X=mean_preds,
                obs=pd.DataFrame({"perturbation": ["G0", "G1", "G2"]}),
                var=pd.DataFrame(index=gene_names),
                uns={"model": "test_model", "output_type": "mean"},
            )
            pred_adata.write_h5ad(f"{tmpdir}/test_predictions.h5ad")

            cbi = build_from_disk(
                results_dir=tmpdir,
                control_adata=control,
                gene_names=gene_names,
                n_cells_per_pert=20,
                seed=0,
            )
            cbi.validate()
            assert cbi.n_control_cells == 50
            # 3 perturbations x 20 cells each = 60
            assert cbi.n_perturbation_cells == 60
            assert cbi.total_cells == 110

    def test_reads_cell_h5ad(self):
        """Test that disk-based loading works for cell-level model outputs (CPA)."""
        import anndata as ad

        with tempfile.TemporaryDirectory() as tmpdir:
            control = _make_control(n_cells=30, n_genes=5)
            gene_names = [f"G{i}" for i in range(5)]

            # CPA-style output: actual cells per perturbation
            rng = np.random.default_rng(0)
            cells = rng.negative_binomial(n=5, p=0.5, size=(40, 5)).astype(np.float32)
            pred_adata = ad.AnnData(
                X=cells,
                obs=pd.DataFrame({"perturbation": ["G0"] * 20 + ["G1"] * 20}),
                var=pd.DataFrame(index=gene_names),
                uns={"model": "cpa", "output_type": "cells"},
            )
            pred_adata.write_h5ad(f"{tmpdir}/cpa_predictions.h5ad")

            cbi = build_from_disk(
                results_dir=tmpdir,
                control_adata=control,
                gene_names=gene_names,
                n_cells_per_pert=20,
                seed=0,
            )
            cbi.validate()
            assert cbi.n_control_cells == 30
            assert cbi.n_perturbation_cells == 40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
