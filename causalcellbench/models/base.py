"""Abstract base class for virtual cell perturbation models.

Every model wrapper must implement predict_perturbation():
    Given control cells and a gene to perturb, return N simulated single cells.

Models that natively support multi-gene perturbation (CPA, GEARS) override
supports_multi_perturbation and predict_multi_perturbation for the d-sep test.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from anndata import AnnData


@dataclass
class PerturbationResult:
    """Container for simulated perturbation output.

    Attributes
    ----------
    expression_matrix
        Simulated gene expression, shape ``(n_cells, n_genes)``.
    gene_names
        Gene names matching columns of expression_matrix.
    perturbed_genes
        Gene(s) that were perturbed (single string or list for multi-pert).
    cell_type
        Cell type context (e.g., ``"K562"``).
    metadata
        Optional dict for model-specific metadata (e.g., latent codes).
    """

    expression_matrix: np.ndarray
    gene_names: list[str]
    perturbed_genes: list[str]
    cell_type: str
    metadata: dict = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return self.expression_matrix.shape[0]

    @property
    def n_genes(self) -> int:
        return self.expression_matrix.shape[1]

    def validate(self) -> None:
        """Check shape consistency."""
        n_cells, n_genes = self.expression_matrix.shape
        if n_genes != len(self.gene_names):
            raise ValueError(
                f"Gene count mismatch: matrix has {n_genes} columns "
                f"but gene_names has {len(self.gene_names)} entries"
            )
        if n_cells == 0:
            raise ValueError("Empty expression matrix (0 cells)")


class AbstractPerturbationModel(ABC):
    """Base class for all virtual cell model wrappers.

    Subclasses must implement load_model() and predict_perturbation().
    Models with native multi-perturbation support (CPA, GEARS) should
    override supports_multi_perturbation and predict_multi_perturbation.
    """

    @abstractmethod
    def load_model(
        self,
        checkpoint_path: str,
        adata_train: Optional[AnnData] = None,
    ) -> None:
        """Load pretrained model weights.

        Parameters
        ----------
        checkpoint_path
            Path to model checkpoint or pretrained weights directory.
        adata_train
            Optional training AnnData for models that need it during init
            (e.g., CPA needs the training data schema).
        """

    @abstractmethod
    def predict_perturbation(
        self,
        control_adata: AnnData,
        perturbed_gene: str,
        n_cells: int,
        gene_names: list[str],
        seed: int = 0,
    ) -> PerturbationResult:
        """Simulate single-gene perturbation.

        Parameters
        ----------
        control_adata
            AnnData of unperturbed control cells. The model uses these
            as the starting point for perturbation simulation.
        perturbed_gene
            Gene to perturb (CRISPRi knockdown).
        n_cells
            Number of simulated cells to generate.
        gene_names
            Gene names for the output (must match control_adata.var_names
            intersection with model's gene vocabulary).
        seed
            Random seed for reproducibility.

        Returns
        -------
        PerturbationResult
            Simulated expression matrix of shape (n_cells, n_genes).
        """

    @property
    def supports_multi_perturbation(self) -> bool:
        """Whether this model natively supports multi-gene perturbation.

        Only CPA and GEARS return True. Used to gate d-separation test
        eligibility. Models that don't support native multi-pert should
        NOT be tested with d-sep (post-hoc output editing is invalid).
        """
        return False

    def predict_multi_perturbation(
        self,
        control_adata: AnnData,
        perturbed_genes: list[str],
        n_cells: int,
        gene_names: list[str],
        seed: int = 0,
    ) -> PerturbationResult:
        """Simulate multi-gene perturbation (CPA/GEARS only).

        Default implementation raises NotImplementedError. Override in
        models with native multi-perturbation support.

        Parameters
        ----------
        perturbed_genes
            List of genes to perturb simultaneously.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native "
            f"multi-gene perturbation. d-sep test not valid for this model."
        )

    def simulate_all_perturbations(
        self,
        control_adata: AnnData,
        genes_to_perturb: list[str],
        n_cells_per_gene: int,
        gene_names: list[str],
        seed: int = 0,
    ) -> dict[str, PerturbationResult]:
        """Convenience: simulate perturbation for each gene in the list.

        Parameters
        ----------
        genes_to_perturb
            List of genes to perturb (one at a time).
        n_cells_per_gene
            Number of simulated cells per perturbation.

        Returns
        -------
        dict mapping gene name -> PerturbationResult
        """
        results = {}
        for i, gene in enumerate(genes_to_perturb):
            results[gene] = self.predict_perturbation(
                control_adata=control_adata,
                perturbed_gene=gene,
                n_cells=n_cells_per_gene,
                gene_names=gene_names,
                seed=seed + i,
            )
        return results

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for this model (e.g., 'CPA', 'GEARS')."""
