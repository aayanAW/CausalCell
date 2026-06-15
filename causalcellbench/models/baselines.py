"""Baseline models: Elastic Net and Random.

Elastic Net: for each gene h, fit ElasticNet on control cells to predict
h from all other genes. Then for a perturbation of gene g, predict
expression of all other genes given g is knocked down.

Random: return cells drawn from control distribution (shuffled).
Any real model should beat this.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from anndata import AnnData
from sklearn.linear_model import ElasticNet

from causalcellbench.models.base import AbstractPerturbationModel, PerturbationResult
from causalcellbench.models.sampling import sample_cells_from_model_output


class ElasticNetBaseline(AbstractPerturbationModel):
    """Elastic Net perturbation prediction baseline.

    Trains one ElasticNet per target gene on control cells:
        gene_h = f(all other genes)

    For perturbation of gene g: set g's expression to near-zero
    (CRISPRi knockdown), predict all other genes.
    """

    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5):
        self._alpha = alpha
        self._l1_ratio = l1_ratio
        self._models: dict[str, ElasticNet] = {}
        self._gene_names: list[str] = []
        self._control_means: np.ndarray | None = None
        self._fitted = False

    @property
    def name(self) -> str:
        return "ElasticNet"

    def load_model(
        self,
        checkpoint_path: str = "",
        adata_train: Optional[AnnData] = None,
    ) -> None:
        """Fit Elastic Net models on control cell data.

        Parameters
        ----------
        checkpoint_path
            Unused (models are fit from data).
        adata_train
            AnnData of control cells to train on.
        """
        if adata_train is None:
            raise ValueError("ElasticNet requires adata_train (control cells)")

        X = adata_train.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        self._gene_names = list(adata_train.var_names)
        self._control_means = X.mean(axis=0)

        n_genes = X.shape[1]
        self._models = {}

        for g in range(n_genes):
            # Features: all genes except g
            feature_idx = [i for i in range(n_genes) if i != g]
            X_train = X[:, feature_idx]
            y_train = X[:, g]

            model = ElasticNet(
                alpha=self._alpha,
                l1_ratio=self._l1_ratio,
                max_iter=1000,
                random_state=0,
            )
            model.fit(X_train, y_train)
            self._models[self._gene_names[g]] = model

        self._fitted = True

    def predict_perturbation(
        self,
        control_adata: AnnData,
        perturbed_gene: str,
        n_cells: int,
        gene_names: list[str],
        seed: int = 0,
    ) -> PerturbationResult:
        """Predict perturbation effect using Elastic Net."""
        if not self._fitted:
            raise RuntimeError("Call load_model() first")

        # Start from control cell means, apply knockdown
        input_state = self._control_means.copy()
        if perturbed_gene in self._gene_names:
            g_idx = self._gene_names.index(perturbed_gene)
            input_state[g_idx] *= 0.1  # ~90% CRISPRi knockdown

        # Predict all genes from the SAME input state (no cascade).
        # Each gene is predicted independently given the knockdown state.
        mean_pred = input_state.copy()
        for target_gene, model in self._models.items():
            if target_gene == perturbed_gene:
                continue
            t_idx = self._gene_names.index(target_gene)
            feature_idx = [i for i in range(len(self._gene_names)) if i != t_idx]
            features = input_state[feature_idx].reshape(1, -1)
            mean_pred[t_idx] = max(float(model.predict(features)[0]), 0.0)

        # Align to requested gene_names
        aligned = np.array(
            [mean_pred[self._gene_names.index(g)] for g in gene_names],
            dtype=np.float32,
        )

        # NB sample
        cells = sample_cells_from_model_output(
            aligned, control_adata, n_cells, gene_names, seed
        )

        return PerturbationResult(
            expression_matrix=cells,
            gene_names=gene_names,
            perturbed_genes=[perturbed_gene],
            cell_type="",
        )


class RandomBaseline(AbstractPerturbationModel):
    """Random baseline: return shuffled control cells.

    Floor baseline. Any real model must beat this. If it doesn't,
    the causal discovery pipeline is not extracting useful signal.
    """

    @property
    def name(self) -> str:
        return "Random"

    def load_model(
        self, checkpoint_path: str = "", adata_train: Optional[AnnData] = None
    ) -> None:
        pass  # Nothing to load

    def predict_perturbation(
        self,
        control_adata: AnnData,
        perturbed_gene: str,
        n_cells: int,
        gene_names: list[str],
        seed: int = 0,
    ) -> PerturbationResult:
        """Return random cells from control distribution."""
        rng = np.random.default_rng(seed)

        X = control_adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        # Align to requested genes
        var_names = list(control_adata.var_names)
        gene_idx = [var_names.index(g) for g in gene_names]
        X = X[:, gene_idx]

        # Random sample with replacement from control cells
        idx = rng.choice(X.shape[0], size=n_cells, replace=True)
        cells = X[idx]

        # Shuffle gene expression across cells to destroy any structure
        for col in range(cells.shape[1]):
            rng.shuffle(cells[:, col])

        return PerturbationResult(
            expression_matrix=cells,
            gene_names=gene_names,
            perturbed_genes=[perturbed_gene],
            cell_type="",
        )
