"""Covariance-preservation calibrator (Mode 2).

Projects predicted shift vectors onto the top-k principal subspace of the
real perturbation covariance Σ_pert (NOT control covariance — see V4 audit
H7) and shrinks off-axis components.
"""
from __future__ import annotations
import numpy as np


class CovariancePreservingCalibrator:
    def __init__(self, n_components: int = 50, shrinkage: float = 0.5):
        self.n_components = n_components
        self.shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
        self.basis = None

    def fit(self, real_pert_shifts: np.ndarray) -> "CovariancePreservingCalibrator":
        """real_pert_shifts: (n_perturbations, n_genes) shift matrix from real data."""
        X = np.asarray(real_pert_shifts, dtype=np.float64)
        X = X - X.mean(axis=0, keepdims=True)
        # SVD to extract top-k subspace
        max_k = min(self.n_components, min(X.shape) - 1)
        if max_k < 1:
            self.basis = np.eye(X.shape[1])
            return self
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.basis = Vt[:max_k].T  # (n_genes, max_k)
        return self

    def transform(self, delta: np.ndarray) -> np.ndarray:
        if self.basis is None:
            return delta
        coords = delta @ self.basis           # (n_perts, k)
        on_basis = coords @ self.basis.T      # (n_perts, n_genes)
        return self.shrinkage * on_basis + (1 - self.shrinkage) * delta
