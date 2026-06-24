"""Training-time objectives for covariance-preserving perturbation prediction."""
from .covariance_regularizer import (
    covariance_frobenius_penalty,
    CovarianceRegularizedLoss,
)

__all__ = ["covariance_frobenius_penalty", "CovarianceRegularizedLoss"]
