"""Training-time covariance-preserving regularizer (J2).

Motivation. The benchmark's identifiability argument (Proposition: conditional
independence is invariant under per-gene monotone calibration) implies the
substitutability gap cannot be closed post-hoc on the marginal -- it lives in
the joint second-order structure (the gene-gene covariance / precision support
that interventional causal discovery exploits). This module supplies the
training-time remedy the post-hoc calibration sweep could not: a differentiable
penalty that aligns the predicted perturbation-response covariance with the
real-data covariance, added to any reconstruction loss.

    L = L_recon  +  lambda * || Cov(pred_shift) - Cov(real_shift) ||_F^2 / Z

where the shift is (predicted - control_mean), Cov is over the perturbation
axis, and Z normalizes by the squared Frobenius norm of the real covariance so
lambda is scale-free across datasets/gene-set sizes.

This is the method; running it at scale (retraining CPA/GEARS with this term)
is a GPU job. The penalty and its gradient are unit-tested in
tests/test_covariance_regularizer.py. Integration point for CPA: add the
penalty to the reconstruction term in causalcellbench/models/cpa_wrapper.py
train_cpa (or subclass the scvi module's loss); pass lambda via a new
``cov_reg_lambda`` argument.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _row_covariance(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Covariance over the first (perturbation/sample) axis.

    x: (n_samples, n_genes) -> (n_genes, n_genes).
    """
    if x.dim() != 2:
        raise ValueError(
            f"expected 2D (n_samples, n_genes), got shape {tuple(x.shape)}"
        )
    n = x.shape[0]
    if n < 2:
        raise ValueError("covariance requires at least 2 samples")
    xc = x - x.mean(dim=0, keepdim=True)
    return (xc.transpose(0, 1) @ xc) / (n - 1 + eps)


def covariance_frobenius_penalty(
    pred_shift: Tensor,
    real_cov: Tensor,
    normalize: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Differentiable ||Cov(pred_shift) - real_cov||_F^2 penalty.

    Parameters
    ----------
    pred_shift : (n_perturbations, n_genes) predicted shift matrix (requires grad).
    real_cov   : (n_genes, n_genes) precomputed real-data perturbation covariance
                 (a constant target; detach if it carries grad).
    normalize  : divide by ||real_cov||_F^2 so lambda is scale-free.

    Returns a scalar tensor.
    """
    real_cov = real_cov.detach()
    pred_cov = _row_covariance(pred_shift, eps=eps)
    diff = pred_cov - real_cov
    num = (diff * diff).sum()
    if normalize:
        denom = (real_cov * real_cov).sum().clamp_min(eps)
        return num / denom
    return num


class CovarianceRegularizedLoss(torch.nn.Module):
    """Wrap any per-sample reconstruction loss with the covariance penalty.

    loss = recon_loss + lambda_cov * covariance_frobenius_penalty(pred_shift, real_cov)

    real_cov is registered as a (non-trained) buffer; fit it once from the real
    training data via ``fit_real_covariance``.
    """

    def __init__(self, lambda_cov: float = 1.0, normalize: bool = True):
        super().__init__()
        if lambda_cov < 0:
            raise ValueError("lambda_cov must be non-negative")
        self.lambda_cov = float(lambda_cov)
        self.normalize = bool(normalize)
        self.register_buffer("real_cov", torch.empty(0))

    def fit_real_covariance(self, real_shift: Tensor) -> "CovarianceRegularizedLoss":
        self.real_cov = _row_covariance(real_shift.detach())
        return self

    def forward(self, recon_loss: Tensor, pred_shift: Tensor) -> Tensor:
        if self.real_cov.numel() == 0:
            raise RuntimeError("call fit_real_covariance(real_shift) before forward")
        penalty = covariance_frobenius_penalty(
            pred_shift, self.real_cov, normalize=self.normalize
        )
        return recon_loss + self.lambda_cov * penalty
