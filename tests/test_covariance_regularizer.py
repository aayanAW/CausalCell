"""Numerical validation of the training-time covariance regularizer (J2)."""

import torch

from causalcellbench.training.covariance_regularizer import (
    CovarianceRegularizedLoss,
    covariance_frobenius_penalty,
    _row_covariance,
)


def _make(seed=0, n=64, g=12):
    torch.manual_seed(seed)
    real = torch.randn(n, g)
    real_cov = _row_covariance(real)
    return real, real_cov


def test_zero_penalty_when_covariance_matches():
    real, real_cov = _make()
    # pred with identical covariance (use the real shifts themselves)
    pen = covariance_frobenius_penalty(real, real_cov)
    assert pen.item() < 1e-6


def test_penalty_positive_and_decreases_toward_real():
    real, real_cov = _make()
    far = torch.randn(64, 12) * 5.0  # different covariance
    near = 0.9 * real + 0.1 * torch.randn(64, 12)  # closer to real
    pen_far = covariance_frobenius_penalty(far, real_cov).item()
    pen_near = covariance_frobenius_penalty(near, real_cov).item()
    assert pen_far > 0
    assert pen_near < pen_far


def test_gradient_step_reduces_penalty():
    real, real_cov = _make()
    pred = torch.randn(64, 12, requires_grad=True)
    opt = torch.optim.SGD([pred], lr=0.1)
    p0 = covariance_frobenius_penalty(pred, real_cov)
    for _ in range(50):
        opt.zero_grad()
        loss = covariance_frobenius_penalty(pred, real_cov)
        loss.backward()
        opt.step()
    p1 = covariance_frobenius_penalty(pred, real_cov)
    assert p1.item() < p0.item()  # gradient descent on the penalty works
    assert pred.grad is not None


def test_combined_loss_adds_penalty():
    real, _ = _make()
    crit = CovarianceRegularizedLoss(lambda_cov=2.0).fit_real_covariance(real)
    recon = torch.tensor(1.0)
    pred = torch.randn(64, 12)
    total = crit(recon, pred)
    expected_pen = covariance_frobenius_penalty(pred, _row_covariance(real))
    assert torch.allclose(total, recon + 2.0 * expected_pen, atol=1e-5)


def test_lambda_zero_is_recon_only():
    real, _ = _make()
    crit = CovarianceRegularizedLoss(lambda_cov=0.0).fit_real_covariance(real)
    recon = torch.tensor(3.14)
    total = crit(recon, torch.randn(64, 12))
    assert torch.allclose(total, recon)


def test_normalization_is_scale_free():
    real, real_cov = _make()
    pred = torch.randn(64, 12)
    p_unit = covariance_frobenius_penalty(pred, real_cov, normalize=True)
    # scaling both pred and target by c leaves the normalized penalty unchanged
    p_scaled = covariance_frobenius_penalty(10 * pred, 100 * real_cov, normalize=True)
    assert abs(p_unit.item() - p_scaled.item()) < 1e-4
