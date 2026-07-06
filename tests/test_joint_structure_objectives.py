"""Numerical validation of the joint-structure objectives (Track B).

Covers the three untried, better-motivated penalties the J2 covariance null
points to: precision-support, MMD, Sinkhorn-OT. Each test checks (a) the
penalty is ~0 when predicted == real, (b) it is larger for a distribution far
from real than for one near it, and (c) a gradient step on the penalty reduces
it (the term is trainable).
"""
import torch

from causalcellbench.training.joint_structure_objectives import (
    JointStructureLoss,
    fit_real_precision,
    mmd_penalty,
    precision_support_penalty,
    sinkhorn_penalty,
    _differentiable_precision,
)


def _data(seed=0, n=128, g=10):
    torch.manual_seed(seed)
    # correlated real data so precision has real off-diagonal structure
    base = torch.randn(n, g)
    mix = torch.randn(g, g) * 0.3 + torch.eye(g)
    real = base @ mix
    return real


# --------------------------- precision-support --------------------------- #
def test_precision_zero_when_matched():
    real = _data()
    theta = fit_real_precision(real)
    pen = precision_support_penalty(real, theta)
    assert pen.item() < 1e-3


def test_precision_larger_for_wrong_support():
    real = _data()
    theta = fit_real_precision(real)
    # far: independent noise (diagonal precision => wrong off-diagonal support)
    far = torch.randn(128, 10)
    near = real + 0.05 * torch.randn(128, 10)
    p_far = precision_support_penalty(far, theta).item()
    p_near = precision_support_penalty(near, theta).item()
    assert p_far > p_near


def test_precision_gradient_reduces():
    real = _data()
    theta = fit_real_precision(real)
    pred = torch.randn(128, 10, requires_grad=True)
    opt = torch.optim.Adam([pred], lr=0.05)
    p0 = precision_support_penalty(pred, theta).item()
    for _ in range(100):
        opt.zero_grad()
        loss = precision_support_penalty(pred, theta)
        loss.backward()
        opt.step()
    p1 = precision_support_penalty(pred, theta).item()
    assert pred.grad is not None
    assert p1 < p0


# --------------------------------- MMD ---------------------------------- #
def test_mmd_zero_when_matched():
    real = _data()
    pen = mmd_penalty(real, real)
    assert pen.item() < 1e-4


def test_mmd_larger_for_shifted():
    real = _data()
    far = real + 5.0
    near = real + 0.1
    assert mmd_penalty(far, real).item() > mmd_penalty(near, real).item()


def test_mmd_gradient_reduces():
    real = _data()
    pred = (real + 3.0).clone().requires_grad_(True)
    opt = torch.optim.Adam([pred], lr=0.1)
    p0 = mmd_penalty(pred, real).item()
    for _ in range(100):
        opt.zero_grad()
        loss = mmd_penalty(pred, real)
        loss.backward()
        opt.step()
    p1 = mmd_penalty(pred, real).item()
    assert pred.grad is not None
    assert p1 < p0


# ------------------------------- Sinkhorn ------------------------------- #
def test_sinkhorn_smaller_when_matched():
    real = _data()
    far = real + 5.0
    assert sinkhorn_penalty(real, real).item() < sinkhorn_penalty(far, real).item()


def test_sinkhorn_gradient_reduces():
    real = _data()
    pred = (real + 3.0).clone().requires_grad_(True)
    opt = torch.optim.Adam([pred], lr=0.1)
    p0 = sinkhorn_penalty(pred, real).item()
    for _ in range(80):
        opt.zero_grad()
        loss = sinkhorn_penalty(pred, real)
        loss.backward()
        opt.step()
    p1 = sinkhorn_penalty(pred, real).item()
    assert pred.grad is not None
    assert p1 < p0


# ---------------------------- unified module ---------------------------- #
def test_module_adds_penalty():
    real = _data()
    for term in ("precision", "mmd", "sinkhorn"):
        crit = JointStructureLoss(term=term, lam=2.0).fit_real(real)
        recon = torch.tensor(1.0)
        pred = torch.randn(128, 10)
        total = crit(recon, pred)
        assert torch.isfinite(total)
        assert total.item() >= recon.item() - 1e-6  # penalty is non-negative-ish


def test_module_requires_fit():
    crit = JointStructureLoss(term="mmd", lam=1.0)
    try:
        crit.penalty(torch.randn(8, 10))
        assert False, "should have raised"
    except RuntimeError:
        pass
