"""Joint-structure-preserving training objectives (Track B, precision/OT).

Motivation. The covariance-Frobenius penalty (covariance_regularizer.py) was
DISCARDED: a pre-registered lambda sweep found no held-out causal-F1 gain (J2
null). Frobenius distance matches covariance *magnitude*, but interventional
causal discovery (GIES / inspre) exploits *conditional independence* = the
ZEROS of the precision (inverse-covariance) matrix, not covariance magnitude.
The identifiability proposition (J7) makes this precise: the CPDAG is a function
of the precision-matrix support, invariant to per-gene monotone calibration. So
the right training-time target is the *precision support*, or the full joint
distribution -- not the second moment.

This module supplies three differentiable penalties, added to any recon loss as
``L = L_recon + lambda * L_term``. They are the untried, better-motivated
objectives the J2 null points to:

  1. precision_support_penalty  -- align sparse inverse-covariance SUPPORT
     (graphical-lasso-style; the conditional-independence structure).
  2. mmd_penalty                -- RBF-kernel MMD between predicted and real
     joint perturbation-response rows (matches the whole distribution).
  3. sinkhorn_penalty           -- entropic-regularized OT (Wasserstein) between
     predicted and real joint distributions.

Each is unit-tested in tests/test_joint_structure_objectives.py. Running them at
scale (retraining CPA end-to-end) is a GPU job; the no-GPU proxy sweep
(scripts/b_precision_objective_sweep.py) is the local pre-registered stand-in.
"""
from __future__ import annotations

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# shared
# --------------------------------------------------------------------------- #
def _row_covariance(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Covariance over the first (sample) axis. x:(n,g) -> (g,g)."""
    if x.dim() != 2:
        raise ValueError(f"expected 2D (n_samples, n_genes), got {tuple(x.shape)}")
    n = x.shape[0]
    if n < 2:
        raise ValueError("covariance requires at least 2 samples")
    xc = x - x.mean(dim=0, keepdim=True)
    return (xc.transpose(0, 1) @ xc) / (n - 1 + eps)


def _differentiable_precision(x: Tensor, ridge: float = 1e-2, eps: float = 1e-8) -> Tensor:
    """Ridge-regularized precision (inverse covariance) of the sample rows.

    A ridge on the diagonal keeps the inverse well-conditioned and
    differentiable even when n_samples < n_genes (the perturbation-axis regime).
    """
    g = x.shape[1]
    cov = _row_covariance(x, eps=eps)
    cov = cov + ridge * torch.eye(g, dtype=x.dtype, device=x.device)
    return torch.linalg.inv(cov)


def _soft_support(theta: Tensor, tau: float = 1e-2) -> Tensor:
    """Differentiable soft |support| gate in [0,1): |t|/(|t|+tau).

    Approaches 1 for entries with magnitude >> tau (an "edge" in the
    conditional-independence graph) and 0 for entries << tau (a zero =
    conditional independence). tau sets the magnitude scale of "nonzero".
    """
    a = theta.abs()
    return a / (a + tau)


# --------------------------------------------------------------------------- #
# 1. precision-support
# --------------------------------------------------------------------------- #
def precision_support_penalty(
    pred_shift: Tensor,
    real_precision: Tensor,
    tau: float = 1e-2,
    ridge: float = 1e-2,
    normalize: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Align the OFF-DIAGONAL support of the predicted precision with real.

    Penalizes squared difference of the soft-support gates of the off-diagonal
    precision entries -- i.e. mismatch in the conditional-independence graph
    (which pairs are direct-neighbours), NOT covariance magnitude. This is the
    statistic interventional causal discovery actually reads.

    Parameters
    ----------
    pred_shift     : (n_perturbations, n_genes), requires grad.
    real_precision : (n_genes, n_genes) precomputed real precision target
                     (constant; use fit_real_precision()).
    tau            : support gate scale.
    ridge          : diagonal ridge for the predicted precision inverse.
    normalize      : divide by the number of off-diagonal entries.
    """
    real_precision = real_precision.detach()
    g = pred_shift.shape[1]
    theta_pred = _differentiable_precision(pred_shift, ridge=ridge, eps=eps)
    off = ~torch.eye(g, dtype=torch.bool, device=pred_shift.device)
    s_pred = _soft_support(theta_pred, tau)[off]
    s_real = _soft_support(real_precision, tau)[off]
    diff = s_pred - s_real
    num = (diff * diff).sum()
    if normalize:
        return num / off.sum().clamp_min(1)
    return num


def fit_real_precision(real_shift: Tensor, ridge: float = 1e-2) -> Tensor:
    """Precompute the real-data precision target (detached constant)."""
    return _differentiable_precision(real_shift.detach(), ridge=ridge)


# --------------------------------------------------------------------------- #
# 2. MMD (RBF, multi-bandwidth)
# --------------------------------------------------------------------------- #
def _pdist2(a: Tensor, b: Tensor) -> Tensor:
    """Squared euclidean distances, (na,g)x(nb,g) -> (na,nb)."""
    a2 = (a * a).sum(1, keepdim=True)
    b2 = (b * b).sum(1, keepdim=True).transpose(0, 1)
    return (a2 + b2 - 2.0 * a @ b.transpose(0, 1)).clamp_min(0.0)


def mmd_penalty(
    pred_shift: Tensor,
    real_shift: Tensor,
    bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
    eps: float = 1e-8,
) -> Tensor:
    """Biased multi-bandwidth RBF-kernel MMD^2 between predicted and real rows.

    Uses the plain V-statistic (full n x n / m x m means, self-terms included) --
    the standard BIASED MMD^2 estimator; smooth and well-behaved for gradient
    training. Matches the whole joint perturbation-response distribution (all
    moments), not just the second. real_shift is treated as a constant target.
    """
    real_shift = real_shift.detach()
    xx = _pdist2(pred_shift, pred_shift)
    yy = _pdist2(real_shift, real_shift)
    xy = _pdist2(pred_shift, real_shift)
    # median heuristic scale, then multiply by each bandwidth
    with torch.no_grad():
        med = xy.median().clamp_min(eps)
    mmd = pred_shift.new_zeros(())
    for bw in bandwidths:
        s = bw * med
        kxx = torch.exp(-xx / (2 * s))
        kyy = torch.exp(-yy / (2 * s))
        kxy = torch.exp(-xy / (2 * s))
        mmd = mmd + kxx.mean() + kyy.mean() - 2.0 * kxy.mean()
    return (mmd / len(bandwidths)).clamp_min(0.0)


# --------------------------------------------------------------------------- #
# 3. Sinkhorn OT (entropic Wasserstein)
# --------------------------------------------------------------------------- #
def sinkhorn_penalty(
    pred_shift: Tensor,
    real_shift: Tensor,
    blur: float = 0.5,
    n_iter: int = 50,
    eps: float = 1e-8,
) -> Tensor:
    """Entropic-regularized OT cost between predicted and real row-distributions.

    Uniform marginals; squared-euclidean ground cost normalized by its own
    median; ``blur`` is the entropic regularization strength. Differentiable via
    unrolled Sinkhorn iterations.
    """
    real_shift = real_shift.detach()
    C = _pdist2(pred_shift, real_shift)
    with torch.no_grad():
        scale = C.median().clamp_min(eps)
    C = C / scale
    n, m = C.shape
    reg = blur
    log_a = pred_shift.new_full((n,), -torch.log(torch.tensor(float(n))))
    log_b = pred_shift.new_full((m,), -torch.log(torch.tensor(float(m))))
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)
    Cr = C / reg
    for _ in range(n_iter):
        # f_i = -reg * logsumexp_j( log_b_j + g_j/reg - C_ij/reg )
        f = -reg * torch.logsumexp(log_b[None, :] + g[None, :] / reg - Cr, dim=1)
        g = -reg * torch.logsumexp(log_a[:, None] + f[:, None] / reg - Cr, dim=0)
    # transport plan and cost
    logP = log_a[:, None] + log_b[None, :] + (f[:, None] + g[None, :] - C) / reg
    P = torch.exp(logP)
    return (P * C).sum()


# --------------------------------------------------------------------------- #
# unified module
# --------------------------------------------------------------------------- #
class JointStructureLoss(torch.nn.Module):
    """Wrap a recon loss with ONE joint-structure term (pre-registered: one at a time).

    term ∈ {"precision", "mmd", "sinkhorn"}.
    loss = recon_loss + lam * term(pred_shift, real).
    For "precision" the real target is the precomputed precision buffer; for
    "mmd"/"sinkhorn" it is the raw real_shift buffer.
    """

    def __init__(self, term: str = "precision", lam: float = 1.0, **kw):
        super().__init__()
        if term not in ("precision", "mmd", "sinkhorn"):
            raise ValueError(f"unknown term {term!r}")
        if lam < 0:
            raise ValueError("lam must be non-negative")
        self.term = term
        self.lam = float(lam)
        self.kw = kw
        self.register_buffer("real_target", torch.empty(0))

    def fit_real(self, real_shift: Tensor) -> "JointStructureLoss":
        if self.term == "precision":
            ridge = self.kw.get("ridge", 1e-2)
            self.real_target = fit_real_precision(real_shift, ridge=ridge)
        else:
            self.real_target = real_shift.detach()
        return self

    def penalty(self, pred_shift: Tensor) -> Tensor:
        if self.real_target.numel() == 0:
            raise RuntimeError("call fit_real(real_shift) before penalty/forward")
        if self.term == "precision":
            return precision_support_penalty(
                pred_shift, self.real_target,
                tau=self.kw.get("tau", 1e-2), ridge=self.kw.get("ridge", 1e-2),
            )
        if self.term == "mmd":
            return mmd_penalty(pred_shift, self.real_target,
                               bandwidths=self.kw.get("bandwidths", (0.5, 1.0, 2.0, 4.0, 8.0)))
        return sinkhorn_penalty(pred_shift, self.real_target,
                                blur=self.kw.get("blur", 0.5),
                                n_iter=self.kw.get("n_iter", 50))

    def forward(self, recon_loss: Tensor, pred_shift: Tensor) -> Tensor:
        return recon_loss + self.lam * self.penalty(pred_shift)
