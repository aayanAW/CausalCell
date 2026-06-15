"""DCDI-G: Differentiable Causal Discovery for Interventional Data (Gaussian).

Self-contained implementation of the linear Gaussian variant of DCDI
(Brouillard et al., NeurIPS 2020). Key advantage over GIES: handles
soft (imperfect) interventions matching CRISPRi's ~80-90% knockdown.

Uses:
  - Weighted adjacency matrix W learned via gradient descent
  - DAG constraint: h(W) = tr(e^{W*W}) - d = 0  (NOTEARS-style)
  - Augmented Lagrangian for constrained optimization
  - Soft interventions: intervened variables have shifted means

No R dependency. Requires only numpy, scipy, torch.

Reference: Brouillard et al., NeurIPS 2020.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import expm

from causalcellbench.causal.causalbench_bridge import CausalBenchInput

logger = logging.getLogger(__name__)


# =====================================================================
# DAG Constraint
# =====================================================================

class _TrExpScipy(torch.autograd.Function):
    """Autograd function: trace of matrix exponential via scipy."""

    @staticmethod
    def forward(ctx, W):
        W_np = W.detach().cpu().numpy()
        E = expm(W_np)
        ctx.save_for_backward(torch.as_tensor(E, dtype=W.dtype, device=W.device))
        return torch.tensor(np.trace(E), dtype=W.dtype, device=W.device)

    @staticmethod
    def backward(ctx, grad_output):
        (E,) = ctx.saved_tensors
        return E.t() * grad_output


def _dag_constraint(W: torch.Tensor) -> torch.Tensor:
    """h(W) = tr(e^{W*W}) - d.  Returns 0 iff W is a DAG."""
    d = W.shape[0]
    return _TrExpScipy.apply(W * W) - d


# =====================================================================
# DCDI-G Model (Linear Gaussian with soft interventions)
# =====================================================================

class _DCDIGaussian(nn.Module):
    """Linear Gaussian structural equation model.

    For each variable j:  X_j = sum_i W_{ij} * X_i + noise_j
    Under soft intervention on j:  X_j = sum_i W_{ij} * X_i + shift_j + noise_j

    Parameters
    ----------
    d : number of variables
    n_regimes : number of intervention regimes (including observational)
    """

    def __init__(self, d: int, n_regimes: int):
        super().__init__()
        self.d = d
        # Weighted adjacency matrix (unconstrained, will be sigmoid-masked)
        self.W = nn.Parameter(torch.zeros(d, d))
        # Per-variable log noise variance
        self.log_var = nn.Parameter(torch.zeros(d))
        # Intervention shifts: (n_regimes, d) — regime 0 is observational (shift=0)
        self.shifts = nn.Parameter(torch.zeros(n_regimes, d))

    def get_adjacency(self, threshold: float = 0.3) -> np.ndarray:
        """Return thresholded adjacency matrix."""
        W = self.W.detach().cpu().numpy()
        A = (np.abs(W) > threshold).astype(int)
        np.fill_diagonal(A, 0)
        return A

    def get_weighted_adjacency(self) -> np.ndarray:
        """Return raw weighted adjacency (no threshold)."""
        W = self.W.detach().cpu().numpy()
        np.fill_diagonal(W, 0)
        return W

    def log_likelihood(
        self,
        X: torch.Tensor,
        regimes: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute log-likelihood under linear Gaussian SEM.

        Parameters
        ----------
        X : (n, d) data matrix
        regimes : (n,) integer regime indices
        masks : (n, d) binary mask (1 = not intervened, 0 = intervened)
                For soft interventions, all entries are 1 (we use shifts instead).
        """
        W = self.W
        # Zero diagonal
        W_masked = W * (1 - torch.eye(self.d, device=W.device))

        # Predicted mean: X @ W (each row: predicted from parents)
        pred = X @ W_masked

        # Add intervention shifts per regime
        shift = self.shifts[regimes]  # (n, d)
        pred = pred + shift

        # Residuals
        residuals = X - pred
        var = torch.exp(self.log_var) + 1e-8  # (d,)

        # Gaussian log-likelihood per variable
        ll = -0.5 * (residuals ** 2 / var + torch.log(var) + np.log(2 * np.pi))

        if masks is not None:
            # For perfect interventions, mask out intervened variables
            ll = ll * masks

        return ll.sum(dim=1).mean()


# =====================================================================
# Training Loop
# =====================================================================

def _train_dcdi(
    X: np.ndarray,
    regimes: np.ndarray,
    masks: Optional[np.ndarray],
    n_regimes: int,
    max_epochs: int = 200,
    lr: float = 0.01,
    lambda_init: float = 0.0,
    mu_init: float = 0.001,
    mu_mult: float = 10.0,
    mu_max: float = 1e12,
    h_tol: float = 1e-8,
    l1_penalty: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """Train DCDI-G using augmented Lagrangian.

    Returns
    -------
    W : (d, d) weighted adjacency matrix.
    """
    torch.manual_seed(seed)
    n, d = X.shape

    X_t = torch.tensor(X, dtype=torch.float32)
    regimes_t = torch.tensor(regimes, dtype=torch.long)
    masks_t = torch.tensor(masks, dtype=torch.float32) if masks is not None else None

    model = _DCDIGaussian(d, n_regimes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lam = lambda_init
    mu = mu_init
    h_prev = np.inf

    for epoch in range(max_epochs):
        optimizer.zero_grad()

        # Negative log-likelihood
        nll = -model.log_likelihood(X_t, regimes_t, masks_t)

        # DAG constraint
        eye_d = torch.eye(d, device=model.W.device)
        W_abs = torch.abs(model.W) * (1 - eye_d)
        h = _dag_constraint(W_abs)

        # L1 sparsity
        l1 = l1_penalty * torch.sum(torch.abs(model.W) * (1 - eye_d))

        # Augmented Lagrangian
        loss = nll + l1 + lam * h + 0.5 * mu * h * h

        loss.backward()
        optimizer.step()

        # Zero diagonal after each step
        with torch.no_grad():
            model.W.data.fill_diagonal_(0.0)

        h_val = h.item()

        # Update Lagrangian parameters
        if (epoch + 1) % 50 == 0:
            if h_val > 0.25 * h_prev:
                mu = min(mu * mu_mult, mu_max)
            lam += mu * h_val
            h_prev = h_val

            if epoch < max_epochs - 1:
                logger.debug(
                    "  DCDI epoch %d: nll=%.4f, h=%.6f, mu=%.2f, lam=%.4f",
                    epoch + 1, nll.item(), h_val, mu, lam,
                )

        if h_val < h_tol:
            logger.info("  DCDI converged at epoch %d (h=%.2e)", epoch + 1, h_val)
            break

    return model.get_weighted_adjacency()


# =====================================================================
# Public API (matches run_gies interface)
# =====================================================================

def run_dcdi(
    cb_input: CausalBenchInput,
    seed: int = 0,
    partition_size: int = 50,
    intervention_type: str = "imperfect",
    max_epochs: int = 200,
    threshold: float = 0.3,
    l1_penalty: float = 0.01,
) -> list[tuple[str, str]]:
    """Run DCDI-G on CausalBench-formatted input.

    Parameters
    ----------
    cb_input
        Formatted input from ``build_causalbench_input()``.
    seed
        Random seed.
    partition_size
        Number of genes per partition (scalability).
    intervention_type
        ``"perfect"`` or ``"imperfect"`` (soft). Use ``"imperfect"``
        for CRISPRi data (~80-90% knockdown).
    max_epochs
        Maximum training epochs per partition.
    threshold
        Edge weight threshold for adjacency matrix.
    l1_penalty
        L1 sparsity penalty on adjacency weights.

    Returns
    -------
    list of (source_gene, target_gene) edge tuples.
    """
    gene_names = cb_input.gene_names
    n_genes = len(gene_names)

    # Build regime mapping: "non-targeting" = 0, others sorted alphabetically
    unique_labels = sorted(set(cb_input.interventions) - {"non-targeting"})
    unique_labels = ["non-targeting"] + unique_labels  # ensure regime 0 = observational
    label_to_regime = {label: idx for idx, label in enumerate(unique_labels)}
    n_regimes = len(unique_labels)

    all_edges: list[tuple[str, str]] = []

    # Partition genes for scalability
    rng = np.random.default_rng(seed)
    gene_order = rng.permutation(n_genes)
    partitions = [
        gene_order[i: i + partition_size]
        for i in range(0, n_genes, partition_size)
    ]

    for p_idx, partition_idx in enumerate(partitions):
        partition_genes = [gene_names[i] for i in partition_idx]
        partition_gene_set = set(partition_genes)
        n_part = len(partition_idx)

        logger.info("  DCDI partition %d/%d: %d genes", p_idx + 1, len(partitions), n_part)

        # Extract data for this partition
        X = cb_input.expression_matrix[:, partition_idx].astype(np.float64)

        # Build intervention masks for soft interventions
        # For imperfect interventions: all variables are observed (mask=1),
        # but intervened variables get a shift parameter
        if intervention_type == "perfect":
            masks = np.ones((X.shape[0], n_part), dtype=np.float64)
            for cell_idx, label in enumerate(cb_input.interventions):
                if label != "non-targeting" and label in partition_gene_set:
                    local_idx = partition_genes.index(label)
                    masks[cell_idx, local_idx] = 0.0  # mask out for perfect intervention
        else:
            masks = None  # soft interventions use shifts, not masks

        # Remap regimes: only include regimes relevant to this partition
        # (observational + perturbations of genes in this partition)
        partition_relevant = {"non-targeting"} | partition_gene_set
        relevant_labels = sorted(
            l for l in unique_labels if l in partition_relevant
        )
        part_label_to_regime = {l: i for i, l in enumerate(relevant_labels)}
        part_regimes = np.array([
            part_label_to_regime.get(l, 0)  # default to observational
            for l in cb_input.interventions
        ])
        n_part_regimes = len(relevant_labels)

        try:
            W = _train_dcdi(
                X, part_regimes, masks, n_part_regimes,
                max_epochs=max_epochs, l1_penalty=l1_penalty, seed=seed + p_idx,
            )

            # Extract edges from weighted adjacency
            for i in range(n_part):
                for j in range(n_part):
                    if i != j and abs(W[i, j]) > threshold:
                        all_edges.append((partition_genes[i], partition_genes[j]))

        except Exception as e:
            logger.warning("  DCDI partition %d failed: %s", p_idx, e)
            continue

    logger.info("  DCDI total: %d edges", len(all_edges))
    return all_edges
