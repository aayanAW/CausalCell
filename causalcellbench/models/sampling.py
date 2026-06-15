"""Cell sampling utilities for CausalCellBench.

Models produce mean expression predictions. Causal discovery algorithms
(GIES, DCDI) require single-cell-level data. This module bridges the gap
using Negative Binomial sampling calibrated from real control cells.

WHY NEGATIVE BINOMIAL (not Gaussian):
    scRNA-seq count data is overdispersed and zero-inflated. A Gaussian
    model with non-negative clipping produces distributions that look
    nothing like real biology — the Wasserstein distance between
    Gaussian-sampled and real cells would be enormous. The Negative
    Binomial (NB) distribution is the standard generative model for
    scRNA-seq (scVI, DESeq2, ZINB-WaVE all use it).

NB parameterization:
    X ~ NB(r, p) where:
        r = dispersion (shape) parameter, estimated from control cells
        p = r / (r + mu)  where mu = model's predicted mean
        E[X] = mu, Var[X] = mu + mu^2/r  (overdispersed when r < inf)

Dispersion estimation:
    Method-of-moments: r = mu^2 / (var - mu) for var > mu
    With DESeq2-style shrinkage: fit log(r) ~ log(mu) trend, shrink
    per-gene estimates toward the trend for stability.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from anndata import AnnData


def estimate_nb_dispersion(
    control_adata: AnnData,
    gene_names: list[str],
    shrinkage: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-gene Negative Binomial dispersion from control cells.

    Parameters
    ----------
    control_adata
        AnnData of unperturbed control cells.
    gene_names
        Genes to estimate dispersion for.
    shrinkage
        If True, apply DESeq2-style shrinkage (recommended).
        Fits log(r) ~ log(mu) trend and shrinks per-gene estimates.

    Returns
    -------
    (mu, r) tuple:
        mu: np.ndarray (n_genes,) — per-gene mean expression
        r: np.ndarray (n_genes,) — per-gene dispersion parameter
    """
    # Check for duplicate gene names (would cause silent column misattribution)
    if len(set(gene_names)) != len(gene_names):
        dupes = [g for g in gene_names if gene_names.count(g) > 1]
        raise ValueError(f"Duplicate gene names in input: {set(dupes)}")

    var_names_list = list(control_adata.var_names)
    if len(set(var_names_list)) != len(var_names_list):
        dupes = [g for g in var_names_list if var_names_list.count(g) > 1]
        raise ValueError(f"Duplicate gene names in control_adata.var_names: {set(dupes)}")

    # Get gene indices (uses dict for O(1) lookup, consistent with bridge)
    var_to_idx = {name: i for i, name in enumerate(var_names_list)}
    gene_idx = []
    missing = []
    for g in gene_names:
        if g in var_to_idx:
            gene_idx.append(var_to_idx[g])
        else:
            missing.append(g)
    if missing:
        raise ValueError(
            f"{len(missing)} genes not found in control data: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    # Extract expression matrix
    X = control_adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X[:, gene_idx], dtype=np.float64)

    n_cells, n_genes = X.shape
    if n_cells < 10:
        warnings.warn(
            f"Only {n_cells} control cells for dispersion estimation. "
            f"Results may be unreliable. Recommend >= 100 cells."
        )

    # Per-gene mean and variance
    mu = X.mean(axis=0)
    var = X.var(axis=0, ddof=1)  # unbiased variance

    # Method of moments: r = mu^2 / (var - mu)
    # Valid when var > mu (overdispersed). When var <= mu (underdispersed
    # or Poisson-like), set r to a large value (near-Poisson regime).
    with np.errstate(divide="ignore", invalid="ignore"):
        overdispersed = var > mu + 1e-8
        r = np.where(
            overdispersed,
            mu ** 2 / (var - mu),
            1e4,  # near-Poisson fallback
        )

    # Floor and ceiling (0.1 floor avoids degenerate heavy-tailed distributions)
    r = np.clip(r, 0.1, 1e6)

    # Handle zero-mean genes (no expression in controls)
    zero_mean = mu < 1e-8
    r[zero_mean] = 1.0
    mu[zero_mean] = 1e-8

    # DESeq2-style shrinkage: fit trend, shrink toward it
    if shrinkage and n_genes >= 20:
        r = _shrink_dispersion(mu, r)

    return mu.astype(np.float64), r.astype(np.float64)


def _shrink_dispersion(
    mu: np.ndarray,
    r_raw: np.ndarray,
    shrinkage_weight: float = 0.5,
) -> np.ndarray:
    """Shrink per-gene dispersion estimates toward a mean-dispersion trend.

    Follows the DESeq2 approach: genes with similar mean expression
    should have similar dispersion. Shrinking toward the trend reduces
    noise in per-gene estimates, especially for low-expression genes.

    Parameters
    ----------
    mu
        Per-gene mean expression.
    r_raw
        Raw method-of-moments dispersion estimates.
    shrinkage_weight
        Weight toward trend (0 = no shrinkage, 1 = full shrinkage).

    Returns
    -------
    Shrunk dispersion estimates.
    """
    # Fit log(r) ~ log(mu) using robust linear regression
    valid = (mu > 1e-4) & np.isfinite(r_raw) & (r_raw > 0.01) & (r_raw < 1e5)
    if valid.sum() < 10:
        return r_raw  # not enough data to fit trend

    log_mu = np.log(mu[valid] + 1)
    log_r = np.log(r_raw[valid])

    # Simple robust linear fit (median-based to resist outliers)
    # log(r) = a + b * log(mu)
    from numpy.polynomial import polynomial as P

    try:
        coeffs = P.polyfit(log_mu, log_r, deg=1)
        r_trend = np.exp(P.polyval(np.log(mu + 1), coeffs))
        r_trend = np.clip(r_trend, 0.01, 1e6)
    except (np.linalg.LinAlgError, ValueError):
        return r_raw  # fit failed, return raw

    # Shrink: weighted geometric mean of raw and trend
    log_r_shrunk = (
        (1 - shrinkage_weight) * np.log(r_raw + 1e-8)
        + shrinkage_weight * np.log(r_trend + 1e-8)
    )
    return np.exp(log_r_shrunk)


def sample_cells_nb(
    mean_prediction: np.ndarray,
    dispersion_r: np.ndarray,
    n_cells: int,
    seed: int = 0,
) -> np.ndarray:
    """Sample cells from Negative Binomial distribution.

    Parameters
    ----------
    mean_prediction
        Model's predicted mean expression, shape ``(n_genes,)``.
    dispersion_r
        Per-gene NB dispersion parameter, shape ``(n_genes,)``.
        Estimated from control cells via ``estimate_nb_dispersion()``.
    n_cells
        Number of cells to generate.
    seed
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated count matrix, shape ``(n_cells, n_genes)``.
        Integer-valued, non-negative (natural from NB).
    """
    if mean_prediction.ndim != 1:
        raise ValueError(
            f"Expected 1D mean prediction, got shape {mean_prediction.shape}"
        )
    if len(mean_prediction) != len(dispersion_r):
        raise ValueError(
            f"mean_prediction length {len(mean_prediction)} != "
            f"dispersion_r length {len(dispersion_r)}"
        )

    # Guard against NaN/inf in model predictions
    mean_prediction = np.nan_to_num(
        mean_prediction, nan=0.0, posinf=1e4, neginf=0.0
    )

    rng = np.random.default_rng(seed)
    n_genes = len(mean_prediction)

    # Vectorized Gamma-Poisson compound (equivalent to NB, accepts float r):
    #   rate ~ Gamma(shape=r, scale=mu/r)  for each (cell, gene)
    #   X ~ Poisson(rate)
    # This gives E[X]=mu, Var[X]=mu+mu^2/r (NB distribution).
    # Using Gamma-Poisson avoids NumPy Generator's int-only constraint
    # on negative_binomial(n=...).
    mu = np.maximum(mean_prediction, 1e-8).astype(np.float64)
    r = np.maximum(dispersion_r, 0.1).astype(np.float64)

    # shape=(n_cells, n_genes), broadcast from (n_genes,)
    rates = rng.gamma(
        shape=r[np.newaxis, :],
        scale=(mu / r)[np.newaxis, :],
        size=(n_cells, n_genes),
    )
    cells = rng.poisson(lam=rates).astype(np.float32)

    return cells


def sample_perturbed_cells_overlay(
    ctrl_X: np.ndarray,
    mean_pred: np.ndarray,
    n_cells: int,
    seed: int = 0,
) -> np.ndarray:
    """Generate perturbed cells by overlaying fold-changes on control cells.

    CRITICAL: Independent NB sampling destroys gene-gene correlations,
    causing GIES to find ~0 edges even from correct perturbation means.
    This method preserves the control covariance structure by:
      1. Bootstrap sampling from control cells
      2. Scaling each gene by the predicted fold-change ratio
      3. Adding Poisson noise for count-level variability

    Parameters
    ----------
    ctrl_X
        Control expression matrix, shape ``(n_ctrl, n_genes)``.
        Should be the SAME subsampled controls that GIES sees.
    mean_pred
        Predicted mean expression under perturbation, shape ``(n_genes,)``.
    n_cells
        Number of cells to generate.
    seed
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated count matrix, shape ``(n_cells, n_genes)``.
    """
    rng = np.random.default_rng(seed)
    n_ctrl = ctrl_X.shape[0]

    # Bootstrap from controls
    idx = rng.choice(n_ctrl, size=n_cells, replace=True)
    cells = ctrl_X[idx].copy().astype(np.float64)

    # Compute fold-change ratios
    ctrl_means = ctrl_X.mean(axis=0).astype(np.float64)
    ratios = np.ones_like(ctrl_means)
    # Use low threshold (1e-6) to include low-expression genes;
    # pseudocount (0.01) prevents division-by-zero without losing signal
    nonzero = ctrl_means > 1e-6
    ctrl_means_safe = np.maximum(ctrl_means, 0.01)
    ratios[nonzero] = np.maximum(mean_pred[nonzero], 0.0) / ctrl_means_safe[nonzero]
    ratios = np.clip(ratios, 0.01, 10.0)  # prevent extreme scaling

    # Apply fold-change to each cell (preserves covariance structure)
    cells *= ratios[np.newaxis, :]

    # Add Poisson noise for count-level variability
    cells = np.maximum(cells, 0.0)
    cells = rng.poisson(lam=cells).astype(np.float32)

    return cells


def sample_cells_from_model_output(
    mean_prediction: np.ndarray,
    control_adata: AnnData,
    n_cells: int,
    gene_names: list[str],
    seed: int = 0,
    dispersion_cache: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """High-level API: generate N simulated cells from a model's mean prediction.

    This is the main entry point for model wrappers. Uses NB sampling
    with dispersion estimated from control cells.

    Parameters
    ----------
    mean_prediction
        Model's predicted mean expression, shape ``(n_genes,)``.
    control_adata
        AnnData of unperturbed control cells (for dispersion estimation).
    n_cells
        Number of cells to generate.
    gene_names
        Gene names matching columns of mean_prediction.
    seed
        Random seed.
    dispersion_cache
        Optional pre-computed (mu, r) tuple from ``estimate_nb_dispersion()``.
        Pass this to avoid re-estimating dispersion on every call.

    Returns
    -------
    np.ndarray
        Simulated count matrix, shape ``(n_cells, n_genes)``.
    """
    if dispersion_cache is not None:
        _, r = dispersion_cache
    else:
        _, r = estimate_nb_dispersion(control_adata, gene_names)

    return sample_cells_nb(mean_prediction, r, n_cells, seed)


def sample_cells_from_distribution(
    mean_prediction: np.ndarray,
    std_prediction: np.ndarray,
    n_cells: int,
    seed: int = 0,
) -> np.ndarray:
    """Generate cells when model provides both mean and std (e.g., CPA VAE).

    For CPA, the VAE posterior already captures the NB-like distribution
    from scVI training, so we sample from the posterior directly rather
    than imposing our own NB.

    Parameters
    ----------
    mean_prediction
        Mean expression, shape ``(n_genes,)``.
    std_prediction
        Per-gene standard deviation, shape ``(n_genes,)``.
    n_cells
        Number of cells to generate.
    seed
        Random seed.

    Returns
    -------
    np.ndarray
        Shape ``(n_cells, n_genes)``. Non-negative.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=(n_cells, len(mean_prediction)))
    cells = mean_prediction[np.newaxis, :] + noise * std_prediction[np.newaxis, :]
    return np.maximum(cells, 0.0).astype(np.float32)
