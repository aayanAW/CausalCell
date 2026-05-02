"""Post-hoc effect-size calibration for virtual cell model predictions.

Implements quantile matching, the standard distributional alignment technique
introduced by Bolstad et al. (Bioinformatics 2003) for microarray normalization
and adapted from Reinhard et al. (IEEE CG&A 2001) for image color transfer.

The algorithm is rank-preserving distribution mapping:
    For each predicted value x_i:
        FC_calibrated = sign(FC_pred) * F_real^{-1}(F_model(|FC_pred|))

This corrects the magnitude distribution of model predictions to match the
empirical distribution of real perturbation fold-changes, while preserving
the rank order of which genes the model thinks are most affected.

References:
    Bolstad et al. 2003. A comparison of normalization methods for high
        density oligonucleotide array data based on variance and bias.
        Bioinformatics 19(2):185-193.
    Reinhard et al. 2001. Color transfer between images.
        IEEE Computer Graphics and Applications 21(5):34-41.

The implementation uses numpy.interp for the inverse-CDF mapping, which is
the standard approach. No new algorithm is invented; we apply an established
statistical technique to a new domain (virtual cell model output calibration).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QuantileCalibrator:
    """Quantile-match model log2FC magnitudes to a reference distribution.

    Parameters
    ----------
    n_quantiles
        Number of quantile knots used to discretize the empirical CDFs.
        1000 is plenty for typical perturbation datasets (~200 perts x 200 genes).
    pseudocount
        Added to expression values before log to prevent log(0).
        0.01 matches the value used in overlay sampling.
    """

    n_quantiles: int = 1000
    pseudocount: float = 0.01

    # Fitted attributes
    real_quantiles_: np.ndarray | None = None
    quantile_points_: np.ndarray | None = None
    fitted_: bool = False

    def fit(
        self,
        real_log2fc_abs: np.ndarray,
    ) -> "QuantileCalibrator":
        """Estimate the empirical CDF of |log2FC| values from real data.

        Parameters
        ----------
        real_log2fc_abs
            Flat 1-D array of absolute log2 fold-change values from real
            perturbation data, pooled across all (perturbation, gene) pairs.
        """
        x = np.asarray(real_log2fc_abs, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("No finite real log2FC values to fit on.")

        self.quantile_points_ = np.linspace(0.0, 1.0, self.n_quantiles)
        self.real_quantiles_ = np.quantile(x, self.quantile_points_)
        self.fitted_ = True
        return self

    def transform(
        self,
        pred_expr: np.ndarray,
        ctrl_means: np.ndarray,
    ) -> np.ndarray:
        """Calibrate model predictions to match the fitted real distribution.

        Parameters
        ----------
        pred_expr
            Model's predicted mean expression, shape (n_perturbations, n_genes).
            These are expression values, not fold-changes.
        ctrl_means
            Per-gene control mean expression, shape (n_genes,).

        Returns
        -------
        Calibrated predicted expression matrix, same shape as pred_expr.
        Magnitudes are remapped to match the empirical real-FC distribution;
        rank order, sign of each fold-change, and per-perturbation gene
        ordering are preserved.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")

        pred = np.asarray(pred_expr, dtype=np.float64)
        ctrl = np.asarray(ctrl_means, dtype=np.float64)

        if pred.ndim != 2:
            raise ValueError(
                f"pred_expr must be 2D (n_perts, n_genes); got {pred.shape}"
            )
        if ctrl.shape[0] != pred.shape[1]:
            raise ValueError(
                f"ctrl_means length {ctrl.shape[0]} != pred_expr cols {pred.shape[1]}"
            )

        # Compute log2 fold-changes per (perturbation, gene)
        ps = self.pseudocount
        log2fc = np.log2((pred + ps) / (ctrl[None, :] + ps))
        signs = np.sign(log2fc)
        abs_fc = np.abs(log2fc)

        # Build the empirical CDF of model |log2FC| via rank interpolation
        flat = abs_fc.ravel()
        finite_mask = np.isfinite(flat)
        if not finite_mask.any():
            return pred  # nothing to calibrate

        # Empirical CDF: rank / n
        # ranks tied at average; gives values in (0, 1] which we use as
        # the input quantile to look up against the real distribution
        finite_vals = flat[finite_mask]
        n = finite_vals.size
        # argsort-of-argsort gives ranks 0..n-1; convert to (rank+1)/n for CDF
        order = np.argsort(finite_vals, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64) / n
        # ranks now contain the empirical CDF F_model(|fc|) for each value

        # Look up real-distribution quantile at each model rank
        # numpy.interp is the standard tool for inverse-CDF lookup
        calibrated_abs = np.interp(
            ranks,
            self.quantile_points_,
            self.real_quantiles_,
        )

        # Place calibrated values back into the full array, preserving sign;
        # non-finite cells fall back to zero (no perturbation effect)
        flat_out = np.zeros_like(flat)
        flat_out[finite_mask] = calibrated_abs
        abs_calibrated = flat_out.reshape(abs_fc.shape)
        log2fc_calibrated = signs * abs_calibrated

        # Convert calibrated log2FC back to expression scale
        # expr_cal = (ctrl + ps) * 2^log2fc_cal - ps
        pred_calibrated = (ctrl[None, :] + ps) * np.power(2.0, log2fc_calibrated) - ps
        # Clip negatives that can arise from the pseudocount subtraction
        pred_calibrated = np.maximum(pred_calibrated, 0.0)
        return pred_calibrated.astype(pred_expr.dtype, copy=False)

    def fit_transform(
        self,
        real_log2fc_abs: np.ndarray,
        pred_expr: np.ndarray,
        ctrl_means: np.ndarray,
    ) -> np.ndarray:
        """Convenience: fit on real data, then transform predictions."""
        return self.fit(real_log2fc_abs).transform(pred_expr, ctrl_means)


def empirical_real_log2fc(
    real_pert_expr: np.ndarray,
    real_pert_labels: list[str],
    ctrl_means: np.ndarray,
    pseudocount: float = 0.01,
) -> np.ndarray:
    """Compute |log2FC| values from real perturbation data, pooled.

    For each perturbation regime, take the per-gene mean across cells, compute
    log2(mean / ctrl_mean), and pool absolute values across all (regime, gene)
    pairs. Returns a flat 1-D array suitable as input to QuantileCalibrator.fit.

    Parameters
    ----------
    real_pert_expr
        Concatenated real perturbation expression matrix, shape (n_cells, n_genes).
    real_pert_labels
        Per-cell perturbation label, length n_cells.
    ctrl_means
        Per-gene control mean, shape (n_genes,).
    pseudocount
        Same pseudocount used in calibration; default 0.01.
    """
    expr = np.asarray(real_pert_expr, dtype=np.float64)
    ctrl = np.asarray(ctrl_means, dtype=np.float64)
    labels = np.asarray(real_pert_labels)

    # Per-perturbation mean
    unique_labels = np.unique(labels)
    means = []
    for lab in unique_labels:
        mask = labels == lab
        if mask.sum() == 0:
            continue
        means.append(expr[mask].mean(axis=0))
    if not means:
        raise ValueError("No perturbation labels found.")

    means_arr = np.stack(means, axis=0)
    log2fc = np.log2((means_arr + pseudocount) / (ctrl[None, :] + pseudocount))
    return np.abs(log2fc).ravel()
