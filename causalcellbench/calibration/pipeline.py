"""Calibration pipeline composer.

All modes operate on log2-fold-change matrices (n_perturbations, n_genes).
The quantile mode here is a thin re-implementation operating directly on
log2FC space, so that composition with covariance and sparsity (which also
operate on log2FC) is unit-consistent.
"""
from __future__ import annotations
import numpy as np
from itertools import permutations
from typing import Sequence

from causalcellbench.calibration.covariance_preservation import (
    CovariancePreservingCalibrator,
)
from causalcellbench.calibration.sparsity_injection import (
    SparsityInjectionCalibrator,
)


class _Log2FCQuantileMatcher:
    """Quantile-match |log2FC| of predictions to a reference real |log2FC|
    distribution, preserving sign."""

    def __init__(self, n_quantiles: int = 1000):
        self.n_quantiles = int(n_quantiles)
        self.q_points: np.ndarray | None = None
        self.real_q: np.ndarray | None = None

    def fit(self, real_log_fc: np.ndarray) -> "_Log2FCQuantileMatcher":
        abs_fc = np.abs(np.asarray(real_log_fc, dtype=np.float64).flatten())
        abs_fc = abs_fc[np.isfinite(abs_fc)]
        if abs_fc.size == 0:
            self.q_points = np.linspace(0, 1, self.n_quantiles)
            self.real_q = np.zeros_like(self.q_points)
            return self
        self.q_points = np.linspace(0.0, 1.0, self.n_quantiles)
        self.real_q = np.quantile(abs_fc, self.q_points)
        return self

    def transform(self, pred_log_fc: np.ndarray) -> np.ndarray:
        if self.q_points is None:
            return pred_log_fc
        sign = np.sign(pred_log_fc)
        abs_fc = np.abs(pred_log_fc)
        flat = abs_fc.flatten()
        n = flat.size
        if n == 0:
            return pred_log_fc
        order = np.argsort(flat, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64) / n
        calibrated_abs = np.interp(ranks, self.q_points, self.real_q)
        return (sign * calibrated_abs.reshape(abs_fc.shape)).astype(pred_log_fc.dtype)


class CalibrationPipeline:
    def __init__(self, order: Sequence[str]):
        valid = {"q", "quantile", "c", "covariance", "s", "sparsity"}
        unknown = [m for m in order if m not in valid]
        if unknown:
            raise ValueError(f"unknown modes: {unknown}")
        self.order = list(order)
        self.calibrators: dict[str, object] = {}

    @staticmethod
    def _canon(mode: str) -> str:
        return {"q": "quantile", "c": "covariance", "s": "sparsity"}.get(mode, mode)

    def fit(self, *, real_log_fc, real_pert_shifts=None, pred_log_fc):
        # `running` mirrors the cumulative output each mode actually receives at
        # transform time, so composition-order-dependent modes (sparsity) are fit
        # on the right distribution rather than the raw predictions.
        running = np.asarray(pred_log_fc, dtype=np.float64).copy()
        for mode in self.order:
            cmode = self._canon(mode)
            if cmode == "quantile":
                cal = _Log2FCQuantileMatcher().fit(real_log_fc)
            elif cmode == "covariance":
                # Fit the covariance basis on real log2FC (not raw count shifts) so
                # the projection is unit-consistent with the log2FC transform input.
                cal = CovariancePreservingCalibrator()
                cal.fit(real_log_fc)
            elif cmode == "sparsity":
                target = float((np.abs(real_log_fc) < 0.05).mean())
                cal = SparsityInjectionCalibrator(target_near_zero_frac=target)
                cal.fit(running)
            self.calibrators[mode] = cal
            running = cal.transform(running)
        return self

    def transform(self, x):
        out = np.asarray(x, dtype=np.float64).copy()
        for mode in self.order:
            cal = self.calibrators[mode]
            out = cal.transform(out)
        return out


def all_orderings() -> list[list[str]]:
    return [list(p) for p in permutations(["s", "q", "c"])]
