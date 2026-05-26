"""Sparsity-injection calibrator (Mode 3).

Soft-thresholds predicted log2FC values such that the near-zero fraction
matches a target (typically empirical real-data near-zero fraction).
"""
from __future__ import annotations
import numpy as np


class SparsityInjectionCalibrator:
    def __init__(self, target_near_zero_frac: float = 0.37,
                 near_zero_band: float = 0.05, tau_eps: float = 1e-4):
        self.target_near_zero_frac = float(target_near_zero_frac)
        self.near_zero_band = float(near_zero_band)
        self.tau_eps = float(tau_eps)
        self.threshold = None

    def fit(self, log_fc_pred: np.ndarray) -> "SparsityInjectionCalibrator":
        abs_fc = np.abs(np.asarray(log_fc_pred, dtype=np.float64).flatten())
        # Binary search for tau such that fraction(|FC| < tau) ≈ target
        lo, hi = 0.0, float(abs_fc.max() + self.tau_eps) if abs_fc.size else 1.0
        while hi - lo > self.tau_eps:
            mid = (lo + hi) / 2.0
            frac = float((abs_fc < mid).mean())
            if frac < self.target_near_zero_frac:
                lo = mid
            else:
                hi = mid
        self.threshold = (lo + hi) / 2.0
        return self

    def transform(self, log_fc: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            return log_fc
        tau = self.threshold
        sign = np.sign(log_fc)
        abs_fc = np.abs(log_fc)
        # Standard soft-thresholding (audit H8: use the textbook form, not the
        # non-standard denominator I sketched in V3)
        shrunk = np.maximum(abs_fc - tau, 0.0)
        return sign * shrunk
