#!/usr/bin/env bash
# Phase 4: Calibration as diagnostic. 3 modes + 6-order ablation + 80/20 train/test split.
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1
if [ -f "$ROOT/.venv_gears/bin/activate" ]; then source "$ROOT/.venv_gears/bin/activate"; fi
export OMP_NUM_THREADS=1 PYTHONPATH="$ROOT"

# Generate Mode 2 + Mode 3 if not present
mkdir -p "$ROOT/causalcellbench/calibration"

# Modules already in place — skip overwrite if present
if [ -f "$ROOT/causalcellbench/calibration/pipeline.py" ] \
   && [ -f "$ROOT/causalcellbench/calibration/covariance_preservation.py" ] \
   && [ -f "$ROOT/causalcellbench/calibration/sparsity_injection.py" ]; then
    echo "  Calibration modules already present, skipping module write"
else

# Mode 2: covariance preservation using Σ_pert (per H7 audit fix)
cat > "$ROOT/causalcellbench/calibration/covariance_preservation.py" <<'PY'
"""Covariance-preservation calibrator (Mode 2).

Projects predicted shift vectors onto the top-k principal subspace of the
real perturbation covariance Σ_pert (NOT control covariance — see V4 audit
H7) and shrinks off-axis components.
"""
from __future__ import annotations
import numpy as np


class CovariancePreservingCalibrator:
    def __init__(self, n_components: int = 50, shrinkage: float = 0.5):
        self.n_components = n_components
        self.shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
        self.basis = None

    def fit(self, real_pert_shifts: np.ndarray) -> "CovariancePreservingCalibrator":
        """real_pert_shifts: (n_perturbations, n_genes) shift matrix from real data."""
        X = np.asarray(real_pert_shifts, dtype=np.float64)
        X = X - X.mean(axis=0, keepdims=True)
        # SVD to extract top-k subspace
        max_k = min(self.n_components, min(X.shape) - 1)
        if max_k < 1:
            self.basis = np.eye(X.shape[1])
            return self
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.basis = Vt[:max_k].T  # (n_genes, max_k)
        return self

    def transform(self, delta: np.ndarray) -> np.ndarray:
        if self.basis is None:
            return delta
        coords = delta @ self.basis           # (n_perts, k)
        on_basis = coords @ self.basis.T      # (n_perts, n_genes)
        return self.shrinkage * on_basis + (1 - self.shrinkage) * delta
PY

# Mode 3: sparsity injection
cat > "$ROOT/causalcellbench/calibration/sparsity_injection.py" <<'PY'
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
PY

# Pipeline composer (per H10: 6-permutation ablation)
cat > "$ROOT/causalcellbench/calibration/pipeline.py" <<'PY'
"""Calibration pipeline composer.

Supports arbitrary ordering of three modes:
- quantile : QuantileCalibrator (existing Mode 1)
- covariance : CovariancePreservingCalibrator (Mode 2)
- sparsity : SparsityInjectionCalibrator (Mode 3)
"""
from __future__ import annotations
import numpy as np
from itertools import permutations

from causalcellbench.calibration.quantile_calibration import QuantileCalibrator
from causalcellbench.calibration.covariance_preservation import (
    CovariancePreservingCalibrator,
)
from causalcellbench.calibration.sparsity_injection import (
    SparsityInjectionCalibrator,
)

_MODE_REGISTRY = {
    "quantile": QuantileCalibrator,
    "covariance": CovariancePreservingCalibrator,
    "sparsity": SparsityInjectionCalibrator,
}


class CalibrationPipeline:
    def __init__(self, order: list[str]):
        unknown = [m for m in order if m not in _MODE_REGISTRY]
        if unknown:
            raise ValueError(f"unknown modes: {unknown}")
        self.order = list(order)
        self.calibrators = {}

    def fit(self, *, real_log_fc, real_pert_shifts, pred_log_fc):
        for mode in self.order:
            cls = _MODE_REGISTRY[mode]
            if mode == "quantile":
                cal = cls().fit(real_log_fc.flatten(), pred_log_fc.flatten())
            elif mode == "covariance":
                cal = cls().fit(real_pert_shifts)
            elif mode == "sparsity":
                target = float((np.abs(real_log_fc) < 0.05).mean())
                cal = cls(target_near_zero_frac=target).fit(pred_log_fc)
            self.calibrators[mode] = cal
        return self

    def transform(self, x):
        out = np.asarray(x, dtype=np.float64).copy()
        for mode in self.order:
            out = self.calibrators[mode].transform(out)
        return out


def all_orderings():
    return [list(p) for p in permutations(["sparsity","quantile","covariance"])]
PY

# Update __init__.py
cat > "$ROOT/causalcellbench/calibration/__init__.py" <<'PY'
"""Calibration module — three modes (quantile, covariance, sparsity)
and a composable Pipeline. See ARCHITECTURE_V4_AUDIT_AND_UPGRADE.md."""
from causalcellbench.calibration.quantile_calibration import (
    QuantileCalibrator,
    empirical_real_log2fc,
)
from causalcellbench.calibration.covariance_preservation import (
    CovariancePreservingCalibrator,
)
from causalcellbench.calibration.sparsity_injection import (
    SparsityInjectionCalibrator,
)
from causalcellbench.calibration.pipeline import (
    CalibrationPipeline,
    all_orderings,
)

__all__ = [
    "QuantileCalibrator",
    "CovariancePreservingCalibrator",
    "SparsityInjectionCalibrator",
    "CalibrationPipeline",
    "empirical_real_log2fc",
    "all_orderings",
]
PY
fi  # end of "skip if modules already present"

# Pre-flight test: import works
python3 -c "
from causalcellbench.calibration import CalibrationPipeline, all_orderings
print('  calibration modules import OK')
print('  orderings:', all_orderings())
" || { echo "  Calibration import FAILED"; exit 1; }

# Run sweep (placeholder: actual sweep depends on Phase 1 + Phase 2 deliverables)
mkdir -p "$ROOT/results/calibration_modes" "$ROOT/data/model_outputs_phase4"
python3 "$ROOT/scripts/v4/phase_4_calibration_sweep.py" 2>&1 | tail -80
echo ""
echo "  Running GIES on each calibrated condition (slow, ~5 min each × 27 = ~2.25h)"
python3 "$ROOT/scripts/v4/phase_4_calibration_gies.py" 2>&1 | tail -60
