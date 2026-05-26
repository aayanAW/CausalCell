"""Effect-size calibration for virtual cell model predictions.

Three modes (V4 audit-driven):
- Quantile  : rank-preserving distribution mapping (Bolstad 2003; Reinhard 2001).
- Covariance: project predicted shifts onto top-k Σ_pert subspace (V4 H7 fix).
- Sparsity  : soft-threshold to match empirical near-zero fraction (V4 H8).

Pipeline composes them in any order; 6 orderings of the 3 modes form the
calibration ablation space used in Phase 4.
"""
from causalcellbench.calibration.quantile_calibrator import (
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
