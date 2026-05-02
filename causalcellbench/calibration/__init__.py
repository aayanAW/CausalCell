"""Effect-size calibration for virtual cell model predictions.

The ``QuantileCalibrator`` implements rank-preserving distribution mapping
(Bolstad et al., Bioinformatics 2003; Reinhard et al., IEEE CG&A 2001),
applied here to align virtual cell model |log2FC| outputs with the empirical
distribution of real perturbation effects.
"""

from causalcellbench.calibration.quantile_calibrator import (
    QuantileCalibrator,
    empirical_real_log2fc,
)

__all__ = ["QuantileCalibrator", "empirical_real_log2fc"]
