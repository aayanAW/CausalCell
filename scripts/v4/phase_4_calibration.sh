#!/usr/bin/env bash
# Phase 4: Calibration as diagnostic. 3 modes + 6-order ablation + 80/20 train/test split.
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1
if [ -f "$ROOT/.venv_gears/bin/activate" ]; then source "$ROOT/.venv_gears/bin/activate"; fi
export OMP_NUM_THREADS=1 PYTHONPATH="$ROOT"

# Generate Mode 2 + Mode 3 if not present
mkdir -p "$ROOT/causalcellbench/calibration"

# Calibration modules are committed to the repo (causalcellbench/calibration/).
# Do not regenerate them here: the former heredoc fallback wrote broken V3 stubs
# (wrong module name + wrong fit/transform signatures) that overwrote the good V4
# modules whenever any one file was missing. Hard-fail instead.
for m in pipeline covariance_preservation sparsity_injection __init__; do
    f="$ROOT/causalcellbench/calibration/${m}.py"
    [ -f "$f" ] || { echo "ERROR: missing $f -- restore with: git checkout causalcellbench/calibration/"; exit 1; }
done
echo "  Calibration modules present."

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
