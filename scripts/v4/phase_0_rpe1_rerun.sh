#!/usr/bin/env bash
# Phase 0: re-run RPE1 GIES after the pipeline bug fix.
# The bug-fixed scripts/phase9_rpe1.py now actually fits ElasticNet on RPE1
# ctrl cells and propagates predictions. Cache invalidation already wired.
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1

# Sanity: data must be present.
if [ ! -f "$ROOT/data/rpe1_real.h5ad" ] && [ ! -f "$ROOT/data/rpe1_subset_n200.h5ad" ]; then
    echo "  WARN: RPE1 data not found locally. Will attempt Modal download."
    # Try to pull from Modal volume
    modal volume get ccbench-data raw/rpe1_real.h5ad "$ROOT/data/rpe1_real.h5ad" 2>&1 | tail -3 || true
fi
if [ ! -f "$ROOT/data/rpe1_real.h5ad" ]; then
    echo "  ERROR: RPE1 data unavailable. Phase 0 cannot run."
    exit 1
fi

# Activate venv if available
if [ -f "$ROOT/.venv_gears/bin/activate" ]; then
    source "$ROOT/.venv_gears/bin/activate"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ROOT"

python3 "$ROOT/scripts/phase9_rpe1.py" 2>&1 | tail -100

# Read the corrected JSON
python3 - "$ROOT/results/phase9_rpe1/phase9_results.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
prec = d.get("gies",{}).get("elastic_net_precision",-1)
print(f"  RPE1 corrected ElasticNet precision: {prec:.4f}")
if prec >= 0.30:
    print("  DECISION Δ2.1: RPE1 cross-context HOLDS. Keep §5.7 expanded.")
elif prec >= 0.10:
    print("  DECISION Δ2.1: RPE1 cross-context WEAK. Restructure §5.7 to 'baseline survives weakly'.")
else:
    print("  DECISION Δ2.1: RPE1 cross-context FAILS. Drop §5.7; lean on Norman.")
PY
