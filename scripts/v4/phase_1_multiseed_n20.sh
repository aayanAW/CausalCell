#!/usr/bin/env bash
# Phase 1: bump K562 multi-seed from n=5 to n=20.
# Uses the project's run_multi_seed.py which writes per-seed JSONs + aggregates.
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1
if [ -f "$ROOT/.venv_gears/bin/activate" ]; then source "$ROOT/.venv_gears/bin/activate"; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ROOT"

SEEDS=(42 123 456 789 1024 2048 4096 7919 13337 31337 65537 100003 142857 271828 314159 524287 1000003 1299709 1999993 2147483647)

python3 "$ROOT/scripts/run_multi_seed.py" \
    --n-genes 200 \
    --seeds "${SEEDS[@]}" \
    --extra-args="--skip-random --partition-size 20" 2>&1 | tail -50

# Verify n=20 in JSON
python3 - "$ROOT/results/multi_seed/aggregated_results.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
n_seeds = d.get("n_seeds") or len(d.get("seeds", []))
print(f"  Multi-seed aggregated: n_seeds={n_seeds}")
if n_seeds >= 20:
    print("  PHASE 1 SUCCESS: n_seeds >= 20")
else:
    print(f"  WARN: expected n_seeds >= 20, got {n_seeds}")
PY
