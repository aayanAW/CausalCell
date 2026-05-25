#!/usr/bin/env bash
# Phase 1: bump K562 multi-seed from n=5 to n=20.
# Adds 15 new seeds. Each seed runs the full vanilla eval pipeline.
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1

if [ -f "$ROOT/.venv_gears/bin/activate" ]; then
    source "$ROOT/.venv_gears/bin/activate"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ROOT"

NEW_SEEDS=(2048 4096 7919 13337 31337 65537 100003 142857 271828 314159 524287 1000003 1299709 1999993 2147483647)
for seed in "${NEW_SEEDS[@]}"; do
    cache_dir="$ROOT/results/gies_cache/n200_s${seed}_p20"
    if [ -d "$cache_dir" ] && [ -f "$cache_dir/elastic_net_edges.json" ]; then
        echo "  seed=$seed: cached, skip"
        continue
    fi
    echo "  seed=$seed: running..."
    t0=$(date +%s)
    python3 "$ROOT/scripts/run_eval_local.py" \
        --data-path "$ROOT/data/k562_subset_n200_slim.h5ad" \
        --pre-subset --n-genes 200 --partition-size 20 \
        --n-cells 100 --n-ctrl 200 --seed "$seed" \
        --model-outputs-dir "$ROOT/data/model_outputs_real_n200" \
        --gies-cache-dir "$ROOT/results/gies_cache" \
        --skip-random 2>&1 | tail -5
    t1=$(date +%s)
    echo "  seed=$seed: done in $((t1-t0))s"
done

# Re-aggregate
python3 "$ROOT/scripts/aggregate_multi_seed.py" 2>&1 | tail -20

# Verify n=20 in JSON
python3 - "$ROOT/results/multi_seed/aggregated_results.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
n_seeds = d.get("n_seeds") or len(d.get("seeds", []))
print(f"  Multi-seed aggregated: n_seeds={n_seeds}")
if n_seeds < 20:
    print(f"  WARN: expected n_seeds >= 20, got {n_seeds}")
PY
