#!/usr/bin/env bash
# Phase 3: STATE multi-seed on K562 at 1200 cells.
# Launches Modal STATE function with 24h timeout, checkpoint per 100 cells.
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT"

if ! command -v modal &>/dev/null; then
    echo "  ERROR: modal CLI not in PATH"
    exit 1
fi

# Launch detached STATE embedding job at 1200 cells
out_dir="$ROOT/data/model_outputs_real_n200_state_multiseed"
mkdir -p "$out_dir"
if [ -f "$out_dir/state_predictions_1200.h5ad" ]; then
    echo "  STATE 1200-cell embeddings cached, skipping launch"
else
    echo "  Launching STATE 1200-cell embedding (detached, 24h timeout)..."
    nohup modal run --detach "$ROOT/scripts/modal_run_state_v3.py" \
        --n-cells-per-pert 5 --n-ctrl 200 \
        > "$ROOT/logs/phase3_state.log" 2>&1 &
    disown
fi

# Poll for completion
MAX_WAIT_HR=20
WAITED=0
POLL=600  # 10 min
while [ $WAITED -lt $((MAX_WAIT_HR*3600)) ]; do
    modal volume get ccbench-data \
        "state_cache/state_predictions_1200.h5ad" \
        "$out_dir/state_predictions_1200.h5ad" 2>/dev/null || true
    if [ -f "$out_dir/state_predictions_1200.h5ad" ]; then break; fi
    sleep $POLL
    WAITED=$((WAITED + POLL))
    echo "  Phase 3 poll @ $((WAITED/60))m: waiting for STATE embedding..."
done

# Run GIES on 3 seeds against the 1200-cell predictions
if [ -f "$out_dir/state_predictions_1200.h5ad" ]; then
    for seed in 42 123 456; do
        out="$ROOT/results/state/state_multiseed_s${seed}.json"
        if [ -f "$out" ]; then continue; fi
        python3 "$ROOT/scripts/state_eval_gies.py" \
            --predictions "$out_dir/state_predictions_1200.h5ad" \
            --seed "$seed" \
            --output "$out" 2>&1 | tail -5 || true
    done
    # Aggregate
    python3 - "$ROOT" <<'PY' || true
import json, sys, os, statistics
root = sys.argv[1]
f1s = []
for seed in (42, 123, 456):
    p = os.path.join(root, "results", "state", f"state_multiseed_s{seed}.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        f1s.append(d.get("state", {}).get("f1") or d.get("f1"))
agg = {
    "n_seeds": len(f1s),
    "seeds": [42,123,456][:len(f1s)],
    "per_seed_f1": f1s,
    "f1_mean": statistics.mean(f1s) if f1s else None,
    "f1_std": statistics.stdev(f1s) if len(f1s)>1 else None,
}
out = os.path.join(root, "results", "state", "state_multiseed_f1.json")
with open(out, "w") as f:
    json.dump(agg, f, indent=2)
print(f"  STATE multi-seed: n={agg['n_seeds']}, mean={agg['f1_mean']}, std={agg['f1_std']}")
PY
fi
