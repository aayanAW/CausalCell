#!/usr/bin/env bash
# Kickoff: wait until the background run_multi_seed.py finishes,
# then launch orchestrate.sh. Designed for tmux + caffeinate detach.
set -u
ROOT="/Users/aayanalwani/virtual cells"
LOG="$ROOT/logs/v4_kickoff.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "================================================================"
echo "V4 KICKOFF — waiting for background multi-seed, then orchestrator"
echo "  started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

# Wait for run_multi_seed processes to exit (no-op if none active)
while pgrep -f 'run_multi_seed' >/dev/null 2>&1; do
    n_seeds=$(ls "$ROOT/results/multi_seed/" | grep -c 'eval_results_n200_s.*json' || echo 0)
    echo "  [$(date +%H:%M:%S)] multi-seed bg active — per-seed JSONs so far: $n_seeds / 20"
    sleep 120
done
echo "  [$(date +%H:%M:%S)] no run_multi_seed processes active — proceeding"

# Aggregate if not already
n_seeds=$(ls "$ROOT/results/multi_seed/" | grep -c 'eval_results_n200_s.*json' || echo 0)
echo "  per-seed JSONs: $n_seeds"
if [ "$n_seeds" -lt 20 ]; then
    echo "  Re-running run_multi_seed to fill missing seeds + aggregate"
    cd "$ROOT" || exit 1
    if [ -f "$ROOT/.venv_gears/bin/activate" ]; then source "$ROOT/.venv_gears/bin/activate"; fi
    export OMP_NUM_THREADS=1 PYTHONPATH="$ROOT"
    python3 "$ROOT/scripts/run_multi_seed.py" \
        --n-genes 200 \
        --seeds 42 123 456 789 1024 2048 4096 7919 13337 31337 65537 100003 142857 271828 314159 524287 1000003 1299709 1999993 2147483647 \
        --extra-args="--skip-random --partition-size 20" 2>&1 | tail -50
fi

# Reset state: mark Phase 1 completed so orchestrator skips to Phase 2 (which is deferred → instant skip → Phase 4)
python3 - <<'PY'
import json
p = "/Users/aayanalwani/virtual cells/logs/v4_state.json"
with open(p) as f: s = json.load(f)
if "phase1_multiseed_n20" not in s.get("completed_phases", []):
    s.setdefault("completed_phases", []).append("phase1_multiseed_n20")
if "phase0_rpe1_rerun" not in s.get("failed_phases", []):
    s.setdefault("failed_phases", []).append("phase0_rpe1_rerun")
s["status"] = "resuming"
with open(p, "w") as f: json.dump(s, f, indent=2)
print("  state reset:", json.dumps(s, indent=2))
PY

# Run orchestrator
echo "  Launching orchestrate.sh"
bash "$ROOT/scripts/v4/orchestrate.sh"
echo "  orchestrate.sh exited at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
