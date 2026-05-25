#!/usr/bin/env bash
# V4 master orchestrator.
#
# Runs Phases 1-11 of ARCHITECTURE_V4_AUDIT_AND_UPGRADE.md sequentially.
# Resumes from where it left off via /Users/aayanalwani/virtual\ cells/logs/v4_state.json.
# Designed to run inside tmux + caffeinate so it survives laptop close.
#
# Launch (detached, survives laptop close):
#   caffeinate -di -s tmux new -d -s perturbcausal-v4 \
#     "bash '/Users/aayanalwani/virtual cells/scripts/v4/orchestrate.sh'"
#
# Watch:
#   tmux attach -t perturbcausal-v4   # detach with Ctrl-b d
#   tail -f /Users/aayanalwani/virtual\ cells/logs/v4_orchestrator.log
#   cat /Users/aayanalwani/virtual\ cells/logs/v4_state.json
#
# Stop:
#   tmux kill-session -t perturbcausal-v4

set -u

ROOT="/Users/aayanalwani/virtual cells"
LOG="$ROOT/logs/v4_orchestrator.log"
STATE="$ROOT/logs/v4_state.json"
PHASES_DIR="$ROOT/scripts/v4"

cd "$ROOT" || exit 1
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "================================================================"
echo "PERTURBCAUSAL V4 ORCHESTRATOR — started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  branch: $(git rev-parse --abbrev-ref HEAD)"
echo "  commit: $(git rev-parse --short HEAD)"
echo "  pid:    $$"
echo "  hostname: $(hostname)"
echo "================================================================"

# Initialize state file if missing
if [ ! -f "$STATE" ]; then
    cat > "$STATE" <<'EOF'
{
  "started_at": null,
  "current_phase": null,
  "completed_phases": [],
  "failed_phases": [],
  "status": "initialized"
}
EOF
fi

write_state() {
    local phase="$1"
    local status="$2"
    local now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    python3 - "$STATE" "$phase" "$status" "$now" <<'PY'
import json, sys
state_path, phase, status, now = sys.argv[1:5]
with open(state_path) as f:
    state = json.load(f)
if state.get("started_at") is None:
    state["started_at"] = now
state["current_phase"] = phase if status not in ("completed","failed") else None
state["last_update"] = now
if status == "completed":
    if phase not in state["completed_phases"]:
        state["completed_phases"].append(phase)
elif status == "failed":
    if phase not in state["failed_phases"]:
        state["failed_phases"].append(phase)
state["status"] = status
with open(state_path, "w") as f:
    json.dump(state, f, indent=2)
PY
}

is_completed() {
    local phase="$1"
    python3 -c "
import json
with open('$STATE') as f:
    state = json.load(f)
import sys; sys.exit(0 if '$phase' in state.get('completed_phases',[]) else 1)
"
}

run_phase() {
    local phase_num="$1"
    local phase_name="$2"
    local phase_script="$PHASES_DIR/phase_${phase_num}_${phase_name}.sh"
    local phase_label="phase${phase_num}_${phase_name}"

    echo
    echo "----------------------------------------------------------------"
    echo "[$(date +%H:%M:%S)] PHASE $phase_num: $phase_name"
    echo "----------------------------------------------------------------"

    if is_completed "$phase_label"; then
        echo "  SKIP: already completed"
        return 0
    fi
    if [ ! -f "$phase_script" ]; then
        echo "  STUB: $phase_script does not exist. Skipping."
        return 0
    fi
    write_state "$phase_label" "in_progress"
    set +e
    bash "$phase_script"
    local rc=$?
    set -e
    if [ $rc -eq 0 ]; then
        write_state "$phase_label" "completed"
        echo "  OK ($(date +%H:%M:%S))"
    else
        write_state "$phase_label" "failed"
        echo "  FAILED rc=$rc ($(date +%H:%M:%S))"
        # Decision: continue to next phase even on failure. Failures are logged.
        # Hard-stop on critical failures via env var V4_HARD_STOP_ON_FAILURE=1.
        if [ "${V4_HARD_STOP_ON_FAILURE:-0}" = "1" ]; then
            echo "  Hard-stop triggered. Exiting."
            exit $rc
        fi
    fi
}

# Phase sequence
run_phase 0 rpe1_rerun
run_phase 1 multiseed_n20
run_phase 2 deep_rpe1_norman
run_phase 3 state_multiseed
run_phase 4 calibration
run_phase 5 trackb
run_phase 6 geneformer_contamination
run_phase 7 baselines_ablations
run_phase 8 engineering
run_phase 9 writing
run_phase 10 submission_prep
run_phase 11 reproduction

echo
echo "================================================================"
echo "V4 ORCHESTRATOR finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Final state: $STATE"
cat "$STATE"
echo "================================================================"
