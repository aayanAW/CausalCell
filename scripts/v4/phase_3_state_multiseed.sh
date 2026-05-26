#!/usr/bin/env bash
# Phase 3 DEFERRED: STATE multi-seed at 1200 cells.
# The existing scripts/modal_run_state_v3.py works but doesn't accept the
# --n-cells-per-pert override that the V4 plan calls for. Multi-seed evaluation
# also requires running state_eval_gies.py with --seed support (not currently
# implemented). This is left as a follow-up.
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1

mkdir -p "$ROOT/results/state"
cat > "$ROOT/results/state/PHASE3_DEFERRED.md" <<'MD'
Phase 3 deferred. See scripts/v4/phase_3_state_multiseed.sh for rationale.
Follow-up: extend scripts/modal_run_state_v3.py to accept --n-cells-per-pert
and update scripts/state_eval_gies.py to accept --seed, then re-launch.
MD
echo "  PHASE 3 DEFERRED: see results/state/PHASE3_DEFERRED.md"
exit 0
