#!/usr/bin/env bash
# Wait for the 3 STATE 1200-cell (n=5/pert) runs, then eval + aggregate the
# tighter multi-seed CI. Long-running (~10h).
set -u
cd "/Users/aayanalwani/virtual cells" || exit 1
PY="/Users/aayanalwani/virtual cells/.venv_gears/bin/python"; [ -x "$PY" ] || PY=python3
SD=results/model_outputs_real_n200_state
mkdir -p "$SD"
for i in $(seq 1 140); do
  st=$(modal volume ls ccbench-data results/model_outputs_real_n200_state 2>/dev/null)
  n=0; for s in 42 123 456; do echo "$st" | grep -q "seed${s}_n5" && n=$((n+1)); done
  echo "[$(date +%H:%M)] state_n5: $n/3"
  [ "$n" -ge 3 ] && break
  sleep 300
done
echo "[$(date +%H:%M)] evaluating 1200-cell STATE seeds"
for s in 42 123 456; do
  dst="$SD/state_predictions_seed${s}_n5.h5ad"
  modal volume get --force ccbench-data "results/model_outputs_real_n200_state/state_predictions_seed${s}_n5.h5ad" "$dst" 2>/dev/null
  [ -f "$dst" ] && "$PY" scripts/state_eval_gies.py --seed "$s" --pred-path "$dst" 2>&1 | grep -iE 'STATE:|Wrote' | tail -2
done
"$PY" scripts/v4/phase_3_state_aggregate.py 2>&1 | tail -6
echo "STATE-1200 FINALIZE COMPLETE"
