#!/usr/bin/env bash
# Auto-resume the remaining Phase-2/3 jobs once the Modal spend cap lifts.
# GEARS now has the cell-cap fix (no GO-graph hang). Long cap-retry window
# (~2h/job) so it self-completes when the limit is raised. Geneformer on new
# datasets is intentionally skipped (similarity-heuristic, low value).
set -u
cd "/Users/aayanalwani/virtual cells" || exit 1
M="python3 -m modal run --detach"

launch_retry () {  # $1=human name, $2=done-check-cmd, $3=launch-cmd
    local name="$1" check="$2" cmd="$3"
    if eval "$check" 2>/dev/null | grep -q FOUND; then echo "[$(date +%H:%M)] $name present — skip"; return 0; fi
    for t in $(seq 1 80); do   # ~2h of cap-retry at 90s
        local out; out=$(eval "$cmd" 2>&1)
        if echo "$out" | grep -qiE 'spawned|Spawned function'; then echo "[$(date +%H:%M)] $name launched"; return 0; fi
        echo "$out" | grep -qi 'spend limit' && { echo "[$(date +%H:%M)] $name capped (try $t)"; sleep 90; } || { echo "[$(date +%H:%M)] $name err"; sleep 60; }
    done
    echo "[$(date +%H:%M)] $name gave up"; return 1
}
wait_file () {  # $1=name $2=volpath
    for i in $(seq 1 90); do
        modal volume ls ccbench-data "$(dirname $2)" 2>/dev/null | grep -q "$(basename $2)" && { echo "[$(date +%H:%M)] $1 DONE"; return 0; }
        sleep 100
    done; echo "[$(date +%H:%M)] $1 wait-timeout"; return 1
}

launch_retry gears_rpe1 "modal volume ls ccbench-data results | grep -q gears_rpe1_predictions && echo FOUND" \
    "$M scripts/modal_run_gears_real.py --dataset rpe1 --gene-list data/gene_lists/rpe1_genes.txt --epochs 20"
wait_file gears_rpe1 results/gears_rpe1_predictions.h5ad
launch_retry gears_norman "modal volume ls ccbench-data results | grep -q gears_norman_predictions && echo FOUND" \
    "$M scripts/modal_run_gears_real.py --dataset norman --gene-list data/gene_lists/norman_genes.txt --epochs 20"
wait_file gears_norman results/gears_norman_predictions.h5ad
launch_retry state123 "modal volume ls ccbench-data results/model_outputs_real_n200_state | grep -q seed123 && echo FOUND" \
    "$M scripts/modal_run_state_v3.py --seed 123 --n-cells-per-pert 1 --n-ctrl-keep 120"
launch_retry state456 "modal volume ls ccbench-data results/model_outputs_real_n200_state | grep -q seed456 && echo FOUND" \
    "$M scripts/modal_run_state_v3.py --seed 456 --n-cells-per-pert 1 --n-ctrl-keep 120"
wait_file state123 results/model_outputs_real_n200_state/state_predictions_seed123.h5ad
wait_file state456 results/model_outputs_real_n200_state/state_predictions_seed456.h5ad
echo "[$(date +%H:%M)] remaining jobs done — running finalize"
bash scripts/v4/finalize_phase2_3.sh
