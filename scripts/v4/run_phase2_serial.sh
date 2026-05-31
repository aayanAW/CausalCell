#!/usr/bin/env bash
# Cap-aware serial Phase-2 driver: runs the 5 deep-model jobs ONE AT A TIME.
# Each launch retries on the Modal billing-cycle spend-limit error (handles
# limit-raise propagation delay), then polls the volume for the prediction
# before starting the next job. One GPU active at a time -> no scheduler thrash.
set -u
cd "/Users/aayanalwani/virtual cells" || exit 1
MODAL="python3 -m modal run --detach"

run_job () {
    local name="$1"; shift
    local cmd="$*"
    if modal volume ls ccbench-data results 2>/dev/null | grep -q "${name}_predictions.h5ad"; then
        echo "[$(date +%H:%M)] $name already present — skip"; return 0
    fi
    # launch with cap-retry (up to ~30 min of propagation wait)
    local launched=0
    for t in $(seq 1 20); do
        local out; out=$(eval "$cmd" 2>&1)
        if echo "$out" | grep -qi 'spawned'; then
            echo "[$(date +%H:%M)] $name launched: $(echo "$out" | grep -i spawned | head -1)"
            launched=1; break
        elif echo "$out" | grep -qi 'spend limit'; then
            echo "[$(date +%H:%M)] $name capped (try $t) — wait 90s"; sleep 90
        else
            echo "[$(date +%H:%M)] $name launch error: $(echo "$out" | grep -iE 'error' | head -1)"; sleep 30
        fi
    done
    [ "$launched" = 1 ] || { echo "[$(date +%H:%M)] $name FAILED to launch"; return 1; }
    # poll for prediction (~110 min)
    for i in $(seq 1 66); do
        sleep 100
        if modal volume ls ccbench-data results 2>/dev/null | grep -q "${name}_predictions.h5ad"; then
            echo "[$(date +%H:%M)] $name DONE"; return 0
        fi
    done
    echo "[$(date +%H:%M)] $name TIMEOUT"; return 1
}

run_job gears_rpe1      "$MODAL scripts/modal_run_gears_real.py --dataset rpe1 --gene-list data/gene_lists/rpe1_genes.txt --epochs 20"
run_job cpa_rpe1        "$MODAL scripts/modal_run_cpa_real.py --dataset rpe1 --mode train --n-genes 0"
run_job geneformer_rpe1 "$MODAL scripts/modal_run_geneformer_real.py --dataset rpe1 --gene-list data/gene_lists/rpe1_genes.txt"
run_job gears_norman    "$MODAL scripts/modal_run_gears_real.py --dataset norman --gene-list data/gene_lists/norman_genes.txt --epochs 20"
run_job geneformer_norman "$MODAL scripts/modal_run_geneformer_real.py --dataset norman --gene-list data/gene_lists/norman_genes.txt"
echo "[$(date +%H:%M)] SERIAL PHASE 2 COMPLETE"
modal volume ls ccbench-data results 2>/dev/null | grep -iE 'rpe1|norman'
