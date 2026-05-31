#!/usr/bin/env bash
# Run the Phase 2 deep-model jobs ONE AT A TIME to avoid GPU-capacity preemption
# (6 concurrent GPU requests thrashed the scheduler). Each job is spawn+detach;
# we poll the Modal volume for its prediction file before launching the next.
set -u
cd "/Users/aayanalwani/virtual cells" || exit 1
MODAL="python3 -m modal run --detach"

run_job () {
    local name="$1"; shift
    local cmd="$*"
    if modal volume ls ccbench-data results 2>/dev/null | grep -q "${name}_predictions.h5ad"; then
        echo "[$(date +%H:%M)] $name already present — skip"
        return 0
    fi
    echo "[$(date +%H:%M)] launching $name"
    eval "$cmd" 2>&1 | grep -iE 'spawned|error' | tail -1
    # poll up to ~110 min for the prediction file
    for i in $(seq 1 66); do
        sleep 100
        if modal volume ls ccbench-data results 2>/dev/null | grep -q "${name}_predictions.h5ad"; then
            echo "[$(date +%H:%M)] $name DONE (after ~$((i*100/60)) min)"
            return 0
        fi
    done
    echo "[$(date +%H:%M)] $name TIMEOUT — moving on"
    return 1
}

run_job gears_rpe1     "$MODAL scripts/modal_run_gears_real.py --dataset rpe1 --gene-list data/gene_lists/rpe1_genes.txt --epochs 20"
run_job cpa_rpe1       "$MODAL scripts/modal_run_cpa_real.py --dataset rpe1 --mode train --n-genes 0"
run_job geneformer_rpe1 "$MODAL scripts/modal_run_geneformer_real.py --dataset rpe1 --gene-list data/gene_lists/rpe1_genes.txt"
run_job gears_norman   "$MODAL scripts/modal_run_gears_real.py --dataset norman --gene-list data/gene_lists/norman_genes.txt --epochs 20"
run_job geneformer_norman "$MODAL scripts/modal_run_geneformer_real.py --dataset norman --gene-list data/gene_lists/norman_genes.txt"

echo "[$(date +%H:%M)] SERIAL PHASE 2 COMPLETE"
modal volume ls ccbench-data results 2>/dev/null | grep -iE 'rpe1|norman'
