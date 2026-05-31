#!/usr/bin/env bash
# Phase 2 (un-deferred): GIES eval of deep-model predictions on Norman + RPE1.
#
# Prereq: deep-model predictions downloaded from Modal into
#   data/model_outputs_real_norman/{gears,cpa,geneformer}_predictions.h5ad
#   data/model_outputs_real_rpe1/{gears,cpa,geneformer}_predictions.h5ad
# (download_phase2_predictions.sh handles the rename from
#  {model}_{dataset}_predictions.h5ad on the Modal volume.)
#
# Runs the same local GIES pipeline used for K562, with --pert-col perturbation
# and the canonical gene list so the real-data ground truth and the model
# predictions are scored on an identical gene subset.
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1
PY="$ROOT/.venv_gears/bin/python"
[ -x "$PY" ] || PY=python3

run_one () {
    local ds="$1" data_path="$2"
    local genes="data/gene_lists/${ds}_genes.txt"
    local outdir="data/model_outputs_real_${ds}"
    echo "=================================================================="
    echo " Phase 2 eval: ${ds}"
    echo "=================================================================="
    if [ ! -f "$genes" ]; then echo "  MISSING gene list $genes"; return 1; fi
    "$PY" scripts/run_eval_local.py \
        --data-path "$data_path" \
        --pert-col perturbation \
        --gene-list "$genes" \
        --partition-size 20 \
        --n-cells 100 \
        --n-ctrl 200 \
        --seed 42 \
        --model-outputs-dir "$outdir" \
        --gies-cache-dir "results/gies_cache_${ds}" \
        2>&1 | tee "results/phase2_eval_${ds}.log"
    # run_eval_local writes results/eval_results_n<N>.json; copy to a stable name
    local latest
    latest=$(ls -t results/eval_results_n*.json 2>/dev/null | head -1)
    [ -n "$latest" ] && cp "$latest" "results/phase2_eval_${ds}.json" \
        && echo "  -> results/phase2_eval_${ds}.json"
}

run_one rpe1   "data/rpe1_subset_n200_slim.h5ad"
run_one norman "data/norman_subset_n200_slim.h5ad"

echo "Done. See results/phase2_eval_{rpe1,norman}.json"
