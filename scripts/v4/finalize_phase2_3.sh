#!/usr/bin/env bash
# One-shot finalizer for Phase 2 (deep models on RPE1+Norman) and Phase 3
# (STATE multi-seed). Run AFTER predictions land on the Modal volume. Idempotent:
# skips eval steps whose inputs are missing, so it can be re-run as jobs finish.
set -u
cd "/Users/aayanalwani/virtual cells" || exit 1
PY="/Users/aayanalwani/virtual cells/.venv_gears/bin/python"; [ -x "$PY" ] || PY="python3"
VOL="ccbench-data"

echo "==================== 1. Download Phase-2 predictions ===================="
for ds in rpe1 norman; do
    outdir="data/model_outputs_real_${ds}"; mkdir -p "$outdir"
    for m in gears cpa geneformer; do
        dst="${outdir}/${m}_predictions.h5ad"
        [ -f "$dst" ] && { echo "  have $dst"; continue; }
        modal volume get --force "$VOL" "results/${m}_${ds}_predictions.h5ad" "$dst" 2>/dev/null \
            && echo "  pulled ${m}_${ds}" || echo "  (missing on volume: ${m}_${ds})"
    done
done

echo "==================== 2. Phase-2 GIES eval (per dataset) ================="
run_eval () {
    local ds="$1" data="$2" pert_col="$3"
    [ -f "$data" ] || { echo "  SKIP $ds — no $data"; return; }
    "$PY" scripts/run_eval_local.py --data-path "$data" --pert-col "$pert_col" \
        --gene-list "data/gene_lists/${ds}_genes.txt" --partition-size 20 \
        --n-cells 100 --n-ctrl 200 --seed 42 \
        --model-outputs-dir "data/model_outputs_real_${ds}" \
        --gies-cache-dir "results/gies_cache_${ds}" 2>&1 | tail -20
    local latest; latest=$(ls -t results/eval_results_n*.json 2>/dev/null | head -1)
    [ -n "$latest" ] && cp "$latest" "results/phase2_eval_${ds}.json" && echo "  -> phase2_eval_${ds}.json"
}
run_eval rpe1   "data/rpe1_subset_n200_slim.h5ad"   gene
run_eval norman "data/norman_subset_n200_slim.h5ad" perturbation

echo "==================== 3. Cross-dataset aggregate ========================"
"$PY" scripts/v4/phase_2_aggregate.py 2>&1 | tail -15

echo "==================== 4. Phase-6 contamination probe ===================="
"$PY" scripts/v4/phase_6_contamination.py 2>&1 | tail -15

echo "==================== 5. STATE multi-seed eval =========================="
mkdir -p data/model_outputs_real_n200_state
for s in 42 123 456; do
    src="results/model_outputs_real_n200_state/state_predictions_seed${s}.h5ad"
    dst="data/model_outputs_real_n200_state/state_predictions_seed${s}.h5ad"
    [ -f "$dst" ] || modal volume get --force "$VOL" "$src" "$dst" 2>/dev/null \
        && echo "  have state seed${s}" || echo "  (missing state seed${s})"
    if [ -f "$dst" ]; then
        "$PY" scripts/state_eval_gies.py --seed "$s" --pred-path "$dst" 2>&1 | grep -iE 'STATE:|Wrote' | tail -2
    fi
done
"$PY" scripts/v4/phase_3_state_aggregate.py 2>&1 | tail -6

echo "==================== FINALIZE COMPLETE ================================="
