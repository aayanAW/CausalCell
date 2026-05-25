#!/usr/bin/env bash
# Phase 2: train GEARS, CPA, Geneformer on RPE1 + Norman via Modal.
# Each model fitted on each dataset → predictions h5ad → GIES eval.
# Modal jobs are detached by design; this script launches them and polls.
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT"

# Pre-flight: Modal CLI available?
if ! command -v modal &>/dev/null; then
    echo "  ERROR: modal CLI not in PATH"
    exit 1
fi

# Pre-flight: confirm Modal volume contains data
modal volume ls ccbench-data raw 2>&1 | grep -E 'rpe1|norman' || {
    echo "  WARN: rpe1/norman data not on ccbench-data volume. Aborting Phase 2."
    exit 1
}

# Launch matrix: 3 models × 2 datasets = 6 Modal jobs
DATASETS=(rpe1 norman2019)
MODELS=(gears cpa geneformer)
mkdir -p "$ROOT/data/model_outputs_phase2"

LAUNCHED=0
SKIPPED=0
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        out_dir="$ROOT/data/model_outputs_phase2/${model}_${dataset}"
        if [ -f "$out_dir/predictions.h5ad" ]; then
            echo "  ${model}/${dataset}: cached predictions exist, skip"
            SKIPPED=$((SKIPPED+1))
            continue
        fi
        # Use the unified runner if it exists; otherwise the per-model scripts.
        runner="$ROOT/scripts/modal_runner.py"
        if [ ! -f "$runner" ]; then
            runner="$ROOT/scripts/modal_run_${model}_real.py"
        fi
        if [ ! -f "$runner" ]; then
            echo "  ${model}/${dataset}: no runner script found, skip"
            continue
        fi
        echo "  Launching ${model}/${dataset} via Modal (detached)..."
        # Modal launch detached
        nohup modal run --detach "$runner" \
            --model "$model" --dataset "$dataset" \
            > "$ROOT/logs/phase2_${model}_${dataset}.log" 2>&1 &
        disown
        LAUNCHED=$((LAUNCHED+1))
        sleep 5  # stagger
    done
done
echo "  Phase 2 launched: $LAUNCHED jobs, skipped $SKIPPED"

# Wait for completion (polling)
# Each Modal job writes its predictions to the Modal volume; we pull on completion.
MAX_WAIT_HR=12
WAIT_SECS=$((MAX_WAIT_HR * 3600))
WAITED=0
POLL=300  # 5 min
while [ $WAITED -lt $WAIT_SECS ]; do
    ALL_DONE=1
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            out_dir="$ROOT/data/model_outputs_phase2/${model}_${dataset}"
            if [ ! -f "$out_dir/predictions.h5ad" ]; then
                # Try pulling from Modal volume
                mkdir -p "$out_dir"
                modal volume get ccbench-data \
                    "phase2/${model}_${dataset}/predictions.h5ad" \
                    "$out_dir/predictions.h5ad" 2>/dev/null || true
                if [ ! -f "$out_dir/predictions.h5ad" ]; then
                    ALL_DONE=0
                fi
            fi
        done
    done
    if [ $ALL_DONE -eq 1 ]; then break; fi
    sleep $POLL
    WAITED=$((WAITED + POLL))
    echo "  Phase 2 poll @ $((WAITED/60))m: waiting for jobs..."
done

# Run GIES on each set of predictions
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        out_dir="$ROOT/data/model_outputs_phase2/${model}_${dataset}"
        if [ -f "$out_dir/predictions.h5ad" ]; then
            echo "  Running GIES on ${model}/${dataset}"
            python3 "$ROOT/scripts/state_eval_gies.py" \
                --predictions "$out_dir/predictions.h5ad" \
                --dataset "$dataset" \
                --model "$model" \
                --output "$ROOT/results/phase2/${model}_${dataset}_f1.json" 2>&1 | tail -5 || \
                echo "  ${model}/${dataset}: GIES eval failed"
        fi
    done
done

# Aggregate per-dataset across models
python3 - "$ROOT" <<'PY' || true
import json, sys, os
root = sys.argv[1]
for dataset in ("rpe1", "norman2019"):
    agg = {"dataset": dataset, "models": {}}
    for model in ("gears", "cpa", "geneformer"):
        p = os.path.join(root, "results", "phase2", f"{model}_{dataset}_f1.json")
        if os.path.exists(p):
            with open(p) as f:
                agg["models"][model] = json.load(f)
    out = os.path.join(root, "results", "multi_seed", f"aggregated_{dataset}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"  Aggregated {dataset}: {len(agg['models'])} models")
PY
