#!/usr/bin/env bash
# Download Phase 2 deep-model predictions from the Modal volume and rename them
# into the per-dataset directory layout the local eval expects.
#
# Modal volume writes:   results/{model}_{dataset}_predictions.h5ad
# Eval (load_real_model_predictions) reads:
#                        data/model_outputs_real_{dataset}/{model}_predictions.h5ad
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1
VOL="ccbench-data"

for ds in rpe1 norman; do
    outdir="data/model_outputs_real_${ds}"
    mkdir -p "$outdir"
    for model in gears cpa geneformer; do
        src="results/${model}_${ds}_predictions.h5ad"
        dst="${outdir}/${model}_predictions.h5ad"
        echo "  pulling $src -> $dst"
        modal volume get --force "$VOL" "$src" "$dst" 2>&1 | tail -1 \
            || echo "    (not present yet: $src)"
    done
done
echo "Done."
