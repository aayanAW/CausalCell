#!/usr/bin/env bash
# Phase 2 DEFERRED: deep models on RPE1 + Norman.
# The existing scripts/modal_run_{gears,cpa,geneformer}_real.py are hardcoded
# for the Replogle K562 dataset. Running them on RPE1 or Norman requires a
# proper modal_runner.py that takes --dataset as a first-class argument and
# routes to the appropriate Modal volume path + AnnData column mapping.
#
# This is real engineering work (~1-2 days) and is left as a follow-up. The
# orchestrator emits this notice and continues so the rest of the pipeline
# (Phase 4-11) can complete tonight.
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1

mkdir -p "$ROOT/data/model_outputs_phase2"
cat > "$ROOT/data/model_outputs_phase2/DEFERRED.md" <<'MD'
Phase 2 deferred. See scripts/v4/phase_2_deep_rpe1_norman.sh for rationale.
Follow-up: build scripts/modal_runner.py that accepts --model + --dataset,
then re-launch with:

  bash scripts/v4/phase_2_deep_rpe1_norman.sh
MD
echo "  PHASE 2 DEFERRED: see data/model_outputs_phase2/DEFERRED.md"
exit 0
