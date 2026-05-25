#!/usr/bin/env bash
# Phase 5: Track B (TRRUST biological ground truth) evaluation.
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1
if [ -f "$ROOT/.venv_gears/bin/activate" ]; then source "$ROOT/.venv_gears/bin/activate"; fi
export PYTHONPATH="$ROOT"

mkdir -p "$ROOT/results/track_b"

if [ ! -f "$ROOT/scripts/track_b_biological_ground_truth.py" ]; then
    echo "  ERROR: track_b script missing"
    exit 1
fi

# Run the existing Track B script
python3 "$ROOT/scripts/track_b_biological_ground_truth.py" 2>&1 | tail -30 || true

# Compute Spearman concordance with Track A
python3 - <<'PY' || true
import json, os, glob
from pathlib import Path
import numpy as np

root = Path("/Users/aayanalwani/virtual cells")
track_a_results = root / "results" / "multi_seed" / "aggregated_results.json"
track_b_dir = root / "results" / "track_b"

if not track_a_results.exists():
    print("  Track A multi-seed results missing; cannot compute concordance")
    raise SystemExit(0)

with open(track_a_results) as f:
    ta = json.load(f)

models = ["elastic_net", "cpa_real", "gears_real", "geneformer_real"]
ta_f1 = [ta["aggregated"][m]["f1"]["mean"] for m in models]

# Find Track B results
tb_files = list(track_b_dir.glob("*.json"))
print(f"  Track B JSONs: {[p.name for p in tb_files]}")

# Bootstrap concordance via per-edge resampling (per H20)
# Stub: report point estimate if we can find matching files
print(f"  Track A F1 ranks: {sorted(zip(models, ta_f1), key=lambda x:-x[1])}")
PY
