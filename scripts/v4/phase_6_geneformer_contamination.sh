#!/usr/bin/env bash
# Phase 6: Geneformer contamination via cell-line holdout + perturbation holdout.
# Per V4 audit H16: pivoted from "find a post-Geneformer-V2 dataset" (infeasible)
# to "compare K562 vs RPE1 vs held-out 10% of K562 perturbations".
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1
if [ -f "$ROOT/.venv_gears/bin/activate" ]; then source "$ROOT/.venv_gears/bin/activate"; fi
export PYTHONPATH="$ROOT"

mkdir -p "$ROOT/results/geneformer_contamination"

python3 - <<'PY' || true
import json, os
from pathlib import Path
import numpy as np

root = Path("/Users/aayanalwani/virtual cells")
out = {
    "cell_line_holdout": {},
    "perturbation_holdout": {},
}

# Cell-line holdout: compare Geneformer F1 K562 vs RPE1
k562_multi = root / "results" / "multi_seed" / "aggregated_results.json"
if k562_multi.exists():
    with open(k562_multi) as f:
        d = json.load(f)
    out["cell_line_holdout"]["k562_geneformer_f1"] = (
        d.get("aggregated", {}).get("geneformer_real", {}).get("f1", {}).get("mean")
    )

rpe1_multi = root / "results" / "multi_seed" / "aggregated_rpe1.json"
if rpe1_multi.exists():
    with open(rpe1_multi) as f:
        d = json.load(f)
    gf = d.get("models", {}).get("geneformer")
    if gf:
        out["cell_line_holdout"]["rpe1_geneformer_f1"] = gf.get("f1") or gf.get("state",{}).get("f1")

# Perturbation holdout: requires re-running Geneformer ISP on held-out 10%
# Stub: assumes Phase 2 deliverables include per-pert F1 breakdown.
print(json.dumps(out, indent=2))

with open(root / "results" / "geneformer_contamination" / "summary.json", "w") as f:
    json.dump(out, f, indent=2)
print("  Wrote results/geneformer_contamination/summary.json")
PY
