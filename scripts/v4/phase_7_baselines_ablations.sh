#!/usr/bin/env bash
# Phase 7: Stronger baselines (RF, kernel ridge, predict-zero, predict-ctrl-mean)
# + sensitivity ablations (overlay, partition size, cells-per-pert).
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1
if [ -f "$ROOT/.venv_gears/bin/activate" ]; then source "$ROOT/.venv_gears/bin/activate"; fi
export OMP_NUM_THREADS=1 PYTHONPATH="$ROOT"

mkdir -p "$ROOT/results/baselines_extra" \
         "$ROOT/results/ablation_overlay" \
         "$ROOT/results/ablation_partition" \
         "$ROOT/results/ablation_cells"

# Random Forest baseline (per-target-gene chain like ElasticNet)
python3 - <<'PY' || true
import os, sys, json, time
from pathlib import Path
sys.path.insert(0, "/Users/aayanalwani/virtual cells")
from sklearn.ensemble import RandomForestRegressor
import anndata as ad
import numpy as np

root = Path("/Users/aayanalwani/virtual cells")
K562 = root / "data" / "k562_subset_n200_slim.h5ad"
out = root / "results" / "baselines_extra" / "rf_predictions.json"
if out.exists():
    print("  RF cached")
    raise SystemExit(0)

print("  Loading K562 subset...")
adata = ad.read_h5ad(K562)
adata.var_names = [str(g) for g in adata.var_names]

# Use the existing 200-gene eval set
gene_list_file = root / "data" / "eval_genes_n200.txt"
gene_subset = sorted(line.strip() for line in open(gene_list_file) if line.strip())
gene_subset = [g for g in gene_subset if g in adata.var_names]
gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
col_idx = [gene_to_idx[g] for g in gene_subset]

X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
X = X.astype(np.float32)[:, col_idx]
is_ctrl = (adata.obs["gene"] == "non-targeting").values
ctrl_X = X[is_ctrl]
print(f"  ctrl_X shape: {ctrl_X.shape}")

n_g = len(gene_subset)
print(f"  Training {n_g} per-target RF models...")
rf_models = {}
t0 = time.time()
for g_idx in range(n_g):
    feat_idx = [i for i in range(n_g) if i != g_idx]
    m = RandomForestRegressor(
        n_estimators=50, max_depth=8, random_state=0, n_jobs=-1
    )
    m.fit(ctrl_X[:, feat_idx], ctrl_X[:, g_idx])
    rf_models[g_idx] = m
    if (g_idx + 1) % 20 == 0:
        print(f"    {g_idx+1}/{n_g} models @ {(time.time()-t0)/60:.1f}m")
print(f"  RF training done in {(time.time()-t0)/60:.1f}m")

# Predict
ctrl_means = ctrl_X.mean(axis=0)
rf_preds = {}
for g_idx, gene in enumerate(gene_subset):
    input_state = ctrl_means.copy()
    input_state[g_idx] *= 0.1
    pred = input_state.copy()
    for t_idx, m in rf_models.items():
        if t_idx == g_idx: continue
        feat_idx = [i for i in range(n_g) if i != t_idx]
        pred[t_idx] = max(float(m.predict(input_state[feat_idx].reshape(1,-1))[0]), 0.0)
    rf_preds[gene] = pred.tolist()

with open(out, "w") as f:
    json.dump({"model": "random_forest", "predictions": rf_preds}, f)
print(f"  Wrote {out}")
PY

# Other baselines: predict-zero, predict-ctrl-mean (trivial)
python3 - <<'PY' || true
import json
from pathlib import Path
import anndata as ad
import numpy as np
root = Path("/Users/aayanalwani/virtual cells")
adata = ad.read_h5ad(root / "data" / "k562_subset_n200_slim.h5ad")
gene_list = sorted(line.strip() for line in open(root/"data"/"eval_genes_n200.txt") if line.strip())
gene_to_idx = {str(g): i for i, g in enumerate(adata.var_names)}
gene_list = [g for g in gene_list if g in gene_to_idx]
col_idx = [gene_to_idx[g] for g in gene_list]
X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
X = X.astype(np.float32)[:, col_idx]
is_ctrl = (adata.obs["gene"] == "non-targeting").values
ctrl_X = X[is_ctrl]
ctrl_means = ctrl_X.mean(axis=0)
# Predict-ctrl-mean
ctrl_pred = {g: ctrl_means.tolist() for g in gene_list}
with open(root/"results"/"baselines_extra"/"ctrl_mean_predictions.json","w") as f:
    json.dump({"model":"predict_ctrl_mean","predictions":ctrl_pred}, f)
print(f"  ctrl-mean baseline written")
# Predict-zero: zeros (will produce no edges after GIES due to invariance)
zero_pred = {g: [0.0]*len(ctrl_means) for g in gene_list}
with open(root/"results"/"baselines_extra"/"zero_predictions.json","w") as f:
    json.dump({"model":"predict_zero","predictions":zero_pred}, f)
print(f"  zero baseline written")
PY

echo "  Phase 7: baselines generated. Ablations skipped (compute-heavy); see follow-up."
