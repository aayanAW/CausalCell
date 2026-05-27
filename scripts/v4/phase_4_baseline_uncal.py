"""Compute vanilla (uncalibrated) F1 baseline on the same 40-pert test set
used by phase_4_calibration_gies.py. Reports per-model F1 to compare against
the calibrated sweep."""
from __future__ import annotations
import json, sys, os, time
from pathlib import Path

ROOT = Path("/Users/aayanalwani/virtual cells")
sys.path.insert(0, str(ROOT))

import numpy as np
import anndata as ad

from scripts.run_eval_local import (
    sample_perturbed_cells_overlay,
    run_gies,
    build_cb_input,
    compute_set_metrics,
)

K562 = ROOT / "data" / "k562_subset_n200_slim.h5ad"
GENE_LIST = ROOT / "data" / "eval_genes_n200.txt"
REAL_EDGES = ROOT / "results" / "gies_cache" / "n200_s42_p20" / "real_data_edges.json"
MANIFEST = ROOT / "results" / "calibration_modes" / "sweep_manifest.json"
OUT = ROOT / "results" / "calibration_modes" / "vanilla_baseline_f1.json"
INPUT_DIR = ROOT / "data" / "model_outputs_real_n200"
SEED = 42
PARTITION = 20
N_PERT_CELLS = 100
N_CTRL = 200


def main():
    manifest = json.load(open(MANIFEST))
    test_perts = set(manifest["test_perts"])
    real_edges = set(tuple(e) for e in json.load(open(REAL_EDGES))["edges"])
    print(f"  Test perts: {len(test_perts)}, real edges: {len(real_edges)}")

    adata = ad.read_h5ad(K562)
    adata.var_names = [str(g) for g in adata.var_names]
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float64)
    is_ctrl = (adata.obs["gene"] == "non-targeting").values
    gene_subset = sorted(line.strip() for line in open(GENE_LIST) if line.strip())
    g2i = {str(g): i for i, g in enumerate(adata.var_names)}
    gene_subset = [g for g in gene_subset if g in g2i]
    col_idx = [g2i[g] for g in gene_subset]
    ctrl_X_eval = X[is_ctrl][:, col_idx]
    print(f"  ctrl_X_eval shape: {ctrl_X_eval.shape}")

    out = {}
    for model in ("gears", "cpa", "geneformer"):
        p = INPUT_DIR / f"{model}_predictions.h5ad"
        a = ad.read_h5ad(p)
        a.var_names = [str(g) for g in a.var_names]
        Xp = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        Xp = np.maximum(Xp.astype(np.float64), 0.0)
        pred_var_to_idx = {g: i for i, g in enumerate(a.var_names)}
        pred_cols = [pred_var_to_idx[g] for g in gene_subset if g in pred_var_to_idx]
        Xp_sub = Xp[:, pred_cols]
        obs_col = "gene" if "gene" in a.obs.columns else a.obs.columns[0]
        pred_perts = list(a.obs[obs_col].values)
        # Keep only test perturbations
        pred_means = {}
        for i, pp in enumerate(pred_perts):
            if pp in test_perts:
                pred_means[pp] = Xp_sub[i]
        # Match the test perturbation list ordering from manifest
        keep_perts = [p for p in manifest["test_perts"] if p in pred_means]
        print(f"  {model}: {len(keep_perts)} of {len(test_perts)} test perturbations present")
        # Overlay sample
        rng = np.random.default_rng(SEED)
        n_ctrl_for_overlay = min(N_CTRL, ctrl_X_eval.shape[0])
        ctrl_subset = ctrl_X_eval[rng.choice(ctrl_X_eval.shape[0], n_ctrl_for_overlay, replace=False)]
        pert_matrices = []
        pert_labels = []
        for i_p, p in enumerate(keep_perts):
            cells = sample_perturbed_cells_overlay(ctrl_subset, pred_means[p], N_PERT_CELLS, seed=SEED+i_p)
            pert_matrices.append(cells)
            pert_labels.extend([p]*N_PERT_CELLS)
        cb = build_cb_input(ctrl_subset, pert_matrices, pert_labels, gene_subset)
        t0 = time.time()
        edges = set(tuple(e) for e in run_gies(cb, seed=SEED, partition_size=PARTITION))
        m = compute_set_metrics(edges, real_edges, gene_subset, seed=SEED)
        out[model] = {"f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
                       "n_edges": len(edges), "wall_s": time.time()-t0}
        print(f"  {model:<12} vanilla F1={m['f1']:.4f} edges={len(edges)} wall={time.time()-t0:.0f}s")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Wrote {OUT}")


if __name__ == "__main__":
    main()
