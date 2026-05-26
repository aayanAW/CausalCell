"""Phase 4 followup: run overlay+GIES on each calibrated h5ad produced by
phase_4_calibration_sweep.py. Produces results/calibration_modes/f1_per_ordering.json.

Each h5ad is loaded, overlay-sampled with the existing pipeline, GIES is run, and
F1 is computed against the same real-data ground truth used by Phase 1.
"""
from __future__ import annotations
import json
import sys
import time
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
OUT = ROOT / "results" / "calibration_modes" / "f1_per_ordering.json"

SEED = 42
PARTITION = 20
N_PERT_CELLS = 100
N_CTRL = 200


def main():
    if not MANIFEST.exists():
        print(f"  No manifest at {MANIFEST}. Run phase_4_calibration_sweep.py first.")
        return 1
    with open(MANIFEST) as f:
        manifest = json.load(f)
    with open(REAL_EDGES) as f:
        real_edges = set(tuple(e) for e in json.load(f)["edges"])
    print(f"  Real edges (seed 42): {len(real_edges)}")

    # Load K562 base for control cells + eval genes
    adata = ad.read_h5ad(K562)
    adata.var_names = [str(g) for g in adata.var_names]
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float64)
    is_ctrl = (adata.obs["gene"] == "non-targeting").values
    gene_subset = sorted([line.strip() for line in open(GENE_LIST) if line.strip()])
    gene_to_idx = {str(g): i for i, g in enumerate(adata.var_names)}
    gene_subset = [g for g in gene_subset if g in gene_to_idx]
    col_idx = [gene_to_idx[g] for g in gene_subset]
    ctrl_X_eval = X[is_ctrl][:, col_idx]
    print(f"  ctrl_X_eval: {ctrl_X_eval.shape}, n_eval_genes: {len(gene_subset)}")

    out = {}
    if OUT.exists():
        with open(OUT) as f:
            out = json.load(f)

    for entry in manifest["orderings"]:
        model = entry["model"]
        order = entry["order"]
        h5_path = ROOT / entry["h5ad"]
        key = f"{model}_{order}"
        if key in out:
            print(f"  {key}: cached, skip")
            continue
        if not h5_path.exists():
            print(f"  {key}: h5ad missing, skip")
            continue
        cal_adata = ad.read_h5ad(h5_path)
        cal_X = cal_adata.X.toarray() if hasattr(cal_adata.X, "toarray") else np.asarray(cal_adata.X)
        cal_X = cal_X.astype(np.float64)
        cal_perts = list(cal_adata.obs["gene"].values)
        # Build pred_means dict for overlay
        pred_means = {p: cal_X[i] for i, p in enumerate(cal_perts)}

        t0 = time.time()
        rng = np.random.default_rng(SEED)
        # Subsample control cells matching the K562 baseline
        n_ctrl_for_overlay = min(N_CTRL, ctrl_X_eval.shape[0])
        ctrl_subset = ctrl_X_eval[rng.choice(ctrl_X_eval.shape[0], n_ctrl_for_overlay, replace=False)]
        ctrl_mean_eval = ctrl_X_eval.mean(axis=0)

        # Overlay sample per perturbation
        pert_matrices = []
        pert_labels = []
        for p in cal_perts:
            pred_vec = pred_means[p]
            cells = sample_perturbed_cells_overlay(
                ctrl_subset, ctrl_mean_eval, pred_vec, N_PERT_CELLS, rng
            )
            pert_matrices.append(cells)
            pert_labels.extend([p] * N_PERT_CELLS)

        cb = build_cb_input(ctrl_subset, pert_matrices, pert_labels, gene_subset)
        edges = set(tuple(e) for e in run_gies(cb, seed=SEED, partition_size=PARTITION))
        metrics = compute_set_metrics(edges, real_edges, gene_subset, seed=SEED)
        wall = time.time() - t0
        out[key] = {
            "model": model,
            "order": order,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "n_edges": len(edges),
            "wall_s": wall,
        }
        print(f"  {key:<20} f1={metrics['f1']:.4f} edges={len(edges)} wall={wall:.0f}s")
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2)
    print(f"\n  Wrote {OUT} with {len(out)} F1 entries.")


if __name__ == "__main__":
    sys.exit(main() or 0)
