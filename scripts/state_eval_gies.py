"""Run the standard overlay+GIES pipeline on STATE predictions.

Produces results/state/state_f1_summary.json with per-condition F1.

The STATE predictions h5ad is expected at:
    data/model_outputs_real_n200_state/state_predictions.h5ad

Pattern: each row = one perturbation's predicted mean, var_names = genes,
obs[<pert_col>] = perturbation gene name.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import anndata as ad
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causalcellbench.causal.gies_runner import run_gies_full
from scripts.run_eval_local import sample_perturbed_cells_overlay

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"
STATE_DIR = RES / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

K562 = DATA / "k562_subset_n200_slim.h5ad"
GENE_LIST = DATA / "eval_genes_n200.txt"
REAL_EDGES_JSON = RES / "gies_cache" / "n200_s42_p20" / "real_data_edges.json"
PRED_PATH = DATA / "model_outputs_real_n200_state" / "state_predictions.h5ad"
PERT_COL_CANDIDATES = ("gene", "perturbation", "pert", "target_gene")
CTRL_LABELS = ("non-targeting", "control", "ctrl")

SEED = 42
PARTITION_SIZE = 20
N_CELLS_PER_PERT = 100
N_CTRL_FOR_OVERLAY = 200


def compute_f1(pred: set, gt: set):
    pred = set(pred); gt = set(gt)
    tp = len(pred & gt)
    if tp == 0:
        return 0.0, 0.0, 0.0, 0
    p = tp / max(1, len(pred))
    r = tp / max(1, len(gt))
    return 2 * p * r / (p + r), p, r, tp


def find_pert_col(p_obs):
    for c in PERT_COL_CANDIDATES:
        if c in p_obs.columns:
            return c
    return p_obs.columns[0]


def _read_h5ad_minimal(path):
    """Read X + obs columns + var_names from an h5ad via h5py, bypassing
    anndata version incompat (state writes a null encoding in uns/log1p/base
    that older anndata can't read)."""
    import h5py
    with h5py.File(path, "r") as f:
        X = np.asarray(f["X"][:])
        # var_names: try _index attr, else first var dataset
        if "var" in f:
            v = f["var"]
            idx_key = v.attrs.get("_index", "_index")
            if isinstance(idx_key, bytes):
                idx_key = idx_key.decode()
            var_names = [
                x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                for x in v[idx_key][:]
            ]
        # obs gene column
        obs = f["obs"]
        gene_ds = obs["gene"]
        if "categories" in gene_ds:
            cats = [c.decode() if isinstance(c, (bytes, np.bytes_)) else c
                    for c in gene_ds["categories"][:]]
            codes = gene_ds["codes"][:]
            gene_labels = [cats[c] if c >= 0 else "" for c in codes]
        else:
            gene_labels = [
                x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                for x in gene_ds[:]
            ]
    return X, gene_labels, var_names


def load_state_pred_means(predictions_path, gene_subset):
    X, pert_labels, var_names = _read_h5ad_minimal(predictions_path)
    print(f"  STATE predictions: shape={X.shape}, n_perts_in_obs={len(set(pert_labels))}")
    X = X.astype(np.float64)
    var_idx = {g: i for i, g in enumerate(var_names)}
    cols = [var_idx[g] for g in gene_subset if g in var_idx]
    print(f"  matched {len(cols)} of {len(gene_subset)} eval genes in STATE var_names")
    Xs = X[:, cols]
    means = {}
    gene_subset_set = set(gene_subset)
    # Aggregate by perturbation label (mean across cells of same perturbation)
    by_pert = {}
    for i, g in enumerate(pert_labels):
        if g in CTRL_LABELS:
            continue
        if g in gene_subset_set:
            by_pert.setdefault(g, []).append(Xs[i])
    for g, rows in by_pert.items():
        means[g] = np.mean(rows, axis=0).astype(np.float64)
    return means


def main():
    print("=" * 72)
    print("STATE GIES eval")
    print("=" * 72)
    if not PRED_PATH.exists():
        print(f"FATAL: {PRED_PATH} not found")
        return 1

    real_edges_data = json.load(open(REAL_EDGES_JSON))
    real_edges = set(tuple(e) for e in real_edges_data["edges"])
    print(f"Real-data GIES ground truth: {len(real_edges)} edges")

    adata = ad.read_h5ad(K562)
    adata.var_names = [str(g) for g in adata.var_names]
    gene_names_full = list(adata.var_names)

    gene_subset = sorted([line.strip() for line in open(GENE_LIST) if line.strip()])
    var_set = set(gene_names_full)
    gene_subset = [g for g in gene_subset if g in var_set]
    print(f"Eval genes: {len(gene_subset)}")

    eval_idx = [gene_names_full.index(g) for g in gene_subset]
    X_full = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X_full = X_full.astype(np.float64)
    X_eval = X_full[:, eval_idx]
    pert_col = "gene"
    is_ctrl = adata.obs[pert_col].values == "non-targeting"
    ctrl_X = X_eval[is_ctrl]
    print(f"Control cells: {ctrl_X.shape[0]} on {len(gene_subset)} eval genes")

    rng = np.random.default_rng(SEED)
    if ctrl_X.shape[0] > N_CTRL_FOR_OVERLAY:
        ctrl_idx = rng.choice(ctrl_X.shape[0], size=N_CTRL_FOR_OVERLAY, replace=False)
        ctrl_sample = ctrl_X[ctrl_idx]
    else:
        ctrl_sample = ctrl_X

    print(f"\nLoading STATE predictions from {PRED_PATH}")
    pred_means = load_state_pred_means(PRED_PATH, gene_subset)
    print(f"  predictions for {len(pred_means)} perturbations")

    # Build overlay-sampled expression matrix
    all_X = [ctrl_sample]
    all_interv = ["non-targeting"] * ctrl_sample.shape[0]
    for g_idx, g in enumerate(gene_subset):
        if g not in pred_means:
            continue
        gene_seed = SEED * 10000 + g_idx
        sampled = sample_perturbed_cells_overlay(
            ctrl_sample, pred_means[g], N_CELLS_PER_PERT, seed=gene_seed
        )
        all_X.append(sampled)
        all_interv.extend([g] * sampled.shape[0])
    expr = np.vstack(all_X)
    print(f"Expression matrix for GIES: {expr.shape}")

    t0 = time.time()
    edges = run_gies_full(
        expr, interventions=all_interv, gene_names=gene_subset,
        seed=SEED, partition_size=PARTITION_SIZE,
    )
    edge_set = set(edges)
    f1, p, r, tp = compute_f1(edge_set, real_edges)
    wall = time.time() - t0
    print(f"\nSTATE: edges={len(edge_set)} F1={f1:.4f} P={p:.4f} R={r:.4f} TP={tp} "
          f"({wall:.1f}s)")

    summary = {
        "real_n_edges": len(real_edges),
        "n_eval_genes": len(gene_subset),
        "n_predictions": len(pred_means),
        "seed": SEED,
        "partition_size": PARTITION_SIZE,
        "state": {
            "n_edges": len(edge_set),
            "f1": f1, "precision": p, "recall": r, "tp": tp,
            "wall_time_s": wall,
        },
    }
    out = STATE_DIR / (f"state_f1_summary_seed{SEED}.json" if SEED != 42
                       else "state_f1_summary.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser(description="STATE GIES eval (multi-seed capable)")
    _p.add_argument("--seed", type=int, default=42)
    _p.add_argument("--pred-path", default=None,
                    help="Override prediction h5ad (e.g. state_predictions_seed123.h5ad)")
    _a = _p.parse_args()
    SEED = _a.seed
    if _a.pred_path:
        PRED_PATH = Path(_a.pred_path)
    sys.exit(main())
