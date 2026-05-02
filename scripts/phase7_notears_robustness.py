"""Phase 7 — NOTEARS-INT robustness check (third causal discovery backend).

NOTEARS (Zheng et al. NeurIPS 2018) formulates DAG learning as continuous
optimization with an acyclicity constraint. The interventional variant
(NOTEARS-INT) masks per-cell loss terms for genes that were perturbed in
that cell, so the model only fits unperturbed genes' linear regressions.

This addresses the algorithmic-robustness critique by adding a third
causal discovery backend alongside GIES (score-based) and inspre
(sparse-inverse). NOTEARS is gradient-based, so it's a fundamentally
different algorithm class.

Output: results/phase7/notears_edges_by_condition.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import anndata as ad
import numpy as np
from notears.linear import notears_linear

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "results" / "phase7"
OUT.mkdir(parents=True, exist_ok=True)

K562 = DATA / "k562_subset_n200_slim.h5ad"
PRED_DIR_VAN = DATA / "model_outputs_real_n200"
PRED_DIR_CAL = DATA / "model_outputs_real_n200_calibrated"
PERT_COL = "gene"
CTRL = "non-targeting"
PSEUDO = 0.01
LAMBDA = 0.05      # L1 penalty
W_THRESH = 0.10    # absolute-weight cutoff for edge inclusion
MAX_ITER = 100


def adjacency_to_edges(W: np.ndarray, gene_names: list[str], thresh: float = W_THRESH) -> set[tuple[str, str]]:
    """Convert NOTEARS weighted adjacency into directed edge set."""
    edges = set()
    n = len(gene_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(W[i, j]) >= thresh:
                edges.add((gene_names[i], gene_names[j]))
    return edges


def f1(pred: set, gt: set) -> tuple[float, float, float]:
    pred = set(pred); gt = set(gt)
    tp = len(pred & gt)
    if tp == 0:
        return 0.0, 0.0, 0.0
    p = tp / max(1, len(pred))
    r = tp / max(1, len(gt))
    return 2 * p * r / (p + r), p, r


def load_pred_means(path: Path, gene_names: list[str]) -> dict[str, np.ndarray]:
    p = ad.read_h5ad(path)
    p_genes = list(p.var_names)
    pred_to_idx = {g: i for i, g in enumerate(p_genes)}
    cols = [pred_to_idx[g] for g in gene_names if g in pred_to_idx]
    pmat = p.X.toarray() if hasattr(p.X, "toarray") else np.asarray(p.X)
    pmat = np.asarray(pmat[:, cols], dtype=np.float64)
    out = {}
    for i in range(pmat.shape[0]):
        gene = str(p.obs[PERT_COL].iloc[i]) if PERT_COL in p.obs.columns else \
               str(p.obs.iloc[i, 0])
        out[gene] = pmat[i]
    return out


def build_ace(adata, gene_names, pred_means=None):
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float64)
    labels = adata.obs[PERT_COL].values
    is_ctrl = labels == CTRL
    ctrl_mean = X[is_ctrl].mean(axis=0)
    n = len(gene_names)
    ace = np.zeros((n, n), dtype=np.float64)
    if pred_means is not None:
        for i, g in enumerate(gene_names):
            if g in pred_means:
                ace[i] = pred_means[g] - ctrl_mean
        return np.nan_to_num(ace, nan=0.0, posinf=0.0, neginf=0.0)
    for i, g in enumerate(gene_names):
        idx = np.where(labels == g)[0]
        if len(idx) >= 5:
            ace[i] = X[idx].mean(axis=0) - ctrl_mean
    return np.nan_to_num(ace, nan=0.0, posinf=0.0, neginf=0.0)


print("=" * 72)
print("Phase 7 — NOTEARS robustness check")
print("=" * 72)

adata = ad.read_h5ad(K562)
adata.var_names = [str(g) for g in adata.var_names]
gene_names = list(adata.var_names)

# Real-data ground truth (NOTEARS on real ACE)
print("Building real ACE matrix ...")
real_ace = build_ace(adata, gene_names, None)
print("Running NOTEARS on real ACE ...")
t0 = time.time()
W_real = notears_linear(real_ace, lambda1=LAMBDA, loss_type="l2",
                          max_iter=MAX_ITER, h_tol=1e-8, w_threshold=0.0)
print(f"  done ({time.time()-t0:.1f}s); ||W||_0 above {W_THRESH}: {(np.abs(W_real)>=W_THRESH).sum()}")
real_edges = adjacency_to_edges(W_real, gene_names)
print(f"  edges (thresh {W_THRESH}): {len(real_edges)}")

results = {"real": {"n_edges": len(real_edges)}, "ground_truth": "NOTEARS on real ACE",
           "lambda": LAMBDA, "w_thresh": W_THRESH}

# ElasticNet baseline
enet_path = PRED_DIR_VAN / "elastic_net_predictions.h5ad"
if enet_path.exists():
    means = load_pred_means(enet_path, gene_names)
    ace = build_ace(adata, gene_names, means)
    print(f"Running NOTEARS on ElasticNet ...")
    t0 = time.time()
    W = notears_linear(ace, lambda1=LAMBDA, loss_type="l2",
                        max_iter=MAX_ITER, h_tol=1e-8, w_threshold=0.0)
    edges = adjacency_to_edges(W, gene_names)
    f, p, r = f1(edges, real_edges)
    print(f"  edges={len(edges)}, F1={f:.4f}, P={p:.4f}, R={r:.4f} ({time.time()-t0:.1f}s)")
    results["elastic_net"] = {"n_edges": len(edges), "f1": f, "precision": p, "recall": r}

# Deep models
for m in ["gears", "cpa", "geneformer"]:
    for tag, d in [("vanilla", PRED_DIR_VAN), ("calibrated", PRED_DIR_CAL)]:
        p = d / f"{m}_predictions.h5ad"
        if not p.exists():
            continue
        means = load_pred_means(p, gene_names)
        ace = build_ace(adata, gene_names, means)
        print(f"Running NOTEARS on {m} ({tag}) ...")
        t0 = time.time()
        try:
            W = notears_linear(ace, lambda1=LAMBDA, loss_type="l2",
                                max_iter=MAX_ITER, h_tol=1e-8, w_threshold=0.0)
            edges = adjacency_to_edges(W, gene_names)
            f, prec, rec = f1(edges, real_edges)
            print(f"  edges={len(edges)}, F1={f:.4f}, P={prec:.4f}, R={rec:.4f} ({time.time()-t0:.1f}s)")
            results[f"{m}_{tag}"] = {"n_edges": len(edges), "f1": f,
                                       "precision": prec, "recall": rec,
                                       "wall_time": time.time()-t0}
        except Exception as e:
            print(f"  FAILED: {e}")
            results[f"{m}_{tag}"] = {"error": str(e)}

with open(OUT / "notears_edges_by_condition.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWrote {OUT/'notears_edges_by_condition.json'}")
