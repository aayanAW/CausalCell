"""Phase 7 — algorithmic robustness via the PC algorithm.

The original CausalCellBench used GIES + inspre. A reviewer concern is that
the GIES-zero-edges-from-CPA-and-Geneformer finding could be algorithm-
specific. We add a third causal discovery backend (PC; Spirtes & Glymour
2000) via the `causal-learn` package and compare F1 across backends per
condition.

PC is constraint-based and observational by construction — it does not use
intervention labels — so it serves as a complementary check: if conclusions
hold across score-based interventional (GIES), sparse-inverse-of-ACE
(inspre), and constraint-based observational (PC), the result is robust to
algorithm class.

Output: results/phase7/pc_edges_by_condition.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import anndata as ad
import numpy as np

from causallearn.search.ConstraintBased.PC import pc

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "results" / "phase7"
OUT.mkdir(parents=True, exist_ok=True)

K562 = DATA / "k562_subset_n200_slim.h5ad"
PERT_COL = "gene"
CTRL_LABEL = "non-targeting"
SEED = 42
PSEUDO = 0.01
PRED_DIR = DATA / "model_outputs_real_n200"
PRED_DIR_CAL = DATA / "model_outputs_real_n200_calibrated"


def load_real_pert_means(adata, gene_names):
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float64)
    labels = adata.obs[PERT_COL].values
    is_ctrl = labels == CTRL_LABEL
    ctrl_mean = X[is_ctrl].mean(axis=0)
    means = {}
    for g in gene_names:
        idx = np.where(labels == g)[0]
        if len(idx) >= 5:
            means[g] = X[idx].mean(axis=0)
    return ctrl_mean, means


def load_pred_means(model_path, gene_names):
    p = ad.read_h5ad(model_path)
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


def build_ace_dataset(pert_means, ctrl_mean, gene_names):
    """Stack rows = perturbations, cols = genes, values = predicted shift from ctrl.
    PC then runs on this matrix with each row as an observation."""
    rows = []
    for g in gene_names:
        if g in pert_means:
            rows.append(pert_means[g] - ctrl_mean)
    return np.stack(rows, axis=0)


def pc_edges(data: np.ndarray, gene_names: list[str], alpha: float = 0.05) -> set[tuple[str, str]]:
    """Run PC and extract directed/undirected edges as (source, target) tuples."""
    # Replace NaN/inf with 0 (no shift) so PC's CIT doesn't reject input
    data = np.nan_to_num(np.asarray(data, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    # Add tiny per-column jitter so std is never exactly zero (prevents
    # FisherZ from producing NaN correlations on constant columns).
    rng = np.random.default_rng(0)
    col_std = data.std(axis=0)
    zero_cols = col_std < 1e-12
    if zero_cols.any():
        jitter = rng.normal(0.0, 1e-8, size=data.shape).astype(np.float64)
        data = data + jitter * zero_cols.astype(np.float64)[None, :]
    cg = pc(data, alpha=alpha, indep_test="fisherz", verbose=False, show_progress=False)
    g = cg.G
    nodes = g.get_nodes()
    edges: set[tuple[str, str]] = set()
    n = len(nodes)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            edge = g.get_edge(nodes[i], nodes[j])
            if edge is None:
                continue
            # In causal-learn's GraphNode adjacency we just take A[i][j] != 0
            # as a directed edge from i to j; PC produces a CPDAG so undirected
            # edges appear in both directions.
            mat = g.graph
            if mat[i, j] != 0:
                edges.add((gene_names[i], gene_names[j]))
    return edges


def compute_f1(pred: set, gt: set) -> tuple[float, float, float]:
    pred = set(pred); gt = set(gt)
    tp = len(pred & gt)
    if tp == 0:
        return 0.0, 0.0, 0.0
    p = tp / max(1, len(pred))
    r = tp / max(1, len(gt))
    return 2 * p * r / (p + r), p, r


print("=" * 72)
print("Phase 7 — PC robustness check")
print("=" * 72)
adata = ad.read_h5ad(K562)
adata.var_names = [str(g) for g in adata.var_names]
gene_names = list(adata.var_names)

ctrl_mean, real_pert_means = load_real_pert_means(adata, gene_names)
print(f"  {len(real_pert_means)} real perturbations on {len(gene_names)} genes")

# Run PC on real ACE data (this is our "real-data PC" ground truth for this backend)
t0 = time.time()
real_ace = build_ace_dataset(real_pert_means, ctrl_mean, gene_names)
print(f"  real ACE matrix shape: {real_ace.shape}, running PC ...")
real_edges = pc_edges(real_ace, gene_names, alpha=0.05)
print(f"  real PC edges: {len(real_edges)} ({time.time()-t0:.1f}s)")

# Conditions to evaluate vs real-PC ground truth
conditions = {
    "elastic_net_van": ROOT / "results" / "gies_cache" / "n200_s42_p20" / "elastic_net_edges.json",
}
# For deep models we use ACE matrices derived from predicted means
deep_models = ["gears", "cpa", "geneformer"]

results = {"real": {"n_edges": len(real_edges)}, "ground_truth": "PC on real ACE"}
OUT_JSON = OUT / "pc_edges_by_condition.json"

def _flush():
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

_flush()  # write real-data result immediately

for m in deep_models:
    p_van = PRED_DIR / f"{m}_predictions.h5ad"
    p_cal = PRED_DIR_CAL / f"{m}_predictions.h5ad"
    if p_van.exists():
        try:
            means_van = load_pred_means(p_van, gene_names)
            ace_van = build_ace_dataset(means_van, ctrl_mean, gene_names)
            t0 = time.time()
            edges_van = pc_edges(ace_van, gene_names, alpha=0.05)
            f1, p, r = compute_f1(edges_van, real_edges)
            print(f"  {m} (vanilla):   PC edges={len(edges_van)}, "
                  f"F1={f1:.4f}, P={p:.4f}, R={r:.4f} ({time.time()-t0:.1f}s)",
                  flush=True)
            results[f"{m}_vanilla"] = {"n_edges": len(edges_van), "f1": f1,
                                        "precision": p, "recall": r}
        except Exception as exc:
            print(f"  {m} (vanilla) FAILED: {exc}", flush=True)
            results[f"{m}_vanilla"] = {"error": str(exc)}
        _flush()
    if p_cal.exists():
        try:
            means_cal = load_pred_means(p_cal, gene_names)
            ace_cal = build_ace_dataset(means_cal, ctrl_mean, gene_names)
            t0 = time.time()
            edges_cal = pc_edges(ace_cal, gene_names, alpha=0.05)
            f1, p, r = compute_f1(edges_cal, real_edges)
            print(f"  {m} (calibrated): PC edges={len(edges_cal)}, "
                  f"F1={f1:.4f}, P={p:.4f}, R={r:.4f} ({time.time()-t0:.1f}s)",
                  flush=True)
            results[f"{m}_calibrated"] = {"n_edges": len(edges_cal), "f1": f1,
                                           "precision": p, "recall": r}
        except Exception as exc:
            print(f"  {m} (calibrated) FAILED: {exc}", flush=True)
            results[f"{m}_calibrated"] = {"error": str(exc)}
        _flush()
print(f"\nWrote {OUT/'pc_edges_by_condition.json'}")
