"""Phase 2 — technical-duplicate noise ceiling.

Mejia et al. (2025) recommend reporting model performance as a fraction of
the technical-duplicate noise ceiling: split real cells in half, recover a
graph from each half, compute F1 between the two recovered graphs. This is
the F1 a *perfect* mean predictor could achieve given Replogle's intrinsic
technical variance.

Without this baseline, comparisons of e.g. ElasticNet F1=0.425 vs Geneformer
F1=0.432 are unfalsifiable — both could be at >95% of the noise ceiling
(in which case the gap is practically meaningless) or at 60% (in which case
both have substantial room to improve).

Output: results/technical_duplicate.json, including F1 between paired halves
across n_splits random splits, and a summary used as the noise ceiling row in
the headline table.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"
OUT = RES / "technical_duplicate"
OUT.mkdir(parents=True, exist_ok=True)

K562 = DATA / "k562_subset_n200_slim.h5ad"
PERT_COL = "gene"
CTRL_LABEL = "non-targeting"
PARTITION_SIZE = 20
N_CTRL_GIES = 200
N_CELLS_PER_PERT = 100
SEED = 42
N_SPLITS = 3   # 3 random half-splits; can be raised if compute allows


def _run_gies_partition(args_tuple):
    import gies as gies_pkg
    p_idx, partition_idx, gene_names, expression_matrix, interventions = args_tuple
    partition_genes = [gene_names[i] for i in partition_idx]
    partition_gene_set = set(partition_genes)
    n_part = len(partition_idx)
    regimes = {}
    for cell_idx, label in enumerate(interventions):
        regimes.setdefault(label, []).append(cell_idx)
    data_list, intervention_targets_list = [], []
    for label, cell_indices in regimes.items():
        data_list.append(expression_matrix[cell_indices][:, partition_idx].astype(np.float64))
        if label == CTRL_LABEL:
            intervention_targets_list.append([])
        elif label in partition_gene_set:
            intervention_targets_list.append([partition_genes.index(label)])
        else:
            intervention_targets_list.append([])
    edges = []
    try:
        adjacency, _ = gies_pkg.fit_bic(data_list, intervention_targets_list)
    except Exception as e:
        return p_idx, edges, str(e)
    for i in range(n_part):
        for j in range(n_part):
            if i == j:
                continue
            if adjacency[i, j] != 0:
                if adjacency[j, i] != 0:
                    if i < j:
                        edges.append((partition_genes[i], partition_genes[j]))
                else:
                    edges.append((partition_genes[i], partition_genes[j]))
    return p_idx, edges, None


def run_gies(expression_matrix, interventions, gene_names, partition_size, seed, n_workers=4):
    import multiprocessing as mp
    n_genes = len(gene_names)
    rng = np.random.default_rng(seed)
    gene_order = rng.permutation(n_genes)
    partitions = [gene_order[i:i + partition_size] for i in range(0, n_genes, partition_size)]
    tasks = [(p_idx, p, gene_names, expression_matrix, interventions)
             for p_idx, p in enumerate(partitions)]
    ctx = mp.get_context("fork")
    pool = ctx.Pool(processes=n_workers)
    try:
        results = pool.map(_run_gies_partition, tasks)
    finally:
        pool.close(); pool.join()
    edges = []
    for _, e, err in results:
        if not err:
            edges.extend(e)
    return set(edges)


def build_cb_input(ctrl_X_sub, pert_data, gene_names, n_cells_per_pert, seed):
    """Build expression matrix + interventions list for GIES from real cells."""
    rng = np.random.default_rng(seed)
    rows = [ctrl_X_sub.astype(np.float32)]
    interv = [CTRL_LABEL] * ctrl_X_sub.shape[0]
    for gene in gene_names:
        if gene not in pert_data:
            continue
        cells = pert_data[gene]
        if cells.shape[0] == 0:
            continue
        if cells.shape[0] > n_cells_per_pert:
            idx = rng.choice(cells.shape[0], size=n_cells_per_pert, replace=False)
            cells = cells[idx]
        rows.append(cells.astype(np.float32))
        interv.extend([gene] * cells.shape[0])
    return np.vstack(rows), interv


def f1_score(pred_set, gt_set):
    pred_set = set(pred_set); gt_set = set(gt_set)
    tp = len(pred_set & gt_set)
    if not tp:
        return 0.0, 0.0, 0.0
    p = tp / len(pred_set)
    r = tp / len(gt_set)
    return 2 * p * r / (p + r), p, r


print("=" * 72)
print("Phase 2 — technical-duplicate noise ceiling")
print("=" * 72)
print("Loading K562 N=200 ...")
adata = ad.read_h5ad(K562)
adata.var_names = [str(g) for g in adata.var_names]
gene_names = list(adata.var_names)

# Index real cells by perturbation
labels = adata.obs[PERT_COL].values
X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
X = X.astype(np.float32)

ctrl_idx = np.where(labels == CTRL_LABEL)[0]
pert_indices: dict[str, np.ndarray] = {}
for g in gene_names:
    idx = np.where(labels == g)[0]
    if len(idx) >= 10:
        pert_indices[g] = idx
print(f"  {len(ctrl_idx)} control cells, {len(pert_indices)} perturbation regimes")

split_results = []
rng_master = np.random.default_rng(SEED)

for split in range(N_SPLITS):
    print(f"\nSplit {split+1}/{N_SPLITS}")
    sub_seed = SEED + split * 1000
    rng = np.random.default_rng(sub_seed)

    # Random half splits per perturbation
    pert_a, pert_b = {}, {}
    for g, idxs in pert_indices.items():
        sh = rng.permutation(idxs)
        h = len(sh) // 2
        pert_a[g] = X[sh[:h]]
        pert_b[g] = X[sh[h:2*h]]

    # Random half split for controls (used for GIES control regime)
    sh_ctrl = rng.permutation(ctrl_idx)
    h_c = len(sh_ctrl) // 2
    ctrl_X_a_full = X[sh_ctrl[:h_c]]
    ctrl_X_b_full = X[sh_ctrl[h_c:2*h_c]]
    n_use = min(N_CTRL_GIES, ctrl_X_a_full.shape[0], ctrl_X_b_full.shape[0])
    idx_a = rng.choice(ctrl_X_a_full.shape[0], size=n_use, replace=False)
    idx_b = rng.choice(ctrl_X_b_full.shape[0], size=n_use, replace=False)
    ctrl_X_a = ctrl_X_a_full[idx_a]
    ctrl_X_b = ctrl_X_b_full[idx_b]

    # Build CB inputs
    expr_a, interv_a = build_cb_input(ctrl_X_a, pert_a, gene_names,
                                       N_CELLS_PER_PERT, sub_seed + 1)
    expr_b, interv_b = build_cb_input(ctrl_X_b, pert_b, gene_names,
                                       N_CELLS_PER_PERT, sub_seed + 2)
    print(f"  half A: {expr_a.shape}, half B: {expr_b.shape}")

    t0 = time.time()
    edges_a = run_gies(expr_a, interv_a, gene_names, PARTITION_SIZE, sub_seed + 3)
    edges_b = run_gies(expr_b, interv_b, gene_names, PARTITION_SIZE, sub_seed + 4)
    t = time.time() - t0
    f1_ab, p_ab, r_ab = f1_score(edges_a, edges_b)
    print(f"  edges A={len(edges_a)}, B={len(edges_b)}, F1(A,B)={f1_ab:.4f}, "
          f"P={p_ab:.4f}, R={r_ab:.4f}, time={t:.1f}s")
    split_results.append({
        "split": split, "seed": sub_seed,
        "edges_a": len(edges_a), "edges_b": len(edges_b),
        "f1_ab": f1_ab, "precision_ab": p_ab, "recall_ab": r_ab,
        "wall_time": t,
    })
    # Cache edge sets
    with open(OUT / f"split{split}_a_edges.json", "w") as f:
        json.dump([list(e) for e in edges_a], f)
    with open(OUT / f"split{split}_b_edges.json", "w") as f:
        json.dump([list(e) for e in edges_b], f)

f1s = np.array([s["f1_ab"] for s in split_results])
ps = np.array([s["precision_ab"] for s in split_results])
rs = np.array([s["recall_ab"] for s in split_results])
summary = {
    "n_splits": N_SPLITS,
    "splits": split_results,
    "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
    "precision_mean": float(ps.mean()),
    "recall_mean": float(rs.mean()),
}
print("\n" + "=" * 72)
print("TECHNICAL DUPLICATE NOISE CEILING")
print("=" * 72)
print(f"  F1(A,B) mean = {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
print(f"  Precision mean = {summary['precision_mean']:.4f}")
print(f"  Recall mean    = {summary['recall_mean']:.4f}")

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote {OUT/'summary.json'}")
