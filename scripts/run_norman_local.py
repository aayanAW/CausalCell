"""Run Norman 2019 substitutability eval locally — bypasses Modal's
heartbeat/preemption issues that have hung the cloud job for 13+ hours.

Reads data/norman2019.h5ad, runs sequential GIES on real data and on
ElasticNet baseline predictions, writes results/norman2019_summary.json.

Usage:
    .venv_gears/bin/python scripts/run_norman_local.py
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
from sklearn.linear_model import ElasticNet

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "norman2019.h5ad"
OUT_SUMMARY = ROOT / "results" / "norman2019_summary.json"
OUT_EDGES = ROOT / "results" / "norman2019_edges.json"

N_GENES = 200
PARTITION_SIZE = 20
SEED = 42


def _gies_worker(args_tuple, q):
    """Run GIES in subprocess so we can timeout-kill if it hangs."""
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
        # Add tiny jitter to prevent zero-variance regimes that produce NaN
        # gauss_int scores (the cause of partition-4's infinite loop in v1).
        sub = expression_matrix[cell_indices][:, partition_idx].astype(np.float64)
        sub = sub + np.random.default_rng(p_idx).normal(0, 1e-6, size=sub.shape)
        data_list.append(sub)
        if label in ("non-targeting", "ctrl", "control"):
            intervention_targets_list.append([])
        elif label in partition_gene_set:
            intervention_targets_list.append([partition_genes.index(label)])
        else:
            intervention_targets_list.append([])
    edges = []
    try:
        adjacency, _ = gies_pkg.fit_bic(data_list, intervention_targets_list)
    except Exception as e:
        q.put((p_idx, edges, f"fit_bic exception: {e}"))
        return
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
    q.put((p_idx, edges, None))


def run_gies_partition(args_tuple, timeout_sec=600):
    """Run a single GIES partition with a hard timeout. If GIES hangs (e.g.
    NaN-driven infinite hill-climbing), we return empty edges + error string
    rather than blocking the whole pipeline."""
    import multiprocessing as mp
    p_idx = args_tuple[0]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_gies_worker, args=(args_tuple, q))
    p.start()
    p.join(timeout=timeout_sec)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
        return p_idx, [], f"timeout after {timeout_sec}s"
    try:
        return q.get(timeout=5)
    except Exception:
        return p_idx, [], "queue empty after process exit"


def main():
    print("=" * 60)
    print("Norman 2019 substitutability eval (local)")
    print("=" * 60)
    if not DATA.exists():
        print(f"FATAL: {DATA} not found")
        return 1

    adata = ad.read_h5ad(DATA)
    print(f"Loaded {adata.shape}")
    print(f"obs columns (first 10): {list(adata.obs.columns)[:10]}")

    pert_col = None
    for c in ["perturbation_name", "perturbation", "gene", "guide_identity",
              "perturbations", "condition", "target_gene"]:
        if c in adata.obs.columns:
            pert_col = c
            break
    if pert_col is None:
        raise KeyError("No perturbation column found in Norman dataset")
    print(f"Using perturbation column: {pert_col}")
    print(f"  n unique values: {adata.obs[pert_col].nunique()}")

    ctrl_label = None
    for c in ["control", "non-targeting", "NTC", "ctrl", "Control"]:
        if (adata.obs[pert_col] == c).sum() > 0:
            ctrl_label = c
            break
    if ctrl_label is None:
        ctrl_label = adata.obs[pert_col].value_counts().index[0]
    print(f"Control label: {ctrl_label}, n_cells: {(adata.obs[pert_col] == ctrl_label).sum()}")

    is_single = ~adata.obs[pert_col].astype(str).str.contains(r"\+", regex=True)
    is_ctrl = adata.obs[pert_col] == ctrl_label
    keep = is_single | is_ctrl
    adata_s = adata[keep].copy()
    print(f"After dropping combinatorials: {adata_s.shape}")

    pert_counts = adata_s.obs[pert_col].value_counts()
    var_set = set(str(v) for v in adata_s.var_names)
    eligible = sorted([str(g) for g, c in pert_counts.items()
                       if g != ctrl_label and c >= 10 and str(g) in var_set])
    print(f"Eligible genes: {len(eligible)}")

    rng = np.random.default_rng(SEED)
    gene_subset = sorted(rng.choice(eligible,
                                     size=min(N_GENES, len(eligible)),
                                     replace=False).tolist())
    print(f"Selected {len(gene_subset)} target genes")

    gene_to_varidx = {str(g): i for i, g in enumerate(adata_s.var_names)}
    col_idx = [gene_to_varidx[g] for g in gene_subset]
    X = adata_s.X.toarray() if hasattr(adata_s.X, "toarray") else np.asarray(adata_s.X)
    X = X.astype(np.float32)

    is_ctrl_arr = (adata_s.obs[pert_col].values == ctrl_label)
    ctrl_X_full = X[is_ctrl_arr][:, col_idx]
    n_ctrl = min(200, ctrl_X_full.shape[0])
    rng2 = np.random.default_rng(SEED)
    ctrl_X = ctrl_X_full[rng2.choice(ctrl_X_full.shape[0], size=n_ctrl, replace=False)]
    print(f"Control matrix: {ctrl_X.shape}")

    pert_indices = {}
    labels_arr = adata_s.obs[pert_col].values
    for g in gene_subset:
        idx = np.where(labels_arr == g)[0]
        if len(idx) > 100:
            sub = rng.choice(idx, size=100, replace=False)
        else:
            sub = idx
        pert_indices[g] = sub

    rows = [ctrl_X.astype(np.float32)]
    interv = ["non-targeting"] * ctrl_X.shape[0]
    for g in gene_subset:
        idx = pert_indices.get(g)
        if idx is None or len(idx) == 0:
            continue
        rows.append(X[idx][:, col_idx].astype(np.float32))
        interv.extend([g] * len(idx))
    expression_matrix = np.vstack(rows)
    print(f"Combined: {expression_matrix.shape}")

    rng3 = np.random.default_rng(SEED)
    n_g = len(gene_subset)
    gene_order = rng3.permutation(n_g)
    parts = [gene_order[i:i + PARTITION_SIZE] for i in range(0, n_g, PARTITION_SIZE)]
    tasks = [(p, parts[p], gene_subset, expression_matrix, interv) for p in range(len(parts))]
    print(f"\nRunning GIES across {len(parts)} partitions on Norman real data...", flush=True)
    t0 = time.time()
    results = []
    for i, t in enumerate(tasks):
        ts = time.time()
        r = run_gies_partition(t)
        results.append(r)
        n_edges = len(r[1]) if r[2] is None else 0
        err_msg = r[2] or "ok"
        print(f"  [real {i+1}/{len(tasks)}] {n_edges} edges, {time.time()-ts:.1f}s ({err_msg})",
              flush=True)
    real_edges = []
    for _, e, err in results:
        if not err:
            real_edges.extend(e)
    real_edges = set(real_edges)
    print(f"Real-data GIES: {len(real_edges)} edges, {time.time()-t0:.1f}s")

    print("\nTraining ElasticNet on Norman 2019 ...", flush=True)
    enet_models = {}
    n_g_local = len(gene_subset)
    for g_idx in range(n_g_local):
        feature_idx = [i for i in range(n_g_local) if i != g_idx]
        m = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=0)
        m.fit(ctrl_X[:, feature_idx], ctrl_X[:, g_idx])
        enet_models[g_idx] = m
    enet_means = {}
    ctrl_means_local = ctrl_X.mean(axis=0)
    for g_idx, gene in enumerate(gene_subset):
        input_state = ctrl_means_local.copy()
        input_state[g_idx] *= 0.1
        mean_pred = input_state.copy()
        for t_idx, model in enet_models.items():
            if t_idx == g_idx:
                continue
            feature_idx = [i for i in range(n_g_local) if i != t_idx]
            mean_pred[t_idx] = max(float(model.predict(input_state[feature_idx].reshape(1, -1))[0]), 0.0)
        enet_means[gene] = mean_pred

    rng4 = np.random.default_rng(SEED)
    rows = [ctrl_X.astype(np.float32)]
    interv = ["non-targeting"] * ctrl_X.shape[0]
    for g in gene_subset:
        if g not in enet_means:
            continue
        n_cells = 100
        boot_idx = rng4.choice(ctrl_X.shape[0], size=n_cells, replace=True)
        boot = ctrl_X[boot_idx].copy().astype(np.float64)
        cm = ctrl_X.mean(axis=0).astype(np.float64)
        ratios = np.clip((enet_means[g] + 0.01) / (cm + 0.01), 0.01, 10.0)
        boot *= ratios[None, :]
        boot = np.maximum(boot, 0.0)
        boot = rng4.poisson(lam=boot).astype(np.float32)
        rows.append(boot)
        interv.extend([g] * n_cells)
    expr_enet = np.vstack(rows)
    tasks = [(p, parts[p], gene_subset, expr_enet, interv) for p in range(len(parts))]

    print(f"\nRunning GIES on ElasticNet predictions on Norman...", flush=True)
    t0 = time.time()
    results = []
    for i, t in enumerate(tasks):
        ts = time.time()
        r = run_gies_partition(t)
        results.append(r)
        n_edges = len(r[1]) if r[2] is None else 0
        err_msg = r[2] or "ok"
        print(f"  [enet {i+1}/{len(tasks)}] {n_edges} edges, {time.time()-ts:.1f}s ({err_msg})",
              flush=True)
    enet_edges = []
    for _, e, err in results:
        if not err:
            enet_edges.extend(e)
    enet_edges = set(enet_edges)
    print(f"ElasticNet GIES: {len(enet_edges)} edges, {time.time()-t0:.1f}s")

    tp = len(enet_edges & real_edges)
    p_ = tp / max(1, len(enet_edges))
    r_ = tp / max(1, len(real_edges))
    f1 = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
    print(f"\nElasticNet vs real-data Norman GIES: F1={f1:.4f}, P={p_:.4f}, R={r_:.4f}, TP={tp}")

    out = {
        "dataset": "Norman 2019",
        "n_genes": n_g,
        "partition_size": PARTITION_SIZE,
        "seed": SEED,
        "real_n_edges": len(real_edges),
        "elastic_net": {
            "n_edges": len(enet_edges),
            "tp": tp, "f1": f1, "precision": p_, "recall": r_,
        },
    }
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SUMMARY, "w") as f:
        json.dump(out, f, indent=2)
    with open(OUT_EDGES, "w") as f:
        json.dump({"real": [list(e) for e in real_edges],
                   "elastic_net": [list(e) for e in enet_edges]}, f)
    print(f"Wrote {OUT_SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
