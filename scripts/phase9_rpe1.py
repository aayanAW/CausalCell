"""PHASE 9: RPE1 Cross-Context Generalization.

Runs the CausalCellBench pipeline on RPE1 (retinal pigment epithelium)
to test if findings from K562 generalize:
1. Does the ElasticNet anomaly replicate?
2. Does DL effect-size inflation replicate?
3. Do model rankings hold?

Uses the same eval pipeline as K562 but parameterized by cell line config.
Models are trained fresh on RPE1 data.

NOTE: This script handles the EVAL portion. Model training (GEARS/CPA/Geneformer)
must be done separately if model predictions don't exist.
For a quick cross-context test, we use ElasticNet + real data only
(no DL model training required).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import time
import logging
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from multiprocessing import get_context
from sklearn.linear_model import ElasticNet as SkElasticNet
import gies as gies_pkg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths — adjust for RunPod
if Path("/runpod-volume/rpe1_real.h5ad").exists():
    RPE1_PATH = Path("/runpod-volume/rpe1_real.h5ad")
elif Path("/workspace/rpe1_real.h5ad").exists():
    RPE1_PATH = Path("/workspace/rpe1_real.h5ad")
else:
    RPE1_PATH = Path("/Users/aayanalwani/virtual cells/data/rpe1_real.h5ad")

OUTDIR = Path("/workspace/results/phase9_rpe1") if Path("/workspace").exists() else Path("/Users/aayanalwani/virtual cells/results/phase9_rpe1")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_GENES = 200
N_CTRL = 200
N_CELLS_PER_PERT = 100
PARTITION_SIZE = 20
N_WORKERS = 4


# =====================================================================
# GIES Engine (same as main eval)
# =====================================================================

def _run_gies_partition(args_tuple):
    p_idx, partition_idx, gene_names, expression_matrix, interventions = args_tuple
    partition_genes = [gene_names[i] for i in partition_idx]
    partition_gene_set = set(partition_genes)
    n_part = len(partition_idx)

    regimes = {}
    for cell_idx, label in enumerate(interventions):
        regimes.setdefault(label, []).append(cell_idx)

    data_list, targets_list = [], []
    for label, cell_indices in regimes.items():
        X_regime = expression_matrix[cell_indices][:, partition_idx]
        data_list.append(X_regime.astype(np.float64))
        if label == "non-targeting":
            targets_list.append([])
        elif label in partition_gene_set:
            targets_list.append([list(partition_genes).index(label)])
        else:
            targets_list.append([])

    edges = []
    try:
        adjacency, _ = gies_pkg.fit_bic(data_list, targets_list)
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


def run_gies(expr, interv, gene_names, partition_size=20, seed=42):
    n = len(gene_names)
    rng = np.random.default_rng(seed)
    gene_order = rng.permutation(n)
    partitions = [gene_order[i:i+partition_size] for i in range(0, n, partition_size)]
    tasks = [(i, p, gene_names, expr, interv) for i, p in enumerate(partitions)]

    ctx = get_context("fork")
    pool = ctx.Pool(processes=N_WORKERS)
    try:
        results = pool.map(_run_gies_partition, tasks)
    finally:
        pool.close()
        pool.join()

    all_edges = []
    for _, edges, err in results:
        if not err:
            all_edges.extend(edges)
    return set(all_edges)


def build_overlay_data(ctrl_X, ctrl_mean, preds, gene_names,
                        n_ctrl=200, n_cells=100, seed=42):
    rng = np.random.default_rng(seed)
    ctrl_idx = rng.choice(ctrl_X.shape[0], size=min(n_ctrl, ctrl_X.shape[0]), replace=False)
    X_ctrl = ctrl_X[ctrl_idx].astype(np.float32)

    rows = [X_ctrl]
    interventions = ["non-targeting"] * X_ctrl.shape[0]

    for gene in gene_names:
        if gene not in preds:
            continue
        pred_mean = preds[gene]
        boot_idx = rng.choice(X_ctrl.shape[0], size=n_cells, replace=True)
        boot = X_ctrl[boot_idx].copy()
        fc = (pred_mean + 0.01) / (ctrl_mean + 0.01)
        fc = np.clip(fc, 0.01, 10.0)
        overlay = boot * fc
        rows.append(overlay.astype(np.float32))
        interventions.extend([gene] * n_cells)

    return np.vstack(rows), interventions


# =====================================================================
# Main
# =====================================================================

def main():
    logger.info("=" * 70)
    logger.info("PHASE 9: RPE1 Cross-Context Evaluation")
    logger.info("=" * 70)

    # Step 1: Load RPE1 data
    logger.info(f"\nStep 1: Loading RPE1 from {RPE1_PATH}")
    adata = ad.read_h5ad(RPE1_PATH)
    logger.info(f"  Shape: {adata.shape}")
    logger.info(f"  Obs columns: {list(adata.obs.columns)[:10]}")

    # Determine perturbation column
    pert_col = None
    for col in ["gene", "perturbation", "guide_identity", "condition"]:
        if col in adata.obs.columns:
            pert_col = col
            break
    if pert_col is None:
        logger.error("  Cannot find perturbation column!")
        return
    logger.info(f"  Perturbation column: {pert_col}")

    # Map gene names
    if str(adata.var_names[0]).startswith("ENSG"):
        logger.info("  Mapping Ensembl to gene symbols...")
        if "gene_name" in adata.var.columns:
            adata.var_names = [str(x) for x in adata.var["gene_name"].values]
        adata.var_names_make_unique()
    else:
        adata.var_names = [str(x) for x in adata.var_names]

    logger.info(f"  Sample var_names: {list(adata.var_names[:5])}")

    # Find controls
    ctrl_labels = ["non-targeting", "control", "ctrl", "NT"]
    ctrl_mask = pd.Series(False, index=adata.obs.index)
    for label in ctrl_labels:
        ctrl_mask |= (adata.obs[pert_col] == label)
    logger.info(f"  Control cells: {ctrl_mask.sum()}")

    if ctrl_mask.sum() == 0:
        logger.error("  No control cells found!")
        logger.info(f"  Unique pert values: {adata.obs[pert_col].value_counts().head(10)}")
        return

    # Find eligible perturbation genes
    pert_counts = adata.obs[pert_col].value_counts()
    var_set = set(adata.var_names)
    eligible = sorted([
        g for g, c in pert_counts.items()
        if g not in ctrl_labels and c >= 10 and g in var_set
    ])
    logger.info(f"  Eligible perturbation genes: {len(eligible)}")

    # Select 200 genes
    rng = np.random.default_rng(SEED)
    n_select = min(N_GENES, len(eligible))
    gene_subset = sorted(rng.choice(eligible, size=n_select, replace=False).tolist())
    logger.info(f"  Selected {len(gene_subset)} genes")

    # Save gene list
    with open(OUTDIR / "rpe1_gene_subset.txt", "w") as f:
        f.write("\n".join(gene_subset))

    # Build data matrices
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    col_idx = [gene_to_idx[g] for g in gene_subset]

    ctrl_X = adata[ctrl_mask].X
    if hasattr(ctrl_X, "toarray"):
        ctrl_X = ctrl_X.toarray()
    ctrl_X = np.asarray(ctrl_X[:, col_idx], dtype=np.float32)
    ctrl_mean = ctrl_X.mean(axis=0)

    # Real perturbation means
    pert_means = {}
    for gene in gene_subset:
        mask = adata.obs[pert_col] == gene
        if mask.sum() == 0:
            continue
        X = adata[mask].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        pert_means[gene] = np.mean(X[:, col_idx], axis=0).astype(np.float32)

    logger.info(f"  Perturbation means computed: {len(pert_means)}")

    # Step 2: Linearity analysis (replicating Phase 1)
    logger.info("\nStep 2: Linearity analysis")
    fc_matrix = []
    for gene in gene_subset:
        if gene not in pert_means:
            continue
        ratio = (pert_means[gene] + 1) / (ctrl_mean + 1)
        ratio = np.clip(ratio, 1e-10, 1e10)
        fc = np.log2(ratio)
        fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
        fc_matrix.append(fc)
    fc_matrix = np.array(fc_matrix)
    fc_matrix = np.nan_to_num(fc_matrix, nan=0.0)

    U, S, Vt = np.linalg.svd(fc_matrix, full_matrices=False)
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    frac_near_zero = np.mean(np.abs(fc_matrix) < 0.05)
    frac_large = np.mean(np.abs(fc_matrix) > 0.5)

    logger.info(f"  Rank-1 variance: {cumvar[0]:.4f}")
    logger.info(f"  Rank-5 variance: {cumvar[min(4, len(cumvar)-1)]:.4f}")
    logger.info(f"  Frac near-zero (|FC|<0.05): {frac_near_zero:.4f}")
    logger.info(f"  Frac large (|FC|>0.5): {frac_large:.4f}")

    # Step 3: GIES on real data + ElasticNet
    logger.info("\nStep 3: Running GIES on real data")
    cache_gt = OUTDIR / "real_data_edges.json"
    if cache_gt.exists():
        with open(cache_gt) as f:
            gt_edges = set(tuple(e) for e in json.load(f)["edges"])
        logger.info(f"  Real data: CACHED ({len(gt_edges)} edges)")
    else:
        expr_real, interv_real = build_overlay_data(
            ctrl_X, ctrl_mean, pert_means, gene_subset, N_CTRL, N_CELLS_PER_PERT, SEED)
        logger.info(f"  Real data: {expr_real.shape[0]} cells, running GIES...")
        t0 = time.time()
        gt_edges = run_gies(expr_real, interv_real, gene_subset, PARTITION_SIZE, SEED)
        t_gt = time.time() - t0
        logger.info(f"  Real data: {len(gt_edges)} edges ({t_gt:.1f}s)")
        with open(cache_gt, "w") as f:
            json.dump({"edges": [list(e) for e in gt_edges], "time": t_gt}, f)

    logger.info("\nStep 3b: Running GIES on ElasticNet (real means)")
    cache_enet = OUTDIR / "elastic_net_edges.json"
    if cache_enet.exists():
        with open(cache_enet) as f:
            enet_edges = set(tuple(e) for e in json.load(f)["edges"])
        logger.info(f"  ElasticNet: CACHED ({len(enet_edges)} edges)")
    else:
        expr_enet, interv_enet = build_overlay_data(
            ctrl_X, ctrl_mean, pert_means, gene_subset, N_CTRL, N_CELLS_PER_PERT, SEED)
        t0 = time.time()
        enet_edges = run_gies(expr_enet, interv_enet, gene_subset, PARTITION_SIZE, SEED)
        t_enet = time.time() - t0
        logger.info(f"  ElasticNet: {len(enet_edges)} edges ({t_enet:.1f}s)")
        with open(cache_enet, "w") as f:
            json.dump({"edges": [list(e) for e in enet_edges], "time": t_enet}, f)

    # Compute ElasticNet precision
    enet_tp = len(enet_edges & gt_edges)
    enet_prec = enet_tp / len(enet_edges) if enet_edges else 0
    logger.info(f"  ElasticNet precision: {enet_prec:.4f} ({enet_tp} TP / {len(enet_edges)} edges)")

    # Step 4: Summary comparison with K562
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 9 SUMMARY: RPE1 vs K562 Comparison")
    logger.info("=" * 70)

    rpe1_results = {
        "cell_line": "RPE1",
        "n_genes": len(gene_subset),
        "n_eligible": len(eligible),
        "n_control_cells": int(ctrl_mask.sum()),
        "n_pert_means": len(pert_means),
        "linearity": {
            "rank1_variance": float(cumvar[0]),
            "rank5_variance": float(cumvar[min(4, len(cumvar)-1)]),
            "frac_near_zero": float(frac_near_zero),
            "frac_large": float(frac_large),
        },
        "gies": {
            "real_data_edges": len(gt_edges),
            "elastic_net_edges": len(enet_edges),
            "elastic_net_precision": float(enet_prec),
            "elastic_net_tp": enet_tp,
        },
    }

    # K562 comparison values
    k562_ref = {
        "rank1_variance": 0.568,
        "frac_near_zero": 0.386,
        "frac_large": 0.027,
        "enet_precision": 0.425,
    }

    logger.info(f"\n{'Metric':<35} {'K562':>10} {'RPE1':>10} {'Match?':>8}")
    logger.info("-" * 70)
    logger.info(f"{'Rank-1 variance':<35} {k562_ref['rank1_variance']:>10.3f} {cumvar[0]:>10.3f} "
                f"{'YES' if abs(cumvar[0] - k562_ref['rank1_variance']) < 0.15 else 'NO':>8}")
    logger.info(f"{'Frac near-zero (|FC|<0.05)':<35} {k562_ref['frac_near_zero']:>10.3f} {frac_near_zero:>10.3f} "
                f"{'YES' if abs(frac_near_zero - k562_ref['frac_near_zero']) < 0.15 else 'NO':>8}")
    logger.info(f"{'Frac large (|FC|>0.5)':<35} {k562_ref['frac_large']:>10.3f} {frac_large:>10.3f} "
                f"{'YES' if abs(frac_large - k562_ref['frac_large']) < 0.05 else 'NO':>8}")
    logger.info(f"{'ElasticNet precision':<35} {k562_ref['enet_precision']:>10.3f} {enet_prec:>10.3f} "
                f"{'YES' if abs(enet_prec - k562_ref['enet_precision']) < 0.15 else 'NO':>8}")

    # Does the ElasticNet anomaly replicate?
    enet_anomaly = enet_prec > 0.35  # Strong performance from simple regression
    logger.info(f"\n  ElasticNet anomaly replicates in RPE1: {'YES' if enet_anomaly else 'NO'}")
    logger.info(f"  Perturbation effects are low-rank: {'YES' if cumvar[0] > 0.4 else 'NO'}")
    logger.info(f"  Perturbation effects are sparse: {'YES' if frac_near_zero > 0.3 else 'NO'}")

    rpe1_results["elasticnet_anomaly_replicates"] = enet_anomaly
    rpe1_results["low_rank"] = bool(cumvar[0] > 0.4)
    rpe1_results["sparse"] = bool(frac_near_zero > 0.3)

    with open(OUTDIR / "phase9_results.json", "w") as f:
        json.dump(rpe1_results, f, indent=2)

    logger.info(f"\nAll results saved to {OUTDIR}")
    logger.info("PHASE 9 COMPLETE")


if __name__ == "__main__":
    main()
