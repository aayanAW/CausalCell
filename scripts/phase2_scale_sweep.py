"""PHASE 2: Map the scale inflection point where Geneformer overtakes GEARS.

Sweeps N=50,75,100,125,150,175,200 by subsampling from the 200-gene dataset.
Uses adaptive partition sizes. Saves all intermediate GIES results.

IMPORTANT: Set OMP_NUM_THREADS=1 before running to avoid BLAS thread deadlock.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import time
import pickle
import logging
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from multiprocessing import Pool, get_context
import gies as gies_pkg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path("/Users/aayanalwani/virtual cells")
DATA_PATH = BASE / "data" / "k562_subset_n200_slim.h5ad"
MODEL_DIR = BASE / "data" / "model_outputs_real_n200"
OUTDIR = BASE / "results" / "phase2"
OUTDIR.mkdir(parents=True, exist_ok=True)

GENE_SCALES = [50, 75, 100, 125, 150, 175, 200]
SEED = 42
N_CTRL = 200
N_CELLS_PER_PERT = 100
N_WORKERS = 4  # conservative for Mac


def _run_gies_partition(args_tuple):
    """Run GIES on a single partition."""
    p_idx, partition_idx, gene_names, expression_matrix, interventions = args_tuple
    partition_genes = [gene_names[i] for i in partition_idx]
    partition_gene_set = set(partition_genes)
    n_part = len(partition_idx)

    regimes = {}
    for cell_idx, label in enumerate(interventions):
        regimes.setdefault(label, []).append(cell_idx)

    data_list = []
    intervention_targets_list = []
    for label, cell_indices in regimes.items():
        X_regime = expression_matrix[cell_indices][:, partition_idx]
        data_list.append(X_regime.astype(np.float64))
        if label == "non-targeting":
            intervention_targets_list.append([])
        elif label in partition_gene_set:
            local_idx = list(partition_genes).index(label)
            intervention_targets_list.append([local_idx])
        else:
            intervention_targets_list.append([])

    edges = []
    try:
        adjacency, score = gies_pkg.fit_bic(data_list, intervention_targets_list)
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


def run_gies(expression_matrix, interventions, gene_names, partition_size=20, seed=42):
    """Run GIES with partitioning and parallelism."""
    n_genes = len(gene_names)
    rng = np.random.default_rng(seed)
    gene_order = rng.permutation(n_genes)
    partitions = [
        gene_order[i: i + partition_size]
        for i in range(0, n_genes, partition_size)
    ]
    tasks = [
        (p_idx, partition_idx, gene_names, expression_matrix, interventions)
        for p_idx, partition_idx in enumerate(partitions)
    ]

    ctx = get_context("fork")
    pool = ctx.Pool(processes=N_WORKERS)
    try:
        results = pool.map(_run_gies_partition, tasks)
    finally:
        pool.close()
        pool.join()

    all_edges = []
    for p_idx, edges, error in results:
        if error:
            logger.warning(f"  Partition {p_idx} failed: {error}")
        else:
            all_edges.extend(edges)
    return set(all_edges)


def compute_metrics(predicted_edges, gt_edges, gene_names):
    """Compute F1, Precision, Recall for predicted vs ground truth edges.

    These are set-based metrics at the model's single operating point
    (GIES returns a fixed edge set, not a ranked list).
    """
    pred_set = set(predicted_edges)
    gt_set = set(gt_edges)
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    n_total = len(gene_names) * (len(gene_names) - 1)
    n_gt = len(gt_set)
    random_floor = n_gt / n_total
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp, "fp": fp, "fn": fn,
        "n_edges": len(pred_set),
        "n_gt": len(gt_set),
        "random_floor": float(random_floor),
    }


# Keep old name for any references
compute_auprc = compute_metrics


def build_interventional_data(ctrl_X, pert_data, gene_subset, gene_to_varidx,
                               n_ctrl=200, n_cells=100, seed=42):
    """Build expression matrix + interventions list for GIES."""
    rng = np.random.default_rng(seed)
    col_idx = [gene_to_varidx[g] for g in gene_subset]

    # Subsample controls
    ctrl_idx = rng.choice(ctrl_X.shape[0], size=min(n_ctrl, ctrl_X.shape[0]), replace=False)
    X_ctrl = ctrl_X[ctrl_idx][:, col_idx].astype(np.float32)

    rows = [X_ctrl]
    interventions = ["non-targeting"] * X_ctrl.shape[0]

    for gene in gene_subset:
        if gene not in pert_data:
            continue
        X_pert = pert_data[gene]
        if X_pert.shape[0] == 0:
            continue
        if X_pert.shape[0] > n_cells:
            idx = rng.choice(X_pert.shape[0], size=n_cells, replace=False)
            X_pert = X_pert[idx]
        rows.append(X_pert[:, col_idx].astype(np.float32))
        interventions.extend([gene] * X_pert.shape[0])

    expression_matrix = np.vstack(rows)
    return expression_matrix, interventions


def build_model_interventional_data(ctrl_X, ctrl_mean, model_preds, gene_subset,
                                      gene_to_varidx, n_ctrl=200, n_cells=100, seed=42):
    """Build overlay-sampled interventional data from model predictions."""
    rng = np.random.default_rng(seed)
    col_idx = [gene_to_varidx[g] for g in gene_subset]

    ctrl_idx = rng.choice(ctrl_X.shape[0], size=min(n_ctrl, ctrl_X.shape[0]), replace=False)
    X_ctrl = ctrl_X[ctrl_idx][:, col_idx].astype(np.float32)

    rows = [X_ctrl]
    interventions = ["non-targeting"] * X_ctrl.shape[0]

    cm = ctrl_mean[col_idx]

    for gene in gene_subset:
        if gene not in model_preds:
            continue
        pred_mean = model_preds[gene]  # Already subset to gene_subset order
        # Overlay sampling
        boot_idx = rng.choice(X_ctrl.shape[0], size=n_cells, replace=True)
        boot_cells = X_ctrl[boot_idx].copy()
        fc = (pred_mean + 0.01) / (cm + 0.01)
        fc = np.clip(fc, 0.01, 10.0)
        overlay = boot_cells * fc
        rows.append(overlay.astype(np.float32))
        interventions.extend([gene] * n_cells)

    expression_matrix = np.vstack(rows)
    return expression_matrix, interventions


def main():
    logger.info("=" * 70)
    logger.info("PHASE 2: Scale Sweep — Mapping the Inflection Point")
    logger.info("=" * 70)

    # Load data
    logger.info("Loading data...")
    adata = ad.read_h5ad(DATA_PATH)
    adata.var_names = [str(x) for x in adata.var_names]
    all_genes = sorted(adata.var_names.tolist())
    gene_to_varidx = {g: i for i, g in enumerate(adata.var_names)}

    ctrl_mask = adata.obs["gene"] == "non-targeting"
    ctrl_X = adata[ctrl_mask].X
    if hasattr(ctrl_X, "toarray"):
        ctrl_X = ctrl_X.toarray()
    ctrl_X = np.asarray(ctrl_X, dtype=np.float32)
    ctrl_mean = ctrl_X.mean(axis=0)

    # Build per-gene perturbed cell matrices
    pert_data_real = {}
    for gene in all_genes:
        mask = adata.obs["gene"] == gene
        if mask.sum() == 0:
            continue
        X = adata[mask].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        pert_data_real[gene] = np.asarray(X, dtype=np.float32)

    # Load model predictions (mean vectors, 200 genes)
    models = {}
    for name in ["gears", "cpa", "geneformer"]:
        path = MODEL_DIR / f"{name}_predictions.h5ad"
        if not path.exists():
            continue
        pred = ad.read_h5ad(path)
        model_preds = {}
        for i, row_gene in enumerate(pred.obs["perturbation"]):
            g = str(row_gene)
            if g in set(all_genes):
                # Get the prediction for our 200 genes
                gene_idx = [list(pred.var_names).index(gn) for gn in all_genes if gn in pred.var_names]
                model_preds[g] = pred.X[i, gene_idx] if not hasattr(pred.X, "toarray") else pred.X[i, gene_idx].toarray().flatten()
        models[name] = model_preds
        logger.info(f"  {name}: {len(model_preds)} perturbations")

    # ElasticNet: train proper regression models on control data
    # NOTE: Previous version used real perturbation means as "predictions" (train/test leakage).
    # Now we train actual ElasticNet models on control cells and predict perturbation effects.
    from sklearn.linear_model import ElasticNet as SkElasticNet
    logger.info("Training ElasticNet on 200 genes (proper held-out prediction)...")
    col_idx_all = [gene_to_varidx[g] for g in all_genes]
    rng = np.random.default_rng(SEED)
    ctrl_idx = rng.choice(ctrl_X.shape[0], size=N_CTRL, replace=False)
    X_ctrl_200 = ctrl_X[ctrl_idx][:, col_idx_all]
    ctrl_mean_200 = X_ctrl_200.mean(axis=0)

    # Train one ElasticNet per target gene using control data
    enet_models = {}
    n_all = len(all_genes)
    for g_idx in range(n_all):
        feature_idx = [i for i in range(n_all) if i != g_idx]
        model = SkElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=0)
        model.fit(X_ctrl_200[:, feature_idx], X_ctrl_200[:, g_idx])
        enet_models[g_idx] = model
    logger.info(f"  Trained {len(enet_models)} ElasticNet models")

    # Predict perturbation effects: knock down target gene -> predict downstream
    enet_preds = {}
    for gene in all_genes:
        if gene not in pert_data_real:
            continue
        g_idx = all_genes.index(gene)
        input_state = ctrl_mean_200.copy()
        input_state[g_idx] *= 0.1  # ~90% CRISPRi knockdown

        mean_pred = input_state.copy()
        for t_idx, enet_model in enet_models.items():
            if t_idx == g_idx:
                continue
            feature_idx = [i for i in range(n_all) if i != t_idx]
            mean_pred[t_idx] = max(float(enet_model.predict(input_state[feature_idx].reshape(1, -1))[0]), 0.0)
        enet_preds[gene] = mean_pred

    models["elastic_net"] = enet_preds

    # Run scale sweep
    results_by_scale = {}

    for n_genes in GENE_SCALES:
        logger.info(f"\n{'='*50}")
        logger.info(f"Scale: N={n_genes} genes")
        logger.info(f"{'='*50}")

        # Subsample genes (deterministic)
        rng_genes = np.random.default_rng(SEED)
        gene_subset = sorted(rng_genes.choice(all_genes, size=n_genes, replace=False).tolist())

        # Adaptive partition size
        if n_genes <= 50:
            partition_size = 15
        elif n_genes <= 100:
            partition_size = 20
        else:
            partition_size = 20

        logger.info(f"  Partition size: {partition_size}")
        gene_subset_set = set(gene_subset)
        gene_to_sub_idx = {g: i for i, g in enumerate(gene_subset)}

        # Cache path
        cache_dir = OUTDIR / f"gies_cache_n{n_genes}_p{partition_size}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        scale_results = {"n_genes": n_genes, "partition_size": partition_size, "conditions": {}}

        # 1. Real data
        cache_path = cache_dir / "real_data_edges.json"
        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            gt_edges = set(tuple(e) for e in cached["edges"])
            t = cached.get("time", 0)
            logger.info(f"  real_data: CACHED ({len(gt_edges)} edges, {t:.1f}s)")
        else:
            logger.info(f"  Running GIES on real_data...")
            expr, interv = build_interventional_data(
                ctrl_X, pert_data_real, gene_subset, gene_to_varidx, N_CTRL, N_CELLS_PER_PERT, SEED)
            t0 = time.time()
            gt_edges = run_gies(expr, interv, gene_subset, partition_size, SEED)
            t = time.time() - t0
            logger.info(f"  real_data: {len(gt_edges)} edges ({t:.1f}s)")
            with open(cache_path, "w") as f:
                json.dump({"edges": [list(e) for e in gt_edges], "time": t}, f)

        scale_results["conditions"]["real_data"] = {"n_edges": len(gt_edges), "time": t}
        scale_results["n_gt_edges"] = len(gt_edges)

        # 2-5. Model conditions
        for model_name in ["elastic_net", "gears", "cpa", "geneformer"]:
            cache_path = cache_dir / f"{model_name}_edges.json"
            if cache_path.exists():
                with open(cache_path) as f:
                    cached = json.load(f)
                model_edges = set(tuple(e) for e in cached["edges"])
                t = cached.get("time", 0)
                logger.info(f"  {model_name}: CACHED ({len(model_edges)} edges, {t:.1f}s)")
            else:
                model_preds = models.get(model_name, {})
                # Subset predictions to gene_subset
                sub_preds = {}
                for gene in gene_subset:
                    if gene not in model_preds:
                        continue
                    full_pred = model_preds[gene]
                    # full_pred is indexed by all_genes order; subset to gene_subset
                    sub_idx = [all_genes.index(g) for g in gene_subset]
                    sub_preds[gene] = full_pred[sub_idx] if len(full_pred) > max(sub_idx) else full_pred[:n_genes]

                logger.info(f"  Running GIES on {model_name} ({len(sub_preds)} perts)...")
                expr, interv = build_model_interventional_data(
                    ctrl_X, ctrl_mean, sub_preds, gene_subset, gene_to_varidx,
                    N_CTRL, N_CELLS_PER_PERT, SEED)
                t0 = time.time()
                model_edges = run_gies(expr, interv, gene_subset, partition_size, SEED)
                t = time.time() - t0
                logger.info(f"  {model_name}: {len(model_edges)} edges ({t:.1f}s)")
                with open(cache_path, "w") as f:
                    json.dump({"edges": [list(e) for e in model_edges], "time": t}, f)

            # Compute metrics
            tp = len(model_edges & gt_edges)
            fp = len(model_edges - gt_edges)
            fn = len(gt_edges - model_edges)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            n_total = n_genes * (n_genes - 1)
            random_auprc = len(gt_edges) / n_total

            scale_results["conditions"][model_name] = {
                "n_edges": len(model_edges),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": float(precision),
                "recall": float(recall),
                "auprc": float(precision),  # at single operating point
                "random_floor": float(random_auprc),
                "time": t,
            }

        results_by_scale[n_genes] = scale_results
        # Checkpoint
        with open(OUTDIR / "scale_sweep_results.json", "w") as f:
            json.dump(results_by_scale, f, indent=2)
        logger.info(f"  Checkpoint saved for N={n_genes}")

    # Build summary table
    logger.info("\n" + "=" * 70)
    logger.info("SCALE SWEEP SUMMARY")
    logger.info("=" * 70)
    header = f"{'N':>4} {'GT':>5} {'ElNet':>7} {'GEARS':>7} {'CPA':>7} {'GF':>7}"
    logger.info(header)
    logger.info("-" * 50)

    rows = []
    for n in GENE_SCALES:
        r = results_by_scale[n]
        gt = r["n_gt_edges"]
        row = {"n_genes": n, "n_gt_edges": gt}
        vals = [f"{n:>4}", f"{gt:>5}"]
        for m in ["elastic_net", "gears", "cpa", "geneformer"]:
            if m in r["conditions"]:
                p = r["conditions"][m]["precision"]
                vals.append(f"{p:>7.4f}")
                row[f"{m}_precision"] = p
                row[f"{m}_edges"] = r["conditions"][m]["n_edges"]
                row[f"{m}_tp"] = r["conditions"][m]["tp"]
            else:
                vals.append(f"{'N/A':>7}")
        logger.info(" ".join(vals))
        rows.append(row)

    pd.DataFrame(rows).to_csv(OUTDIR / "scale_sweep_summary.csv", index=False)
    logger.info(f"\nResults saved to {OUTDIR}")


if __name__ == "__main__":
    main()
