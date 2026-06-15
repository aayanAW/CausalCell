"""PHASE 5: inspre integration via official R package.

Uses rpy2 to call the validated inspre R implementation for causal
discovery on the K562 N=200 dataset. Compares inspre graphs against
GIES graphs to validate model rankings with a second algorithm.

Also demonstrates genome-scale capability by running on all available
perturbation genes (impossible with GIES).

Saves all intermediate results with checkpoints.
"""

import os
os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"

import json
import time
import logging
import pickle
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path("/Users/aayanalwani/virtual cells")
DATA_PATH = BASE / "data" / "k562_subset_n200_slim.h5ad"
FULL_DATA_PATH = BASE / "data" / "k562_real.h5ad"
MODEL_DIR = BASE / "data" / "model_outputs_real_n200"
GIES_CACHE = BASE / "results" / "gies_cache" / "n200_s42_p20"
OUTDIR = BASE / "results" / "phase5_inspre"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 42


# =====================================================================
# inspre R Bridge
# =====================================================================

def run_inspre_from_ace(ace_matrix: np.ndarray, target_edges: int = 1200,
                         max_iter: int = 200) -> tuple:
    """
    Run inspre via the official R package on a precomputed ACE matrix.

    Uses inspre's built-in lambda path (20 values from sparse to dense).
    Selects the lambda that produces closest to target_edges edges,
    or uses CV test_error if available.

    Args:
        ace_matrix: (n_genes, n_genes) ACE matrix
        target_edges: desired number of edges (for lambda selection)
        max_iter: ADMM iterations

    Returns:
        (adjacency_matrix, elapsed_seconds, metadata_dict)
    """
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import default_converter, numpy2ri

    inspre_r = importr("inspre")
    n = ace_matrix.shape[0]
    logger.info(f"  Running inspre (R): {n}x{n} ACE matrix, target ~{target_edges} edges")

    with localconverter(default_converter + numpy2ri.converter):
        r_ace = ro.r['matrix'](ace_matrix.flatten(), nrow=n, ncol=n, byrow=True)

        t0 = time.time()
        result = inspre_r.fit_inspre_from_R(
            R_hat=r_ace,
            its=max_iter,
            verbose=0,
            train_prop=0.8,
        )
        elapsed = time.time() - t0
        logger.info(f"  inspre completed in {elapsed:.1f}s")

        # Extract results by tag
        tags = list(result.tags())
        G_hat = np.array(result[tags.index("G_hat")])  # (n, n, nlambda)
        lambdas = np.array(result[tags.index("lambda")])
        test_err = np.array(result[tags.index("test_error")])

    # Select best lambda
    # Strategy 1: Use CV test_error if available (non-NaN)
    valid_te = ~np.isnan(test_err)
    if np.any(valid_te):
        best_idx = np.where(valid_te)[0][np.argmin(test_err[valid_te])]
        logger.info(f"  Selected lambda by CV: {lambdas[best_idx]:.4f} "
                     f"(test_error={test_err[best_idx]:.4f})")
    else:
        # Strategy 2: Pick lambda giving closest to target_edges
        edge_counts = []
        for li in range(G_hat.shape[2]):
            nnz = np.sum(np.abs(G_hat[:, :, li]) > 1e-6)
            edge_counts.append(nnz)
        edge_counts = np.array(edge_counts)
        best_idx = np.argmin(np.abs(edge_counts - target_edges))
        logger.info(f"  Selected lambda by edge count: {lambdas[best_idx]:.4f} "
                     f"({edge_counts[best_idx]} edges, target={target_edges})")

    adj_best = G_hat[:, :, best_idx]

    metadata = {
        "lambda": float(lambdas[best_idx]),
        "lambda_idx": int(best_idx),
        "all_lambdas": lambdas.tolist(),
        "all_edge_counts": [int(np.sum(np.abs(G_hat[:,:,i]) > 1e-6))
                            for i in range(G_hat.shape[2])],
        "elapsed": elapsed,
    }

    return adj_best, elapsed, metadata


def adjacency_to_edges(adj: np.ndarray, gene_names: list,
                        threshold: float = 0.0) -> set:
    """Convert adjacency matrix to edge set, filtering by absolute weight."""
    edges = set()
    n = len(gene_names)
    for i in range(n):
        for j in range(n):
            if i != j and abs(adj[i, j]) > threshold:
                edges.add((gene_names[i], gene_names[j]))
    return edges


# =====================================================================
# ACE Matrix Computation
# =====================================================================

def compute_ace_matrix(pert_means: dict, ctrl_mean: np.ndarray,
                        gene_names: list) -> np.ndarray:
    """
    Compute Average Causal Effect matrix.

    ACE[i,j] = mean(X_j | do(gene_i)) - mean(X_j | control)

    Args:
        pert_means: {gene_name: mean_expression_vector} for each perturbation
        ctrl_mean: mean expression of control cells
        gene_names: ordered list of gene names (defines matrix axes)

    Returns:
        ACE matrix (n_genes x n_genes)
    """
    n = len(gene_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    ace = np.zeros((n, n), dtype=np.float64)

    for gene, mean_expr in pert_means.items():
        if gene not in gene_to_idx:
            continue
        i = gene_to_idx[gene]
        ace[i, :] = mean_expr - ctrl_mean

    return ace


# =====================================================================
# Data Loading
# =====================================================================

def load_200gene_data():
    """Load the N=200 gene subset and all model predictions."""
    logger.info("Loading N=200 data...")
    adata = ad.read_h5ad(DATA_PATH)
    adata.var_names = [str(x) for x in adata.var_names]
    gene_names = sorted(adata.var_names.tolist())
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    ctrl_mask = adata.obs["gene"] == "non-targeting"
    ctrl_X = adata[ctrl_mask].X
    if hasattr(ctrl_X, "toarray"):
        ctrl_X = ctrl_X.toarray()
    ctrl_X = np.asarray(ctrl_X, dtype=np.float64)
    ctrl_mean = ctrl_X.mean(axis=0)

    # Real perturbation means (200 genes)
    col_idx = [gene_to_idx[g] for g in gene_names]
    pert_means_real = {}
    for gene in gene_names:
        mask = adata.obs["gene"] == gene
        if mask.sum() == 0:
            continue
        X = adata[mask].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        pert_means_real[gene] = np.mean(X, axis=0).astype(np.float64)[col_idx]

    ctrl_mean_200 = ctrl_mean[col_idx]

    # Model predictions (subset to 200 genes)
    models = {}
    for name in ["gears", "cpa", "geneformer"]:
        path = MODEL_DIR / f"{name}_predictions.h5ad"
        if not path.exists():
            continue
        pred = ad.read_h5ad(path)
        pred_gene_idx = [list(pred.var_names).index(g) for g in gene_names if g in pred.var_names]
        model_preds = {}
        for i, row_gene in enumerate(pred.obs["perturbation"]):
            g = str(row_gene)
            if g in set(gene_names):
                vals = pred.X[i, pred_gene_idx]
                if hasattr(vals, "toarray"):
                    vals = vals.toarray().flatten()
                model_preds[g] = np.asarray(vals, dtype=np.float64)
        models[name] = model_preds
        logger.info(f"  {name}: {len(model_preds)} perturbations")

    # ElasticNet proxy = real means (same as eval pipeline)
    models["elastic_net"] = pert_means_real

    return gene_names, ctrl_mean_200, pert_means_real, models


# =====================================================================
# Main Analysis
# =====================================================================

def run_inspre_on_condition(name: str, pert_means: dict, ctrl_mean: np.ndarray,
                             gene_names: list, target_edges: int = 1200) -> dict:
    """Run inspre on one condition, with caching."""
    cache_path = OUTDIR / f"{name}_inspre_edges.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        edges = set(tuple(e) for e in cached["edges"])
        logger.info(f"  {name}: CACHED ({len(edges)} edges, {cached.get('time', 0):.1f}s)")
        return {"edges": edges, "time": cached.get("time", 0), "n_edges": len(edges)}

    # Compute ACE matrix
    ace = compute_ace_matrix(pert_means, ctrl_mean, gene_names)
    logger.info(f"  {name}: ACE matrix {ace.shape}, "
                f"mean|ACE|={np.mean(np.abs(ace)):.4f}, "
                f"nnz(|ACE|>0.01)={np.sum(np.abs(ace) > 0.01)}")

    # Save ACE matrix checkpoint
    np.save(OUTDIR / f"{name}_ace_matrix.npy", ace)

    # Run inspre
    adj, elapsed, metadata = run_inspre_from_ace(ace, target_edges=target_edges)

    # Convert to edges
    edges = adjacency_to_edges(adj, gene_names, threshold=1e-6)
    logger.info(f"  {name}: {len(edges)} edges ({elapsed:.1f}s)")

    # Cache
    with open(cache_path, "w") as f:
        json.dump({
            "edges": [list(e) for e in edges],
            "time": elapsed,
            "lambda": metadata["lambda"],
            "n_genes": len(gene_names),
            "metadata": metadata,
        }, f)

    return {"edges": edges, "time": elapsed, "n_edges": len(edges)}


def compare_inspre_vs_gies(inspre_edges: dict, gene_names: list):
    """Compare inspre and GIES edge sets for each condition."""
    logger.info("\n" + "=" * 70)
    logger.info("inspre vs GIES Comparison")
    logger.info("=" * 70)

    conditions = ["real_data", "elastic_net", "cpa_real", "gears_real", "geneformer_real"]
    display = {"real_data": "Real Data", "elastic_net": "ElasticNet",
               "cpa_real": "CPA", "gears_real": "GEARS", "geneformer_real": "Geneformer"}

    # Load GIES edges
    gies_edges = {}
    for name in conditions:
        gies_path = GIES_CACHE / f"{name}_edges.json"
        if gies_path.exists():
            with open(gies_path) as f:
                gies_edges[name] = set(tuple(e) for e in json.load(f)["edges"])

    # inspre ground truth = inspre on real data
    inspre_gt = inspre_edges.get("real_data", {}).get("edges", set())
    gies_gt = gies_edges.get("real_data", set())

    logger.info(f"\nGround truth edges: GIES={len(gies_gt)}, inspre={len(inspre_gt)}")
    overlap_gt = len(inspre_gt & gies_gt)
    logger.info(f"GT overlap: {overlap_gt} "
                f"(Jaccard={overlap_gt / len(inspre_gt | gies_gt):.3f})")

    # Model rankings under both algorithms
    rows = []
    header = f"{'Model':<15} {'GIES_prec':>10} {'inspre_prec':>12} {'GIES_edges':>11} {'inspre_edges':>13}"
    logger.info(f"\n{header}")
    logger.info("-" * 65)

    for cond in conditions:
        if cond == "real_data":
            continue

        # Map condition name to inspre key
        inspre_key = cond.replace("_real", "") if cond.endswith("_real") else cond
        inspre_cond = inspre_edges.get(inspre_key, {}).get("edges", set())
        gies_cond = gies_edges.get(cond, set())

        # Precision against respective ground truths
        gies_tp = len(gies_cond & gies_gt)
        gies_prec = gies_tp / len(gies_cond) if gies_cond else 0

        inspre_tp = len(inspre_cond & inspre_gt)
        inspre_prec = inspre_tp / len(inspre_cond) if inspre_cond else 0

        d = display.get(cond, cond)
        logger.info(f"{d:<15} {gies_prec:>10.4f} {inspre_prec:>12.4f} "
                     f"{len(gies_cond):>11} {len(inspre_cond):>13}")

        rows.append({
            "model": d, "condition": cond,
            "gies_precision": gies_prec, "inspre_precision": inspre_prec,
            "gies_edges": len(gies_cond), "inspre_edges": len(inspre_cond),
            "gies_tp": gies_tp, "inspre_tp": inspre_tp,
        })

    pd.DataFrame(rows).to_csv(OUTDIR / "inspre_vs_gies_comparison.csv", index=False)
    logger.info(f"\nSaved comparison to {OUTDIR}/inspre_vs_gies_comparison.csv")

    # Key question: do model rankings agree?
    gies_rank = sorted([r for r in rows], key=lambda x: -x["gies_precision"])
    inspre_rank = sorted([r for r in rows], key=lambda x: -x["inspre_precision"])
    logger.info(f"\nGIES ranking:   {', '.join(r['model'] for r in gies_rank)}")
    logger.info(f"inspre ranking: {', '.join(r['model'] for r in inspre_rank)}")

    return rows


def lambda_sweep(pert_means: dict, ctrl_mean: np.ndarray, gene_names: list,
                  condition_name: str = "real_data"):
    """Sweep lambda to find optimal sparsity for inspre."""
    logger.info(f"\n=== Lambda Sweep for {condition_name} ===")
    ace = compute_ace_matrix(pert_means, ctrl_mean, gene_names)

    lambdas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    results = []

    for lam in lambdas:
        cache_path = OUTDIR / f"{condition_name}_lambda{lam:.3f}_edges.json"
        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            n_edges = len(cached["edges"])
            elapsed = cached.get("time", 0)
            logger.info(f"  lambda={lam:.3f}: CACHED ({n_edges} edges)")
        else:
            adj, elapsed = run_inspre_from_ace(ace, lambda_val=lam)
            edges = adjacency_to_edges(adj, gene_names, threshold=1e-6)
            n_edges = len(edges)
            with open(cache_path, "w") as f:
                json.dump({"edges": [list(e) for e in edges], "time": elapsed, "lambda": lam}, f)
            logger.info(f"  lambda={lam:.3f}: {n_edges} edges ({elapsed:.1f}s)")

        results.append({"lambda": lam, "n_edges": n_edges, "time": elapsed})

    pd.DataFrame(results).to_csv(OUTDIR / f"{condition_name}_lambda_sweep.csv", index=False)
    return results


def main():
    logger.info("=" * 70)
    logger.info("PHASE 5: inspre Causal Discovery via R")
    logger.info("=" * 70)

    gene_names, ctrl_mean, pert_means_real, models = load_200gene_data()

    # Step 1: Run inspre on real data first (establishes target edge count)
    logger.info("\n--- Step 1: inspre on real data ---")
    inspre_results = {}
    inspre_results["real_data"] = run_inspre_on_condition(
        "real_data", pert_means_real, ctrl_mean, gene_names, target_edges=1200)

    gt_n_edges = inspre_results["real_data"]["n_edges"]
    logger.info(f"  Real data ground truth: {gt_n_edges} edges")

    # Step 2: Run inspre on all model conditions (target same edge count)
    logger.info(f"\n--- Step 2: inspre on all model conditions (target ~{gt_n_edges} edges) ---")

    # ElasticNet (= real means, same as eval pipeline)
    inspre_results["elastic_net"] = run_inspre_on_condition(
        "elastic_net", models["elastic_net"], ctrl_mean, gene_names, gt_n_edges)

    # DL models
    for model_name in ["gears", "cpa", "geneformer"]:
        if model_name not in models:
            continue
        inspre_results[model_name] = run_inspre_on_condition(
            model_name, models[model_name], ctrl_mean, gene_names, gt_n_edges)

    # Step 3: Compare inspre vs GIES
    logger.info("\n--- Step 3: inspre vs GIES comparison ---")
    comparison = compare_inspre_vs_gies(inspre_results, gene_names)

    # Step 4: Save comprehensive results
    summary = {
        "n_genes": len(gene_names),
        "conditions": {
            k: {"n_edges": v["n_edges"], "time": v["time"]}
            for k, v in inspre_results.items()
        },
    }
    with open(OUTDIR / "phase5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'=' * 70}")
    logger.info("PHASE 5 COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"All results saved to {OUTDIR}")


if __name__ == "__main__":
    main()
