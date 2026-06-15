"""PHASE 5: inspre via subprocess call to official R package.

Uses subprocess to call Rscript (avoids rpy2 API complexity).
Saves ACE matrix as CSV, R reads it, runs inspre, saves result as CSV.
"""

import os
import json
import time
import logging
import subprocess
import tempfile
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path("/workspace")
DATA_PATH = BASE / "k562_subset_n200_slim.h5ad"
MODEL_DIR = BASE / "model_outputs"
GIES_CACHE = BASE / "gies_cache" / "n200_s42_p20"
OUTDIR = BASE / "results" / "phase5_inspre"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Rscript command — use conda env's Rscript directly
RSCRIPT = "/opt/mamba/envs/renv/bin/Rscript"

R_INSPRE_SCRIPT = '''
library(inspre)
args <- commandArgs(trailingOnly = TRUE)
ace_path <- args[1]
out_path <- args[2]
target_edges <- as.integer(args[3])

cat("Loading ACE matrix from", ace_path, "\\n")
R_hat <- as.matrix(read.csv(ace_path, header=FALSE))
n <- nrow(R_hat)
cat("ACE matrix:", n, "x", n, "\\n")

cat("Running inspre...\\n")
t0 <- proc.time()
result <- fit_inspre_from_R(R_hat = R_hat, its = 200, verbose = 1, train_prop = 0.8)
elapsed <- (proc.time() - t0)["elapsed"]
cat("inspre completed in", elapsed, "seconds\\n")

# Extract G_hat (estimated graph) across lambda path
G_hat <- result$G_hat
lambdas <- result$lambda
test_err <- result$test_error

# Select best lambda
valid_te <- !is.nan(test_err) & !is.na(test_err)
if (any(valid_te)) {
    best_idx <- which(valid_te)[which.min(test_err[valid_te])]
    cat("Selected lambda by CV:", lambdas[best_idx], "\\n")
} else {
    # Pick lambda giving closest to target edge count
    edge_counts <- sapply(1:dim(G_hat)[3], function(i) sum(abs(G_hat[,,i]) > 1e-6))
    best_idx <- which.min(abs(edge_counts - target_edges))
    cat("Selected lambda by edge count:", lambdas[best_idx],
        "(", edge_counts[best_idx], "edges, target=", target_edges, ")\\n")
}

G_best <- G_hat[,,best_idx]
cat("Non-zero entries:", sum(abs(G_best) > 1e-6), "\\n")
cat("Lambda:", lambdas[best_idx], "\\n")

# Save adjacency matrix
write.csv(G_best, out_path, row.names=FALSE)

# Also save all edge counts for the lambda path
edge_counts <- sapply(1:dim(G_hat)[3], function(i) sum(abs(G_hat[,,i]) > 1e-6))
meta_path <- paste0(tools::file_path_sans_ext(out_path), "_meta.csv")
write.csv(data.frame(lambda=lambdas, n_edges=edge_counts,
                      test_error=test_err), meta_path, row.names=FALSE)
cat("Saved to", out_path, "and", meta_path, "\\n")
'''


def run_inspre(ace_matrix: np.ndarray, gene_names: list,
               condition_name: str, target_edges: int = 1200) -> dict:
    """Run inspre via R subprocess."""
    cache_path = OUTDIR / f"{condition_name}_inspre_edges.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        edges = set(tuple(e) for e in cached["edges"])
        logger.info(f"  {condition_name}: CACHED ({len(edges)} edges)")
        return {"edges": edges, "time": cached.get("time", 0), "n_edges": len(edges)}

    with tempfile.TemporaryDirectory() as tmpdir:
        ace_path = os.path.join(tmpdir, "ace.csv")
        out_path = os.path.join(tmpdir, "adj.csv")
        r_script_path = os.path.join(tmpdir, "run_inspre.R")

        # Save ACE matrix
        np.savetxt(ace_path, ace_matrix, delimiter=",")

        # Save R script
        with open(r_script_path, "w") as f:
            f.write(R_INSPRE_SCRIPT)

        # Run R
        logger.info(f"  {condition_name}: running inspre via R ({ace_matrix.shape[0]}x{ace_matrix.shape[0]})...")
        t0 = time.time()
        result = subprocess.run(
            f"{RSCRIPT} {r_script_path} {ace_path} {out_path} {target_edges}",
            shell=True, capture_output=True, text=True, timeout=1800,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            logger.error(f"  R failed: {result.stderr[-500:]}")
            return {"edges": set(), "time": elapsed, "n_edges": 0, "error": result.stderr[-200:]}

        logger.info(f"  inspre completed in {elapsed:.1f}s")
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                logger.info(f"    R: {line.strip()}")

        # Read adjacency matrix
        adj = pd.read_csv(out_path).values

        # Convert to edges
        edges = set()
        n = len(gene_names)
        for i in range(n):
            for j in range(n):
                if i != j and abs(adj[i, j]) > 1e-6:
                    edges.add((gene_names[i], gene_names[j]))

        logger.info(f"  {condition_name}: {len(edges)} edges ({elapsed:.1f}s)")

        # Read metadata
        meta_path = os.path.join(tmpdir, "adj_meta.csv")
        meta_df = pd.read_csv(meta_path) if os.path.exists(meta_path) else None

    # Cache
    with open(cache_path, "w") as f:
        json.dump({
            "edges": [list(e) for e in edges],
            "time": elapsed,
            "n_genes": len(gene_names),
            "target_edges": target_edges,
        }, f)

    # Save ACE matrix
    np.save(OUTDIR / f"{condition_name}_ace_matrix.npy", ace_matrix)

    return {"edges": edges, "time": elapsed, "n_edges": len(edges)}


def compute_ace_matrix(pert_means: dict, ctrl_mean: np.ndarray,
                        gene_names: list) -> np.ndarray:
    """ACE[i,j] = mean(X_j | do(gene_i)) - mean(X_j | control)"""
    n = len(gene_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    ace = np.zeros((n, n), dtype=np.float64)
    for gene, mean_expr in pert_means.items():
        if gene not in gene_to_idx:
            continue
        i = gene_to_idx[gene]
        ace[i, :] = mean_expr - ctrl_mean
    return ace


def main():
    logger.info("=" * 70)
    logger.info("PHASE 5: inspre Causal Discovery via R subprocess")
    logger.info("=" * 70)

    # Load data
    logger.info("Loading data...")
    adata = ad.read_h5ad(DATA_PATH)
    adata.var_names = [str(x) for x in adata.var_names]
    gene_names = sorted(adata.var_names.tolist())
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    col_idx = [gene_to_idx[g] for g in gene_names]

    ctrl_mask = adata.obs["gene"] == "non-targeting"
    ctrl_X = adata[ctrl_mask].X
    if hasattr(ctrl_X, "toarray"):
        ctrl_X = ctrl_X.toarray()
    ctrl_X = np.asarray(ctrl_X[:, col_idx], dtype=np.float64)
    ctrl_mean = ctrl_X.mean(axis=0)

    # Real perturbation means
    pert_means_real = {}
    for gene in gene_names:
        mask = adata.obs["gene"] == gene
        if mask.sum() == 0:
            continue
        X = adata[mask].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        pert_means_real[gene] = np.mean(X[:, col_idx], axis=0).astype(np.float64)

    # Model predictions
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

    models["elastic_net"] = pert_means_real

    # Run inspre on all conditions
    conditions = ["real_data", "elastic_net", "gears", "cpa", "geneformer"]
    preds_map = {
        "real_data": pert_means_real,
        "elastic_net": pert_means_real,
        "gears": models.get("gears", {}),
        "cpa": models.get("cpa", {}),
        "geneformer": models.get("geneformer", {}),
    }

    results = {}
    for cond in conditions:
        preds = preds_map[cond]
        if not preds:
            continue
        ace = compute_ace_matrix(preds, ctrl_mean, gene_names)
        target = 1200
        if "real_data" in results:
            target = results["real_data"]["n_edges"]
        results[cond] = run_inspre(ace, gene_names, cond, target_edges=target)

    # Compare with GIES
    logger.info("\n" + "=" * 70)
    logger.info("inspre vs GIES Comparison")
    logger.info("=" * 70)

    inspre_gt = results.get("real_data", {}).get("edges", set())

    # Load GIES edges
    gies_map = {"elastic_net": "elastic_net", "gears": "gears_real",
                "cpa": "cpa_real", "geneformer": "geneformer_real"}
    gies_gt_path = GIES_CACHE / "real_data_edges.json"
    gies_gt = set()
    if gies_gt_path.exists():
        with open(gies_gt_path) as f:
            gies_gt = set(tuple(e) for e in json.load(f)["edges"])

    logger.info(f"Ground truth: inspre={len(inspre_gt)} edges, GIES={len(gies_gt)} edges")
    if inspre_gt and gies_gt:
        overlap = len(inspre_gt & gies_gt)
        logger.info(f"Overlap: {overlap} (Jaccard={overlap/len(inspre_gt | gies_gt):.3f})")

    header = f"{'Model':<15} {'inspre_prec':>12} {'GIES_prec':>10} {'inspre_edges':>13} {'GIES_edges':>11}"
    logger.info(header)
    logger.info("-" * 65)

    rows = []
    for cond in ["elastic_net", "gears", "cpa", "geneformer"]:
        if cond not in results:
            continue
        inspre_edges = results[cond]["edges"]
        inspre_tp = len(inspre_edges & inspre_gt)
        inspre_prec = inspre_tp / len(inspre_edges) if inspre_edges else 0

        gies_key = gies_map.get(cond, cond)
        gies_path = GIES_CACHE / f"{gies_key}_edges.json"
        gies_prec = 0
        gies_n = 0
        if gies_path.exists():
            with open(gies_path) as f:
                gies_edges = set(tuple(e) for e in json.load(f)["edges"])
            gies_tp = len(gies_edges & gies_gt)
            gies_prec = gies_tp / len(gies_edges) if gies_edges else 0
            gies_n = len(gies_edges)

        logger.info(f"{cond:<15} {inspre_prec:>12.4f} {gies_prec:>10.4f} "
                     f"{len(inspre_edges):>13} {gies_n:>11}")
        rows.append({"model": cond, "inspre_precision": inspre_prec,
                      "gies_precision": gies_prec,
                      "inspre_edges": len(inspre_edges), "gies_edges": gies_n})

    pd.DataFrame(rows).to_csv(OUTDIR / "inspre_vs_gies.csv", index=False)

    # Save summary
    summary = {
        "n_genes": len(gene_names),
        "inspre_gt_edges": len(inspre_gt),
        "gies_gt_edges": len(gies_gt),
        "conditions": {k: {"n_edges": v["n_edges"], "time": v["time"]}
                        for k, v in results.items()},
    }
    with open(OUTDIR / "phase5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nAll results saved to {OUTDIR}")
    logger.info("PHASE 5 COMPLETE")


if __name__ == "__main__":
    main()
