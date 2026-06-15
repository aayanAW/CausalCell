"""PHASE 1: Investigate why ElasticNet beats deep learning models for causal discovery.

Analyzes linearity, sparsity, covariance preservation, and edge overlap
to explain the ElasticNet anomaly in CausalCellBench.

Saves intermediate results to results/phase1/ as CSV/pickle checkpoints.
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE = Path("/Users/aayanalwani/virtual cells")
DATA_PATH = BASE / "data" / "k562_subset_n200_slim.h5ad"
MODEL_DIR = BASE / "data" / "model_outputs_real_n200"
GIES_CACHE = BASE / "results" / "gies_cache" / "n200_s42_p20"
RESULTS_JSON = BASE / "results" / "eval_results_n200.json"
OUTDIR = BASE / "results" / "phase1"
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load K562 subset and model predictions."""
    logger.info("Loading K562 data...")
    adata = ad.read_h5ad(DATA_PATH)
    adata.var_names = [str(x) for x in adata.var_names]
    gene_names = sorted(adata.var_names.tolist())
    logger.info(f"  Shape: {adata.shape}, {len(gene_names)} genes")

    # Separate control and perturbed
    ctrl_mask = adata.obs["gene"] == "non-targeting"
    ctrl_X = adata[ctrl_mask].X
    if hasattr(ctrl_X, "toarray"):
        ctrl_X = ctrl_X.toarray()
    ctrl_X = np.asarray(ctrl_X, dtype=np.float32)

    # Build per-gene perturbation means from real data
    pert_means = {}
    for gene in gene_names:
        mask = adata.obs["gene"] == gene
        if mask.sum() == 0:
            continue
        X_pert = adata[mask].X
        if hasattr(X_pert, "toarray"):
            X_pert = X_pert.toarray()
        pert_means[gene] = np.mean(X_pert, axis=0).astype(np.float32)

    ctrl_mean = ctrl_X.mean(axis=0)
    logger.info(f"  Control cells: {ctrl_X.shape[0]}, Perturbations: {len(pert_means)}")

    # Load model predictions
    models = {}
    for name in ["gears", "cpa", "geneformer"]:
        path = MODEL_DIR / f"{name}_predictions.h5ad"
        if path.exists():
            pred = ad.read_h5ad(path)
            # Subset to our 200 genes
            gene_idx = [list(pred.var_names).index(g) for g in gene_names if g in pred.var_names]
            pred_genes = [pred.var_names[i] for i in gene_idx]
            pred_X = pred.X[:, gene_idx]
            if hasattr(pred_X, "toarray"):
                pred_X = pred_X.toarray()
            # Build dict: gene -> mean prediction vector (200 genes)
            models[name] = {}
            for i, row_gene in enumerate(pred.obs["perturbation"]):
                if str(row_gene) in set(gene_names):
                    models[name][str(row_gene)] = pred_X[i]
            logger.info(f"  {name}: {len(models[name])} perturbations loaded")

    return adata, ctrl_X, ctrl_mean, pert_means, gene_names, models


def analysis_1_linearity(ctrl_X, ctrl_mean, pert_means, gene_names):
    """Test how linear perturbation effects are across genes."""
    logger.info("\n=== Analysis 1: Linearity of Perturbation Responses ===")

    # For each perturbed gene, compute fold-change vector
    # Then check if fold-changes are linearly predictable from gene identity
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    fold_changes = []
    pert_labels = []
    for gene, mean_expr in pert_means.items():
        if gene not in gene_to_idx:
            continue
        # Subset to 200 genes
        idx = [gene_to_idx[g] for g in gene_names if g in gene_to_idx]
        fc = np.log2((mean_expr[idx] + 1) / (ctrl_mean[idx] + 1))
        fold_changes.append(fc)
        pert_labels.append(gene)

    fc_matrix = np.array(fold_changes)  # (n_perts, n_genes)
    logger.info(f"  Fold-change matrix: {fc_matrix.shape}")

    # Metric 1: What fraction of variance in fold-changes is captured by
    # a rank-1 (linear) approximation? (SVD)
    U, S, Vt = np.linalg.svd(fc_matrix, full_matrices=False)
    total_var = np.sum(S**2)
    cumvar = np.cumsum(S**2) / total_var
    rank1_var = cumvar[0]
    rank5_var = cumvar[min(4, len(cumvar)-1)]
    rank10_var = cumvar[min(9, len(cumvar)-1)]

    logger.info(f"  Rank-1 variance explained: {rank1_var:.4f}")
    logger.info(f"  Rank-5 variance explained: {rank5_var:.4f}")
    logger.info(f"  Rank-10 variance explained: {rank10_var:.4f}")

    # Metric 2: Per-gene R² of linear model predicting fold-change
    # from one-hot perturbation indicator (this is trivially 1.0 for means)
    # Instead: for each TARGET gene j, how well does a linear function of
    # the PERTURBED gene's expression predict the fold-change?
    # This tests if perturbation effects are linearly proportional to
    # the perturbed gene's baseline expression.

    # Metric 3: Sparsity of fold-changes
    # How many genes are significantly affected per perturbation?
    threshold = 0.1  # |log2FC| > 0.1
    n_affected = np.mean(np.abs(fc_matrix) > threshold, axis=1)
    logger.info(f"  Mean fraction of genes affected per pert (|log2FC|>0.1): {np.mean(n_affected):.4f}")
    logger.info(f"  Mean fraction of genes affected per pert (|log2FC|>0.5): {np.mean(np.abs(fc_matrix) > 0.5):.4f}")

    # Metric 4: Distribution of effect sizes
    all_fc = fc_matrix.flatten()
    logger.info(f"  Mean |log2FC|: {np.mean(np.abs(all_fc)):.4f}")
    logger.info(f"  Median |log2FC|: {np.median(np.abs(all_fc)):.4f}")
    logger.info(f"  95th percentile |log2FC|: {np.percentile(np.abs(all_fc), 95):.4f}")

    results = {
        "rank1_variance": float(rank1_var),
        "rank5_variance": float(rank5_var),
        "rank10_variance": float(rank10_var),
        "singular_values_top20": S[:20].tolist(),
        "mean_frac_affected_01": float(np.mean(n_affected)),
        "mean_frac_affected_05": float(np.mean(np.abs(fc_matrix) > 0.5)),
        "mean_abs_log2fc": float(np.mean(np.abs(all_fc))),
        "median_abs_log2fc": float(np.median(np.abs(all_fc))),
        "p95_abs_log2fc": float(np.percentile(np.abs(all_fc), 95)),
        "cumulative_variance": cumvar[:20].tolist(),
    }

    pd.DataFrame({
        "perturbation": pert_labels,
        "mean_abs_fc": np.mean(np.abs(fc_matrix), axis=1),
        "max_abs_fc": np.max(np.abs(fc_matrix), axis=1),
        "n_affected_01": np.sum(np.abs(fc_matrix) > 0.1, axis=1),
        "n_affected_05": np.sum(np.abs(fc_matrix) > 0.5, axis=1),
    }).to_csv(OUTDIR / "perturbation_effect_sizes.csv", index=False)
    logger.info("  Saved perturbation_effect_sizes.csv")

    return results


def analysis_2_sparsity(pert_means, ctrl_mean, gene_names, models):
    """Compare prediction sparsity across models."""
    logger.info("\n=== Analysis 2: Prediction Sparsity ===")

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    idx = [gene_to_idx[g] for g in gene_names]

    results = {}
    for model_name in ["real", "gears", "cpa", "geneformer"]:
        if model_name == "real":
            preds = pert_means
        else:
            preds = models.get(model_name, {})

        if not preds:
            continue

        all_fc = []
        for gene in gene_names:
            if gene not in preds:
                continue
            pred = preds[gene]
            if model_name == "real":
                fc = np.log2((pred[idx] + 1) / (ctrl_mean[idx] + 1))
            else:
                fc = np.log2((pred + 1) / (ctrl_mean[idx] + 1))
            all_fc.append(fc)

        if not all_fc:
            continue

        fc_matrix = np.array(all_fc)
        abs_fc = np.abs(fc_matrix)

        # Sparsity metrics
        gini = _gini_coefficient(abs_fc.flatten())
        l1_norm = np.mean(np.sum(abs_fc, axis=1))
        l2_norm = np.mean(np.sqrt(np.sum(fc_matrix**2, axis=1)))
        frac_near_zero = np.mean(abs_fc < 0.05)
        frac_large = np.mean(abs_fc > 0.5)

        results[model_name] = {
            "gini_coefficient": float(gini),
            "mean_l1_norm": float(l1_norm),
            "mean_l2_norm": float(l2_norm),
            "frac_near_zero": float(frac_near_zero),
            "frac_large_effect": float(frac_large),
            "mean_abs_fc": float(np.mean(abs_fc)),
        }
        logger.info(f"  {model_name}: Gini={gini:.4f}, L1={l1_norm:.1f}, "
                     f"near_zero={frac_near_zero:.3f}, large={frac_large:.4f}")

    return results


def _gini_coefficient(values):
    """Compute Gini coefficient (0=perfect equality, 1=perfect inequality)."""
    values = np.sort(np.abs(values))
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) / (n * np.sum(values))) - (n + 1) / n


def analysis_3_covariance(adata, ctrl_X, gene_names, models, ctrl_mean):
    """Compare covariance structure between real and model-generated data."""
    logger.info("\n=== Analysis 3: Covariance Preservation ===")

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    col_idx = [gene_to_idx[g] for g in gene_names]

    # Real data covariance (from control cells, 200 genes)
    ctrl_200 = ctrl_X[:, col_idx]
    cov_real = np.corrcoef(ctrl_200.T)  # correlation matrix

    results = {}
    for model_name in ["gears", "cpa", "geneformer"]:
        preds = models.get(model_name, {})
        if not preds:
            continue

        # Build synthetic dataset: overlay sampling from model predictions
        # For covariance analysis, use the mean predictions to generate cells
        n_cells_per = 100
        all_cells = []
        rng = np.random.default_rng(42)

        for gene in gene_names:
            if gene not in preds:
                continue
            pred_mean = preds[gene]
            # Overlay: bootstrap control cells + apply fold-change
            boot_idx = rng.choice(ctrl_200.shape[0], size=n_cells_per, replace=True)
            boot_cells = ctrl_200[boot_idx].copy()

            # Compute fold-change
            cm = ctrl_mean[col_idx]
            fc = (pred_mean + 0.01) / (cm + 0.01)
            fc = np.clip(fc, 0.01, 10.0)

            overlay_cells = boot_cells * fc
            all_cells.append(overlay_cells)

        if not all_cells:
            continue

        model_data = np.vstack(all_cells)
        cov_model = np.corrcoef(model_data.T)

        # Frobenius distance
        frob = np.linalg.norm(cov_real - cov_model, "fro")
        # Correlation of correlation matrices (vectorized upper triangle)
        iu = np.triu_indices(len(gene_names), k=1)
        corr_of_corr = np.corrcoef(cov_real[iu], cov_model[iu])[0, 1]
        # Mean absolute difference
        mad = np.mean(np.abs(cov_real - cov_model))

        results[model_name] = {
            "frobenius_distance": float(frob),
            "correlation_of_correlations": float(corr_of_corr),
            "mean_abs_diff": float(mad),
        }
        logger.info(f"  {model_name}: Frob={frob:.2f}, CorrOfCorr={corr_of_corr:.4f}, MAD={mad:.4f}")

    return results


def analysis_4_edge_overlap(gene_names):
    """Analyze edge overlap between model GIES graphs."""
    logger.info("\n=== Analysis 4: Edge Overlap Between Models ===")

    # Load all edge sets
    edge_sets = {}
    for name in ["real_data", "elastic_net", "overlay_resampled_real",
                  "cpa_real", "gears_real", "geneformer_real"]:
        path = GIES_CACHE / f"{name}_edges.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            edge_sets[name] = set(tuple(e) for e in data["edges"])
            logger.info(f"  {name}: {len(edge_sets[name])} edges")

    if "real_data" not in edge_sets:
        logger.warning("  No real_data edges found!")
        return {}

    gt = edge_sets["real_data"]
    results = {}

    # Pairwise Jaccard similarity
    names = list(edge_sets.keys())
    jaccard_matrix = np.zeros((len(names), len(names)))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            s1, s2 = edge_sets[n1], edge_sets[n2]
            if len(s1 | s2) > 0:
                jaccard_matrix[i, j] = len(s1 & s2) / len(s1 | s2)

    jaccard_df = pd.DataFrame(jaccard_matrix, index=names, columns=names)
    jaccard_df.to_csv(OUTDIR / "edge_jaccard_similarity.csv")
    logger.info("  Saved edge_jaccard_similarity.csv")

    # For each model: what edges does it find that real_data doesn't?
    # And what edges does real_data find that it misses?
    display = {"elastic_net": "ElasticNet", "gears_real": "GEARS",
               "cpa_real": "CPA", "geneformer_real": "Geneformer"}

    for key, display_name in display.items():
        if key not in edge_sets:
            continue
        model_edges = edge_sets[key]
        tp = model_edges & gt
        fp = model_edges - gt
        fn = gt - model_edges
        precision = len(tp) / len(model_edges) if model_edges else 0
        recall = len(tp) / len(gt) if gt else 0

        # What edges are unique to this model (not in any other model)?
        other_models = [k for k in display.keys() if k != key and k in edge_sets]
        other_edges = set()
        for k in other_models:
            other_edges |= edge_sets[k]
        unique_fp = fp - other_edges  # FP edges that ONLY this model predicts

        results[key] = {
            "true_positives": len(tp),
            "false_positives": len(fp),
            "false_negatives": len(fn),
            "precision": float(precision),
            "recall": float(recall),
            "unique_false_positives": len(unique_fp),
        }
        logger.info(f"  {display_name}: TP={len(tp)}, FP={len(fp)}, FN={len(fn)}, "
                     f"Prec={precision:.3f}, Rec={recall:.3f}, UniqueFP={len(unique_fp)}")

    # ElasticNet-specific edges: edges found by ElasticNet but NOT by any DL model
    if "elastic_net" in edge_sets:
        dl_edges = set()
        for k in ["gears_real", "cpa_real", "geneformer_real"]:
            if k in edge_sets:
                dl_edges |= edge_sets[k]
        enet_only = edge_sets["elastic_net"] - dl_edges
        enet_only_tp = enet_only & gt
        logger.info(f"\n  ElasticNet-only edges: {len(enet_only)} "
                     f"({len(enet_only_tp)} true positives)")
        results["elasticnet_only"] = {
            "total": len(enet_only),
            "true_positives": len(enet_only_tp),
            "frac_tp": float(len(enet_only_tp) / len(enet_only)) if enet_only else 0,
        }

    return results, jaccard_df


def main():
    logger.info("=" * 70)
    logger.info("PHASE 1: Investigating the ElasticNet Anomaly")
    logger.info("=" * 70)

    adata, ctrl_X, ctrl_mean, pert_means, gene_names, models = load_data()

    # Run all analyses
    linearity = analysis_1_linearity(ctrl_X, ctrl_mean, pert_means, gene_names)
    sparsity = analysis_2_sparsity(pert_means, ctrl_mean, gene_names, models)
    covariance = analysis_3_covariance(adata, ctrl_X, gene_names, models, ctrl_mean)
    overlap_results = analysis_4_edge_overlap(gene_names)

    if isinstance(overlap_results, tuple):
        overlap, jaccard_df = overlap_results
    else:
        overlap = overlap_results
        jaccard_df = None

    # Save all results
    all_results = {
        "linearity": linearity,
        "sparsity": sparsity,
        "covariance": covariance,
        "edge_overlap": overlap,
    }
    with open(OUTDIR / "phase1_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(OUTDIR / "phase1_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 SUMMARY: Why ElasticNet Beats Deep Learning")
    logger.info("=" * 70)

    logger.info(f"\n1. LINEARITY: Rank-1 SVD explains {linearity['rank1_variance']:.1%} of "
                f"fold-change variance. Rank-5 explains {linearity['rank5_variance']:.1%}.")
    if linearity['rank5_variance'] > 0.5:
        logger.info("   -> Perturbation effects are LOW-RANK (mostly linear).")
        logger.info("   -> ElasticNet is well-suited; DL models may overfit non-linear noise.")
    else:
        logger.info("   -> Effects are high-rank. DL models should theoretically excel.")

    logger.info(f"\n2. SPARSITY:")
    for model, sp in sparsity.items():
        logger.info(f"   {model}: Gini={sp['gini_coefficient']:.3f}, "
                     f"near_zero={sp['frac_near_zero']:.1%}, "
                     f"large={sp['frac_large_effect']:.1%}")

    logger.info(f"\n3. COVARIANCE PRESERVATION:")
    for model, cv in covariance.items():
        logger.info(f"   {model}: CorrOfCorr={cv['correlation_of_correlations']:.4f}")

    logger.info(f"\n4. EDGE OVERLAP:")
    if isinstance(overlap, dict) and "elasticnet_only" in overlap:
        eo = overlap["elasticnet_only"]
        logger.info(f"   ElasticNet-only edges: {eo['total']} ({eo['true_positives']} TP, "
                     f"{eo['frac_tp']:.1%} precision)")

    logger.info(f"\nAll results saved to {OUTDIR}")


if __name__ == "__main__":
    main()
