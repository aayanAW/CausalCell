"""CausalCellBench evaluation pipeline — runs locally on M4 Mac.

No GPU needed. Runs GIES + GRNBoost2 causal discovery on:
  - Real perturbed cells (upper bound)
  - ElasticNet simulated perturbations
  - Model stubs (CPA, GEARS, scGPT, Geneformer — differentiated)
  - Random baseline (floor)

Usage:
    python3 scripts/run_eval_local.py --n-genes 100 --partition-size 30
    python3 scripts/run_eval_local.py --n-genes 50   # faster test run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
from numpy.polynomial import polynomial as P
from scipy.stats import mannwhitneyu
from scipy.stats import wasserstein_distance as _wasserstein_1d
from sklearn.linear_model import ElasticNet
from sklearn.metrics import average_precision_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("eval_pipeline")

# =====================================================================
# Negative Binomial Sampling
# =====================================================================


def estimate_nb_dispersion(control_X: np.ndarray, shrinkage: bool = True):
    """Estimate per-gene NB dispersion from control expression matrix.

    Parameters
    ----------
    control_X : (n_cells, n_genes) dense float array
    shrinkage : apply DESeq2-style log-linear shrinkage

    Returns
    -------
    mu : (n_genes,) mean expression
    r  : (n_genes,) dispersion parameter
    """
    X = np.asarray(control_X, dtype=np.float64)
    mu = X.mean(axis=0)
    var = X.var(axis=0, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        overdispersed = var > mu + 1e-8
        r = np.where(overdispersed, mu**2 / (var - mu), 1e4)

    r = np.clip(r, 0.1, 1e6)
    zero_mean = mu < 1e-8
    r[zero_mean] = 1.0
    mu[zero_mean] = 1e-8

    if shrinkage and len(mu) >= 20:
        r = _shrink_dispersion(mu, r)

    return mu.astype(np.float64), r.astype(np.float64)


def _shrink_dispersion(mu, r_raw, shrinkage_weight=0.5):
    valid = (mu > 1e-4) & np.isfinite(r_raw) & (r_raw > 0.01) & (r_raw < 1e5)
    if valid.sum() < 10:
        return r_raw
    try:
        coeffs = P.polyfit(np.log(mu[valid] + 1), np.log(r_raw[valid]), deg=1)
        r_trend = np.exp(P.polyval(np.log(mu + 1), coeffs))
        r_trend = np.clip(r_trend, 0.01, 1e6)
    except (np.linalg.LinAlgError, ValueError):
        return r_raw
    log_r_shrunk = (1 - shrinkage_weight) * np.log(
        r_raw + 1e-8
    ) + shrinkage_weight * np.log(r_trend + 1e-8)
    return np.exp(log_r_shrunk)


def sample_cells_nb(mean_prediction, dispersion_r, n_cells, seed=0):
    """Sample cells from NB via Gamma-Poisson compound."""
    mean_prediction = np.nan_to_num(mean_prediction, nan=0.0, posinf=1e4, neginf=0.0)
    rng = np.random.default_rng(seed)
    mu = np.maximum(mean_prediction, 1e-8).astype(np.float64)
    r = np.maximum(dispersion_r, 0.1).astype(np.float64)
    rates = rng.gamma(
        shape=r[np.newaxis, :],
        scale=(mu / r)[np.newaxis, :],
        size=(n_cells, len(mu)),
    )
    return rng.poisson(lam=rates).astype(np.float32)


# =====================================================================
# CausalBenchInput
# =====================================================================


@dataclass
class CausalBenchInput:
    expression_matrix: np.ndarray
    interventions: list
    gene_names: list
    n_control_cells: int
    n_perturbation_cells: int
    perturbed_genes: list

    @property
    def total_cells(self):
        return self.n_control_cells + self.n_perturbation_cells

    def validate(self):
        n_cells, n_genes = self.expression_matrix.shape
        assert n_cells == len(self.interventions), (
            f"Rows {n_cells} != interventions {len(self.interventions)}"
        )
        assert n_genes == len(self.gene_names), (
            f"Cols {n_genes} != gene_names {len(self.gene_names)}"
        )
        assert n_cells == self.total_cells, (
            f"Rows {n_cells} != ctrl+pert {self.total_cells}"
        )
        n_ctrl = sum(1 for i in self.interventions if i == "non-targeting")
        assert n_ctrl == self.n_control_cells
        assert np.isfinite(self.expression_matrix).all(), (
            "Non-finite values in expression matrix"
        )


def build_cb_input(ctrl_X, pert_matrices, pert_labels, gene_names):
    if pert_matrices:
        pert_X = np.concatenate(pert_matrices, axis=0).astype(np.float32)
    else:
        pert_X = np.empty((0, len(gene_names)), dtype=np.float32)
    expression_matrix = np.concatenate([ctrl_X.astype(np.float32), pert_X], axis=0)
    interventions = ["non-targeting"] * ctrl_X.shape[0] + pert_labels
    result = CausalBenchInput(
        expression_matrix=expression_matrix,
        interventions=interventions,
        gene_names=gene_names,
        n_control_cells=ctrl_X.shape[0],
        n_perturbation_cells=pert_X.shape[0],
        perturbed_genes=sorted(set(pert_labels)),
    )
    result.validate()
    return result


# =====================================================================
# GIES Runner
# =====================================================================


def _run_gies_partition(args_tuple):
    """Run GIES on a single partition. Designed for parallel execution."""
    import gies as gies_pkg

    p_idx, partition_idx, gene_names, expression_matrix, interventions = args_tuple

    partition_genes = [gene_names[i] for i in partition_idx]
    partition_gene_set = set(partition_genes)
    n_part = len(partition_idx)

    regimes = {}
    for cell_idx, label in enumerate(interventions):
        regimes.setdefault(label, []).append(cell_idx)

    data_list = []
    intervention_targets_list = []

    # Deterministic tiny jitter (std 1e-3 == N(0,1e-6)) breaks the GIES
    # gauss_int degeneracy (1/omega -> inf -> infinite hill-climbing) on genes
    # with zero variance under a given interventional regime — required for the
    # Norman partitions; below numerical noise for well-conditioned data so it
    # does not measurably alter recovered edge sets (see manuscript appendix).
    _jit = np.random.default_rng(1234 + p_idx)
    for label, cell_indices in regimes.items():
        X_regime = expression_matrix[cell_indices][:, partition_idx].astype(np.float64)
        X_regime = X_regime + _jit.normal(0.0, 1e-3, size=X_regime.shape)
        data_list.append(X_regime)
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


def run_gies(cb_input, seed=0, partition_size=30, n_workers=None):
    """Run GIES with CausalBench-style partitioning (parallelized across partitions)."""
    import multiprocessing as mp

    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 7)

    gene_names = cb_input.gene_names
    n_genes = len(gene_names)

    rng = np.random.default_rng(seed)
    gene_order = rng.permutation(n_genes)
    partitions = [
        gene_order[i : i + partition_size] for i in range(0, n_genes, partition_size)
    ]

    # Build args for parallel execution — all numpy/list, fully picklable
    tasks = [
        (
            p_idx,
            partition_idx,
            gene_names,
            cb_input.expression_matrix,
            cb_input.interventions,
        )
        for p_idx, partition_idx in enumerate(partitions)
    ]

    n_parts = len(partitions)
    logger.info(
        "    Running %d partitions in parallel (%d workers)", n_parts, n_workers
    )

    # Use spawn context — fork + BLAS threads deadlocks under Modal/Docker (V3 plan P8.2)
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=n_workers)
    try:
        results = pool.map(_run_gies_partition, tasks)
    finally:
        pool.close()
        pool.join()

    all_edges = []
    for p_idx, edges, error in results:
        if error:
            logger.warning("  GIES partition %d failed: %s", p_idx, error)
        else:
            all_edges.extend(edges)

    return all_edges


# =====================================================================
# GRNBoost2 Runner (observational)
# =====================================================================


def run_grnboost(cb_input, n_top=None):
    """Run GRNBoost2 — ignores intervention structure, uses all cells.

    Calls arboreto's core regression functions directly (without dask)
    to avoid the dask-expr compatibility bug:
        TypeError: Must supply at least one delayed object
    See: https://github.com/aertslab/arboreto/issues/42
    """
    from arboreto.core import (
        SGBM_KWARGS,
        EARLY_STOP_WINDOW_LENGTH,
        to_tf_matrix,
        infer_partial_network,
    )

    expression_matrix = cb_input.expression_matrix
    gene_names = list(cb_input.gene_names)
    tf_names = gene_names  # all genes as potential TFs (default arboreto behavior)
    n_obs, n_genes = expression_matrix.shape
    logger.info("    GRNBoost2: %d cells x %d genes (dask-free)...", n_obs, n_genes)

    tf_matrix, tf_matrix_gene_names = to_tf_matrix(
        expression_matrix, gene_names, tf_names
    )

    link_dfs = []
    for target_idx in range(n_genes):
        target_gene_name = gene_names[target_idx]
        target_gene_expression = expression_matrix[:, target_idx]

        link_df = infer_partial_network(
            regressor_type="GBM",
            regressor_kwargs=SGBM_KWARGS,
            tf_matrix=tf_matrix,
            tf_matrix_gene_names=tf_matrix_gene_names,
            target_gene_name=target_gene_name,
            target_gene_expression=target_gene_expression,
            include_meta=False,
            early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
            seed=42,
        )
        if link_df is not None and len(link_df) > 0:
            link_dfs.append(link_df)

    if not link_dfs:
        return []

    network = pd.concat(link_dfs, ignore_index=True)
    network = network.sort_values(by="importance", ascending=False)

    edges = [
        (row["TF"], row["target"])
        for _, row in network.iterrows()
        if row["TF"] != row["target"]
    ]
    if n_top is not None:
        edges = edges[:n_top]
    return edges


# =====================================================================
# Evaluation Metrics
# =====================================================================


def compute_set_metrics(
    predicted_edges, ground_truth_edges, all_genes, edge_weights=None, seed: int = 0
) -> dict:
    """Compute F1, Precision, Recall, and AUPRC for predicted vs ground truth edges.

    Primary metrics (set-based, no confidence scores needed):
      - F1, Precision, Recall at the model's operating point
    Secondary metric:
      - AUPRC via top-k sweep (varies k from 10 to N to trace a PR curve)
    Legacy metric (kept for backward compatibility):
      - Shuffled-AUPRC (original method — documented as approximate)

    Parameters
    ----------
    predicted_edges : collection of (source, target) tuples
    ground_truth_edges : set of (source, target) tuples
    all_genes : list of gene names
    edge_weights : optional dict of edge -> float confidence
    seed : random seed for shuffle-based AUPRC
    """
    pred_set = set(predicted_edges)
    gt_set = set(ground_truth_edges)
    n_pred = len(pred_set)
    n_gt = len(gt_set)

    if n_pred == 0:
        return {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "auprc": 0.0,
            "auprc_topk": 0.0,
            "precision_at_100": 0.0,
            "n_predicted": 0,
            "n_ground_truth": n_gt,
            "n_true_positive": 0,
        }

    tp = len(pred_set & gt_set)
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # --- Top-k AUPRC: sweep k to build a proper PR curve ---
    # For unweighted edge sets, vary the number of predicted edges considered
    # to trace precision-recall at different operating points
    predicted_list = list(predicted_edges)
    if edge_weights is not None:
        predicted_list.sort(key=lambda e: edge_weights.get(e, 0.0), reverse=True)
    else:
        # Shuffle to remove partition-order bias, then sweep top-k
        rng_auprc = np.random.default_rng(seed)
        rng_auprc.shuffle(predicted_list)

    # Sweep k from 10 to N in steps, computing precision and recall at each k
    # Anchor at (recall=0, precision=1.0) per standard PR curve convention
    precisions_topk, recalls_topk = [1.0], [0.0]
    step = max(1, n_pred // 50)  # ~50 operating points
    for k in range(step, n_pred + 1, step):
        topk = set(predicted_list[:k])
        tp_k = len(topk & gt_set)
        p_k = tp_k / k
        r_k = tp_k / n_gt if n_gt > 0 else 0.0
        precisions_topk.append(p_k)
        recalls_topk.append(r_k)
    # Always include the full set
    if recalls_topk[-1] != recall:
        precisions_topk.append(precision)
        recalls_topk.append(recall)
    # Sort by recall to ensure monotonic x-axis for trapz
    order = np.argsort(recalls_topk)
    recalls_sorted = np.array(recalls_topk)[order]
    precs_sorted = np.array(precisions_topk)[order]
    auprc_topk = (
        float(np.trapz(precs_sorted, recalls_sorted))
        if len(recalls_sorted) > 1
        else precision
    )

    # --- Legacy shuffled AUPRC (for backward compatibility) ---
    y_true = np.array([1 if e in gt_set else 0 for e in predicted_list])
    if edge_weights is not None:
        y_score = np.array([edge_weights.get(e, 0.0) for e in predicted_list])
    else:
        y_score = np.linspace(1.0, 0.0, len(predicted_list))
    auprc_legacy = (
        float(average_precision_score(y_true, y_score)) if y_true.sum() > 0 else 0.0
    )

    # --- Precision@100 ---
    p100 = (
        float(y_true[: min(100, len(y_true))].sum() / min(100, len(y_true)))
        if len(y_true) > 0
        else 0.0
    )

    return {
        "f1": round(f1, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "auprc": round(auprc_legacy, 6),
        "auprc_topk": round(auprc_topk, 6),
        "precision_at_100": round(p100, 6),
        "n_predicted": n_pred,
        "n_ground_truth": n_gt,
        "n_true_positive": tp,
    }


# Backward-compatible alias
def compute_auprc(
    predicted_edges, ground_truth_edges, all_genes, edge_weights=None, seed=0
):
    """Legacy wrapper — calls compute_set_metrics and returns compatible dict."""
    return compute_set_metrics(
        predicted_edges, ground_truth_edges, all_genes, edge_weights, seed
    )


def compute_random_auprc(ground_truth_edges, all_genes, n_samples=50, seed=0):
    if len(all_genes) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    auprcs = []
    for _ in range(n_samples):
        sources = rng.integers(0, len(all_genes), size=1000)
        targets = rng.integers(0, len(all_genes), size=1000)
        edges = [
            (all_genes[s], all_genes[t]) for s, t in zip(sources, targets) if s != t
        ]
        r = compute_auprc(edges, ground_truth_edges, all_genes)
        auprcs.append(r["auprc"])
    return float(np.mean(auprcs))


def precision_at_k(predicted_edges, ground_truth_edges, k):
    if not predicted_edges or k <= 0:
        return 0.0
    top_k = list(predicted_edges)[:k]
    return sum(1 for e in top_k if e in ground_truth_edges) / min(k, len(top_k))


def structural_hamming_distance(predicted_edges, ground_truth_edges):
    pred_set = set(predicted_edges)
    gt_set = set(ground_truth_edges)
    reversed_edges = {
        (s, t) for s, t in pred_set if (t, s) in gt_set and (s, t) not in gt_set
    }
    missing = gt_set - pred_set - {(t, s) for s, t in reversed_edges}
    extra = pred_set - gt_set - reversed_edges
    shd = len(missing) + len(extra) + len(reversed_edges)
    return {
        "shd": shd,
        "missing": len(missing),
        "extra": len(extra),
        "reversed": len(reversed_edges),
    }


def mean_wasserstein_distance(
    predicted_edges, expression_matrix, interventions, gene_names, max_edges=500
):
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    ctrl_idx = [i for i, v in enumerate(interventions) if v == "non-targeting"]
    pert_groups = {}
    for i, v in enumerate(interventions):
        if v != "non-targeting":
            pert_groups.setdefault(v, []).append(i)

    edges_to_eval = list(predicted_edges)[:max_edges]
    distances = []
    for src, tgt in edges_to_eval:
        if src not in pert_groups or tgt not in gene_to_idx:
            continue
        tgt_idx = gene_to_idx[tgt]
        pert_vals = expression_matrix[pert_groups[src], tgt_idx]
        ctrl_vals = expression_matrix[ctrl_idx, tgt_idx]
        if len(pert_vals) < 5 or len(ctrl_vals) < 5:
            continue
        distances.append(_wasserstein_1d(ctrl_vals, pert_vals))

    if not distances:
        return {"mean_wasserstein": 0.0, "n_evaluated": 0}
    return {
        "mean_wasserstein": float(np.mean(distances)),
        "n_evaluated": len(distances),
    }


def false_omission_rate(
    predicted_edges, expression_matrix, interventions, gene_names, alpha: float = 0.05
) -> dict:
    """Compute False Omission Rate with GLOBAL Benjamini-Hochberg correction.

    FOR = fraction of non-predicted edges that show significant differential
    expression (i.e., edges the model missed but are biologically real).

    FDR correction is applied GLOBALLY across all source-target pairs tested
    (not per-source gene), which is the statistically correct approach for
    controlling the false discovery rate across ~N^2 tests.
    """
    predicted_set = set(predicted_edges)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    ctrl_idx = np.array(
        [i for i, v in enumerate(interventions) if v == "non-targeting"]
    )
    pert_groups = {}
    for i, v in enumerate(interventions):
        if v != "non-targeting":
            pert_groups.setdefault(v, []).append(i)
    pert_groups = {k: np.array(v) for k, v in pert_groups.items()}

    # Collect ALL p-values globally across all source-target pairs
    p_values_list = []
    test_pairs = []  # track which (src, tgt) each p-value corresponds to
    for src in sorted(pert_groups.keys()):
        if src not in gene_to_idx:
            continue
        src_cells = pert_groups[src]
        if len(src_cells) < 5:
            continue
        for tgt in gene_names:
            if tgt == src or (src, tgt) in predicted_set:
                continue
            tgt_idx = gene_to_idx[tgt]
            try:
                _, p = mannwhitneyu(
                    expression_matrix[src_cells, tgt_idx],
                    expression_matrix[ctrl_idx, tgt_idx],
                    alternative="two-sided",
                )
                p_values_list.append(p)
                test_pairs.append((src, tgt))
            except ValueError:
                continue

    if not p_values_list:
        return {"for_rate": 0.0, "n_false_omissions": 0, "n_tested": 0}

    p_arr = np.array(p_values_list)
    # Global Benjamini-Hochberg across ALL tests
    n = len(p_arr)
    sorted_idx = np.argsort(p_arr)
    sorted_p = p_arr[sorted_idx]
    thresholds = alpha * np.arange(1, n + 1) / n
    below = sorted_p <= thresholds
    if not below.any():
        n_false_omissions = 0
    else:
        # BH: find the largest k such that p_(k) <= alpha * k / n
        max_k = np.max(np.where(below)[0])
        # All tests with rank <= max_k are significant
        significant_idx = sorted_idx[: max_k + 1]
        n_false_omissions = len(significant_idx)

    return {
        "for_rate": n_false_omissions / n,
        "n_false_omissions": n_false_omissions,
        "n_tested": n,
    }


# =====================================================================
# Perturbation Cell Sampling (covariance-preserving)
# =====================================================================


def sample_perturbed_cells_overlay(ctrl_X, mean_pred, n_cells, seed=0):
    """Generate perturbed cells by overlaying fold-changes on control cells.

    CRITICAL FIX: Independent NB sampling destroys gene-gene correlations,
    causing GIES to find ~0 edges even from correct perturbation means.
    This method preserves the control covariance structure by:
    1. Bootstrap sampling from control cells
    2. Scaling each gene by the predicted fold-change ratio
    3. Adding Poisson noise for count-level variability

    This gives GIES the cross-gene distributional shifts it needs.
    """
    rng = np.random.default_rng(seed)
    n_ctrl = ctrl_X.shape[0]

    # Bootstrap from controls
    idx = rng.choice(n_ctrl, size=n_cells, replace=True)
    cells = ctrl_X[idx].copy().astype(np.float64)

    # Compute fold-change ratios
    ctrl_means = ctrl_X.mean(axis=0).astype(np.float64)
    ratios = np.ones_like(ctrl_means)
    # Use low threshold (1e-6) to include low-expression genes;
    # pseudocount (0.01) prevents division-by-zero without losing signal
    nonzero = ctrl_means > 1e-6
    ctrl_means_safe = np.maximum(ctrl_means, 0.01)
    ratios[nonzero] = np.maximum(mean_pred[nonzero], 0.0) / ctrl_means_safe[nonzero]
    ratios = np.clip(ratios, 0.01, 10.0)  # prevent extreme scaling

    # Apply fold-change to each cell (preserves covariance structure)
    cells *= ratios[np.newaxis, :]

    # Add Poisson noise for count-level variability
    cells = np.maximum(cells, 0.0)
    cells = rng.poisson(lam=cells).astype(np.float32)

    return cells


# =====================================================================
# Real Model Output Loading
# =====================================================================


def load_real_model_predictions(
    model_outputs_dir, model_name, gene_subset, ctrl_X, n_cells, seed=0
):
    """Load real model predictions from .h5ad and prepare for GIES evaluation.

    Handles two output types:
      - "mean": one row per perturbation → overlay-sample to get cells
      - "cells": multiple cells per perturbation → use directly

    Returns (pert_matrices, pert_labels) or (None, None) if file missing.
    """
    filename = f"{model_name}_predictions.h5ad"
    filepath = os.path.join(model_outputs_dir, filename)
    if not os.path.exists(filepath):
        logger.warning("  Model output not found: %s", filepath)
        return None, None

    logger.info("  Loading %s", filepath)
    pred_adata = ad.read_h5ad(filepath)
    output_type = pred_adata.uns.get("output_type", "mean")
    logger.info("    Shape: %s, output_type: %s", pred_adata.shape, output_type)

    # Build gene name → index mapping for the gene_subset
    pred_var = list(pred_adata.var_names)
    pred_var_to_idx = {g: i for i, g in enumerate(pred_var)}  # O(1) lookup
    pred_var_upper = {
        g.upper(): i for i, g in enumerate(pred_var)
    }  # case-insensitive fallback
    gene_idx_map = {}
    for g in gene_subset:
        if g in pred_var_to_idx:
            gene_idx_map[g] = pred_var_to_idx[g]
        elif g.upper() in pred_var_upper:
            gene_idx_map[g] = pred_var_upper[g.upper()]

    if len(gene_idx_map) < len(gene_subset) * 0.5:
        logger.warning(
            "    Only %d/%d genes matched in %s — skipping",
            len(gene_idx_map),
            len(gene_subset),
            model_name,
        )
        return None, None

    logger.info("    Gene overlap: %d/%d", len(gene_idx_map), len(gene_subset))

    # Get perturbation labels from obs
    pert_col = None
    for col in ["perturbation", "condition", "gene", "pert"]:
        if col in pred_adata.obs.columns:
            pert_col = col
            break
    if pert_col is None:
        logger.warning("    No perturbation column found in %s — skipping", model_name)
        return None, None

    pert_matrices = []
    pert_labels = []

    if output_type == "cells":
        # CPA-style: multiple cells per perturbation already in X
        for gene in gene_subset:
            if gene not in gene_idx_map:
                continue
            mask = pred_adata.obs[pert_col] == gene
            if mask.sum() == 0:
                # Try case-insensitive
                mask = pred_adata.obs[pert_col].str.upper() == gene.upper()
            if mask.sum() == 0:
                continue
            gene_cells = pred_adata[mask]
            X_g = gene_cells.X
            if hasattr(X_g, "toarray"):
                X_g = X_g.toarray()
            # Extract only the columns corresponding to gene_subset
            col_indices = [gene_idx_map[g] for g in gene_subset if g in gene_idx_map]
            expr = np.asarray(X_g[:, col_indices], dtype=np.float32)
            # Subsample if too many cells
            if expr.shape[0] > n_cells:
                gene_seed = seed + int(hashlib.md5(gene.encode()).hexdigest()[:8], 16)
                rng_sub = np.random.default_rng(gene_seed)
                expr = expr[rng_sub.choice(expr.shape[0], size=n_cells, replace=False)]
            # Pad columns if gene_subset has genes missing from model output
            if expr.shape[1] < len(gene_subset):
                padded = np.zeros((expr.shape[0], len(gene_subset)), dtype=np.float32)
                j = 0
                for gi, g in enumerate(gene_subset):
                    if g in gene_idx_map:
                        padded[:, gi] = expr[:, j]
                        j += 1
                    else:
                        # Fill with control mean for missing genes
                        padded[:, gi] = ctrl_X[:, gi].mean()
                expr = padded
            # Guard against NaN/Inf from model outputs
            expr = np.nan_to_num(expr, nan=0.0, posinf=1e4, neginf=0.0)
            pert_matrices.append(expr)
            pert_labels.extend([gene] * expr.shape[0])
    else:
        # Mean-prediction models: one row per perturbation → overlay sample
        for gene in gene_subset:
            if gene not in gene_idx_map:
                continue
            mask = pred_adata.obs[pert_col] == gene
            if mask.sum() == 0:
                mask = pred_adata.obs[pert_col].str.upper() == gene.upper()
            if mask.sum() == 0:
                continue
            # Get mean prediction (take first matching row if multiple)
            mean_row = pred_adata[mask].X
            if hasattr(mean_row, "toarray"):
                mean_row = mean_row.toarray()
            mean_row = np.asarray(mean_row[0], dtype=np.float64)
            # Map to gene_subset columns
            mean_pred = np.zeros(len(gene_subset), dtype=np.float64)
            for gi, g in enumerate(gene_subset):
                if g in gene_idx_map:
                    mean_pred[gi] = mean_row[gene_idx_map[g]]
                else:
                    mean_pred[gi] = ctrl_X[:, gi].mean()
            # Guard against NaN/Inf — replace with control mean (no perturbation effect)
            nan_mask = ~np.isfinite(mean_pred)
            if nan_mask.any():
                mean_pred[nan_mask] = ctrl_X[:, nan_mask].mean(axis=0)
            # Overlay sample from means
            gene_seed = seed + int(hashlib.md5(gene.encode()).hexdigest()[:8], 16)
            sampled = sample_perturbed_cells_overlay(
                ctrl_X, mean_pred, n_cells, seed=gene_seed
            )
            pert_matrices.append(sampled)
            pert_labels.extend([gene] * n_cells)

    if not pert_matrices:
        logger.warning("    No perturbations matched for %s", model_name)
        return None, None

    logger.info(
        "    Loaded %d perturbations, %d total cells",
        len(pert_matrices),
        sum(m.shape[0] for m in pert_matrices),
    )
    return pert_matrices, pert_labels


# =====================================================================
# Baselines and Model Stubs
# =====================================================================


def run_elasticnet_baseline(ctrl_X_train, ctrl_X_sample, gene_subset, n_cells, seed=0):
    """Train ElasticNet on controls, predict perturbation means, overlay-sample.

    Uses ctrl_X_train (full controls) for training ElasticNet models,
    and ctrl_X_sample (subsampled controls) for generating perturbed cells.
    """
    logger.info("  Training %d ElasticNet models...", len(gene_subset))
    control_means = ctrl_X_train.mean(axis=0)
    n_genes = len(gene_subset)

    models = {}
    for g_idx in range(n_genes):
        feature_idx = [i for i in range(n_genes) if i != g_idx]
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=0)
        model.fit(ctrl_X_train[:, feature_idx], ctrl_X_train[:, g_idx])
        models[g_idx] = model

    pert_matrices, pert_labels = [], []
    for g_idx, gene in enumerate(gene_subset):
        input_state = control_means.copy()
        input_state[g_idx] *= 0.1  # ~90% CRISPRi knockdown

        mean_pred = input_state.copy()
        for t_idx, enet_model in models.items():
            if t_idx == g_idx:
                continue
            feature_idx = [i for i in range(n_genes) if i != t_idx]
            mean_pred[t_idx] = max(
                float(enet_model.predict(input_state[feature_idx].reshape(1, -1))[0]),
                0.0,
            )

        gene_seed = seed + int(hashlib.md5(gene.encode()).hexdigest()[:8], 16)
        sampled = sample_perturbed_cells_overlay(
            ctrl_X_sample, mean_pred, n_cells, seed=gene_seed
        )
        pert_matrices.append(sampled)
        pert_labels.extend([gene] * n_cells)

    return pert_matrices, pert_labels


def run_random_baseline(ctrl_X, gene_subset, n_cells, seed=0):
    """Random baseline: bootstrap control cells with no perturbation effect.

    Each 'perturbation' regime is just a resampling of control cells.
    GIES should find no edges because there's no distributional shift.
    """
    rng = np.random.default_rng(seed)
    pert_matrices, pert_labels = [], []
    for gene in gene_subset:
        # Pure bootstrap: no fold-change, just resampled controls
        idx = rng.choice(ctrl_X.shape[0], size=n_cells, replace=True)
        cells = ctrl_X[idx].copy()
        pert_matrices.append(cells)
        pert_labels.extend([gene] * n_cells)
    return pert_matrices, pert_labels


def run_model_stub(
    ctrl_X_train,
    ctrl_X_sample,
    gene_subset,
    n_cells,
    model_name,
    knockdown_factor=0.1,
    noise_scale=0.0,
    downstream_frac=0.0,
    downstream_strength=0.0,
    seed=0,
):
    """Differentiated model stub with covariance-preserving overlay sampling.

    Parameters
    ----------
    ctrl_X_train : array (n_full, n_genes)
        Full control matrix for computing means and correlations.
    ctrl_X_sample : array (n_sub, n_genes)
        Subsampled controls for overlay sampling base (must match GIES controls).

    Better stubs model how well each model captures downstream gene effects:
    - knockdown_factor: direct CRISPRi knockdown (perturbed gene)
    - downstream_frac: fraction of other genes that show secondary effects
    - downstream_strength: magnitude of secondary effects (fold-change from ctrl)
    - noise_scale: model-specific prediction noise

    Uses overlay sampling (not independent NB) to preserve gene-gene correlations.
    """
    control_means = ctrl_X_train.mean(axis=0)
    # Compute gene-gene correlation from full controls for downstream propagation
    ctrl_corr = np.corrcoef(ctrl_X_train.T)  # (n_genes, n_genes)
    rng = np.random.default_rng(seed)
    n_genes = len(gene_subset)
    pert_matrices, pert_labels = [], []

    for g_idx, gene in enumerate(gene_subset):
        mean_pred = control_means.copy()
        # Direct knockdown
        mean_pred[g_idx] *= knockdown_factor

        # Downstream effects: propagate through correlated genes
        if downstream_frac > 0 and downstream_strength > 0:
            correlations = np.abs(ctrl_corr[g_idx])
            correlations[g_idx] = 0  # don't double-count target
            n_downstream = max(1, int(n_genes * downstream_frac))
            top_corr_idx = np.argsort(correlations)[-n_downstream:]
            for j in top_corr_idx:
                sign = np.sign(ctrl_corr[g_idx, j])
                # Downstream genes shift proportional to correlation * strength
                shift = sign * correlations[j] * downstream_strength
                mean_pred[j] *= max(1.0 + shift * (knockdown_factor - 1.0), 0.05)

        # Model-specific noise on means
        if noise_scale > 0:
            noise = rng.normal(0, noise_scale * np.abs(control_means), size=n_genes)
            mean_pred = np.maximum(mean_pred + noise, 0.0)

        # Overlay sampling: preserves control covariance structure
        # Uses ctrl_X_sample (subsampled) so bootstrap base matches GIES controls
        gene_seed = seed + int(hashlib.md5(gene.encode()).hexdigest()[:8], 16)
        sampled = sample_perturbed_cells_overlay(
            ctrl_X_sample, mean_pred, n_cells, seed=gene_seed
        )
        pert_matrices.append(sampled)
        pert_labels.extend([gene] * n_cells)

    return pert_matrices, pert_labels


# Model stubs with different downstream propagation behaviors:
# - CPA: learns perturbation embeddings → moderate downstream capture
# - GEARS: GNN on gene network → best downstream propagation
# - scGPT: transformer masking → some downstream via attention
# - Geneformer: rank-value encoding → weakest downstream (embedding-based)
MODEL_CONFIGS = {
    "cpa": {
        "knockdown_factor": 0.10,
        "noise_scale": 0.02,
        "downstream_frac": 0.15,
        "downstream_strength": 0.4,
        "seed": 100,
    },
    "gears": {
        "knockdown_factor": 0.10,
        "noise_scale": 0.03,
        "downstream_frac": 0.25,
        "downstream_strength": 0.6,
        "seed": 200,
    },
    "scgpt": {
        "knockdown_factor": 0.05,
        "noise_scale": 0.03,
        "downstream_frac": 0.10,
        "downstream_strength": 0.3,
        "seed": 300,
    },
    "geneformer": {
        "knockdown_factor": 0.15,
        "noise_scale": 0.04,
        "downstream_frac": 0.05,
        "downstream_strength": 0.2,
        "seed": 400,
    },
}


# =====================================================================
# Prediction Accuracy Metrics
# =====================================================================


def compute_prediction_accuracy(
    model_pert_matrices: list,
    model_pert_labels: list,
    real_pert_matrices: list,
    real_pert_labels: list,
    gene_subset: list,
) -> dict:
    """Compare model predictions to real perturbation means.

    Computes MSE, Pearson r, and cosine similarity between predicted and
    observed mean expression vectors across all perturbations.

    This disentangles prediction accuracy from causal graph quality:
    if a model predicts well but has poor causal graphs, that's informative.
    """
    from scipy.stats import pearsonr
    from scipy.spatial.distance import cosine as cosine_dist

    # Group matrices by gene — each matrix corresponds to one gene's cells
    real_by_gene = {}
    offset = 0
    for mat in real_pert_matrices:
        n = mat.shape[0]
        if n == 0:
            continue
        gene = real_pert_labels[offset]
        real_by_gene[gene] = mat.mean(axis=0)
        offset += n

    model_by_gene = {}
    offset = 0
    for mat in model_pert_matrices:
        n = mat.shape[0]
        if n == 0:
            continue
        gene = model_pert_labels[offset]
        model_by_gene[gene] = mat.mean(axis=0)
        offset += n

    # Compute metrics over shared genes
    shared = sorted(set(real_by_gene) & set(model_by_gene))
    if len(shared) < 2:
        return {
            "mse": float("nan"),
            "pearson_r": float("nan"),
            "cosine_sim": float("nan"),
            "n_genes_compared": len(shared),
        }

    real_vecs = np.array([real_by_gene[g] for g in shared])
    model_vecs = np.array([model_by_gene[g] for g in shared])

    # Per-gene MSE then average
    mse = float(np.mean((real_vecs - model_vecs) ** 2))

    # Flatten for global Pearson r
    r_val, _ = pearsonr(real_vecs.ravel(), model_vecs.ravel())

    # Mean cosine similarity across genes
    cos_sims = []
    for i in range(len(shared)):
        norm_r = np.linalg.norm(real_vecs[i])
        norm_m = np.linalg.norm(model_vecs[i])
        if norm_r > 0 and norm_m > 0:
            cos_sims.append(1.0 - cosine_dist(real_vecs[i], model_vecs[i]))
    mean_cos = float(np.mean(cos_sims)) if cos_sims else float("nan")

    return {
        "mse": round(mse, 6),
        "pearson_r": round(float(r_val), 6),
        "cosine_sim": round(mean_cos, 6),
        "n_genes_compared": len(shared),
    }


# =====================================================================
# Bootstrap Confidence Intervals
# =====================================================================


def bootstrap_evaluate(
    edges_list: list,
    gt_edges: set,
    gene_subset: list,
    n_bootstrap: int = 200,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Compute bootstrap 95% CIs for set-based metrics.

    Resamples predicted edges (with replacement) and computes metrics
    on each bootstrap sample. Reports mean, std, and CI bounds.

    Parameters
    ----------
    edges_list : list of (source, target) tuples — the predicted edges
    gt_edges : set of ground truth edges
    gene_subset : list of gene names
    n_bootstrap : number of bootstrap iterations
    seed : random seed
    confidence : confidence level (default 0.95 for 95% CI)
    """
    rng = np.random.default_rng(seed)
    n_edges = len(edges_list)

    if n_edges == 0:
        empty = {"mean": 0.0, "std": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
        return {m: empty.copy() for m in ["f1", "precision", "recall", "auprc"]}

    boot_metrics = {m: [] for m in ["f1", "precision", "recall", "auprc"]}

    for b in range(n_bootstrap):
        # Resample edges with replacement
        idx = rng.choice(n_edges, size=n_edges, replace=True)
        boot_edges = [edges_list[i] for i in idx]
        # Deduplicate (resampling may duplicate edges)
        boot_edges_unique = list(set(boot_edges))

        m = compute_set_metrics(boot_edges_unique, gt_edges, gene_subset, seed=seed + b)
        for k in boot_metrics:
            boot_metrics[k].append(m[k])

    alpha = 1.0 - confidence
    result = {}
    for metric_name, values in boot_metrics.items():
        arr = np.array(values)
        result[metric_name] = {
            "mean": round(float(np.mean(arr)), 6),
            "std": round(float(np.std(arr)), 6),
            "ci_lo": round(float(np.percentile(arr, 100 * alpha / 2)), 6),
            "ci_hi": round(float(np.percentile(arr, 100 * (1 - alpha / 2))), 6),
        }

    return result


def pairwise_permutation_test(
    edges_a: list,
    edges_b: list,
    gt_edges: set,
    gene_subset: list,
    n_perm: int = 1000,
    seed: int = 0,
) -> dict:
    """DEPRECATED / STATISTICALLY INVALID for single-seed edge sets.

    This pools edges_a ∪ edges_b and repartitions the union, which tests
    "is a given edge assigned to A vs B" under reshuffling of two FIXED edge
    sets — NOT "does predictor A recover the ground truth better than B".
    On single-seed inputs the observed difference is compared against a null
    built from the same two fixed sets, so it returns p ≈ 1.0 and cannot
    reject anything. It must NOT be used for any reported significance claim.

    The valid pairwise test is the paired sign-flip permutation ACROSS SEEDS
    in scripts/run_multi_seed.py (used for all K562 multi-seed p-values). For
    a single-seed cross-dataset comparison, bootstrap over cells/perturbations
    instead, or report point estimates against the permutation chance band
    only (the manuscript treats single-seed cells as suggestive, not
    significant).
    """
    warnings.warn(
        "pairwise_permutation_test pools A∪B and is invalid for single-seed "
        "edge sets (returns p≈1.0); do not cite its p-value. Use the "
        "across-seed paired sign-flip test in run_multi_seed.py.",
        RuntimeWarning,
        stacklevel=2,
    )
    rng = np.random.default_rng(seed)

    m_a = compute_set_metrics(edges_a, gt_edges, gene_subset)
    m_b = compute_set_metrics(edges_b, gt_edges, gene_subset)
    observed_diff = m_a["f1"] - m_b["f1"]

    # Pool all edges from both conditions
    all_edges = list(set(edges_a) | set(edges_b))
    n_a = len(set(edges_a))
    n_total = len(all_edges)

    perm_diffs = []
    for _ in range(n_perm):
        perm_idx = rng.permutation(n_total)
        perm_a = [all_edges[i] for i in perm_idx[:n_a]]
        perm_b = [all_edges[i] for i in perm_idx[n_a:]]
        pm_a = compute_set_metrics(perm_a, gt_edges, gene_subset)
        pm_b = compute_set_metrics(perm_b, gt_edges, gene_subset)
        perm_diffs.append(pm_a["f1"] - pm_b["f1"])

    perm_diffs = np.array(perm_diffs)
    p_value = float(np.mean(np.abs(perm_diffs) >= abs(observed_diff)))

    return {
        "f1_a": m_a["f1"],
        "f1_b": m_b["f1"],
        "observed_diff": round(observed_diff, 6),
        "p_value": round(p_value, 6),
        "significant_005": p_value < 0.05,
    }


# =====================================================================
# Full Evaluation for a Condition
# =====================================================================


def evaluate_condition(name, edges, cb_input, gt_edges, gene_subset, wall_time=0.0):
    """Compute all metrics for a condition.

    Primary: F1, Precision, Recall (set-based, no confidence scores needed)
    Secondary: AUPRC (top-k sweep), P@100, SHD
    Distributional: Wasserstein distance, False Omission Rate
    """
    metrics = compute_set_metrics(edges, gt_edges, gene_subset)
    shd = structural_hamming_distance(edges, gt_edges)
    ws = mean_wasserstein_distance(
        edges, cb_input.expression_matrix, cb_input.interventions, gene_subset
    )
    fr = false_omission_rate(
        edges, cb_input.expression_matrix, cb_input.interventions, gene_subset
    )

    return {
        "n_edges": len(edges),
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "auprc": metrics["auprc"],
        "auprc_topk": metrics["auprc_topk"],
        "precision_at_100": metrics["precision_at_100"],
        "n_true_positive": metrics["n_true_positive"],
        "shd": shd,
        "wasserstein": ws,
        "false_omission": fr,
        "wall_time": wall_time,
    }


# =====================================================================
# Main Pipeline
# =====================================================================


def main(args):
    t_start = time.time()

    logger.info("=" * 70)
    logger.info("CausalCellBench: Local Evaluation Pipeline (M4 Mac)")
    logger.info("=" * 70)
    logger.info(
        "  n_genes=%d, partition_size=%d, n_cells=%d, seed=%d",
        args.n_genes,
        args.partition_size,
        args.n_cells,
        args.seed,
    )

    # ------------------------------------------------------------------
    # Step 1: Load K562
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading K562 data")
    adata = ad.read_h5ad(args.data_path)
    logger.info("  Shape: %s", adata.shape)

    # Map Ensembl IDs → gene symbols (skip if pre-subset)
    if getattr(args, "pre_subset", False):
        logger.info("  Pre-subset mode: skipping Ensembl mapping")
        adata.var_names = [str(x) for x in adata.var_names]
    elif str(adata.var_names[0]).startswith("ENSG"):
        logger.info("  Mapping Ensembl IDs to gene symbols via .var['gene_name']")
        symbols = adata.var["gene_name"].values
        seen = {}
        unique_symbols = []
        for s in symbols:
            s = str(s)
            if s in seen:
                seen[s] += 1
                unique_symbols.append(f"{s}_{seen[s]}")
            else:
                seen[s] = 0
                unique_symbols.append(s)
        adata.var_names = unique_symbols
        adata.var_names_make_unique()
        logger.info("  Sample var_names: %s", list(adata.var_names[:5]))

    # ------------------------------------------------------------------
    # Step 2: Extract controls and select genes
    # ------------------------------------------------------------------
    logger.info("Step 2: Extracting controls and selecting genes")

    pert_col = getattr(
        args, "pert_col", "gene"
    )  # Replogle K562='gene'; Norman/RPE1='perturbation'
    assert pert_col in adata.obs.columns, (
        f"--pert-col '{pert_col}' not in obs {list(adata.obs.columns)}"
    )

    # Find control cells
    control_mask = adata.obs[pert_col] == "non-targeting"
    if control_mask.sum() == 0:
        for label in ["control", "ctrl", "NT"]:
            control_mask = adata.obs[pert_col] == label
            if control_mask.sum() > 0:
                break
    assert control_mask.sum() > 0, "No control cells found"

    control = adata[control_mask].copy()
    logger.info("  Control cells: %d", control.n_obs)

    if getattr(args, "pre_subset", False):
        # Pre-subset mode: all var_names are the gene subset
        gene_subset = sorted(adata.var_names.tolist())
        logger.info(
            "  Pre-subset mode: using all %d var_names as gene subset", len(gene_subset)
        )
    else:
        # Find perturbation-eligible genes (≥10 cells AND in var_names)
        pert_counts = adata.obs[pert_col].value_counts()
        var_set = set(adata.var_names)
        eligible = sorted(
            [
                g
                for g, c in pert_counts.items()
                if g != "non-targeting" and c >= 10 and g in var_set
            ]
        )
        logger.info("  Perturbation-eligible genes: %d", len(eligible))

        # Select gene subset
        rng = np.random.default_rng(args.seed)
        if args.gene_list and os.path.exists(args.gene_list):
            with open(args.gene_list) as f:
                requested = [line.strip() for line in f if line.strip()]
            eligible_set = set(eligible)
            gene_subset = sorted([g for g in requested if g in eligible_set])
            logger.info(
                "  From gene list: %d/%d genes found in eligible set",
                len(gene_subset),
                len(requested),
            )
        else:
            gene_subset = sorted(
                rng.choice(
                    eligible, size=min(args.n_genes, len(eligible)), replace=False
                )
            )
    logger.info("  Selected %d genes", len(gene_subset))

    # Build aligned control matrix
    # CRITICAL: Subsample controls to avoid dominating GIES inference.
    # CausalBench uses matched control counts (~200). With 10K+ controls
    # vs 100 pert cells/gene, control covariance drowns perturbation signal.
    gene_to_varidx = {g: i for i, g in enumerate(adata.var_names)}
    col_idx = [gene_to_varidx[g] for g in gene_subset]

    X_ctrl_full = control.X
    if hasattr(X_ctrl_full, "toarray"):
        X_ctrl_full = X_ctrl_full.toarray()
    X_ctrl_full = np.asarray(X_ctrl_full[:, col_idx], dtype=np.float32)

    # Subsample controls: use n_ctrl (default 200) to balance with pert cells
    n_ctrl = min(args.n_ctrl, X_ctrl_full.shape[0])
    rng_ctrl = np.random.default_rng(args.seed)
    ctrl_idx = rng_ctrl.choice(X_ctrl_full.shape[0], size=n_ctrl, replace=False)
    ctrl_X = X_ctrl_full[ctrl_idx]
    logger.info(
        "  Control matrix: %s (subsampled from %d)", ctrl_X.shape, X_ctrl_full.shape[0]
    )
    # Keep full control matrix for dispersion estimation (needs all cells)
    ctrl_X_full = X_ctrl_full

    # ------------------------------------------------------------------
    # Step 3: Estimate NB dispersion
    # ------------------------------------------------------------------
    logger.info(
        "Step 3: Estimating NB dispersion (from all %d control cells)",
        ctrl_X_full.shape[0],
    )
    mu_disp, r_disp = estimate_nb_dispersion(ctrl_X_full)
    logger.info("  Median r=%.2f, mean mu=%.2f", np.median(r_disp), np.mean(mu_disp))

    # ------------------------------------------------------------------
    # Step 4: Generate all conditions
    # ------------------------------------------------------------------
    conditions_data = {}

    # --- Real perturbed cells (upper bound) ---
    logger.info("Step 4a: Extracting real perturbed cells")
    t0 = time.time()
    real_pert_matrices, real_pert_labels = [], []
    for gene in gene_subset:
        mask = adata.obs[pert_col] == gene
        if mask.sum() < 5:
            continue
        gene_cells = adata[mask]
        X_g = gene_cells.X
        if hasattr(X_g, "toarray"):
            X_g = X_g.toarray()
        expr = np.asarray(X_g[:, col_idx], dtype=np.float32)
        if expr.shape[0] > args.n_cells:
            gene_seed_sub = args.seed + int(
                hashlib.md5(gene.encode()).hexdigest()[:8], 16
            )
            rng_sub = np.random.default_rng(gene_seed_sub)
            expr = expr[rng_sub.choice(expr.shape[0], size=args.n_cells, replace=False)]
        real_pert_matrices.append(expr)
        real_pert_labels.extend([gene] * expr.shape[0])
    cb_real = build_cb_input(ctrl_X, real_pert_matrices, real_pert_labels, gene_subset)
    t_real_prep = time.time() - t0
    logger.info(
        "  Real: %d cells (%d ctrl + %d pert)",
        cb_real.total_cells,
        cb_real.n_control_cells,
        cb_real.n_perturbation_cells,
    )

    # --- Overlay-resampled real (tests if overlay sampling preserves signal) ---
    # Take MEAN expression from real perturbed cells, then overlay-sample.
    # This is the UPPER BOUND for any model: perfect mean predictions + overlay sampling.
    logger.info("Step 4a+: Overlay-resampled real (perfect model upper bound)")
    t0 = time.time()
    nb_real_mats, nb_real_labels = [], []
    for gene in gene_subset:
        mask = adata.obs[pert_col] == gene
        if mask.sum() < 5:
            continue
        gene_cells = adata[mask]
        X_g = gene_cells.X
        if hasattr(X_g, "toarray"):
            X_g = X_g.toarray()
        real_mean = np.asarray(X_g[:, col_idx], dtype=np.float64).mean(axis=0)
        gene_seed = args.seed + int(hashlib.md5(gene.encode()).hexdigest()[:8], 16)
        sampled = sample_perturbed_cells_overlay(
            ctrl_X, real_mean, args.n_cells, seed=gene_seed
        )
        nb_real_mats.append(sampled)
        nb_real_labels.extend([gene] * args.n_cells)
    cb_nb_real = build_cb_input(ctrl_X, nb_real_mats, nb_real_labels, gene_subset)
    conditions_data["overlay_resampled_real"] = {
        "cb": cb_nb_real,
        "prep_time": time.time() - t0,
    }
    logger.info(
        "  Overlay-resampled real: %d cells (%.1fs)",
        cb_nb_real.total_cells,
        conditions_data["overlay_resampled_real"]["prep_time"],
    )

    # --- Perturbation effect diagnostic ---
    logger.info("  Perturbation effect diagnostic (mean |log2FC| per condition):")
    ctrl_means_diag = ctrl_X.mean(axis=0)
    for cond_name_d, data_d in [
        ("real", (real_pert_matrices, real_pert_labels)),
        ("nb_resampled", (nb_real_mats, nb_real_labels)),
    ]:
        mats_d, labs_d = data_d
        if not mats_d:
            continue
        all_means_d = np.stack([m.mean(axis=0) for m in mats_d])
        lfc = np.log2(np.maximum(all_means_d, 0.01) / np.maximum(ctrl_means_diag, 0.01))
        logger.info(
            "    %s: mean |log2FC|=%.3f, max |log2FC|=%.3f, genes w/ |lfc|>0.5: %d/%d",
            cond_name_d,
            np.abs(lfc).mean(),
            np.abs(lfc).max(),
            (np.abs(lfc).max(axis=1) > 0.5).sum(),
            len(mats_d),
        )

    # --- ElasticNet ---
    # Train on ALL controls (better model), but build CausalBenchInput with subsampled
    logger.info("Step 4b: ElasticNet baseline")
    t0 = time.time()
    enet_mats, enet_labels = run_elasticnet_baseline(
        ctrl_X_full, ctrl_X, gene_subset, args.n_cells, seed=args.seed
    )
    cb_enet = build_cb_input(ctrl_X, enet_mats, enet_labels, gene_subset)
    t_enet_prep = time.time() - t0
    logger.info("  ElasticNet: %d cells (%.1fs)", cb_enet.total_cells, t_enet_prep)

    # --- Real model outputs (if provided) ---
    real_models_loaded = set()
    if getattr(args, "model_outputs_dir", None):
        model_outputs_dir = args.model_outputs_dir
        logger.info(
            "Step 4b: Loading real model predictions from %s", model_outputs_dir
        )
        for model_name in ["cpa", "gears", "scgpt", "geneformer"]:
            t0 = time.time()
            mats, labels = load_real_model_predictions(
                model_outputs_dir,
                model_name,
                gene_subset,
                ctrl_X,
                args.n_cells,
                seed=args.seed,
            )
            if mats is not None:
                cb = build_cb_input(ctrl_X, mats, labels, gene_subset)
                conditions_data[f"{model_name}_real"] = {
                    "cb": cb,
                    "prep_time": time.time() - t0,
                }
                real_models_loaded.add(model_name)
                logger.info(
                    "  %s (real): %d cells (%.1fs)",
                    model_name,
                    cb.total_cells,
                    time.time() - t0,
                )

    # --- Model stubs (DEPRECATED — only used when no real model outputs available) ---
    # Stubs use arbitrary parameters and are not suitable for publication.
    # Always prefer real model outputs via --model-outputs-dir.
    use_stubs = getattr(args, "use_stubs", False) and not real_models_loaded
    if use_stubs:
        logger.warning(
            "Step 4: Running model stubs (DEPRECATED — use --model-outputs-dir for real outputs)"
        )
        for model_name, cfg in MODEL_CONFIGS.items():
            logger.info("Step 4: %s stub", model_name)
            t0 = time.time()
            mats, labels = run_model_stub(
                ctrl_X_full,
                ctrl_X,
                gene_subset,
                args.n_cells,
                model_name=model_name,
                **cfg,
            )
            cb = build_cb_input(ctrl_X, mats, labels, gene_subset)
            conditions_data[model_name] = {"cb": cb, "prep_time": time.time() - t0}
            logger.info(
                "  %s: %d cells (%.1fs)",
                model_name,
                cb.total_cells,
                conditions_data[model_name]["prep_time"],
            )
    elif not real_models_loaded:
        logger.info(
            "Step 4: No real model outputs found and stubs disabled. Use --model-outputs-dir or --use-stubs."
        )

    # --- Random baseline ---
    cb_rand = None
    if not getattr(args, "skip_random", False):
        logger.info("Step 4: Random baseline")
        t0 = time.time()
        rand_mats, rand_labels = run_random_baseline(
            ctrl_X, gene_subset, args.n_cells, seed=args.seed
        )
        cb_rand = build_cb_input(ctrl_X, rand_mats, rand_labels, gene_subset)
        t_rand_prep = time.time() - t0
        logger.info("  Random: %d cells (%.1fs)", cb_rand.total_cells, t_rand_prep)
    else:
        logger.info("Step 4: Skipping random baseline (--skip-random)")

    # ------------------------------------------------------------------
    # Step 5: Run GIES on all conditions
    # ------------------------------------------------------------------
    logger.info("Step 5: Running GIES causal discovery")

    all_conditions = {
        "real_data": {"cb": cb_real, "prep_time": t_real_prep},
        "elastic_net": {"cb": cb_enet, "prep_time": t_enet_prep},
        **conditions_data,
    }
    if cb_rand is not None:
        all_conditions["random"] = {"cb": cb_rand, "prep_time": t_rand_prep}

    gies_cache = None
    if getattr(args, "gies_cache_dir", None):
        gies_cache = os.path.join(
            args.gies_cache_dir,
            f"n{len(gene_subset)}_s{args.seed}_p{args.partition_size}",
        )
        os.makedirs(gies_cache, exist_ok=True)
        logger.info("  GIES cache dir: %s", gies_cache)

    edges_by_condition = {}
    # Run GIES for each condition — partitions within each are parallelized
    for cond_name, cond_data in all_conditions.items():
        cache_path = (
            os.path.join(gies_cache, f"{cond_name}_edges.json") if gies_cache else None
        )
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            edges = set(tuple(e) for e in cached["edges"])
            t_gies = cached.get("gies_time", 0)
            logger.info(
                "  GIES on %s... CACHED (%d edges, %.1fs)",
                cond_name,
                len(edges),
                t_gies,
            )
        else:
            logger.info("  GIES on %s...", cond_name)
            t0 = time.time()
            edges = run_gies(
                cond_data["cb"], seed=args.seed, partition_size=args.partition_size
            )
            t_gies = time.time() - t0
            logger.info("    %s: %d edges (%.1fs)", cond_name, len(edges), t_gies)
            if cache_path:
                with open(cache_path, "w") as f:
                    json.dump(
                        {"edges": [list(e) for e in edges], "gies_time": t_gies}, f
                    )
        edges_by_condition[cond_name] = edges
        cond_data["gies_time"] = t_gies

    # ------------------------------------------------------------------
    # Step 5b: Run inspre on all conditions (second causal discovery backend)
    # ------------------------------------------------------------------
    inspre_edges_by_condition = {}
    if getattr(args, "run_inspre", False):
        logger.info("Step 5b: Running inspre causal discovery")
        import subprocess
        import tempfile
        import pandas as pd_inspre

        # Find Rscript
        rscript_paths = [
            "/opt/mamba/envs/renv/bin/Rscript",
            "/usr/local/bin/Rscript",
            "/usr/bin/Rscript",
            "Rscript",
        ]
        rscript = None
        for rp in rscript_paths:
            if os.path.exists(rp) or rp == "Rscript":
                rscript = rp
                break

        if rscript:
            r_inspre_code = """
library(inspre)
args <- commandArgs(trailingOnly = TRUE)
ace_path <- args[1]
out_path <- args[2]
target_edges <- as.integer(args[3])
R_hat <- as.matrix(read.csv(ace_path, header=FALSE))
result <- fit_inspre_from_R(R_hat = R_hat, its = 200, verbose = 0, train_prop = 0.8)
G_hat <- result$G_hat
lambdas <- result$lambda
test_err <- result$test_error
valid_te <- !is.nan(test_err) & !is.na(test_err)
if (any(valid_te)) {
    best_idx <- which(valid_te)[which.min(test_err[valid_te])]
} else {
    edge_counts <- sapply(1:dim(G_hat)[3], function(i) sum(abs(G_hat[,,i]) > 1e-6))
    best_idx <- which.min(abs(edge_counts - target_edges))
}
G_best <- G_hat[,,best_idx]
write.csv(G_best, out_path, row.names=FALSE)
"""
            # Compute ACE matrices for each condition
            ctrl_mean_flat = ctrl_X.mean(axis=0)
            for cond_name, cond_data in all_conditions.items():
                inspre_cache = (
                    os.path.join(gies_cache, f"{cond_name}_inspre_edges.json")
                    if gies_cache
                    else None
                )
                if inspre_cache and os.path.exists(inspre_cache):
                    with open(inspre_cache) as f:
                        cached = json.load(f)
                    inspre_edges_by_condition[cond_name] = set(
                        tuple(e) for e in cached["edges"]
                    )
                    logger.info(
                        "  inspre %s: CACHED (%d edges)",
                        cond_name,
                        len(inspre_edges_by_condition[cond_name]),
                    )
                    continue

                cb = cond_data["cb"]
                # Compute per-gene perturbation means for ACE
                pert_means = {}
                for gene in gene_subset:
                    mask = np.array([lab == gene for lab in cb.interventions])
                    if mask.sum() < 2:
                        continue
                    pert_means[gene] = cb.expression_matrix[mask].mean(axis=0)

                # Build ACE matrix: ACE[i,j] = E[Xj|do(gi)] - E[Xj|ctrl]
                n_g = len(gene_subset)
                gene_to_gidx = {g: i for i, g in enumerate(gene_subset)}
                ace = np.zeros((n_g, n_g), dtype=np.float64)
                for gene, pm in pert_means.items():
                    if gene in gene_to_gidx:
                        ace[gene_to_gidx[gene], :] = pm - ctrl_mean_flat

                target_n = max(len(edges_by_condition.get("real_data", [])), 10)
                with tempfile.TemporaryDirectory() as tmpdir:
                    ace_path = os.path.join(tmpdir, "ace.csv")
                    out_path = os.path.join(tmpdir, "adj.csv")
                    r_path = os.path.join(tmpdir, "run.R")
                    np.savetxt(ace_path, ace, delimiter=",")
                    with open(r_path, "w") as f:
                        f.write(r_inspre_code)
                    try:
                        t0 = time.time()
                        result = subprocess.run(
                            [rscript, r_path, ace_path, out_path, str(target_n)],
                            capture_output=True,
                            text=True,
                            timeout=600,
                        )
                        t_inspre = time.time() - t0
                        if result.returncode == 0 and os.path.exists(out_path):
                            adj = pd_inspre.read_csv(out_path).values
                            edges = set()
                            for i in range(n_g):
                                for j in range(n_g):
                                    if i != j and abs(adj[i, j]) > 1e-6:
                                        edges.add((gene_subset[i], gene_subset[j]))
                            inspre_edges_by_condition[cond_name] = edges
                            logger.info(
                                "  inspre %s: %d edges (%.1fs)",
                                cond_name,
                                len(edges),
                                t_inspre,
                            )
                            if inspre_cache:
                                with open(inspre_cache, "w") as f:
                                    json.dump(
                                        {
                                            "edges": [list(e) for e in edges],
                                            "time": t_inspre,
                                        },
                                        f,
                                    )
                        else:
                            logger.warning(
                                "  inspre %s failed: %s",
                                cond_name,
                                result.stderr[-200:] if result.stderr else "unknown",
                            )
                    except subprocess.TimeoutExpired:
                        logger.warning("  inspre %s timed out", cond_name)
                    except FileNotFoundError:
                        logger.warning("  Rscript not found — skipping inspre")
                        break
        else:
            logger.warning("  Rscript not found — skipping inspre")

    # ------------------------------------------------------------------
    # Step 6: Run GRNBoost2 on real data (observational baseline)
    # ------------------------------------------------------------------
    edges_grnboost_real = None
    t_grnboost = 0.0
    grn_cache_path = (
        os.path.join(gies_cache, "grnboost2_edges.json") if gies_cache else None
    )
    if grn_cache_path and os.path.exists(grn_cache_path):
        with open(grn_cache_path) as f:
            cached = json.load(f)
        edges_grnboost_real = set(tuple(e) for e in cached["edges"])
        t_grnboost = cached.get("time", 0)
        logger.info(
            "Step 6: GRNBoost2 CACHED (%d edges, %.1fs)",
            len(edges_grnboost_real),
            t_grnboost,
        )
    else:
        try:
            logger.info("Step 6: Running GRNBoost2 on real data")
            t0 = time.time()
            edges_grnboost_real = run_grnboost(
                cb_real, n_top=len(edges_by_condition["real_data"]) * 2
            )
            t_grnboost = time.time() - t0
            logger.info(
                "  GRNBoost2: %d edges (%.1fs)", len(edges_grnboost_real), t_grnboost
            )
            if grn_cache_path:
                with open(grn_cache_path, "w") as f:
                    json.dump(
                        {
                            "edges": [list(e) for e in edges_grnboost_real],
                            "time": t_grnboost,
                        },
                        f,
                    )
        except Exception as e:
            logger.warning("  GRNBoost2 failed (dask version issue): %s", e)
            logger.info("  Skipping GRNBoost2 — will fix dependency later")

    # ------------------------------------------------------------------
    # Step 6b: Run DCDI-G on real data (soft intervention baseline)
    # ------------------------------------------------------------------
    edges_dcdi_real = None
    t_dcdi = 0.0
    if getattr(args, "run_dcdi", False):
        try:
            from causalcellbench.causal.dcdi_runner import run_dcdi

            logger.info("Step 6b: Running DCDI-G on real data (soft interventions)")
            t0 = time.time()
            edges_dcdi_real = run_dcdi(
                cb_real,
                seed=args.seed,
                partition_size=args.partition_size,
                intervention_type="imperfect",
                max_epochs=getattr(args, "dcdi_epochs", 200),
            )
            t_dcdi = time.time() - t0
            logger.info("  DCDI-G: %d edges (%.1fs)", len(edges_dcdi_real), t_dcdi)
        except Exception as e:
            logger.warning("  DCDI-G failed: %s", e)

    # ------------------------------------------------------------------
    # Step 7: Evaluate all conditions
    # ------------------------------------------------------------------
    logger.info("Step 7: Computing evaluation metrics")

    gt_edges = set(edges_by_condition["real_data"])
    logger.info("  Ground truth: %d edges from real-data GIES", len(gt_edges))

    random_floor = compute_random_auprc(
        gt_edges, gene_subset, n_samples=50, seed=args.seed
    )
    logger.info("  Random floor AUPRC: %.4f", random_floor)

    results = {}
    for cond_name, cond_data in all_conditions.items():
        edges = list(edges_by_condition[cond_name])
        wall_time = cond_data["prep_time"] + cond_data.get("gies_time", 0)
        if cond_name == "real_data":
            results[cond_name] = {
                "n_edges": len(edges),
                "f1": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "auprc": 1.0,
                "auprc_topk": 1.0,
                "precision_at_100": 1.0,
                "n_true_positive": len(edges),
                "shd": {"shd": 0, "missing": 0, "extra": 0, "reversed": 0},
                "wasserstein": mean_wasserstein_distance(
                    edges,
                    cond_data["cb"].expression_matrix,
                    cond_data["cb"].interventions,
                    gene_subset,
                ),
                "false_omission": false_omission_rate(
                    edges,
                    cond_data["cb"].expression_matrix,
                    cond_data["cb"].interventions,
                    gene_subset,
                ),
                "wall_time": wall_time,
            }
        else:
            results[cond_name] = evaluate_condition(
                cond_name,
                edges,
                cond_data["cb"],
                gt_edges,
                gene_subset,
                wall_time,
            )

    # GRNBoost2 on real data (if available)
    if edges_grnboost_real is not None:
        results["grnboost2_real"] = evaluate_condition(
            "grnboost2_real",
            edges_grnboost_real,
            cb_real,
            gt_edges,
            gene_subset,
            t_grnboost,
        )

    # DCDI-G on real data (if available)
    if edges_dcdi_real is not None:
        results["dcdi_real"] = evaluate_condition(
            "dcdi_real",
            edges_dcdi_real,
            cb_real,
            gt_edges,
            gene_subset,
            t_dcdi,
        )

    # inspre results (if available) — evaluate against GIES ground truth
    inspre_results = {}
    if inspre_edges_by_condition:
        logger.info("  Evaluating inspre results against GIES ground truth:")
        for cond_name, inspre_edges in inspre_edges_by_condition.items():
            cb = all_conditions.get(cond_name, {}).get("cb")
            if cb is None:
                continue
            inspre_metrics = compute_set_metrics(
                list(inspre_edges), gt_edges, gene_subset
            )
            inspre_results[cond_name] = {
                "n_edges": len(inspre_edges),
                "f1": inspre_metrics["f1"],
                "precision": inspre_metrics["precision"],
                "recall": inspre_metrics["recall"],
                "n_true_positive": inspre_metrics["n_true_positive"],
            }
            logger.info(
                "    inspre %s: F1=%.4f, P=%.4f, R=%.4f (%d edges)",
                cond_name,
                inspre_metrics["f1"],
                inspre_metrics["precision"],
                inspre_metrics["recall"],
                len(inspre_edges),
            )

        # Also compute inspre-on-real as alternative ground truth
        if "real_data" in inspre_edges_by_condition:
            inspre_gt = inspre_edges_by_condition["real_data"]
            logger.info(
                "  Evaluating GIES results against inspre ground truth (%d edges):",
                len(inspre_gt),
            )
            for cond_name in results:
                if cond_name == "real_data":
                    continue
                gies_edges = list(edges_by_condition.get(cond_name, []))
                if gies_edges:
                    m = compute_set_metrics(gies_edges, inspre_gt, gene_subset)
                    logger.info("    GIES %s vs inspre-GT: F1=%.4f", cond_name, m["f1"])

    # ------------------------------------------------------------------
    # Step 7b: Bootstrap confidence intervals
    # ------------------------------------------------------------------
    logger.info("Step 7b: Computing bootstrap confidence intervals")
    bootstrap_results = {}
    for cond_name in results:
        if cond_name == "real_data":
            continue  # trivially 1.0
        edges = list(edges_by_condition.get(cond_name, []))
        if cond_name == "grnboost2_real" and edges_grnboost_real is not None:
            edges = list(edges_grnboost_real)
        elif cond_name == "dcdi_real" and edges_dcdi_real is not None:
            edges = list(edges_dcdi_real)
        if edges:
            boot = bootstrap_evaluate(
                edges, gt_edges, gene_subset, n_bootstrap=200, seed=args.seed
            )
            bootstrap_results[cond_name] = boot
            f1_ci = boot["f1"]
            logger.info(
                "  %s: F1=%.4f [%.4f, %.4f]",
                cond_name,
                f1_ci["mean"],
                f1_ci["ci_lo"],
                f1_ci["ci_hi"],
            )

    # ------------------------------------------------------------------
    # Step 7c: Prediction accuracy (if real model outputs available)
    # ------------------------------------------------------------------
    prediction_accuracy = {}
    if real_pert_matrices and real_pert_labels:
        for model_name in real_models_loaded:
            cond_key = f"{model_name}_real"
            if cond_key in conditions_data:
                cond = conditions_data[cond_key]
                # Re-extract model matrices from the cb_input using dict-based grouping
                cb = cond["cb"]
                model_mats_flat = cb.expression_matrix[cb.n_control_cells :]
                model_labels = cb.interventions[cb.n_control_cells :]
                # Group rows by gene using a dict (safe even if labels are non-contiguous)
                from collections import defaultdict

                gene_rows = defaultdict(list)
                for i, lab in enumerate(model_labels):
                    gene_rows[lab].append(model_mats_flat[i])
                model_mats = [np.array(rows) for rows in gene_rows.values()]
                model_labs = []
                for gene, rows in gene_rows.items():
                    model_labs.extend([gene] * len(rows))

                acc = compute_prediction_accuracy(
                    model_mats,
                    model_labs,
                    real_pert_matrices,
                    real_pert_labels,
                    gene_subset,
                )
                prediction_accuracy[model_name] = acc
                logger.info(
                    "  %s prediction accuracy: MSE=%.4f, r=%.4f, cos=%.4f",
                    model_name,
                    acc["mse"],
                    acc["pearson_r"],
                    acc["cosine_sim"],
                )

    # ------------------------------------------------------------------
    # Step 7d: Pairwise significance tests (key model comparisons)
    # ------------------------------------------------------------------
    pairwise_tests = {}
    comparison_pairs = [
        ("gears_real", "cpa_real"),
        ("gears_real", "geneformer_real"),
        ("elastic_net", "gears_real"),
        ("gears_real", "grnboost2_real"),
        ("cpa_real", "geneformer_real"),
    ]
    for cond_a, cond_b in comparison_pairs:
        edges_a = list(edges_by_condition.get(cond_a, []))
        edges_b_key = cond_b
        if cond_b == "grnboost2_real" and edges_grnboost_real is not None:
            edges_b_list = list(edges_grnboost_real)
        else:
            edges_b_list = list(edges_by_condition.get(cond_b, []))
        if edges_a and edges_b_list:
            test = pairwise_permutation_test(
                edges_a, edges_b_list, gt_edges, gene_subset, n_perm=500, seed=args.seed
            )
            key = f"{cond_a}_vs_{cond_b}"
            pairwise_tests[key] = test
            sig = "*" if test["significant_005"] else ""
            logger.info(
                "  %s vs %s: dF1=%.4f, p=%.4f %s",
                cond_a,
                cond_b,
                test["observed_diff"],
                test["p_value"],
                sig,
            )

    # ------------------------------------------------------------------
    # Step 8: Save and print results
    # ------------------------------------------------------------------
    total_time = time.time() - t_start
    summary = {
        "pipeline": "CausalCellBench Local Evaluation v2",
        "n_genes": len(gene_subset),
        "gene_subset": gene_subset,
        "partition_size": args.partition_size,
        "n_cells_per_pert": args.n_cells,
        "seed": args.seed,
        "n_control_cells_total": control.n_obs,
        "n_control_cells_gies": n_ctrl,
        "n_eligible_genes": len(gene_subset),
        "n_gt_edges": len(gt_edges),
        "random_floor_auprc": random_floor,
        "nb_dispersion_median_r": float(np.median(r_disp)),
        "nb_dispersion_mean_mu": float(np.mean(mu_disp)),
        "conditions": results,
        "inspre_results": inspre_results,
        "bootstrap_ci": bootstrap_results,
        "prediction_accuracy": prediction_accuracy,
        "pairwise_significance": pairwise_tests,
        "total_wall_time": total_time,
    }

    os.makedirs("results", exist_ok=True)
    out_path = f"results/eval_results_n{len(gene_subset)}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # --- Print summary ---
    print()
    print("=" * 100)
    print("CAUSALCELLBENCH EVALUATION RESULTS")
    print("=" * 100)
    print(
        f"  Genes: {len(gene_subset)} | Controls: {n_ctrl} (of {control.n_obs}) | "
        f"GT edges: {len(gt_edges)} | Random floor: {random_floor:.4f}"
    )
    print()
    header = f"  {'Condition':<18} {'Edges':>6} {'F1':>7} {'Prec':>7} {'Rec':>7} {'AUPRC':>7} {'P@100':>7} {'TP':>5} {'SHD':>6} {'Time':>7}"
    print(header)
    print(
        f"  {'-' * 18} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 5} {'-' * 6} {'-' * 7}"
    )

    display_order = [
        "real_data",
        "overlay_resampled_real",
        "elastic_net",
        "cpa_real",
        "gears_real",
        "scgpt_real",
        "geneformer_real",
        "cpa",
        "gears",
        "scgpt",
        "geneformer",
        "grnboost2_real",
        "dcdi_real",
        "random",
    ]
    display_names = {
        "real_data": "Real Data (UB)",
        "overlay_resampled_real": "Overlay Real",
        "elastic_net": "ElasticNet (reg)",
        "cpa_real": "CPA",
        "gears_real": "GEARS",
        "scgpt_real": "scGPT",
        "geneformer_real": "Geneformer",
        "cpa": "CPA (stub)",
        "gears": "GEARS (stub)",
        "scgpt": "scGPT (stub)",
        "geneformer": "Geneformer (stub)",
        "grnboost2_real": "GRNBoost2 (obs)",
        "dcdi_real": "DCDI-G (soft)",
        "random": "Random (floor)",
    }

    for key in display_order:
        if key not in results:
            continue
        c = results[key]
        name = display_names.get(key, key)
        shd_v = c.get("shd", {}).get("shd", 0)
        auprc_display = c.get("auprc_topk", c["auprc"])
        print(
            f"  {name:<18} {c['n_edges']:>6} {c.get('f1', 0):>7.4f} {c.get('precision', 0):>7.4f} "
            f"{c.get('recall', 0):>7.4f} {auprc_display:>7.4f} {c.get('precision_at_100', 0):>7.4f} "
            f"{c.get('n_true_positive', 0):>5} {shd_v:>6} {c['wall_time']:>6.1f}s"
        )

    print()
    print(f"  Random floor AUPRC: {random_floor:.4f}")
    print(f"  Total pipeline time: {total_time:.1f}s")
    print(f"  Results saved to: {out_path}")
    print()

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CausalCellBench local evaluation")
    parser.add_argument(
        "--data-path", default="data/k562_real.h5ad", help="Path to dataset h5ad"
    )
    parser.add_argument(
        "--pert-col",
        default="gene",
        help="obs column with perturbation labels (Replogle='gene', Norman/RPE1='perturbation')",
    )
    parser.add_argument(
        "--n-genes", type=int, default=100, help="Number of genes to evaluate"
    )
    parser.add_argument(
        "--partition-size", type=int, default=30, help="GIES partition size"
    )
    parser.add_argument(
        "--n-cells", type=int, default=100, help="Cells per perturbation"
    )
    parser.add_argument(
        "--n-ctrl",
        type=int,
        default=200,
        help="Control cells for GIES (matched to pert count, prevents control domination)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gene-list",
        default=None,
        help="File with one gene per line (overrides --n-genes and --seed selection)",
    )
    parser.add_argument(
        "--model-outputs-dir",
        default=None,
        help="Directory with {model}_predictions.h5ad files from real model runs",
    )
    parser.add_argument(
        "--use-stubs",
        action="store_true",
        help="Run model stubs (DEPRECATED — only use when no real model outputs available)",
    )
    parser.add_argument(
        "--skip-random",
        action="store_true",
        help="Skip random baseline GIES (saves ~2.5h for 200 genes)",
    )
    parser.add_argument(
        "--gies-cache-dir",
        default=None,
        help="Cache GIES edges per condition (enables incremental re-runs)",
    )
    parser.add_argument(
        "--run-inspre",
        action="store_true",
        help="Also run inspre causal discovery (handles cycles, scalable)",
    )
    parser.add_argument(
        "--run-dcdi",
        action="store_true",
        help="Also run DCDI-G causal discovery (slower, handles soft interventions)",
    )
    parser.add_argument(
        "--dcdi-epochs",
        type=int,
        default=200,
        help="DCDI training epochs per partition",
    )
    parser.add_argument(
        "--pre-subset",
        action="store_true",
        help="Data is already subset to target genes+cells (skip Ensembl mapping & gene selection)",
    )
    args = parser.parse_args()
    main(args)
