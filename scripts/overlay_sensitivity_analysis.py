"""Overlay sampling sensitivity analysis for CausalCellBench.

Tests whether causal discovery results are robust to arbitrary choices
in the overlay sampling procedure:
  - Fold-change clip range
  - Pseudocount for low-expression genes
  - Noise model (Poisson, NB, None)
  - Control cell threshold

Key question: do model RANKINGS remain stable across parameter choices?
If yes, overlay sampling is defensible regardless of absolute metric shifts.

Usage:
    python3 scripts/overlay_sensitivity_analysis.py
    python3 scripts/overlay_sensitivity_analysis.py --n-genes 50  # faster
    python3 scripts/overlay_sensitivity_analysis.py --models elastic_net,cpa_real
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from numpy.polynomial import polynomial as P
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import ElasticNet

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("overlay_sensitivity")

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "overlay_sensitivity"


# =====================================================================
# Negative Binomial dispersion estimation (from run_eval_local.py)
# =====================================================================

def estimate_nb_dispersion(
    control_X: np.ndarray, shrinkage: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate per-gene NB dispersion from control expression matrix.

    Parameters
    ----------
    control_X : (n_cells, n_genes) dense float array
    shrinkage : apply DESeq2-style log-linear shrinkage

    Returns
    -------
    mu : (n_genes,) mean expression
    r  : (n_genes,) dispersion parameter (size parameter)
    """
    X = np.asarray(control_X, dtype=np.float64)
    mu = X.mean(axis=0)
    var = X.var(axis=0, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        overdispersed = var > mu + 1e-8
        r = np.where(overdispersed, mu ** 2 / (var - mu), 1e4)

    r = np.clip(r, 0.1, 1e6)
    zero_mean = mu < 1e-8
    r[zero_mean] = 1.0
    mu[zero_mean] = 1e-8

    if shrinkage and len(mu) >= 20:
        r = _shrink_dispersion(mu, r)

    return mu.astype(np.float64), r.astype(np.float64)


def _shrink_dispersion(
    mu: np.ndarray, r_raw: np.ndarray, shrinkage_weight: float = 0.5
) -> np.ndarray:
    valid = (mu > 1e-4) & np.isfinite(r_raw) & (r_raw > 0.01) & (r_raw < 1e5)
    if valid.sum() < 10:
        return r_raw
    try:
        coeffs = P.polyfit(np.log(mu[valid] + 1), np.log(r_raw[valid]), deg=1)
        r_trend = np.exp(P.polyval(np.log(mu + 1), coeffs))
        r_trend = np.clip(r_trend, 0.01, 1e6)
    except (np.linalg.LinAlgError, ValueError):
        return r_raw
    log_r_shrunk = (
        (1 - shrinkage_weight) * np.log(r_raw + 1e-8)
        + shrinkage_weight * np.log(r_trend + 1e-8)
    )
    return np.exp(log_r_shrunk)


# =====================================================================
# Overlay sampling with configurable parameters
# =====================================================================

@dataclass
class OverlayParams:
    """Configurable overlay sampling parameters."""
    clip_lo: float = 0.01
    clip_hi: float = 10.0
    pseudocount: float = 0.01
    noise_model: str = "poisson"  # "poisson", "nb", "none"
    ctrl_threshold: float = 1e-6

    @property
    def label(self) -> str:
        return (
            f"clip[{self.clip_lo},{self.clip_hi}]_"
            f"pc{self.pseudocount}_"
            f"{self.noise_model}_"
            f"thr{self.ctrl_threshold}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_lo": self.clip_lo,
            "clip_hi": self.clip_hi,
            "pseudocount": self.pseudocount,
            "noise_model": self.noise_model,
            "ctrl_threshold": self.ctrl_threshold,
        }


def sample_perturbed_cells_overlay(
    ctrl_X: np.ndarray,
    mean_pred: np.ndarray,
    n_cells: int,
    params: OverlayParams,
    nb_dispersion: Optional[np.ndarray] = None,
    seed: int = 0,
) -> np.ndarray:
    """Generate perturbed cells by overlaying fold-changes on control cells.

    Preserves control covariance structure by:
    1. Bootstrap sampling from control cells
    2. Scaling each gene by the predicted fold-change ratio
    3. Adding noise (Poisson, NB, or None) for count-level variability

    Parameters
    ----------
    ctrl_X : (n_ctrl, n_genes) control expression matrix
    mean_pred : (n_genes,) predicted mean expression under perturbation
    n_cells : number of cells to generate
    params : OverlayParams with clip range, pseudocount, noise model, threshold
    nb_dispersion : (n_genes,) NB dispersion for noise_model="nb"
    seed : random seed
    """
    rng = np.random.default_rng(seed)
    n_ctrl = ctrl_X.shape[0]

    # Bootstrap from controls
    idx = rng.choice(n_ctrl, size=n_cells, replace=True)
    cells = ctrl_X[idx].copy().astype(np.float64)

    # Compute fold-change ratios
    ctrl_means = ctrl_X.mean(axis=0).astype(np.float64)
    ratios = np.ones_like(ctrl_means)
    nonzero = ctrl_means > params.ctrl_threshold
    ctrl_means_safe = np.maximum(ctrl_means, params.pseudocount)
    ratios[nonzero] = (
        np.maximum(mean_pred[nonzero], 0.0) / ctrl_means_safe[nonzero]
    )
    ratios = np.clip(ratios, params.clip_lo, params.clip_hi)

    # Apply fold-change to each cell (preserves covariance structure)
    cells *= ratios[np.newaxis, :]

    # Noise model
    cells = np.maximum(cells, 0.0)

    if params.noise_model == "poisson":
        cells = rng.poisson(lam=cells).astype(np.float32)
    elif params.noise_model == "nb":
        if nb_dispersion is None:
            raise ValueError("nb_dispersion required for noise_model='nb'")
        r = np.maximum(nb_dispersion, 0.1).astype(np.float64)
        mu = np.maximum(cells, 1e-8)
        rates = rng.gamma(
            shape=r[np.newaxis, :],
            scale=(mu / r[np.newaxis, :]),
            size=cells.shape,
        )
        cells = rng.poisson(lam=rates).astype(np.float32)
    elif params.noise_model == "none":
        # Just round to integers (count data)
        cells = np.round(cells).astype(np.float32)
    else:
        raise ValueError(f"Unknown noise_model: {params.noise_model}")

    return cells


# =====================================================================
# CausalBenchInput (from run_eval_local.py)
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
    def total_cells(self) -> int:
        return self.n_control_cells + self.n_perturbation_cells

    def validate(self) -> None:
        n_cells, n_genes = self.expression_matrix.shape
        assert n_cells == len(self.interventions)
        assert n_genes == len(self.gene_names)
        assert n_cells == self.total_cells
        assert np.isfinite(self.expression_matrix).all()


def build_cb_input(
    ctrl_X: np.ndarray,
    pert_matrices: List[np.ndarray],
    pert_labels: List[str],
    gene_names: List[str],
) -> CausalBenchInput:
    if pert_matrices:
        pert_X = np.concatenate(pert_matrices, axis=0).astype(np.float32)
    else:
        pert_X = np.empty((0, len(gene_names)), dtype=np.float32)
    expression_matrix = np.concatenate(
        [ctrl_X.astype(np.float32), pert_X], axis=0
    )
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
# GIES runner (from run_eval_local.py)
# =====================================================================

def _run_gies_partition(args_tuple):
    """Run GIES on a single partition. Designed for parallel execution."""
    import gies as gies_pkg
    p_idx, partition_idx, gene_names, expression_matrix, interventions = args_tuple

    partition_genes = [gene_names[i] for i in partition_idx]
    partition_gene_set = set(partition_genes)
    n_part = len(partition_idx)

    regimes: Dict[str, list] = {}
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


def run_gies(
    cb_input: CausalBenchInput,
    seed: int = 0,
    partition_size: int = 30,
    n_workers: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Run GIES with CausalBench-style partitioning."""
    import multiprocessing as mp

    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 7)

    gene_names = cb_input.gene_names
    n_genes = len(gene_names)

    rng = np.random.default_rng(seed)
    gene_order = rng.permutation(n_genes)
    partitions = [
        gene_order[i: i + partition_size]
        for i in range(0, n_genes, partition_size)
    ]

    tasks = [
        (p_idx, partition_idx, gene_names,
         cb_input.expression_matrix, cb_input.interventions)
        for p_idx, partition_idx in enumerate(partitions)
    ]

    n_parts = len(partitions)
    logger.info("      GIES: %d partitions, %d workers", n_parts, n_workers)

    ctx = mp.get_context("fork")
    pool = ctx.Pool(processes=n_workers)
    try:
        results = pool.map(_run_gies_partition, tasks)
    finally:
        pool.close()
        pool.join()

    all_edges: List[Tuple[str, str]] = []
    for p_idx, edges, error in results:
        if error:
            logger.warning("    GIES partition %d failed: %s", p_idx, error)
        else:
            all_edges.extend(edges)

    return all_edges


# =====================================================================
# ElasticNet baseline (from run_eval_local.py)
# =====================================================================

def run_elasticnet_predictions(
    ctrl_X_train: np.ndarray,
    gene_subset: List[str],
) -> Dict[str, np.ndarray]:
    """Train ElasticNet on controls and return mean predictions per gene.

    Returns dict mapping gene -> (n_genes,) mean predicted expression.
    """
    n_genes = len(gene_subset)
    control_means = ctrl_X_train.mean(axis=0)

    models = {}
    for g_idx in range(n_genes):
        feature_idx = [i for i in range(n_genes) if i != g_idx]
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=0)
        model.fit(ctrl_X_train[:, feature_idx], ctrl_X_train[:, g_idx])
        models[g_idx] = model

    predictions: Dict[str, np.ndarray] = {}
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
        predictions[gene] = mean_pred

    return predictions


# =====================================================================
# Real model prediction loading
# =====================================================================

def load_model_mean_predictions(
    model_outputs_dir: str,
    model_name: str,
    gene_subset: List[str],
    ctrl_X: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:
    """Load real model predictions and return mean predictions per gene.

    Returns dict mapping gene -> (n_genes,) mean predicted expression,
    or None if file missing.
    """
    filename = f"{model_name}_predictions.h5ad"
    filepath = os.path.join(model_outputs_dir, filename)
    if not os.path.exists(filepath):
        return None

    pred_adata = ad.read_h5ad(filepath)
    output_type = pred_adata.uns.get("output_type", "mean")

    # Build gene mapping
    pred_var = list(pred_adata.var_names)
    pred_var_to_idx = {g: i for i, g in enumerate(pred_var)}
    pred_var_upper = {g.upper(): i for i, g in enumerate(pred_var)}
    gene_idx_map: Dict[str, int] = {}
    for g in gene_subset:
        if g in pred_var_to_idx:
            gene_idx_map[g] = pred_var_to_idx[g]
        elif g.upper() in pred_var_upper:
            gene_idx_map[g] = pred_var_upper[g.upper()]

    if len(gene_idx_map) < len(gene_subset) * 0.5:
        logger.warning("  Only %d/%d genes matched for %s",
                       len(gene_idx_map), len(gene_subset), model_name)
        return None

    # Perturbation column
    pert_col = None
    for col in ["perturbation", "condition", "gene", "pert"]:
        if col in pred_adata.obs.columns:
            pert_col = col
            break
    if pert_col is None:
        return None

    predictions: Dict[str, np.ndarray] = {}
    for gene in gene_subset:
        if gene not in gene_idx_map:
            continue
        mask = pred_adata.obs[pert_col] == gene
        if mask.sum() == 0:
            mask = pred_adata.obs[pert_col].str.upper() == gene.upper()
        if mask.sum() == 0:
            continue

        if output_type == "cells":
            # Average across sampled cells to get the mean prediction
            X_g = pred_adata[mask].X
            if hasattr(X_g, "toarray"):
                X_g = X_g.toarray()
            raw_mean = np.asarray(X_g, dtype=np.float64).mean(axis=0)
        else:
            raw_mean = pred_adata[mask].X
            if hasattr(raw_mean, "toarray"):
                raw_mean = raw_mean.toarray()
            raw_mean = np.asarray(raw_mean[0], dtype=np.float64)

        # Map to gene_subset columns
        mean_pred = np.zeros(len(gene_subset), dtype=np.float64)
        for gi, g in enumerate(gene_subset):
            if g in gene_idx_map:
                mean_pred[gi] = raw_mean[gene_idx_map[g]]
            else:
                mean_pred[gi] = ctrl_X[:, gi].mean()

        nan_mask = ~np.isfinite(mean_pred)
        if nan_mask.any():
            mean_pred[nan_mask] = ctrl_X[:, nan_mask].mean(axis=0)

        predictions[gene] = mean_pred

    if not predictions:
        return None

    logger.info("  Loaded %s: %d/%d perturbation predictions",
                model_name, len(predictions), len(gene_subset))
    return predictions


# =====================================================================
# Metrics
# =====================================================================

def edge_set_metrics(
    pred_edges: List[Tuple[str, str]],
    gt_edges: set,
) -> Dict[str, float]:
    """Compute precision, recall, F1 of predicted edges vs ground truth."""
    pred_set = set(pred_edges)
    if not pred_set and not gt_edges:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_edges": 0}
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_edges": 0}
    if not gt_edges:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_edges": len(pred_set)}

    tp = len(pred_set & gt_edges)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_edges) if gt_edges else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "f1": round(f1, 5),
        "n_edges": len(pred_set),
        "n_tp": tp,
    }


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two edge sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


# =====================================================================
# CPA native cell comparison
# =====================================================================

def compare_cpa_overlay_vs_native(
    ctrl_X: np.ndarray,
    gene_subset: List[str],
    model_outputs_dir: str,
    params: OverlayParams,
    nb_dispersion: Optional[np.ndarray],
    n_cells: int,
    partition_size: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """Compare GIES edges from CPA overlay-sampled vs CPA native cells.

    Validates whether overlay sampling distorts causal discovery results.
    Returns None if CPA native cells file is missing.
    """
    cells_path = os.path.join(model_outputs_dir, "cpa_predictions_cells.h5ad")
    mean_path = os.path.join(model_outputs_dir, "cpa_predictions.h5ad")
    if not os.path.exists(cells_path) or not os.path.exists(mean_path):
        return None

    logger.info("  CPA native vs overlay comparison")

    # Load CPA native cells
    cells_adata = ad.read_h5ad(cells_path)
    mean_adata = ad.read_h5ad(mean_path)

    # Build gene mapping for native cells
    pred_var = list(cells_adata.var_names)
    pred_var_to_idx = {g: i for i, g in enumerate(pred_var)}
    pred_var_upper = {g.upper(): i for i, g in enumerate(pred_var)}
    gene_idx_map: Dict[str, int] = {}
    for g in gene_subset:
        if g in pred_var_to_idx:
            gene_idx_map[g] = pred_var_to_idx[g]
        elif g.upper() in pred_var_upper:
            gene_idx_map[g] = pred_var_upper[g.upper()]

    if len(gene_idx_map) < len(gene_subset) * 0.5:
        logger.warning("  CPA gene overlap too low: %d/%d", len(gene_idx_map), len(gene_subset))
        return None

    pert_col = None
    for col in ["perturbation", "condition", "gene", "pert"]:
        if col in cells_adata.obs.columns:
            pert_col = col
            break
    if pert_col is None:
        return None

    # Build native cells input
    native_mats, native_labels = [], []
    overlay_mats, overlay_labels = [], []

    # Also get mean predictions for overlay
    mean_pert_col = None
    for col in ["perturbation", "condition", "gene", "pert"]:
        if col in mean_adata.obs.columns:
            mean_pert_col = col
            break

    for gene in gene_subset:
        if gene not in gene_idx_map:
            continue

        # Native cells
        mask = cells_adata.obs[pert_col] == gene
        if mask.sum() == 0:
            mask = cells_adata.obs[pert_col].str.upper() == gene.upper()
        if mask.sum() == 0:
            continue

        X_g = cells_adata[mask].X
        if hasattr(X_g, "toarray"):
            X_g = X_g.toarray()

        col_indices = [gene_idx_map[g] for g in gene_subset if g in gene_idx_map]
        expr = np.asarray(X_g[:, col_indices], dtype=np.float32)

        if expr.shape[0] > n_cells:
            gene_seed = seed + int(hashlib.md5(gene.encode()).hexdigest()[:8], 16)
            rng_sub = np.random.default_rng(gene_seed)
            expr = expr[rng_sub.choice(expr.shape[0], size=n_cells, replace=False)]

        # Pad if needed
        if expr.shape[1] < len(gene_subset):
            padded = np.zeros((expr.shape[0], len(gene_subset)), dtype=np.float32)
            j = 0
            for gi, g in enumerate(gene_subset):
                if g in gene_idx_map:
                    padded[:, gi] = expr[:, j]
                    j += 1
                else:
                    padded[:, gi] = ctrl_X[:, gi].mean()
            expr = padded

        expr = np.nan_to_num(expr, nan=0.0, posinf=1e4, neginf=0.0)
        native_mats.append(expr)
        native_labels.extend([gene] * expr.shape[0])

        # Overlay from mean prediction
        if mean_pert_col is not None:
            mean_mask = mean_adata.obs[mean_pert_col] == gene
            if mean_mask.sum() == 0:
                mean_mask = mean_adata.obs[mean_pert_col].str.upper() == gene.upper()
            if mean_mask.sum() > 0:
                mean_row = mean_adata[mean_mask].X
                if hasattr(mean_row, "toarray"):
                    mean_row = mean_row.toarray()
                raw_mean = np.asarray(mean_row[0], dtype=np.float64)

                mean_pred = np.zeros(len(gene_subset), dtype=np.float64)
                for gi, g in enumerate(gene_subset):
                    if g in gene_idx_map:
                        mean_pred[gi] = raw_mean[gene_idx_map[g]]
                    else:
                        mean_pred[gi] = ctrl_X[:, gi].mean()

                gene_seed = seed + int(hashlib.md5(gene.encode()).hexdigest()[:8], 16)
                sampled = sample_perturbed_cells_overlay(
                    ctrl_X, mean_pred, n_cells, params, nb_dispersion, seed=gene_seed
                )
                overlay_mats.append(sampled)
                overlay_labels.extend([gene] * n_cells)

    if not native_mats or not overlay_mats:
        return None

    # Run GIES on both
    cb_native = build_cb_input(ctrl_X, native_mats, native_labels, gene_subset)
    cb_overlay = build_cb_input(ctrl_X, overlay_mats, overlay_labels, gene_subset)

    logger.info("    Native: %d cells, Overlay: %d cells",
                cb_native.total_cells, cb_overlay.total_cells)

    edges_native = run_gies(cb_native, seed=seed, partition_size=partition_size)
    edges_overlay = run_gies(cb_overlay, seed=seed, partition_size=partition_size)

    native_set = set(edges_native)
    overlay_set = set(edges_overlay)

    return {
        "n_native_edges": len(native_set),
        "n_overlay_edges": len(overlay_set),
        "jaccard": round(jaccard_similarity(native_set, overlay_set), 5),
        "overlap": len(native_set & overlay_set),
        "native_only": len(native_set - overlay_set),
        "overlay_only": len(overlay_set - native_set),
    }


# =====================================================================
# Main sensitivity analysis
# =====================================================================

def build_param_grid() -> List[OverlayParams]:
    """Build the full parameter sweep grid."""
    clip_ranges = [
        (0.1, 5.0),
        (0.01, 10.0),    # default
        (0.001, 100.0),
        (1e-4, 1000.0),
    ]
    pseudocounts = [0.001, 0.01, 0.1, 1.0]
    noise_models = ["poisson", "nb", "none"]

    # Full grid: 4 * 4 * 3 = 48 combinations
    grid: List[OverlayParams] = []
    for (clip_lo, clip_hi), pc, noise in itertools.product(
        clip_ranges, pseudocounts, noise_models
    ):
        grid.append(OverlayParams(
            clip_lo=clip_lo,
            clip_hi=clip_hi,
            pseudocount=pc,
            noise_model=noise,
            ctrl_threshold=1e-6,  # fixed at default
        ))
    return grid


def run_sensitivity_for_model(
    model_name: str,
    mean_predictions: Dict[str, np.ndarray],
    ctrl_X: np.ndarray,
    gene_subset: List[str],
    gt_edges: set,
    nb_dispersion: np.ndarray,
    param_grid: List[OverlayParams],
    n_cells: int,
    partition_size: int,
    seed: int,
    cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run GIES across all overlay parameter combos for one model.

    Returns list of result dicts (one per parameter combo).
    """
    results: List[Dict[str, Any]] = []

    for pi, params in enumerate(param_grid):
        param_label = params.label
        logger.info("  [%d/%d] %s | %s", pi + 1, len(param_grid), model_name, param_label)

        # Check cache
        if cache_dir is not None:
            cache_file = cache_dir / f"{model_name}_{param_label}_edges.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    cached = json.load(f)
                edges = [tuple(e) for e in cached["edges"]]
                metrics = edge_set_metrics(edges, gt_edges)
                result = {
                    "model": model_name,
                    **params.to_dict(),
                    **metrics,
                    "cached": True,
                }
                results.append(result)
                logger.info("    CACHED: %d edges, F1=%.4f", metrics["n_edges"], metrics["f1"])
                continue

        # Generate overlay-sampled cells for each perturbation
        pert_matrices, pert_labels = [], []
        for gene, mean_pred in mean_predictions.items():
            gene_seed = seed + int(hashlib.md5(gene.encode()).hexdigest()[:8], 16)
            sampled = sample_perturbed_cells_overlay(
                ctrl_X, mean_pred, n_cells, params, nb_dispersion, seed=gene_seed
            )
            pert_matrices.append(sampled)
            pert_labels.extend([gene] * n_cells)

        if not pert_matrices:
            logger.warning("    No perturbation matrices generated")
            continue

        cb_input = build_cb_input(ctrl_X, pert_matrices, pert_labels, gene_subset)

        # Run GIES
        t0 = time.time()
        edges = run_gies(cb_input, seed=seed, partition_size=partition_size)
        t_gies = time.time() - t0

        # Cache edges
        if cache_dir is not None:
            cache_file = cache_dir / f"{model_name}_{param_label}_edges.json"
            with open(cache_file, "w") as f:
                json.dump({"edges": [list(e) for e in edges], "time": t_gies}, f)

        metrics = edge_set_metrics(edges, gt_edges)
        result = {
            "model": model_name,
            **params.to_dict(),
            **metrics,
            "gies_time": round(t_gies, 2),
            "cached": False,
        }
        results.append(result)
        logger.info("    %d edges, F1=%.4f, P=%.4f, R=%.4f (%.1fs)",
                     metrics["n_edges"], metrics["f1"],
                     metrics["precision"], metrics["recall"], t_gies)

    return results


def analyze_ranking_stability(
    results_df: pd.DataFrame,
    models: List[str],
) -> Dict[str, Any]:
    """Analyze whether model rankings are stable across parameter choices.

    For each parameter combination, rank the models by F1.
    Then compute rank correlation (Kendall tau, Spearman) across all combos.
    """
    if len(models) < 2:
        return {"stable": True, "note": "Only one model, ranking trivially stable"}

    # Build a matrix: rows = param combos, columns = models, values = F1
    param_cols = ["clip_lo", "clip_hi", "pseudocount", "noise_model"]
    grouped = results_df.groupby(param_cols)

    rank_matrix: List[List[int]] = []
    f1_matrix: List[Dict[str, float]] = []

    for name, group in grouped:
        f1_by_model: Dict[str, float] = {}
        for _, row in group.iterrows():
            if row["model"] in models:
                f1_by_model[row["model"]] = row["f1"]

        # Only include combos where all models have results
        if len(f1_by_model) == len(models):
            # Rank models (higher F1 = rank 1)
            sorted_models = sorted(f1_by_model.keys(), key=lambda m: f1_by_model[m], reverse=True)
            ranks = {m: r + 1 for r, m in enumerate(sorted_models)}
            rank_matrix.append([ranks[m] for m in models])
            f1_matrix.append(f1_by_model)

    if len(rank_matrix) < 2:
        return {"stable": True, "note": "Too few complete combos for rank analysis",
                "n_combos": len(rank_matrix)}

    rank_arr = np.array(rank_matrix)  # (n_combos, n_models)

    # Pairwise rank correlation between all param combos
    taus: List[float] = []
    for i in range(len(rank_arr)):
        for j in range(i + 1, len(rank_arr)):
            tau, _ = kendalltau(rank_arr[i], rank_arr[j])
            if np.isfinite(tau):
                taus.append(tau)

    # Check if any model's rank changes
    rank_ranges = {}
    for mi, model in enumerate(models):
        model_ranks = rank_arr[:, mi]
        rank_ranges[model] = {
            "min_rank": int(model_ranks.min()),
            "max_rank": int(model_ranks.max()),
            "median_rank": float(np.median(model_ranks)),
            "rank_std": float(np.std(model_ranks)),
        }

    # Modal ranking (most frequent rank order)
    rank_tuples = [tuple(row) for row in rank_arr]
    from collections import Counter
    rank_counts = Counter(rank_tuples)
    modal_ranking, modal_count = rank_counts.most_common(1)[0]

    stability = {
        "n_param_combos": len(rank_arr),
        "n_models": len(models),
        "models": models,
        "mean_kendall_tau": round(float(np.mean(taus)), 4) if taus else None,
        "min_kendall_tau": round(float(np.min(taus)), 4) if taus else None,
        "modal_ranking": {m: int(r) for m, r in zip(models, modal_ranking)},
        "modal_ranking_frequency": f"{modal_count}/{len(rank_arr)}",
        "rank_ranges": rank_ranges,
        "stable": all(v["max_rank"] - v["min_rank"] <= 1 for v in rank_ranges.values()),
        "perfectly_stable": modal_count == len(rank_arr),
    }

    return stability


def main():
    parser = argparse.ArgumentParser(
        description="Overlay sampling sensitivity analysis for CausalCellBench"
    )
    parser.add_argument("--n-genes", type=int, default=100)
    parser.add_argument("--n-cells", type=int, default=100)
    parser.add_argument("--n-ctrl", type=int, default=200)
    parser.add_argument("--partition-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model list (default: elastic_net + all available real models). "
             "Options: elastic_net, cpa_real, gears_real, geneformer_real"
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to K562 .h5ad (auto-detected if not given)"
    )
    parser.add_argument(
        "--model-outputs-dir", type=str, default=None,
        help="Path to model outputs (auto-detected if not given)"
    )
    parser.add_argument("--no-cpa-comparison", action="store_true",
                        help="Skip CPA native vs overlay comparison")
    args = parser.parse_args()

    t_start = time.time()

    logger.info("=" * 70)
    logger.info("Overlay Sampling Sensitivity Analysis")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Resolve data paths
    # ------------------------------------------------------------------
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        slim = BASE_DIR / "data" / "k562_subset_n200_slim.h5ad"
        full = BASE_DIR / "data" / "k562_real.h5ad"
        data_path = slim if slim.exists() else full
    assert data_path.exists(), f"Data file not found: {data_path}"

    if args.model_outputs_dir:
        model_outputs_dir = args.model_outputs_dir
    else:
        # Prefer n200 outputs
        n200 = BASE_DIR / "data" / "model_outputs_real_n200"
        real = BASE_DIR / "data" / "model_outputs_real"
        model_outputs_dir = str(n200 if n200.exists() else real)

    logger.info("  Data: %s", data_path)
    logger.info("  Model outputs: %s", model_outputs_dir)
    logger.info("  n_genes=%d, n_cells=%d, partition_size=%d, seed=%d",
                args.n_genes, args.n_cells, args.partition_size, args.seed)

    # ------------------------------------------------------------------
    # Setup output directory
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = RESULTS_DIR / "gies_cache"
    cache_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Load K562 data
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading K562 data")
    adata = ad.read_h5ad(str(data_path))
    logger.info("  Shape: %s", adata.shape)

    # Determine if pre-subset
    pre_subset = not str(adata.var_names[0]).startswith("ENSG")
    if not pre_subset and "gene_name" in adata.var.columns:
        logger.info("  Mapping Ensembl IDs to gene symbols")
        symbols = adata.var["gene_name"].values
        seen: Dict[str, int] = {}
        unique_symbols: List[str] = []
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
    else:
        adata.var_names = [str(x) for x in adata.var_names]

    # ------------------------------------------------------------------
    # Extract controls and select genes
    # ------------------------------------------------------------------
    logger.info("Step 2: Extracting controls and gene subset")
    pert_col = "gene"
    control_mask = adata.obs[pert_col] == "non-targeting"
    if control_mask.sum() == 0:
        for label in ["control", "ctrl", "NT"]:
            control_mask = adata.obs[pert_col] == label
            if control_mask.sum() > 0:
                break
    assert control_mask.sum() > 0, "No control cells found"

    control = adata[control_mask].copy()
    logger.info("  Control cells: %d", control.n_obs)

    if pre_subset:
        gene_subset = sorted(adata.var_names.tolist())
        logger.info("  Pre-subset mode: %d genes", len(gene_subset))
    else:
        pert_counts = adata.obs[pert_col].value_counts()
        var_set = set(adata.var_names)
        eligible = sorted([
            g for g, c in pert_counts.items()
            if g != "non-targeting" and c >= 10 and g in var_set
        ])
        rng = np.random.default_rng(args.seed)
        gene_subset = sorted(
            rng.choice(eligible, size=min(args.n_genes, len(eligible)), replace=False)
        )
    logger.info("  Selected %d genes", len(gene_subset))

    # Build control matrices
    gene_to_varidx = {g: i for i, g in enumerate(adata.var_names)}
    col_idx = [gene_to_varidx[g] for g in gene_subset]

    X_ctrl_full = control.X
    if hasattr(X_ctrl_full, "toarray"):
        X_ctrl_full = X_ctrl_full.toarray()
    X_ctrl_full = np.asarray(X_ctrl_full[:, col_idx], dtype=np.float32)

    n_ctrl = min(args.n_ctrl, X_ctrl_full.shape[0])
    rng_ctrl = np.random.default_rng(args.seed)
    ctrl_idx = rng_ctrl.choice(X_ctrl_full.shape[0], size=n_ctrl, replace=False)
    ctrl_X = X_ctrl_full[ctrl_idx]
    logger.info("  Control matrix: %s (from %d)", ctrl_X.shape, X_ctrl_full.shape[0])

    # ------------------------------------------------------------------
    # Estimate NB dispersion (needed for NB noise model)
    # ------------------------------------------------------------------
    logger.info("Step 3: Estimating NB dispersion")
    _, nb_dispersion = estimate_nb_dispersion(X_ctrl_full)
    logger.info("  Median dispersion r=%.2f", np.median(nb_dispersion))

    # ------------------------------------------------------------------
    # Ground truth: GIES on real perturbed cells
    # ------------------------------------------------------------------
    logger.info("Step 4: Building ground truth (GIES on real perturbed cells)")
    gt_cache = cache_dir / "ground_truth_real_edges.json"
    if gt_cache.exists():
        with open(gt_cache) as f:
            cached = json.load(f)
        gt_edges_list = [tuple(e) for e in cached["edges"]]
        logger.info("  Ground truth CACHED: %d edges", len(gt_edges_list))
    else:
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
                gene_seed = args.seed + int(hashlib.md5(gene.encode()).hexdigest()[:8], 16)
                rng_sub = np.random.default_rng(gene_seed)
                expr = expr[rng_sub.choice(expr.shape[0], size=args.n_cells, replace=False)]
            real_pert_matrices.append(expr)
            real_pert_labels.extend([gene] * expr.shape[0])

        cb_real = build_cb_input(ctrl_X, real_pert_matrices, real_pert_labels, gene_subset)
        t0 = time.time()
        gt_edges_list = run_gies(cb_real, seed=args.seed, partition_size=args.partition_size)
        t_gt = time.time() - t0
        logger.info("  Ground truth: %d edges (%.1fs)", len(gt_edges_list), t_gt)

        with open(gt_cache, "w") as f:
            json.dump({"edges": [list(e) for e in gt_edges_list], "time": t_gt}, f)

    gt_edges = set(gt_edges_list)
    logger.info("  Ground truth edge count: %d", len(gt_edges))

    # ------------------------------------------------------------------
    # Load model mean predictions
    # ------------------------------------------------------------------
    logger.info("Step 5: Loading model mean predictions")

    all_model_preds: Dict[str, Dict[str, np.ndarray]] = {}

    # ElasticNet
    logger.info("  Training ElasticNet...")
    t0 = time.time()
    enet_preds = run_elasticnet_predictions(X_ctrl_full, gene_subset)
    logger.info("  ElasticNet: %d predictions (%.1fs)", len(enet_preds), time.time() - t0)
    all_model_preds["elastic_net"] = enet_preds

    # Real model outputs
    for model_name in ["cpa", "gears", "geneformer"]:
        preds = load_model_mean_predictions(
            model_outputs_dir, model_name, gene_subset, ctrl_X
        )
        if preds is not None:
            all_model_preds[f"{model_name}_real"] = preds

    # Filter to requested models
    if args.models:
        requested = [m.strip() for m in args.models.split(",")]
        all_model_preds = {k: v for k, v in all_model_preds.items() if k in requested}

    logger.info("  Models to sweep: %s", list(all_model_preds.keys()))

    # ------------------------------------------------------------------
    # Parameter sweep
    # ------------------------------------------------------------------
    logger.info("Step 6: Parameter sweep (%d models x 48 param combos)",
                len(all_model_preds))
    param_grid = build_param_grid()
    logger.info("  Grid size: %d", len(param_grid))

    all_results: List[Dict[str, Any]] = []

    for model_name, preds in all_model_preds.items():
        logger.info("--- Model: %s (%d perturbation predictions) ---",
                     model_name, len(preds))
        model_results = run_sensitivity_for_model(
            model_name=model_name,
            mean_predictions=preds,
            ctrl_X=ctrl_X,
            gene_subset=gene_subset,
            gt_edges=gt_edges,
            nb_dispersion=nb_dispersion,
            param_grid=param_grid,
            n_cells=args.n_cells,
            partition_size=args.partition_size,
            seed=args.seed,
            cache_dir=cache_dir,
        )
        all_results.extend(model_results)

    # ------------------------------------------------------------------
    # CPA native vs overlay comparison
    # ------------------------------------------------------------------
    cpa_comparison = None
    if not args.no_cpa_comparison and model_outputs_dir:
        logger.info("Step 7: CPA native vs overlay comparison")
        default_params = OverlayParams()  # default settings
        cpa_comparison = compare_cpa_overlay_vs_native(
            ctrl_X=ctrl_X,
            gene_subset=gene_subset,
            model_outputs_dir=model_outputs_dir,
            params=default_params,
            nb_dispersion=nb_dispersion,
            n_cells=args.n_cells,
            partition_size=args.partition_size,
            seed=args.seed,
        )
        if cpa_comparison:
            logger.info("  CPA comparison: Jaccard=%.4f, Native=%d edges, Overlay=%d edges",
                         cpa_comparison["jaccard"],
                         cpa_comparison["n_native_edges"],
                         cpa_comparison["n_overlay_edges"])
        else:
            logger.info("  CPA native cells not available, skipping comparison")

    # ------------------------------------------------------------------
    # Analysis: ranking stability
    # ------------------------------------------------------------------
    logger.info("Step 8: Analyzing ranking stability")
    results_df = pd.DataFrame(all_results)
    models_in_results = sorted(results_df["model"].unique().tolist())
    stability = analyze_ranking_stability(results_df, models_in_results)

    logger.info("  Models analyzed: %s", models_in_results)
    logger.info("  Rankings stable (rank shift <= 1): %s", stability.get("stable"))
    logger.info("  Perfectly stable (identical across all combos): %s",
                stability.get("perfectly_stable"))
    if stability.get("mean_kendall_tau") is not None:
        logger.info("  Mean Kendall tau: %.4f", stability["mean_kendall_tau"])

    # Per-model summary
    logger.info("\n  Per-model F1 summary across parameter sweep:")
    for model in models_in_results:
        model_df = results_df[results_df["model"] == model]
        logger.info("    %s: F1 mean=%.4f, std=%.4f, min=%.4f, max=%.4f, "
                     "edges mean=%.1f, std=%.1f",
                     model,
                     model_df["f1"].mean(), model_df["f1"].std(),
                     model_df["f1"].min(), model_df["f1"].max(),
                     model_df["n_edges"].mean(), model_df["n_edges"].std())

    # Per-parameter effect
    logger.info("\n  Parameter marginal effects on F1:")
    for param_name in ["clip_lo", "pseudocount", "noise_model"]:
        grouped = results_df.groupby(param_name)["f1"].agg(["mean", "std"])
        for val, row in grouped.iterrows():
            logger.info("    %s=%s: F1 mean=%.4f +/- %.4f",
                         param_name, val, row["mean"], row["std"])

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    logger.info("Step 9: Saving results")

    # CSV with all results
    csv_path = RESULTS_DIR / "sensitivity_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("  CSV: %s", csv_path)

    # JSON summary
    summary = {
        "config": {
            "n_genes": len(gene_subset),
            "n_cells": args.n_cells,
            "n_ctrl": n_ctrl,
            "partition_size": args.partition_size,
            "seed": args.seed,
            "data_path": str(data_path),
            "model_outputs_dir": model_outputs_dir,
        },
        "ground_truth_n_edges": len(gt_edges),
        "models": models_in_results,
        "n_param_combos": len(param_grid),
        "total_runs": len(all_results),
        "ranking_stability": stability,
        "per_model_summary": {},
        "per_parameter_summary": {},
    }

    # Per-model summary
    for model in models_in_results:
        model_df = results_df[results_df["model"] == model]
        summary["per_model_summary"][model] = {
            "f1_mean": round(model_df["f1"].mean(), 5),
            "f1_std": round(model_df["f1"].std(), 5),
            "f1_min": round(model_df["f1"].min(), 5),
            "f1_max": round(model_df["f1"].max(), 5),
            "precision_mean": round(model_df["precision"].mean(), 5),
            "recall_mean": round(model_df["recall"].mean(), 5),
            "n_edges_mean": round(model_df["n_edges"].mean(), 2),
            "n_edges_std": round(model_df["n_edges"].std(), 2),
        }

    # Per-parameter marginal effects
    for param_name in ["clip_lo", "pseudocount", "noise_model"]:
        param_summary: Dict[str, Dict[str, float]] = {}
        for val, group in results_df.groupby(param_name):
            param_summary[str(val)] = {
                "f1_mean": round(group["f1"].mean(), 5),
                "f1_std": round(group["f1"].std(), 5),
                "n_edges_mean": round(group["n_edges"].mean(), 2),
            }
        summary["per_parameter_summary"][param_name] = param_summary

    # CPA comparison
    if cpa_comparison:
        summary["cpa_native_vs_overlay"] = cpa_comparison

    json_path = RESULTS_DIR / "sensitivity_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("  JSON: %s", json_path)

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info("SENSITIVITY ANALYSIS COMPLETE (%.1f min)", elapsed / 60)
    logger.info("=" * 70)
    logger.info("")
    logger.info("KEY FINDINGS:")
    logger.info("  Ground truth: %d edges from real perturbed cells", len(gt_edges))
    logger.info("  Tested %d parameter combinations across %d models",
                len(param_grid), len(models_in_results))

    if stability.get("perfectly_stable"):
        logger.info("  RANKINGS PERFECTLY STABLE: identical model ordering across all combos")
    elif stability.get("stable"):
        logger.info("  RANKINGS STABLE: model rank shifts <= 1 position")
    else:
        logger.info("  RANKINGS UNSTABLE: some models shift >1 rank position")
        logger.info("  Rank ranges: %s", json.dumps(stability.get("rank_ranges", {}), indent=2))

    if cpa_comparison:
        logger.info("  CPA overlay vs native Jaccard: %.4f", cpa_comparison["jaccard"])
        if cpa_comparison["jaccard"] > 0.5:
            logger.info("  -> Overlay sampling preserves majority of CPA signal")
        elif cpa_comparison["jaccard"] > 0.2:
            logger.info("  -> Moderate distortion from overlay sampling")
        else:
            logger.info("  -> WARNING: Substantial distortion from overlay sampling")

    logger.info("")
    logger.info("Results: %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
