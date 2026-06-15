"""Phase 4 — calibration mode sweep (Mode 1 vs 2 vs 3 vs all permutations).

Applies the existing QuantileCalibrator + new CovariancePreserving + SparsityInjection
to model predictions with 80/20 perturbation train/test split (V4 audit H6 fix).
Writes calibrated predictions h5ad files to data/model_outputs_phase4/.
Downstream GIES + F1 is left to a follow-up run_eval invocation (slow).

Output: results/calibration_modes/sweep_manifest.json listing every produced
calibrated prediction file + its calibration order.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from itertools import permutations

ROOT = Path("/Users/aayanalwani/virtual cells")
sys.path.insert(0, str(ROOT))

import numpy as np
import anndata as ad

from causalcellbench.calibration import (
    CalibrationPipeline,
    all_orderings,
)

INPUT_DIR = ROOT / "data" / "model_outputs_real_n200"
OUT_DIR = ROOT / "data" / "model_outputs_phase4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results" / "calibration_modes"
RESULTS.mkdir(parents=True, exist_ok=True)
K562 = ROOT / "data" / "k562_subset_n200_slim.h5ad"
EPS = 1e-2

MODELS = ("gears", "cpa", "geneformer")
MODE_KEYS = ("q", "c", "s")  # Q=quantile, C=covariance, S=sparsity

CTRL_LABEL = "non-targeting"
PERT_COL = "gene"


def load_real_data():
    """Return (ctrl_means, pert_means_dict, real_log_fc, real_pert_shifts, gene_names)."""
    adata = ad.read_h5ad(K562)
    adata.var_names = [str(g) for g in adata.var_names]
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float64)
    is_ctrl = (adata.obs[PERT_COL] == CTRL_LABEL).values
    ctrl_X = X[is_ctrl]
    ctrl_means = ctrl_X.mean(axis=0)
    gene_names = list(adata.var_names)
    # Per-perturbation real means
    pert_means = {}
    for gene in set(adata.obs[PERT_COL].values):
        if gene == CTRL_LABEL:
            continue
        mask = (adata.obs[PERT_COL] == gene).values
        if mask.sum() < 5:
            continue
        pert_means[gene] = X[mask].mean(axis=0)
    # Real log-FC matrix (n_perts, n_genes)
    shifts = []
    pert_order = sorted(pert_means.keys())
    for g in pert_order:
        shifts.append(np.log2((pert_means[g] + EPS) / (ctrl_means + EPS)))
    real_log_fc = np.asarray(shifts)
    # Real perturbation shift matrix (raw, not log) for covariance fitting
    real_pert_shifts = np.asarray([pert_means[g] - ctrl_means for g in pert_order])
    return ctrl_means, pert_means, real_log_fc, real_pert_shifts, gene_names, pert_order


def load_model_pred(model_name: str, gene_names: list[str]):
    """Return predicted_log_fc matrix (n_perts, n_genes) + perturbation list."""
    p = INPUT_DIR / f"{model_name}_predictions.h5ad"
    if not p.exists():
        raise FileNotFoundError(p)
    adata = ad.read_h5ad(p)
    adata.var_names = [str(g) for g in adata.var_names]
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float64)
    # Map predicted gene order to reference gene order
    pred_gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    col_idx = [pred_gene_to_idx[g] for g in gene_names if g in pred_gene_to_idx]
    pred = X[:, col_idx]
    # Perturbation labels
    obs_col = PERT_COL if PERT_COL in adata.obs.columns else adata.obs.columns[0]
    pert_labels = list(adata.obs[obs_col].values)
    return pred, pert_labels


def train_test_split_perts(perts: list[str], train_frac: float = 0.8, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(perts)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    train_idx = sorted(idx[:n_train].tolist())
    test_idx = sorted(idx[n_train:].tolist())
    train_perts = [perts[i] for i in train_idx]
    test_perts = [perts[i] for i in test_idx]
    return train_perts, test_perts, train_idx, test_idx


def fit_and_transform(order: list[str], *, real_log_fc_train, real_pert_shifts_train,
                      pred_log_fc_train, pred_log_fc_test):
    """Fit calibration pipeline on train, apply to test. Returns calibrated_test_log_fc."""
    p = CalibrationPipeline(order=order)
    p.fit(
        real_log_fc=real_log_fc_train,
        real_pert_shifts=real_pert_shifts_train,
        pred_log_fc=pred_log_fc_train,
    )
    return p.transform(pred_log_fc_test)


def main():
    print("=" * 60)
    print("Phase 4 calibration sweep")
    print("=" * 60)
    ctrl_means, real_pert_means, real_log_fc, real_pert_shifts, gene_names, pert_order = load_real_data()
    print(f"  Loaded real K562: {len(pert_order)} perturbations, {len(gene_names)} genes")
    train_perts, test_perts, train_idx, test_idx = train_test_split_perts(
        pert_order, train_frac=0.8, seed=42
    )
    print(f"  Train perts: {len(train_perts)}, Test perts: {len(test_perts)}")

    real_log_fc_train = real_log_fc[train_idx]
    real_pert_shifts_train = real_pert_shifts[train_idx]

    sweep = {"orderings": [], "train_perts": train_perts, "test_perts": test_perts}
    # 3 single-mode + 6 full-pipeline orderings = 9 conditions per model
    all_orders = [[m] for m in MODE_KEYS] + all_orderings()

    for model in MODELS:
        try:
            pred, pred_pert_labels = load_model_pred(model, gene_names)
        except FileNotFoundError:
            print(f"  {model}: predictions not found, skip")
            continue
        # Build pred_log_fc (n_predicted_perts, n_genes). Clip pred ≥ 0 because
        # some models (GEARS) emit small negative expression values.
        pred_safe = np.maximum(pred, 0.0)
        pred_log_fc = np.log2((pred_safe + EPS) / (ctrl_means + EPS))
        # Sanity: any remaining NaN/Inf → 0 (means no fold-change)
        pred_log_fc = np.nan_to_num(pred_log_fc, nan=0.0, posinf=10.0, neginf=-10.0)
        # Align pred_pert_labels with the real perturbation order
        pert_to_pred_idx = {p: i for i, p in enumerate(pred_pert_labels)}
        # Construct train/test split using the same pert ordering
        train_pred_idx = [pert_to_pred_idx[p] for p in train_perts if p in pert_to_pred_idx]
        test_pred_idx = [pert_to_pred_idx[p] for p in test_perts if p in pert_to_pred_idx]
        pred_log_fc_train = pred_log_fc[train_pred_idx]
        pred_log_fc_test = pred_log_fc[test_pred_idx]
        for order in all_orders:
            order_label = "".join(order)
            cal_log_fc_test = fit_and_transform(
                order,
                real_log_fc_train=real_log_fc_train,
                real_pert_shifts_train=real_pert_shifts_train,
                pred_log_fc_train=pred_log_fc_train,
                pred_log_fc_test=pred_log_fc_test,
            )
            # Exact inverse of log_fc = log2((pred+EPS)/(ctrl+EPS)): pred = (ctrl+EPS)*2^log_fc - EPS
            cal_raw = (ctrl_means + EPS) * (2 ** cal_log_fc_test) - EPS
            cal_raw = np.maximum(cal_raw, 0.0).astype(np.float32)
            # Write h5ad (only test perturbations). Label rows with the SURVIVING test
            # perturbations (those present in predictions), matching test_pred_idx order.
            out_h5 = OUT_DIR / f"{model}_{order_label}_test_predictions.h5ad"
            new_adata = ad.AnnData(X=cal_raw)
            new_adata.var_names = gene_names
            new_adata.obs["gene"] = [p for p in test_perts if p in pert_to_pred_idx]
            new_adata.write_h5ad(out_h5)
            sweep["orderings"].append({
                "model": model,
                "order": order_label,
                "h5ad": str(out_h5.relative_to(ROOT)),
                "n_test_perts": int(len(test_pred_idx)),
            })
            # Stats
            frac_large_cal = float((np.abs(cal_log_fc_test) > 0.5).mean())
            frac_large_real = float((np.abs(real_log_fc[test_idx]) > 0.5).mean())
            print(f"  {model:<12} order={order_label:<3} "
                  f"frac|FC|>.5 cal={frac_large_cal:.3f} real(test)={frac_large_real:.3f}")

    manifest = RESULTS / "sweep_manifest.json"
    with open(manifest, "w") as f:
        json.dump(sweep, f, indent=2)
    print(f"\n  Wrote {manifest}")
    print(f"  Total calibrated h5ad files: {len(sweep['orderings'])}")
    print(f"  Follow-up: run GIES on each via state_eval_gies.py to populate F1.")


if __name__ == "__main__":
    main()
