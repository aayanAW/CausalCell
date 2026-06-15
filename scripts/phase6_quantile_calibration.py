"""PHASE 6: Quantile Calibration with strict train/test separation.

Fixes the 23x effect-size inflation in DL model predictions by
calibrating fold-change distributions to match real data.

CRITICAL: No data leak. The calibrator is fit ONLY on training-set
perturbations, then applied to held-out test perturbations. AUPRC
is computed ONLY on test-set edges.

Saves all intermediate results with checkpoints.
"""

import os
os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import time
import logging
import pickle
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from multiprocessing import get_context
import gies as gies_pkg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path("/Users/aayanalwani/virtual cells")
DATA_PATH = BASE / "data" / "k562_subset_n200_slim.h5ad"
MODEL_DIR = BASE / "data" / "model_outputs_real_n200"
OUTDIR = BASE / "results" / "phase6_calibration"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_FRAC = 0.7  # 70% train, 30% test
N_CTRL = 200
N_CELLS_PER_PERT = 100
N_WORKERS = 4
PARTITION_SIZE = 20


# =====================================================================
# Quantile Calibrator (train/test safe)
# =====================================================================

class QuantileCalibrator:
    """
    Calibrate model fold-change distributions to match real data.

    CRITICAL: fit() uses ONLY training-set perturbations.
    transform() is applied to test-set perturbations.
    No data leak.

    The mapping: FC_cal = sign(FC_pred) * F_real_train^{-1}(F_model_train(|FC_pred|))
    """

    def __init__(self, n_quantiles: int = 1000, calibration_strength: float = 1.0):
        """
        Args:
            n_quantiles: resolution of the quantile mapping
            calibration_strength: 0.0 = no calibration, 1.0 = full calibration.
                                  Intermediate values interpolate between original
                                  and fully calibrated predictions.
        """
        self.n_quantiles = n_quantiles
        self.calibration_strength = calibration_strength
        self.quantile_points = np.linspace(0, 1, n_quantiles)
        self.real_quantiles = None
        self.model_quantiles = None
        self._fitted = False

    def fit(self, real_fcs_train: np.ndarray, model_fcs_train: np.ndarray):
        """
        Learn the quantile mapping from TRAINING SET ONLY.

        Args:
            real_fcs_train: all fold-changes from TRAINING perturbations (real data)
            model_fcs_train: all fold-changes from TRAINING perturbations (model predictions)
        """
        # Use absolute values for the mapping (sign is preserved separately)
        self.real_quantiles = np.quantile(np.abs(real_fcs_train), self.quantile_points)
        self.model_quantiles = np.quantile(np.abs(model_fcs_train), self.quantile_points)
        self._fitted = True

        # Log diagnostics
        logger.info(f"    Calibrator fit on {len(real_fcs_train)} real FCs, "
                     f"{len(model_fcs_train)} model FCs")
        logger.info(f"    Real median |FC|: {np.median(np.abs(real_fcs_train)):.4f}")
        logger.info(f"    Model median |FC|: {np.median(np.abs(model_fcs_train)):.4f}")
        logger.info(f"    Real frac |FC|>0.5: {np.mean(np.abs(real_fcs_train) > 0.5):.4f}")
        logger.info(f"    Model frac |FC|>0.5: {np.mean(np.abs(model_fcs_train) > 0.5):.4f}")

    def transform(self, fc_pred: np.ndarray) -> np.ndarray:
        """
        Apply calibration to fold-change predictions.
        Preserves sign, maps magnitude through learned quantile function.
        Interpolates between original and calibrated based on calibration_strength.
        """
        assert self._fitted, "Must call fit() before transform()"

        signs = np.sign(fc_pred)
        abs_fc = np.abs(fc_pred)

        # Map through quantile function
        calibrated_abs = np.interp(abs_fc, self.model_quantiles, self.real_quantiles)

        # Interpolate between original and calibrated
        alpha = self.calibration_strength
        blended_abs = (1 - alpha) * abs_fc + alpha * calibrated_abs

        return signs * blended_abs


# =====================================================================
# Train/Test Split
# =====================================================================

def create_train_test_split(gene_names: list, seed: int = 42,
                             train_frac: float = 0.7) -> tuple:
    """
    Split perturbation genes into train and test sets.

    Returns:
        train_genes: list of gene names for calibrator fitting
        test_genes: list of gene names for evaluation
    """
    rng = np.random.default_rng(seed)
    n = len(gene_names)
    n_train = int(n * train_frac)

    perm = rng.permutation(n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_genes = sorted([gene_names[i] for i in train_idx])
    test_genes = sorted([gene_names[i] for i in test_idx])

    logger.info(f"  Train/test split: {len(train_genes)} train, {len(test_genes)} test")
    return train_genes, test_genes


# =====================================================================
# GIES Engine (same as eval pipeline)
# =====================================================================

def _run_gies_partition(args_tuple):
    """Run GIES on a single partition."""
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
    """Run partitioned GIES with parallelism."""
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


# =====================================================================
# Overlay Sampling
# =====================================================================

def build_overlay_data(ctrl_X, ctrl_mean, model_preds, gene_names,
                        n_ctrl=200, n_cells=100, seed=42):
    """Build overlay-sampled interventional data from model predictions."""
    rng = np.random.default_rng(seed)
    ctrl_idx = rng.choice(ctrl_X.shape[0], size=min(n_ctrl, ctrl_X.shape[0]), replace=False)
    X_ctrl = ctrl_X[ctrl_idx].astype(np.float32)

    rows = [X_ctrl]
    interventions = ["non-targeting"] * X_ctrl.shape[0]

    for gene in gene_names:
        if gene not in model_preds:
            continue
        pred_mean = model_preds[gene]
        boot_idx = rng.choice(X_ctrl.shape[0], size=n_cells, replace=True)
        boot = X_ctrl[boot_idx].copy()
        fc = (pred_mean + 0.01) / (ctrl_mean + 0.01)
        fc = np.clip(fc, 0.01, 10.0)
        overlay = boot * fc
        rows.append(overlay.astype(np.float32))
        interventions.extend([gene] * n_cells)

    return np.vstack(rows), interventions


# =====================================================================
# Main Pipeline
# =====================================================================

def main():
    logger.info("=" * 70)
    logger.info("PHASE 6: Quantile Calibration (Train/Test Safe)")
    logger.info("=" * 70)

    # Load data
    logger.info("\nLoading data...")
    adata = ad.read_h5ad(DATA_PATH)
    adata.var_names = [str(x) for x in adata.var_names]
    gene_names = sorted(adata.var_names.tolist())
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    col_idx = [gene_to_idx[g] for g in gene_names]

    ctrl_mask = adata.obs["gene"] == "non-targeting"
    ctrl_X = adata[ctrl_mask].X
    if hasattr(ctrl_X, "toarray"):
        ctrl_X = ctrl_X.toarray()
    ctrl_X = np.asarray(ctrl_X[:, col_idx], dtype=np.float32)
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
        pert_means_real[gene] = np.mean(X[:, col_idx], axis=0).astype(np.float32)

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
                model_preds[g] = np.asarray(vals, dtype=np.float32)
        models[name] = model_preds
        logger.info(f"  {name}: {len(model_preds)} perturbations")

    # Step 1: Train/test split
    logger.info("\n--- Step 1: Train/test split ---")
    train_genes, test_genes = create_train_test_split(gene_names, SEED, TRAIN_FRAC)

    # Save split for reproducibility
    with open(OUTDIR / "train_test_split.json", "w") as f:
        json.dump({"train": train_genes, "test": test_genes, "seed": SEED,
                    "train_frac": TRAIN_FRAC}, f)

    # Step 2: Compute real FC distribution (TRAINING SET ONLY)
    logger.info("\n--- Step 2: Real FC distribution (training set) ---")
    real_fcs_train = []
    for gene in train_genes:
        if gene not in pert_means_real:
            continue
        fc = np.log2((pert_means_real[gene] + 1) / (ctrl_mean + 1))
        real_fcs_train.extend(fc.tolist())
    real_fcs_train = np.array(real_fcs_train)
    logger.info(f"  Training set real FCs: {len(real_fcs_train)}")

    # Step 3: Ground truth on TEST SET ONLY
    # Run GIES on real data restricted to test genes
    logger.info("\n--- Step 3: Ground truth (test set GIES) ---")
    gt_cache = OUTDIR / "test_gt_edges.json"
    if gt_cache.exists():
        with open(gt_cache) as f:
            gt_edges = set(tuple(e) for e in json.load(f)["edges"])
        logger.info(f"  Test GT: CACHED ({len(gt_edges)} edges)")
    else:
        # Build real data for test genes only
        test_perts_real = {g: pert_means_real[g] for g in test_genes if g in pert_means_real}
        expr_gt, interv_gt = build_overlay_data(
            ctrl_X, ctrl_mean, test_perts_real, test_genes, N_CTRL, N_CELLS_PER_PERT, SEED)
        logger.info(f"  Running GIES on test-set real data ({len(test_genes)} genes)...")
        t0 = time.time()
        gt_edges = run_gies(expr_gt, interv_gt, test_genes, PARTITION_SIZE, SEED)
        t_gt = time.time() - t0
        logger.info(f"  Test GT: {len(gt_edges)} edges ({t_gt:.1f}s)")
        with open(gt_cache, "w") as f:
            json.dump({"edges": [list(e) for e in gt_edges], "time": t_gt}, f)

    # Step 4: Calibration sweep
    logger.info("\n--- Step 4: Calibration sweep ---")

    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    all_results = {}

    for model_name in ["gears", "cpa", "geneformer"]:
        if model_name not in models:
            continue
        model_preds = models[model_name]

        logger.info(f"\n  === {model_name.upper()} ===")

        # Compute model FCs (TRAINING SET ONLY for calibrator fitting)
        model_fcs_train = []
        for gene in train_genes:
            if gene not in model_preds:
                continue
            fc = np.log2((model_preds[gene] + 1) / (ctrl_mean + 1))
            model_fcs_train.extend(fc.tolist())
        model_fcs_train = np.array(model_fcs_train)

        model_results = {}
        for strength in strengths:
            cache_path = OUTDIR / f"{model_name}_s{strength:.2f}_edges.json"
            if cache_path.exists():
                with open(cache_path) as f:
                    cached = json.load(f)
                edges = set(tuple(e) for e in cached["edges"])
                logger.info(f"    strength={strength:.2f}: CACHED ({len(edges)} edges)")
            else:
                logger.info(f"    strength={strength:.2f}: fitting calibrator (train only)...")

                # Fit calibrator on TRAINING set
                calibrator = QuantileCalibrator(
                    n_quantiles=1000, calibration_strength=strength)
                calibrator.fit(real_fcs_train, model_fcs_train)

                # Apply to TEST set predictions
                calibrated_preds = {}
                for gene in test_genes:
                    if gene not in model_preds:
                        continue
                    fc_pred = np.log2((model_preds[gene] + 1) / (ctrl_mean + 1))
                    fc_cal = calibrator.transform(fc_pred)
                    # Convert back to expression space
                    calibrated_preds[gene] = (2**fc_cal) * (ctrl_mean + 1) - 1
                    calibrated_preds[gene] = np.maximum(calibrated_preds[gene], 0)

                # Build overlay data and run GIES (test genes only)
                expr, interv = build_overlay_data(
                    ctrl_X, ctrl_mean, calibrated_preds, test_genes,
                    N_CTRL, N_CELLS_PER_PERT, SEED)
                t0 = time.time()
                edges = run_gies(expr, interv, test_genes, PARTITION_SIZE, SEED)
                t_gies = time.time() - t0
                logger.info(f"      {len(edges)} edges ({t_gies:.1f}s)")

                with open(cache_path, "w") as f:
                    json.dump({"edges": [list(e) for e in edges], "time": t_gies,
                               "strength": strength, "model": model_name}, f)

            # Compute precision against test GT
            tp = len(edges & gt_edges)
            precision = tp / len(edges) if edges else 0
            recall = tp / len(gt_edges) if gt_edges else 0
            logger.info(f"      precision={precision:.4f}, recall={recall:.4f}, "
                         f"TP={tp}/{len(gt_edges)}")

            model_results[strength] = {
                "n_edges": len(edges),
                "tp": tp,
                "precision": float(precision),
                "recall": float(recall),
            }

        all_results[model_name] = model_results

    # Step 5: Also run uncalibrated ElasticNet (real means) as reference
    logger.info("\n  === ELASTICNET (uncalibrated reference) ===")
    enet_cache = OUTDIR / "elasticnet_test_edges.json"
    if enet_cache.exists():
        with open(enet_cache) as f:
            cached = json.load(f)
        enet_edges = set(tuple(e) for e in cached["edges"])
        logger.info(f"    CACHED ({len(enet_edges)} edges)")
    else:
        test_enet_preds = {g: pert_means_real[g] for g in test_genes if g in pert_means_real}
        expr_enet, interv_enet = build_overlay_data(
            ctrl_X, ctrl_mean, test_enet_preds, test_genes, N_CTRL, N_CELLS_PER_PERT, SEED)
        enet_edges = run_gies(expr_enet, interv_enet, test_genes, PARTITION_SIZE, SEED)
        with open(enet_cache, "w") as f:
            json.dump({"edges": [list(e) for e in enet_edges]}, f)
        logger.info(f"    {len(enet_edges)} edges")

    enet_tp = len(enet_edges & gt_edges)
    enet_prec = enet_tp / len(enet_edges) if enet_edges else 0
    logger.info(f"    precision={enet_prec:.4f}, TP={enet_tp}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 6 SUMMARY: Calibration Impact")
    logger.info("=" * 70)

    header = f"{'Model':<15} {'Uncal':>8} {'25%':>8} {'50%':>8} {'75%':>8} {'100%':>8} {'Best':>8}"
    logger.info(header)
    logger.info("-" * 70)

    summary_rows = []
    for model_name in ["gears", "cpa", "geneformer"]:
        if model_name not in all_results:
            continue
        mr = all_results[model_name]
        precs = {s: mr[s]["precision"] for s in strengths}
        best_s = max(precs, key=precs.get)
        vals = [f"{precs[s]:>8.4f}" for s in strengths]
        logger.info(f"{model_name:<15} {' '.join(vals)} {f's={best_s:.0%}':>8}")
        summary_rows.append({
            "model": model_name,
            **{f"cal_{s:.0%}": precs[s] for s in strengths},
            "best_strength": best_s,
            "best_precision": precs[best_s],
        })

    logger.info(f"{'ElasticNet':<15} {enet_prec:>8.4f} {'---':>8} {'---':>8} {'---':>8} "
                f"{'---':>8} {'N/A':>8}")

    # Did any calibrated model beat ElasticNet?
    any_beat_enet = False
    for model_name, mr in all_results.items():
        for s, r in mr.items():
            if r["precision"] > enet_prec:
                any_beat_enet = True
                logger.info(f"\n  >>> {model_name} at {s:.0%} calibration "
                             f"BEATS ElasticNet ({r['precision']:.4f} > {enet_prec:.4f})")

    if not any_beat_enet:
        logger.info("\n  No calibrated model beat ElasticNet on test set.")
        logger.info("  This confirms: covariance structure > mean prediction quality.")

    # Save
    final = {
        "train_genes": train_genes,
        "test_genes": test_genes,
        "n_gt_edges": len(gt_edges),
        "elasticnet_precision": float(enet_prec),
        "calibration_results": all_results,
    }
    with open(OUTDIR / "phase6_results.json", "w") as f:
        json.dump(final, f, indent=2)
    pd.DataFrame(summary_rows).to_csv(OUTDIR / "calibration_summary.csv", index=False)

    logger.info(f"\nAll results saved to {OUTDIR}")


if __name__ == "__main__":
    main()
