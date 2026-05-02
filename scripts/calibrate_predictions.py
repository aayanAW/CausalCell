"""Apply post-hoc effect-size calibration to existing virtual cell model predictions.

Reads the .h5ad outputs from ``data/model_outputs_real_n200/`` (vanilla model
predictions), fits a QuantileCalibrator on the empirical |log2FC| distribution
of real K562 perturbations at the same gene scale, applies the calibration to
each model's predicted means, and writes calibrated .h5ad files to
``data/model_outputs_real_n200_calibrated/``.

After this script runs, the existing eval pipeline can be invoked with
``--model-outputs-dir data/model_outputs_real_n200_calibrated`` to produce
the calibrated counterpart of the headline N=200 K562 results.

Usage:
    python3 scripts/calibrate_predictions.py
    python3 scripts/calibrate_predictions.py --models gears,cpa,geneformer
    python3 scripts/calibrate_predictions.py --diagnostics

The calibration is rank-preserving (Bolstad 2003; Reinhard 2001), so per-gene
relative attribution is unchanged; only the magnitude distribution is corrected.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np

from causalcellbench.calibration import (
    QuantileCalibrator,
    empirical_real_log2fc,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("calibrate")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
DEFAULT_INPUT_DIR = DATA / "model_outputs_real_n200"
DEFAULT_OUTPUT_DIR = DATA / "model_outputs_real_n200_calibrated"
DEFAULT_K562_SUBSET = DATA / "k562_subset_n200_slim.h5ad"


def load_real_perturbation_data(
    h5ad_path: Path,
    pert_col: str = "gene",
    control_label: str = "non-targeting",
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Read the real K562 subset and return (pert_expr, pert_labels, ctrl_means, gene_names).

    pert_expr : (n_pert_cells, n_genes) dense array of real perturbation cells
    pert_labels : per-cell perturbation gene name
    ctrl_means : (n_genes,) per-gene control mean expression
    gene_names : list of gene names in column order
    """
    logger.info("Reading real K562 subset from %s", h5ad_path)
    adata = ad.read_h5ad(h5ad_path)
    adata.var_names = [str(x) for x in adata.var_names]
    gene_names = list(adata.var_names)

    if pert_col not in adata.obs.columns:
        raise KeyError(
            f"Perturbation column '{pert_col}' missing; have {list(adata.obs.columns)}"
        )

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float64)

    is_ctrl = adata.obs[pert_col] == control_label
    if is_ctrl.sum() == 0:
        raise ValueError(f"No control cells found with label '{control_label}'.")

    ctrl_means = X[is_ctrl.values].mean(axis=0)

    pert_mask = ~is_ctrl
    pert_expr = X[pert_mask.values]
    pert_labels = adata.obs[pert_col].values[pert_mask.values].tolist()

    logger.info(
        "  %d control cells, %d perturbation cells across %d gene labels",
        int(is_ctrl.sum()),
        int(pert_mask.sum()),
        len(set(pert_labels)),
    )
    return pert_expr, pert_labels, ctrl_means, gene_names


def calibrate_model_file(
    in_path: Path,
    out_path: Path,
    calibrator: QuantileCalibrator,
    ctrl_means_full: np.ndarray,
    full_gene_names: list[str],
) -> dict:
    """Read a vanilla model .h5ad, calibrate, write to out_path. Return diagnostics dict."""
    logger.info("Calibrating %s -> %s", in_path.name, out_path.name)
    pred = ad.read_h5ad(in_path)
    pred_genes = list(pred.var_names)

    X = pred.X.toarray() if hasattr(pred.X, "toarray") else np.asarray(pred.X)
    X = X.astype(np.float64)  # (n_perts, n_genes_pred)

    # Align control means to the prediction's gene order. Genes missing from
    # the K562 subset (which is 200 genes) get a small fallback so calibration
    # can run on the full 8.5k gene matrix; only the 200 evaluation genes drive
    # downstream causal recovery, and those are guaranteed to be present.
    full_to_idx = {g: i for i, g in enumerate(full_gene_names)}
    ctrl_aligned = np.full(len(pred_genes), 1e-3, dtype=np.float64)
    matched = 0
    for j, g in enumerate(pred_genes):
        if g in full_to_idx:
            ctrl_aligned[j] = ctrl_means_full[full_to_idx[g]]
            matched += 1
    logger.info(
        "    %d/%d prediction genes aligned to real ctrl means",
        matched, len(pred_genes),
    )

    # Pre-cal diagnostic
    ps = calibrator.pseudocount
    pre_log2fc = np.log2((X + ps) / (ctrl_aligned[None, :] + ps))
    pre_abs = np.abs(pre_log2fc)
    pre_diag = {
        "frac_large_pre": float((pre_abs > 0.5).mean()),
        "mean_abs_pre": float(pre_abs.mean()),
        "frac_near_zero_pre": float((pre_abs < 0.1).mean()),
    }

    # Apply calibration
    X_cal = calibrator.transform(X, ctrl_aligned)

    # Post-cal diagnostic
    post_log2fc = np.log2((X_cal + ps) / (ctrl_aligned[None, :] + ps))
    post_abs = np.abs(post_log2fc)
    post_diag = {
        "frac_large_post": float((post_abs > 0.5).mean()),
        "mean_abs_post": float(post_abs.mean()),
        "frac_near_zero_post": float((post_abs < 0.1).mean()),
    }

    # Write calibrated file with the same schema as the input
    pred_cal = ad.AnnData(
        X=X_cal.astype(np.float32),
        obs=pred.obs.copy(),
        var=pred.var.copy(),
    )
    pred_cal.uns["output_type"] = pred.uns.get("output_type", "mean")
    pred_cal.uns["model"] = pred.uns.get("model", in_path.stem.replace("_predictions", ""))
    pred_cal.uns["calibration"] = {
        "method": "quantile_matching",
        "reference": "real_K562_perturbations_log2FC",
        "pseudocount": calibrator.pseudocount,
        "n_quantile_knots": calibrator.n_quantiles,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_cal.write_h5ad(out_path)
    logger.info("    wrote %s (%d perts, %d genes)", out_path.name, X_cal.shape[0], X_cal.shape[1])

    diag = {**pre_diag, **post_diag}
    diag["inflation_ratio_pre_to_real"] = (
        pre_diag["frac_large_pre"] / 0.027 if pre_diag["frac_large_pre"] > 0 else 0
    )
    diag["inflation_ratio_post_to_real"] = (
        post_diag["frac_large_post"] / 0.027 if post_diag["frac_large_post"] > 0 else 0
    )
    return diag


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory with vanilla {model}_predictions.h5ad files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write calibrated predictions",
    )
    parser.add_argument(
        "--k562-subset",
        type=Path,
        default=DEFAULT_K562_SUBSET,
        help="Path to the K562 200-gene subset .h5ad",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gears,cpa,geneformer",
        help="Comma-separated list of model names to calibrate",
    )
    parser.add_argument(
        "--n-quantiles",
        type=int,
        default=1000,
        help="Number of quantile knots for the calibrator",
    )
    parser.add_argument(
        "--pseudocount",
        type=float,
        default=0.01,
        help="Pseudocount for log2FC computation; matches overlay sampling",
    )
    parser.add_argument(
        "--diagnostics-out",
        type=Path,
        default=None,
        help="Optional path to write per-model calibration diagnostics JSON",
    )
    args = parser.parse_args()

    pert_expr, pert_labels, ctrl_means, gene_names = load_real_perturbation_data(
        args.k562_subset
    )

    # Compute the empirical |log2FC| distribution from real perturbations.
    # This is the reference distribution every model gets calibrated to.
    real_abs_log2fc = empirical_real_log2fc(
        pert_expr, pert_labels, ctrl_means, pseudocount=args.pseudocount,
    )
    logger.info(
        "Real K562 |log2FC| reference: %d values, mean=%.3f, frac>0.5=%.4f",
        real_abs_log2fc.size,
        float(real_abs_log2fc.mean()),
        float((real_abs_log2fc > 0.5).mean()),
    )

    calibrator = QuantileCalibrator(
        n_quantiles=args.n_quantiles,
        pseudocount=args.pseudocount,
    ).fit(real_abs_log2fc)

    diagnostics: dict[str, dict] = {}
    for model in args.models.split(","):
        model = model.strip()
        in_path = args.input_dir / f"{model}_predictions.h5ad"
        if not in_path.exists():
            logger.warning("  %s not found; skipping", in_path)
            continue
        out_path = args.output_dir / f"{model}_predictions.h5ad"
        diagnostics[model] = calibrate_model_file(
            in_path, out_path, calibrator, ctrl_means, gene_names,
        )
        d = diagnostics[model]
        logger.info(
            "  %s: frac>0.5  %.3f -> %.3f   (real ref: 0.027, ratios %.1fx -> %.1fx)",
            model,
            d["frac_large_pre"], d["frac_large_post"],
            d["inflation_ratio_pre_to_real"], d["inflation_ratio_post_to_real"],
        )

    if args.diagnostics_out:
        args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.diagnostics_out, "w") as f:
            json.dump(
                {
                    "real_reference": {
                        "n_values": int(real_abs_log2fc.size),
                        "mean_abs": float(real_abs_log2fc.mean()),
                        "frac_large": float((real_abs_log2fc > 0.5).mean()),
                        "frac_near_zero": float((real_abs_log2fc < 0.1).mean()),
                    },
                    "per_model": diagnostics,
                },
                f,
                indent=2,
            )
        logger.info("Wrote diagnostics to %s", args.diagnostics_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
