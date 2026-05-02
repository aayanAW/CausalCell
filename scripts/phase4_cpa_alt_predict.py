"""Re-run prediction step on the already-trained CPA alt-loss model.

The training in `phase4_cpa_alt_loss.py` succeeded and saved a checkpoint to
`data/model_outputs_real_n200_alt_loss/cpa_alt_loss_checkpoint/`, but the
prediction step crashed because `cpa.CPA.predict()` does not take `cf_pert`.
The correct API in cpa-tools 0.8.8 is to:

  1. Take the control AnnData
  2. Set the perturbation column to the target gene
  3. Call `predict(adata=...)` which infers conditional on the new label

Usage:
    PYTHONPATH=. python3 scripts/phase4_cpa_alt_predict.py
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import anndata as ad
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cpa_alt_predict")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_DIR = DATA / "model_outputs_real_n200_alt_loss"
CKPT_DIR = OUT_DIR / "cpa_alt_loss_checkpoint"
K562 = DATA / "k562_subset_n200_slim.h5ad"
GENE_LIST = DATA / "eval_genes_n200.txt"
PERT_COL = "gene"
CTRL = "non-targeting"


def main():
    import torch
    log.info(f"PyTorch: {torch.__version__}")

    log.info("Loading K562 subset ...")
    adata = ad.read_h5ad(K562)
    adata.var_names = [str(g) for g in adata.var_names]
    log.info(f"  shape: {adata.shape}")

    if GENE_LIST.exists():
        gene_subset = sorted([line.strip() for line in open(GENE_LIST) if line.strip()])
    else:
        rng = np.random.default_rng(42)
        pert_counts = adata.obs[PERT_COL].value_counts()
        var_set = set(adata.var_names)
        eligible = sorted([g for g, c in pert_counts.items()
                           if g != CTRL and c >= 10 and g in var_set])
        gene_subset = sorted(rng.choice(eligible, size=200, replace=False).tolist())
    log.info(f"  gene subset: {len(gene_subset)}")

    keep_mask = (adata.obs[PERT_COL] == CTRL) | (adata.obs[PERT_COL].isin(gene_subset))
    adata = adata[keep_mask].copy()
    adata.obs["is_control"] = (adata.obs[PERT_COL] == CTRL).astype(int)
    adata.obs["dose"] = 1.0
    adata.obs["cov_celltype"] = "K562"

    import cpa
    log.info("Loading saved CPA model ...")
    cpa.CPA.setup_anndata(
        adata,
        perturbation_key=PERT_COL,
        dosage_key="dose",
        control_group=CTRL,
        is_count_data=True,
        categorical_covariate_keys=["cov_celltype"],
        deg_uns_key=None,
        deg_uns_cat_key="cov_celltype",
        max_comb_len=1,
    )
    try:
        model = cpa.CPA.load(str(CKPT_DIR), adata=adata)
    except Exception as e:
        log.error(f"Could not load model: {e}")
        return 1
    log.info("Model loaded.")

    # Per-perturbation predict: copy controls, relabel to target gene, predict
    log.info("Predicting per-perturbation by relabeling control cells ...")
    ctrl_mask = adata.obs["is_control"] == 1
    ctrl_adata = adata[ctrl_mask].copy()
    if ctrl_adata.n_obs > 1000:
        rng = np.random.default_rng(42)
        idx = rng.choice(ctrl_adata.n_obs, size=1000, replace=False)
        ctrl_adata = ctrl_adata[idx].copy()
    log.info(f"  context cells: {ctrl_adata.n_obs}")

    pred_means = []
    pred_genes = []
    failures = 0
    for i, gene in enumerate(gene_subset):
        try:
            cf = ctrl_adata.copy()
            cf.obs[PERT_COL] = gene
            cf.obs["is_control"] = 0
            try:
                cpa.CPA.setup_anndata(
                    cf,
                    perturbation_key=PERT_COL, dosage_key="dose",
                    control_group=CTRL, is_count_data=True,
                    categorical_covariate_keys=["cov_celltype"],
                    deg_uns_key=None, deg_uns_cat_key="cov_celltype",
                    max_comb_len=1,
                )
            except Exception:
                pass

            # CPA 0.8.8 writes predictions to adata.obsm['CPA_pred'] in place;
            # the call returns None.
            model.predict(adata=cf, batch_size=128, n_samples=10, return_mean=True)
            if "CPA_pred" not in cf.obsm:
                raise RuntimeError("predict() did not populate obsm['CPA_pred']")
            X = np.asarray(cf.obsm["CPA_pred"])
            mean_pred = X.mean(axis=0)
            pred_means.append(mean_pred)
            pred_genes.append(gene)
            if (i + 1) % 25 == 0:
                log.info(f"  predicted {i+1}/{len(gene_subset)}")
        except Exception as e:
            failures += 1
            if failures <= 5:
                log.warning(f"  {gene}: {e}")
    log.info(f"Total predictions: {len(pred_means)} / {len(gene_subset)}; failures: {failures}")

    if not pred_means:
        log.error("No predictions; aborting")
        return 1

    pred_array = np.stack(pred_means, axis=0).astype(np.float32)
    pred_adata = ad.AnnData(
        X=pred_array,
        obs={"perturbation": pred_genes},
        var={"feature_name": list(adata.var_names)},
    )
    pred_adata.var_names = list(adata.var_names)
    pred_adata.uns["output_type"] = "mean"
    pred_adata.uns["model"] = "cpa-alt-loss-gauss"
    pred_adata.uns["recon_loss"] = "gauss"

    out_path = OUT_DIR / "cpa_predictions.h5ad"
    pred_adata.write_h5ad(out_path)
    log.info(f"Wrote {out_path}: {pred_adata.shape}")

    diag = {
        "model": "cpa-alt-loss-gauss",
        "recon_loss": "gauss",
        "n_predictions": len(pred_genes),
        "n_failures": failures,
        "out_path": str(out_path),
    }
    with open(OUT_DIR / "training_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
