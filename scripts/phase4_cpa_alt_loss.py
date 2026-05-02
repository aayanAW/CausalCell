"""Phase 4 — Mechanism via alternative loss: train CPA with Gaussian (MSE)
reconstruction loss instead of the default Negative-Binomial.

The paper's central mechanism claim is that the per-gene MSE training
objective produces dense, inflated predictions across all deep architectures.
Vanilla CPA in the existing pipeline already uses NB loss (a more
sophisticated count-aware likelihood). To probe the mechanism claim, we
retrain CPA with the Gaussian recon loss — the textbook per-gene MSE
objective — and check whether (a) the per-perturbation Δ-variance ratio
drops further, (b) inter-perturbation cosine similarity rises further,
(c) downstream causal F1 changes against the GIES ground truth.

Note: this trains *one* model variant, not a full sweep. It's a single
mechanism probe, not an ablation. Output is a `.h5ad` predictions file in
the same schema as the existing CPA outputs.

Usage:
    PYTHONPATH=. python3 scripts/phase4_cpa_alt_loss.py
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
import scanpy as sc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("cpa_alt_loss")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_DIR = DATA / "model_outputs_real_n200_alt_loss"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K562 = DATA / "k562_subset_n200_slim.h5ad"
GENE_LIST = DATA / "eval_genes_n200.txt"
PERT_COL = "gene"
CTRL = "non-targeting"

# CPA hyperparameters: keep close to v1 defaults
SEED = 42
EPOCHS = 50
BATCH_SIZE = 128
LATENT_DIM = 64


def main():
    import torch
    log.info(f"PyTorch: {torch.__version__}, MPS: {torch.backends.mps.is_available()}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info(f"Device: {device}")

    log.info("Loading K562 subset ...")
    adata = ad.read_h5ad(K562)
    adata.var_names = [str(g) for g in adata.var_names]
    log.info(f"  shape: {adata.shape}")

    # Read gene list (this is what the existing CPA was trained on)
    if GENE_LIST.exists():
        gene_subset = sorted([line.strip() for line in open(GENE_LIST) if line.strip()])
    else:
        rng = np.random.default_rng(SEED)
        pert_counts = adata.obs[PERT_COL].value_counts()
        var_set = set(adata.var_names)
        eligible = sorted([g for g, c in pert_counts.items()
                           if g != CTRL and c >= 10 and g in var_set])
        gene_subset = sorted(rng.choice(eligible, size=200, replace=False).tolist())
    log.info(f"  gene subset: {len(gene_subset)}")

    # Subset perturbations to the eval gene set + controls
    keep_mask = (adata.obs[PERT_COL] == CTRL) | (adata.obs[PERT_COL].isin(gene_subset))
    adata = adata[keep_mask].copy()
    log.info(f"  after subsetting to eval perturbations: {adata.shape}")

    # CPA expects a "control" key column with bool/int
    adata.obs["is_control"] = (adata.obs[PERT_COL] == CTRL).astype(int)
    adata.obs["dose"] = 1.0
    adata.obs["cov_celltype"] = "K562"

    # Setup CPA
    try:
        import cpa
    except ImportError as e:
        log.error(f"cpa-tools not importable: {e}")
        return 1
    log.info(f"CPA package loaded; default config setup ...")

    # Try to set up the CPA workflow.  cpa-tools is sensitive to the exact
    # API version; wrap the calls in try/except to fail loud if the API
    # changed.
    try:
        cpa.CPA.setup_anndata(
            adata,
            perturbation_key=PERT_COL,
            dosage_key="dose",
            control_group=CTRL,
            batch_key=None,
            is_count_data=True,
            categorical_covariate_keys=["cov_celltype"],
            deg_uns_key=None,
            deg_uns_cat_key="cov_celltype",
            max_comb_len=1,
        )
    except Exception as e:
        log.warning(f"setup_anndata raised: {e}; trying minimal setup ...")
        try:
            cpa.CPA.setup_anndata(
                adata,
                perturbation_key=PERT_COL,
                dosage_key="dose",
                control_group=CTRL,
                is_count_data=True,
                max_comb_len=1,
            )
        except Exception as e2:
            log.error(f"CPA setup_anndata failed: {e2}")
            return 1

    log.info("Configuring CPA model with recon_loss='gauss' (Gaussian, per-gene MSE) ...")
    model_params = {
        "n_latent": LATENT_DIM,
        "recon_loss": "gauss",       # ← ALT LOSS: Gaussian MSE instead of vanilla NB
        "doser_type": "logsigm",
        "n_hidden_encoder": 256,
        "n_layers_encoder": 3,
        "n_hidden_decoder": 256,
        "n_layers_decoder": 3,
        "use_batch_norm_encoder": True,
        "use_layer_norm_encoder": False,
        "use_batch_norm_decoder": True,
        "use_layer_norm_decoder": False,
        "dropout_rate_encoder": 0.0,
        "dropout_rate_decoder": 0.0,
        "variational": True,
    }
    trainer_params = {
        "n_epochs_kl_warmup": 0,
        "n_epochs_pretrain_ae": 5,
        "n_epochs_adv_warmup": 5,
        "n_epochs_mixup_warmup": 0,
        "mixup_alpha": 0.0,
        "adv_steps": 2,
        "n_hidden_adv": 64,
        "n_layers_adv": 2,
        "use_batch_norm_adv": True,
        "use_layer_norm_adv": False,
        "dropout_rate_adv": 0.3,
        "reg_adv": 20,
        "pen_adv": 5.0,
        "lr": 0.0003,
        "wd": 4e-7,
        "adv_lr": 0.0003,
        "adv_wd": 4e-7,
        "doser_lr": 0.0003,
        "doser_wd": 4e-7,
        "do_clip_grad": True,
        "adv_loss": "cce",
        "gradient_clip_value": 1.0,
        "step_size_lr": 45,
    }

    try:
        model = cpa.CPA(adata=adata, **model_params)
    except Exception as e:
        log.error(f"CPA model init failed: {e}")
        return 1
    log.info("Training CPA with Gaussian loss ...")

    t0 = time.time()
    try:
        model.train(
            max_epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            plan_kwargs=trainer_params,
            early_stopping_patience=10,
            check_val_every_n_epoch=5,
            save_path=str(OUT_DIR / "cpa_alt_loss_checkpoint"),
        )
    except Exception as e:
        log.error(f"Training failed: {e}")
        return 1
    train_time = time.time() - t0
    log.info(f"  done ({train_time:.1f}s)")

    # Predict perturbation effects
    log.info("Predicting perturbation effects ...")
    pred_means = []
    pred_genes = []
    ctrl_cells = adata[adata.obs["is_control"] == 1].copy()
    if ctrl_cells.n_obs > 1000:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(ctrl_cells.n_obs, size=1000, replace=False)
        ctrl_cells = ctrl_cells[idx].copy()

    for gene in gene_subset:
        try:
            pred_adata = model.predict(
                ctrl_cells,
                cf_pert={PERT_COL: gene, "dose": 1.0, "cov_celltype": "K562"},
            )
            mean_pred = np.asarray(pred_adata.X.mean(axis=0)).flatten()
            pred_means.append(mean_pred)
            pred_genes.append(gene)
        except Exception as e:
            log.warning(f"  {gene}: {e}")
            continue

    if not pred_means:
        log.error("No predictions produced")
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
    log.info(f"Wrote {out_path}: {pred_adata.shape}, train_time={train_time:.1f}s")

    diag = {
        "model": "cpa-alt-loss-gauss",
        "recon_loss": "gauss",
        "epochs": EPOCHS,
        "n_predictions": len(pred_genes),
        "train_time_seconds": train_time,
        "out_path": str(out_path),
    }
    with open(OUT_DIR / "training_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
