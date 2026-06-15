"""Run CPA training + prediction locally on CPU/MPS.

Trains CPA on K562 CRISPRi data and produces real perturbation
predictions as .h5ad for the CausalCellBench evaluation pipeline.

CPA outputs full single cells (not means), so the eval pipeline
uses output_type="cells" — no overlay sampling needed.

Usage:
    .venv_gears/bin/python3 scripts/run_cpa_local.py --n-genes 50 --max-epochs 100
    .venv_gears/bin/python3 scripts/run_cpa_local.py --gene-list data/eval_genes_n50.txt
"""

import argparse
import os
import sys
import time
import json

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def main():
    parser = argparse.ArgumentParser(description="Train CPA locally on K562")
    parser.add_argument("--data-path", default="data/k562_real.h5ad",
                        help="Path to K562 h5ad file")
    parser.add_argument("--output-dir", default="data/model_outputs_real",
                        help="Directory for output predictions")
    parser.add_argument("--n-genes", type=int, default=50,
                        help="Number of perturbation genes")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Maximum training epochs")
    parser.add_argument("--n-latent", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-cells-per-pert", type=int, default=100,
                        help="Cells to generate per perturbation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gene-list", default=None,
                        help="File with one gene per line (overrides --n-genes)")
    parser.add_argument("--recon-loss", default="nb",
                        choices=["nb", "zinb", "gauss"])
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU (no CUDA available)")

    import cpa
    print(f"CPA version: {getattr(cpa, '__version__', 'unknown')}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if not os.path.exists(args.data_path):
        print(f"ERROR: {args.data_path} not found")
        sys.exit(1)

    print(f"Loading data from {args.data_path}...")
    adata = ad.read_h5ad(args.data_path)
    print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    # Ensembl -> symbol mapping
    if any(str(v).startswith("ENSG") for v in adata.var_names[:5]):
        if "gene_name" in adata.var.columns:
            print("Converting Ensembl IDs to gene symbols...")
            adata.var["ensembl_id"] = adata.var_names.copy()
            adata.var_names = adata.var["gene_name"].astype(str).values
            adata.var_names_make_unique()

    # Find perturbation column
    perturbation_key = None
    for key in ["gene", "perturbation", "guide_identity", "condition", "cond_harm", "target_gene"]:
        if key in adata.obs.columns:
            perturbation_key = key
            break
    if perturbation_key is None:
        print(f"ERROR: No perturbation column. Available: {list(adata.obs.columns)}")
        sys.exit(1)
    print(f"Perturbation key: '{perturbation_key}'")

    # Find control label
    unique_perts = adata.obs[perturbation_key].astype(str).unique()
    control_label = None
    for c in ["non-targeting", "ctrl", "control", "NT", "non_targeting"]:
        if c in unique_perts:
            control_label = c
            break
    if control_label is None:
        print(f"ERROR: No control label found. Unique: {sorted(unique_perts)[:20]}")
        sys.exit(1)

    is_control = adata.obs[perturbation_key].astype(str) == control_label
    n_control = is_control.sum()
    perturbation_names = sorted([p for p in unique_perts if p != control_label])
    print(f"Control cells: {n_control}, Perturbations: {len(perturbation_names)}")

    # Use counts layer if available
    if "counts" in adata.layers:
        print("Using 'counts' layer")
        adata.X = adata.layers["counts"].copy()
    elif "raw_counts" in adata.layers:
        print("Using 'raw_counts' layer")
        adata.X = adata.layers["raw_counts"].copy()

    # ------------------------------------------------------------------
    # 3. Select genes
    # ------------------------------------------------------------------
    gene_set = set(adata.var_names)
    valid_pert_genes = [g for g in perturbation_names if g in gene_set]
    print(f"Valid perturbation genes (in var_names): {len(valid_pert_genes)}")

    if args.gene_list and os.path.exists(args.gene_list):
        with open(args.gene_list) as f:
            requested = [line.strip() for line in f if line.strip()]
        selected_genes = [g for g in requested if g in set(valid_pert_genes)]
        print(f"From gene list: {len(selected_genes)}/{len(requested)} found")
    else:
        rng_sel = np.random.default_rng(args.seed)
        if args.n_genes > 0 and args.n_genes < len(valid_pert_genes):
            selected_genes = sorted(
                rng_sel.choice(valid_pert_genes, size=args.n_genes, replace=False)
            )
        else:
            selected_genes = sorted(valid_pert_genes)
    print(f"Selected {len(selected_genes)} genes for prediction")

    # ------------------------------------------------------------------
    # 4. Subset data to selected perturbations + controls
    # ------------------------------------------------------------------
    mask = is_control.copy()
    for gene in selected_genes:
        mask |= (adata.obs[perturbation_key].astype(str) == gene)
    adata_sub = adata[mask].copy()

    # Densify if sparse (CPA works with dense)
    if sp.issparse(adata_sub.X):
        adata_sub.X = np.asarray(adata_sub.X.todense())

    print(f"Training subset: {adata_sub.n_obs} cells, {adata_sub.n_vars} genes")

    # ------------------------------------------------------------------
    # 5. Create dosage column and train/valid split
    # ------------------------------------------------------------------
    dosage_key = "cpa_dosage"
    adata_sub.obs[dosage_key] = np.where(
        adata_sub.obs[perturbation_key].astype(str) == control_label,
        0.0, 1.0
    ).astype(np.float32)

    rng = np.random.default_rng(args.seed)

    # Hold out ~5% of perturbations as OOD
    n_ood = max(2, len(selected_genes) // 20)
    ood_perts = set(rng.choice(selected_genes, size=n_ood, replace=False))
    print(f"OOD perturbations: {sorted(ood_perts)}")

    split = np.full(adata_sub.n_obs, "train", dtype=object)
    for i in range(adata_sub.n_obs):
        pert = str(adata_sub.obs[perturbation_key].iloc[i])
        if pert in ood_perts:
            split[i] = "ood"
        elif rng.random() < 0.15:
            split[i] = "valid"
    adata_sub.obs["cpa_split"] = split
    for s in ["train", "valid", "ood"]:
        print(f"  {s}: {(split == s).sum()} cells")

    # ------------------------------------------------------------------
    # 6. Setup and train CPA
    # ------------------------------------------------------------------
    print("\nSetting up CPA...")
    try:
        cpa.CPA.setup_anndata(
            adata_sub,
            perturbation_key=perturbation_key,
            control_group=control_label,
            dosage_key=dosage_key,
            is_count_data=(args.recon_loss in ("nb", "zinb")),
            max_comb_len=1,
        )
    except TypeError:
        cpa.CPA.setup_anndata(
            adata_sub,
            drug_key=perturbation_key,
            dose_key=dosage_key,
            control_key=control_label,
        )

    model = cpa.CPA(
        adata=adata_sub,
        split_key="cpa_split",
        train_split="train",
        valid_split="valid",
        test_split="ood",
        n_latent=args.n_latent,
        recon_loss=args.recon_loss,
        variational=True,
        n_hidden_encoder=256,
        n_layers_encoder=3,
        n_hidden_decoder=256,
        n_layers_decoder=2,
        dropout_rate_encoder=0.1,
        dropout_rate_decoder=0.0,
        doser_type="linear",
        seed=args.seed,
    )

    plan_kwargs = {
        "lr": 1e-4,
        "n_epochs_kl_warmup": 20,
        "n_epochs_adv_warmup": 50,
        "n_epochs_pretrain_ae": 10,
    }

    print(f"\nTraining CPA for up to {args.max_epochs} epochs...")
    t_train_start = time.time()

    model.train(
        max_epochs=args.max_epochs,
        use_gpu=use_gpu,
        batch_size=args.batch_size,
        plan_kwargs=plan_kwargs,
        early_stopping_patience=15,
        check_val_every_n_epoch=5,
        save_path=os.path.join(args.output_dir, "cpa_logs"),
    )

    t_train = time.time() - t_train_start
    print(f"Training complete in {t_train:.1f}s ({t_train / 60:.1f} min)")

    # Save model
    model_dir = os.path.join(args.output_dir, "cpa_checkpoint")
    model.save(model_dir, overwrite=True, save_anndata=True)
    print(f"Model saved to {model_dir}")

    # ------------------------------------------------------------------
    # 7. Predict perturbation effects
    # ------------------------------------------------------------------
    # CPA predicts counterfactual cells: take control cells, set perturbation,
    # and generate predicted expression under that perturbation
    control_mask = adata_sub.obs[perturbation_key].astype(str) == control_label
    control_adata = adata_sub[control_mask].copy()
    print(f"\nPredicting {len(selected_genes)} perturbations "
          f"({args.n_cells_per_pert} cells each)...")

    all_cells = []
    all_labels = []
    failed = []
    t_pred_start = time.time()

    rng_pred = np.random.default_rng(args.seed + 1)

    for i, pert in enumerate(selected_genes):
        if i % 10 == 0:
            elapsed = time.time() - t_pred_start
            print(f"  {i}/{len(selected_genes)} ({elapsed:.1f}s)")

        try:
            n_ctrl = control_adata.n_obs
            if n_ctrl >= args.n_cells_per_pert:
                idx = rng_pred.choice(n_ctrl, size=args.n_cells_per_pert, replace=False)
            else:
                idx = rng_pred.choice(n_ctrl, size=args.n_cells_per_pert, replace=True)

            cf_adata = control_adata[idx].copy()
            cf_adata.obs[perturbation_key] = pert
            cf_adata.obs[dosage_key] = 1.0

            model.predict(
                adata=cf_adata,
                batch_size=min(256, args.n_cells_per_pert),
                n_samples=20,
                return_mean=True,
            )

            pred_key = "CPA_pred"
            if pred_key in cf_adata.obsm:
                expr = np.asarray(cf_adata.obsm[pred_key], dtype=np.float32)
                # For the eval pipeline, we save the MEAN across cells
                # (the eval pipeline expects one row per perturbation for mean output,
                # but CPA natively outputs cells — save cells for flexibility)
                all_cells.append(expr)
                all_labels.extend([pert] * expr.shape[0])
            else:
                print(f"  WARN: No CPA_pred key for '{pert}', keys: {list(cf_adata.obsm.keys())}")
                failed.append(pert)

        except Exception as e:
            print(f"  WARN: Failed on '{pert}': {e}")
            failed.append(pert)

    t_pred = time.time() - t_pred_start
    n_success = len(set(all_labels))
    print(f"Prediction complete: {n_success} OK, {len(failed)} failed ({t_pred:.1f}s)")

    if not all_cells:
        print("ERROR: No successful predictions!")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 8. Save results — two formats
    # ------------------------------------------------------------------
    pred_matrix = np.concatenate(all_cells, axis=0)

    # Save full cells version (CPA native output)
    result_cells = ad.AnnData(
        X=pred_matrix.astype(np.float32),
        obs=pd.DataFrame({"perturbation": all_labels}),
        var=adata_sub.var[[]].copy() if pred_matrix.shape[1] == adata_sub.n_vars else pd.DataFrame(
            index=[f"gene_{i}" for i in range(pred_matrix.shape[1])]
        ),
        uns={
            "model": "cpa",
            "output_type": "cells",
            "n_cells_per_pert": args.n_cells_per_pert,
            "n_perturbations": n_success,
            "n_failed": len(failed),
            "failed_perturbations": failed[:50],
            "ood_perturbations": sorted(ood_perts),
            "cell_type": "K562",
            "training_time_seconds": t_train,
            "inference_time_seconds": t_pred,
            "max_epochs": args.max_epochs,
            "n_latent": args.n_latent,
            "recon_loss": args.recon_loss,
            "seed": args.seed,
        },
    )

    # Save cells output as secondary (for reference, but not used by GIES — too slow)
    cells_path = os.path.join(args.output_dir, "cpa_predictions_cells.h5ad")
    result_cells.write_h5ad(cells_path)
    print(f"\nSaved cells output to {cells_path}")
    print(f"  Shape: {result_cells.shape}")
    print(f"  X mean: {pred_matrix.mean():.4f}, std: {pred_matrix.std():.4f}")

    # Save mean version as PRIMARY output (GIES uses overlay sampling on means)
    # CPA cells have zero_fraction=0.0 (fully dense) which makes GIES ~100x slower
    unique_perts_pred = sorted(set(all_labels))
    mean_matrix = np.zeros((len(unique_perts_pred), pred_matrix.shape[1]), dtype=np.float32)
    for j, p in enumerate(unique_perts_pred):
        pmask = np.array(all_labels) == p
        mean_matrix[j] = pred_matrix[pmask].mean(axis=0)

    result_mean = ad.AnnData(
        X=mean_matrix,
        obs=pd.DataFrame({"perturbation": unique_perts_pred}),
        var=result_cells.var.copy(),
        uns={
            "model": "cpa",
            "output_type": "mean",
            "cell_type": "K562",
            "n_perturbations": len(unique_perts_pred),
        },
    )
    mean_path = os.path.join(args.output_dir, "cpa_predictions.h5ad")
    result_mean.write_h5ad(mean_path)
    print(f"Saved mean output (PRIMARY) to {mean_path}")
    print(f"  Shape: {result_mean.shape}")

    total_time = t_train + t_pred
    print(f"\nTotal: train={t_train/60:.1f}min, pred={t_pred:.1f}s, "
          f"total={total_time/60:.1f}min")


if __name__ == "__main__":
    main()
