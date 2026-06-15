"""Run GEARS training + prediction locally on Apple M4 (MPS).

Trains GEARS on K562 CRISPRi data and produces real perturbation
mean predictions as .h5ad for the CausalCellBench evaluation pipeline.

Usage:
    # From project root, using the GEARS venv:
    .venv_gears/bin/python3 scripts/run_gears_local.py --n-genes 200 --epochs 20

    # Quick 50-gene test:
    .venv_gears/bin/python3 scripts/run_gears_local.py --n-genes 50 --epochs 5
"""

import argparse
import os
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def main():
    parser = argparse.ArgumentParser(description="Train GEARS locally on K562")
    parser.add_argument("--data-path", default="data/k562_real.h5ad",
                        help="Path to K562 h5ad file")
    parser.add_argument("--output-dir", default="data/model_outputs",
                        help="Directory for output predictions")
    parser.add_argument("--cache-dir", default="data/gears_cache",
                        help="GEARS cache directory")
    parser.add_argument("--n-genes", type=int, default=200,
                        help="Number of perturbation genes")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (must match eval pipeline, default=42)")
    parser.add_argument("--gene-list", default=None,
                        help="File with one gene per line (overrides --n-genes and --seed selection)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, mps, cpu, cuda")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # Monkey-patch Series.nonzero() for newer pandas (removed in 2.0+)
    import pandas as _pd
    if not hasattr(_pd.Series, 'nonzero'):
        _pd.Series.nonzero = lambda self: (self.to_numpy().nonzero()[0],)

    # Import GEARS
    from gears import PertData, GEARS

    # ---------------------------------------------------------------
    # 1. Load K562 data
    # ---------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    if not os.path.exists(args.data_path):
        print(f"ERROR: {args.data_path} not found")
        sys.exit(1)

    print(f"Loading data from {args.data_path}...")
    adata = ad.read_h5ad(args.data_path)
    print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

    # ---------------------------------------------------------------
    # 2. Identify perturbation structure
    # ---------------------------------------------------------------
    perturbation_key = None
    for key in ["gene", "perturbation", "guide_identity", "condition"]:
        if key in adata.obs.columns:
            perturbation_key = key
            break
    if perturbation_key is None:
        print(f"ERROR: No perturbation column. Available: {list(adata.obs.columns)}")
        sys.exit(1)
    print(f"Perturbation key: '{perturbation_key}'")

    control_label = None
    unique_perts = adata.obs[perturbation_key].astype(str).unique()
    for c in ["non-targeting", "ctrl", "control", "NT"]:
        if c in unique_perts:
            control_label = c
            break
    if control_label is None:
        print(f"ERROR: No control label found in {sorted(unique_perts)[:20]}")
        sys.exit(1)

    is_control = adata.obs[perturbation_key].astype(str) == control_label
    n_ctrl = is_control.sum()
    print(f"Control cells: {n_ctrl}")

    # Convert Ensembl IDs to gene symbols if needed
    if any(str(v).startswith("ENSG") for v in adata.var_names[:5]):
        if "gene_name" in adata.var.columns:
            print("Converting Ensembl IDs to gene symbols...")
            adata.var["_ensembl_id"] = adata.var_names.copy()
            adata.var_names = adata.var["gene_name"].astype(str).values
            adata.var_names_make_unique()

    # Get valid perturbation genes
    pert_values = adata.obs[perturbation_key].value_counts()
    pert_genes = [g for g in pert_values.index
                  if g != control_label
                  and g.lower() not in ("ctrl", "control", "non-targeting", "nt")]
    gene_set = set(adata.var_names)
    valid_pert_genes = [g for g in pert_genes if g in gene_set]
    print(f"Valid perturbation genes: {len(valid_pert_genes)} / {len(pert_genes)}")

    # Select subset
    if args.gene_list and os.path.exists(args.gene_list):
        with open(args.gene_list) as f:
            requested = [line.strip() for line in f if line.strip()]
        selected_genes = [g for g in requested if g in set(valid_pert_genes)]
        print(f"From gene list: {len(selected_genes)}/{len(requested)} genes found")
    else:
        rng = np.random.default_rng(args.seed)
        if args.n_genes > 0 and args.n_genes < len(valid_pert_genes):
            selected_genes = sorted(rng.choice(valid_pert_genes, size=args.n_genes, replace=False))
        else:
            selected_genes = sorted(valid_pert_genes)
    print(f"Selected {len(selected_genes)} genes")

    # ---------------------------------------------------------------
    # 3. Prepare GEARS-format AnnData
    # ---------------------------------------------------------------
    # Subsample to fit in memory: cap control cells and pert cells per gene
    max_ctrl = 2000  # enough for training, prevents 10K+ control domination
    max_pert_per_gene = 100
    rng_sub = np.random.default_rng(args.seed)

    ctrl_idx = np.where(is_control)[0]
    if len(ctrl_idx) > max_ctrl:
        ctrl_idx = rng_sub.choice(ctrl_idx, size=max_ctrl, replace=False)

    pert_idx_list = []
    for gene in selected_genes:
        gene_mask = adata.obs[perturbation_key] == gene
        gene_idx = np.where(gene_mask)[0]
        if len(gene_idx) > max_pert_per_gene:
            gene_idx = rng_sub.choice(gene_idx, size=max_pert_per_gene, replace=False)
        pert_idx_list.append(gene_idx)

    all_idx = np.concatenate([ctrl_idx] + pert_idx_list)
    adata_sub = adata[all_idx].copy()
    print(f"Subsampled: {len(ctrl_idx)} ctrl + {sum(len(p) for p in pert_idx_list)} pert = {adata_sub.n_obs} cells")
    if not sp.issparse(adata_sub.X):
        adata_sub.X = sp.csr_matrix(adata_sub.X)

    conditions = []
    for val in adata_sub.obs[perturbation_key]:
        if val == control_label or val.lower() in ("ctrl", "control", "non-targeting", "nt"):
            conditions.append("ctrl")
        elif "+" in val:
            conditions.append(val)
        else:
            conditions.append(f"{val}+ctrl")
    adata_sub.obs["condition"] = conditions
    adata_sub.obs["cell_type"] = "K562"
    # GEARS requires condition_name: "{cell_type}_{condition}_{dose}"
    adata_sub.obs["condition_name"] = [
        f"K562_{c}_1+1" for c in conditions
    ]
    if "gene_name" not in adata_sub.var.columns:
        adata_sub.var["gene_name"] = adata_sub.var_names.tolist()

    print(f"GEARS AnnData: {adata_sub.n_obs} cells, "
          f"{adata_sub.obs['condition'].nunique()} conditions")

    # ---------------------------------------------------------------
    # 4. Monkey-patch GEARS for compatibility
    # ---------------------------------------------------------------
    import gears.data_utils as gears_du
    import gears.pertdata as gears_pd

    def _safe_rank(adata_rk, *a, **kw):
        # Skip slow Wilcoxon rank_genes_groups for large gene sets (>100 perturbations)
        # and use manual DE key population instead. GEARS uses these only for
        # training gene subset selection — the model still learns perturbation effects.
        n_conds = adata_rk.obs.get("condition", pd.Series()).nunique()
        if n_conds <= 60:
            try:
                import scanpy as _sc
                groupby = kw.get("groupby",
                                 "condition" if "condition" in adata_rk.obs.columns else None)
                if groupby is None:
                    raise ValueError("No groupby key")
                print(f"  Running rank_genes_groups ({n_conds} conditions)...")
                _sc.tl.rank_genes_groups(
                    adata_rk, groupby=groupby,
                    reference="ctrl" if "ctrl" in adata_rk.obs[groupby].values else "rest",
                    method="wilcoxon",
                )
            except Exception as exc:
                print(f"  Warning: rank_genes_groups failed ({exc}), populating DE keys manually")
        else:
            print(f"  Skipping rank_genes_groups ({n_conds} conditions — too slow), using manual DE keys")
        conditions_list = [c for c in adata_rk.obs.get("condition",
                           adata_rk.obs.iloc[:, 0]).unique() if c != "ctrl"]
        gene_names_rk = list(adata_rk.var_names)
        if "rank_genes_groups_cov_all" not in adata_rk.uns:
            adata_rk.uns["rank_genes_groups_cov_all"] = {
                "K562_" + cond + "_1+1": gene_names_rk for cond in conditions_list
            }
        if "top_non_dropout_de_20" not in adata_rk.uns:
            adata_rk.uns["top_non_dropout_de_20"] = {
                "K562_" + cond + "_1+1": gene_names_rk[:20] for cond in conditions_list
            }
        return adata_rk
    gears_du.rank_genes_groups_by_cov = _safe_rank

    def _fixed_get_dropout(adata_dd):
        unique_conds = adata_dd.obs["condition"].unique()
        cond2idx = {c: np.where(adata_dd.obs["condition"] == c)[0]
                    for c in unique_conds}
        cond2mean = {}
        for c, idx in cond2idx.items():
            Xc = adata_dd.X[idx]
            if sp.issparse(Xc):
                Xc = Xc.toarray()
            cond2mean[c] = np.asarray(Xc).mean(axis=0).flatten()
        ctrl_mean = cond2mean.get("ctrl", np.zeros(adata_dd.n_vars))
        gene_id2idx = {g: i for i, g in enumerate(adata_dd.var_names)}
        gene_idx2id = {i: g for g, i in gene_id2idx.items()}
        non_zeros_gene_idx = {}
        non_dropout_gene_idx = {}
        top_non_dropout_de_20 = {}
        top_non_zero_de_20 = {}
        de_keys = adata_dd.uns.get("rank_genes_groups_cov_all", {})
        for pert_key, top_genes in de_keys.items():
            cond = pert_key
            parts = pert_key.split("_", 1)
            if len(parts) == 2 and parts[0].upper() == "K562":
                rest = parts[1]
                for suffix in ("_1+1", "_1"):
                    if rest.endswith(suffix):
                        cond = rest[:-len(suffix)]
                        break
                else:
                    cond = rest
            if cond not in cond2mean and pert_key in cond2mean:
                cond = pert_key
            if cond not in cond2mean:
                continue
            X_mean = cond2mean[cond]
            non_zero = np.where(X_mean != 0)[0]
            zero = np.where(X_mean == 0)[0]
            true_zeros = np.intersect1d(zero, np.where(ctrl_mean == 0)[0])
            non_dropouts = np.concatenate((non_zero, true_zeros))
            gene_idx_top = [gene_id2idx[g] for g in top_genes if g in gene_id2idx]
            nd20 = [i for i in gene_idx_top if i in set(non_dropouts)][:20]
            nz20 = [i for i in gene_idx_top if i in set(non_zero)][:20]
            non_zeros_gene_idx[pert_key] = np.sort(non_zero)
            non_dropout_gene_idx[pert_key] = np.sort(non_dropouts)
            top_non_dropout_de_20[pert_key] = np.array([gene_idx2id[i] for i in nd20])
            top_non_zero_de_20[pert_key] = np.array([gene_idx2id[i] for i in nz20])
        adata_dd.uns["top_non_dropout_de_20"] = top_non_dropout_de_20
        adata_dd.uns["non_dropout_gene_idx"] = non_dropout_gene_idx
        adata_dd.uns["non_zeros_gene_idx"] = non_zeros_gene_idx
        adata_dd.uns["top_non_zero_de_20"] = top_non_zero_de_20
        return adata_dd
    gears_du.get_dropout_non_zero_genes = _fixed_get_dropout
    gears_pd.get_dropout_non_zero_genes = _fixed_get_dropout

    # Fix PyTorch 2.1+ in-place operation error in GEARS loss function
    import gears.utils as gears_utils
    def _fixed_loss_fct(pred, y, perts, ctrl=None, direction_lambda=1e-3, dict_filter=None):
        gamma = 2
        perts_arr = np.array(perts)
        losses = torch.zeros(1, device=pred.device)
        for p in set(perts_arr):
            pert_idx = np.where(perts_arr == p)[0]
            if p != 'ctrl':
                retain_idx = dict_filter[p]
                pred_p = pred[pert_idx][:, retain_idx]
                y_p = y[pert_idx][:, retain_idx]
            else:
                pred_p = pred[pert_idx]
                y_p = y[pert_idx]
            losses = losses + torch.sum((pred_p - y_p)**(2 + gamma)) / pred_p.shape[0] / pred_p.shape[1]
            if p != 'ctrl':
                losses = losses + torch.sum(direction_lambda * (torch.sign(y_p - ctrl[retain_idx]) - torch.sign(pred_p - ctrl[retain_idx]))**2) / pred_p.shape[0] / pred_p.shape[1]
            else:
                losses = losses + torch.sum(direction_lambda * (torch.sign(y_p - ctrl) - torch.sign(pred_p - ctrl))**2) / pred_p.shape[0] / pred_p.shape[1]
        return losses / (len(set(perts_arr)))
    gears_utils.loss_fct = _fixed_loss_fct
    # Also patch the module-level reference used in gears.py
    import gears.gears as gears_mod
    gears_mod.loss_fct = _fixed_loss_fct
    print("Monkey-patched GEARS loss_fct (fixed in-place += for PyTorch 2.1+)")

    # ---------------------------------------------------------------
    # 5. Process through GEARS pipeline
    # ---------------------------------------------------------------
    dataset_name = "k562_crispri"
    dataset_dir = os.path.join(args.cache_dir, dataset_name)

    pert_data = PertData(args.cache_dir)
    processed_h5ad = os.path.join(dataset_dir, "perturb_processed.h5ad")
    if os.path.exists(processed_h5ad):
        print(f"Reusing existing GEARS cache at {dataset_dir}")
    else:
        import shutil
        if os.path.exists(dataset_dir):
            print(f"Deleting stale GEARS cache at {dataset_dir}")
            shutil.rmtree(dataset_dir)
        print("Processing dataset through GEARS pipeline...")
        print(f"  {adata_sub.n_obs} cells, {adata_sub.n_vars} genes, "
              f"{adata_sub.obs['condition'].nunique()} conditions")
        t0 = time.time()
        pert_data.new_data_process(dataset_name=dataset_name, adata=adata_sub)
        print(f"  Data processing took {time.time() - t0:.1f}s")

    pert_data.load(data_path=dataset_dir)
    pert_data.prepare_split(split="simulation", seed=args.seed)
    pert_data.get_dataloader(batch_size=args.batch_size,
                             test_batch_size=args.batch_size * 4)

    gears_genes = list(pert_data.adata.var_names)
    print(f"GEARS gene vocab: {len(gears_genes)} genes")
    pert_list = getattr(pert_data, 'pert_list', None)
    if pert_list is None:
        # Build from condition column
        pert_list = [c for c in pert_data.adata.obs["condition"].unique()
                     if c != "ctrl"]
    print(f"Perturbation list: {len(pert_list)} perturbations")

    # ---------------------------------------------------------------
    # 6. Train GEARS
    # ---------------------------------------------------------------
    # GEARS uses torch internally; for MPS we need to handle device
    # GEARS only supports "cuda" or "cpu" in its device arg
    gears_device = "cuda" if device == "cuda" else "cpu"
    if device == "mps":
        print("NOTE: GEARS doesn't natively support MPS; using CPU")
        print("  (MPS would require patching GEARS internals)")

    gears_model = GEARS(pert_data, device=gears_device)
    gears_model.model_initialize(hidden_size=args.hidden_size)

    print(f"\nTraining GEARS for {args.epochs} epochs "
          f"(hidden={args.hidden_size}, lr={args.lr}, device={gears_device})...")
    t_train = time.time()
    gears_model.train(epochs=args.epochs, lr=args.lr)
    train_time = time.time() - t_train
    print(f"Training complete in {train_time:.1f}s ({train_time/60:.1f} min)")

    # Save model
    model_dir = os.path.join(args.cache_dir, "gears_model")
    os.makedirs(model_dir, exist_ok=True)
    gears_model.save_model(model_dir)
    print(f"Model saved to {model_dir}")

    # ---------------------------------------------------------------
    # 7. Predict perturbation effects
    # ---------------------------------------------------------------
    print(f"\nPredicting {len(selected_genes)} perturbations...")
    mean_predictions = []
    pert_labels = []
    skipped = []
    t_pred = time.time()

    for i, gene in enumerate(selected_genes):
        if i % 25 == 0:
            print(f"  {i}/{len(selected_genes)} ({time.time()-t_pred:.1f}s)")
        try:
            pred_dict = gears_model.predict([[gene]])
            if isinstance(pred_dict, tuple):
                pred_dict = pred_dict[0]
            if isinstance(pred_dict, dict):
                for key in [gene, f"{gene}+ctrl", f"{gene}_ctrl"]:
                    if key in pred_dict:
                        mean_expr = np.asarray(pred_dict[key],
                                               dtype=np.float32).flatten()
                        break
                else:
                    key = list(pred_dict.keys())[0]
                    mean_expr = np.asarray(pred_dict[key],
                                           dtype=np.float32).flatten()
            else:
                mean_expr = np.asarray(pred_dict, dtype=np.float32).flatten()

            mean_predictions.append(mean_expr)
            pert_labels.append(gene)
        except Exception as e:
            print(f"  WARNING: Failed on {gene}: {e}")
            skipped.append(gene)

    pred_time = time.time() - t_pred
    print(f"Prediction: {len(pert_labels)} OK, {len(skipped)} skipped "
          f"({pred_time:.1f}s)")

    if not mean_predictions:
        print("ERROR: No successful predictions!")
        sys.exit(1)

    # ---------------------------------------------------------------
    # 8. Save results
    # ---------------------------------------------------------------
    pred_matrix = np.stack(mean_predictions, axis=0)

    if pred_matrix.shape[1] == len(gears_genes):
        var_index = gears_genes
    else:
        var_index = [f"gene_{i}" for i in range(pred_matrix.shape[1])]

    result = ad.AnnData(
        X=pred_matrix.astype(np.float32),
        obs=pd.DataFrame({"perturbation": pert_labels}),
        var=pd.DataFrame(index=var_index),
        uns={
            "model": "gears",
            "output_type": "mean",
            "cell_type": "K562",
            "n_predicted": len(pert_labels),
            "n_skipped": len(skipped),
            "skipped_genes": skipped[:100],
            "epochs": args.epochs,
            "hidden_size": args.hidden_size,
            "seed": args.seed,
            "train_time_seconds": train_time,
            "prediction_time_seconds": pred_time,
            "device": gears_device,
        },
    )

    output_path = os.path.join(args.output_dir, "gears_predictions.h5ad")
    result.write_h5ad(output_path)
    print(f"\nSaved to {output_path}")
    print(f"Shape: {result.shape}")
    print(f"Total: train={train_time/60:.1f}min, pred={pred_time:.1f}s")


if __name__ == "__main__":
    main()
