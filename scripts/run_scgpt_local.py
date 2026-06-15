"""Run scGPT perturbation prediction locally on CPU.

Fine-tunes scGPT's TransformerGenerator on K562 CRISPRi data
using GEARS' data pipeline, and produces mean perturbation predictions.

Without a pretrained checkpoint, trains from random init.
This is acceptable because the benchmark tests whether the architecture
can learn perturbation effects from training data, not whether
pretraining helps (that's a separate question).

Usage:
    .venv_gears/bin/python3 scripts/run_scgpt_local.py --gene-list data/eval_genes_n50.txt
    .venv_gears/bin/python3 scripts/run_scgpt_local.py --n-genes 50 --epochs 10
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
    parser = argparse.ArgumentParser(description="Train scGPT locally on K562")
    parser.add_argument("--data-path", default="data/k562_real.h5ad")
    parser.add_argument("--output-dir", default="data/model_outputs_real")
    parser.add_argument("--cache-dir", default="data/scgpt_cache")
    parser.add_argument("--n-genes", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pool-size", type=int, default=100,
                        help="Control cells to pool per prediction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gene-list", default=None)
    parser.add_argument("--checkpoint", default="",
                        help="Path to pretrained scGPT checkpoint dir")
    args = parser.parse_args()

    device = "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    from gears import PertData
    from gears.utils import create_cell_graph_dataset_for_prediction
    from torch_geometric.loader import DataLoader

    from scgpt.model import TransformerGenerator
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.utils import map_raw_id_to_vocab_id
    from scgpt.loss import masked_mse_loss

    print("scGPT + GEARS imported")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load K562 data
    # ------------------------------------------------------------------
    if not os.path.exists(args.data_path):
        print(f"ERROR: {args.data_path} not found")
        sys.exit(1)

    print(f"Loading data from {args.data_path}...")
    adata = ad.read_h5ad(args.data_path)
    print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

    # Ensembl -> symbols
    if any(str(v).startswith("ENSG") for v in adata.var_names[:5]):
        if "gene_name" in adata.var.columns:
            print("Converting Ensembl IDs to gene symbols...")
            adata.var["_ensembl_id"] = adata.var_names.copy()
            adata.var_names = adata.var["gene_name"].astype(str).values
            adata.var_names_make_unique()

    # Find perturbation key and control label
    perturbation_key = None
    for key in ["gene", "perturbation", "guide_identity", "condition"]:
        if key in adata.obs.columns:
            perturbation_key = key
            break
    if perturbation_key is None:
        print(f"ERROR: No perturbation column found")
        sys.exit(1)

    unique_perts = adata.obs[perturbation_key].astype(str).unique()
    control_label = None
    for c in ["non-targeting", "ctrl", "control", "NT"]:
        if c in unique_perts:
            control_label = c
            break
    if control_label is None:
        print(f"ERROR: No control label found")
        sys.exit(1)

    is_control = adata.obs[perturbation_key].astype(str) == control_label
    print(f"Perturbation key: '{perturbation_key}', control: '{control_label}'")
    print(f"Control cells: {is_control.sum()}")

    # Select genes
    pert_values = adata.obs[perturbation_key].value_counts()
    pert_genes = [g for g in pert_values.index
                  if g != control_label
                  and g.lower() not in ("ctrl", "control", "non-targeting", "nt")]
    gene_set = set(adata.var_names)
    valid_pert_genes = [g for g in pert_genes if g in gene_set]
    print(f"Valid perturbation genes: {len(valid_pert_genes)}")

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
    print(f"Selected {len(selected_genes)} genes")

    # ------------------------------------------------------------------
    # 2. Prepare GEARS-format data
    # ------------------------------------------------------------------
    mask = is_control.copy()
    for gene in selected_genes:
        mask |= (adata.obs[perturbation_key].astype(str) == gene)
    adata_sub = adata[mask].copy()
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
    if "gene_name" not in adata_sub.var.columns:
        adata_sub.var["gene_name"] = adata_sub.var_names.tolist()

    print(f"GEARS AnnData: {adata_sub.n_obs} cells, "
          f"{adata_sub.obs['condition'].nunique()} conditions")

    # Monkey-patch GEARS
    import gears.data_utils as gears_du
    import gears.pertdata as gears_pd

    def _safe_rank(adata_rk, *a, **kw):
        try:
            import scanpy as _sc
            groupby = kw.get("groupby", "condition" if "condition" in adata_rk.obs.columns else None)
            if groupby is None:
                raise ValueError("No groupby key")
            _sc.tl.rank_genes_groups(
                adata_rk, groupby=groupby,
                reference="ctrl" if "ctrl" in adata_rk.obs[groupby].values else "rest",
                method="wilcoxon",
            )
        except Exception as exc:
            print(f"  Warning: rank_genes_groups failed ({exc})")
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
        cond2idx = {c: np.where(adata_dd.obs["condition"] == c)[0] for c in unique_conds}
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

    # Process through GEARS
    import shutil
    dataset_name = "k562_crispri"
    dataset_dir = os.path.join(args.cache_dir, dataset_name)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    pert_data = PertData(args.cache_dir)
    print("Processing through GEARS pipeline...")
    t0 = time.time()
    pert_data.new_data_process(dataset_name=dataset_name, adata=adata_sub)
    print(f"  Processing took {time.time() - t0:.1f}s")
    pert_data.load(data_path=dataset_dir)
    pert_data.prepare_split(split="simulation", seed=args.seed)
    pert_data.get_dataloader(batch_size=args.batch_size,
                             test_batch_size=args.batch_size)

    gene_list = pert_data.gene_names.values.tolist()
    n_genes_total = len(gene_list)
    print(f"GEARS gene list: {n_genes_total} genes")

    # ------------------------------------------------------------------
    # 3. Build vocabulary and model
    # ------------------------------------------------------------------
    # Load pretrained vocab or build from GEARS gene list
    special_tokens = ["<pad>", "<cls>", "<eoc>"]

    if args.checkpoint and os.path.exists(os.path.join(args.checkpoint, "vocab.json")):
        vocab = GeneVocab.from_file(os.path.join(args.checkpoint, "vocab.json"))
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        vocab.set_default_index(vocab["<pad>"])
        print(f"Loaded vocab from checkpoint: {len(vocab)} tokens")

        with open(os.path.join(args.checkpoint, "args.json")) as f:
            model_config = json.load(f)
    else:
        # Use scGPT's default gene vocab
        import scgpt
        default_vocab_path = os.path.join(
            os.path.dirname(scgpt.__file__), "tokenizer", "default_gene_vocab.json"
        )
        if os.path.exists(default_vocab_path):
            vocab = GeneVocab.from_file(default_vocab_path)
            for s in special_tokens:
                if s not in vocab:
                    vocab.append_token(s)
            vocab.set_default_index(vocab["<pad>"])
            print(f"Loaded default scGPT vocab: {len(vocab)} tokens")
        else:
            vocab = GeneVocab(gene_list_or_vocab=special_tokens + gene_list)
            vocab.set_default_index(vocab["<pad>"])
            print(f"Built vocab from gene list: {len(vocab)} tokens")

        model_config = {
            "embsize": 512,
            "nheads": 8,
            "nlayers": 12,
            "d_hid": 512,
            "n_layers_cls": 3,
        }

    embsize = model_config.get("embsize", 512)
    nhead = model_config.get("nheads", 8)
    nlayers = model_config.get("nlayers", 12)
    d_hid = model_config.get("d_hid", 512)
    n_layers_cls = model_config.get("n_layers_cls", 3)

    # Reduce model size for CPU training
    # Full scGPT has 12 layers x 512 emb — way too slow on CPU
    # Use 2 layers, 128 emb for CPU feasibility (~20 min)
    nlayers_use = min(nlayers, 2)
    embsize = min(embsize, 128)
    d_hid = min(d_hid, 128)
    nhead = min(nhead, 4)
    if nlayers_use < nlayers:
        print(f"Reducing for CPU: layers={nlayers_use}, emb={embsize}, hid={d_hid}, heads={nhead}")

    # Mark genes in vocab
    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    n_in_vocab = (pert_data.adata.var["id_in_vocab"] == 1).sum()
    print(f"Genes in vocab: {n_in_vocab}/{n_genes_total}")

    gene_ids = np.array(
        [vocab[g] if g in vocab else vocab["<pad>"] for g in gene_list],
        dtype=int,
    )

    # Control cells for prediction
    ctrl_mask = pert_data.adata.obs["condition"] == "ctrl"
    ctrl_adata = pert_data.adata[ctrl_mask].copy()
    print(f"Control cells: {ctrl_adata.n_obs}")

    # Initialize model
    ntokens = len(vocab)
    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers_use,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=0.0,
        pad_token="<pad>",
        pad_value=0,
        pert_pad_id=0,
        use_fast_transformer=False,
    )

    # Load pretrained weights if available
    if args.checkpoint:
        weights_file = os.path.join(args.checkpoint, "best_model.pt")
        if os.path.exists(weights_file):
            pretrained_dict = torch.load(weights_file, map_location="cpu")
            if pretrained_dict:
                model_dict = model.state_dict()
                loadable = {
                    k: v for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                model_dict.update(loadable)
                model.load_state_dict(model_dict)
                print(f"Loaded {len(loadable)}/{len(pretrained_dict)} pretrained params")
            else:
                print("Checkpoint empty, training from scratch")
    else:
        print("No checkpoint, training from random init")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params/1e6:.1f}M")

    # ------------------------------------------------------------------
    # 4. Fine-tune
    # ------------------------------------------------------------------
    max_seq_len = 512  # Reduced from 1536 for CPU feasibility

    print(f"\nTraining scGPT for {args.epochs} epochs (lr={args.lr}, device={device})...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    best_val_loss = float("inf")
    best_state = None
    t_train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_data in pert_data.dataloader["train_loader"]:
            bsz = len(batch_data.y)
            batch_data.to(device)

            x = batch_data.x
            ori_values = x[:, 0].view(bsz, n_genes_total)
            pert_flags = x[:, 1].long().view(bsz, n_genes_total)
            target_values = batch_data.y

            input_gene_ids = torch.arange(n_genes_total, device=device, dtype=torch.long)
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[:max_seq_len]

            input_values = ori_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_subset = target_values[:, input_gene_ids]

            mapped_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_ids = mapped_ids.repeat(bsz, 1)

            src_mask = torch.zeros_like(input_values, dtype=torch.bool, device=device)

            out = model(
                mapped_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_mask,
            )
            pred = out["mlm_output"]
            all_mask = torch.ones_like(input_values, dtype=torch.bool)
            loss = masked_mse_loss(pred, target_subset, all_mask)

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        # Validation
        val_loss_str = ""
        if "val_loader" in pert_data.dataloader:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch_data in pert_data.dataloader["val_loader"]:
                    bsz = len(batch_data.y)
                    batch_data.to(device)
                    x = batch_data.x
                    ori_values = x[:, 0].view(bsz, n_genes_total)
                    pert_flags = x[:, 1].long().view(bsz, n_genes_total)
                    target_values = batch_data.y

                    input_gene_ids = torch.arange(n_genes_total, device=device, dtype=torch.long)
                    mapped_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                    mapped_ids = mapped_ids.repeat(bsz, 1)
                    src_mask = torch.zeros_like(ori_values, dtype=torch.bool, device=device)

                    out = model(
                        mapped_ids, ori_values, pert_flags,
                        src_key_padding_mask=src_mask,
                    )
                    pred = out["mlm_output"]
                    all_mask = torch.ones_like(ori_values, dtype=torch.bool)
                    vloss = masked_mse_loss(pred, target_values, all_mask)
                    val_loss += vloss.item()
                    n_val += 1

            avg_val = val_loss / max(n_val, 1)
            val_loss_str = f", val_loss={avg_val:.4f}"

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t_train_start
        print(f"  Epoch {epoch}/{args.epochs}: loss={avg_loss:.4f}{val_loss_str} "
              f"({elapsed/60:.1f} min total)")

    train_time = time.time() - t_train_start
    print(f"\nTraining complete: {train_time:.1f}s ({train_time/60:.1f} min)")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val_loss={best_val_loss:.4f})")

    # ------------------------------------------------------------------
    # 5. Predict perturbation effects
    # ------------------------------------------------------------------
    model.eval()

    gene_set_gears = set(gene_list)
    predict_genes = [g for g in selected_genes if g in gene_set_gears]
    print(f"\nPredicting {len(predict_genes)} perturbations (pool_size={args.pool_size})...")

    mean_predictions = []
    pert_labels = []
    skipped = []
    t_pred_start = time.time()

    for i, gene in enumerate(predict_genes):
        if i % 10 == 0:
            elapsed = time.time() - t_pred_start
            print(f"  {i}/{len(predict_genes)} ({elapsed:.1f}s)")

        try:
            cell_graphs = create_cell_graph_dataset_for_prediction(
                [gene], ctrl_adata, gene_list, device,
                num_samples=args.pool_size,
            )
            loader = DataLoader(cell_graphs, batch_size=args.batch_size, shuffle=False)

            preds = []
            with torch.no_grad():
                for batch_data in loader:
                    batch_data.to(device)
                    pred_values = model.pred_perturb(
                        batch_data,
                        include_zero_gene="all",
                        gene_ids=gene_ids,
                        amp=False,  # No AMP on CPU
                    )
                    preds.append(pred_values.cpu())

            preds = torch.cat(preds, dim=0)
            mean_expr = preds.numpy().mean(axis=0).astype(np.float32)
            mean_predictions.append(mean_expr)
            pert_labels.append(gene)

        except Exception as e:
            print(f"  WARNING: Failed on {gene}: {e}")
            skipped.append(gene)

    pred_time = time.time() - t_pred_start
    print(f"Prediction: {len(pert_labels)} OK, {len(skipped)} skipped ({pred_time:.1f}s)")

    if not mean_predictions:
        print("ERROR: No successful predictions!")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    pred_matrix = np.stack(mean_predictions, axis=0)

    if pred_matrix.shape[1] == len(gene_list):
        var_index = gene_list
    else:
        var_index = [f"gene_{i}" for i in range(pred_matrix.shape[1])]

    result = ad.AnnData(
        X=pred_matrix.astype(np.float32),
        obs=pd.DataFrame({"perturbation": pert_labels}),
        var=pd.DataFrame(index=var_index),
        uns={
            "model": "scgpt",
            "output_type": "mean",
            "cell_type": "K562",
            "n_predicted": len(pert_labels),
            "n_skipped": len(skipped),
            "skipped_genes": skipped[:100],
            "epochs": args.epochs,
            "nlayers": nlayers_use,
            "embsize": embsize,
            "train_time_seconds": train_time,
            "prediction_time_seconds": pred_time,
            "seed": args.seed,
            "device": device,
        },
    )

    output_path = os.path.join(args.output_dir, "scgpt_predictions.h5ad")
    result.write_h5ad(output_path)
    print(f"\nSaved to {output_path}")
    print(f"Shape: {result.shape}")
    print(f"X mean: {pred_matrix.mean():.4f}, std: {pred_matrix.std():.4f}")
    print(f"Total: train={train_time/60:.1f}min, pred={pred_time:.1f}s")


if __name__ == "__main__":
    main()
