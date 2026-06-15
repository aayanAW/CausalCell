"""Run Geneformer ISP locally on CPU.

Manual in-silico perturbation (ISP) using the pretrained Geneformer V2-104M
model. Instead of the full InSilicoPerturber class, this script does
ISP manually:
  1. Tokenize control cells (rank-value encoding)
  2. For each cell, get original CLS+gene embeddings
  3. For each perturbation gene, delete its token and re-embed
  4. Compute cosine similarity between original/perturbed embeddings
  5. Map similarity to expression fold-change predictions

Outputs mean expression predictions as .h5ad for the eval pipeline.

Usage:
    .venv_gf/bin/python3 scripts/run_geneformer_local.py --n-genes 50
    .venv_gf/bin/python3 scripts/run_geneformer_local.py --gene-list data/eval_genes_n50.txt
"""

import argparse
import os
import sys
import time
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pathlib import Path


def tokenize_cell(expression: np.ndarray, gene_ensembl_ids: list,
                  token_dict: dict, gene_median_dict: dict,
                  max_len: int = 4096) -> list:
    """Tokenize a single cell using Geneformer rank-value encoding.

    Genes are ranked by expression/median ratio (descending), then mapped
    to token IDs. Only genes present in both the token dictionary and
    median dictionary are included.
    """
    # Compute expression / median ratio for ranking
    ratios = []
    for i, ens_id in enumerate(gene_ensembl_ids):
        expr = expression[i]
        if expr <= 0:
            continue
        median = gene_median_dict.get(ens_id, None)
        if median is None or median <= 0:
            continue
        token_id = token_dict.get(ens_id, None)
        if token_id is None:
            continue
        ratios.append((token_id, expr / median))

    # Sort by ratio descending (most expressed genes first)
    ratios.sort(key=lambda x: x[1], reverse=True)

    # Take top max_len tokens
    tokens = [r[0] for r in ratios[:max_len]]
    return tokens


def get_embeddings(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Get CLS and gene embeddings from model."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
    # Use last hidden state
    hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
    cls_emb = hidden[:, 0, :]  # CLS token embedding
    gene_embs = hidden[:, 1:, :]  # Gene token embeddings
    return cls_emb, gene_embs


def main():
    parser = argparse.ArgumentParser(description="Run Geneformer ISP locally")
    parser.add_argument("--data-path", default="data/k562_real.h5ad")
    parser.add_argument("--output-dir", default="data/model_outputs_real")
    parser.add_argument("--model-path", default="/tmp/geneformer_repo/Geneformer-V2-104M")
    parser.add_argument("--n-genes", type=int, default=50)
    parser.add_argument("--max-ncells", type=int, default=20,
                        help="Number of control cells to use (fewer = faster)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gene-list", default=None)
    parser.add_argument("--max-token-len", type=int, default=2048,
                        help="Max tokens per cell (lower = faster)")
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    device = torch.device("cpu")
    print("Running on CPU")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dictionaries
    # ------------------------------------------------------------------
    from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
    print(f"Loading token dictionary from {TOKEN_DICTIONARY_FILE}...")
    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        token_dict = pickle.load(f)

    # Load gene median dictionary
    token_dict_path = Path(TOKEN_DICTIONARY_FILE)
    median_dict_path = token_dict_path.parent / "gene_median_dictionary_gc104M.pkl"
    with open(median_dict_path, "rb") as f:
        gene_median_dict = pickle.load(f)

    # Load gene name to ID mapping
    gene_name_dict_path = token_dict_path.parent / "gene_name_id_dict_gc104M.pkl"
    with open(gene_name_dict_path, "rb") as f:
        gene_name_id_dict = pickle.load(f)

    # Build reverse mappings
    token_to_ensembl = {v: k for k, v in token_dict.items()
                        if isinstance(k, str) and k.startswith("ENSG")}
    # gene_name_id_dict: {gene_name: ensembl_id}
    symbol_to_ensembl = {}
    for name, ens_id in gene_name_id_dict.items():
        if isinstance(ens_id, str) and ens_id.startswith("ENSG"):
            symbol_to_ensembl[name] = ens_id
    ensembl_to_symbol = {v: k for k, v in symbol_to_ensembl.items()}

    ensembl_count = sum(1 for k in token_dict if isinstance(k, str) and k.startswith("ENSG"))
    print(f"Token dict: {len(token_dict)} entries, {ensembl_count} Ensembl IDs")
    print(f"Gene median dict: {len(gene_median_dict)} entries")
    print(f"Symbol-to-Ensembl: {len(symbol_to_ensembl)} mappings")

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    if not os.path.exists(args.data_path):
        print(f"ERROR: {args.data_path} not found")
        sys.exit(1)

    print(f"Loading data from {args.data_path}...")
    adata = ad.read_h5ad(args.data_path)
    print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

    # Map var_names to Ensembl IDs
    if any(str(v).startswith("ENSG") for v in adata.var_names[:5]):
        # var_names are already Ensembl IDs
        gene_ensembl_ids = list(adata.var_names)
        if "gene_name" in adata.var.columns:
            gene_symbols = list(adata.var["gene_name"].astype(str))
        else:
            gene_symbols = [ensembl_to_symbol.get(e, e) for e in gene_ensembl_ids]
    else:
        # var_names are symbols, need to map to Ensembl
        gene_symbols = list(adata.var_names)
        gene_ensembl_ids = [symbol_to_ensembl.get(s, "") for s in gene_symbols]

    n_mapped = sum(1 for e in gene_ensembl_ids if e.startswith("ENSG"))
    print(f"Genes mapped to Ensembl: {n_mapped} / {adata.n_vars}")

    # Find perturbation column
    perturbation_key = None
    for key in ["gene", "perturbation", "guide_identity", "condition"]:
        if key in adata.obs.columns:
            perturbation_key = key
            break
    if perturbation_key is None:
        print(f"ERROR: No perturbation column found")
        sys.exit(1)

    # Find control label
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
    n_ctrl = is_control.sum()
    pert_names = sorted([p for p in unique_perts if p != control_label])
    print(f"Control: {n_ctrl} cells, Perturbations: {len(pert_names)}")

    # ------------------------------------------------------------------
    # 3. Select genes
    # ------------------------------------------------------------------
    gene_set = set(gene_symbols) if gene_symbols[0] != gene_ensembl_ids[0] else set(gene_symbols)
    valid_pert_genes = [g for g in pert_names if g in gene_set]

    # Also check if perturbation gene has Ensembl ID in token dict
    valid_pert_genes_tokenizable = []
    for g in valid_pert_genes:
        ens = symbol_to_ensembl.get(g)
        if ens and ens in token_dict:
            valid_pert_genes_tokenizable.append(g)
    print(f"Perturbation genes in Geneformer vocabulary: {len(valid_pert_genes_tokenizable)}")

    if args.gene_list and os.path.exists(args.gene_list):
        with open(args.gene_list) as f:
            requested = [line.strip() for line in f if line.strip()]
        selected_genes = [g for g in requested if g in set(valid_pert_genes_tokenizable)]
        print(f"From gene list: {len(selected_genes)}/{len(requested)} found and tokenizable")
    else:
        rng_sel = np.random.default_rng(args.seed)
        if args.n_genes > 0 and args.n_genes < len(valid_pert_genes_tokenizable):
            selected_genes = sorted(
                rng_sel.choice(valid_pert_genes_tokenizable, size=args.n_genes, replace=False)
            )
        else:
            selected_genes = sorted(valid_pert_genes_tokenizable)
    print(f"Selected {len(selected_genes)} genes for ISP")

    # Map selected genes to Ensembl IDs and token IDs
    gene_to_ensembl = {g: symbol_to_ensembl[g] for g in selected_genes}
    gene_to_token = {g: token_dict[symbol_to_ensembl[g]] for g in selected_genes}

    # ------------------------------------------------------------------
    # 4. Prepare control cells
    # ------------------------------------------------------------------
    control_adata = adata[is_control].copy()
    # Use counts layer if available
    if "counts" in control_adata.layers:
        X_ctrl = control_adata.layers["counts"]
    elif "raw_counts" in control_adata.layers:
        X_ctrl = control_adata.layers["raw_counts"]
    else:
        X_ctrl = control_adata.X

    if sp.issparse(X_ctrl):
        X_ctrl = np.asarray(X_ctrl.todense())
    else:
        X_ctrl = np.asarray(X_ctrl)

    # Subsample control cells
    rng = np.random.default_rng(args.seed)
    if X_ctrl.shape[0] > args.max_ncells:
        idx = rng.choice(X_ctrl.shape[0], size=args.max_ncells, replace=False)
        X_ctrl = X_ctrl[idx]
    n_cells = X_ctrl.shape[0]
    print(f"Using {n_cells} control cells")

    # Compute control mean expression (full adata, for output)
    X_all_ctrl = adata[is_control].X
    if sp.issparse(X_all_ctrl):
        control_mean = np.asarray(X_all_ctrl.mean(axis=0)).flatten()
    else:
        control_mean = X_all_ctrl.mean(axis=0).flatten()

    # ------------------------------------------------------------------
    # 5. Tokenize control cells
    # ------------------------------------------------------------------
    print("\nTokenizing control cells...")
    t_tok_start = time.time()
    cell_tokens = []
    for i in range(n_cells):
        tokens = tokenize_cell(
            X_ctrl[i], gene_ensembl_ids, token_dict, gene_median_dict,
            max_len=args.max_token_len
        )
        cell_tokens.append(tokens)

    avg_len = np.mean([len(t) for t in cell_tokens])
    print(f"Tokenized {n_cells} cells, avg length: {avg_len:.0f} tokens "
          f"({time.time() - t_tok_start:.1f}s)")

    # ------------------------------------------------------------------
    # 6. Load model
    # ------------------------------------------------------------------
    print(f"\nLoading Geneformer model from {args.model_path}...")
    from transformers import BertForMaskedLM, BertConfig

    config = BertConfig.from_pretrained(args.model_path)
    model = BertForMaskedLM.from_pretrained(args.model_path, config=config)
    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params / 1e6:.1f}M parameters")

    # ------------------------------------------------------------------
    # 7. Get original embeddings for all cells
    # ------------------------------------------------------------------
    print("\nComputing original embeddings...")
    pad_token_id = config.pad_token_id if config.pad_token_id is not None else 0

    # Find max sequence length (no CLS token — raw gene tokens only)
    # Cap at max_position_embeddings - 1 to avoid off-by-one edge cases
    max_seq_len = min(
        max(len(t) for t in cell_tokens),
        config.max_position_embeddings - 1
    )
    print(f"Max sequence length: {max_seq_len}")

    # Build token-to-position mapping for each cell
    cell_token_positions = []  # list of dicts: {token_id: position_in_sequence}
    for tokens in cell_tokens:
        pos_map = {}
        for pos, tid in enumerate(tokens[:max_seq_len]):
            pos_map[tid] = pos  # 0-indexed, no CLS offset
        cell_token_positions.append(pos_map)

    # Original forward passes
    t_emb_start = time.time()
    original_gene_embs = []  # List of (n_tokens, hidden) per cell

    for i in range(n_cells):
        tokens = cell_tokens[i][:max_seq_len]
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=True)
        hidden = outputs.last_hidden_state[0]  # (seq_len, hidden)
        original_gene_embs.append(hidden.numpy())

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t_emb_start
            print(f"  {i+1}/{n_cells} cells ({elapsed:.1f}s)")

    t_emb = time.time() - t_emb_start
    print(f"Original embeddings: {t_emb:.1f}s ({t_emb / n_cells:.2f}s/cell)")

    # ------------------------------------------------------------------
    # 8. Perturb and compute cosine similarities
    # ------------------------------------------------------------------
    print(f"\nRunning ISP for {len(selected_genes)} genes...")
    t_isp_start = time.time()

    mean_predictions = []
    pert_labels = []
    failed = []

    for gi, gene_sym in enumerate(selected_genes):
        gene_token = gene_to_token[gene_sym]
        gene_ens = gene_to_ensembl[gene_sym]

        if gi % 5 == 0:
            elapsed = time.time() - t_isp_start
            print(f"  {gi}/{len(selected_genes)} genes ({elapsed:.1f}s)")

        # Collect per-gene cosine similarities across cells
        # For each affected gene token, track cos_sim across cells
        gene_cos_sims = {}  # {affected_token_id: [cos_sims]}

        n_cells_with_gene = 0
        for ci in range(n_cells):
            tokens = cell_tokens[ci][:max_seq_len]
            pos_map = cell_token_positions[ci]

            if gene_token not in pos_map:
                continue  # Gene not expressed in this cell
            n_cells_with_gene += 1

            # Remove the perturbation gene's token (pos_map is 0-indexed)
            gene_pos = pos_map[gene_token]
            pert_tokens = [t for j, t in enumerate(tokens) if j != gene_pos]

            if len(pert_tokens) == 0:
                continue

            # Forward pass with perturbed sequence
            input_ids = torch.tensor([pert_tokens], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask,
                                     output_hidden_states=True)
            pert_hidden = outputs.last_hidden_state[0].numpy()  # (seq_len-1, hidden)

            # Compute cosine similarity for each gene token present in both
            orig_hidden = original_gene_embs[ci]  # (seq_len, hidden)

            for tid, orig_pos in pos_map.items():
                if tid == gene_token:
                    continue  # Skip the perturbed gene itself

                # Find this token in the perturbed sequence
                # After removing gene_token, positions shift
                pert_tokens_list = pert_tokens
                try:
                    pert_pos = pert_tokens_list.index(tid)
                except ValueError:
                    continue  # Token not in perturbed sequence

                # Cosine similarity between original and perturbed embedding
                if orig_pos >= orig_hidden.shape[0] or pert_pos >= pert_hidden.shape[0]:
                    continue
                orig_emb = orig_hidden[orig_pos]
                pert_emb = pert_hidden[pert_pos]

                cos_sim = (np.dot(orig_emb, pert_emb) /
                           (np.linalg.norm(orig_emb) * np.linalg.norm(pert_emb) + 1e-8))

                if tid not in gene_cos_sims:
                    gene_cos_sims[tid] = []
                gene_cos_sims[tid].append(cos_sim)

        if n_cells_with_gene == 0:
            failed.append(gene_sym)
            continue

        # Map cosine similarities to expression predictions
        pred = control_mean.copy().astype(np.float64)

        # Set perturbed gene to near-zero (knockdown)
        if gene_sym in gene_symbols:
            gene_idx_in_adata = gene_symbols.index(gene_sym)
            pred[gene_idx_in_adata] *= 0.05

        # Apply per-gene effects based on cosine similarity
        for affected_tid, cos_sims in gene_cos_sims.items():
            affected_ens = token_to_ensembl.get(affected_tid)
            if affected_ens is None:
                continue
            affected_sym = ensembl_to_symbol.get(affected_ens)
            if affected_sym is None or affected_sym not in gene_symbols:
                continue
            if affected_sym == gene_sym:
                continue

            mean_cos = np.mean(cos_sims)
            # Map cosine similarity to expression scaling
            # Lower similarity = more disruption = more effect
            shift = max(0.0, 1.0 - mean_cos)
            alpha = np.log(10.0)
            fc = min(np.exp(alpha * shift), 10.0)
            if fc > 1.0:
                idx = gene_symbols.index(affected_sym)
                pred[idx] *= (1.0 / fc)

        mean_predictions.append(pred.astype(np.float32))
        pert_labels.append(gene_sym)

    t_isp = time.time() - t_isp_start
    print(f"\nISP complete: {len(pert_labels)} OK, {len(failed)} failed ({t_isp:.1f}s)")

    if not mean_predictions:
        print("ERROR: No successful predictions!")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    pred_matrix = np.stack(mean_predictions, axis=0)

    # Use gene symbols as var_names (eval pipeline expects symbols, not Ensembl IDs)
    var_df = adata.var[[]].copy() if pred_matrix.shape[1] == adata.n_vars else pd.DataFrame(
        index=[f"gene_{i}" for i in range(pred_matrix.shape[1])]
    )
    if any(str(v).startswith("ENSG") for v in var_df.index[:5]):
        if "gene_name" in adata.var.columns:
            gene_names = adata.var["gene_name"].astype(str).values.copy()
            gene_names = np.array([x if x else "unknown" for x in gene_names])
            # Make duplicate names unique by appending suffix (don't drop them)
            seen = {}
            for i, name in enumerate(gene_names):
                if name in seen:
                    seen[name] += 1
                    gene_names[i] = f"{name}-{seen[name]}"
                else:
                    seen[name] = 0
            var_df = pd.DataFrame(index=gene_names)

    result = ad.AnnData(
        X=pred_matrix,
        obs=pd.DataFrame({"perturbation": pert_labels}),
        var=var_df,
        uns={
            "model": "geneformer",
            "output_type": "mean",
            "model_name": "Geneformer-V2-104M",
            "cell_type": "K562",
            "n_cells_used": n_cells,
            "n_predicted": len(pert_labels),
            "n_failed": len(failed),
            "failed_genes": failed[:50],
            "isp_time_seconds": t_isp,
            "max_token_len": args.max_token_len,
            "seed": args.seed,
        },
    )

    out_path = os.path.join(args.output_dir, "geneformer_predictions.h5ad")
    result.write_h5ad(out_path)
    print(f"\nSaved to {out_path}")
    print(f"  Shape: {result.shape}")
    print(f"  X mean: {pred_matrix.mean():.4f}, std: {pred_matrix.std():.4f}")

    total_time = t_isp + t_emb
    print(f"Total time: emb={t_emb:.0f}s, ISP={t_isp:.0f}s, total={total_time/60:.1f}min")


if __name__ == "__main__":
    main()
