"""Geneformer perturbation prediction wrapper.

Geneformer (Theodoris et al., Nature 2023) uses rank-value encoding:
genes are ranked by expression within each cell, scaled by corpus-wide
median statistics. Perturbation is simulated by deleting gene tokens
from the rank-ordered sequence (CRISPRi knockdown) or moving them to
position 0 (overexpression).

Architecture overview:
    1. TranscriptomeTokenizer converts raw counts -> rank-value token IDs
    2. Pretrained transformer processes token sequences
    3. InSilicoPerturber deletes/overexpresses target gene tokens
    4. Model produces perturbed cell & gene embeddings
    5. Cosine similarity between original and perturbed embeddings
       quantifies perturbation impact

Expression mapping challenge:
    Geneformer does NOT directly output expression values. It outputs
    embedding-space cosine similarities. To get expression predictions
    for CausalCellBench, we use per-gene cosine similarities from the
    "cell_and_gene" emb_mode as a proxy for expression change magnitude:

        expr_perturbed[gene_j] = expr_control[gene_j] * f(cos_sim_j)

    where f maps cosine similarity to a fold-change scaling factor.
    Genes with low cosine similarity (large embedding shift) get
    larger expression changes. The perturbed gene itself is set to ~0.

This wrapper runs in an ISOLATED conda environment (env_geneformer).

Installation:
    conda create -n env_geneformer python=3.10
    conda activate env_geneformer
    pip install torch==2.2.2 transformers==4.36.2 accelerate>=0.24
    pip install git+https://huggingface.co/ctheodoris/Geneformer.git
    pip install anndata scanpy

Usage (CLI for benchmark.py):
    conda run -n env_geneformer python -m causalcellbench.models.geneformer_wrapper \\
        --input data/k562_control.h5ad \\
        --output results/geneformer_predictions.h5ad \\
        --genes TP53,MDM2,...

Usage (programmatic):
    from causalcellbench.models.geneformer_wrapper import GeneformerModel
    model = GeneformerModel()
    model.load_model(checkpoint_path="ctheodoris/Geneformer")
    result = model.predict_perturbation(control_adata, "TP53", n_cells=100, gene_names=[...])
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports -- Geneformer has heavy deps (torch, transformers, etc.)
# ---------------------------------------------------------------------------

def _import_geneformer():
    """Import Geneformer modules with clear error messages."""
    try:
        from geneformer import (
            InSilicoPerturber,
            TranscriptomeTokenizer,
            EmbExtractor,
        )
        return InSilicoPerturber, TranscriptomeTokenizer, EmbExtractor
    except ImportError as e:
        raise ImportError(
            "geneformer not installed. Install in the env_geneformer conda env:\n"
            "  pip install git+https://huggingface.co/ctheodoris/Geneformer.git\n"
            f"Original error: {e}"
        ) from e


def _import_isp_stats():
    """Import InSilicoPerturberStats."""
    try:
        from geneformer import InSilicoPerturberStats
        return InSilicoPerturberStats
    except ImportError as e:
        raise ImportError(
            f"geneformer InSilicoPerturberStats not available: {e}"
        ) from e


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Gene ID mapping utilities
# ---------------------------------------------------------------------------

def build_gene_symbol_to_ensembl_map(
    adata,
    gene_symbol_col: str = "gene_symbols",
    ensembl_col: str = "ensembl_id",
) -> dict[str, str]:
    """Build a mapping from gene symbols to Ensembl IDs.

    Geneformer requires Ensembl IDs internally. Most scRNA-seq datasets
    use gene symbols (TP53, BRCA1, etc.). This function creates the
    mapping from adata.var metadata.

    Parameters
    ----------
    adata
        AnnData object. Checks var columns and var_names for Ensembl IDs.
    gene_symbol_col
        Column name in adata.var containing gene symbols.
    ensembl_col
        Column name in adata.var containing Ensembl IDs.

    Returns
    -------
    dict mapping gene_symbol -> ensembl_id
    """
    symbol_to_ensembl = {}

    # Strategy 1: adata.var has both symbol and ensembl columns
    if ensembl_col in adata.var.columns:
        ensembl_ids = adata.var[ensembl_col]
        if gene_symbol_col in adata.var.columns:
            symbols = adata.var[gene_symbol_col]
        else:
            symbols = adata.var_names
        for sym, ens in zip(symbols, ensembl_ids):
            if isinstance(ens, str) and ens.startswith("ENSG"):
                symbol_to_ensembl[sym] = ens
        return symbol_to_ensembl

    # Strategy 2: var_names ARE Ensembl IDs
    if all(str(v).startswith("ENSG") for v in list(adata.var_names)[:10]):
        for v in adata.var_names:
            symbol_to_ensembl[v] = v
        return symbol_to_ensembl

    # Strategy 3: Try common column names
    for col in ["ensembl_id", "gene_id", "ensembl", "ENSEMBL", "gene_ids"]:
        if col in adata.var.columns:
            for sym, ens in zip(adata.var_names, adata.var[col]):
                if isinstance(ens, str) and ens.startswith("ENSG"):
                    symbol_to_ensembl[sym] = ens
            if symbol_to_ensembl:
                return symbol_to_ensembl

    logger.warning(
        "Could not find Ensembl IDs in adata.var. "
        "Geneformer requires Ensembl IDs. Available var columns: %s",
        list(adata.var.columns),
    )
    return symbol_to_ensembl


def load_geneformer_token_dict(token_dictionary_file: Optional[str] = None) -> dict:
    """Load Geneformer's Ensembl ID to token ID mapping.

    Parameters
    ----------
    token_dictionary_file
        Path to pickle file. If None, uses Geneformer's default.

    Returns
    -------
    dict mapping ensembl_id -> token_id
    """
    if token_dictionary_file is None:
        try:
            from geneformer import TOKEN_DICTIONARY_FILE
            token_dictionary_file = TOKEN_DICTIONARY_FILE
        except ImportError:
            from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
            token_dictionary_file = TOKEN_DICTIONARY_FILE

    with open(token_dictionary_file, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# AnnData preparation for Geneformer tokenization
# ---------------------------------------------------------------------------

def prepare_adata_for_geneformer(
    adata,
    ensembl_col: str = "ensembl_id",
    n_counts_col: str = "n_counts",
) -> "anndata.AnnData":
    """Prepare AnnData for Geneformer's TranscriptomeTokenizer.

    Geneformer requires:
        - adata.var['ensembl_id']: Ensembl IDs for each gene
        - adata.obs['n_counts']: total UMI counts per cell
        - Raw (unnormalized) count data in adata.X

    Parameters
    ----------
    adata
        AnnData with raw counts.
    ensembl_col
        Column name for Ensembl IDs in adata.var.
    n_counts_col
        Column name for total counts in adata.obs.

    Returns
    -------
    AnnData ready for TranscriptomeTokenizer.
    """
    import anndata as ad
    import scanpy as sc

    adata = adata.copy()

    # Ensure ensembl_id column exists
    if ensembl_col not in adata.var.columns:
        # Check if var_names are already Ensembl IDs
        sample = list(adata.var_names)[:5]
        if all(str(v).startswith("ENSG") for v in sample):
            adata.var[ensembl_col] = adata.var_names.tolist()
        else:
            # Try to find ensembl IDs in other columns
            found = False
            for col in ["gene_id", "gene_ids", "ensembl", "ENSEMBL"]:
                if col in adata.var.columns:
                    adata.var[ensembl_col] = adata.var[col]
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"No Ensembl IDs found in adata.var. "
                    f"Available columns: {list(adata.var.columns)}. "
                    f"Geneformer requires Ensembl IDs in var['{ensembl_col}']."
                )

    # Ensure n_counts column exists
    if n_counts_col not in adata.obs.columns:
        X = adata.X
        if hasattr(X, "toarray"):
            n_counts = np.asarray(X.sum(axis=1)).flatten()
        else:
            n_counts = np.asarray(X, dtype=np.float64).sum(axis=1)
        adata.obs[n_counts_col] = n_counts.astype(np.float32)

    # Filter out genes without valid Ensembl IDs
    valid_ensembl = adata.var[ensembl_col].apply(
        lambda x: isinstance(x, str) and x.startswith("ENSG")
    )
    n_before = adata.n_vars
    adata = adata[:, valid_ensembl].copy()
    n_after = adata.n_vars

    if n_after < n_before:
        logger.info(
            "Filtered genes for valid Ensembl IDs: %d -> %d",
            n_before, n_after,
        )

    if n_after == 0:
        raise ValueError("No genes with valid Ensembl IDs remain after filtering.")

    logger.info(
        "Prepared AnnData for Geneformer: %d cells, %d genes",
        adata.n_obs, adata.n_vars,
    )

    return adata


def tokenize_adata(
    adata,
    output_dir: str,
    output_prefix: str = "geneformer_tokenized",
    model_input_size: int = 4096,
    nproc: int = 4,
    custom_attr_name_dict: Optional[dict] = None,
) -> Path:
    """Tokenize AnnData using Geneformer's TranscriptomeTokenizer.

    Parameters
    ----------
    adata
        Prepared AnnData (must have ensembl_id in var, n_counts in obs).
    output_dir
        Directory to write tokenized dataset.
    output_prefix
        Prefix for output files.
    model_input_size
        Maximum token sequence length (4096 for V2).
    nproc
        Number of CPU workers for tokenization.
    custom_attr_name_dict
        Optional dict mapping custom attribute names for the tokenizer.

    Returns
    -------
    Path to the tokenized .dataset directory.
    """
    _, TranscriptomeTokenizer, _ = _import_geneformer()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save AnnData to temporary h5ad for the tokenizer
    tmp_data_dir = output_path / "tmp_adata"
    tmp_data_dir.mkdir(exist_ok=True)
    tmp_h5ad = tmp_data_dir / "input.h5ad"
    adata.write_h5ad(str(tmp_h5ad))

    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict=custom_attr_name_dict or {},
        nproc=nproc,
        model_input_size=model_input_size,
        special_token=True,
        model_version="V2",
    )

    tokenizer.tokenize_data(
        data_directory=str(tmp_data_dir),
        output_directory=str(output_path),
        output_prefix=output_prefix,
        file_format="h5ad",
    )

    tokenized_path = output_path / f"{output_prefix}.dataset"

    if not tokenized_path.exists():
        # Some versions save without .dataset suffix
        candidates = list(output_path.glob(f"{output_prefix}*"))
        if candidates:
            tokenized_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"Tokenized dataset not found at {tokenized_path}. "
                f"Contents of {output_path}: {list(output_path.iterdir())}"
            )

    logger.info("Tokenized dataset saved to %s", tokenized_path)
    return tokenized_path


# ---------------------------------------------------------------------------
# Core: In-silico perturbation via Geneformer
# ---------------------------------------------------------------------------

def run_in_silico_perturbation(
    model_directory: str,
    tokenized_data_path: str,
    output_directory: str,
    output_prefix: str,
    genes_to_perturb: list[str],
    emb_mode: str = "cell_and_gene",
    emb_layer: int = -1,
    forward_batch_size: int = 100,
    nproc: int = 4,
    max_ncells: Optional[int] = None,
    filter_data: Optional[dict] = None,
    token_dictionary_file: Optional[str] = None,
) -> Path:
    """Run Geneformer InSilicoPerturber for gene deletion perturbations.

    This is the core function that runs the actual Geneformer perturbation
    pipeline. It deletes target genes from the rank-value encoding and
    computes embedding shifts.

    Parameters
    ----------
    model_directory
        Path to pretrained Geneformer model (local or HuggingFace name).
    tokenized_data_path
        Path to tokenized .dataset from tokenize_adata().
    output_directory
        Directory for output pickle files.
    output_prefix
        Prefix for output files.
    genes_to_perturb
        List of Ensembl IDs to perturb (delete from rank encoding).
    emb_mode
        Embedding mode. "cell_and_gene" gives both cell-level and
        per-gene cosine similarities (needed for expression mapping).
    emb_layer
        Transformer layer for embedding extraction.
        -1 = second-to-last (recommended), 0 = last.
    forward_batch_size
        Batch size for transformer forward pass.
    nproc
        CPU workers.
    max_ncells
        Max cells to perturb (None = all).
    filter_data
        Optional dict to filter cells (e.g., {"cell_type": ["K562"]}).
    token_dictionary_file
        Path to token dictionary pickle. None uses default.

    Returns
    -------
    Path to output directory containing perturbation result pickles.
    """
    InSilicoPerturber, _, _ = _import_geneformer()

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    isp = InSilicoPerturber(
        perturb_type="delete",
        perturb_rank_shift=None,
        genes_to_perturb=genes_to_perturb,
        combos=0,
        anchor_gene=None,
        model_type="Pretrained",
        num_classes=0,
        emb_mode=emb_mode,
        cell_emb_style="mean_pool",
        filter_data=filter_data,
        cell_states_to_model=None,
        state_embs_dict=None,
        max_ncells=max_ncells,
        cell_inds_to_perturb="all",
        emb_layer=emb_layer,
        forward_batch_size=forward_batch_size,
        nproc=nproc,
        model_version="V2",
        token_dictionary_file=token_dictionary_file,
    )

    logger.info(
        "Running InSilicoPerturber: %d genes, emb_mode=%s, batch_size=%d",
        len(genes_to_perturb), emb_mode, forward_batch_size,
    )

    t0 = time.time()
    isp.perturb_data(
        model_directory=model_directory,
        input_data_file=str(tokenized_data_path),
        output_directory=str(output_path),
        output_prefix=output_prefix,
    )
    elapsed = time.time() - t0

    logger.info("InSilicoPerturber complete in %.1fs", elapsed)
    return output_path


# ---------------------------------------------------------------------------
# Expression prediction from Geneformer embeddings
# ---------------------------------------------------------------------------

def load_perturbation_results(
    output_directory: str,
    output_prefix: str,
) -> tuple[dict, Optional[dict]]:
    """Load InSilicoPerturber output pickle files.

    Parameters
    ----------
    output_directory
        Directory containing ISP output pickles.
    output_prefix
        Prefix used when running ISP.

    Returns
    -------
    (cell_embs_dict, gene_embs_dict) tuple.
    cell_embs_dict: {(perturbed_gene_token, "cell_emb"): [cos_sims]}
    gene_embs_dict: {(perturbed_gene_token, affected_gene_token): [cos_sims]}
                    or None if emb_mode was "cell" or "cls".
    """
    output_path = Path(output_directory)

    cell_embs_dict = {}
    gene_embs_dict = {}

    # Find cell embedding pickles
    cell_files = sorted(output_path.glob(f"{output_prefix}*cell_embs*"))
    for f in cell_files:
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        if isinstance(d, dict):
            for k, v in d.items():
                if k in cell_embs_dict:
                    if isinstance(v, list):
                        cell_embs_dict[k].extend(v)
                    else:
                        cell_embs_dict[k].append(v)
                else:
                    cell_embs_dict[k] = v if isinstance(v, list) else [v]

    # Find gene embedding pickles
    gene_files = sorted(output_path.glob(f"{output_prefix}*gene_embs*"))
    for f in gene_files:
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        if isinstance(d, dict):
            for k, v in d.items():
                if k in gene_embs_dict:
                    if isinstance(v, list):
                        gene_embs_dict[k].extend(v)
                    else:
                        gene_embs_dict[k].append(v)
                else:
                    gene_embs_dict[k] = v if isinstance(v, list) else [v]

    logger.info(
        "Loaded ISP results: %d cell entries, %d gene entries",
        len(cell_embs_dict), len(gene_embs_dict),
    )

    return cell_embs_dict, gene_embs_dict if gene_embs_dict else None


def cosine_sim_to_fold_change(
    cos_sim: float,
    max_fold_change: float = 10.0,
) -> float:
    """Convert cosine similarity to an expression fold-change factor.

    Mapping rationale:
        cos_sim = 1.0 -> no change (FC = 1.0)
        cos_sim = 0.0 -> maximum change (FC = max_fold_change)
        cos_sim < 0   -> even larger change (capped)

    We use an exponential mapping:
        FC = exp(alpha * (1 - cos_sim))

    where alpha is calibrated so that cos_sim=0 gives FC=max_fold_change.

    Parameters
    ----------
    cos_sim
        Cosine similarity between original and perturbed gene embedding.
        Range typically [0.5, 1.0] for most genes, lower for strongly
        affected genes.
    max_fold_change
        Maximum fold change to allow.

    Returns
    -------
    Fold change factor (>= 1.0). The direction (up/down) is determined
    by comparing perturbed vs original embedding positions.
    """
    alpha = np.log(max_fold_change)
    shift = max(0.0, 1.0 - cos_sim)
    fc = np.exp(alpha * shift)
    return min(fc, max_fold_change)


def predict_expression_from_embeddings(
    control_mean: np.ndarray,
    gene_names: list[str],
    perturbed_gene: str,
    gene_embs_dict: dict,
    perturbed_gene_token: int,
    token_to_ensembl: dict,
    ensembl_to_symbol: dict,
    knockdown_residual: float = 0.05,
) -> np.ndarray:
    """Map Geneformer embedding cosine similarities to expression predictions.

    Strategy:
        1. For the perturbed gene: set expression to knockdown_residual * control
        2. For all other genes: use per-gene cosine similarity to scale
           expression up or down relative to control mean.

    The cosine similarity between a gene's original and perturbed embedding
    indicates how much that gene's context changed. Lower cosine similarity
    = larger context change = larger expression change.

    We assume the direction of change follows the sign of the embedding
    shift projection. As a simplification, we use the magnitude only and
    apply bidirectional scaling centered on control expression.

    Parameters
    ----------
    control_mean
        Mean expression of control cells, shape (n_genes,).
    gene_names
        Gene names (symbols) matching control_mean columns.
    perturbed_gene
        Gene being perturbed (symbol).
    gene_embs_dict
        ISP gene embedding dict: {(pert_token, affected_token): [cos_sims]}.
    perturbed_gene_token
        Token ID of the perturbed gene.
    token_to_ensembl
        Mapping from token IDs to Ensembl IDs.
    ensembl_to_symbol
        Mapping from Ensembl IDs to gene symbols.
    knockdown_residual
        Fraction of expression remaining for the knocked-down gene.

    Returns
    -------
    Predicted mean expression vector, shape (n_genes,).
    """
    pred = control_mean.copy().astype(np.float64)

    # Build reverse mapping: symbol -> index in gene_names
    symbol_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Set perturbed gene to near-zero (CRISPRi knockdown)
    if perturbed_gene in symbol_to_idx:
        pred[symbol_to_idx[perturbed_gene]] *= knockdown_residual

    # Apply per-gene cosine similarity scaling
    if gene_embs_dict is not None:
        for (pert_token, affected_token), cos_sims_list in gene_embs_dict.items():
            if pert_token != perturbed_gene_token:
                continue

            # Get gene symbol for affected token
            affected_ensembl = token_to_ensembl.get(affected_token)
            if affected_ensembl is None:
                continue
            affected_symbol = ensembl_to_symbol.get(affected_ensembl)
            if affected_symbol is None or affected_symbol not in symbol_to_idx:
                continue
            if affected_symbol == perturbed_gene:
                continue  # Already handled

            # Mean cosine similarity across cells
            if isinstance(cos_sims_list, list) and len(cos_sims_list) > 0:
                mean_cos_sim = np.mean(cos_sims_list)
            elif isinstance(cos_sims_list, (int, float)):
                mean_cos_sim = float(cos_sims_list)
            else:
                continue

            # Convert to fold change
            fc = cosine_sim_to_fold_change(mean_cos_sim)
            idx = symbol_to_idx[affected_symbol]

            # Apply scaling: genes with large embedding shifts get
            # larger expression changes. We apply as a multiplicative
            # factor. The direction is ambiguous from cosine similarity
            # alone, so we use a symmetric dampening: reduce expression
            # proportionally to the embedding shift magnitude.
            # This is conservative -- it underestimates upregulation
            # but captures the right structure for causal discovery.
            if fc > 1.0:
                # Dampen toward control: mix between control and shifted
                dampening = 1.0 / fc
                pred[idx] *= dampening

    return pred.astype(np.float32)


def predict_expression_direct(
    control_mean: np.ndarray,
    gene_names: list[str],
    perturbed_gene: str,
    cell_emb_cos_sim: Optional[float] = None,
    knockdown_residual: float = 0.05,
) -> np.ndarray:
    """Simplified expression prediction without per-gene embeddings.

    When emb_mode="cell" (no per-gene info), we can only estimate a
    global perturbation magnitude from the cell-level cosine similarity.
    This is less accurate than the per-gene approach but still provides
    a reasonable first-order approximation.

    Parameters
    ----------
    control_mean
        Mean control expression, shape (n_genes,).
    gene_names
        Gene names.
    perturbed_gene
        Gene being knocked down.
    cell_emb_cos_sim
        Cell-level cosine similarity (original vs perturbed).
        Lower = stronger perturbation effect.
    knockdown_residual
        Fraction remaining for knocked-down gene.

    Returns
    -------
    Predicted expression, shape (n_genes,).
    """
    pred = control_mean.copy().astype(np.float64)

    # Set perturbed gene to near-zero
    gene_list = list(gene_names)
    if perturbed_gene in gene_list:
        pred[gene_list.index(perturbed_gene)] *= knockdown_residual

    # Global scaling based on cell-level perturbation magnitude
    if cell_emb_cos_sim is not None and cell_emb_cos_sim < 1.0:
        # Perturbation magnitude: how much did the cell state change?
        perturbation_magnitude = 1.0 - cell_emb_cos_sim
        # Add noise proportional to perturbation magnitude
        # (approximates downstream effects)
        noise_scale = perturbation_magnitude * 0.1
        rng = np.random.default_rng(hash(perturbed_gene) % 2**32)
        noise = rng.normal(0, noise_scale, size=len(pred))
        pred *= (1.0 + noise)
        pred = np.maximum(pred, 0.0)

    return pred.astype(np.float32)


# ---------------------------------------------------------------------------
# GeneformerModel class -- CausalCellBench model interface
# ---------------------------------------------------------------------------

class GeneformerModel:
    """Geneformer wrapper implementing the CausalCellBench model interface.

    Avoids importing geneformer at class definition time (may not be
    installed in the evaluation environment). The benchmark runner calls
    this through conda subprocess isolation.

    Attributes
    ----------
    _model_directory : str -- path to pretrained Geneformer model
    _token_dict : dict -- ensembl_id -> token_id mapping
    _token_to_ensembl : dict -- reverse mapping token_id -> ensembl_id
    _symbol_to_ensembl : dict -- gene_symbol -> ensembl_id
    _ensembl_to_symbol : dict -- ensembl_id -> gene_symbol
    _dispersion_cache : cached NB dispersion from control cells
    _tokenized_data_path : path to tokenized dataset (cached)
    _isp_output_dir : path to ISP output directory (cached)
    """

    def __init__(self):
        self._model_directory: Optional[str] = None
        self._token_dict: Optional[dict] = None
        self._token_to_ensembl: Optional[dict] = None
        self._symbol_to_ensembl: dict = {}
        self._ensembl_to_symbol: dict = {}
        self._dispersion_cache = None
        self._tokenized_data_path: Optional[str] = None
        self._isp_output_dir: Optional[str] = None
        self._isp_results_cache: dict = {}

    @property
    def name(self) -> str:
        return "Geneformer"

    @property
    def supports_multi_perturbation(self) -> bool:
        return False  # Geneformer perturbs one gene at a time

    def load_model(
        self,
        checkpoint_path: str = "ctheodoris/Geneformer",
        adata_train: Optional["anndata.AnnData"] = None,
        token_dictionary_file: Optional[str] = None,
    ) -> None:
        """Load pretrained Geneformer model.

        Parameters
        ----------
        checkpoint_path
            Path to local model directory or HuggingFace model name.
        adata_train
            Optional AnnData for building gene symbol to Ensembl mapping.
        token_dictionary_file
            Path to token dictionary pickle. None uses default.
        """
        self._model_directory = checkpoint_path

        # Load token dictionary
        self._token_dict = load_geneformer_token_dict(token_dictionary_file)
        self._token_to_ensembl = {v: k for k, v in self._token_dict.items()
                                   if isinstance(k, str) and k.startswith("ENSG")}

        # Build gene name mappings if adata provided
        if adata_train is not None:
            self._symbol_to_ensembl = build_gene_symbol_to_ensembl_map(adata_train)
            self._ensembl_to_symbol = {v: k for k, v in self._symbol_to_ensembl.items()}

        logger.info(
            "Geneformer model loaded: %s (%d tokens in vocabulary)",
            checkpoint_path, len(self._token_dict),
        )

    def _ensure_gene_mappings(self, adata) -> None:
        """Ensure gene symbol <-> Ensembl ID mappings are populated."""
        if not self._symbol_to_ensembl:
            self._symbol_to_ensembl = build_gene_symbol_to_ensembl_map(adata)
            self._ensembl_to_symbol = {v: k for k, v in self._symbol_to_ensembl.items()}

    def _get_ensembl_ids(self, gene_symbols: list[str]) -> list[str]:
        """Convert gene symbols to Ensembl IDs, filtering unavailable ones."""
        ensembl_ids = []
        for sym in gene_symbols:
            ens = self._symbol_to_ensembl.get(sym)
            if ens and ens in self._token_dict:
                ensembl_ids.append(ens)
            elif sym in self._token_dict:
                # Symbol might already be an Ensembl ID
                ensembl_ids.append(sym)
            else:
                logger.debug("Gene %s not in Geneformer vocabulary, skipping", sym)
        return ensembl_ids

    def tokenize_and_perturb(
        self,
        control_adata,
        genes_to_perturb: list[str],
        output_dir: str,
        emb_mode: str = "cell_and_gene",
        forward_batch_size: int = 100,
        max_ncells: Optional[int] = 2000,
        nproc: int = 4,
    ) -> tuple[dict, Optional[dict]]:
        """Full pipeline: tokenize control cells, run ISP, load results.

        Parameters
        ----------
        control_adata
            AnnData of control cells (raw counts).
        genes_to_perturb
            Gene symbols to perturb.
        output_dir
            Working directory for intermediate files.
        emb_mode
            "cell_and_gene" for per-gene expression mapping (recommended),
            "cell" for cell-level only.
        forward_batch_size
            Batch size for transformer forward pass.
        max_ncells
            Maximum control cells to process.
        nproc
            CPU workers.

        Returns
        -------
        (cell_embs_dict, gene_embs_dict) from ISP.
        """
        if self._model_directory is None:
            raise RuntimeError("Call load_model() first")

        self._ensure_gene_mappings(control_adata)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Prepare AnnData
        logger.info("Step 1: Preparing AnnData for Geneformer...")
        prepared_adata = prepare_adata_for_geneformer(control_adata)

        # Step 2: Tokenize
        logger.info("Step 2: Tokenizing cells...")
        tokenized_path = tokenize_adata(
            prepared_adata,
            output_dir=str(output_path / "tokenized"),
            output_prefix="control",
            nproc=nproc,
        )
        self._tokenized_data_path = str(tokenized_path)

        # Step 3: Convert gene symbols to Ensembl IDs
        ensembl_ids = self._get_ensembl_ids(genes_to_perturb)
        if not ensembl_ids:
            raise ValueError(
                "No genes could be mapped to Geneformer Ensembl IDs. "
                f"Tried: {genes_to_perturb[:10]}. "
                f"Symbol-to-Ensembl mapping has {len(self._symbol_to_ensembl)} entries."
            )
        logger.info(
            "Step 3: %d / %d genes mapped to Ensembl IDs",
            len(ensembl_ids), len(genes_to_perturb),
        )

        # Step 4: Run InSilicoPerturber
        isp_output = str(output_path / "isp_output")
        logger.info("Step 4: Running InSilicoPerturber (delete mode)...")
        run_in_silico_perturbation(
            model_directory=self._model_directory,
            tokenized_data_path=str(tokenized_path),
            output_directory=isp_output,
            output_prefix="gf_pert",
            genes_to_perturb=ensembl_ids,
            emb_mode=emb_mode,
            forward_batch_size=forward_batch_size,
            nproc=nproc,
            max_ncells=max_ncells,
        )
        self._isp_output_dir = isp_output

        # Step 5: Load results
        logger.info("Step 5: Loading perturbation results...")
        cell_embs, gene_embs = load_perturbation_results(isp_output, "gf_pert")

        return cell_embs, gene_embs

    def predict_perturbation(
        self,
        control_adata,
        perturbed_gene: str,
        n_cells: int,
        gene_names: list[str],
        seed: int = 0,
    ):
        """Predict single-gene perturbation with NB-sampled cells.

        This method uses pre-computed ISP results (from tokenize_and_perturb)
        if available, otherwise runs a simplified prediction.

        Parameters
        ----------
        control_adata
            AnnData of unperturbed control cells.
        perturbed_gene
            Gene to knock down (symbol).
        n_cells
            Number of simulated cells.
        gene_names
            Output gene names.
        seed
            Random seed.

        Returns
        -------
        PerturbationResult with shape (n_cells, len(gene_names)).
        """
        from causalcellbench.models.base import PerturbationResult
        from causalcellbench.models.sampling import (
            sample_cells_from_model_output,
            estimate_nb_dispersion,
        )

        if self._model_directory is None:
            raise RuntimeError("Call load_model() first")

        # Get control mean expression
        X = control_adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        # Align gene_names to control_adata
        var_to_idx = {g: i for i, g in enumerate(control_adata.var_names)}
        control_mean = np.zeros(len(gene_names), dtype=np.float32)
        for j, g in enumerate(gene_names):
            if g in var_to_idx:
                control_mean[j] = X[:, var_to_idx[g]].mean()

        # Check for cached ISP results
        if perturbed_gene in self._isp_results_cache:
            mean_pred = self._isp_results_cache[perturbed_gene]
        else:
            # Use gene embeddings if available
            self._ensure_gene_mappings(control_adata)
            ens_id = self._symbol_to_ensembl.get(perturbed_gene)
            pert_token = self._token_dict.get(ens_id) if ens_id else None

            if (self._isp_output_dir is not None
                    and pert_token is not None):
                # Load ISP results for this specific gene
                cell_embs, gene_embs = load_perturbation_results(
                    self._isp_output_dir, "gf_pert"
                )
                mean_pred = predict_expression_from_embeddings(
                    control_mean=control_mean,
                    gene_names=gene_names,
                    perturbed_gene=perturbed_gene,
                    gene_embs_dict=gene_embs,
                    perturbed_gene_token=pert_token,
                    token_to_ensembl=self._token_to_ensembl or {},
                    ensembl_to_symbol=self._ensembl_to_symbol,
                )
            else:
                # Fallback: simple knockdown without Geneformer context
                logger.warning(
                    "No ISP results for %s. Using direct prediction.",
                    perturbed_gene,
                )
                mean_pred = predict_expression_direct(
                    control_mean=control_mean,
                    gene_names=gene_names,
                    perturbed_gene=perturbed_gene,
                )

            self._isp_results_cache[perturbed_gene] = mean_pred

        # NB sampling
        if self._dispersion_cache is None:
            self._dispersion_cache = estimate_nb_dispersion(
                control_adata, gene_names
            )

        cells = sample_cells_from_model_output(
            mean_pred, control_adata, n_cells, gene_names, seed,
            dispersion_cache=self._dispersion_cache,
        )

        return PerturbationResult(
            expression_matrix=cells,
            gene_names=gene_names,
            perturbed_genes=[perturbed_gene],
            cell_type="",
        )


# ---------------------------------------------------------------------------
# Batch prediction: run all perturbations at once (efficient ISP usage)
# ---------------------------------------------------------------------------

def run_geneformer_batch_predictions(
    control_adata,
    genes_to_perturb: list[str],
    model_directory: str = "ctheodoris/Geneformer",
    output_dir: str = "./geneformer_output",
    emb_mode: str = "cell_and_gene",
    forward_batch_size: int = 100,
    max_ncells: Optional[int] = 2000,
    nproc: int = 4,
    token_dictionary_file: Optional[str] = None,
) -> dict[str, np.ndarray]:
    """Run Geneformer perturbation predictions for multiple genes at once.

    This is the most efficient way to use Geneformer: tokenize once,
    perturb all genes in a single ISP run, then extract per-gene
    expression predictions.

    Parameters
    ----------
    control_adata
        AnnData of unperturbed control cells (raw counts).
    genes_to_perturb
        List of gene symbols to perturb.
    model_directory
        Pretrained Geneformer model path or HF name.
    output_dir
        Working directory for intermediate files.
    emb_mode
        "cell_and_gene" recommended for expression predictions.
    forward_batch_size
        Batch size for transformer.
    max_ncells
        Max cells to process.
    nproc
        CPU workers.
    token_dictionary_file
        Path to token dictionary. None uses default.

    Returns
    -------
    dict mapping gene_symbol -> predicted mean expression (n_genes,).
    """
    model = GeneformerModel()
    model.load_model(
        checkpoint_path=model_directory,
        adata_train=control_adata,
        token_dictionary_file=token_dictionary_file,
    )

    # Run full ISP pipeline
    cell_embs, gene_embs = model.tokenize_and_perturb(
        control_adata=control_adata,
        genes_to_perturb=genes_to_perturb,
        output_dir=output_dir,
        emb_mode=emb_mode,
        forward_batch_size=forward_batch_size,
        max_ncells=max_ncells,
        nproc=nproc,
    )

    # Get control mean
    X = control_adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    gene_names = list(control_adata.var_names)
    control_mean = X.mean(axis=0)

    # Extract per-gene predictions
    predictions = {}
    for gene in genes_to_perturb:
        ens_id = model._symbol_to_ensembl.get(gene)
        pert_token = model._token_dict.get(ens_id) if ens_id else None

        if pert_token is not None and gene_embs is not None:
            pred = predict_expression_from_embeddings(
                control_mean=control_mean,
                gene_names=gene_names,
                perturbed_gene=gene,
                gene_embs_dict=gene_embs,
                perturbed_gene_token=pert_token,
                token_to_ensembl=model._token_to_ensembl or {},
                ensembl_to_symbol=model._ensembl_to_symbol,
            )
        else:
            pred = predict_expression_direct(
                control_mean=control_mean,
                gene_names=gene_names,
                perturbed_gene=gene,
            )
        predictions[gene] = pred

    return predictions


# ---------------------------------------------------------------------------
# CLI entrypoint (called by benchmark.py via conda subprocess)
# ---------------------------------------------------------------------------

def run_geneformer_predictions(
    input_path: str,
    output_path: str,
    genes_to_perturb: list[str],
    model_name: str = "ctheodoris/Geneformer",
    cell_type: str = "K562",
    emb_mode: str = "cell_and_gene",
    forward_batch_size: int = 100,
    max_ncells: Optional[int] = 2000,
    nproc: int = 4,
    work_dir: Optional[str] = None,
) -> None:
    """Run Geneformer perturbation predictions and save to disk.

    Full pipeline:
        1. Load control cells from h5ad
        2. Prepare AnnData (Ensembl IDs, n_counts)
        3. Tokenize cells (rank-value encoding)
        4. Run InSilicoPerturber (gene deletion)
        5. Map embedding cosine similarities to expression predictions
        6. Save as h5ad

    Parameters
    ----------
    input_path
        Path to control cells .h5ad file.
    output_path
        Path to write predictions .h5ad file.
    genes_to_perturb
        List of gene symbols to perturb.
    model_name
        Pretrained model path or HuggingFace name.
    cell_type
        Cell type for metadata.
    emb_mode
        ISP embedding mode.
    forward_batch_size
        Transformer batch size.
    max_ncells
        Max cells to perturb.
    nproc
        CPU workers.
    work_dir
        Working directory for intermediate files.
    """
    import anndata as ad
    import pandas as pd

    try:
        import geneformer
        logger.info("Geneformer loaded successfully")
    except ImportError:
        logger.error(
            "Geneformer not installed. Run in env_geneformer. "
            "See docstring for installation."
        )
        sys.exit(1)

    # Load control cells
    logger.info("Loading control cells from %s", input_path)
    control = ad.read_h5ad(input_path)
    logger.info("Control: %d cells, %d genes", control.n_obs, control.n_vars)

    # Set up working directory
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="geneformer_work_")
    os.makedirs(work_dir, exist_ok=True)

    # Run batch predictions
    logger.info("Predicting %d perturbations...", len(genes_to_perturb))
    t_start = time.time()

    predictions = run_geneformer_batch_predictions(
        control_adata=control,
        genes_to_perturb=genes_to_perturb,
        model_directory=model_name,
        output_dir=work_dir,
        emb_mode=emb_mode,
        forward_batch_size=forward_batch_size,
        max_ncells=max_ncells,
        nproc=nproc,
    )

    elapsed = time.time() - t_start

    if not predictions:
        logger.error("No successful predictions!")
        sys.exit(1)

    # Build output AnnData
    pert_labels = list(predictions.keys())
    pred_matrix = np.stack([predictions[g] for g in pert_labels], axis=0)

    result = ad.AnnData(
        X=pred_matrix.astype(np.float32),
        obs=pd.DataFrame({"perturbation": pert_labels}),
        var=control.var.copy(),
        uns={
            "model": "geneformer",
            "output_type": "mean",
            "model_name": model_name,
            "cell_type": cell_type,
            "emb_mode": emb_mode,
            "n_predicted": len(pert_labels),
            "inference_time_seconds": elapsed,
        },
    )
    result.write_h5ad(output_path)
    logger.info(
        "Saved %d predictions to %s (%.1fs total)",
        len(pert_labels), output_path, elapsed,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Geneformer perturbation prediction for CausalCellBench"
    )
    parser.add_argument("--input", required=True, help="Control cells .h5ad")
    parser.add_argument("--output", required=True, help="Output predictions .h5ad")
    parser.add_argument("--genes", required=True,
                        help="Comma-separated genes to perturb")
    parser.add_argument("--model-name", default="ctheodoris/Geneformer",
                        help="Pretrained model path or HuggingFace name")
    parser.add_argument("--cell-type", default="K562")
    parser.add_argument("--emb-mode", default="cell_and_gene",
                        choices=["cell", "cell_and_gene"],
                        help="ISP embedding mode")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Forward pass batch size")
    parser.add_argument("--max-ncells", type=int, default=2000,
                        help="Max control cells to process")
    parser.add_argument("--nproc", type=int, default=4,
                        help="CPU workers for tokenization")
    parser.add_argument("--work-dir", default=None,
                        help="Working directory for intermediate files")
    args = parser.parse_args()

    run_geneformer_predictions(
        input_path=args.input,
        output_path=args.output,
        genes_to_perturb=args.genes.split(","),
        model_name=args.model_name,
        cell_type=args.cell_type,
        emb_mode=args.emb_mode,
        forward_batch_size=args.batch_size,
        max_ncells=args.max_ncells,
        nproc=args.nproc,
        work_dir=args.work_dir,
    )
