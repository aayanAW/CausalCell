"""scGPT perturbation prediction wrapper.

scGPT (Cui et al., Nature Methods 2024) is a transformer-based foundation
model pretrained on 33M+ single cells. For perturbation prediction, it uses
a TransformerGenerator architecture that:
  1. Encodes control cell gene expression via gene token + value embeddings
  2. Incorporates perturbation flags indicating which gene(s) are perturbed
  3. Predicts post-perturbation expression via masked language modeling

The perturbation pipeline reuses GEARS' PertData infrastructure for data
loading and cell graph construction, but replaces the GNN backbone with
scGPT's pretrained transformer encoder.

Key checkpoint files (in model_dir):
  - vocab.json   : GeneVocab mapping gene names -> token IDs
  - args.json    : Model architecture config (embsize, nheads, nlayers, etc.)
  - best_model.pt: Trained model weights

This wrapper is designed to run in an ISOLATED conda environment
(env_scgpt) due to flash-attn + PyTorch version conflicts.

Installation (in isolated env):
    conda create -n env_scgpt python=3.10
    conda activate env_scgpt
    pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu122
    pip install flash-attn==2.5.0 --no-build-isolation
    pip install scgpt==0.2.1
    pip install cell-gears anndata scanpy

Usage as standalone script:
    conda run -n env_scgpt python -m causalcellbench.models.scgpt_wrapper \\
        --input data/k562_control.h5ad \\
        --output results/scgpt_predictions.h5ad \\
        --genes TP53,MDM2,... \\
        --checkpoint /path/to/scgpt/checkpoint
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# scGPT special tokens
PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"
EOC_TOKEN = "<eoc>"
SPECIAL_TOKENS = [PAD_TOKEN, CLS_TOKEN, EOC_TOKEN]

# Default model hyperparameters (whole-human pretrained model)
DEFAULT_EMBSIZE = 512
DEFAULT_NHEADS = 8
DEFAULT_NLAYERS = 12
DEFAULT_D_HID = 512
DEFAULT_N_LAYERS_CLS = 3

# Perturbation prediction defaults
DEFAULT_INCLUDE_ZERO_GENE = "all"
DEFAULT_MAX_SEQ_LEN = 1536
DEFAULT_POOL_SIZE = 300  # number of control cells to sample for prediction

# Prefixes of pretrained params to load for fine-tuning
# (decoder is re-initialized for perturbation task)
FINETUNE_PARAM_PREFIXES = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _import_scgpt():
    """Import scGPT with clear error messages."""
    try:
        from scgpt.model import TransformerGenerator
        from scgpt.tokenizer.gene_tokenizer import GeneVocab
        from scgpt.utils import map_raw_id_to_vocab_id
        return TransformerGenerator, GeneVocab, map_raw_id_to_vocab_id
    except ImportError as e:
        raise ImportError(
            "scGPT not installed. Install in the env_scgpt conda env:\n"
            "  pip install scgpt==0.2.1 torch==2.1.2\n"
            f"Original error: {e}"
        ) from e


def _import_gears():
    """Import GEARS utilities for cell graph construction."""
    try:
        from gears import PertData
        from gears.utils import create_cell_graph_dataset_for_prediction
        return PertData, create_cell_graph_dataset_for_prediction
    except ImportError as e:
        raise ImportError(
            "cell-gears not installed. Required for scGPT perturbation pipeline:\n"
            "  pip install cell-gears\n"
            f"Original error: {e}"
        ) from e


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_scgpt_checkpoint(
    checkpoint_dir: str,
    device: str = "cuda",
    use_fast_transformer: bool = True,
    load_param_prefixes: Optional[List[str]] = None,
) -> tuple:
    """Load a pretrained scGPT checkpoint for perturbation prediction.

    Parameters
    ----------
    checkpoint_dir
        Directory containing vocab.json, args.json, best_model.pt.
    device
        Compute device ('cuda' or 'cpu').
    use_fast_transformer
        Whether to use flash attention. Set False if flash-attn not installed.
    load_param_prefixes
        If provided, only load parameters matching these prefixes.
        Used when fine-tuning: load encoder weights but re-init decoder.
        If None, load all parameters.

    Returns
    -------
    (model, vocab, model_config, gene_ids)
        model: TransformerGenerator on device
        vocab: GeneVocab instance
        model_config: dict from args.json
        gene_ids: None (set later when GEARS data is available)
    """
    import torch
    TransformerGenerator, GeneVocab, _ = _import_scgpt()

    model_dir = Path(checkpoint_dir)
    vocab_file = model_dir / "vocab.json"
    config_file = model_dir / "args.json"
    weights_file = model_dir / "best_model.pt"

    # Validate checkpoint files exist
    for f, name in [(vocab_file, "vocab.json"), (config_file, "args.json"),
                    (weights_file, "best_model.pt")]:
        if not f.exists():
            raise FileNotFoundError(
                f"Missing {name} in checkpoint directory: {model_dir}"
            )

    # Load vocabulary
    vocab = GeneVocab.from_file(vocab_file)
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab[PAD_TOKEN])
    logger.info("Loaded vocabulary: %d tokens", len(vocab))

    # Load model config
    with open(config_file, "r") as f:
        model_config = json.load(f)

    embsize = model_config.get("embsize", DEFAULT_EMBSIZE)
    nhead = model_config.get("nheads", DEFAULT_NHEADS)
    nlayers = model_config.get("nlayers", DEFAULT_NLAYERS)
    d_hid = model_config.get("d_hid", DEFAULT_D_HID)
    n_layers_cls = model_config.get("n_layers_cls", DEFAULT_N_LAYERS_CLS)

    logger.info(
        "Model config: embsize=%d, nhead=%d, nlayers=%d, d_hid=%d",
        embsize, nhead, nlayers, d_hid,
    )

    # Initialize model
    ntokens = len(vocab)
    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=0.0,  # no dropout at inference
        pad_token=PAD_TOKEN,
        pad_value=0,
        pert_pad_id=0,
        use_fast_transformer=use_fast_transformer,
    )

    # Load weights
    pretrained_dict = torch.load(weights_file, map_location=device)

    if load_param_prefixes is not None and len(load_param_prefixes) > 0:
        # Selective loading (for fine-tuning: load encoder, re-init decoder)
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if any(k.startswith(p) for p in load_param_prefixes)
        }
        logger.info(
            "Selectively loading %d parameters (prefixes: %s)",
            len(pretrained_dict), load_param_prefixes,
        )

    model_dict = model.state_dict()
    # Only load params that match in name and shape
    loadable = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    skipped = set(pretrained_dict.keys()) - set(loadable.keys())
    if skipped:
        logger.warning("Skipped %d params (shape mismatch or missing): %s",
                        len(skipped), list(skipped)[:5])

    model_dict.update(loadable)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    logger.info(
        "Loaded %d/%d parameters onto %s",
        len(loadable), len(pretrained_dict), device,
    )

    return model, vocab, model_config, None


def download_scgpt_checkpoint(
    model_name: str = "scGPT_human",
    cache_dir: Optional[str] = None,
) -> str:
    """Download a pretrained scGPT checkpoint from HuggingFace or the official source.

    Parameters
    ----------
    model_name
        Model identifier. Options:
        - 'scGPT_human': Whole-human pretrained model (33M cells)
        - 'scGPT_cp':    Cell-type pretrained model
    cache_dir
        Directory to cache downloaded checkpoints.
        Defaults to ~/.cache/scgpt/

    Returns
    -------
    Path to the downloaded checkpoint directory.
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "scgpt")

    ckpt_dir = os.path.join(cache_dir, model_name)
    if os.path.exists(ckpt_dir) and os.path.exists(os.path.join(ckpt_dir, "best_model.pt")):
        logger.info("Using cached checkpoint: %s", ckpt_dir)
        return ckpt_dir

    os.makedirs(ckpt_dir, exist_ok=True)

    # Try HuggingFace hub download
    try:
        from huggingface_hub import snapshot_download
        logger.info("Downloading %s from HuggingFace Hub...", model_name)
        downloaded_path = snapshot_download(
            repo_id=f"ctheodoris/{model_name}",
            local_dir=ckpt_dir,
            local_dir_use_symlinks=False,
        )
        logger.info("Downloaded checkpoint to %s", downloaded_path)
        return downloaded_path
    except Exception as e:
        logger.warning("HuggingFace download failed: %s", e)

    # Fallback: try scGPT's built-in download
    try:
        import scgpt
        if hasattr(scgpt, "utils") and hasattr(scgpt.utils, "download_model"):
            scgpt.utils.download_model(model_name, ckpt_dir)
            return ckpt_dir
    except Exception as e:
        logger.warning("scGPT built-in download failed: %s", e)

    raise RuntimeError(
        f"Could not download scGPT checkpoint '{model_name}'. "
        f"Please download manually and place in {ckpt_dir}.\n"
        f"Expected files: vocab.json, args.json, best_model.pt"
    )


# ---------------------------------------------------------------------------
# Gene vocabulary mapping
# ---------------------------------------------------------------------------

def build_gene_id_mapping(
    vocab,
    gene_names: List[str],
) -> np.ndarray:
    """Map gene names to scGPT vocabulary token IDs.

    Parameters
    ----------
    vocab
        GeneVocab instance from scGPT checkpoint.
    gene_names
        Gene names from the dataset (e.g., from PertData.gene_names).

    Returns
    -------
    np.ndarray of int, shape (n_genes,). Token IDs in vocab space.
    Genes not in vocab are mapped to the pad token ID.
    """
    pad_id = vocab[PAD_TOKEN]
    gene_ids = np.array(
        [vocab[g] if g in vocab else pad_id for g in gene_names],
        dtype=int,
    )
    n_matched = np.sum(gene_ids != pad_id)
    logger.info(
        "Gene vocabulary matching: %d/%d genes found in scGPT vocab",
        n_matched, len(gene_names),
    )
    return gene_ids


def map_gene_symbols_to_ensembl(
    gene_symbols: List[str],
    species: str = "human",
) -> Dict[str, str]:
    """Map gene symbols to Ensembl IDs.

    scGPT's vocabulary may use either gene symbols or Ensembl IDs
    depending on the checkpoint. This helper resolves mismatches.

    Parameters
    ----------
    gene_symbols
        List of gene symbols (e.g., ['TP53', 'MDM2', ...]).
    species
        'human' or 'mouse'.

    Returns
    -------
    Dict mapping symbol -> ensembl ID (or symbol if not found).
    """
    try:
        # Try using gget for ID conversion
        import gget
        results = gget.info(gene_symbols, species=species)
        if results is not None and hasattr(results, "index"):
            return {
                row.get("symbol", sym): idx
                for idx, row in results.iterrows()
                for sym in gene_symbols
                if row.get("symbol", "") == sym
            }
    except (ImportError, Exception) as e:
        logger.debug("gget not available for gene ID mapping: %s", e)

    # Fallback: return identity mapping (assume vocab uses symbols)
    return {g: g for g in gene_symbols}


# ---------------------------------------------------------------------------
# Perturbation prediction
# ---------------------------------------------------------------------------

def predict_perturbation(
    model,
    pert_list: List[List[str]],
    ctrl_adata,
    gene_names: List[str],
    gene_ids: np.ndarray,
    device: str = "cuda",
    pool_size: int = DEFAULT_POOL_SIZE,
    batch_size: int = 64,
    include_zero_gene: str = DEFAULT_INCLUDE_ZERO_GENE,
    amp: bool = True,
) -> Dict[str, np.ndarray]:
    """Predict perturbation effects using scGPT + GEARS cell graph pipeline.

    This follows the approach from the scGPT perturbation tutorial:
    1. For each perturbation, create cell graphs from control cells
       using GEARS' create_cell_graph_dataset_for_prediction
    2. Pass batches through DataLoader
    3. Call model.pred_perturb() for each batch
    4. Average predictions across cells

    Parameters
    ----------
    model
        TransformerGenerator model on device.
    pert_list
        List of perturbations. Each element is a list of gene names
        (single-gene: [['TP53'], ['MDM2']], combo: [['TP53', 'MDM2']]).
    ctrl_adata
        AnnData of control cells (condition == 'ctrl').
    gene_names
        Ordered gene names matching the GEARS data pipeline.
    gene_ids
        Token IDs in scGPT vocab space for each gene in gene_names.
    device
        Compute device.
    pool_size
        Number of control cells to sample per perturbation prediction.
    batch_size
        Batch size for inference.
    include_zero_gene
        Strategy for zero-expression genes: 'all', 'batch-wise'.
    amp
        Use automatic mixed precision.

    Returns
    -------
    Dict mapping perturbation key (e.g., 'TP53') -> mean predicted
    expression array of shape (n_genes,).
    """
    import torch
    from torch_geometric.loader import DataLoader

    _, create_cell_graph = _import_gears()

    model.eval()
    results = {}

    for pert in pert_list:
        pert_key = "_".join(pert)

        # Create cell graphs for this perturbation
        try:
            cell_graphs = create_cell_graph(
                pert, ctrl_adata, gene_names, device,
                num_samples=pool_size,
            )
        except Exception as e:
            logger.warning(
                "Failed to create cell graphs for %s: %s", pert_key, e
            )
            continue

        loader = DataLoader(
            cell_graphs, batch_size=batch_size, shuffle=False,
        )

        preds = []
        with torch.no_grad():
            for batch_data in loader:
                batch_data.to(device)
                pred_values = model.pred_perturb(
                    batch_data,
                    include_zero_gene=include_zero_gene,
                    gene_ids=gene_ids,
                    amp=amp,
                )
                preds.append(pred_values.cpu())

        if preds:
            preds = torch.cat(preds, dim=0)
            results[pert_key] = preds.numpy().mean(axis=0).astype(np.float32)
            logger.debug(
                "Predicted %s: %d cells averaged, output shape %s",
                pert_key, preds.shape[0], results[pert_key].shape,
            )
        else:
            logger.warning("No predictions for %s", pert_key)

    return results


def predict_single_perturbation(
    model,
    gene: str,
    ctrl_adata,
    gene_names: List[str],
    gene_ids: np.ndarray,
    device: str = "cuda",
    pool_size: int = DEFAULT_POOL_SIZE,
    batch_size: int = 64,
    include_zero_gene: str = DEFAULT_INCLUDE_ZERO_GENE,
    amp: bool = True,
) -> np.ndarray:
    """Predict mean expression after single-gene CRISPRi knockdown.

    Convenience wrapper around predict_perturbation for single-gene case.

    Parameters
    ----------
    gene
        Gene to perturb (knockdown).

    Returns
    -------
    np.ndarray of shape (n_genes,) -- mean post-perturbation expression.
    """
    results = predict_perturbation(
        model=model,
        pert_list=[[gene]],
        ctrl_adata=ctrl_adata,
        gene_names=gene_names,
        gene_ids=gene_ids,
        device=device,
        pool_size=pool_size,
        batch_size=batch_size,
        include_zero_gene=include_zero_gene,
        amp=amp,
    )
    key = gene
    if key not in results:
        raise RuntimeError(f"Prediction failed for gene {gene}")
    return results[key]


# ---------------------------------------------------------------------------
# Fine-tuning on perturbation data
# ---------------------------------------------------------------------------

def finetune_scgpt_perturbation(
    checkpoint_dir: str,
    pert_data,
    gene_ids: np.ndarray,
    device: str = "cuda",
    epochs: int = 15,
    lr: float = 1e-4,
    batch_size: int = 64,
    schedule_gamma: float = 0.9,
    amp: bool = True,
    use_fast_transformer: bool = True,
    save_dir: Optional[str] = None,
) -> tuple:
    """Fine-tune a pretrained scGPT model on perturbation data.

    Loads encoder weights from the pretrained checkpoint and trains
    the full model (encoder + decoder) on perturbation prediction
    using MSE loss on predicted vs. observed expression.

    Parameters
    ----------
    checkpoint_dir
        Directory with pretrained scGPT checkpoint.
    pert_data
        GEARS PertData object with loaded and split data.
    gene_ids
        Token IDs for each gene in scGPT vocab.
    device
        Compute device.
    epochs
        Number of training epochs.
    lr
        Learning rate.
    batch_size
        Training batch size.
    schedule_gamma
        Learning rate decay factor per epoch.
    amp
        Use automatic mixed precision.
    use_fast_transformer
        Use flash attention.
    save_dir
        If provided, save fine-tuned model here.

    Returns
    -------
    (model, best_val_loss)
    """
    import torch
    from torch import nn
    from scgpt.loss import masked_mse_loss

    # Load model with selective parameter loading (encoder only)
    model, vocab, config, _ = load_scgpt_checkpoint(
        checkpoint_dir,
        device=device,
        use_fast_transformer=use_fast_transformer,
        load_param_prefixes=FINETUNE_PARAM_PREFIXES,
    )
    model.train()

    n_genes = len(gene_ids)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=schedule_gamma
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    include_zero_gene = DEFAULT_INCLUDE_ZERO_GENE
    max_seq_len = DEFAULT_MAX_SEQ_LEN

    _, _, map_raw_id = _import_scgpt()

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_data in pert_data.dataloader["train_loader"]:
            batch_sz = len(batch_data.y)
            batch_data.to(device)

            x = batch_data.x
            ori_gene_values = x[:, 0].view(batch_sz, n_genes)
            pert_flags = x[:, 1].long().view(batch_sz, n_genes)
            target_values = batch_data.y

            # Select gene subset based on include_zero_gene strategy
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(
                    n_genes, device=device, dtype=torch.long
                )
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )

            # Truncate if exceeds max sequence length
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(
                    len(input_gene_ids), device=device
                )[:max_seq_len]

            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_subset = target_values[:, input_gene_ids]

            # Map raw gene indices to scGPT vocab IDs
            mapped_ids = map_raw_id(input_gene_ids, gene_ids)
            mapped_ids = mapped_ids.repeat(batch_sz, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    mapped_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                )
                output_values = output_dict["mlm_output"]

                # All positions are prediction targets
                masked_positions = torch.ones_like(
                    input_values, dtype=torch.bool
                )
                loss = masked_mse_loss(
                    output_values, target_subset, masked_positions
                )

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()
        logger.info(
            "Epoch %d/%d: train_loss=%.4f, lr=%.2e",
            epoch, epochs, avg_loss, scheduler.get_last_lr()[0],
        )

        # Validation
        if "val_loader" in pert_data.dataloader:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch_data in pert_data.dataloader["val_loader"]:
                    batch_sz = len(batch_data.y)
                    batch_data.to(device)
                    x = batch_data.x
                    ori_gene_values = x[:, 0].view(batch_sz, n_genes)
                    pert_flags = x[:, 1].long().view(batch_sz, n_genes)
                    target_values = batch_data.y

                    input_gene_ids = torch.arange(
                        n_genes, device=device, dtype=torch.long
                    )
                    mapped_ids = map_raw_id(input_gene_ids, gene_ids)
                    mapped_ids = mapped_ids.repeat(batch_sz, 1)
                    src_mask = torch.zeros_like(
                        ori_gene_values, dtype=torch.bool, device=device
                    )

                    with torch.cuda.amp.autocast(enabled=amp):
                        out = model(
                            mapped_ids,
                            ori_gene_values,
                            pert_flags,
                            src_key_padding_mask=src_mask,
                        )
                        pred = out["mlm_output"]
                        mask = torch.ones_like(
                            ori_gene_values, dtype=torch.bool
                        )
                        loss = masked_mse_loss(pred, target_values, mask)

                    val_loss += loss.item()
                    n_val += 1

            avg_val = val_loss / max(n_val, 1)
            logger.info("  val_loss=%.4f", avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    model.eval()

    # Save fine-tuned model
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        import torch as _torch
        _torch.save(model.state_dict(), save_path / "best_model.pt")
        vocab.save_json(save_path / "vocab.json")
        with open(save_path / "args.json", "w") as f:
            json.dump(config, f)
        logger.info("Saved fine-tuned model to %s", save_path)

    return model, best_val_loss


# ---------------------------------------------------------------------------
# ScGPTModel class (CausalCellBench interface)
# ---------------------------------------------------------------------------

class ScGPTModel:
    """scGPT wrapper implementing the CausalCellBench model interface.

    Follows the same pattern as GEARSModel: avoids importing heavy deps
    at class definition time. The benchmark runner calls this through
    conda subprocess or Modal isolation.

    Attributes
    ----------
    _model : TransformerGenerator or None
    _vocab : GeneVocab or None
    _gene_ids : np.ndarray or None (vocab-mapped gene IDs)
    _gene_list : list[str] -- gene names from GEARS data pipeline
    _ctrl_adata : AnnData of control cells for prediction
    _pert_data : PertData instance (if fine-tuning)
    _device : compute device string
    _dispersion_cache : cached NB dispersion for cell sampling
    """

    def __init__(self, device: Optional[str] = None):
        self._model = None
        self._vocab = None
        self._gene_ids: Optional[np.ndarray] = None
        self._gene_list: list[str] = []
        self._ctrl_adata = None
        self._pert_data = None
        self._device = device or ("cuda" if _has_cuda() else "cpu")
        self._dispersion_cache = None
        self._model_config: Optional[dict] = None

    @property
    def name(self) -> str:
        return "scGPT"

    @property
    def supports_multi_perturbation(self) -> bool:
        # scGPT does NOT natively support multi-gene perturbation
        # in the same way GEARS does (no combo graph input).
        # It can do single-gene masking only.
        return False

    def load_model(
        self,
        checkpoint_path: str = "",
        adata_train=None,
        data_path: str = "./scgpt_data",
        data_name: str = "adamson",
        use_fast_transformer: bool = True,
        finetune_epochs: int = 15,
        finetune_lr: float = 1e-4,
    ) -> None:
        """Load pretrained scGPT model.

        Two modes:
        1. Pretrained only: provide checkpoint_path, use zero-shot or
           previously fine-tuned weights.
        2. Fine-tune: provide checkpoint_path AND adata_train (or data_name).
           Loads pretrained encoder, fine-tunes on perturbation data.

        Parameters
        ----------
        checkpoint_path
            Directory with scGPT checkpoint files.
        adata_train
            If provided, fine-tune the model on this perturbation data.
        data_path
            Directory for GEARS data cache.
        data_name
            Built-in GEARS dataset name (for fine-tuning).
        use_fast_transformer
            Use flash attention (requires flash-attn).
        finetune_epochs
            Training epochs if fine-tuning.
        finetune_lr
            Learning rate if fine-tuning.
        """
        if not checkpoint_path:
            # Try to download the default whole-human model
            checkpoint_path = download_scgpt_checkpoint("scGPT_human")

        PertData, _ = _import_gears()

        # Set up GEARS data pipeline (needed for gene list + cell graphs)
        pert_data = PertData(data_path)

        if adata_train is not None:
            # Fine-tune mode: process custom AnnData through GEARS
            from causalcellbench.models.gears_wrapper import prepare_adata_for_gears
            gears_adata = prepare_adata_for_gears(adata_train)
            pert_data.new_data_process(
                dataset_name="scgpt_custom",
                adata=gears_adata,
            )
            pert_data.load(data_path=os.path.join(data_path, "scgpt_custom"))
        else:
            pert_data.load(data_name=data_name)

        pert_data.prepare_split(split="simulation", seed=1)
        pert_data.get_dataloader(batch_size=64, test_batch_size=64)

        self._pert_data = pert_data
        self._gene_list = pert_data.gene_names.values.tolist()

        # Build gene ID mapping
        model, vocab, config, _ = load_scgpt_checkpoint(
            checkpoint_path,
            device=self._device,
            use_fast_transformer=use_fast_transformer,
            load_param_prefixes=(
                FINETUNE_PARAM_PREFIXES if adata_train is not None else None
            ),
        )
        self._vocab = vocab
        self._model_config = config
        self._gene_ids = build_gene_id_mapping(vocab, self._gene_list)

        if adata_train is not None:
            # Fine-tune on perturbation data
            logger.info("Fine-tuning scGPT for %d epochs...", finetune_epochs)
            model, val_loss = finetune_scgpt_perturbation(
                checkpoint_dir=checkpoint_path,
                pert_data=pert_data,
                gene_ids=self._gene_ids,
                device=self._device,
                epochs=finetune_epochs,
                lr=finetune_lr,
                use_fast_transformer=use_fast_transformer,
            )
            logger.info("Fine-tuning complete, best val_loss=%.4f", val_loss)

        self._model = model

        # Extract control cells for prediction
        ctrl_mask = pert_data.adata.obs["condition"] == "ctrl"
        self._ctrl_adata = pert_data.adata[ctrl_mask].copy()
        logger.info(
            "scGPT loaded: %d genes in vocab, %d control cells",
            len(self._gene_list), self._ctrl_adata.n_obs,
        )

    def predict_perturbation_mean(self, gene: str) -> np.ndarray:
        """Get the raw mean prediction for a single gene.

        Returns
        -------
        np.ndarray of shape (n_genes,) in GEARS gene order.
        """
        if self._model is None:
            raise RuntimeError("Call load_model() first")

        return predict_single_perturbation(
            model=self._model,
            gene=gene,
            ctrl_adata=self._ctrl_adata,
            gene_names=self._gene_list,
            gene_ids=self._gene_ids,
            device=self._device,
        )

    def predict_perturbation(
        self,
        control_adata,
        perturbed_gene: str,
        n_cells: int,
        gene_names: list[str],
        seed: int = 0,
    ):
        """Predict single-gene perturbation with NB-sampled cells.

        Parameters
        ----------
        control_adata
            AnnData of unperturbed control cells.
        perturbed_gene
            Gene to knock down.
        n_cells
            Number of simulated cells.
        gene_names
            Output gene names.
        seed
            Random seed.

        Returns
        -------
        PerturbationResult
        """
        from causalcellbench.models.base import PerturbationResult
        from causalcellbench.models.sampling import (
            sample_cells_from_model_output,
            estimate_nb_dispersion,
        )

        if self._model is None:
            raise RuntimeError("Call load_model() first")

        # Get scGPT mean prediction
        mean_scgpt = self.predict_perturbation_mean(perturbed_gene)

        # Align scGPT output (in GEARS gene order) to requested gene_names
        scgpt_to_idx = {g: i for i, g in enumerate(self._gene_list)}
        aligned_mean = np.zeros(len(gene_names), dtype=np.float32)
        for j, g in enumerate(gene_names):
            if g in scgpt_to_idx:
                aligned_mean[j] = mean_scgpt[scgpt_to_idx[g]]
            else:
                # Gene not in scGPT/GEARS vocabulary -- use control mean
                if g in control_adata.var_names:
                    idx = list(control_adata.var_names).index(g)
                    X = control_adata.X
                    if hasattr(X, "toarray"):
                        col = X[:, idx].toarray().flatten()
                    else:
                        col = np.asarray(X[:, idx]).flatten()
                    aligned_mean[j] = col.mean()

        # NB sampling for cell-level data
        if self._dispersion_cache is None:
            self._dispersion_cache = estimate_nb_dispersion(
                control_adata, gene_names
            )

        cells = sample_cells_from_model_output(
            aligned_mean, control_adata, n_cells, gene_names, seed,
            dispersion_cache=self._dispersion_cache,
        )

        return PerturbationResult(
            expression_matrix=cells,
            gene_names=gene_names,
            perturbed_genes=[perturbed_gene],
            cell_type="",
            metadata={"model_config": self._model_config},
        )


# ---------------------------------------------------------------------------
# Standalone prediction (CLI)
# ---------------------------------------------------------------------------

def run_scgpt_predictions(
    input_path: str,
    output_path: str,
    genes_to_perturb: list[str],
    checkpoint_path: str,
    data_name: str = "adamson",
    data_path: str = "./scgpt_data",
    cell_type: str = "K562",
    pool_size: int = DEFAULT_POOL_SIZE,
    batch_size: int = 64,
    use_fast_transformer: bool = True,
    device: Optional[str] = None,
) -> None:
    """Run scGPT perturbation predictions and save to disk.

    This is the main entry point for CLI / Modal usage. Loads pretrained
    model, predicts perturbation effects for specified genes, saves
    mean predictions as .h5ad.

    Parameters
    ----------
    input_path
        Path to control cells .h5ad file.
    output_path
        Path to write predictions .h5ad file.
    genes_to_perturb
        List of genes to perturb (one at a time).
    checkpoint_path
        Path to scGPT pretrained checkpoint directory.
    data_name
        GEARS dataset name (for gene list and cell graph construction).
    data_path
        Directory for GEARS data cache.
    cell_type
        Cell type context for metadata.
    pool_size
        Number of control cells to sample per prediction.
    batch_size
        Inference batch size.
    use_fast_transformer
        Use flash attention.
    device
        'cuda' or 'cpu' (auto-detect if None).
    """
    import anndata as ad
    import pandas as pd

    if device is None:
        device = "cuda" if _has_cuda() else "cpu"

    # Load control cells
    logger.info("Loading control cells from %s", input_path)
    control = ad.read_h5ad(input_path)
    logger.info("Control: %d cells, %d genes", control.n_obs, control.n_vars)

    # Set up GEARS data pipeline
    PertData, _ = _import_gears()
    pert_data = PertData(data_path)
    pert_data.load(data_name=data_name)
    pert_data.prepare_split(split="simulation", seed=1)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=batch_size)

    gene_list = pert_data.gene_names.values.tolist()

    # Get control cells from GEARS data
    ctrl_mask = pert_data.adata.obs["condition"] == "ctrl"
    ctrl_adata = pert_data.adata[ctrl_mask].copy()

    # Load scGPT model
    logger.info("Loading scGPT checkpoint from %s", checkpoint_path)
    model, vocab, config, _ = load_scgpt_checkpoint(
        checkpoint_path,
        device=device,
        use_fast_transformer=use_fast_transformer,
    )

    gene_ids = build_gene_id_mapping(vocab, gene_list)

    # Filter to genes that exist in GEARS gene list
    valid_genes = [g for g in genes_to_perturb if g in gene_list]
    skipped = [g for g in genes_to_perturb if g not in gene_list]
    if skipped:
        logger.warning(
            "Skipping %d genes not in GEARS gene list: %s",
            len(skipped), skipped[:10],
        )

    if not valid_genes:
        logger.error("No valid genes to predict!")
        sys.exit(1)

    # Predict perturbations
    logger.info("Predicting %d perturbations...", len(valid_genes))
    pert_list = [[g] for g in valid_genes]
    t_start = time.time()

    results = predict_perturbation(
        model=model,
        pert_list=pert_list,
        ctrl_adata=ctrl_adata,
        gene_names=gene_list,
        gene_ids=gene_ids,
        device=device,
        pool_size=pool_size,
        batch_size=batch_size,
    )

    elapsed = time.time() - t_start

    # Collect results
    mean_predictions = []
    pert_labels = []
    for gene in valid_genes:
        if gene in results:
            mean_predictions.append(results[gene])
            pert_labels.append(gene)
        else:
            logger.warning("No result for gene %s", gene)
            skipped.append(gene)

    if not mean_predictions:
        logger.error("No successful predictions!")
        sys.exit(1)

    pred_matrix = np.stack(mean_predictions, axis=0)

    # Build output AnnData
    result = ad.AnnData(
        X=pred_matrix.astype(np.float32),
        obs=pd.DataFrame({"perturbation": pert_labels}),
        var=pd.DataFrame(index=gene_list[:pred_matrix.shape[1]]),
        uns={
            "model": "scgpt",
            "output_type": "mean",
            "checkpoint": str(checkpoint_path),
            "cell_type": cell_type,
            "n_predicted": len(pert_labels),
            "n_skipped": len(skipped),
            "skipped_genes": skipped[:50],
            "inference_time_seconds": elapsed,
            "device": device,
            "pool_size": pool_size,
            "data_name": data_name,
        },
    )
    result.write_h5ad(output_path)
    logger.info(
        "Saved %d predictions to %s (skipped %d, %.1fs total)",
        len(pert_labels), output_path, len(skipped), elapsed,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="scGPT perturbation prediction for CausalCellBench"
    )
    parser.add_argument("--input", required=True, help="Control cells .h5ad")
    parser.add_argument("--output", required=True, help="Output predictions .h5ad")
    parser.add_argument("--genes", required=True,
                        help="Comma-separated genes to perturb")
    parser.add_argument("--checkpoint", required=True,
                        help="scGPT checkpoint directory")
    parser.add_argument("--data-name", default="adamson",
                        help="GEARS dataset name")
    parser.add_argument("--data-path", default="./scgpt_data",
                        help="GEARS data cache directory")
    parser.add_argument("--cell-type", default="K562")
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE,
                        help="Number of control cells to sample per prediction")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None,
                        help="cuda or cpu (auto-detect if omitted)")
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Disable flash attention")
    args = parser.parse_args()

    run_scgpt_predictions(
        input_path=args.input,
        output_path=args.output,
        genes_to_perturb=args.genes.split(","),
        checkpoint_path=args.checkpoint,
        data_name=args.data_name,
        data_path=args.data_path,
        cell_type=args.cell_type,
        pool_size=args.pool_size,
        batch_size=args.batch_size,
        use_fast_transformer=not args.no_flash_attn,
        device=args.device,
    )
