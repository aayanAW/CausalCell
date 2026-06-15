"""GEARS (Graph-Enhanced Gene Activation and Repression Simulator) wrapper.

GEARS (Roohani et al., Nat Biotech 2024) uses a GNN on gene-gene interaction
graphs to predict post-perturbation gene expression. It accepts perturbation
gene sets as input and can predict unseen single and combinatorial effects.
Native multi-perturbation support makes it eligible for the d-separation test.

Outputs: mean prediction per gene (no cell-level sampling).
Bridge applies NB sampling via causalcellbench.models.sampling.

GEARS API overview (from snap-stanford/GEARS):
    PertData(data_path)              -- manages datasets and dataloaders
    PertData.load(data_name=...)     -- load built-in dataset (norman, replogle_k562_essential, etc.)
    PertData.new_data_process(name, adata)  -- process custom AnnData
        Required: adata.obs['condition'], adata.obs['cell_type'], adata.var['gene_name']
        Control cells: condition == 'ctrl'
        Single perturbation: condition == 'GENE+ctrl'
        Combo: condition == 'GENE1+GENE2'
    PertData.prepare_split(split='simulation', seed=1)
    PertData.get_dataloader(batch_size=32)
    GEARS(pert_data, device='cuda')
    GEARS.model_initialize(hidden_size=64, ...)
    GEARS.train(epochs=20, lr=1e-3)
    GEARS.predict([['gene1'], ['gene2', 'gene3']])
        Returns: dict mapping 'gene1' -> np.ndarray(n_genes,),
                               'gene2+gene3' -> np.ndarray(n_genes,)
    GEARS.save_model(path) / GEARS.load_pretrained(path)

Installation:
    conda create -n env_gears python=3.10
    conda activate env_gears
    pip install torch torch_geometric cell-gears anndata scanpy

Usage (CLI for benchmark.py):
    conda run -n env_gears python -m causalcellbench.models.gears_wrapper \\
        --input data/k562_control.h5ad \\
        --output results/gears_predictions.h5ad \\
        --genes TP53,MDM2,...

Usage (programmatic):
    from causalcellbench.models.gears_wrapper import GEARSModel
    model = GEARSModel()
    model.load_model(checkpoint_path="checkpoints/gears_k562")
    result = model.predict_perturbation(control_adata, "TP53", n_cells=100, gene_names=[...])
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports -- GEARS has heavy deps (torch, torch_geometric, etc.)
# ---------------------------------------------------------------------------

def _import_gears():
    """Import GEARS with clear error messages."""
    try:
        from gears import PertData, GEARS
        return PertData, GEARS
    except ImportError as e:
        raise ImportError(
            "cell-gears not installed.  Install in the env_gears conda env:\n"
            "  pip install cell-gears torch torch_geometric\n"
            f"Original error: {e}"
        ) from e


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def prepare_adata_for_gears(
    adata,
    perturbation_key: str = "gene",
    control_label: str = "non-targeting",
    cell_type_label: str = "K562",
) -> "anndata.AnnData":
    """Convert a CRISPRi perturbation AnnData into the format GEARS expects.

    GEARS requires:
        - adata.obs['condition']: 'ctrl' for controls, 'GENE+ctrl' for single KD
        - adata.obs['cell_type']: cell type string
        - adata.var['gene_name']: gene identifiers (usually same as var_names)

    Parameters
    ----------
    adata
        AnnData with perturbation labels (e.g., Replogle K562 CRISPRi screen).
    perturbation_key
        Column in adata.obs containing gene perturbation labels.
    control_label
        Value in perturbation_key that indicates control cells.
    cell_type_label
        Cell type string to assign (GEARS uses this for conditioning).

    Returns
    -------
    AnnData formatted for GEARS new_data_process().
    """
    import anndata as ad

    adata = adata.copy()

    # Build 'condition' column: GEARS expects 'ctrl' or 'GENE+ctrl'
    conditions = []
    for val in adata.obs[perturbation_key]:
        if val == control_label or val.lower() in ("ctrl", "control", "non-targeting"):
            conditions.append("ctrl")
        elif "+" in val:
            # Already a combo perturbation, keep as-is
            conditions.append(val)
        else:
            # Single perturbation: GEARS format is 'GENE+ctrl'
            conditions.append(f"{val}+ctrl")
    adata.obs["condition"] = conditions

    # Cell type
    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = cell_type_label

    # Gene names in var
    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = adata.var_names.tolist()

    logger.info(
        "Prepared AnnData for GEARS: %d cells, %d genes, %d conditions",
        adata.n_obs, adata.n_vars,
        adata.obs["condition"].nunique(),
    )
    n_ctrl = (adata.obs["condition"] == "ctrl").sum()
    logger.info("  Control cells: %d", n_ctrl)

    return adata


# ---------------------------------------------------------------------------
# Core functions: train_gears / predict_perturbation
# ---------------------------------------------------------------------------

def train_gears(
    adata,
    perturbation_key: str = "gene",
    control_label: str = "non-targeting",
    cell_type_label: str = "K562",
    data_path: str = "./gears_data",
    dataset_name: str = "k562_crispri",
    split: str = "simulation",
    seed: int = 1,
    batch_size: int = 32,
    hidden_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str | None = None,
    save_dir: str | None = None,
):
    """Train a GEARS model on perturbation data.

    Parameters
    ----------
    adata
        AnnData with perturbation experiment data. Must contain both control
        and perturbed cells with labels in obs[perturbation_key].
    perturbation_key
        Column in adata.obs with perturbation labels.
    control_label
        Value that marks control cells.
    cell_type_label
        Cell type to assign.
    data_path
        Directory for GEARS to cache processed data.
    dataset_name
        Name for the processed dataset (used as subdirectory).
    split
        Train/val/test split strategy. 'simulation' holds out unseen
        perturbations for testing (recommended for our use case).
    seed
        Random seed for splitting.
    batch_size
        Training batch size.
    hidden_size
        GNN hidden dimension.
    epochs
        Number of training epochs.
    lr
        Learning rate.
    device
        'cuda', 'cpu', or None (auto-detect).
    save_dir
        If provided, save trained model here.

    Returns
    -------
    Tuple of (GEARS model, PertData object).
    """
    PertData, GEARS = _import_gears()

    if device is None:
        device = "cuda" if _has_cuda() else "cpu"
    logger.info("GEARS training on device: %s", device)

    # Prepare AnnData
    gears_adata = prepare_adata_for_gears(
        adata,
        perturbation_key=perturbation_key,
        control_label=control_label,
        cell_type_label=cell_type_label,
    )

    # Process through GEARS data pipeline
    data_dir = Path(data_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    pert_data = PertData(str(data_dir))

    dataset_dir = data_dir / dataset_name
    if dataset_dir.exists() and (dataset_dir / "perturb_processed.h5ad").exists():
        logger.info("Loading previously processed dataset from %s", dataset_dir)
        pert_data.load(data_path=str(dataset_dir))
    else:
        logger.info("Processing new dataset '%s' through GEARS pipeline...", dataset_name)
        pert_data.new_data_process(
            dataset_name=dataset_name,
            adata=gears_adata,
        )
        pert_data.load(data_path=str(dataset_dir))

    pert_data.prepare_split(split=split, seed=seed)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=batch_size * 4)

    # Build model
    gears_model = GEARS(pert_data, device=device)
    gears_model.model_initialize(hidden_size=hidden_size)

    # Train
    logger.info("Training GEARS for %d epochs (lr=%.1e, hidden=%d)", epochs, lr, hidden_size)
    t0 = time.time()
    gears_model.train(epochs=epochs, lr=lr)
    elapsed = time.time() - t0
    logger.info("GEARS training complete in %.1f seconds", elapsed)

    # Save
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        gears_model.save_model(str(save_path))
        logger.info("Saved GEARS model to %s", save_path)

    return gears_model, pert_data


def predict_perturbation(
    model,
    gene_name: str,
) -> np.ndarray:
    """Predict mean expression vector for a single-gene perturbation.

    Parameters
    ----------
    model
        A trained GEARS model instance.
    gene_name
        Gene to perturb (knockdown).

    Returns
    -------
    np.ndarray of shape (n_genes,) -- predicted mean post-perturbation expression.
    """
    pred_dict = model.predict([[gene_name]])

    # GEARS returns dict: {'gene_name' -> np.ndarray(n_genes,)}
    # For single gene, key is the gene name itself
    if isinstance(pred_dict, tuple):
        # Uncertainty mode returns (pred_dict, logvar_dict)
        pred_dict = pred_dict[0]

    if isinstance(pred_dict, dict):
        # Key could be 'gene_name' or 'gene_name+ctrl' depending on version
        for key in [gene_name, f"{gene_name}+ctrl", f"{gene_name}_ctrl"]:
            if key in pred_dict:
                return np.asarray(pred_dict[key], dtype=np.float32).flatten()
        # Fallback: take first (only) entry
        key = list(pred_dict.keys())[0]
        logger.debug("GEARS prediction key: %s (expected %s)", key, gene_name)
        return np.asarray(pred_dict[key], dtype=np.float32).flatten()
    else:
        return np.asarray(pred_dict, dtype=np.float32).flatten()


def predict_multi_perturbation(
    model,
    gene_names: list[str],
) -> np.ndarray:
    """Predict mean expression for a multi-gene (combinatorial) perturbation.

    Parameters
    ----------
    model
        A trained GEARS model instance.
    gene_names
        List of genes to perturb simultaneously.

    Returns
    -------
    np.ndarray of shape (n_genes,).
    """
    pred_dict = model.predict([gene_names])

    if isinstance(pred_dict, tuple):
        pred_dict = pred_dict[0]

    if isinstance(pred_dict, dict):
        # Combo key: 'gene1+gene2' or 'gene1_gene2'
        joined_plus = "+".join(gene_names)
        joined_under = "_".join(gene_names)
        for key in [joined_plus, joined_under]:
            if key in pred_dict:
                return np.asarray(pred_dict[key], dtype=np.float32).flatten()
        key = list(pred_dict.keys())[0]
        return np.asarray(pred_dict[key], dtype=np.float32).flatten()
    else:
        return np.asarray(pred_dict, dtype=np.float32).flatten()


def load_pretrained_gears(
    checkpoint_path: str,
    data_path: str = "./gears_data",
    data_name: str = "norman",
    device: str | None = None,
):
    """Load a pretrained GEARS model from disk.

    Parameters
    ----------
    checkpoint_path
        Directory containing saved GEARS model files.
    data_path
        Directory with processed GEARS data.
    data_name
        Dataset name (for PertData loading).
    device
        Compute device.

    Returns
    -------
    Tuple of (GEARS model, PertData object).
    """
    PertData, GEARS = _import_gears()

    if device is None:
        device = "cuda" if _has_cuda() else "cpu"

    pert_data = PertData(str(data_path))
    pert_data.load(data_name=data_name)
    pert_data.prepare_split(split="simulation", seed=1)
    pert_data.get_dataloader(batch_size=32)

    gears_model = GEARS(pert_data, device=device)
    gears_model.model_initialize(hidden_size=64)
    gears_model.load_pretrained(checkpoint_path)
    logger.info("Loaded pretrained GEARS from %s", checkpoint_path)

    return gears_model, pert_data


# ---------------------------------------------------------------------------
# AbstractPerturbationModel implementation
# ---------------------------------------------------------------------------

class GEARSModel:
    """GEARS wrapper implementing the CausalCellBench model interface.

    Extends AbstractPerturbationModel conceptually but avoids importing
    torch/gears at class definition time (they may not be installed in
    the eval environment).  The benchmark.py runner calls this through
    conda subprocess isolation anyway.

    Attributes
    ----------
    _model : GEARS instance or None
    _pert_data : PertData instance or None
    _gene_list : list[str] -- GEARS gene vocabulary
    _dispersion_cache : cached NB dispersion from control cells
    """

    def __init__(self):
        self._model = None
        self._pert_data = None
        self._gene_list: list[str] = []
        self._dispersion_cache = None

    @property
    def name(self) -> str:
        return "GEARS"

    @property
    def supports_multi_perturbation(self) -> bool:
        return True

    def load_model(
        self,
        checkpoint_path: str = "",
        adata_train=None,
        data_path: str = "./gears_data",
        data_name: str = "norman",
        device: str | None = None,
    ) -> None:
        """Load pretrained GEARS model.

        Parameters
        ----------
        checkpoint_path
            Directory containing saved model weights.
        adata_train
            If provided and checkpoint_path is empty, train from scratch.
        data_path
            Path to GEARS data directory.
        data_name
            Built-in dataset name (for pretrained models).
        device
            Compute device.
        """
        if checkpoint_path and Path(checkpoint_path).exists():
            self._model, self._pert_data = load_pretrained_gears(
                checkpoint_path=checkpoint_path,
                data_path=data_path,
                data_name=data_name,
                device=device,
            )
        elif adata_train is not None:
            self._model, self._pert_data = train_gears(
                adata=adata_train,
                data_path=data_path,
                device=device,
            )
        else:
            raise ValueError(
                "Either checkpoint_path (to existing weights) or adata_train "
                "(to train from scratch) must be provided."
            )

        self._gene_list = list(self._pert_data.adata.var_names)
        logger.info("GEARS loaded with %d genes in vocabulary", len(self._gene_list))

    def predict_perturbation_mean(self, gene: str) -> np.ndarray:
        """Get the raw mean prediction for a single gene.

        Returns
        -------
        np.ndarray of shape (n_gears_genes,) in GEARS gene order.
        """
        if self._model is None:
            raise RuntimeError("Call load_model() first")
        return predict_perturbation(self._model, gene)

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
            Output gene names (may differ from GEARS vocabulary).
        seed
            Random seed.

        Returns
        -------
        PerturbationResult with expression_matrix of shape (n_cells, len(gene_names)).
        """
        from causalcellbench.models.base import PerturbationResult
        from causalcellbench.models.sampling import (
            sample_cells_from_model_output,
            estimate_nb_dispersion,
        )

        if self._model is None:
            raise RuntimeError("Call load_model() first")

        # Get GEARS mean prediction
        mean_gears = predict_perturbation(self._model, perturbed_gene)

        # Align GEARS output (in GEARS gene order) to requested gene_names
        gears_to_idx = {g: i for i, g in enumerate(self._gene_list)}
        aligned_mean = np.zeros(len(gene_names), dtype=np.float32)
        for j, g in enumerate(gene_names):
            if g in gears_to_idx:
                aligned_mean[j] = mean_gears[gears_to_idx[g]]
            else:
                # Gene not in GEARS vocabulary -- use control mean as fallback
                if g in control_adata.var_names:
                    idx = list(control_adata.var_names).index(g)
                    X = control_adata.X
                    if hasattr(X, "toarray"):
                        col = X[:, idx].toarray().flatten()
                    else:
                        col = np.asarray(X[:, idx]).flatten()
                    aligned_mean[j] = col.mean()

        # Cache dispersion estimation for efficiency
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
        )

    def predict_multi_perturbation(
        self,
        control_adata,
        perturbed_genes: list[str],
        n_cells: int,
        gene_names: list[str],
        seed: int = 0,
    ):
        """Predict combinatorial perturbation (GEARS native capability).

        Required for d-separation test eligibility.
        """
        from causalcellbench.models.base import PerturbationResult
        from causalcellbench.models.sampling import (
            sample_cells_from_model_output,
            estimate_nb_dispersion,
        )

        if self._model is None:
            raise RuntimeError("Call load_model() first")

        mean_gears = predict_multi_perturbation(self._model, perturbed_genes)

        gears_to_idx = {g: i for i, g in enumerate(self._gene_list)}
        aligned_mean = np.zeros(len(gene_names), dtype=np.float32)
        for j, g in enumerate(gene_names):
            if g in gears_to_idx:
                aligned_mean[j] = mean_gears[gears_to_idx[g]]

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
            perturbed_genes=perturbed_genes,
            cell_type="",
        )


# ---------------------------------------------------------------------------
# CLI entrypoint (called by benchmark.py via conda subprocess)
# ---------------------------------------------------------------------------

def run_gears_predictions(
    input_path: str,
    output_path: str,
    genes_to_perturb: list[str],
    data_name: str = "norman",
    data_path: str = "./gears_data",
    checkpoint_path: str = "",
    train_epochs: int = 20,
    cell_type: str = "K562",
    device: str | None = None,
) -> None:
    """Run GEARS perturbation predictions and save means to disk.

    Two modes:
    1. Pretrained: provide checkpoint_path to load existing model.
    2. Train from scratch: omit checkpoint_path and the model trains
       on the built-in dataset specified by data_name.

    The output .h5ad has one row per perturbation with mean expression
    in GEARS gene ordering. The bridge layer handles NB sampling.
    """
    import anndata as ad
    import pandas as pd

    PertData, GEARS_cls = _import_gears()

    control = ad.read_h5ad(input_path)
    logger.info("Control: %d cells, %d genes", control.n_obs, control.n_vars)

    if device is None:
        device = "cuda" if _has_cuda() else "cpu"

    # Set up PertData
    data_dir = Path(data_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    pert_data = PertData(str(data_dir))
    pert_data.load(data_name=data_name)
    pert_data.prepare_split(split="simulation", seed=1)
    pert_data.get_dataloader(batch_size=32)

    gears_model = GEARS_cls(pert_data, device=device)
    gears_model.model_initialize(hidden_size=64)

    if checkpoint_path and Path(checkpoint_path).exists():
        gears_model.load_pretrained(checkpoint_path)
        logger.info("Loaded GEARS weights from %s", checkpoint_path)
    else:
        logger.info("No checkpoint provided. Training GEARS for %d epochs...", train_epochs)
        gears_model.train(epochs=train_epochs, lr=1e-3)
        # Save the trained model
        save_dir = data_dir / "trained_model"
        save_dir.mkdir(exist_ok=True)
        gears_model.save_model(str(save_dir))
        logger.info("Saved trained model to %s", save_dir)

    # GEARS gene vocabulary
    gears_genes = list(pert_data.adata.var_names)

    # Predict perturbations
    mean_predictions = []
    pert_labels = []
    skipped = []
    t_start = time.time()

    for i, gene in enumerate(genes_to_perturb):
        if i % 50 == 0:
            logger.info("  Predicting %d/%d genes (%.1fs elapsed)",
                        i, len(genes_to_perturb), time.time() - t_start)
        try:
            mean_expr = predict_perturbation(gears_model, gene)
            mean_predictions.append(mean_expr)
            pert_labels.append(gene)
        except Exception as e:
            logger.warning("Failed on gene %s: %s", gene, e)
            skipped.append(gene)

    elapsed = time.time() - t_start

    if not mean_predictions:
        logger.error("No successful predictions! Skipped: %s", skipped[:20])
        sys.exit(1)

    pred_matrix = np.stack(mean_predictions, axis=0)

    # Use GEARS gene vocabulary for var_names
    if pred_matrix.shape[1] == len(gears_genes):
        var_index = gears_genes
    elif pred_matrix.shape[1] == control.n_vars:
        var_index = list(control.var_names)
    else:
        logger.warning(
            "GEARS output dim %d doesn't match GEARS vocab (%d) or control (%d). "
            "Using positional names.",
            pred_matrix.shape[1], len(gears_genes), control.n_vars,
        )
        var_index = [f"gene_{i}" for i in range(pred_matrix.shape[1])]

    result = ad.AnnData(
        X=pred_matrix.astype(np.float32),
        obs=pd.DataFrame({"perturbation": pert_labels}),
        var=pd.DataFrame(index=var_index),
        uns={
            "model": "gears",
            "output_type": "mean",
            "data_name": data_name,
            "cell_type": cell_type,
            "n_predicted": len(pert_labels),
            "n_skipped": len(skipped),
            "skipped_genes": skipped[:50],
            "inference_time_seconds": elapsed,
            "device": device,
        },
    )
    result.write_h5ad(output_path)
    logger.info(
        "Saved %d predictions to %s (skipped %d, %.1fs total)",
        len(pert_labels), output_path, len(skipped), elapsed,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="GEARS perturbation prediction for CausalCellBench"
    )
    parser.add_argument("--input", required=True, help="Path to control cells .h5ad")
    parser.add_argument("--output", required=True, help="Output predictions .h5ad")
    parser.add_argument("--genes", required=True, help="Comma-separated genes to perturb")
    parser.add_argument("--data-name", default="norman",
                        help="GEARS built-in dataset name (norman, replogle_k562_essential, etc.)")
    parser.add_argument("--data-path", default="./gears_data",
                        help="Directory for GEARS data cache")
    parser.add_argument("--checkpoint", default="",
                        help="Path to pretrained GEARS checkpoint directory")
    parser.add_argument("--train-epochs", type=int, default=20,
                        help="Training epochs if no checkpoint provided")
    parser.add_argument("--cell-type", default="K562")
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detect if omitted)")
    args = parser.parse_args()

    run_gears_predictions(
        input_path=args.input,
        output_path=args.output,
        genes_to_perturb=args.genes.split(","),
        data_name=args.data_name,
        data_path=args.data_path,
        checkpoint_path=args.checkpoint,
        train_epochs=args.train_epochs,
        cell_type=args.cell_type,
        device=args.device,
    )
