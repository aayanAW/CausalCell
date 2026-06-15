"""CPA (Compositional Perturbation Autoencoder) wrapper.

CPA (Lotfollahi et al., MSB 2023) is a conditional VAE that disentangles
basal cell state from perturbation effects in latent space. The model learns
perturbation embeddings that are combined additively with the basal latent
representation, then decoded to predict post-perturbation gene expression.

Key architectural insight: CPA factorizes the latent space as
    z = z_basal + z_pert + z_covariate
and decodes to predict gene expression:
    x_hat = decoder(z)

For counterfactual prediction, control cells are encoded to z_basal,
the target perturbation embedding is added, and the result is decoded.

Package: cpa-tools (built on scvi-tools)
API reference: cpa.CPA.setup_anndata(), model.train(), model.predict()

This wrapper runs in an ISOLATED conda environment (env_cpa).

Installation:
    conda create -n env_cpa python=3.10
    conda activate env_cpa
    pip install cpa-tools anndata scanpy

Usage:
    conda run -n env_cpa python -m causalcellbench.models.cpa_wrapper \\
        --input data/k562_real.h5ad \\
        --output results/cpa_predictions.h5ad \\
        --genes TP53,MDM2,... \\
        --checkpoint /path/to/cpa/model
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import guard
# ---------------------------------------------------------------------------

_CPA_AVAILABLE = None


def _check_cpa_available() -> bool:
    """Check if cpa-tools is importable."""
    global _CPA_AVAILABLE
    if _CPA_AVAILABLE is None:
        try:
            import cpa  # noqa: F401

            _CPA_AVAILABLE = True
        except ImportError:
            _CPA_AVAILABLE = False
    return _CPA_AVAILABLE


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_adata_for_cpa(
    adata: "anndata.AnnData",
    perturbation_key: str = "gene",
    control_label: str = "non-targeting",
    dosage_key: Optional[str] = None,
    batch_key: Optional[str] = None,
    covariate_keys: Optional[list[str]] = None,
    is_count_data: bool = True,
    max_comb_len: int = 1,
    copy: bool = True,
) -> "anndata.AnnData":
    """Prepare an AnnData for CPA: add required obs columns and register.

    CPA's ``setup_anndata`` requires:
      - A perturbation column in ``.obs`` identifying the perturbation condition
      - A control group identifier
      - Optionally: dosage, batch, and categorical covariate columns

    For CRISPRi Perturb-seq data (Replogle et al.), the perturbation
    column contains the targeted gene name, and the control label is
    ``"non-targeting"`` (or ``"ctrl"`` depending on preprocessing).

    Parameters
    ----------
    adata
        AnnData with raw or normalized expression in ``.X``.
    perturbation_key
        Column in ``.obs`` identifying the perturbation (e.g., ``"gene"``).
    control_label
        Value in ``perturbation_key`` column that marks control cells.
    dosage_key
        Column for perturbation dosage. If None, a dummy column
        ``"cpa_dosage"`` is created with 1.0 for perturbed, 0.0 for control.
    batch_key
        Column for batch/experiment information.
    covariate_keys
        List of categorical covariate column names.
    is_count_data
        Whether ``.X`` contains raw counts (affects reconstruction loss).
    max_comb_len
        Maximum combinatorial perturbation length (1 for single-gene CRISPRi).
    copy
        If True, operate on a copy of adata.

    Returns
    -------
    AnnData registered with CPA's setup_anndata.
    """
    import anndata as ad

    if not _check_cpa_available():
        raise ImportError(
            "cpa-tools is not installed. Install with: pip install cpa-tools"
        )
    import cpa

    if copy:
        adata = adata.copy()

    # Validate perturbation key exists
    if perturbation_key not in adata.obs.columns:
        raise ValueError(
            f"Perturbation key '{perturbation_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    # Ensure perturbation column is string type
    adata.obs[perturbation_key] = adata.obs[perturbation_key].astype(str)

    # Validate control_label exists in the perturbation column
    unique_perts = set(adata.obs[perturbation_key].unique())
    if control_label not in unique_perts:
        raise ValueError(
            f"Control label '{control_label}' not found in "
            f"adata.obs['{perturbation_key}']. "
            f"Found labels (first 10): {sorted(unique_perts)[:10]}"
        )

    # Create dosage column if not provided or if specified key is missing
    if dosage_key is None:
        dosage_key = "cpa_dosage"
    if dosage_key not in adata.obs.columns:
        adata.obs[dosage_key] = np.where(
            adata.obs[perturbation_key] == control_label, 0.0, 1.0
        ).astype(np.float32)
        logger.info(
            "Created dummy dosage column '%s' (1.0 for perturbed, 0.0 for control)",
            dosage_key,
        )

    # Ensure covariate columns are categorical
    if covariate_keys:
        for key in covariate_keys:
            if key in adata.obs.columns:
                adata.obs[key] = adata.obs[key].astype("category")

    # CPA setup_anndata -- handles both old and new API parameter names.
    # Newer cpa-tools (>=0.8) uses perturbation_key/control_group/dosage_key.
    # Older versions use drug_key/control_key/dose_key.
    try:
        # Try newer API first (cpa-tools >= 0.8)
        cpa.CPA.setup_anndata(
            adata,
            perturbation_key=perturbation_key,
            control_group=control_label,
            dosage_key=dosage_key,
            batch_key=batch_key,
            categorical_covariate_keys=covariate_keys or [],
            is_count_data=is_count_data,
            max_comb_len=max_comb_len,
        )
    except TypeError:
        # Fall back to older API (drug_key/dose_key/control_key)
        logger.warning("Falling back to older cpa-tools API (drug_key/dose_key)")
        cpa.CPA.setup_anndata(
            adata,
            drug_key=perturbation_key,
            dose_key=dosage_key,
            control_key=control_label,
            categorical_covariate_keys=covariate_keys or [],
        )

    logger.info(
        "CPA setup_anndata complete: %d cells, %d genes, "
        "%d perturbations (control='%s')",
        adata.n_obs,
        adata.n_vars,
        len(unique_perts) - 1,  # exclude control
        control_label,
    )
    return adata


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_cpa(
    adata: "anndata.AnnData",
    perturbation_key: str = "gene",
    control_label: str = "non-targeting",
    dosage_key: Optional[str] = None,
    batch_key: Optional[str] = None,
    covariate_keys: Optional[list[str]] = None,
    # Model architecture
    n_latent: int = 64,
    recon_loss: str = "nb",
    variational: bool = True,
    n_hidden_encoder: int = 256,
    n_layers_encoder: int = 3,
    n_hidden_decoder: int = 256,
    n_layers_decoder: int = 2,
    dropout_rate_encoder: float = 0.1,
    dropout_rate_decoder: float = 0.0,
    doser_type: str = "linear",
    # Training
    max_epochs: int = 300,
    batch_size: int = 512,
    early_stopping_patience: int = 15,
    check_val_every_n_epoch: int = 5,
    lr: float = 1e-4,
    # Split
    split_key: Optional[str] = None,
    train_split: str = "train",
    valid_split: str = "valid",
    test_split: str = "ood",
    holdout_fraction: float = 0.15,
    # Save
    save_path: Optional[str] = None,
    seed: int = 0,
    use_gpu: Optional[Union[str, int, bool]] = None,
) -> "cpa.CPA":
    """Train a CPA model on perturbation data.

    This function handles the full pipeline: data preparation, model
    instantiation, and training with early stopping.

    Parameters
    ----------
    adata
        AnnData with expression in ``.X`` and perturbation info in ``.obs``.
    perturbation_key
        Column in ``.obs`` identifying the perturbation.
    control_label
        Value marking control (unperturbed) cells.
    dosage_key
        Column for perturbation dosage (None creates dummy column).
    batch_key
        Column for batch correction.
    covariate_keys
        Categorical covariate columns.
    n_latent
        Latent space dimensionality.
    recon_loss
        Reconstruction loss: ``"nb"`` (negative binomial, for counts),
        ``"zinb"`` (zero-inflated NB), or ``"gauss"`` (Gaussian, for
        normalized data).
    variational
        Whether to use variational inference (True = VAE with KL,
        False = deterministic AE). VAE enables cell-level sampling.
    n_hidden_encoder, n_layers_encoder
        Encoder architecture.
    n_hidden_decoder, n_layers_decoder
        Decoder architecture.
    dropout_rate_encoder, dropout_rate_decoder
        Dropout rates.
    doser_type
        Dose-response function type: ``"linear"``, ``"logsigm"``, ``"sigm"``.
    max_epochs
        Maximum training epochs.
    batch_size
        Training batch size.
    early_stopping_patience
        Stop after this many epochs without validation improvement.
    check_val_every_n_epoch
        Validation frequency.
    lr
        Learning rate.
    split_key
        Column in ``.obs`` with train/valid/ood splits. If None, a random
        split is created using ``holdout_fraction``.
    train_split, valid_split, test_split
        Values in ``split_key`` for each data split.
    holdout_fraction
        Fraction of cells for validation if ``split_key`` is None.
    save_path
        Directory to save model checkpoint and training logs.
    seed
        Random seed.
    use_gpu
        GPU device specification (True/False/device_id).

    Returns
    -------
    Trained cpa.CPA model.
    """
    if not _check_cpa_available():
        raise ImportError(
            "cpa-tools is not installed. Install with: pip install cpa-tools"
        )
    import cpa

    # Auto-detect GPU
    if use_gpu is None:
        try:
            import torch

            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False

    # Create train/valid split if not provided
    if split_key is None or split_key not in adata.obs.columns:
        rng = np.random.default_rng(seed)
        split_key = "cpa_split"
        n = adata.n_obs
        assignments = rng.choice(
            [train_split, valid_split],
            size=n,
            p=[1 - holdout_fraction, holdout_fraction],
        )
        adata.obs[split_key] = assignments
        logger.info(
            "Created random split: %d train, %d valid",
            (assignments == train_split).sum(),
            (assignments == valid_split).sum(),
        )

    # Prepare adata (register with CPA)
    adata = prepare_adata_for_cpa(
        adata,
        perturbation_key=perturbation_key,
        control_label=control_label,
        dosage_key=dosage_key,
        batch_key=batch_key,
        covariate_keys=covariate_keys,
        is_count_data=(recon_loss in ("nb", "zinb")),
        copy=False,  # Already working on it
    )

    # Model instantiation
    model = cpa.CPA(
        adata=adata,
        split_key=split_key,
        train_split=train_split,
        valid_split=valid_split,
        test_split=test_split,
        n_latent=n_latent,
        recon_loss=recon_loss,
        variational=variational,
        n_hidden_encoder=n_hidden_encoder,
        n_layers_encoder=n_layers_encoder,
        n_hidden_decoder=n_hidden_decoder,
        n_layers_decoder=n_layers_decoder,
        dropout_rate_encoder=dropout_rate_encoder,
        dropout_rate_decoder=dropout_rate_decoder,
        doser_type=doser_type,
        seed=seed,
    )
    logger.info(
        "CPA model created: n_latent=%d, recon_loss='%s', variational=%s",
        n_latent,
        recon_loss,
        variational,
    )

    # Trainer kwargs for optimization schedule
    plan_kwargs = {
        "lr": lr,
        "n_epochs_kl_warmup": 20 if variational else None,
        "n_epochs_adv_warmup": 50,
        "n_epochs_pretrain_ae": 10,
    }

    # Train
    t_start = time.time()
    model.train(
        max_epochs=max_epochs,
        use_gpu=use_gpu,
        batch_size=batch_size,
        plan_kwargs=plan_kwargs,
        early_stopping_patience=early_stopping_patience,
        check_val_every_n_epoch=check_val_every_n_epoch,
        save_path=save_path,
    )
    t_train = time.time() - t_start
    logger.info("Training complete in %.1f seconds (%.1f min)", t_train, t_train / 60)

    # Save model
    if save_path:
        save_dir = Path(save_path) / "cpa_model"
        model.save(str(save_dir), overwrite=True, save_anndata=True)
        logger.info("Model saved to %s", save_dir)

    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_perturbation(
    model: "cpa.CPA",
    adata: "anndata.AnnData",
    target_perturbation: str,
    perturbation_key: str = "gene",
    control_label: str = "non-targeting",
    dosage_key: Optional[str] = None,
    n_samples: int = 20,
    batch_size: int = 256,
    return_mean: bool = True,
) -> np.ndarray:
    """Predict expression under a specific perturbation using a trained CPA model.

    The counterfactual prediction workflow:
    1. Select control cells from adata
    2. Copy them and overwrite the perturbation annotation with the target
    3. Set dosage to 1.0 for the target perturbation
    4. Call model.predict() which encodes control cells, adds the target
       perturbation embedding, and decodes to predict expression

    Parameters
    ----------
    model
        Trained cpa.CPA model instance.
    adata
        AnnData registered with CPA (same schema as training data).
        Must contain control cells.
    target_perturbation
        Name of the perturbation to predict (must be in the model's
        perturbation vocabulary from training).
    perturbation_key
        Column in ``.obs`` with perturbation labels.
    control_label
        Value marking control cells.
    dosage_key
        Column for dosage values. If None, uses ``"cpa_dosage"``.
    n_samples
        Number of posterior samples (only meaningful when model is
        variational). More samples = smoother mean estimate.
    batch_size
        Batch size for inference.
    return_mean
        If True, average over posterior samples. If False, return
        all samples (shape: n_samples x n_cells x n_genes).

    Returns
    -------
    np.ndarray
        Predicted expression matrix. Shape ``(n_control_cells, n_genes)``
        if ``return_mean=True``, else ``(n_samples, n_control_cells, n_genes)``.
    """
    import anndata as ad

    if dosage_key is None:
        dosage_key = "cpa_dosage"

    # Select control cells
    control_mask = adata.obs[perturbation_key].astype(str) == control_label
    if control_mask.sum() == 0:
        raise ValueError(
            f"No control cells found (perturbation_key='{perturbation_key}', "
            f"control_label='{control_label}')"
        )

    # Create counterfactual adata: control cells with target perturbation label
    cf_adata = adata[control_mask].copy()
    cf_adata.obs[perturbation_key] = target_perturbation

    # Set dosage to 1.0 for the target perturbation
    if dosage_key in cf_adata.obs.columns:
        cf_adata.obs[dosage_key] = 1.0
    else:
        cf_adata.obs[dosage_key] = 1.0

    # CPA predict stores results in adata.obsm['CPA_pred']
    model.predict(
        adata=cf_adata,
        batch_size=batch_size,
        n_samples=n_samples,
        return_mean=return_mean,
    )

    pred_key = "CPA_pred"
    if pred_key not in cf_adata.obsm:
        raise RuntimeError(
            f"CPA predict did not populate adata.obsm['{pred_key}']. "
            f"Available keys: {list(cf_adata.obsm.keys())}"
        )

    return np.asarray(cf_adata.obsm[pred_key], dtype=np.float32)


def predict_perturbation_batch(
    model: "cpa.CPA",
    adata: "anndata.AnnData",
    target_perturbations: list[str],
    perturbation_key: str = "gene",
    control_label: str = "non-targeting",
    dosage_key: Optional[str] = None,
    n_cells: int = 100,
    n_samples: int = 20,
    batch_size: int = 256,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Predict expression for multiple perturbations.

    For each target perturbation, selects ``n_cells`` control cells (with
    replacement if needed), applies the counterfactual perturbation label,
    and runs CPA prediction.

    Parameters
    ----------
    model
        Trained cpa.CPA model.
    adata
        AnnData registered with CPA, containing control cells.
    target_perturbations
        List of perturbation names to predict.
    perturbation_key
        Perturbation column in ``.obs``.
    control_label
        Control cell identifier.
    dosage_key
        Dosage column name.
    n_cells
        Number of cells to predict per perturbation.
    n_samples
        Posterior samples for variational models.
    batch_size
        Inference batch size.
    seed
        Random seed for control cell subsampling.

    Returns
    -------
    Dict mapping perturbation name to expression matrix (n_cells, n_genes).
    """
    import anndata as ad

    if dosage_key is None:
        dosage_key = "cpa_dosage"

    # Get available perturbations from model's encoder
    available_perts = set()
    if hasattr(model, "pert_encoder"):
        available_perts = set(model.pert_encoder.keys())
        logger.info("Model knows %d perturbations", len(available_perts))

    # Select control cells
    control_mask = adata.obs[perturbation_key].astype(str) == control_label
    n_control = int(control_mask.sum())
    if n_control == 0:
        raise ValueError(f"No control cells found for label '{control_label}'")

    control_adata = adata[control_mask].copy()

    results = {}
    rng = np.random.default_rng(seed)
    t_start = time.time()

    for i, pert in enumerate(target_perturbations):
        if i % 50 == 0 and i > 0:
            logger.info(
                "  %d/%d perturbations (%.1fs)",
                i,
                len(target_perturbations),
                time.time() - t_start,
            )

        # Check if perturbation is in model vocabulary
        if available_perts and pert not in available_perts:
            logger.warning(
                "Perturbation '%s' not in model vocabulary, skipping", pert
            )
            continue

        try:
            # Subsample control cells
            if n_control >= n_cells:
                idx = rng.choice(n_control, size=n_cells, replace=False)
            else:
                idx = rng.choice(n_control, size=n_cells, replace=True)

            sub_adata = control_adata[idx].copy()

            # Set counterfactual perturbation
            sub_adata.obs[perturbation_key] = pert
            if dosage_key in sub_adata.obs.columns:
                sub_adata.obs[dosage_key] = 1.0
            else:
                sub_adata.obs[dosage_key] = 1.0

            # Predict
            model.predict(
                adata=sub_adata,
                batch_size=batch_size,
                n_samples=n_samples,
                return_mean=True,
            )

            pred_key = "CPA_pred"
            if pred_key in sub_adata.obsm:
                expr = np.asarray(sub_adata.obsm[pred_key], dtype=np.float32)
            else:
                raise RuntimeError(f"CPA_pred not found in obsm for '{pert}'")

            results[pert] = expr

        except Exception as e:
            logger.warning("Failed on perturbation '%s': %s", pert, e)
            continue

    elapsed = time.time() - t_start
    logger.info(
        "Predicted %d/%d perturbations in %.1fs",
        len(results),
        len(target_perturbations),
        elapsed,
    )
    return results


# ---------------------------------------------------------------------------
# Gene name mapping utilities
# ---------------------------------------------------------------------------


def map_ensembl_to_symbols(
    adata: "anndata.AnnData",
    gene_name_col: str = "gene_name",
) -> "anndata.AnnData":
    """Convert Ensembl ID var_names to gene symbols if mapping is available.

    Many Perturb-seq datasets store Ensembl IDs as var_names with a
    ``gene_name`` column in ``.var`` containing the symbol. CPA's
    perturbation vocabulary uses gene symbols, so we need to align.

    Parameters
    ----------
    adata
        AnnData whose ``.var_names`` may be Ensembl IDs.
    gene_name_col
        Column in ``.var`` with gene symbols.

    Returns
    -------
    AnnData with gene symbols as var_names (copy if converted).
    """
    if gene_name_col not in adata.var.columns:
        logger.info(
            "No '%s' column in adata.var; assuming var_names are already symbols",
            gene_name_col,
        )
        return adata

    # Check if var_names look like Ensembl IDs
    sample_names = list(adata.var_names[:5])
    looks_like_ensembl = any(
        n.startswith("ENSG") or n.startswith("ENSMUSG") for n in sample_names
    )

    if not looks_like_ensembl:
        logger.info("var_names don't look like Ensembl IDs, keeping as-is")
        return adata

    adata = adata.copy()
    symbols = adata.var[gene_name_col].values

    # Handle duplicates: append suffix to duplicate gene symbols
    seen: dict[str, int] = {}
    unique_symbols = []
    for s in symbols:
        s = str(s)
        if s in seen:
            seen[s] += 1
            unique_symbols.append(f"{s}_{seen[s]}")
        else:
            seen[s] = 0
            unique_symbols.append(s)

    # Store original Ensembl IDs
    adata.var["ensembl_id"] = adata.var_names.copy()
    adata.var_names = unique_symbols
    adata.var_names_make_unique()

    n_converted = sum(1 for a, b in zip(sample_names, unique_symbols) if a != b)
    logger.info(
        "Converted var_names from Ensembl IDs to gene symbols "
        "(%d genes, %d had duplicates)",
        len(unique_symbols),
        sum(1 for v in seen.values() if v > 0),
    )
    return adata


# ---------------------------------------------------------------------------
# High-level wrapper conforming to AbstractPerturbationModel
# ---------------------------------------------------------------------------


class CPAModel:
    """CPA wrapper implementing the CausalCellBench model interface.

    This class wraps a trained CPA model and provides the standard
    predict_perturbation() interface that the benchmark pipeline expects.

    Attributes
    ----------
    model : cpa.CPA
        The underlying CPA model.
    perturbation_key : str
        Column name for perturbation labels.
    control_label : str
        Identifier for control cells.
    dosage_key : str
        Column name for dosage values.
    """

    def __init__(
        self,
        perturbation_key: str = "gene",
        control_label: str = "non-targeting",
        dosage_key: str = "cpa_dosage",
    ):
        self.model: Optional["cpa.CPA"] = None
        self.perturbation_key = perturbation_key
        self.control_label = control_label
        self.dosage_key = dosage_key
        self._adata_train: Optional["anndata.AnnData"] = None

    @property
    def name(self) -> str:
        return "CPA"

    @property
    def supports_multi_perturbation(self) -> bool:
        """CPA natively supports multi-gene perturbation via additive
        perturbation embeddings in latent space."""
        return True

    def load_model(
        self,
        checkpoint_path: str,
        adata_train: Optional["anndata.AnnData"] = None,
    ) -> None:
        """Load a pretrained CPA model from checkpoint.

        Parameters
        ----------
        checkpoint_path
            Directory containing saved CPA model (from model.save()).
        adata_train
            Optional AnnData for models that need the training data schema.
            CPA's load() can reload the saved anndata if it was saved with
            ``save_anndata=True``.
        """
        if not _check_cpa_available():
            raise ImportError("cpa-tools is not installed")
        import cpa

        try:
            import torch

            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False

        self.model = cpa.CPA.load(
            checkpoint_path,
            adata=adata_train,
            use_gpu=use_gpu,
        )
        self._adata_train = adata_train
        logger.info("Loaded CPA model from %s", checkpoint_path)

    def train(
        self,
        adata: "anndata.AnnData",
        **kwargs,
    ) -> None:
        """Train CPA on the provided data. Stores the model internally.

        Parameters
        ----------
        adata
            AnnData with perturbation data.
        **kwargs
            Forwarded to :func:`train_cpa`.
        """
        kwargs.setdefault("perturbation_key", self.perturbation_key)
        kwargs.setdefault("control_label", self.control_label)
        kwargs.setdefault("dosage_key", self.dosage_key)

        self.model = train_cpa(adata, **kwargs)
        self._adata_train = adata

    def predict_perturbation(
        self,
        control_adata: "anndata.AnnData",
        perturbed_gene: str,
        n_cells: int = 100,
        gene_names: Optional[list[str]] = None,
        seed: int = 0,
        n_samples: int = 20,
    ) -> np.ndarray:
        """Predict single-gene perturbation effect.

        Parameters
        ----------
        control_adata
            AnnData of control cells registered with CPA.
        perturbed_gene
            Gene to perturb (CRISPRi knockdown).
        n_cells
            Number of simulated cells to generate.
        gene_names
            Gene names for the output (if None, uses control_adata.var_names).
        seed
            Random seed for cell subsampling.
        n_samples
            Number of posterior samples for variational models.

        Returns
        -------
        np.ndarray
            Predicted expression matrix of shape ``(n_cells, n_genes)``.
        """
        if self.model is None:
            raise RuntimeError(
                "No model loaded. Call load_model() or train() first."
            )

        rng = np.random.default_rng(seed)

        # Get control cells
        control_mask = (
            control_adata.obs[self.perturbation_key].astype(str)
            == self.control_label
        )
        n_control = int(control_mask.sum())
        if n_control == 0:
            raise ValueError(
                f"No control cells with label '{self.control_label}'"
            )

        # Subsample control cells
        control_idx = np.where(control_mask)[0]
        if n_control >= n_cells:
            sel = rng.choice(control_idx, size=n_cells, replace=False)
        else:
            sel = rng.choice(control_idx, size=n_cells, replace=True)

        # Create counterfactual adata
        cf_adata = control_adata[sel].copy()
        cf_adata.obs[self.perturbation_key] = perturbed_gene
        if self.dosage_key in cf_adata.obs.columns:
            cf_adata.obs[self.dosage_key] = 1.0
        else:
            cf_adata.obs[self.dosage_key] = 1.0

        # Run CPA prediction
        self.model.predict(
            adata=cf_adata,
            batch_size=min(256, n_cells),
            n_samples=n_samples,
            return_mean=True,
        )

        pred_key = "CPA_pred"
        if pred_key not in cf_adata.obsm:
            raise RuntimeError(
                f"CPA did not produce predictions in obsm['{pred_key}']. "
                f"Keys: {list(cf_adata.obsm.keys())}"
            )

        expr = np.asarray(cf_adata.obsm[pred_key], dtype=np.float32)

        # Align genes if requested
        if gene_names is not None and gene_names != list(
            control_adata.var_names
        ):
            var_to_idx = {
                n: i for i, n in enumerate(control_adata.var_names)
            }
            col_idx = [var_to_idx[g] for g in gene_names if g in var_to_idx]
            expr = expr[:, col_idx]

        return expr

    def predict_multi_perturbation(
        self,
        control_adata: "anndata.AnnData",
        perturbed_genes: list[str],
        n_cells: int = 100,
        gene_names: Optional[list[str]] = None,
        seed: int = 0,
        n_samples: int = 20,
    ) -> np.ndarray:
        """Predict multi-gene perturbation (CPA native capability).

        CPA represents multi-perturbation as a combination of perturbation
        embeddings (typically additive). The perturbation string is formed
        by joining gene names with '+'.

        Parameters
        ----------
        perturbed_genes
            List of genes to perturb simultaneously.
        (other params as in predict_perturbation)

        Returns
        -------
        np.ndarray of shape (n_cells, n_genes).
        """
        # CPA encodes multi-perturbation as "gene1+gene2" in the pert column
        combo_label = "+".join(sorted(perturbed_genes))
        return self.predict_perturbation(
            control_adata=control_adata,
            perturbed_gene=combo_label,
            n_cells=n_cells,
            gene_names=gene_names,
            seed=seed,
            n_samples=n_samples,
        )


# ---------------------------------------------------------------------------
# Standalone CLI runner
# ---------------------------------------------------------------------------


def run_cpa_predictions(
    input_path: str,
    output_path: str,
    genes_to_perturb: list[str],
    checkpoint_path: str,
    perturbation_key: str = "gene",
    control_label: str = "non-targeting",
    n_cells_per_pert: int = 100,
    cell_type: str = "K562",
    seed: int = 0,
) -> None:
    """Run CPA perturbation predictions with VAE sampling.

    CPA outputs CELL-LEVEL predictions (not just means) because it
    samples from the VAE posterior. The output .h5ad has output_type='cells'.
    """
    import anndata as ad
    import pandas as pd

    if not _check_cpa_available():
        logger.error("cpa-tools not installed. Run in env_cpa.")
        sys.exit(1)

    adata = ad.read_h5ad(input_path)
    logger.info("Loaded: %d cells, %d genes", adata.n_obs, adata.n_vars)

    # Handle Ensembl -> symbol mapping if needed
    adata = map_ensembl_to_symbols(adata)

    # Initialize wrapper and load model
    wrapper = CPAModel(
        perturbation_key=perturbation_key,
        control_label=control_label,
    )
    wrapper.load_model(checkpoint_path, adata_train=adata)

    # Filter to genes that are in the model vocabulary
    available_perts = set()
    if hasattr(wrapper.model, "pert_encoder"):
        available_perts = set(wrapper.model.pert_encoder.keys())

    if available_perts:
        valid_genes = [g for g in genes_to_perturb if g in available_perts]
        skipped = [g for g in genes_to_perturb if g not in available_perts]
        if skipped:
            logger.warning(
                "Skipping %d genes not in CPA vocabulary: %s",
                len(skipped),
                skipped[:10],
            )
    else:
        valid_genes = genes_to_perturb
        logger.warning(
            "Could not check perturbation vocabulary; attempting all genes"
        )

    # Run predictions
    all_cells = []
    all_labels = []
    t_start = time.time()

    for i, gene in enumerate(valid_genes):
        if i % 50 == 0:
            logger.info(
                "  %d/%d genes (%.1fs)", i, len(valid_genes), time.time() - t_start
            )

        try:
            expr = wrapper.predict_perturbation(
                control_adata=adata,
                perturbed_gene=gene,
                n_cells=n_cells_per_pert,
                seed=seed + i,
            )
            all_cells.append(expr)
            all_labels.extend([gene] * expr.shape[0])
        except Exception as e:
            logger.warning("Failed on gene %s: %s", gene, e)
            continue

    if not all_cells:
        logger.error("No successful predictions!")
        sys.exit(1)

    pred_matrix = np.concatenate(all_cells, axis=0)

    result = ad.AnnData(
        X=pred_matrix.astype(np.float32),
        obs=pd.DataFrame({"perturbation": all_labels}),
        var=adata.var.copy() if pred_matrix.shape[1] == adata.n_vars else pd.DataFrame(
            index=[f"gene_{i}" for i in range(pred_matrix.shape[1])]
        ),
        uns={
            "model": "cpa",
            "output_type": "cells",
            "checkpoint": str(checkpoint_path),
            "cell_type": cell_type,
            "n_cells_per_pert": n_cells_per_pert,
            "n_perturbations": len(set(all_labels)),
            "inference_time_seconds": time.time() - t_start,
        },
    )
    result.write_h5ad(output_path)
    logger.info(
        "Saved %d cells (%d perturbations) to %s",
        pred_matrix.shape[0],
        len(set(all_labels)),
        output_path,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="CPA perturbation predictions")
    parser.add_argument("--input", required=True, help="Input AnnData .h5ad")
    parser.add_argument("--output", required=True, help="Output predictions .h5ad")
    parser.add_argument("--genes", required=True, help="Comma-separated genes")
    parser.add_argument("--checkpoint", required=True, help="CPA model directory")
    parser.add_argument(
        "--perturbation-key", default="gene", help="Perturbation obs column"
    )
    parser.add_argument(
        "--control-label", default="non-targeting", help="Control cell label"
    )
    parser.add_argument("--n-cells", type=int, default=100)
    parser.add_argument("--cell-type", default="K562")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_cpa_predictions(
        input_path=args.input,
        output_path=args.output,
        genes_to_perturb=args.genes.split(","),
        checkpoint_path=args.checkpoint,
        perturbation_key=args.perturbation_key,
        control_label=args.control_label,
        n_cells_per_pert=args.n_cells,
        cell_type=args.cell_type,
        seed=args.seed,
    )
