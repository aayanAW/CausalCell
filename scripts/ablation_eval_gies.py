"""Per-variant GIES eval for the architectural ablation (#17).

Takes the 6 variant predictions from results/ablation/{variant}_predictions.h5ad,
runs the standard overlay-sampling + GIES pipeline (matching the rest of the
benchmark at K562 N=200, partition_size=20, seed=42), and writes per-variant
F1 against the cached real-data GIES ground truth.

Output: results/ablation/ablation_f1_summary.json

Wall time: ~10-15 min per variant on M4 CPU = ~1-1.5 hours total.

Usage:
    PYTHONPATH=. python3 scripts/ablation_eval_gies.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import anndata as ad
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causalcellbench.causal.gies_runner import run_gies_full
from scripts.run_eval_local import sample_perturbed_cells_overlay

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"
ABLATION_DIR = RES / "ablation"

K562 = DATA / "k562_subset_n200_slim.h5ad"
GENE_LIST = DATA / "eval_genes_n200.txt"
REAL_EDGES_JSON = RES / "gies_cache" / "n200_s42_p20" / "real_data_edges.json"
PERT_COL = "gene"
CTRL = "non-targeting"

SEED = 42
PARTITION_SIZE = 20
N_CELLS_PER_PERT = 100
N_CTRL_FOR_OVERLAY = 200

VARIANTS = [
    "loss_gauss",
    "latent_16",
    "latent_256",
    "decoder_1layer",
    "no_adversarial",
    "layer_norm",
]


def compute_f1(pred: set, gt: set):
    pred = set(pred); gt = set(gt)
    tp = len(pred & gt)
    if tp == 0:
        return 0.0, 0.0, 0.0, 0
    p = tp / max(1, len(pred))
    r = tp / max(1, len(gt))
    return 2 * p * r / (p + r), p, r, tp


def load_pred_means(predictions_path, gene_subset):
    """Load variant predictions h5ad and return {gene: per-perturbation mean (n_genes,)}.

    The ablation script wrote: AnnData(X=pred_mat, obs={"gene": pred_genes}, var=ad.var)
    where each row is one perturbation's mean prediction.
    """
    p = ad.read_h5ad(predictions_path)
    p_var = list(p.var_names) if hasattr(p, 'var_names') else list(p.var.index)
    p_obs_gene = list(p.obs[PERT_COL]) if PERT_COL in p.obs.columns else list(p.obs.iloc[:, 0])
    if hasattr(p.X, "toarray"):
        X = p.X.toarray()
    else:
        X = np.asarray(p.X, dtype=np.float64)
    # Slice to the eval gene_subset, in the same order
    p_var_idx = {g: i for i, g in enumerate(p_var)}
    cols = [p_var_idx[g] for g in gene_subset if g in p_var_idx]
    if len(cols) != len(gene_subset):
        missing = [g for g in gene_subset if g not in p_var_idx]
        print(f"  WARN: {len(missing)} eval genes missing in predictions; first: {missing[:5]}")
    Xs = X[:, cols]
    means = {}
    for i, g in enumerate(p_obs_gene):
        if g in set(gene_subset):
            means[g] = Xs[i].astype(np.float64)
    return means


def main():
    print("=" * 72)
    print("Architectural ablation per-variant GIES eval")
    print("=" * 72)

    real_edges_data = json.load(open(REAL_EDGES_JSON))
    real_edges = set(tuple(e) for e in real_edges_data["edges"])
    print(f"Real-data GIES ground truth: {len(real_edges)} edges (cache n200_s42_p20)")

    adata = ad.read_h5ad(K562)
    adata.var_names = [str(g) for g in adata.var_names]
    gene_names_full = list(adata.var_names)
    print(f"Loaded K562 subset: {adata.shape}")

    # Subset to the same eval gene set used everywhere else
    gene_subset = sorted([line.strip() for line in open(GENE_LIST) if line.strip()])
    var_set = set(gene_names_full)
    gene_subset = [g for g in gene_subset if g in var_set]
    print(f"Eval genes: {len(gene_subset)}")

    # Slice adata to eval genes
    eval_idx = [gene_names_full.index(g) for g in gene_subset]
    X_full = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X_full = X_full.astype(np.float64)
    X_eval = X_full[:, eval_idx]

    # Control matrix at eval scope
    is_ctrl = adata.obs[PERT_COL].values == CTRL
    ctrl_X = X_eval[is_ctrl]
    print(f"Control cells: {ctrl_X.shape[0]} on {len(gene_subset)} eval genes")

    rng = np.random.default_rng(SEED)
    if ctrl_X.shape[0] > N_CTRL_FOR_OVERLAY:
        ctrl_idx = rng.choice(ctrl_X.shape[0], size=N_CTRL_FOR_OVERLAY, replace=False)
        ctrl_sample = ctrl_X[ctrl_idx]
    else:
        ctrl_sample = ctrl_X

    summary = {
        "real_n_edges": len(real_edges),
        "n_eval_genes": len(gene_subset),
        "seed": SEED,
        "partition_size": PARTITION_SIZE,
        "n_cells_per_pert": N_CELLS_PER_PERT,
        "n_ctrl_for_overlay": N_CTRL_FOR_OVERLAY,
    }

    out_path = ABLATION_DIR / "ablation_f1_summary.json"

    for variant in VARIANTS:
        pred_path = ABLATION_DIR / f"{variant}_predictions.h5ad"
        if not pred_path.exists():
            print(f"  [SKIP] {variant}: no predictions at {pred_path}")
            summary[variant] = {"error": "no predictions"}
            continue

        print(f"\n=== Variant: {variant} ===")
        t0 = time.time()
        try:
            pred_means = load_pred_means(pred_path, gene_subset)
            print(f"  loaded predictions for {len(pred_means)} perturbations")

            # Build overlay-sampled expression matrix + intervention labels
            all_X = [ctrl_sample]
            all_interv = ["non-targeting"] * ctrl_sample.shape[0]

            for g_idx, g in enumerate(gene_subset):
                if g not in pred_means:
                    continue
                gene_seed = SEED * 10000 + g_idx
                sampled = sample_perturbed_cells_overlay(
                    ctrl_sample, pred_means[g], N_CELLS_PER_PERT, seed=gene_seed
                )
                all_X.append(sampled)
                all_interv.extend([g] * sampled.shape[0])

            expr = np.vstack(all_X)
            print(f"  expression matrix: {expr.shape}")

            edges = run_gies_full(
                expr,
                interventions=all_interv,
                gene_names=gene_subset,
                seed=SEED,
                partition_size=PARTITION_SIZE,
            )
            edge_set = set(edges)
            f1, p, r, tp = compute_f1(edge_set, real_edges)
            wall = time.time() - t0
            print(f"  edges={len(edge_set)}, F1={f1:.4f}, P={p:.4f}, R={r:.4f}, "
                  f"TP={tp}, time={wall:.1f}s")
            summary[variant] = {
                "n_edges": len(edge_set),
                "f1": f1,
                "precision": p,
                "recall": r,
                "tp": tp,
                "wall_time_s": wall,
            }
        except Exception as exc:
            print(f"  FAILED: {exc}")
            summary[variant] = {"error": str(exc)}

        # Flush after each variant
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
