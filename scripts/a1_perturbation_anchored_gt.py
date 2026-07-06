#!/usr/bin/env python3
"""A1 — Perturbation-anchored ground truth (non-circular).

Pre-registered in prereg_A1_anchored_gt.md.

Truth from interventions, not from an algorithm. On the K562 N=200 panel the 200
perturbed genes ARE the 200 measured genes. For each perturbed gene A and each
measured gene B != A, we test whether B's expression shifts under A-knockout vs
non-targeting control. An edge A->B is declared when the shift passes an
effect-size + FDR threshold. Because A was actually knocked out, A->B is a real
interventional (causal) edge -- GIES never touches the truth, breaking the S2
circularity.

Outputs:
  data/ground_truth/perturbation_anchored_k562_n200.json  (edges @ all grid points + metadata)
  results/hardening/a1_anchored_gt_build.json             (build stats)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SLIM = ROOT / "data" / "k562_subset_n200_slim.h5ad"
GT_OUT = ROOT / "data" / "ground_truth" / "perturbation_anchored_k562_n200.json"
BUILD_OUT = ROOT / "results" / "hardening" / "a1_anchored_gt_build.json"
CTRL = "non-targeting"

# Pre-registered grid; primary operating point declared BEFORE re-scoring.
Q_GRID = [0.01, 0.05]
FC_GRID = [0.25, 0.5, 1.0]
PRIMARY = {"q": 0.05, "fc": 0.5}


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def main():
    BUILD_OUT.parent.mkdir(parents=True, exist_ok=True)
    GT_OUT.parent.mkdir(parents=True, exist_ok=True)

    a = ad.read_h5ad(SLIM)
    genes = a.var["gene_name"].astype(str).values
    gidx = {g: i for i, g in enumerate(genes)}
    labels = a.obs["gene"].astype(str).values
    X = a.X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)

    # Standard scRNA normalization for DE: library-size normalize -> log1p.
    lib = X.sum(axis=1, keepdims=True)
    med = np.median(lib[lib > 0])
    Xn = np.log1p(X / np.maximum(lib, 1e-9) * med)

    ctrl_mask = labels == CTRL
    ctrl = Xn[ctrl_mask]  # (n_ctrl, 200)
    ctrl_mean = ctrl.mean(axis=0)
    pert_genes = [g for g in np.unique(labels) if g != CTRL]

    # For every perturbed gene A: Welch t vs control for all measured B; log2FC.
    # log2FC on normalized-linear means (expm1 back), pseudocount for stability.
    rows = []  # (A, B, log2fc, p, effect)
    ctrl_lin_mean = np.expm1(ctrl_mean)  # linear-scale control mean per gene
    n_tested = 0
    for A in pert_genes:
        pm = labels == A
        if pm.sum() < 3:
            continue
        pert = Xn[pm]  # (n_A, 200)
        # Welch t-test per gene B (vectorized across genes)
        t, p = stats.ttest_ind(pert, ctrl, axis=0, equal_var=False)
        p = np.nan_to_num(p, nan=1.0)
        pert_lin_mean = np.expm1(pert.mean(axis=0))
        log2fc = np.log2((pert_lin_mean + 1e-3) / (ctrl_lin_mean + 1e-3))
        fdr = bh_fdr(p)
        ai = gidx[A]
        for bi, B in enumerate(genes):
            if bi == ai:
                continue  # no self-edges
            n_tested += 1
            rows.append((A, B, float(log2fc[bi]), float(fdr[bi]), float(t[bi])))

    rows_arr = rows
    # Build edge sets at every grid point.
    edge_sets = {}
    for q in Q_GRID:
        for fc in FC_GRID:
            key = f"q{q}_fc{fc}"
            edges = [
                [A, B]
                for (A, B, l2, f, t) in rows_arr
                if f <= q and abs(l2) >= fc
            ]
            edge_sets[key] = edges

    primary_key = f"q{PRIMARY['q']}_fc{PRIMARY['fc']}"
    primary_edges = edge_sets[primary_key]

    meta = {
        "n_measured_genes": int(len(genes)),
        "n_perturbed_genes": int(len(pert_genes)),
        "n_control_cells": int(ctrl_mask.sum()),
        "n_directed_pairs_tested": int(n_tested),
        "q_grid": Q_GRID,
        "fc_grid": FC_GRID,
        "primary_operating_point": PRIMARY,
        "primary_key": primary_key,
        "edge_counts": {k: len(v) for k, v in edge_sets.items()},
        "primary_n_edges": len(primary_edges),
        "normalization": "library-size (median) + log1p; Welch t on log; log2FC on expm1 means (+1e-3)",
        "genes": list(map(str, genes)),
    }

    with open(GT_OUT, "w") as fh:
        json.dump({"meta": meta, "edge_sets": edge_sets}, fh)
    with open(BUILD_OUT, "w") as fh:
        json.dump(meta, fh, indent=2)

    # console: only the deciding stats
    print(json.dumps({
        "primary_key": primary_key,
        "primary_n_edges": len(primary_edges),
        "edge_counts": meta["edge_counts"],
        "n_pairs_tested": n_tested,
    }))


if __name__ == "__main__":
    main()
