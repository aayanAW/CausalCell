"""Phase 1 — distribution diagnostics: mode collapse vs. inflation reconciliation.

The literature (Mejia et al. 2025; Ahlmann-Eltze & Huber 2025) reports that VCMs
exhibit MODE COLLAPSE — predictions revert toward the dataset mean response,
compressing per-perturbation variance.

Phase 1 of the original CausalCellBench reported INFLATION — predictions exceed
|log2FC|>0.5 in 63% of (perturbation, gene) entries vs. 2.7% in real biology,
a ~23x ratio.

These two observations CAN coexist if models predict an identical, dense
response pattern across all perturbations: high entropy across genes (looks
like inflation when pooled across (pert, gene) entries) and low entropy across
perturbations (looks like mode collapse).

This script tests that hypothesis directly. Outputs:
  results/phase1/distribution_reconciliation.json  (numbers)
  results/phase1/distribution_reconciliation.csv   (per-perturbation rows)
  results/phase1/distribution_reconciliation.txt   (verdict)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "results" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)

K562 = DATA / "k562_subset_n200_slim.h5ad"
MODELS = ["gears", "cpa", "geneformer"]
PRED_DIR = DATA / "model_outputs_real_n200"
PERT_COL = "gene"
CTRL_LABEL = "non-targeting"
PSEUDO = 0.01
MIN_CELLS = 5

print("=" * 72)
print("Phase 1 — Distribution diagnostics: mode collapse vs. inflation")
print("=" * 72)

# ----------------------------------------------------------------------
# 1. Load real data, build per-perturbation real means + control means
# ----------------------------------------------------------------------
print("Loading real K562 N=200 subset ...")
adata = ad.read_h5ad(K562)
adata.var_names = [str(g) for g in adata.var_names]
gene_names = list(adata.var_names)
n_genes = len(gene_names)
gene_to_idx = {g: i for i, g in enumerate(gene_names)}

X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
X = X.astype(np.float64)

is_ctrl = adata.obs[PERT_COL].values == CTRL_LABEL
ctrl_means = X[is_ctrl].mean(axis=0)

real_pert_means: dict[str, np.ndarray] = {}
real_pert_vars: dict[str, np.ndarray] = {}
real_pert_ncells: dict[str, int] = {}
for gene in sorted(set(adata.obs[PERT_COL].values)):
    if gene == CTRL_LABEL:
        continue
    if gene not in gene_to_idx:
        continue
    mask = adata.obs[PERT_COL].values == gene
    if mask.sum() < MIN_CELLS:
        continue
    cells = X[mask]
    real_pert_means[gene] = cells.mean(axis=0)
    real_pert_vars[gene] = cells.var(axis=0, ddof=1)
    real_pert_ncells[gene] = int(mask.sum())

print(f"  perturbations with >={MIN_CELLS} cells: {len(real_pert_means)}")
common_perts = sorted(real_pert_means.keys())

# ----------------------------------------------------------------------
# 2. For each model, build aligned per-perturbation predicted means
# ----------------------------------------------------------------------
def load_model_means(model: str) -> dict[str, np.ndarray]:
    p = PRED_DIR / f"{model}_predictions.h5ad"
    if not p.exists():
        return {}
    pred = ad.read_h5ad(p)
    pred_genes = list(pred.var_names)
    pred_to_idx = {g: i for i, g in enumerate(pred_genes)}
    cols = [pred_to_idx[g] for g in gene_names if g in pred_to_idx]
    pmat = pred.X.toarray() if hasattr(pred.X, "toarray") else np.asarray(pred.X)
    pmat = np.asarray(pmat[:, cols], dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for i in range(pmat.shape[0]):
        gene = str(pred.obs[PERT_COL].iloc[i]) if PERT_COL in pred.obs.columns else \
               str(pred.obs.iloc[i, 0])
        if gene == CTRL_LABEL:
            continue
        out[gene] = pmat[i]
    return out


model_means: dict[str, dict[str, np.ndarray]] = {}
for m in MODELS:
    mm = load_model_means(m)
    common = [g for g in common_perts if g in mm]
    print(f"  {m}: {len(mm)} predicted perturbations, {len(common)} aligned with real")
    model_means[m] = {g: mm[g] for g in common}

# Restrict to perturbations every model has
common_perts = sorted(set(common_perts) & set.intersection(*[set(model_means[m].keys()) for m in MODELS]))
print(f"  shared across all models: {len(common_perts)}")

# ----------------------------------------------------------------------
# 3. Diagnostic 1: Per-perturbation Δ-variance ratio (Var_pred / Var_real)
#    Mode collapse predicts ratios << 1.
#    Inflation predicts ratios >> 1.
# ----------------------------------------------------------------------
print("\nDiagnostic 1: per-perturbation Δ-variance ratio")

def per_pert_delta_var_ratio(model: str) -> np.ndarray:
    """For each perturbation p, ratio of Var(predicted shift) / Var(real shift),
    where the shift is computed across the n_genes gene axis."""
    ratios: list[float] = []
    for g in common_perts:
        real_shift = real_pert_means[g] - ctrl_means
        pred_shift = model_means[model][g] - ctrl_means
        v_real = float(np.var(real_shift, ddof=1) + 1e-12)
        v_pred = float(np.var(pred_shift, ddof=1) + 1e-12)
        ratios.append(v_pred / v_real)
    return np.array(ratios)


delta_var_ratios = {m: per_pert_delta_var_ratio(m) for m in MODELS}
delta_var_summary = {}
for m in MODELS:
    r = delta_var_ratios[m]
    delta_var_summary[m] = {
        "median": float(np.median(r)),
        "mean": float(np.mean(r)),
        "frac_below_1": float(np.mean(r < 1)),
        "frac_below_0.5": float(np.mean(r < 0.5)),
        "frac_above_5": float(np.mean(r > 5)),
        "p10": float(np.percentile(r, 10)),
        "p90": float(np.percentile(r, 90)),
    }
    print(f"  {m}: median={delta_var_summary[m]['median']:.3f}, "
          f"frac<1={delta_var_summary[m]['frac_below_1']:.3f}, "
          f"frac>5={delta_var_summary[m]['frac_above_5']:.3f}")

# ----------------------------------------------------------------------
# 4. Diagnostic 2: Inter-perturbation prediction similarity
#    Mode collapse predicts high similarity (predictions look the same).
# ----------------------------------------------------------------------
print("\nDiagnostic 2: inter-perturbation cosine similarity of shift vectors")

def inter_pert_cosine(shifts: dict[str, np.ndarray]) -> tuple[float, float, float]:
    """Mean off-diagonal cosine of shift matrix; return (mean, p10, p90)."""
    perts = sorted(shifts.keys())
    M = np.stack([shifts[p] for p in perts], axis=0)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    Mn = M / norms
    sim = Mn @ Mn.T
    n = sim.shape[0]
    off = sim[~np.eye(n, dtype=bool)]
    return float(np.mean(off)), float(np.percentile(off, 10)), float(np.percentile(off, 90))


real_shifts = {g: real_pert_means[g] - ctrl_means for g in common_perts}
real_cos_mean, real_cos_p10, real_cos_p90 = inter_pert_cosine(real_shifts)
print(f"  real:        mean cos={real_cos_mean:.3f}  [p10={real_cos_p10:.3f}, p90={real_cos_p90:.3f}]")

inter_pert_summary = {"real": {"mean_cosine": real_cos_mean, "p10": real_cos_p10, "p90": real_cos_p90}}
for m in MODELS:
    shifts_m = {g: model_means[m][g] - ctrl_means for g in common_perts}
    cm, cp10, cp90 = inter_pert_cosine(shifts_m)
    inter_pert_summary[m] = {"mean_cosine": cm, "p10": cp10, "p90": cp90}
    print(f"  {m}: mean cos={cm:.3f}  [p10={cp10:.3f}, p90={cp90:.3f}]")

# ----------------------------------------------------------------------
# 5. Diagnostic 3: Per-perturbation DEG count distribution
#    For each perturbation, count genes with |log2FC|>0.5.
# ----------------------------------------------------------------------
print("\nDiagnostic 3: per-perturbation DEG count (|log2FC|>0.5)")

def deg_count_per_pert(get_means_fn, threshold: float = 0.5) -> np.ndarray:
    counts = []
    for g in common_perts:
        m = get_means_fn(g)
        l2fc = np.log2((m + PSEUDO) / (ctrl_means + PSEUDO))
        counts.append(int(np.sum(np.abs(l2fc) > threshold)))
    return np.array(counts)


real_degs = deg_count_per_pert(lambda g: real_pert_means[g])
deg_summary = {"real": {"median": int(np.median(real_degs)),
                          "mean": float(np.mean(real_degs)),
                          "p10": int(np.percentile(real_degs, 10)),
                          "p90": int(np.percentile(real_degs, 90))}}
print(f"  real:        median DEGs={deg_summary['real']['median']}  "
      f"[p10={deg_summary['real']['p10']}, p90={deg_summary['real']['p90']}]")

for m in MODELS:
    cnts = deg_count_per_pert(lambda g, mm=m: model_means[mm][g])
    deg_summary[m] = {"median": int(np.median(cnts)),
                       "mean": float(np.mean(cnts)),
                       "p10": int(np.percentile(cnts, 10)),
                       "p90": int(np.percentile(cnts, 90))}
    print(f"  {m}: median DEGs={deg_summary[m]['median']}  "
          f"[p10={deg_summary[m]['p10']}, p90={deg_summary[m]['p90']}]")

# ----------------------------------------------------------------------
# 6. Diagnostic 4: Cross-threshold inflation curve
# ----------------------------------------------------------------------
print("\nDiagnostic 4: fraction (|log2FC|>τ) at multiple thresholds")

THRESHOLDS = [0.1, 0.25, 0.5, 1.0, 2.0]

def frac_above(get_means_fn, tau: float) -> float:
    fracs = []
    for g in common_perts:
        m = get_means_fn(g)
        l2fc = np.log2((m + PSEUDO) / (ctrl_means + PSEUDO))
        fracs.append(np.mean(np.abs(l2fc) > tau))
    return float(np.mean(fracs))


threshold_curve = {"thresholds": THRESHOLDS, "real": [], **{m: [] for m in MODELS}}
for tau in THRESHOLDS:
    threshold_curve["real"].append(frac_above(lambda g: real_pert_means[g], tau))
    for m in MODELS:
        threshold_curve[m].append(frac_above(lambda g, mm=m: model_means[mm][g], tau))

print(f"  thresholds: {THRESHOLDS}")
print(f"  real:       {[f'{x:.4f}' for x in threshold_curve['real']]}")
for m in MODELS:
    print(f"  {m}: {[f'{x:.4f}' for x in threshold_curve[m]]}")

# ----------------------------------------------------------------------
# 7. Verdict
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)

verdict_lines = []

# Models with median Δ-variance ratio < 1 → mode collapse confirmed
mc_models = [m for m in MODELS if delta_var_summary[m]["median"] < 1.0]
inf_models = [m for m in MODELS if delta_var_summary[m]["median"] > 5.0]
high_cosine_models = [m for m in MODELS
                       if inter_pert_summary[m]["mean_cosine"] > inter_pert_summary["real"]["mean_cosine"] + 0.10]

if mc_models and inf_models:
    classification = "MIXED — some models collapse, others inflate"
elif mc_models:
    classification = "MODE COLLAPSE confirmed"
elif inf_models:
    classification = "INFLATION confirmed"
else:
    classification = "NEITHER strongly — distribution is approximately preserved"

verdict_lines.append(f"Per-pert Δ-variance: {classification}")
verdict_lines.append(
    f"  models with median Δ-variance ratio < 1 (mode collapse): {mc_models}"
)
verdict_lines.append(
    f"  models with median Δ-variance ratio > 5 (inflation):     {inf_models}"
)
verdict_lines.append(
    f"  models with elevated inter-pert cosine vs real:           {high_cosine_models}"
)

# DEG concentration
real_med = deg_summary["real"]["median"]
deg_high_models = [m for m in MODELS
                    if deg_summary[m]["median"] > real_med * 3]
verdict_lines.append(
    f"DEG concentration: models predicting >3x real median DEGs/pert: {deg_high_models}"
)

# The reconciliation question
if high_cosine_models and (mc_models or any(deg_summary[m]["median"] > real_med * 3 for m in MODELS)):
    reconciliation = (
        "OUTCOME (A): mode collapse + inflation coexist. "
        "Models predict similar dense response pattern across perturbations, "
        "yielding low per-perturbation variance (mode collapse) and high "
        "fraction-of-large-effects when pooled (inflation). Both literature "
        "threads describe the same underlying failure from different angles."
    )
elif inf_models and not mc_models:
    reconciliation = (
        "OUTCOME (B): inflation is real and isolated. "
        "Per-perturbation variance is comparable to or larger than real, "
        "and inter-perturbation similarity is not elevated. Original "
        "inflation framing survives."
    )
else:
    reconciliation = (
        "OUTCOME (C): inflation framing requires revision. "
        "Per-perturbation variance is below real baseline; the original "
        "'inflation' was a pooling artifact across (perturbation, gene) "
        "entries with high inter-perturbation similarity."
    )

verdict_lines.append("")
verdict_lines.append(reconciliation)
print("\n".join(verdict_lines))

# ----------------------------------------------------------------------
# 8. Save
# ----------------------------------------------------------------------
out = {
    "n_perturbations": len(common_perts),
    "n_genes": n_genes,
    "delta_var_ratio_per_pert": {m: delta_var_ratios[m].tolist() for m in MODELS},
    "delta_var_summary": delta_var_summary,
    "inter_perturbation_cosine": inter_pert_summary,
    "deg_count_summary": deg_summary,
    "threshold_curve": threshold_curve,
    "verdict": classification,
    "reconciliation": reconciliation,
    "verdict_lines": verdict_lines,
}
with open(OUT / "distribution_reconciliation.json", "w") as f:
    json.dump(out, f, indent=2)

# CSV per-perturbation rows for follow-up plots
rows = []
for i, g in enumerate(common_perts):
    row = {"perturbation": g, "n_real_cells": real_pert_ncells.get(g, 0),
           "real_deg_count": int(np.sum(np.abs(np.log2((real_pert_means[g] + PSEUDO) /
                                                      (ctrl_means + PSEUDO))) > 0.5))}
    for m in MODELS:
        row[f"{m}_delta_var_ratio"] = float(delta_var_ratios[m][i])
        l2fc_m = np.log2((model_means[m][g] + PSEUDO) / (ctrl_means + PSEUDO))
        row[f"{m}_deg_count"] = int(np.sum(np.abs(l2fc_m) > 0.5))
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT / "distribution_reconciliation.csv", index=False)

with open(OUT / "distribution_reconciliation.txt", "w") as f:
    f.write("\n".join(verdict_lines))

print(f"\nWrote: {OUT/'distribution_reconciliation.json'}")
print(f"Wrote: {OUT/'distribution_reconciliation.csv'}")
print(f"Wrote: {OUT/'distribution_reconciliation.txt'}")
