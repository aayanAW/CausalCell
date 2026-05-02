"""Phase 4 — distribution diagnostics for CPA alt-loss vs vanilla.

Compares the per-perturbation Δ-variance, inter-perturbation cosine, and
DEG concentration of CPA-alt-loss-Gauss against CPA-vanilla-NB.

Output: results/phase4/cpa_alt_loss_diagnostics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "results" / "phase4"
OUT.mkdir(parents=True, exist_ok=True)

K562 = DATA / "k562_subset_n200_slim.h5ad"
PERT_COL = "gene"
CTRL = "non-targeting"
PSEUDO = 0.01

print("=" * 72)
print("Phase 4 — CPA alt-loss distribution diagnostics")
print("=" * 72)

adata = ad.read_h5ad(K562)
adata.var_names = [str(g) for g in adata.var_names]
gene_names = list(adata.var_names)
n_g = len(gene_names)

X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
X = X.astype(np.float64)
is_ctrl = adata.obs[PERT_COL].values == CTRL
ctrl_mean = X[is_ctrl].mean(axis=0)

# Real per-pert means
real_pert_means = {}
real_pert_vars = {}
for g in gene_names:
    idx = np.where(adata.obs[PERT_COL].values == g)[0]
    if len(idx) >= 5:
        cells = X[idx]
        real_pert_means[g] = cells.mean(axis=0)
        real_pert_vars[g] = cells.var(axis=0, ddof=1)


def load_pred_means(path, target_genes):
    p = ad.read_h5ad(path)
    p_genes = list(p.var_names)
    pred_to_idx = {g: i for i, g in enumerate(p_genes)}
    cols = [pred_to_idx[g] for g in target_genes if g in pred_to_idx]
    pmat = p.X.toarray() if hasattr(p.X, "toarray") else np.asarray(p.X)
    pmat = np.asarray(pmat[:, cols], dtype=np.float64)
    out = {}
    for i in range(pmat.shape[0]):
        gene = str(p.obs[PERT_COL].iloc[i]) if PERT_COL in p.obs.columns else \
               str(p.obs.iloc[i, 0])
        out[gene] = pmat[i]
    return out


# Load both CPA variants — restricted to the 200 eval genes
van_means = load_pred_means(DATA / "model_outputs_real_n200" / "cpa_predictions.h5ad",
                              gene_names)
alt_means = load_pred_means(DATA / "model_outputs_real_n200_alt_loss" / "cpa_predictions.h5ad",
                              gene_names)
common = sorted(set(van_means) & set(alt_means) & set(real_pert_means))
print(f"Common perturbations: {len(common)}")


def per_pert_dvar_ratio(model_means, real_means, ctrl_mean):
    ratios = []
    for g in common:
        real_shift = real_means[g] - ctrl_mean
        pred_shift = model_means[g] - ctrl_mean
        v_real = float(np.var(real_shift, ddof=1) + 1e-12)
        v_pred = float(np.var(pred_shift, ddof=1) + 1e-12)
        ratios.append(v_pred / v_real)
    return np.array(ratios)


def inter_pert_cos(model_means, ctrl_mean):
    perts = sorted(model_means.keys())
    M = np.stack([model_means[p] - ctrl_mean for p in perts], axis=0)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    Mn = M / norms
    sim = Mn @ Mn.T
    n = sim.shape[0]
    off = sim[~np.eye(n, dtype=bool)]
    return float(np.mean(off)), float(np.percentile(off, 10)), float(np.percentile(off, 90))


def deg_count(model_means, ctrl_mean, threshold=0.5):
    counts = []
    for g in common:
        m = model_means[g]
        l2fc = np.log2((m + PSEUDO) / (ctrl_mean + PSEUDO))
        counts.append(int((np.abs(l2fc) > threshold).sum()))
    return np.array(counts)


print("\n=== Per-perturbation Δ-variance ratio (lower = more collapsed) ===")
r_van = per_pert_dvar_ratio(van_means, real_pert_means, ctrl_mean)
r_alt = per_pert_dvar_ratio(alt_means, real_pert_means, ctrl_mean)
print(f"  CPA vanilla (NB):    median={np.median(r_van):.4f}, frac<1={np.mean(r_van<1):.3f}")
print(f"  CPA alt-loss (Gauss): median={np.median(r_alt):.4f}, frac<1={np.mean(r_alt<1):.3f}")

print("\n=== Inter-perturbation cosine similarity (higher = more collapsed) ===")
c_real_van, _, _ = inter_pert_cos({g: real_pert_means[g] for g in common}, ctrl_mean)
c_van, c_van_p10, c_van_p90 = inter_pert_cos(van_means, ctrl_mean)
c_alt, c_alt_p10, c_alt_p90 = inter_pert_cos(alt_means, ctrl_mean)
print(f"  Real:                mean={c_real_van:.4f}")
print(f"  CPA vanilla (NB):    mean={c_van:.4f} [{c_van_p10:.3f}, {c_van_p90:.3f}]")
print(f"  CPA alt-loss (Gauss): mean={c_alt:.4f} [{c_alt_p10:.3f}, {c_alt_p90:.3f}]")

print("\n=== Per-perturbation DEG count (genes with |log2FC|>0.5) ===")
real_deg = deg_count(real_pert_means, ctrl_mean)
van_deg = deg_count(van_means, ctrl_mean)
alt_deg = deg_count(alt_means, ctrl_mean)
print(f"  Real:                median={int(np.median(real_deg))}, p90={int(np.percentile(real_deg,90))}")
print(f"  CPA vanilla (NB):    median={int(np.median(van_deg))}, p90={int(np.percentile(van_deg,90))}")
print(f"  CPA alt-loss (Gauss): median={int(np.median(alt_deg))}, p90={int(np.percentile(alt_deg,90))}")

out = {
    "n_perturbations": len(common),
    "delta_var_ratio": {
        "vanilla_NB": {"median": float(np.median(r_van)), "frac_lt_1": float(np.mean(r_van<1))},
        "alt_loss_Gauss": {"median": float(np.median(r_alt)), "frac_lt_1": float(np.mean(r_alt<1))},
    },
    "inter_pert_cosine": {
        "real": {"mean": c_real_van},
        "vanilla_NB": {"mean": c_van, "p10": c_van_p10, "p90": c_van_p90},
        "alt_loss_Gauss": {"mean": c_alt, "p10": c_alt_p10, "p90": c_alt_p90},
    },
    "deg_count": {
        "real": {"median": int(np.median(real_deg)), "p90": int(np.percentile(real_deg,90))},
        "vanilla_NB": {"median": int(np.median(van_deg)), "p90": int(np.percentile(van_deg,90))},
        "alt_loss_Gauss": {"median": int(np.median(alt_deg)), "p90": int(np.percentile(alt_deg,90))},
    },
}

# Verdict
delta_van_more_collapsed = np.median(r_van) < np.median(r_alt)
cos_van_higher = c_van > c_alt
deg_van_lower = np.median(van_deg) < np.median(alt_deg)

if delta_van_more_collapsed and cos_van_higher and deg_van_lower:
    verdict = "Alt-loss (Gaussian) reduces mode collapse on all three diagnostics"
elif (delta_van_more_collapsed + cos_van_higher + deg_van_lower) >= 2:
    verdict = "Alt-loss reduces mode collapse on majority of diagnostics"
elif (delta_van_more_collapsed + cos_van_higher + deg_van_lower) >= 1:
    verdict = "Alt-loss has mixed effect; marginal change"
else:
    verdict = "Alt-loss does NOT reduce mode collapse — failure attributable to objective is not supported"

print(f"\nVERDICT: {verdict}")
out["verdict"] = verdict

with open(OUT / "cpa_alt_loss_diagnostics.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Wrote {OUT/'cpa_alt_loss_diagnostics.json'}")
