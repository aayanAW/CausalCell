#!/usr/bin/env python3
"""J2 (local, multi-seed): does a LEARNED covariance-aware transform close the
gap where fixed marginal calibration did not?

The identifiability result (Prop. ci-invariance) says a per-gene monotone
(marginal) calibration cannot change conditional-independence structure, and the
calibration sweep confirmed no marginal ordering beats vanilla on held-out
perturbations. The proposed remedy is a *training-time* covariance objective. As
a no-GPU proxy for retraining CPA end-to-end, we train a small parametric head
g_phi on CPA's predicted shifts under a held-out perturbation split, with two
objectives:

    recon-only :  || g(delta_cpa) - delta_real ||^2
    cov-reg    :  recon  +  lambda * || Cov(g(delta_cpa)) - Cov(delta_real) ||_F^2 / Z

g_phi has parameters fit on TRAIN perturbations and generalizes to TEST
perturbations, so unlike quantile matching it is a *learned joint* transform.
Predictions go through the identical overlay -> GIES pipeline; F1 is scored
against the real-data GIES ground truth on the held-out perturbations, across
5 random splits. Reports vanilla CPA, recon-only, and cov-reg with mean +/- SD
and the paired cov-reg gain. Positive => supports the training-time remedy; a
null is reported straight. Fully local, no GPU, no Modal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from causalcellbench.training.covariance_regularizer import (  # noqa: E402
    covariance_frobenius_penalty,
)
from scripts.run_eval_local import (  # noqa: E402
    build_cb_input,
    compute_set_metrics,
    run_gies,
    sample_perturbed_cells_overlay,
)

SLIM = ROOT / "data" / "k562_subset_n200_slim.h5ad"
CPA = ROOT / "data" / "model_outputs_real_n200" / "cpa_predictions.h5ad"
OUT = ROOT / "results" / "phase4" / "j2_covariance_regularizer.json"
SEEDS = [42, 123, 456, 789, 1024]
N_TEST = 40
N_CELLS = 100
N_CTRL = 200
PARTITION = 20
LAMBDA = 1.0
EPOCHS = 400
CTRL = "non-targeting"


def per_pert_means(adata, genes, pert_col="gene"):
    sym = (
        adata.var["gene_name"].astype(str).values
        if "gene_name" in adata.var
        else np.array(genes)
    )
    idx = {g: i for i, g in enumerate(sym)}
    cols = [idx[g] for g in genes]
    X = adata.X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)[
        :, cols
    ]
    labels = adata.obs[pert_col].astype(str).values
    return {p: X[labels == p].mean(axis=0) for p in np.unique(labels)}


def gies_edges_for(means_by_pert, ctrl_X, genes):
    mats, labs = [], []
    for p, mu in means_by_pert.items():
        cells = sample_perturbed_cells_overlay(
            ctrl_X, np.asarray(mu), N_CELLS, seed=abs(hash(p)) % 9999
        )
        mats.append(cells)
        labs.extend([p] * cells.shape[0])
    cb = build_cb_input(ctrl_X, mats, labs, genes)
    return set(run_gies(cb, seed=42, partition_size=PARTITION))


def train_head(Xtr, Ytr, real_cov_tr, lam, seed):
    torch.manual_seed(seed)
    G = Xtr.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(G, G), torch.nn.ReLU(), torch.nn.Linear(G, G)
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        opt.zero_grad()
        out = net(Xtr)
        recon = ((out - Ytr) ** 2).mean()
        pen = (
            covariance_frobenius_penalty(out, real_cov_tr)
            if lam > 0
            else torch.tensor(0.0)
        )
        (recon + lam * pen).backward()
        opt.step()
    return net


def run_seed(
    seed, perts, d_real, d_cpa, real_means, cpa_means, ctrl_mean, ctrl_cells, genes
):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(perts))
    te = sorted(perm[:N_TEST].tolist())
    tr = [i for i in range(len(perts)) if i not in set(te)]
    Xtr = torch.tensor(d_cpa[tr], dtype=torch.float32)
    Ytr = torch.tensor(d_real[tr], dtype=torch.float32)
    real_cov_tr = torch.tensor(np.cov(d_real[tr], rowvar=False), dtype=torch.float32)

    gt = gies_edges_for({perts[i]: real_means[perts[i]] for i in te}, ctrl_cells, genes)
    f1 = lambda mb: compute_set_metrics(
        list(gies_edges_for(mb, ctrl_cells, genes)), gt, genes
    )["f1"]

    out = {
        "vanilla_cpa": f1({perts[i]: cpa_means[perts[i]] for i in te}),
        "gt_edges": len(gt),
    }
    for name, lam in [("recon_only", 0.0), ("cov_reg", LAMBDA)]:
        net = train_head(Xtr, Ytr, real_cov_tr, lam, seed)
        with torch.no_grad():
            corr = {
                perts[i]: ctrl_mean
                + net(torch.tensor(d_cpa[i], dtype=torch.float32)).numpy()
                for i in te
            }
        out[name] = f1(corr)
    print(
        f"seed {seed}: vanilla={out['vanilla_cpa']:.4f} recon={out['recon_only']:.4f} "
        f"cov_reg={out['cov_reg']:.4f}"
    )
    return out


def main():
    slim = ad.read_h5ad(SLIM)
    genes = (
        list(slim.var["gene_name"].astype(str).values)
        if "gene_name" in slim.var
        else list(slim.var_names)
    )
    real_means = per_pert_means(slim, genes)
    ctrl_mean = real_means[CTRL]
    cpa = ad.read_h5ad(CPA)
    cpa_pcol = "gene" if "gene" in cpa.obs else cpa.obs.columns[0]
    cpa_means = per_pert_means(cpa, genes, pert_col=cpa_pcol)
    perts = sorted(set(real_means) & set(cpa_means) - {CTRL})
    d_real = np.stack([real_means[p] - ctrl_mean for p in perts])
    d_cpa = np.stack([cpa_means[p] - ctrl_mean for p in perts])
    print(f"{len(perts)} shared perturbations")

    sym = list(slim.var["gene_name"].astype(str).values)
    cc = slim.X[slim.obs["gene"].astype(str).values == CTRL]
    cc = np.asarray(cc.todense() if hasattr(cc, "todense") else cc, dtype=np.float64)[
        :, [sym.index(g) for g in genes]
    ]
    ctrl_cells = cc[
        np.random.default_rng(0).choice(
            cc.shape[0], min(N_CTRL, cc.shape[0]), replace=False
        )
    ]

    rows = [
        run_seed(
            s, perts, d_real, d_cpa, real_means, cpa_means, ctrl_mean, ctrl_cells, genes
        )
        for s in SEEDS
    ]

    def agg(k):
        v = np.array([r[k] for r in rows])
        return float(v.mean()), float(v.std(ddof=1))

    van_m, van_s = agg("vanilla_cpa")
    rec_m, rec_s = agg("recon_only")
    cov_m, cov_s = agg("cov_reg")
    d_cr = np.array([r["cov_reg"] - r["recon_only"] for r in rows])
    d_cv = np.array([r["cov_reg"] - r["vanilla_cpa"] for r in rows])
    res = {
        "seeds": SEEDS,
        "lambda": LAMBDA,
        "per_seed": rows,
        "vanilla_cpa": {"mean": van_m, "sd": van_s},
        "recon_only": {"mean": rec_m, "sd": rec_s},
        "cov_reg": {"mean": cov_m, "sd": cov_s},
        "cov_reg_minus_recon": {
            "mean": float(d_cr.mean()),
            "sd": float(d_cr.std(ddof=1)),
            "n_positive": int((d_cr > 0).sum()),
        },
        "cov_reg_minus_vanilla": {
            "mean": float(d_cv.mean()),
            "sd": float(d_cv.std(ddof=1)),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print("\n=== J2 MULTI-SEED RESULT ===")
    print(f"vanilla CPA      : {van_m:.4f} ± {van_s:.4f}")
    print(f"recon-only head  : {rec_m:.4f} ± {rec_s:.4f}")
    print(f"cov-reg head     : {cov_m:.4f} ± {cov_s:.4f}")
    print(
        f"cov-reg − recon  : {d_cr.mean():+.4f} ± {d_cr.std(ddof=1):.4f} ({int((d_cr > 0).sum())}/5 positive)"
    )
    print(f"cov-reg − vanilla: {d_cv.mean():+.4f} ± {d_cv.std(ddof=1):.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
