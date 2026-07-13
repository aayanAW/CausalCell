#!/usr/bin/env python3
"""J2 covariance-regularizer lambda sweep (autoresearch-loop guards).

Pre-registered (preregistration_j2_lambda.md). FIT-GATE GREEN.
Guard 1: multi-seed significance gate. Guard 2: locked test split touched once.
Guard 3: pre-registered grid + stopping rule.

Per seed, perturbations split train(120)/val(40)/test(40). For each lambda in the
grid, a learned head g_phi is trained on train and scored on VAL via the overlay
-> GIES pipeline. lambda* = argmax mean VAL F1 across seeds. The TEST split is
sealed and evaluated ONCE at the end for vanilla CPA, recon-only (lambda=0), and
lambda*. A keep is reported only if lambda* beats lambda=0 on TEST across >=3/5
seeds with mean improvement exceeding the run-to-run SD.
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
OUT = ROOT / "results" / "phase4" / "j2_lambda_sweep.json"
LEDGER = ROOT / "results" / "phase4" / "j2_lambda_decision_ledger.jsonl"
SEEDS = [42, 123, 456, 789, 1024]
LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
N_VAL, N_TEST = 40, 40
N_CELLS, N_CTRL, PARTITION, EPOCHS = 100, 200, 20, 400
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


def gies_edges(means_by_pert, ctrl_X, genes):
    mats, labs = [], []
    for p, mu in means_by_pert.items():
        cells = sample_perturbed_cells_overlay(
            ctrl_X, np.asarray(mu), N_CELLS, seed=abs(hash(p)) % 9999
        )
        mats.append(cells)
        labs.extend([p] * cells.shape[0])
    return set(
        run_gies(
            build_cb_input(ctrl_X, mats, labs, genes), seed=42, partition_size=PARTITION
        )
    )


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


def main():
    slim = ad.read_h5ad(SLIM)
    genes = list(slim.var["gene_name"].astype(str).values)
    real_means = per_pert_means(slim, genes)
    ctrl_mean = real_means[CTRL]
    cpa = ad.read_h5ad(CPA)
    cpa_pcol = "gene" if "gene" in cpa.obs else cpa.obs.columns[0]
    cpa_means = per_pert_means(cpa, genes, pert_col=cpa_pcol)
    perts = sorted(set(real_means) & set(cpa_means) - {CTRL})
    d_real = {p: real_means[p] - ctrl_mean for p in perts}
    d_cpa = {p: cpa_means[p] - ctrl_mean for p in perts}

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

    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    ledger = open(LEDGER, "w")

    # ---- Pass 1: VALIDATION sweep over lambda ----
    val_f1 = {lam: [] for lam in LAMBDAS}
    splits = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = list(rng.permutation(len(perts)))
        test_i = [perts[i] for i in perm[:N_TEST]]
        val_i = [perts[i] for i in perm[N_TEST : N_TEST + N_VAL]]
        tr_i = [perts[i] for i in perm[N_TEST + N_VAL :]]
        splits[seed] = (tr_i, val_i, test_i)
        Xtr = torch.tensor(np.stack([d_cpa[p] for p in tr_i]), dtype=torch.float32)
        Ytr = torch.tensor(np.stack([d_real[p] for p in tr_i]), dtype=torch.float32)
        real_cov_tr = torch.tensor(
            np.cov(np.stack([d_real[p] for p in tr_i]), rowvar=False),
            dtype=torch.float32,
        )
        val_gt = gies_edges({p: real_means[p] for p in val_i}, ctrl_cells, genes)
        for lam in LAMBDAS:
            net = train_head(Xtr, Ytr, real_cov_tr, lam, seed)
            with torch.no_grad():
                corr = {
                    p: ctrl_mean
                    + net(torch.tensor(d_cpa[p], dtype=torch.float32)).numpy()
                    for p in val_i
                }
            f1 = compute_set_metrics(
                list(gies_edges(corr, ctrl_cells, genes)), val_gt, genes
            )["f1"]
            val_f1[lam].append(f1)
            ledger.write(
                json.dumps({"phase": "val", "seed": seed, "lambda": lam, "val_f1": f1})
                + "\n"
            )
            ledger.flush()
        print(f"[val] seed {seed} done")

    val_mean = {lam: float(np.mean(v)) for lam, v in val_f1.items()}
    lam_star = max([l for l in LAMBDAS if l > 0], key=lambda l: val_mean[l])
    print(f"val mean F1 per lambda: { {k: round(v, 4) for k, v in val_mean.items()} }")
    print(
        f"lambda* (best val, excl 0) = {lam_star}; recon(0) val = {val_mean[0.0]:.4f}"
    )

    # ---- Pass 2: LOCKED TEST, touched once (vanilla, recon, lambda*) ----
    test = {"vanilla": [], "recon": [], "covreg_star": []}
    for seed in SEEDS:
        tr_i, _, test_i = splits[seed]
        Xtr = torch.tensor(np.stack([d_cpa[p] for p in tr_i]), dtype=torch.float32)
        Ytr = torch.tensor(np.stack([d_real[p] for p in tr_i]), dtype=torch.float32)
        real_cov_tr = torch.tensor(
            np.cov(np.stack([d_real[p] for p in tr_i]), rowvar=False),
            dtype=torch.float32,
        )
        test_gt = gies_edges({p: real_means[p] for p in test_i}, ctrl_cells, genes)
        f1 = lambda mb: compute_set_metrics(
            list(gies_edges(mb, ctrl_cells, genes)), test_gt, genes
        )["f1"]
        test["vanilla"].append(f1({p: cpa_means[p] for p in test_i}))
        for name, lam in [("recon", 0.0), ("covreg_star", lam_star)]:
            net = train_head(Xtr, Ytr, real_cov_tr, lam, seed)
            with torch.no_grad():
                corr = {
                    p: ctrl_mean
                    + net(torch.tensor(d_cpa[p], dtype=torch.float32)).numpy()
                    for p in test_i
                }
            test[name].append(f1(corr))
        ledger.write(
            json.dumps(
                {
                    "phase": "test",
                    "seed": seed,
                    "vanilla": test["vanilla"][-1],
                    "recon": test["recon"][-1],
                    "covreg_star": test["covreg_star"][-1],
                    "lambda_star": lam_star,
                }
            )
            + "\n"
        )
        ledger.flush()
        print(
            f"[test] seed {seed}: vanilla={test['vanilla'][-1]:.4f} recon={test['recon'][-1]:.4f} "
            f"covreg*={test['covreg_star'][-1]:.4f}"
        )

    d = np.array(test["covreg_star"]) - np.array(test["recon"])
    keep = (d > 0).sum() >= 3 and d.mean() > d.std(ddof=1)
    res = {
        "lambdas": LAMBDAS,
        "seeds": SEEDS,
        "n_val_trials": len(LAMBDAS) * len(SEEDS),
        "val_mean_f1_per_lambda": val_mean,
        "lambda_star": lam_star,
        "test": {
            k: {
                "mean": float(np.mean(v)),
                "sd": float(np.std(v, ddof=1)),
                "per_seed": v,
            }
            for k, v in test.items()
        },
        "covreg_minus_recon_test": {
            "mean": float(d.mean()),
            "sd": float(d.std(ddof=1)),
            "n_positive": int((d > 0).sum()),
        },
        "KEEP_covreg_over_recon": bool(keep),
        "verdict": (
            "KEEP: covariance penalty robustly helps"
            if keep
            else "DISCARD: covariance penalty does not clear the multi-seed gate"
        ),
    }
    OUT.write_text(json.dumps(res, indent=2))
    ledger.close()
    print("\n=== J2 LAMBDA SWEEP (locked-test) ===")
    print(f"lambda* = {lam_star}")
    print(
        f"TEST  vanilla={res['test']['vanilla']['mean']:.4f}  recon={res['test']['recon']['mean']:.4f}  "
        f"covreg*={res['test']['covreg_star']['mean']:.4f}"
    )
    print(
        f"covreg* - recon (test) = {d.mean():+.4f} ± {d.std(ddof=1):.4f} ({int((d > 0).sum())}/5 positive)"
    )
    print(res["verdict"])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
