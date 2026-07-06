#!/usr/bin/env python3
"""B — precision-aware / OT objective, no-GPU proxy sweep (pre-registered).

prereg_B_precision_objective.md. Mirrors scripts/j2_lambda_sweep.py EXACTLY
(learned head Linear200->ReLU->Linear200, epochs 400, lr 1e-3, overlay->GIES,
train 120 / val 40 / test 40, locked test touched once, 5 seeds) with TWO
changes:
  1. loss term: recon + lam * L_term, term in {precision, mmd, sinkhorn}
     (the untried objectives the J2 covariance null motivates), one at a time.
  2. scoring: causal F1 vs the A1 PERTURBATION-ANCHORED ground truth (non-
     circular), restricted per split to the split's perturbations, NOT GIES-on-
     real.

Keep/discard (Guard 1): the selected (term, lam*) is an improvement only if it
beats recon-only (lam=0) on TEST across >=3/5 seeds AND mean test-F1 gain exceeds
the run-to-run SD. Otherwise: honest null; the definitive answer awaits the GPU
CPA retrain (this proxy is a CPU stand-in, never the definitive result).
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
from causalcellbench.training.joint_structure_objectives import (  # noqa: E402
    JointStructureLoss,
)
from scripts.run_eval_local import (  # noqa: E402
    build_cb_input,
    compute_set_metrics,
    run_gies,
    sample_perturbed_cells_overlay,
)

SLIM = ROOT / "data" / "k562_subset_n200_slim.h5ad"
CPA = ROOT / "data" / "model_outputs_real_n200" / "cpa_predictions.h5ad"
ANCHORED = ROOT / "data" / "ground_truth" / "perturbation_anchored_k562_n200.json"
OUT = ROOT / "results" / "phaseB" / "b_precision_objective_sweep.json"
LEDGER = ROOT / "results" / "phaseB" / "b_decision_ledger.jsonl"
SEEDS = [42, 123, 456, 789, 1024]
LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
TERMS = ["precision", "mmd", "sinkhorn"]
N_VAL, N_TEST = 40, 40
N_CELLS, N_CTRL, PARTITION, EPOCHS = 100, 200, 20, 400
CTRL = "non-targeting"


def per_pert_means(adata, genes, pert_col="gene"):
    sym = (adata.var["gene_name"].astype(str).values
           if "gene_name" in adata.var else np.array(genes))
    idx = {g: i for i, g in enumerate(sym)}
    cols = [idx[g] for g in genes]
    X = adata.X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)[:, cols]
    labels = adata.obs[pert_col].astype(str).values
    return {p: X[labels == p].mean(axis=0) for p in np.unique(labels)}


def gies_edges(means_by_pert, ctrl_X, genes):
    mats, labs = [], []
    for p, mu in means_by_pert.items():
        cells = sample_perturbed_cells_overlay(ctrl_X, np.asarray(mu), N_CELLS,
                                               seed=abs(hash(p)) % 9999)
        mats.append(cells); labs.extend([p] * cells.shape[0])
    return set(run_gies(build_cb_input(ctrl_X, mats, labs, genes),
                        seed=42, partition_size=PARTITION))


def train_head(Xtr, Ytr, real_shift_tr, term, lam, seed):
    torch.manual_seed(seed)
    G = Xtr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(G, G), torch.nn.ReLU(), torch.nn.Linear(G, G))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit = JointStructureLoss(term=term, lam=lam).fit_real(real_shift_tr) if lam > 0 else None
    for _ in range(EPOCHS):
        opt.zero_grad()
        out = net(Xtr)
        recon = ((out - Ytr) ** 2).mean()
        loss = crit(recon, out) if crit is not None else recon
        loss.backward()
        opt.step()
    return net


def main():
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    anchored = json.load(open(ANCHORED))
    prim_key = anchored["meta"]["primary_key"]
    anchored_edges_all = set(tuple(e) for e in anchored["edge_sets"][prim_key])

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

    sym = list(genes)
    cc = slim.X[slim.obs["gene"].astype(str).values == CTRL]
    cc = np.asarray(cc.todense() if hasattr(cc, "todense") else cc, dtype=np.float64)[
        :, [sym.index(g) for g in genes]]
    ctrl_cells = cc[np.random.default_rng(0).choice(cc.shape[0], min(N_CTRL, cc.shape[0]),
                                                     replace=False)]

    def anchored_on(pert_subset):
        """Anchored GT restricted to edges whose SOURCE is in the split."""
        ss = set(pert_subset)
        return {(a, b) for (a, b) in anchored_edges_all if a in ss}

    ledger = open(LEDGER, "w")
    # ---- Pass 1: VALIDATION sweep over (term, lambda) ----
    val_f1 = {(t, lam): [] for t in TERMS for lam in LAMBDAS if lam > 0}
    val_recon = []  # lam=0 shared across terms
    splits = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = list(rng.permutation(len(perts)))
        test_i = [perts[i] for i in perm[:N_TEST]]
        val_i = [perts[i] for i in perm[N_TEST:N_TEST + N_VAL]]
        tr_i = [perts[i] for i in perm[N_TEST + N_VAL:]]
        splits[seed] = (tr_i, val_i, test_i)
        Xtr = torch.tensor(np.stack([d_cpa[p] for p in tr_i]), dtype=torch.float32)
        Ytr = torch.tensor(np.stack([d_real[p] for p in tr_i]), dtype=torch.float32)
        real_shift_tr = torch.tensor(np.stack([d_real[p] for p in tr_i]), dtype=torch.float32)
        val_gt = anchored_on(val_i)

        def eval_head(net):
            with torch.no_grad():
                corr = {p: ctrl_mean + net(torch.tensor(d_cpa[p], dtype=torch.float32)).numpy()
                        for p in val_i}
            return compute_set_metrics(list(gies_edges(corr, ctrl_cells, genes)),
                                       val_gt, genes)["f1"]

        # recon-only (lam=0), once per seed
        net0 = train_head(Xtr, Ytr, real_shift_tr, "precision", 0.0, seed)
        f0 = eval_head(net0); val_recon.append(f0)
        ledger.write(json.dumps({"phase": "val", "seed": seed, "term": "recon",
                                 "lambda": 0.0, "val_f1": f0}) + "\n"); ledger.flush()
        for t in TERMS:
            for lam in [l for l in LAMBDAS if l > 0]:
                net = train_head(Xtr, Ytr, real_shift_tr, t, lam, seed)
                f1 = eval_head(net)
                val_f1[(t, lam)].append(f1)
                ledger.write(json.dumps({"phase": "val", "seed": seed, "term": t,
                                         "lambda": lam, "val_f1": f1}) + "\n"); ledger.flush()
        print(f"[val] seed {seed} done")

    val_mean = {k: float(np.mean(v)) for k, v in val_f1.items()}
    recon_val = float(np.mean(val_recon))
    (term_star, lam_star) = max(val_mean, key=lambda k: val_mean[k])
    print(f"recon(0) val F1={recon_val:.4f}; best (term,lam)=({term_star},{lam_star}) "
          f"val F1={val_mean[(term_star, lam_star)]:.4f}")

    # ---- Pass 2: LOCKED TEST touched once (vanilla, recon, term*/lam*) ----
    test = {"vanilla": [], "recon": [], "star": []}
    for seed in SEEDS:
        tr_i, _, test_i = splits[seed]
        Xtr = torch.tensor(np.stack([d_cpa[p] for p in tr_i]), dtype=torch.float32)
        Ytr = torch.tensor(np.stack([d_real[p] for p in tr_i]), dtype=torch.float32)
        real_shift_tr = torch.tensor(np.stack([d_real[p] for p in tr_i]), dtype=torch.float32)
        test_gt = anchored_on(test_i)

        def f1_of(mb):
            return compute_set_metrics(list(gies_edges(mb, ctrl_cells, genes)),
                                       test_gt, genes)["f1"]
        test["vanilla"].append(f1_of({p: cpa_means[p] for p in test_i}))
        for name, (t, lam) in [("recon", ("precision", 0.0)), ("star", (term_star, lam_star))]:
            net = train_head(Xtr, Ytr, real_shift_tr, t, lam, seed)
            with torch.no_grad():
                corr = {p: ctrl_mean + net(torch.tensor(d_cpa[p], dtype=torch.float32)).numpy()
                        for p in test_i}
            test[name].append(f1_of(corr))
        ledger.write(json.dumps({"phase": "test", "seed": seed, "vanilla": test["vanilla"][-1],
                                 "recon": test["recon"][-1], "star": test["star"][-1],
                                 "term_star": term_star, "lambda_star": lam_star}) + "\n")
        ledger.flush()
        print(f"[test] seed {seed}: vanilla={test['vanilla'][-1]:.4f} "
              f"recon={test['recon'][-1]:.4f} star={test['star'][-1]:.4f}")

    d = np.array(test["star"]) - np.array(test["recon"])
    keep = bool((d > 0).sum() >= 3 and d.mean() > d.std(ddof=1))
    res = {
        "terms": TERMS, "lambdas": LAMBDAS, "seeds": SEEDS,
        "n_val_trials": len(val_f1), "scored_against": "A1 perturbation-anchored GT (primary)",
        "val_mean_f1": {f"{t}|{lam}": v for (t, lam), v in val_mean.items()},
        "recon_val_f1": recon_val, "term_star": term_star, "lambda_star": lam_star,
        "test": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "per_seed": v}
                 for k, v in test.items()},
        "star_minus_recon_test": {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                                  "per_seed": d.tolist(), "n_win": int((d > 0).sum())},
        "keep": keep,
        "verdict": ("KEEP" if keep else "DISCARD") + " (no-GPU proxy; definitive test = GPU CPA retrain)",
    }
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\n=== B PROXY VERDICT: {res['verdict']} ===")
    print(f"star({term_star},lam={lam_star}) test={res['test']['star']['mean']:.4f}±{res['test']['star']['sd']:.4f} "
          f"vs recon={res['test']['recon']['mean']:.4f}±{res['test']['recon']['sd']:.4f}; "
          f"delta={d.mean():+.4f}±{d.std(ddof=1):.4f} ({int((d>0).sum())}/5 win)")
    print("saved", OUT)


if __name__ == "__main__":
    main()
