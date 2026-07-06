#!/usr/bin/env python
"""Genome-scale causal recovery via inspre (ACE-based, NO co-partition ceiling).

The GIES eval saturates at N~200 partly because GIES runs on random 20-gene
partitions (~9.6% of anchored edges are structurally recoverable). inspre works
on the full perturbations x genes ACE matrix, so it has NO partition ceiling —
this is the one backend/scale where the substitutability verdict could flip.

Question: at N in {500,1000,1500}, does real-data-inspre separate from ElasticNet
and from a permutation chance floor, against the NON-CIRCULAR anchored GT? If yes,
there is headroom at scale and GPU VCM inference on these panels is justified. If
everything still collapses to chance, the saturation is fundamental, not a
partition artifact.

Local, no GPU. Conditions: real_data, elastic_net, overlay_resampled_real (perfect
means upper bound). VCM arm (gears/cpa) needs GPU predictions on these panels -> a
separate contingent step, only if this shows headroom.

Run in .venv_gears.
"""
import argparse, json, os, sys, time, tempfile, subprocess
import numpy as np
import anndata as ad
from scipy import stats

RSCRIPT = "/usr/local/bin/Rscript"
ATLAS = "data/k562_real.h5ad"
PERT_COL = "gene"
CTRL = "non-targeting"
MIN_CELLS = 50

R_INSPRE = r'''
library(inspre)
args <- commandArgs(trailingOnly = TRUE)
ace_path <- args[1]; out_path <- args[2]; target_edges <- as.integer(args[3])
R_hat <- as.matrix(read.csv(ace_path, header=FALSE))
result <- fit_inspre_from_R(R_hat = R_hat, its = 200, verbose = 0, train_prop = 0.8)
G_hat <- result$G_hat; test_err <- result$test_error
valid_te <- !is.nan(test_err) & !is.na(test_err)
edge_counts <- sapply(1:dim(G_hat)[3], function(i){ sl<-G_hat[,,i]; sum(abs(sl)>1e-6 & is.finite(sl)) })
if (any(valid_te)) { best_idx <- which(valid_te)[which.min(test_err[valid_te])]
} else { best_idx <- which.min(abs(edge_counts - target_edges)) }
if (length(best_idx)==0 || is.na(best_idx)) best_idx <- which.max(edge_counts)
G_best <- G_hat[,,best_idx]; G_best[!is.finite(G_best)] <- 0
write.csv(G_best, out_path, row.names=FALSE)
'''


def inspre_edges(ace, genes, target_edges=1200, timeout=3000):
    with tempfile.TemporaryDirectory() as td:
        ap = os.path.join(td, "ace.csv"); op = os.path.join(td, "adj.csv"); rp = os.path.join(td, "run.R")
        np.savetxt(ap, ace, delimiter=",")
        open(rp, "w").write(R_INSPRE)
        res = subprocess.run(f"{RSCRIPT} {rp} {ap} {op} {target_edges}",
                             shell=True, capture_output=True, text=True, timeout=timeout)
        if res.returncode != 0:
            raise RuntimeError(f"inspre failed: {res.stderr[-400:]}")
        import pandas as pd
        adj = np.nan_to_num(pd.read_csv(op).values, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(genes)
    return {(genes[i], genes[j]) for i in range(n) for j in range(n) if i != j and abs(adj[i, j]) > 1e-6}


def f1(pred, gt):
    pred, gt = set(pred), set(gt)
    tp = len(pred & gt)
    if tp == 0: return 0.0, 0.0, 0.0
    p = tp / max(1, len(pred)); r = tp / max(1, len(gt))
    return 2 * p * r / (p + r), p, r


def load_panel(N, seed):
    """Pick N genes both perturbed(>=MIN_CELLS) and measured, top by cell count; extract to memory.

    Atlas var_names are Ensembl IDs; perturbation labels are gene SYMBOLS. Bridge via
    var['gene_name']. Work in symbol space (columns renamed to symbols).
    """
    a = ad.read_h5ad(ATLAS, backed="r")
    labels = a.obs[PERT_COL].astype(str).values
    sym = a.var["gene_name"].astype(str).values           # symbol per column
    sym_to_col = {}
    for j, s in enumerate(sym):
        sym_to_col.setdefault(s, j)                        # first occurrence wins
    measured = set(sym_to_col)
    vc = {}
    for g in labels:
        vc[g] = vc.get(g, 0) + 1
    cand = [(g, c) for g, c in vc.items() if g != CTRL and c >= MIN_CELLS and g in measured]
    cand.sort(key=lambda x: -x[1])
    pool = [g for g, _ in cand]
    genes = sorted(pool[:N]) if len(pool) > N else sorted(pool)
    col_idx = [sym_to_col[g] for g in genes]
    keep_labels = set(genes) | {CTRL}
    row_mask = np.isin(labels, list(keep_labels))
    sub = a[row_mask].to_memory()
    Xall = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
    X = Xall[:, col_idx].astype(np.float64)               # columns = panel symbols, in `genes` order
    lab = sub.obs[PERT_COL].astype(str).values
    return genes, X, lab


def build_means(genes, X, lab):
    ctrl = X[lab == CTRL]
    ctrl_mean = ctrl.mean(0)
    real_means = {}
    for g in genes:
        m = lab == g
        if m.sum() >= 1:
            real_means[g] = X[m].mean(0)
    return ctrl, ctrl_mean, real_means


def build_ace(means, ctrl_mean, genes):
    n = len(genes); ace = np.zeros((n, n))
    for i, g in enumerate(genes):
        if g in means:
            ace[i] = means[g] - ctrl_mean
    return np.nan_to_num(ace)


def elasticnet_means(ctrl, genes, seed=0):
    """Predict each gene's perturbation effect from control co-expression (sparse linear)."""
    from sklearn.linear_model import ElasticNet
    n = len(genes)
    # per-target model: predict gene j from all others in control; ACE proxy = -coef direction
    # (matches the paper's ElasticNet baseline: control-only sparse regression)
    means = {}
    Xc = ctrl - ctrl.mean(0)
    for i, g in enumerate(genes):
        y = Xc[:, i]
        Xin = np.delete(Xc, i, axis=1)
        m = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=seed).fit(Xin, y)
        row = np.zeros(n)
        coef = m.coef_
        row[np.arange(n) != i] = coef
        means[g] = row  # effect proxy row
    return means


def anchored_gt(X, lab, genes, q=0.05, tau_fc=0.5):
    gi = {g: i for i, g in enumerate(genes)}
    Xl = np.log1p(X)
    ctrl = Xl[lab == CTRL]
    edges = []
    perts = [g for g in genes if (lab == g).sum() >= 20]
    for A in perts:
        m = lab == A
        pv = np.ones(len(genes)); lfc = np.zeros(len(genes))
        for bi, B in enumerate(genes):
            if B == A: continue
            t, p = stats.ttest_ind(Xl[m, bi], ctrl[:, bi], equal_var=False)
            pv[bi] = p if np.isfinite(p) else 1.0
            lfc[bi] = np.log2((np.expm1(Xl[m, bi].mean()) + 1) / (np.expm1(ctrl[:, bi].mean()) + 1))
        order = np.argsort(pv); ranks = np.empty_like(order); ranks[order] = np.arange(1, len(pv) + 1)
        bh = pv * len(pv) / ranks
        for bi, B in enumerate(genes):
            if B != A and bh[bi] <= q and abs(lfc[bi]) >= tau_fc:
                edges.append((A, B))
    return set(edges), len(perts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 1500])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/instrument_fix/genome_scale_inspre.json")
    args = ap.parse_args()
    sizes = args.sizes[:1] if args.smoke else args.sizes
    seeds = args.seeds[:1] if args.smoke else args.seeds
    conds = ["real_data", "elastic_net", "overlay_resampled_real"]

    out = {"sizes": sizes, "seeds": seeds, "min_cells": MIN_CELLS, "results": {}}
    for N in sizes:
        out["results"][N] = {}
        for seed in seeds:
            t0 = time.time()
            genes, X, lab = load_panel(N, seed)
            ctrl, ctrl_mean, real_means = build_means(genes, X, lab)
            agt, n_perts = anchored_gt(X, lab, genes)
            aces = {"real_data": build_ace(real_means, ctrl_mean, genes)}
            # overlay_resampled_real = perfect means UB; at ACE level identical to real_data ACE
            # (ACE IS the mean shift) -> instead build a mean+Poisson-resampled ACE to mirror the
            # overlay's information loss: resample cells then recompute means.
            rng = np.random.default_rng(seed)
            enet = elasticnet_means(ctrl if ctrl.shape[0] <= 2000 else ctrl[rng.choice(ctrl.shape[0], 2000, replace=False)], genes, seed=seed)
            aces["elastic_net"] = build_ace(enet, ctrl_mean, genes)
            # overlay UB: perfect means passed through overlay-sampling then re-meaned
            ov_means = {}
            for g in genes:
                if g in real_means:
                    ratios = np.ones(len(genes))
                    nz = ctrl_mean > 1e-6
                    ratios[nz] = np.clip(np.maximum(real_means[g][nz], 0) / np.maximum(ctrl_mean, 0.01)[nz], 0.01, 10)
                    base = ctrl[rng.choice(ctrl.shape[0], min(100, ctrl.shape[0]), replace=True)] * ratios[None, :]
                    ov_means[g] = rng.poisson(np.maximum(base, 0)).mean(0)
            aces["overlay_resampled_real"] = build_ace(ov_means, ctrl_mean, genes)

            gt_budget = max(len(agt), 200)
            rec = {"n_genes": len(genes), "anchored_edges": len(agt), "anchor_perts": n_perts, "conditions": {}}
            n_universe = len(genes) * (len(genes) - 1)
            for c in conds:
                tc = time.time()
                edges = inspre_edges(aces[c], genes, target_edges=gt_budget)
                fv, pv, rv = f1(edges, agt)
                etp = len(edges) * len(agt) / n_universe if len(edges) and len(agt) else 0
                rand_f1 = 2 * etp / (len(edges) + len(agt)) if (len(edges) + len(agt)) else 0
                rec["conditions"][c] = {"f1": round(fv, 4), "precision": round(pv, 4), "recall": round(rv, 4),
                                        "n_edges": len(edges), "random_f1": round(rand_f1, 4),
                                        "f1_over_random": round(fv / rand_f1, 2) if rand_f1 else None,
                                        "secs": round(time.time() - tc, 1)}
                print(f"  N={N} seed{seed} {c:22s} F1={fv:.4f} x{(fv/rand_f1 if rand_f1 else 0):.2f}rand edges={len(edges)} ({time.time()-tc:.0f}s)", flush=True)
            out["results"][N][seed] = rec
            print(f"N={N} seed{seed} done ({time.time()-t0:.0f}s) anchored_edges={len(agt)}", flush=True)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            json.dump(out, open(args.out, "w"), indent=2)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
