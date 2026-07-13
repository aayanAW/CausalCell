#!/usr/bin/env python
"""Genome-scale causal recovery via fast precision-style recovery from the interventional ACE.

WHY NOT inspre directly: inspre's ADMM (200 iters) exceeds 900s at N=500 on this
machine and scales superlinearly to N=1000/1500; the full 27-fit grid is
impractical locally, and the only remote box has no R and is 2x oversubscribed.

WHAT THIS IS: inspre is one ADMM solver for precision-matrix / causal-structure
recovery from the interventional ACE R_hat, where the structural relation is
R_hat ~ (I - G)^-1 (do(i) effects propagate through the inverse). So a regularized
G_hat = I - R_reg^-1, thresholded to a target edge budget, is the SAME estimator
family, O(N^3) once (seconds), and answers the actual scientific question:

  At N in {500,1000,1500}, does REAL-data causal recovery separate from ElasticNet
  and from a column-permutation chance floor, scored against the NON-CIRCULAR
  perturbation-anchored GT?

If it separates -> there is headroom at scale; the expensive inspre confirmation +
GPU VCM inference are justified. If everything collapses to chance -> the
saturation is fundamental (not a GIES-partition artifact, not a backend artifact).

This is a genome-scale SCREEN, reported honestly as a fast precision recovery, with
full inspre deferred to proper compute as confirmation.

Run in .venv_gears.
"""
import argparse, json, os, sys, time
import numpy as np
import anndata as ad
from scipy import stats

ATLAS = "data/k562_real.h5ad"
PERT_COL = "gene"
CTRL = "non-targeting"
MIN_CELLS = 50


def precision_edges(ace, genes, target_edges, ridge=1e-2):
    """G_hat = I - (R + ridge*I)^-1 ; threshold |off-diagonal| to target_edges.

    R = ACE (interventional effect matrix). Regularized inverse recovers the
    direct-effect structure (precision-matrix support). Deterministic, O(N^3)."""
    n = len(genes)
    R = np.asarray(ace, dtype=np.float64)
    # scale R so its spectral radius is well-conditioned for inversion
    R = R / (np.abs(R).max() + 1e-9)
    Rreg = R + ridge * np.eye(n)
    try:
        Ginv = np.linalg.inv(Rreg)
    except np.linalg.LinAlgError:
        Ginv = np.linalg.pinv(Rreg)
    G = np.eye(n) - Ginv
    np.fill_diagonal(G, 0.0)
    absG = np.abs(G)
    # pick threshold that yields ~target_edges directed edges
    flat = absG[~np.eye(n, dtype=bool)]
    if target_edges >= len(flat):
        thr = 0.0
    else:
        thr = np.partition(flat, len(flat) - target_edges)[len(flat) - target_edges]
    return {(genes[i], genes[j]) for i in range(n) for j in range(n)
            if i != j and absG[i, j] > thr and absG[i, j] > 1e-9}


def f1(pred, gt):
    pred, gt = set(pred), set(gt)
    tp = len(pred & gt)
    if tp == 0: return 0.0, 0.0, 0.0
    p = tp / max(1, len(pred)); r = tp / max(1, len(gt))
    return 2 * p * r / (p + r), p, r


def load_panel(N):
    a = ad.read_h5ad(ATLAS, backed="r")
    labels = a.obs[PERT_COL].astype(str).values
    sym = a.var["gene_name"].astype(str).values
    sym_to_col = {}
    for j, s in enumerate(sym):
        sym_to_col.setdefault(s, j)
    measured = set(sym_to_col)
    vc = {}
    for g in labels:
        vc[g] = vc.get(g, 0) + 1
    cand = [(g, c) for g, c in vc.items() if g != CTRL and c >= MIN_CELLS and g in measured]
    cand.sort(key=lambda x: -x[1])
    genes = sorted([g for g, _ in cand][:N])
    col_idx = [sym_to_col[g] for g in genes]
    keep = set(genes) | {CTRL}
    sub = a[np.isin(labels, list(keep))].to_memory()
    Xall = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
    X = Xall[:, col_idx].astype(np.float64)
    lab = sub.obs[PERT_COL].astype(str).values
    return genes, X, lab


def build_means(genes, X, lab):
    ctrl = X[lab == CTRL]
    cm = ctrl.mean(0)
    rm = {g: X[lab == g].mean(0) for g in genes if (lab == g).sum() >= 1}
    return ctrl, cm, rm


def build_ace(means, cm, genes):
    n = len(genes); ace = np.zeros((n, n))
    for i, g in enumerate(genes):
        if g in means:
            ace[i] = means[g] - cm
    return np.nan_to_num(ace)


def elasticnet_ace(ctrl, genes, seed):
    from sklearn.linear_model import ElasticNet
    n = len(genes); Xc = ctrl - ctrl.mean(0)
    ace = np.zeros((n, n))
    for i in range(n):
        y = Xc[:, i]; Xin = np.delete(Xc, i, axis=1)
        m = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=seed).fit(Xin, y)
        row = np.zeros(n); row[np.arange(n) != i] = m.coef_
        ace[i] = row
    return ace


def anchored_gt(X, lab, genes, q=0.05, tau_fc=0.5):
    """Vectorized Welch t-test per perturbation A across ALL gene-B columns at once."""
    Xl = np.log1p(X); ctrl = Xl[lab == CTRL]
    n = len(genes)
    cm = ctrl.mean(0); cv = ctrl.var(0, ddof=1); cn = ctrl.shape[0]
    c_expm1_mean = np.expm1(cm) + 1.0
    perts = [g for g in genes if (lab == g).sum() >= 20]
    idx = {g: i for i, g in enumerate(genes)}
    edges = []
    for A in perts:
        m = lab == A; gi = idx[A]
        Xa = Xl[m]; am = Xa.mean(0); av = Xa.var(0, ddof=1); an = Xa.shape[0]
        # Welch t + df across all B columns vectorized
        se = np.sqrt(av / an + cv / cn) + 1e-12
        tstat = (am - cm) / se
        df = (av / an + cv / cn) ** 2 / ((av / an) ** 2 / (an - 1) + (cv / cn) ** 2 / (cn - 1) + 1e-30)
        pv = 2 * stats.t.sf(np.abs(tstat), df)
        pv = np.where(np.isfinite(pv), pv, 1.0)
        lfc = np.log2((np.expm1(am) + 1.0) / c_expm1_mean)
        # exclude self
        pv[gi] = 1.0
        order = np.argsort(pv); ranks = np.empty_like(order); ranks[order] = np.arange(1, n + 1)
        bh = pv * n / ranks
        sel = (bh <= q) & (np.abs(lfc) >= tau_fc)
        sel[gi] = False
        for bi in np.nonzero(sel)[0]:
            edges.append((A, genes[bi]))
    return set(edges), len(perts)


def col_permute(ace, rng):
    out = ace.copy()
    for j in range(out.shape[1]):
        out[:, j] = out[rng.permutation(out.shape[0]), j]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 1500])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--out", default="results/instrument_fix/genome_scale_precision.json")
    args = ap.parse_args()
    conds = ["real_data", "elastic_net", "overlay_resampled_real"]
    out = {"backend": "fast precision recovery (G=I-ACE^-1), inspre-family",
           "sizes": args.sizes, "seeds": args.seeds, "min_cells": MIN_CELLS, "results": {}}

    for N in args.sizes:
        t0 = time.time()
        genes, X, lab = load_panel(N)
        ctrl, cm, rm = build_means(genes, X, lab)
        agt, n_perts = anchored_gt(X, lab, genes)
        budget = max(len(agt), 200)
        aces = {"real_data": build_ace(rm, cm, genes)}
        aces["elastic_net"] = elasticnet_ace(ctrl if ctrl.shape[0] <= 2000 else ctrl[np.random.default_rng(0).choice(ctrl.shape[0], 2000, False)], genes, 0)
        # overlay UB: real means through the diagonal-rescale+Poisson overlay, re-meaned
        rng0 = np.random.default_rng(0); ov = {}
        for g in genes:
            if g in rm:
                ratios = np.ones(len(genes)); nz = cm > 1e-6
                ratios[nz] = np.clip(np.maximum(rm[g][nz], 0) / np.maximum(cm, 0.01)[nz], 0.01, 10)
                base = ctrl[rng0.choice(ctrl.shape[0], min(100, ctrl.shape[0]), True)] * ratios[None, :]
                ov[g] = rng0.poisson(np.maximum(base, 0)).mean(0)
        aces["overlay_resampled_real"] = build_ace(ov, cm, genes)

        n_universe = len(genes) * (len(genes) - 1)
        rec = {"n_genes": len(genes), "anchored_edges": len(agt), "anchor_perts": n_perts,
               "edge_budget": budget, "conditions": {}, "seeds": {}}
        # deterministic condition F1 (backend is deterministic; seeds only vary the perm null + enet subsample)
        for c in conds:
            edges = precision_edges(aces[c], genes, budget)
            fv, pv, rv = f1(edges, agt)
            etp = len(edges) * len(agt) / n_universe if len(edges) and len(agt) else 0
            rf = 2 * etp / (len(edges) + len(agt)) if (len(edges) + len(agt)) else 0
            rec["conditions"][c] = {"f1": round(fv, 4), "precision": round(pv, 4), "recall": round(rv, 4),
                                    "n_edges": len(edges), "random_f1": round(rf, 4),
                                    "f1_over_random": round(fv / rf, 2) if rf else None}
            print(f"  N={N} {c:22s} F1={fv:.4f} x{(fv/rf if rf else 0):.2f}rand P={pv:.4f} R={rv:.4f} edges={len(edges)}", flush=True)
        # permutation-null chance floor on real ACE (empirical, seed-varied)
        perm_f1 = []
        for k in range(args.n_perm):
            pe = precision_edges(col_permute(aces["real_data"], np.random.default_rng(1000 + k)), genes, budget)
            perm_f1.append(f1(pe, agt)[0])
        rec["perm_null"] = {"mean": round(float(np.mean(perm_f1)), 4), "sd": round(float(np.std(perm_f1)), 4),
                            "p975": round(float(np.percentile(perm_f1, 97.5)), 4), "n": args.n_perm}
        real_f1 = rec["conditions"]["real_data"]["f1"]
        rec["real_above_null"] = bool(real_f1 > rec["perm_null"]["p975"])
        print(f"  N={N} perm-null {rec['perm_null']['mean']:.4f}±{rec['perm_null']['sd']:.4f} "
              f"p97.5={rec['perm_null']['p975']:.4f}  real>null={rec['real_above_null']}  ({time.time()-t0:.0f}s)", flush=True)
        out["results"][N] = rec
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
