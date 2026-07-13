#!/usr/bin/env python3
"""A2 — External K562 GRN validation on a coverage-optimized panel.

Fixes the J1 null, which was a gene-SELECTION artifact: the essential-gene panel
had only 3 TRRUST edges. Here we re-pick a 200-gene panel to maximize curated
edge coverage (252 TRRUST edges, 84x more), then ask whether GIES-on-real (and
predictors we can build without a GPU) recover the CURATED external network above
a permutation chance baseline. Also scores against the panel's own
perturbation-anchored GT (A1 method) for cross-reference.

Conditions that can be built locally on the NEW panel: real_data,
overlay_resampled_real (perfect-mean upper bound), elastic_net. The deep VCMs
predicted on the OLD essential panel; re-inferring them on this panel is
GPU-gated and out of scope (stated in reporting).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.run_eval_local import (  # noqa: E402
    build_cb_input,
    run_elasticnet_baseline,
    run_gies,
    sample_perturbed_cells_overlay,
)

PANEL = ROOT / "data" / "ground_truth" / "a2_coverage_panel.json"
SLIM = ROOT / "data" / "k562_a2coverage_n200_slim.h5ad"
OUT = ROOT / "results" / "hardening" / "a2_external_grn.json"
SEEDS = [42, 123, 456, 789, 1024]
N_CELLS, N_CTRL, PARTITION = 100, 2000, 20
CTRL = "non-targeting"
MIN_CELLS = 50


def anchored_edges(adata, genes, q=0.05, fc=0.5):
    """A1-method anchored GT on this panel (perturbation -> responder)."""
    gidx = {g: i for i, g in enumerate(genes)}
    labels = adata.obs["gene"].astype(str).values
    X = adata.X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)
    lib = X.sum(1, keepdims=True)
    med = np.median(lib[lib > 0])
    Xn = np.log1p(X / np.maximum(lib, 1e-9) * med)
    ctrl = Xn[labels == CTRL]
    ctrl_lin = np.expm1(ctrl.mean(0))
    edges = set()
    for A in [g for g in np.unique(labels) if g != CTRL and g in gidx]:
        pm = labels == A
        if pm.sum() < 3:
            continue
        pert = Xn[pm]
        t, p = stats.ttest_ind(pert, ctrl, axis=0, equal_var=False)
        p = np.nan_to_num(p, nan=1.0)
        l2 = np.log2((np.expm1(pert.mean(0)) + 1e-3) / (ctrl_lin + 1e-3))
        # BH
        order = np.argsort(p); n = p.size
        r = p[order] * n / (np.arange(n) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        fdr = np.empty(n); fdr[order] = np.clip(r, 0, 1)
        ai = gidx[A]
        for bi in range(len(genes)):
            if bi == ai:
                continue
            if fdr[bi] <= q and abs(l2[bi]) >= fc:
                edges.add((A, genes[bi]))
    return edges


def score(pred, gt):
    pred, gt = set(pred), set(gt)
    if not pred or not gt:
        return dict(f1=0.0, precision=0.0, recall=0.0, tp=0, n_pred=len(pred))
    tp = len(pred & gt)
    prec = tp / len(pred); rec = tp / len(gt)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return dict(f1=f1, precision=prec, recall=rec, tp=tp, n_pred=len(pred))


def perm_chance(pred_edges, gt_edges, genes, n_perm=1000, seed=0):
    """Chance recall: shuffle GT targets, recompute precision/recall lift."""
    rng = np.random.default_rng(seed)
    genes = list(genes)
    gt = list(gt_edges)
    pred = set(pred_edges)
    obs_tp = len(pred & set(gt))
    null_tp = []
    for _ in range(n_perm):
        shuf = {(a, genes[rng.integers(len(genes))]) for a, b in gt}
        null_tp.append(len(pred & shuf))
    null_tp = np.array(null_tp)
    mean_null = null_tp.mean() + 1e-9
    return dict(obs_tp=int(obs_tp), null_tp_mean=float(mean_null),
                lift=float(obs_tp / mean_null),
                p_value=float((null_tp >= obs_tp).mean()))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel = json.load(open(PANEL))
    genes = panel["panel_genes"]
    trrust = {(a, b) for a, b in panel["trrust_edges"]}

    a = ad.read_h5ad(SLIM)
    sym = a.var["gene_name"].astype(str).values
    assert list(sym) == list(genes), "panel gene order mismatch"
    labels = a.obs["gene"].astype(str).values
    X = a.X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)

    ctrl_X_full = X[labels == CTRL]
    # eligible perturbed genes on this panel
    from collections import Counter
    cnt = Counter(labels)
    pert_genes = [g for g in genes if cnt.get(g, 0) >= MIN_CELLS and g != CTRL]

    anch = anchored_edges(a, genes)
    print(json.dumps({"panel_genes": len(genes), "trrust_edges": len(trrust),
                      "anchored_edges": len(anch), "eligible_perts": len(pert_genes)}))

    results = {"meta": {**panel["meta"], "anchored_edges_on_panel": len(anch),
                        "eligible_perts": len(pert_genes)},
               "seeds": SEEDS, "conditions": {}}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        # subsample controls for GIES
        ci = rng.choice(ctrl_X_full.shape[0], size=min(N_CTRL, ctrl_X_full.shape[0]),
                        replace=False)
        ctrl_X = ctrl_X_full[ci]

        conds = {}
        # real_data: real perturbed cells
        rmats, rlabs = [], []
        for g in pert_genes:
            gc = X[labels == g]
            if gc.shape[0] > N_CELLS:
                gc = gc[rng.choice(gc.shape[0], N_CELLS, replace=False)]
            rmats.append(gc); rlabs += [g] * gc.shape[0]
        conds["real_data"] = build_cb_input(ctrl_X, rmats, rlabs, genes)

        # overlay_resampled_real: perfect means -> overlay
        omats, olabs = [], []
        for g in pert_genes:
            gc = X[labels == g]
            mean_g = gc.mean(0)
            gs = seed + int(hashlib.md5(g.encode()).hexdigest()[:8], 16)
            omats.append(sample_perturbed_cells_overlay(ctrl_X, mean_g, N_CELLS, seed=gs))
            olabs += [g] * N_CELLS
        conds["overlay_resampled_real"] = build_cb_input(ctrl_X, omats, olabs, genes)

        # elastic_net (only over eligible perts to match real_data support)
        # run_elasticnet_baseline builds over all gene_subset; restrict to pert_genes
        emats, elabs = run_elasticnet_baseline(ctrl_X, ctrl_X, pert_genes, N_CELLS, seed=seed)
        conds["elastic_net"] = build_cb_input(ctrl_X, emats, elabs, genes)

        for cname, cb in conds.items():
            edges = set(run_gies(cb, seed=seed, partition_size=PARTITION))
            row = results["conditions"].setdefault(cname, {"trrust": [], "anchored": [],
                                                            "trrust_lift": [], "n_edges": []})
            row["n_edges"].append(len(edges))
            row["trrust"].append(score(edges, trrust))
            row["anchored"].append(score(edges, anch))
            row["trrust_lift"].append(perm_chance(edges, trrust, genes, seed=seed))
        print(f"seed {seed} done: " + ", ".join(
            f"{c}={results['conditions'][c]['trrust'][-1]['tp']}tp/"
            f"{results['conditions'][c]['trrust'][-1]['n_pred']}"
            for c in conds))

    # aggregate
    for c, row in results["conditions"].items():
        row["summary"] = {
            "trrust_f1_mean": float(np.mean([x["f1"] for x in row["trrust"]])),
            "trrust_prec_mean": float(np.mean([x["precision"] for x in row["trrust"]])),
            "trrust_recall_mean": float(np.mean([x["recall"] for x in row["trrust"]])),
            "trrust_lift_mean": float(np.mean([x["lift"] for x in row["trrust_lift"]])),
            "trrust_lift_sd": float(np.std([x["lift"] for x in row["trrust_lift"]], ddof=1)),
            "trrust_p_median": float(np.median([x["p_value"] for x in row["trrust_lift"]])),
            "anchored_f1_mean": float(np.mean([x["f1"] for x in row["anchored"]])),
            "n_edges_mean": float(np.mean(row["n_edges"])),
        }
    json.dump(results, open(OUT, "w"), indent=2)
    print("\n=== TRRUST lift over chance (mean±sd, median p) ===")
    for c, row in results["conditions"].items():
        s = row["summary"]
        print(f"{c:24s} lift={s['trrust_lift_mean']:.2f}±{s['trrust_lift_sd']:.2f} "
              f"p={s['trrust_p_median']:.3f} TRRUST-F1={s['trrust_f1_mean']:.4f} "
              f"anchored-F1={s['anchored_f1_mean']:.4f}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
