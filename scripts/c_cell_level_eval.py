#!/usr/bin/env python
"""C: cell-level vs mean+overlay causal evaluation (prereg_C_cell_level_eval.md).

Four conditions, identical panel/partitions/edge-budget/GT per seed:
  A real cells        -> GIES        (real joint structure; the ceiling)
  B overlay(real mean)-> GIES        (what the paper currently reports)
  C predicted cells   -> GIES        (model joint; the FIXED instrument)
  D overlay(pred mean)-> GIES        (same model, current instrument)

Decisive tests (paired within seed):
  T1  C > D  (does the model's joint structure reach GIES when cells bypass overlay?)
  T2  A > B  (does real joint structure carry recoverable edges the overlay discards?)

Run in .venv_gears. Dataset-agnostic: pass --dataset rpe1|norman.
"""
import argparse, json, os, sys, time
import numpy as np
import anndata as ad

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from run_eval_local import (
    build_cb_input, run_gies, sample_perturbed_cells_overlay,
    compute_set_metrics,
)

DATASETS = {
    "rpe1": dict(
        pred="data/model_outputs_real_rpe1/cpa_predictions.h5ad",
        real="data/rpe1_subset_n200_slim.h5ad",
        panel_json="results/phase2_eval_rpe1.json",
        real_pert_col="gene", pred_pert_col="perturbation",
        ctrl_label="non-targeting",
    ),
    "norman": dict(
        pred="data/model_outputs_real_norman/cpa_predictions.h5ad",
        real="data/norman_subset_n200_slim.h5ad",
        panel_json="results/phase2_eval_norman.json",
        real_pert_col="perturbation", pred_pert_col="perturbation",
        ctrl_label="control",
    ),
}


def anchored_gt(real, pert_col, ctrl_label, panel, q=0.05, tau_fc=0.5):
    """Perturbation-anchored GT: knock out A -> B changes => edge A->B. Welch t-test + BH-FDR."""
    from scipy import stats
    gi = {g: i for i, g in enumerate(panel)}
    X = real[:, panel].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = np.log1p(X.astype(np.float64))
    labels = real.obs[pert_col].astype(str).values
    ctrl_mask = labels == ctrl_label
    ctrl = X[ctrl_mask]
    edges, pvals, effects, srcs, tgts = [], [], [], [], []
    perturbed = [g for g in panel if (labels == g).sum() >= 20 and g in gi]
    for A in perturbed:
        m = labels == A
        if m.sum() < 20:
            continue
        pv = np.full(len(panel), 1.0); lfc = np.zeros(len(panel))
        for bi, B in enumerate(panel):
            if B == A:
                continue
            t, p = stats.ttest_ind(X[m, bi], ctrl[:, bi], equal_var=False)
            pv[bi] = p if np.isfinite(p) else 1.0
            lfc[bi] = np.log2((np.expm1(X[m, bi].mean()) + 1) / (np.expm1(ctrl[:, bi].mean()) + 1))
        # BH-FDR across the tested B for this A
        order = np.argsort(pv); ranks = np.empty_like(order); ranks[order] = np.arange(1, len(pv) + 1)
        bh = pv * len(pv) / ranks
        for bi, B in enumerate(panel):
            if B == A:
                continue
            if bh[bi] <= q and abs(lfc[bi]) >= tau_fc:
                edges.append((A, B))
    return set(edges), len(perturbed)


def build_conditions(ds, panel, n_cells, seed):
    """Return dict name-> (pert_matrices, pert_labels, ctrl_X). All share ctrl_X & panel."""
    real = ad.read_h5ad(ds["real"])
    pred = ad.read_h5ad(ds["pred"])
    gi_real = {g: i for i, g in enumerate(real.var_names)}
    gi_pred = {g: i for i, g in enumerate(pred.var_names)}
    pcols_real = [g for g in panel if g in gi_real]
    assert pcols_real == panel, "panel gene missing from real"
    def sub(adata, cols):
        X = adata[:, cols].X
        return (X.toarray() if hasattr(X, "toarray") else np.asarray(X)).astype(np.float32)
    real_lab = real.obs[ds["real_pert_col"]].astype(str).values
    pred_lab = pred.obs[ds["pred_pert_col"]].astype(str).values
    ctrl_mask = real_lab == ds["ctrl_label"]
    ctrl_X = sub(real, panel)[ctrl_mask]
    rng = np.random.default_rng(seed)
    if ctrl_X.shape[0] > 2000:
        ctrl_X = ctrl_X[rng.choice(ctrl_X.shape[0], 2000, replace=False)]

    perturbed = [g for g in panel if (real_lab == g).sum() >= 20 and (pred_lab == g).sum() >= 1]
    A_mats, B_mats, C_mats, D_mats, labs = [], [], [], [], []
    realX_all = sub(real, panel); predX_all = sub(pred, panel)
    for g in perturbed:
        rm = real_lab == g; pm = pred_lab == g
        real_cells = realX_all[rm]; pred_cells = predX_all[pm]
        if real_cells.shape[0] > n_cells:
            real_cells = real_cells[rng.choice(real_cells.shape[0], n_cells, replace=False)]
        if pred_cells.shape[0] > n_cells:
            pred_cells = pred_cells[rng.choice(pred_cells.shape[0], n_cells, replace=False)]
        real_mean = realX_all[rm].mean(0)
        pred_mean = predX_all[pm].mean(0)
        # A real cells; C predicted cells; B overlay(real mean); D overlay(pred mean)
        A_mats.append(real_cells)
        C_mats.append(pred_cells)
        B_mats.append(sample_perturbed_cells_overlay(ctrl_X, real_mean, n_cells, seed))
        D_mats.append(sample_perturbed_cells_overlay(ctrl_X, pred_mean, n_cells, seed))
        labs.append(g)
    def expand(mats):
        pl = []
        for g, m in zip(labs, mats):
            pl.extend([g] * m.shape[0])
        return mats, pl
    return {
        "A_real_cells": expand(A_mats),
        "B_overlay_real": expand(B_mats),
        "C_pred_cells": expand(C_mats),
        "D_overlay_pred": expand(D_mats),
    }, ctrl_X, perturbed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rpe1", choices=list(DATASETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    ap.add_argument("--partition-size", type=int, default=20)
    ap.add_argument("--n-cells", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="1 seed, quick")
    args = ap.parse_args()
    ds = DATASETS[args.dataset]
    panel = json.load(open(ds["panel_json"]))["gene_subset"]
    seeds = args.seeds[:1] if args.smoke else args.seeds

    # anchored GT once (uses full real cells, not seed-dependent)
    real = ad.read_h5ad(ds["real"])
    agt, n_anchor_perts = anchored_gt(real, ds["real_pert_col"], ds["ctrl_label"], panel)
    print(f"[{args.dataset}] panel={len(panel)} anchored_edges={len(agt)} anchor_perts={n_anchor_perts}", flush=True)
    del real

    conds = ["A_real_cells", "B_overlay_real", "C_pred_cells", "D_overlay_pred"]
    per_seed = {c: {} for c in conds}
    circ_f1 = {c: {} for c in conds}   # vs GIES-on-real-cells (A's edges)
    edge_sets = {c: {} for c in conds}
    for seed in seeds:
        t0 = time.time()
        cdata, ctrl_X, perturbed = build_conditions(ds, panel, args.n_cells, seed)
        # SAME seed -> run_gies uses same partition permutation for all 4 conditions
        edges_by = {}
        for c in conds:
            mats, labs = cdata[c]
            cb = build_cb_input(ctrl_X, mats, labs, panel)
            edges = run_gies(cb, seed=seed, partition_size=args.partition_size)
            edges_by[c] = set(edges)
            edge_sets[c][seed] = len(edges)
        gt_circ = edges_by["A_real_cells"]  # condition A's own edges = circular GT
        for c in conds:
            m_anch = compute_set_metrics(list(edges_by[c]), agt, panel)
            per_seed[c][seed] = m_anch["f1"]
            m_circ = compute_set_metrics(list(edges_by[c]), gt_circ, panel)
            circ_f1[c][seed] = m_circ["f1"]
        # jaccard C vs D and A vs B (edge-set overlap)
        def jac(x, y):
            return len(x & y) / max(len(x | y), 1)
        if seed == seeds[0]:
            jac_CD = jac(edges_by["C_pred_cells"], edges_by["D_overlay_pred"])
            jac_AB = jac(edges_by["A_real_cells"], edges_by["B_overlay_real"])
        print(f"  seed{seed} ({time.time()-t0:.0f}s): "
              + " ".join(f"{c[0]}={per_seed[c][seed]:.4f}" for c in conds), flush=True)

    def stat(d):
        v = np.array([d[s] for s in seeds]); return float(v.mean()), float(v.std())
    def gate(hi, lo):
        h = np.array([per_seed[hi][s] for s in seeds]); l = np.array([per_seed[lo][s] for s in seeds])
        diff = h - l; nwin = int((diff > 0).sum())
        return dict(mean_gain=float(diff.mean()), sd=float(diff.std()),
                    n_win=nwin, n_seeds=len(seeds),
                    keep=bool(nwin >= 4 and diff.mean() > diff.std()))

    out = dict(
        dataset=args.dataset, panel_genes=len(panel), seeds=seeds,
        partition_size=args.partition_size, n_cells=args.n_cells,
        anchored_edges=len(agt), anchor_perts=n_anchor_perts,
        anchored_f1={c: dict(mean=stat(per_seed[c])[0], sd=stat(per_seed[c])[1],
                             per_seed=per_seed[c]) for c in conds},
        circular_f1={c: dict(mean=stat(circ_f1[c])[0], sd=stat(circ_f1[c])[1],
                             per_seed=circ_f1[c]) for c in conds},
        n_edges={c: edge_sets[c] for c in conds},
        jaccard_CD=jac_CD, jaccard_AB=jac_AB,
        T1_C_vs_D=gate("C_pred_cells", "D_overlay_pred"),
        T2_A_vs_B=gate("A_real_cells", "B_overlay_real"),
    )
    outpath = args.out or f"results/instrument_fix/c_cell_level_{args.dataset}.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    json.dump(out, open(outpath, "w"), indent=2)
    print("\n=== SUMMARY (anchored GT) ===", flush=True)
    for c in conds:
        print(f"  {c:16s} F1={out['anchored_f1'][c]['mean']:.4f}±{out['anchored_f1'][c]['sd']:.4f}", flush=True)
    print(f"  T1 C>D: {out['T1_C_vs_D']['n_win']}/{len(seeds)} gain "
          f"{out['T1_C_vs_D']['mean_gain']:+.4f}±{out['T1_C_vs_D']['sd']:.4f} KEEP={out['T1_C_vs_D']['keep']}", flush=True)
    print(f"  T2 A>B: {out['T2_A_vs_B']['n_win']}/{len(seeds)} gain "
          f"{out['T2_A_vs_B']['mean_gain']:+.4f}±{out['T2_A_vs_B']['sd']:.4f} KEEP={out['T2_A_vs_B']['keep']}", flush=True)
    print(f"  Jaccard(C,D)={out['jaccard_CD']:.3f}  Jaccard(A,B)={out['jaccard_AB']:.3f}", flush=True)
    print(f"saved {outpath}", flush=True)


if __name__ == "__main__":
    main()
