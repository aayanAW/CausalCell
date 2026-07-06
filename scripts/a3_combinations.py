#!/usr/bin/env python3
"""A3 — Combination-perturbation causal recovery (held-out Norman doubles).

Pre-registered in prereg_A3_combinations.md.

131 double perturbations (73 genes, all with singles, 71/73 measured) are held
out. We ask: on the unseen doubles -- where an ADDITIVE/linear baseline provably
cannot capture epistasis -- does any predictor recover the doubles' causal
structure better than the additive baseline?

Predictors (both buildable locally):
  real_doubles          : real held-out double cells (within-regime reference).
  additive_singles      : predicted double mean = ctrl + (single_A - ctrl) +
                          (single_B - ctrl); the explicit no-epistasis baseline
                          built from REAL singles (a strong additive oracle). This
                          is the linear/additive arm: a predictor that composes
                          single effects additively CANNOT capture epistasis.
Ground truth (both scored, per prereg_A3_combinations.md):
  PRIMARY   = GIES-on-real-doubles (within-regime reference: the doubles' own
              real-data causal structure). Each predictor's recovered edges are
              scored vs the edges GIES recovers from the REAL held-out doubles.
  SECONDARY = perturbation-anchored edges from the REAL doubles (A1 method):
              double (A,B) -> responder C when C shifts vs control. Causal-by-
              construction and non-circular.
NOTE (deviation from locked prereg): the panel keeps only doubles whose BOTH
parent genes are measured on the panel AND have >= MIN_CELLS single-perturbation
cells (needed to build the additive-of-singles arm). This drops 131 -> 129
doubles (2 excluded); disclosed in results meta as n_doubles and
n_doubles_prereg.

VCM double-prediction (GEARS/CPA on unseen pairs) requires GPU inference and is
out of scope locally; the additive baselines are the informative local contrast.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.run_eval_local import (  # noqa: E402
    build_cb_input,
    run_gies,
    sample_perturbed_cells_overlay,
)

NORMAN = ROOT / "data" / "norman2019.h5ad"
OUT = ROOT / "results" / "hardening" / "a3_combinations.json"
SEEDS = [42, 123, 456, 789, 1024]
N_CELLS, N_CTRL, PARTITION = 100, 2000, 20
MIN_CELLS = 20


def bh(p):
    p = np.asarray(p, float); n = p.size
    o = np.argsort(p); r = p[o] * n / (np.arange(n) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(r, 0, 1)
    return out


def anchored_from_doubles(dbl_means, ctrl_mean_lin, ctrl_log, X_by_pert_log, genes,
                          q=0.05, fc=0.5):
    """Anchored GT: double (A,B) -> C when C shifts vs control (Welch t + FC)."""
    gidx = {g: i for i, g in enumerate(genes)}
    edges = set()
    for dbl, pert_log in X_by_pert_log.items():
        if pert_log.shape[0] < 3:
            continue
        t, p = stats.ttest_ind(pert_log, ctrl_log, axis=0, equal_var=False)
        p = np.nan_to_num(p, nan=1.0)
        l2 = np.log2((np.expm1(pert_log.mean(0)) + 1e-3) / (ctrl_mean_lin + 1e-3))
        fdr = bh(p)
        parents = [g for g in dbl.split("_") if g in gidx]
        for bi in range(len(genes)):
            if genes[bi] in parents:
                continue
            if fdr[bi] <= q and abs(l2[bi]) >= fc:
                for A in parents:
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


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(NORMAN)
    sym = a.var_names.astype(str).values
    pert = a.obs["perturbation"].astype(str).values
    npv = a.obs["nperts"].astype(int).values
    X = a.X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)

    doubles = sorted(set(pert[npv == 2]))
    # panel = measured double-genes that have a single
    single_set = set(pert[npv == 1])
    dbl_genes = set()
    for d in doubles:
        dbl_genes |= set(d.split("_"))
    genes = sorted(g for g in dbl_genes if g in set(sym) and g in single_set)
    gidx = {g: i for i, g in enumerate(genes)}
    col = [np.where(sym == g)[0][0] for g in genes]

    # keep only doubles whose BOTH parents are in the panel
    doubles = [d for d in doubles if all(g in gidx for g in d.split("_"))]

    # log-normalize on panel columns
    Xp = X[:, col]
    lib = X.sum(1, keepdims=True); med = np.median(lib[lib > 0])
    Xlog = np.log1p(Xp / np.maximum(lib, 1e-9) * med)

    ctrl_mask = npv == 0
    ctrl_log = Xlog[ctrl_mask]
    ctrl_lin = np.expm1(ctrl_log.mean(0))
    ctrl_raw = Xp[ctrl_mask]  # raw counts on panel for overlay sampling
    ctrl_raw_mean = ctrl_raw.mean(0)

    # per-double real cells (raw) + logs for anchored GT
    dbl_cells_raw, dbl_log = {}, {}
    for d in doubles:
        m = pert == d
        if m.sum() < MIN_CELLS:
            continue
        dbl_cells_raw[d] = Xp[m]
        dbl_log[d] = Xlog[m]
    doubles = list(dbl_cells_raw.keys())

    # single means (raw) for additive baseline
    single_mean_raw = {}
    for g in genes:
        m = pert == g
        if m.sum() >= MIN_CELLS:
            single_mean_raw[g] = Xp[m].mean(0)

    # doubles usable by additive-of-singles (both singles present)
    add_doubles = [d for d in doubles if all(g in single_mean_raw for g in d.split("_"))]

    anch = anchored_from_doubles(None, ctrl_lin, ctrl_log, dbl_log, genes)
    meta = dict(panel_genes=len(genes), n_doubles=len(doubles), n_doubles_prereg=131,
                n_additive_doubles=len(add_doubles), anchored_edges=len(anch),
                min_cells=MIN_CELLS,
                deviation="kept doubles with both parents measured AND >=MIN_CELLS singles; 131->{} (2 dropped)".format(len(doubles)),
                scoring="PRIMARY=GIES-on-real-doubles (within-regime); SECONDARY=anchored")
    print(json.dumps(meta))

    results = {"meta": meta, "seeds": SEEDS, "conditions": {}}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        ci = rng.choice(ctrl_raw.shape[0], size=min(N_CTRL, ctrl_raw.shape[0]), replace=False)
        ctrl_X = ctrl_raw[ci]

        conds = {}
        # real_doubles
        rm, rl = [], []
        for d in add_doubles:
            gc = dbl_cells_raw[d]
            if gc.shape[0] > N_CELLS:
                gc = gc[rng.choice(gc.shape[0], N_CELLS, replace=False)]
            rm.append(gc); rl += [d] * gc.shape[0]
        conds["real_doubles"] = build_cb_input(ctrl_X, rm, rl, genes)

        # additive_singles: mean = ctrl + (sA-ctrl) + (sB-ctrl)
        am, al = [], []
        for d in add_doubles:
            A, B = d.split("_")
            mean_pred = np.maximum(ctrl_raw_mean + (single_mean_raw[A] - ctrl_raw_mean)
                                   + (single_mean_raw[B] - ctrl_raw_mean), 0.0)
            gs = seed + int(hashlib.md5(d.encode()).hexdigest()[:8], 16)
            am.append(sample_perturbed_cells_overlay(ctrl_X, mean_pred, N_CELLS, seed=gs))
            al += [d] * N_CELLS
        conds["additive_singles"] = build_cb_input(ctrl_X, am, al, genes)

        # PRIMARY reference: GIES on the REAL doubles (within-regime GT).
        # NOTE: real_doubles scored against this GT is self-identity (F1==1.0);
        # it is the trivial CEILING, not a competitor. The primary metric is
        # informative ONLY for non-real predictors (additive_singles here, and
        # would-be VCMs): what fraction of the real doubles' own causal
        # structure they recover. A permutation shuffle gives the chance floor.
        gies_edges = {c: set(run_gies(cb, seed=seed, partition_size=PARTITION))
                      for c, cb in conds.items()}
        primary_gt = gies_edges["real_doubles"]
        rng2 = np.random.default_rng(seed + 7)
        gl = list(genes)
        shuf_gt = {(a, gl[rng2.integers(len(gl))]) for a, b in primary_gt}
        for cname in conds:
            edges = gies_edges[cname]
            row = results["conditions"].setdefault(cname, {"anchored": [], "primary": [],
                                                            "primary_chance": [], "n_edges": []})
            row["n_edges"].append(len(edges))
            row["anchored"].append(score(edges, anch))       # SECONDARY (non-circular)
            row["primary"].append(score(edges, primary_gt))  # PRIMARY (within-regime)
            row["primary_chance"].append(score(edges, shuf_gt))
        print(f"seed {seed}: " + ", ".join(
            f"{c}=anch{results['conditions'][c]['anchored'][-1]['f1']:.3f}/"
            f"prim{results['conditions'][c]['primary'][-1]['f1']:.3f}" for c in conds))

    for c, row in results["conditions"].items():
        f1a = [x["f1"] for x in row["anchored"]]
        f1p = [x["f1"] for x in row["primary"]]
        f1pc = [x["f1"] for x in row["primary_chance"]]
        row["summary"] = dict(
            anchored_f1_mean=float(np.mean(f1a)), anchored_f1_sd=float(np.std(f1a, ddof=1)),
            anchored_per_seed=[round(x, 4) for x in f1a],
            primary_f1_mean=float(np.mean(f1p)), primary_f1_sd=float(np.std(f1p, ddof=1)),
            primary_per_seed=[round(x, 4) for x in f1p],
            primary_chance_mean=float(np.mean(f1pc)),
            n_edges_mean=float(np.mean(row["n_edges"])))
    results["conditions"]["real_doubles"]["summary"]["primary_note"] = \
        "self-identity ceiling (F1==1.0 by construction); excluded from gates"

    # HONEST keep/discard gate: real_doubles vs additive_singles on the ANCHORED
    # (non-circular, GIES-output-independent) GT only. The primary GT is derived
    # from real_doubles' own GIES output, so a real-vs-additive gate on it is
    # circular and is NOT computed. For the primary metric we instead report the
    # additive baseline's within-regime recovery vs its permutation chance floor.
    r = np.array(results["conditions"]["real_doubles"]["summary"]["anchored_per_seed"])
    a = np.array(results["conditions"]["additive_singles"]["summary"]["anchored_per_seed"])
    d = r - a
    results["gate"] = {"reference": "anchored (non-circular)",
                       "comparison": "real_doubles vs additive_singles",
                       "mean_gain": float(d.mean()), "sd": float(d.std(ddof=1)),
                       "n_win": int((d > 0).sum()),
                       "pass_gate": bool((d > 0).sum() >= 3 and d.mean() > d.std(ddof=1))}
    add_p = results["conditions"]["additive_singles"]["summary"]
    results["primary_within_regime"] = {
        "note": "additive baseline's recovery of GIES-on-real-doubles structure vs chance",
        "additive_primary_f1": add_p["primary_f1_mean"],
        "additive_primary_chance_f1": add_p["primary_chance_mean"],
        "real_doubles_primary_f1": 1.0}
    json.dump(results, open(OUT, "w"), indent=2)
    print("\n=== Held-out doubles: F1 (mean±sd) ===")
    for c, row in results["conditions"].items():
        s = row["summary"]
        note = "  [primary=self-identity ceiling]" if c == "real_doubles" else ""
        print(f"{c:18s} anchored={s['anchored_f1_mean']:.4f}±{s['anchored_f1_sd']:.4f} "
              f"primary={s['primary_f1_mean']:.4f}{note}")
    g = results["gate"]
    print(f"GATE (anchored, non-circular): real>additive {g['n_win']}/5, "
          f"gain {g['mean_gain']:+.4f}±{g['sd']:.4f}, PASS={g['pass_gate']}")
    print(f"PRIMARY within-regime: additive recovers {add_p['primary_f1_mean']:.4f} of real-doubles "
          f"GIES structure (chance {add_p['primary_chance_mean']:.4f})")
    print("saved", OUT)


if __name__ == "__main__":
    main()
