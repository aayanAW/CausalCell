#!/usr/bin/env python
"""Falsification battery for the positive-control Branch-A result.

Pre-registered in prereg_positive_control.md Amendment 2 (2026-07-08). A hostile
self-check argued the cross-split "positive control" certifies only split-half
RELIABILITY of the marginal perturbation-effect matrix (a DE test), not causal-
structure recovery. This runs the three pre-committed falsification tests:

  F1  DE-baseline   : rank (A,B) by |Welch-t of B under A-KO on H2| vs GT_total_H1.
                      Predicted TIE with the matched |ACE| estimand (cond5 fails).
  F2  direct-edge GT: transitive reduction of GT_total_H1 + TRRUST-curated-in-panel.
                      Predicted COLLAPSE of real toward the null on direct edges.
  F3  VCM arm       : CPA/GEARS/Geneformer predicted total-effect ACE vs GT_total
                      and GT_direct, cross-split. Predicted VCM ~= real on total
                      (inversion) and VCM ~= real ~= null on direct (floor).

Reuses the validated primitives in scripts/positive_control.py (does not modify it).
Run in .venv_gears:  .venv_gears/bin/python scripts/pc_falsify.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import positive_control as pc  # validated primitives

ROOT = pc.ROOT
CTRL = pc.CTRL
OUT_DIR = ROOT / "results" / "positive_control"
VCM_DIR = ROOT / "data" / "model_outputs_real_n200"
VCMS = {
    "cpa": "cpa_predictions.h5ad",
    "gears": "gears_predictions.h5ad",
    "geneformer": "geneformer_predictions.h5ad",
}


# ----------------------------------------------------------------------------- new arms
def de_tstat_matrix(Xp, labp, panel, rows=None) -> np.ndarray:
    """|Welch-t| of B under A-KO vs control (the DE statistic that built the GT), on `rows`."""
    if rows is not None:
        Xp, labp = Xp[rows], labp[rows]
    Xl = np.log1p(Xp)
    ctrl = Xl[labp == CTRL]
    n = len(panel)
    if ctrl.shape[0] < 2:
        return np.zeros((n, n))
    cm, cv, cn = ctrl.mean(0), ctrl.var(0, ddof=1), ctrl.shape[0]
    idx = {g: i for i, g in enumerate(panel)}
    T = np.zeros((n, n))
    for A in panel:
        m = labp == A
        if m.sum() < 20:
            continue
        gi = idx[A]
        Xa = Xl[m]
        am, av, an = Xa.mean(0), Xa.var(0, ddof=1), Xa.shape[0]
        se = np.sqrt(av / an + cv / cn) + 1e-12
        t = np.abs((am - cm) / se)
        t[gi] = 0.0
        T[gi] = t
    return np.nan_to_num(T)


def transitive_reduction(edges: set, panel: list) -> set:
    """Drop (A,C) if a 2-hop path A->B->C exists in `edges` (imperfect on cycles)."""
    out: dict = {}
    inn: dict = {}
    for a, c in edges:
        out.setdefault(a, set()).add(c)
        inn.setdefault(c, set()).add(a)
    reduced = set()
    for a, c in edges:
        mids = out.get(a, set()) & inn.get(c, set())
        mids.discard(a)
        mids.discard(c)
        if not mids:  # no 2-hop explanation -> keep as direct
            reduced.add((a, c))
    return reduced


def trrust_direct(panel: list, tf_targets: dict) -> set:
    pset = set(panel)
    return {
        (tf, tg)
        for tf in panel
        if tf in tf_targets
        for tg in (tf_targets[tf] & pset)
        if tg != tf
    }


def load_vcm_ace(path: Path, panel: list, cm: np.ndarray, pool_genes: list):
    """Predicted total-effect ACE from a VCM: row A = pred_mean[A][panel] - real ctrl mean.
    Returns (ace, coverage) where coverage = # panel perts with a prediction."""
    a = ad.read_h5ad(path)
    perts = a.obs["perturbation"].astype(str).values
    vgenes = list(a.var.index.astype(str))
    vgi = {g: i for i, g in enumerate(vgenes)}
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    prow = {p: i for i, p in enumerate(perts)}
    cols = [vgi.get(g, -1) for g in panel]
    valid = np.array([c >= 0 for c in cols])
    cols_arr = np.array([c if c >= 0 else 0 for c in cols])
    n = len(panel)
    ace = np.zeros((n, n))
    cov = 0
    for i, g in enumerate(panel):
        if g in prow:  # this perturbation has a prediction
            pred = X[prow[g], cols_arr].astype(float).copy()
            pred[~valid] = cm[~valid]  # unmeasured target -> no predicted effect
            ace[i] = pred - cm
            cov += 1
    return np.nan_to_num(ace), cov


# ----------------------------------------------------------------------------- gate
def falsify_gate(
    stratum,
    N,
    genes,
    X,
    lab,
    on_target,
    cell_count,
    tf_set,
    tf_targets,
    vcm_aces,
    seeds,
    n_perm,
    vcm_perts=None,
):
    if stratum == "SVCM":
        # panel = perturbations that ALL VCMs predicted (full VCM coverage), top-N by cell count
        cand = [g for g in genes if vcm_perts and g in vcm_perts]
        cand.sort(key=lambda g: -cell_count[g])
        panel = sorted(cand[:N])
    else:
        panel = pc.select_panel(
            stratum, N, genes, on_target, cell_count, tf_set, tf_targets
        )
    if len(panel) < 20:
        return {
            "stratum": stratum,
            "N": N,
            "panel_size": len(panel),
            "skipped": "panel<20",
        }
    cols = [genes.index(g) for g in panel]
    rowmask = np.isin(lab, panel + [CTRL])
    Xp, labp = X[np.ix_(rowmask, cols)], lab[rowmask]

    # VCM ACEs restricted to this panel (fixed predicted means)
    vcm_panel = {}
    for name, (full_ace, full_panel) in vcm_aces.items():
        # full_ace is keyed to the pool genes; subselect panel rows/cols
        gi = {g: i for i, g in enumerate(full_panel)}
        sub = np.zeros((len(panel), len(panel)))
        ok = all(g in gi for g in panel)
        if ok:
            pidx = [gi[g] for g in panel]
            sub = full_ace[np.ix_(pidx, pidx)]
        vcm_panel[name] = sub

    conds = ["real_ace", "de_tstat", "overlay", "corr_all"] + list(vcm_panel.keys())
    gt_types = ["total", "direct_transred", "trrust"]
    # accumulate per-seed F1 per (condition, gt_type) and null per gt_type
    acc = {c: {gt: [] for gt in gt_types} for c in conds}
    null = {gt: [] for gt in gt_types}
    gt_sizes = {gt: [] for gt in gt_types}

    for s in seeds:
        rng = np.random.default_rng(s)
        h1, h2 = [], []
        for g in panel + [CTRL]:
            gi = rng.permutation(np.nonzero(labp == g)[0])
            cut = gi.size // 2
            h1.extend(gi[:cut].tolist())
            h2.extend(gi[cut:].tolist())
        h1, h2 = np.array(sorted(h1)), np.array(sorted(h2))

        gt_total = pc.anchored_gt(Xp, labp, panel, rows=h1)
        if len(gt_total) < 1:
            continue
        gts = {
            "total": gt_total,
            "direct_transred": transitive_reduction(gt_total, panel),
            "trrust": trrust_direct(panel, tf_targets),
        }
        budget = len(gt_total)

        ace_real, cm = pc.build_ace(Xp, labp, panel, rows=h2, rng=None)
        mats = {
            "real_ace": ace_real,
            "de_tstat": de_tstat_matrix(Xp, labp, panel, rows=h2),
            "overlay": pc.overlay_ace(Xp, labp, panel, cm, rows=h2, rng=rng),
            "corr_all": pc.corr_matrix(Xp[h2]),
        }
        for name, m in vcm_panel.items():
            mats[name] = m

        for gt in gt_types:
            gtset = gts[gt]
            gt_sizes[gt].append(len(gtset))
            if len(gtset) < 1:
                for c in conds:
                    acc[c][gt].append(0.0)
                continue
            b = budget if gt == "total" else max(len(gtset), 1)
            for c in conds:
                acc[c][gt].append(
                    pc.f1_pr(pc.edges_from_matrix(mats[c], panel, b), gtset)[0]
                )
            for k in range(n_perm):
                prng = np.random.default_rng(1_000_000 + s * 1000 + k)
                null[gt].append(
                    pc.f1_pr(
                        pc.edges_from_matrix(pc.col_permute(ace_real, prng), panel, b),
                        gtset,
                    )[0]
                )

    if not acc["real_ace"]["total"]:
        return {
            "stratum": stratum,
            "N": N,
            "panel_size": len(panel),
            "skipped": "no_gt",
        }

    out = {
        "stratum": stratum,
        "N": N,
        "panel_size": len(panel),
        "gt_sizes": {gt: round(float(np.mean(v)), 1) for gt, v in gt_sizes.items()},
        "conditions": {},
        "null": {},
    }
    for gt in gt_types:
        nn = np.asarray(null[gt], float)
        out["null"][gt] = {
            "mean": round(float(nn.mean()), 5) if nn.size else None,
            "p975": round(float(np.percentile(nn, 97.5)), 5) if nn.size else None,
        }
    for c in conds:
        out["conditions"][c] = {gt: pc.summ(acc[c][gt]) for gt in gt_types}

    # decisive readouts
    def m(c, gt):
        return out["conditions"][c][gt]["mean"]

    def lo(c, gt):
        return out["conditions"][c][gt]["ci_lo"]

    out["readouts"] = {
        "F1_cond5_real_gt_de_total": bool(
            m("real_ace", "total") - m("de_tstat", "total") > pc.MDE_PRE
        ),
        "F1_real_minus_de_total": round(
            m("real_ace", "total") - m("de_tstat", "total"), 5
        ),
        "F2_real_above_null_direct_transred": bool(
            out["null"]["direct_transred"]["p975"] is not None
            and lo("real_ace", "direct_transred")
            > out["null"]["direct_transred"]["p975"]
        ),
        "F2_real_above_null_trrust": bool(
            out["null"]["trrust"]["p975"] is not None
            and lo("real_ace", "trrust") > out["null"]["trrust"]["p975"]
        ),
        "F2_real_above_null_total": bool(
            lo("real_ace", "total") > out["null"]["total"]["p975"]
        ),
    }
    # F3 inversion: does any VCM tie/beat real on total?
    for name in vcm_panel:
        out["readouts"][f"F3_{name}_vs_real_total"] = round(
            m(name, "total") - m("real_ace", "total"), 5
        )
        out["readouts"][f"F3_{name}_above_null_direct"] = bool(
            out["null"]["direct_transred"]["p975"] is not None
            and lo(name, "direct_transred") > out["null"]["direct_transred"]["p975"]
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[200])
    ap.add_argument("--strata", nargs="+", default=["S0", "S1", "S4", "SVCM"])
    ap.add_argument("--seeds", type=int, nargs="+", default=pc.SEEDS)
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--pool-size", type=int, default=700)
    ap.add_argument("--cap-cells", type=int, default=2000)
    ap.add_argument("--ctrl-cap", type=int, default=20000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR / "pc_falsify.json"))
    args = ap.parse_args()
    if args.smoke:
        args.strata, args.seeds, args.n_perm, args.pool_size, args.cap_cells = (
            ["S0"],
            [2027, 4099, 6113],
            15,
            350,
            800,
        )
        args.out = str(OUT_DIR / "pc_falsify_smoke.json")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # VCM-predicted pert intersection (force these into the pool so SVCM has full coverage)
    vcm_perts = None
    for fn in VCMS.values():
        p = VCM_DIR / fn
        if p.exists():
            ps = set(ad.read_h5ad(p, backed="r").obs["perturbation"].astype(str).values)
            vcm_perts = ps if vcm_perts is None else (vcm_perts & ps)
    vcm_perts = vcm_perts or set()
    print(f"[vcm] predicted-pert intersection: {len(vcm_perts)}", flush=True)

    print(f"[pool] loading top-{args.pool_size} perts + VCM set ...", flush=True)
    t0 = time.time()
    genes, X, lab, on_target, cell_count = pc.load_pool(
        args.pool_size, args.cap_cells, args.ctrl_cap, force_genes=vcm_perts
    )
    tf_set, tf_targets = pc.load_trrust()
    ctrl_cm = X[lab == CTRL].mean(0)
    print(
        f"[pool] {len(genes)} genes, {X.shape[0]} cells  ({time.time() - t0:.0f}s)",
        flush=True,
    )

    # preload VCM ACEs keyed to the FULL pool gene set (subselected per panel later)
    vcm_aces = {}
    for name, fn in VCMS.items():
        p = VCM_DIR / fn
        if p.exists():
            ace, cov = load_vcm_ace(p, genes, ctrl_cm, genes)
            vcm_aces[name] = (ace, genes)
            print(f"[vcm] {name}: coverage {cov}/{len(genes)} pool perts", flush=True)
        else:
            print(f"[vcm] {name}: MISSING {p}", flush=True)

    gates = []
    for N in args.sizes:
        for st in args.strata:
            gt0 = time.time()
            g = falsify_gate(
                st,
                N,
                genes,
                X,
                lab,
                on_target,
                cell_count,
                tf_set,
                tf_targets,
                vcm_aces,
                args.seeds,
                args.n_perm,
                vcm_perts=vcm_perts,
            )
            gates.append(g)
            if "skipped" in g:
                print(f"  {st} N={N}: SKIP {g['skipped']}", flush=True)
            else:
                r = g["readouts"]
                c = g["conditions"]
                print(
                    f"  {st} N={N} panel={g['panel_size']} GT total~{g['gt_sizes']['total']:.0f} "
                    f"direct~{g['gt_sizes']['direct_transred']:.0f} trrust~{g['gt_sizes']['trrust']:.0f}",
                    flush=True,
                )
                print(
                    f"      TOTAL: real={c['real_ace']['total']['mean']:.4f} de={c['de_tstat']['total']['mean']:.4f} "
                    f"overlay={c['overlay']['total']['mean']:.4f} corr={c['corr_all']['total']['mean']:.4f} "
                    f"null.p975={g['null']['total']['p975']:.4f}",
                    flush=True,
                )
                print(
                    f"      DIRECT(transred): real={c['real_ace']['direct_transred']['mean']:.4f} "
                    f"null.p975={g['null']['direct_transred']['p975']:.4f} "
                    f"real>null={r['F2_real_above_null_direct_transred']}",
                    flush=True,
                )
                for name in [k for k in c if k in VCMS]:
                    print(
                        f"      VCM {name}: total={c[name]['total']['mean']:.4f} (Δreal={r[f'F3_{name}_vs_real_total']:+.4f}) "
                        f"direct={c[name]['direct_transred']['mean']:.4f} direct>null={r[f'F3_{name}_above_null_direct']}",
                        flush=True,
                    )
                print(
                    f"      READOUT: F1 cond5(real>de)={r['F1_cond5_real_gt_de_total']} (Δ={r['F1_real_minus_de_total']:+.4f}) | "
                    f"F2 real>null[direct]={r['F2_real_above_null_direct_transred']} [trrust]={r['F2_real_above_null_trrust']} "
                    f"({time.time() - gt0:.0f}s)",
                    flush=True,
                )

    json.dump({"gates": gates}, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
