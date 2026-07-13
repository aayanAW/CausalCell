#!/usr/bin/env python
"""Working positive control — SNR-restricted recoverable regime.

Pre-registered in prereg_positive_control.md (+ Amendment 1, 2026-07-08).

Question: is there a restricted, biologically motivated regime where REAL K562
Perturb-seq, scored through an ESTIMAND-MATCHED recovery, provably beats a
permutation null against the non-circular perturbation-anchored (total-effect)
ground truth -- and does it survive a CELL-SPLIT (GT on held-out cells, ACE on
disjoint cells) that breaks the same-data tautology?

Design (all from the prereg + amendment, none post-hoc):
  * Primary estimand E-match = threshold |ACE| itself (total-effect ranking),
    the SAME graph object as the reachability GT (fixes the C2 construct mismatch).
    E-prec (precision-support) and E-corr (co-expression) are DIAGNOSTIC/BASELINE.
  * CROSS-SPLIT is the decisive Branch-A test: per seed, split each pert's cells
    + controls into disjoint halves H1/H2; GT on H1, ACE on H2, score H2 vs H1.
    This makes real>null load-bearing (breaks the same-data alignment). The
    in-sample version (GT+ACE on all cells) is reported as an UPPER-BOUND only.
  * Strata S0..S4 enumerated before looking; only {matched estimand} x {strata} x {N}
    on the CROSS-SPLIT block can declare Branch A; BH-FDR across the family;
    real must beat overlay AND both correlational baselines in-window.
  * Full-power cells (cap 2000/pert, all controls); multi-seed via fresh split.

Run in .venv_gears:  .venv_gears/bin/python scripts/positive_control.py [--smoke]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
ATLAS = ROOT / "data" / "k562_real.h5ad"
TRRUST = ROOT / "data" / "ground_truth" / "trrust_rawdata.human.tsv"
OUT_DIR = ROOT / "results" / "positive_control"
PERT_COL = "gene"
CTRL = "non-targeting"
MIN_CELLS = 50

SEEDS = [2027, 4099, 6113, 8117, 10133, 12157, 14173, 16183, 18191, 20201]
Q_GT, TAU_FC = 0.05, 0.5
MDE_PRE = 0.010
STRONG_FC = 1.0
HIGH_COV = 200
MIN_GT_EDGES = 50  # a gate with fewer GT edges is low-power, cannot declare A
STRATA = ["S0", "S1", "S2", "S3", "S4"]


# ----------------------------------------------------------------------------- utils
def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def f1_pr(pred: set, gt: set) -> tuple[float, float, float]:
    pred, gt = set(pred), set(gt)
    tp = len(pred & gt)
    if tp == 0:
        return 0.0, 0.0, 0.0
    p = tp / max(1, len(pred))
    r = tp / max(1, len(gt))
    return 2 * p * r / (p + r), p, r


def config_hash(d: dict) -> str:
    return hashlib.sha1(
        json.dumps(d, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def summ(v) -> dict:
    v = np.asarray(v, float)
    n = v.size
    m = float(v.mean()) if n else 0.0
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    half = stats.t.ppf(0.975, max(1, n - 1)) * sd / np.sqrt(n) if n > 1 else 0.0
    return {
        "mean": round(m, 5),
        "sd": round(sd, 5),
        "ci_lo": round(m - half, 5),
        "ci_hi": round(m + half, 5),
        "n": int(n),
        "per_seed": [round(float(x), 5) for x in v],
    }


# ----------------------------------------------------------------------------- data
def load_pool(pool_size: int, cap_cells: int, ctrl_cap: int, force_genes=None):
    """One atlas read. Top-`pool_size` perts (>=MIN_CELLS) as a dense (cells x genes)
    matrix with <=cap_cells per pert and <=ctrl_cap controls, plus per-pert on-target
    log2FC and cell counts for stratum selection. `force_genes` (optional) are
    additionally included if measured & >=MIN_CELLS (e.g. the VCM-predicted pert set)."""
    a = ad.read_h5ad(ATLAS, backed="r")
    labels = a.obs[PERT_COL].astype(str).values
    sym = a.var["gene_name"].astype(str).values
    sym_to_col = {}
    for j, s in enumerate(sym):
        sym_to_col.setdefault(s, j)
    measured = set(sym_to_col)
    counts: dict[str, int] = {}
    for g in labels:
        counts[g] = counts.get(g, 0) + 1
    cand = [
        (g, c)
        for g, c in counts.items()
        if g != CTRL and c >= MIN_CELLS and g in measured
    ]
    cand.sort(key=lambda x: -x[1])
    chosen = set(g for g, _ in cand[:pool_size])
    if force_genes:
        chosen |= {
            g
            for g in force_genes
            if g in measured and counts.get(g, 0) >= MIN_CELLS and g != CTRL
        }
    genes = sorted(chosen)
    col_idx = [sym_to_col[g] for g in genes]

    rng = np.random.default_rng(0)
    keep_rows: list[int] = []
    for g in genes:
        idx = np.nonzero(labels == g)[0]
        if idx.size > cap_cells:
            idx = rng.choice(idx, cap_cells, replace=False)
        keep_rows.extend(idx.tolist())
    cidx = np.nonzero(labels == CTRL)[0]
    if cidx.size > ctrl_cap:
        cidx = rng.choice(cidx, ctrl_cap, replace=False)
    keep_rows.extend(cidx.tolist())
    keep_rows = np.array(sorted(keep_rows))

    sub = a[keep_rows].to_memory()
    Xall = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
    X = Xall[:, col_idx].astype(np.float64)
    lab = sub.obs[PERT_COL].astype(str).values

    Xl = np.log1p(X)
    cm = Xl[lab == CTRL].mean(0)
    c_lin = np.expm1(cm) + 1.0
    gi = {g: i for i, g in enumerate(genes)}
    on_target, cell_count = {}, {}
    for g in genes:
        m = lab == g
        cell_count[g] = int(m.sum())
        am = Xl[m].mean(0)
        on_target[g] = float(np.log2((np.expm1(am[gi[g]]) + 1.0) / c_lin[gi[g]]))
    return genes, X, lab, on_target, cell_count


def load_trrust() -> tuple[set[str], dict[str, set[str]]]:
    """Return (set of TFs, TF -> set of targets) from TRRUST (tab-sep, no header)."""
    tfs: set[str] = set()
    targets: dict[str, set[str]] = {}
    if not TRRUST.exists():
        return tfs, targets
    for line in TRRUST.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0] and parts[1]:
            tf, tgt = parts[0].strip(), parts[1].strip()
            tfs.add(tf)
            targets.setdefault(tf, set()).add(tgt)
    return tfs, targets


def select_panel(
    stratum, N, genes, on_target, cell_count, tf_set, tf_targets
) -> list[str]:
    gset = set(genes)
    if stratum == "S0":
        cand = sorted(genes, key=lambda g: -cell_count[g])
    elif stratum == "S1":
        cand = sorted(
            (g for g in genes if abs(on_target[g]) >= STRONG_FC),
            key=lambda g: -cell_count[g],
        )
    elif stratum == "S2":
        cand = sorted(
            (g for g in genes if cell_count[g] >= HIGH_COV),
            key=lambda g: -abs(on_target[g]),
        )
    elif stratum == "S3":
        # perturbed TRRUST TFs with >=1 target also in the pool, unioned with those in-pool targets
        tf_in = [g for g in genes if g in tf_set and (tf_targets.get(g, set()) & gset)]
        panel_set: set[str] = set(tf_in)
        for tf in tf_in:
            panel_set |= tf_targets[tf] & gset
        cand = sorted(panel_set, key=lambda g: -cell_count[g])
    elif stratum == "S4":
        cand = sorted(
            (
                g
                for g in genes
                if abs(on_target[g]) >= STRONG_FC and cell_count[g] >= HIGH_COV
            ),
            key=lambda g: -cell_count[g],
        )
    else:
        raise ValueError(stratum)
    return sorted(cand[:N])


# ----------------------------------------------------------------------------- GT + estimands
def anchored_gt(Xp, labp, panel, rows=None) -> set:
    """Total-effect GT on `panel`, optionally restricted to row indices `rows` (cell-split H1)."""
    if rows is not None:
        Xp, labp = Xp[rows], labp[rows]
    Xl = np.log1p(Xp)
    ctrl = Xl[labp == CTRL]
    if ctrl.shape[0] < 2:
        return set()
    n = len(panel)
    cm, cv, cn = ctrl.mean(0), ctrl.var(0, ddof=1), ctrl.shape[0]
    c_lin = np.expm1(cm) + 1.0
    idx = {g: i for i, g in enumerate(panel)}
    edges = set()
    for A in panel:
        m = labp == A
        if m.sum() < 20:
            continue
        gi = idx[A]
        Xa = Xl[m]
        am, av, an = Xa.mean(0), Xa.var(0, ddof=1), Xa.shape[0]
        se = np.sqrt(av / an + cv / cn) + 1e-12
        t = (am - cm) / se
        df = (av / an + cv / cn) ** 2 / (
            (av / an) ** 2 / (an - 1) + (cv / cn) ** 2 / (cn - 1) + 1e-30
        )
        pv = np.where(
            np.isfinite(2 * stats.t.sf(np.abs(t), df)),
            2 * stats.t.sf(np.abs(t), df),
            1.0,
        )
        lfc = np.log2((np.expm1(am) + 1.0) / c_lin)
        pv[gi] = 1.0
        sel = (bh_adjust(pv) <= Q_GT) & (np.abs(lfc) >= TAU_FC)
        sel[gi] = False
        for bi in np.nonzero(sel)[0]:
            edges.add((A, panel[bi]))
    return edges


def build_ace(Xp, labp, panel, rows=None, rng=None):
    """ACE row i = mean(expr | do i) - mean(expr | ctrl). `rows` restricts to a cell
    subset (cross-split H2); `rng` bootstrap-resamples cells within (per-seed SD)."""
    idx_all = np.arange(len(labp)) if rows is None else np.asarray(rows)
    lab_s = labp[idx_all]
    n = len(panel)
    ctrl_local = idx_all[lab_s == CTRL]
    if ctrl_local.size == 0:
        return np.zeros((n, n)), np.zeros(len(panel[0:0]) or n)
    if rng is not None:
        ctrl_local = rng.choice(ctrl_local, ctrl_local.size, replace=True)
    cm = Xp[ctrl_local].mean(0)
    ace = np.zeros((n, n))
    for i, g in enumerate(panel):
        gl = idx_all[lab_s == g]
        if gl.size == 0:
            continue
        if rng is not None:
            gl = rng.choice(gl, gl.size, replace=True)
        ace[i] = Xp[gl].mean(0) - cm
    return np.nan_to_num(ace), cm


def overlay_ace(Xp, labp, panel, cm, rows=None, rng=None) -> np.ndarray:
    """Real means through the diagonal-rescale + Poisson overlay, re-meaned (the confound arm)."""
    idx_all = np.arange(len(labp)) if rows is None else np.asarray(rows)
    lab_s = labp[idx_all]
    n = len(panel)
    ctrl_local = idx_all[lab_s == CTRL]
    ctrl_X = Xp[ctrl_local]
    if ctrl_X.shape[0] == 0:
        return np.zeros((n, n))
    r = rng if rng is not None else np.random.default_rng(0)
    nz = cm > 1e-6
    ace = np.zeros((n, n))
    for i, g in enumerate(panel):
        gl = idx_all[lab_s == g]
        if gl.size == 0:
            continue
        rm = Xp[gl].mean(0)
        ratios = np.ones(n)
        ratios[nz] = np.clip(np.maximum(rm[nz], 0) / np.maximum(cm, 0.01)[nz], 0.01, 10)
        base = (
            ctrl_X[r.choice(ctrl_X.shape[0], min(100, ctrl_X.shape[0]), True)]
            * ratios[None, :]
        )
        ace[i] = r.poisson(np.maximum(base, 0)).mean(0) - cm
    return np.nan_to_num(ace)


def elastic_ace(ctrl_X, n, seed) -> np.ndarray:
    from sklearn.linear_model import ElasticNet

    Xc = ctrl_X - ctrl_X.mean(0)
    if Xc.shape[0] > 2000:
        Xc = Xc[np.random.default_rng(seed).choice(Xc.shape[0], 2000, False)]
    ace = np.zeros((n, n))
    for i in range(n):
        y = Xc[:, i]
        Xin = np.delete(Xc, i, axis=1)
        m = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=seed).fit(
            Xin, y
        )
        row = np.zeros(n)
        row[np.arange(n) != i] = m.coef_
        ace[i] = row
    return ace


def corr_matrix(X_cells) -> np.ndarray:
    C = np.nan_to_num(np.corrcoef(np.log1p(X_cells).T))
    np.fill_diagonal(C, 0.0)
    return np.abs(C)


def edges_from_matrix(M, panel, budget) -> set:
    """Top-`budget` off-diagonal |M_ij| as directed edges i->j (returns exactly budget)."""
    n = len(panel)
    absM = np.abs(M).copy()
    np.fill_diagonal(absM, 0.0)
    off = ~np.eye(n, dtype=bool)
    flat = absM[off]
    if budget >= flat.size or budget <= 0:
        thr = -1.0
    else:
        thr = np.partition(flat, flat.size - budget)[
            flat.size - budget
        ]  # budget-th largest value
    return {
        (panel[i], panel[j])
        for i in range(n)
        for j in range(n)
        if i != j and absM[i, j] >= thr and absM[i, j] > 1e-12
    }


def precision_edges(ace, panel, budget, ridge=1e-2) -> set:
    """Direct-edge precision support (the MISMATCHED estimand, diagnostic only)."""
    n = len(panel)
    R = ace / (np.abs(ace).max() + 1e-9)
    Rreg = R + ridge * np.eye(n)
    try:
        Ginv = np.linalg.inv(Rreg)
    except np.linalg.LinAlgError:
        Ginv = np.linalg.pinv(Rreg)
    G = np.eye(n) - Ginv
    np.fill_diagonal(G, 0.0)
    return edges_from_matrix(G, panel, budget)


def col_permute(M, rng) -> np.ndarray:
    out = M.copy()
    for j in range(out.shape[1]):
        out[:, j] = out[rng.permutation(out.shape[0]), j]
    return out


# ----------------------------------------------------------------------------- one block (in-sample OR cross-split)
def score_block(Xp, labp, panel, seeds, n_perm, split: bool):
    """Return per-seed real/overlay F1 (matched estimand), pooled null, and fixed
    baselines (elastic, corr-ctrl, corr-all, prec). If split=True, GT is built on H1
    and ACE on the disjoint H2 (per seed); else GT on all cells, ACE bootstrap on all."""
    n = len(panel)
    real_f1, real_p, real_r = [], [], []
    over_f1 = []
    prec_f1 = []
    null_f1 = []
    gt_sizes = []
    edge_counts = []
    # fixed baselines from seed-0 configuration (point estimates; guards, not deciders)
    base = {}
    for si, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        if split:
            h1, h2 = [], []
            for g in panel + [CTRL]:
                gi = np.nonzero(labp == g)[0]
                gi = rng.permutation(gi)
                cut = gi.size // 2
                h1.extend(gi[:cut].tolist())
                h2.extend(gi[cut:].tolist())
            h1, h2 = np.array(sorted(h1)), np.array(sorted(h2))
            gt = anchored_gt(Xp, labp, panel, rows=h1)
            ace_real, cm = build_ace(
                Xp, labp, panel, rows=h2, rng=None
            )  # H2 is already a fresh split
            ace_over = overlay_ace(Xp, labp, panel, cm, rows=h2, rng=rng)
            ace_ref_rows = h2
        else:
            gt = anchored_gt(Xp, labp, panel)  # all cells (deterministic target)
            ace_real, cm = build_ace(
                Xp, labp, panel, rows=None, rng=rng
            )  # bootstrap for SD
            ace_over = overlay_ace(Xp, labp, panel, cm, rows=None, rng=rng)
            ace_ref_rows = None
        gt_sizes.append(len(gt))
        if len(gt) < 1:
            continue
        budget = len(gt)
        e_real = edges_from_matrix(ace_real, panel, budget)
        fr, pr, rr = f1_pr(e_real, gt)
        real_f1.append(fr)
        real_p.append(pr)
        real_r.append(rr)
        edge_counts.append(len(e_real))
        over_f1.append(f1_pr(edges_from_matrix(ace_over, panel, budget), gt)[0])
        prec_f1.append(f1_pr(precision_edges(ace_real, panel, budget), gt)[0])
        for k in range(n_perm):
            prng = np.random.default_rng(1_000_000 + s * 1000 + k)
            null_f1.append(
                f1_pr(
                    edges_from_matrix(col_permute(ace_real, prng), panel, budget), gt
                )[0]
            )
        if not base:
            # baselines on the FIRST seed that yields a non-empty GT (guards vs empty
            # seed-0 GT in the cross-split path, which previously KeyError'd downstream)
            idx = np.arange(len(labp)) if ace_ref_rows is None else ace_ref_rows
            lab_s = labp[idx]
            ctrl_X = Xp[idx][lab_s == CTRL]
            all_X = Xp[idx]
            ace_enet = elastic_ace(ctrl_X, n, seeds[0])
            base["elastic_f1"] = round(
                f1_pr(edges_from_matrix(ace_enet, panel, budget), gt)[0], 5
            )
            base["corr_ctrl_f1"] = round(
                f1_pr(edges_from_matrix(corr_matrix(ctrl_X), panel, budget), gt)[0], 5
            )
            base["corr_all_f1"] = round(
                f1_pr(edges_from_matrix(corr_matrix(all_X), panel, budget), gt)[0], 5
            )

    if not real_f1:
        return None
    null = np.asarray(null_f1, float)
    real_arr = np.asarray(real_f1, float)
    over_arr = np.asarray(over_f1, float)
    null_band = {
        "mean": round(float(null.mean()), 5),
        "sd": round(float(null.std()), 5),
        "p975": round(float(np.percentile(null, 97.5)), 5),
        "n": int(null.size),
    }
    # paired real>overlay (per seed), one-sided real>null (across seeds vs pooled null mean)
    d = real_arr - over_arr[: real_arr.size]
    t_ro, p_ro = stats.ttest_1samp(d, 0.0) if d.size > 1 else (0.0, 1.0)
    p_real_gt_over = (p_ro / 2 if t_ro > 0 else 1.0) if d.size > 1 else 1.0
    m = real_arr - null.mean()
    t_rn, p_rn = stats.ttest_1samp(m, 0.0) if real_arr.size > 1 else (0.0, 1.0)
    p_real_gt_null = (p_rn / 2 if t_rn > 0 else 1.0) if real_arr.size > 1 else 1.0
    return {
        "gt_edges_mean": round(float(np.mean(gt_sizes)), 1),
        "real": {
            **summ(real_arr),
            "precision": round(float(np.mean(real_p)), 5),
            "recall": round(float(np.mean(real_r)), 5),
            "edges_mean": round(float(np.mean(edge_counts)), 1),
        },
        "overlay": summ(over_arr),
        "prec_diag": summ(prec_f1),
        "null": null_band,
        **base,
        "p_real_gt_null": round(float(p_real_gt_null), 5),
        "p_real_gt_overlay": round(float(p_real_gt_over), 5),
    }


def run_gate(
    stratum, N, genes, X, lab, on_target, cell_count, tf_set, tf_targets, seeds, n_perm
):
    panel = select_panel(stratum, N, genes, on_target, cell_count, tf_set, tf_targets)
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

    xs = score_block(
        Xp, labp, panel, seeds, n_perm, split=False
    )  # in-sample (upper bound)
    cs = score_block(
        Xp, labp, panel, seeds, n_perm, split=True
    )  # cross-split (decisive)
    g = {
        "stratum": stratum,
        "N": N,
        "panel_size": len(panel),
        "in_sample": xs,
        "cross_split": cs,
    }
    if cs is None or xs is None:
        g["skipped"] = "no_gt_edges"
        return g

    # Branch-A conditions on the CROSS-SPLIT block only
    rm, nb, ov = cs["real"], cs["null"], cs["overlay"]
    corr_max = max(cs["corr_ctrl_f1"], cs["corr_all_f1"])
    g["cond1_ci_above_null"] = bool(rm["ci_lo"] > nb["p975"])
    g["cond2_margin_gt_mde"] = bool((rm["mean"] - nb["mean"]) > MDE_PRE)
    g["cond3_real_gt_overlay"] = bool(
        rm["mean"] > ov["mean"] and cs["p_real_gt_overlay"] < 0.05
    )
    g["cond4_real_ge_corr"] = bool(rm["mean"] >= corr_max)
    g["low_power"] = bool(cs["gt_edges_mean"] < MIN_GT_EDGES)
    g["p_one_sided"] = cs["p_real_gt_null"]
    g["corr_max_f1"] = round(corr_max, 5)
    return g


def write_summary(result, path):
    lines = [
        "# Positive-control scorecard",
        "",
        f"**Branch: {result['branch']}**  ({result['n_pass']} pass / {result['n_inversion']} inversion / {result['n_gates']} gates)  git `{result['git_sha']}`",
        "",
        "Decisive block = CROSS-SPLIT (GT on held-out cells, ACE on disjoint cells).",
        "",
        "| stratum | N | panel | GTedges | real F1 (xsplit) [CI] | null p97.5 | overlay | corr_max | in-sample real | C1 C2 C3 C4 | FDRq | PASS |",
        "|--|--|--|--|--|--|--|--|--|--|--|--|",
    ]
    for g in result["gates"]:
        if "skipped" in g:
            lines.append(
                f"| {g['stratum']} | {g['N']} | {g['panel_size']} | — | SKIP: {g['skipped']} | | | | | | | |"
            )
            continue
        cs, xs = g["cross_split"], g["in_sample"]
        rm = cs["real"]
        c = f"{int(g['cond1_ci_above_null'])} {int(g['cond2_margin_gt_mde'])} {int(g['cond3_real_gt_overlay'])} {int(g['cond4_real_ge_corr'])}"
        lp = " (low-power)" if g.get("low_power") else ""
        lines.append(
            f"| {g['stratum']} | {g['N']} | {g['panel_size']} | {cs['gt_edges_mean']:.0f} | "
            f"{rm['mean']:.4f} [{rm['ci_lo']:.4f},{rm['ci_hi']:.4f}] | {cs['null']['p975']:.4f} | "
            f"{cs['overlay']['mean']:.4f} | {g['corr_max_f1']:.4f} | {xs['real']['mean']:.4f} | {c} | "
            f"{g.get('fdr_q', '—')} | {'YES' if g.get('pass') else ('INV' if g.get('inversion') else 'no')}{lp} |"
        )
    lines += [
        "",
        "cond1 real CI-lo>null p97.5 · cond2 margin>MDE(0.010) · cond3 real>overlay (paired p<0.05) · cond4 real>=max(corr_ctrl,corr_all).",
        "Branch A requires a cross-split gate to pass all four + FDR q<=0.05 and not be low-power.",
    ]
    Path(path).write_text("\n".join(lines))


def write_figure(result, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    gates = [g for g in result["gates"] if "skipped" not in g]
    if not gates:
        return
    labels = [f"{g['stratum']}·N{g['N']}" for g in gates]
    real = [g["cross_split"]["real"]["mean"] for g in gates]
    lo = [
        g["cross_split"]["real"]["mean"] - g["cross_split"]["real"]["ci_lo"]
        for g in gates
    ]
    hi = [
        g["cross_split"]["real"]["ci_hi"] - g["cross_split"]["real"]["mean"]
        for g in gates
    ]
    nullp = [g["cross_split"]["null"]["p975"] for g in gates]
    over = [g["cross_split"]["overlay"]["mean"] for g in gates]
    x = np.arange(len(gates))
    fig, ax = plt.subplots(figsize=(max(6, len(gates) * 0.9), 4.2))
    ax.errorbar(
        x,
        real,
        yerr=[lo, hi],
        fmt="o",
        color="#1b6",
        capsize=3,
        label="real (cross-split)",
    )
    ax.plot(x, nullp, "s--", color="#888", label="null p97.5")
    ax.plot(x, over, "^", color="#c60", alpha=0.7, label="overlay")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("edge-set F1 (matched estimand)")
    ax.set_title(f"Positive control — cross-split · Branch {result['branch']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 500])
    ap.add_argument("--strata", nargs="+", default=STRATA)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--pool-size", type=int, default=700)
    ap.add_argument("--cap-cells", type=int, default=2000)
    ap.add_argument("--ctrl-cap", type=int, default=20000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR / "pc_grid.json"))
    args = ap.parse_args()
    if args.smoke:
        args.sizes, args.strata, args.seeds, args.n_perm = (
            [200],
            ["S0", "S1"],
            [2027, 4099, 6113],
            20,
        )
        args.pool_size, args.cap_cells = 350, 800
        args.out = str(OUT_DIR / "pc_smoke.json")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {
        "sizes": args.sizes,
        "strata": args.strata,
        "seeds": args.seeds,
        "n_perm": args.n_perm,
        "pool_size": args.pool_size,
        "cap_cells": args.cap_cells,
        "ctrl_cap": args.ctrl_cap,
        "q_gt": Q_GT,
        "tau_fc": TAU_FC,
        "mde": MDE_PRE,
        "strong_fc": STRONG_FC,
        "high_cov": HIGH_COV,
    }
    print(f"[pool] loading top-{args.pool_size} perts ...", flush=True)
    t0 = time.time()
    genes, X, lab, on_target, cell_count = load_pool(
        args.pool_size, args.cap_cells, args.ctrl_cap
    )
    tf_set, tf_targets = load_trrust()
    print(
        f"[pool] {len(genes)} genes, {X.shape[0]} cells, {len(tf_set)} TRRUST TFs  ({time.time() - t0:.0f}s)",
        flush=True,
    )

    gates, ledger = [], []
    for N in args.sizes:
        for st in args.strata:
            gt0 = time.time()
            g = run_gate(
                st,
                N,
                genes,
                X,
                lab,
                on_target,
                cell_count,
                tf_set,
                tf_targets,
                args.seeds,
                args.n_perm,
            )
            gates.append(g)
            ledger.append(
                {
                    "git_sha": git_sha(),
                    "config_hash": config_hash({**cfg, "N": N, "stratum": st}),
                    **{
                        k: g.get(k)
                        for k in (
                            "stratum",
                            "N",
                            "panel_size",
                            "low_power",
                            "cond1_ci_above_null",
                            "cond2_margin_gt_mde",
                            "cond3_real_gt_overlay",
                            "cond4_real_ge_corr",
                            "p_one_sided",
                        )
                    },
                }
            )
            if "skipped" in g:
                print(
                    f"  {st} N={N}: SKIP ({g['skipped']}, panel={g['panel_size']})",
                    flush=True,
                )
            else:
                cs, xs = g["cross_split"], g["in_sample"]
                rm = cs["real"]
                print(
                    f"  {st} N={N} panel={g['panel_size']} GT~{cs['gt_edges_mean']:.0f}: "
                    f"XSPLIT real={rm['mean']:.4f}[{rm['ci_lo']:.4f},{rm['ci_hi']:.4f}] "
                    f"null.p975={cs['null']['p975']:.4f} over={cs['overlay']['mean']:.4f} "
                    f"corr={g['corr_max_f1']:.4f} | insample={xs['real']['mean']:.4f} | "
                    f"C{int(g['cond1_ci_above_null'])}{int(g['cond2_margin_gt_mde'])}"
                    f"{int(g['cond3_real_gt_overlay'])}{int(g['cond4_real_ge_corr'])}"
                    f"{' LP' if g['low_power'] else ''} ({time.time() - gt0:.0f}s)",
                    flush=True,
                )

    live = [g for g in gates if "skipped" not in g]
    if live:
        adj = bh_adjust(np.array([g["p_one_sided"] for g in live]))
        for g, q in zip(live, adj):
            g["fdr_q"] = round(float(q), 5)
            g["pass"] = bool(
                g["cond1_ci_above_null"]
                and g["cond2_margin_gt_mde"]
                and g["cond3_real_gt_overlay"]
                and g["cond4_real_ge_corr"]
                and q <= 0.05
                and not g["low_power"]
            )
            g["inversion"] = bool(
                g["cond1_ci_above_null"]
                and g["cond2_margin_gt_mde"]
                and not g["cond3_real_gt_overlay"]
            )
    n_pass = sum(g.get("pass", False) for g in live)
    n_inv = sum(g.get("inversion", False) for g in live)
    branch = "A" if n_pass > 0 else ("C_inversion" if n_inv > 0 else "B")
    result = {
        "config": cfg,
        "git_sha": git_sha(),
        "branch": branch,
        "n_gates": len(live),
        "n_pass": int(n_pass),
        "n_inversion": int(n_inv),
        "gates": gates,
    }
    json.dump(result, open(args.out, "w"), indent=2)
    with open(OUT_DIR / "decision_ledger.jsonl", "w") as fh:
        for r in ledger:
            fh.write(json.dumps(r) + "\n")
    if not args.smoke:
        write_summary(result, OUT_DIR / "pc_summary.md")
        write_figure(result, ROOT / "figs" / "pc_real_vs_null_by_stratum.png")
    print(
        f"\n=== BRANCH {branch} === ({n_pass} pass / {n_inv} inversion / {len(live)} gates)  saved {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
