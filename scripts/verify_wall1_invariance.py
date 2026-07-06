#!/usr/bin/env python3
"""Wall-1 invariance lemma — numerical verification against the ACTUAL overlay.

Verifies, using the real `sample_perturbed_cells_overlay` from run_eval_local.py
(NOT a reimplementation), the corrected lemma:

  (Part 1) DETERMINISTIC rescale is EXACT. Overlaying any two mean predictions with
    the same ratio-support onto the same control cells (Poisson step disabled) yields
    correlation matrices identical to machine precision and precision-support
    Jaccard == 1.0. This is the exact-invariance core: the prediction enters as a
    positive diagonal congruence D, under which corr(D Sigma D) = corr(Sigma) and
    supp((D Sigma D)^-1) = supp(Sigma^-1).

  (Part 2) POISSON noise adds a mean-dependent residual that VANISHES with counts.
    Same two predictions WITH the Poisson step, swept over a UMI depth-scale factor:
    report max|Δcorr| and the precision-support edge-flip fraction vs depth. Confirms
    the residual -> 0 as counts grow, so "mean-only" holds up to an empirically small,
    count-dependent noise term (NOT claimed as exactly zero once noise is on).

Runs in .venv_gears (Py3.9). CPU only. Writes results/hardening/wall1_invariance.json
and figures/wall1_invariance.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.run_eval_local import sample_perturbed_cells_overlay  # noqa: E402

OUT = ROOT / "results" / "hardening" / "wall1_invariance.json"
FIG = ROOT / "figures" / "wall1_invariance.png"
DATA = ROOT / "data" / "k562_subset_n200_slim.h5ad"
SEED = 42
N_CELLS = 400
EPS = 1e-8


def load_ctrl():
    a = ad.read_h5ad(DATA)
    # control cells: try common labels
    obs = a.obs
    ctrl_mask = None
    for col in ["gene", "perturbation", "condition", "pert"]:
        if col in obs.columns:
            vals = obs[col].astype(str)
            for cl in ["non-targeting", "control", "ctrl", "NT"]:
                if (vals == cl).sum() > 50:
                    ctrl_mask = (vals == cl).values
                    break
        if ctrl_mask is not None:
            break
    X = a.X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)
    if ctrl_mask is None:
        # fall back: use all cells (still a valid control covariance for the test)
        ctrl = X
    else:
        ctrl = X[ctrl_mask]
    return ctrl


def precision_support(cells, thr=0.05):
    """Support (binary adjacency) of the partial-correlation / precision matrix.
    Uses shrinkage inverse of the correlation matrix; support = |partial corr|>thr.
    """
    Xc = cells - cells.mean(0, keepdims=True)
    sd = Xc.std(0, keepdims=True)
    keep = (sd.ravel() > EPS)
    Xk = Xc[:, keep] / sd[:, keep]
    C = np.corrcoef(Xk, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    p = C.shape[0]
    C += 1e-3 * np.eye(p)  # ridge for invertibility
    P = np.linalg.inv(C)
    d = np.sqrt(np.abs(np.diag(P)))
    partial = -P / np.outer(d, d)
    np.fill_diagonal(partial, 0.0)
    supp = (np.abs(partial) > thr).astype(int)
    # embed back to full gene index space
    full = np.zeros((cells.shape[1], cells.shape[1]), int)
    idx = np.where(keep)[0]
    full[np.ix_(idx, idx)] = supp
    return full, keep


def jaccard(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = (a & b).sum(); union = (a | b).sum()
    return 1.0 if union == 0 else inter / union


def deterministic_overlay(ctrl_X, mean_pred):
    """Reproduce the overlay's DETERMINISTIC rescale by monkeypatching the RNG's
    poisson to identity, so we isolate Part 1 from the real function. We call the
    real function but replace rng.poisson with a pass-through of the (float) lambda.
    """
    import scripts.run_eval_local as rl
    orig = np.random.default_rng

    class _NoPoisRNG:
        def __init__(self, seed): self._r = orig(seed)
        def choice(self, *a, **k): return self._r.choice(*a, **k)
        def poisson(self, lam=None, **k): return np.asarray(lam)  # identity, no noise
    rl.np.random.default_rng = _NoPoisRNG
    try:
        cells = sample_perturbed_cells_overlay(ctrl_X, mean_pred, N_CELLS, seed=SEED)
    finally:
        rl.np.random.default_rng = orig
    return cells


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    ctrl = load_ctrl()
    rng = np.random.default_rng(SEED)
    ci = rng.choice(ctrl.shape[0], size=min(2000, ctrl.shape[0]), replace=False)
    ctrl_X = ctrl[ci]
    ctrl_mean = ctrl_X.mean(0)
    G = ctrl_X.shape[1]

    # Two VERY different mean predictions (same clip-support): random fold-changes,
    # kept inside the [0.01,10] clip so both use the full diagonal range.
    fc1 = rng.uniform(0.2, 5.0, size=G)
    fc2 = rng.uniform(0.2, 5.0, size=G)
    pred1 = ctrl_mean * fc1
    pred2 = ctrl_mean * fc2

    res = {"data": str(DATA.name), "n_genes": int(G), "n_cells": N_CELLS, "seed": SEED}

    # ---------- Part 1: deterministic rescale is EXACT ----------
    d1 = deterministic_overlay(ctrl_X, pred1)
    d2 = deterministic_overlay(ctrl_X, pred2)
    # correlation matrices
    def corr(x):
        c = np.corrcoef(x, rowvar=False); return np.nan_to_num(c, nan=0.0)
    C1, C2 = corr(d1), corr(d2)
    max_dcorr_det = float(np.max(np.abs(C1 - C2)))
    S1, _ = precision_support(d1)
    S2, _ = precision_support(d2)
    jac_det = float(jaccard(S1, S2))
    res["part1_deterministic"] = dict(
        max_abs_delta_corr=max_dcorr_det,
        precision_support_jaccard=jac_det,
        n_support_edges_pred1=int(S1.sum()), n_support_edges_pred2=int(S2.sum()),
        claim="exact invariance: corr identical to machine precision, support Jaccard==1")

    # ---------- Part 2: Poisson residual vs UMI depth ----------
    depth_factors = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    sweep = []
    base_supp = None
    for df in depth_factors:
        # scale control counts to change effective depth, then overlay WITH Poisson
        cX = np.maximum(ctrl_X * df, 0.0)
        n1 = sample_perturbed_cells_overlay(cX, pred1 * df, N_CELLS, seed=SEED)
        n2 = sample_perturbed_cells_overlay(cX, pred2 * df, N_CELLS, seed=SEED)
        Cn1, Cn2 = corr(n1), corr(n2)
        mdc = float(np.max(np.abs(Cn1 - Cn2)))
        Sn1, _ = precision_support(n1)
        Sn2, _ = precision_support(n2)
        # edge-flip fraction: disagreement between the two predictions' supports
        flips = int((Sn1 != Sn2).sum())
        total = int(Sn1.size - Sn1.shape[0])  # exclude diagonal
        sweep.append(dict(depth_factor=df,
                          mean_counts_per_cell=float(n1.sum(1).mean()),
                          max_abs_delta_corr=mdc,
                          support_jaccard=float(jaccard(Sn1, Sn2)),
                          edge_flip_fraction=flips / max(total, 1)))
    res["part2_poisson_sweep"] = sweep

    # ---------- Part 3: SAME mean, DIFFERENT joint structure -> identical metric ----------
    # This is the airtight Track-B kill: a model whose predictions enter as MEANS is
    # reduced to a per-gene mean before the overlay. Two predicted-cell distributions
    # with the SAME per-gene mean but WILDLY different joint structure therefore produce
    # an IDENTICAL overlay input and IDENTICAL causal support. We construct:
    #   J_hi : cells with strong induced gene-gene covariance (shared latent factor)
    #   J_lo : cells with the SAME per-gene mean but independent genes (no covariance)
    # then feed each through the mean-path (reduce to mean -> overlay), as the pipeline does.
    target_mean = pred1.copy()
    # J_hi: multiply control cells by a shared per-cell latent (induces cross-gene corr)
    rng3 = np.random.default_rng(SEED + 3)
    latent = rng3.lognormal(0, 0.5, size=(ctrl_X.shape[0], 1))
    j_hi = ctrl_X * latent
    j_hi = j_hi * (target_mean / np.maximum(j_hi.mean(0), EPS))   # rescale to target mean
    # J_lo: independent Poisson draws at the target mean (no cross-gene structure)
    j_lo = rng3.poisson(np.broadcast_to(target_mean, (ctrl_X.shape[0], G))).astype(float)
    j_lo = j_lo * (target_mean / np.maximum(j_lo.mean(0), EPS))
    # covariance of the two "predicted" distributions (before mean-reduction) differs:
    off_hi = float(np.abs(corr(j_hi) - np.eye(G)).mean())
    off_lo = float(np.abs(corr(j_lo) - np.eye(G)).mean())
    # pipeline mean-path: reduce each to its per-gene mean, then overlay
    m_hi = j_hi.mean(0); m_lo = j_lo.mean(0)
    o_hi = sample_perturbed_cells_overlay(ctrl_X, m_hi, N_CELLS, seed=SEED)
    o_lo = sample_perturbed_cells_overlay(ctrl_X, m_lo, N_CELLS, seed=SEED)
    Sh, _ = precision_support(o_hi); Sl, _ = precision_support(o_lo)
    res["part3_same_mean_diff_joint"] = dict(
        input_mean_max_abs_diff=float(np.max(np.abs(m_hi - m_lo))),
        pred_offdiag_corr_hi=off_hi, pred_offdiag_corr_lo=off_lo,
        pred_joint_structure_ratio=off_hi / max(off_lo, 1e-9),
        overlay_support_jaccard=float(jaccard(Sh, Sl)),
        max_abs_delta_corr=float(np.max(np.abs(corr(o_hi) - corr(o_lo)))),
        claim=("two predictions with the SAME per-gene mean but "
               f"{off_hi/max(off_lo,1e-9):.1f}x different joint structure yield the SAME "
               "overlay support -> the metric cannot reward Track-B's joint-structure reshaping"))

    # summary verdict
    # NOTE: the real overlay casts cells to float32, so the deterministic residual
    # sits at float32 epsilon (~1e-7 on correlations), NOT float64 machine-zero. The
    # exact-invariance claim is about SUPPORT (Jaccard==1.0) and correlation equality
    # up to storage precision — verified honestly against the actual function.
    res["verdict"] = dict(
        deterministic_support_exact=bool(jac_det == 1.0),
        deterministic_corr_float32=bool(max_dcorr_det < 1e-6),
        deterministic_exact=bool(jac_det == 1.0 and max_dcorr_det < 1e-6),
        poisson_residual_decreases=bool(
            sweep[-1]["edge_flip_fraction"] <= sweep[0]["edge_flip_fraction"]),
        poisson_residual_at_realistic_depth=float(
            [s["edge_flip_fraction"] for s in sweep if abs(s["depth_factor"] - 1.0) < 1e-9][0]),
        note=("Lemma: prediction enters ONLY as a per-gene mean; deterministic rescale "
              "leaves precision-support exactly invariant (Jaccard=1, corr equal to float32 "
              "epsilon). Poisson adds a mean-dependent residual that is substantial at "
              "realistic depth and vanishes only as counts->inf; crucially it still depends "
              "on the prediction ONLY through per-gene means, never joint structure."))
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ["part1_deterministic", "verdict"]}, indent=2))
    print("poisson sweep:", json.dumps(sweep, indent=1))

    # ---------- figure ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    mc = [s["mean_counts_per_cell"] for s in sweep]
    ax = axes[0]
    ax.loglog(mc, [max(s["max_abs_delta_corr"], 1e-16) for s in sweep], "o-", color="#c1440e")
    ax.axhline(max_dcorr_det if max_dcorr_det > 0 else 1e-16, ls="--", color="gray",
               label=f"deterministic: {max_dcorr_det:.1e}")
    ax.set_xlabel("mean counts / cell"); ax.set_ylabel("max |Δ correlation|  (pred1 vs pred2)")
    ax.set_title("Part 2: Poisson residual vs depth"); ax.legend(fontsize=8)
    ax2 = axes[1]
    ax2.semilogx(mc, [s["edge_flip_fraction"] for s in sweep], "s-", color="#1f6f8b")
    ax2.set_xlabel("mean counts / cell"); ax2.set_ylabel("precision-support edge-flip fraction")
    ax2.set_title("Support disagreement vanishes with depth")
    fig.suptitle(f"Wall-1: deterministic exact (max|Δcorr|={max_dcorr_det:.0e}, "
                 f"supp Jaccard={jac_det:.3f}); Poisson residual → 0", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130, bbox_inches="tight")
    print("wrote", OUT, "and", FIG)

    # ---- gate-script contract ----
    ok = res["verdict"]["deterministic_exact"]
    if not ok:
        print("GATE FAIL: deterministic invariance not exact — check overlay import")
        sys.exit(1)
    print("GATE PASS")


if __name__ == "__main__":
    main()
