#!/usr/bin/env python3
"""A3 mechanism discriminator — mean-matched null.

Pre-registered in prereg_A3_mechanism_discriminator.md. THIS SCRIPT IS THE GATE
(and its own test suite): it asserts the reused A3 machinery reproduces the locked
A3 numbers, then runs the pre-registered B-vs-C gate.

Discriminates WHY real Norman doubles beat additive-of-singles:
  A. real_doubles       : real double CELLS (covariance + non-additive means)
  B. real_means_overlay : real doubles' per-gene MEAN through the overlay
                          (joint structure destroyed, non-additive means kept)  [NEW]
  C. additive_singles   : additive-of-singles mean through the overlay          [baseline]

B vs C on the anchored GT isolates the marginal-mean channel (both go through the
identical overlay). A vs B isolates the cell-path joint structure.

Runs in .venv_gears (Py3.9). CPU only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.run_eval_local import (  # noqa: E402
    build_cb_input,
    run_gies,
    sample_perturbed_cells_overlay,
)
from scripts.a3_combinations import anchored_from_doubles, score  # noqa: E402

NORMAN = ROOT / "data" / "norman2019.h5ad"
OUT = ROOT / "results" / "hardening" / "a3_mechanism_discriminator.json"
SEEDS = [42, 123, 456, 789, 1024]
N_CELLS, N_CTRL, PARTITION = 100, 2000, 20
MIN_CELLS = 20

# locked A3 reference numbers (from RESULTS_AB_FIXES.md / a3_combinations.json)
A3_REAL_REF = (0.0552, 0.0056)
A3_ADD_REF = (0.0359, 0.0037)


def build_panel():
    a = ad.read_h5ad(NORMAN)
    sym = a.var_names.astype(str).values
    pert = a.obs["perturbation"].astype(str).values
    npv = a.obs["nperts"].astype(int).values
    X = a.X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)

    doubles = sorted(set(pert[npv == 2]))
    single_set = set(pert[npv == 1])
    dbl_genes = set()
    for d in doubles:
        dbl_genes |= set(d.split("_"))
    genes = sorted(g for g in dbl_genes if g in set(sym) and g in single_set)
    gidx = {g: i for i, g in enumerate(genes)}
    col = [np.where(sym == g)[0][0] for g in genes]
    doubles = [d for d in doubles if all(g in gidx for g in d.split("_"))]

    Xp = X[:, col]
    lib = X.sum(1, keepdims=True); med = np.median(lib[lib > 0])
    Xlog = np.log1p(Xp / np.maximum(lib, 1e-9) * med)

    ctrl_mask = npv == 0
    ctrl_log = Xlog[ctrl_mask]
    ctrl_lin = np.expm1(ctrl_log.mean(0))
    ctrl_raw = Xp[ctrl_mask]
    ctrl_raw_mean = ctrl_raw.mean(0)

    dbl_cells_raw, dbl_log = {}, {}
    for d in doubles:
        m = pert == d
        if m.sum() < MIN_CELLS:
            continue
        dbl_cells_raw[d] = Xp[m]
        dbl_log[d] = Xlog[m]
    doubles = list(dbl_cells_raw.keys())

    single_mean_raw = {}
    for g in genes:
        m = pert == g
        if m.sum() >= MIN_CELLS:
            single_mean_raw[g] = Xp[m].mean(0)

    add_doubles = [d for d in doubles if all(g in single_mean_raw for g in d.split("_"))]
    anch = anchored_from_doubles(None, ctrl_lin, ctrl_log, dbl_log, genes)
    return dict(genes=genes, ctrl_raw=ctrl_raw, ctrl_raw_mean=ctrl_raw_mean,
                dbl_cells_raw=dbl_cells_raw, single_mean_raw=single_mean_raw,
                add_doubles=add_doubles, anch=anch)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    P = build_panel()
    genes = P["genes"]; anch = P["anch"]; add_doubles = P["add_doubles"]
    meta = dict(panel_genes=len(genes), n_doubles=len(add_doubles),
                anchored_edges=len(anch), seeds=SEEDS,
                conditions=["real_doubles", "real_means_overlay", "additive_singles"],
                design="B=real double MEANS through overlay (mean-matched null)")
    print(json.dumps(meta))

    results = {"meta": meta, "conditions": {}}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        ctrl_all = P["ctrl_raw"]
        ci = rng.choice(ctrl_all.shape[0], size=min(N_CTRL, ctrl_all.shape[0]), replace=False)
        ctrl_X = ctrl_all[ci]

        conds = {}
        # A. real_doubles (cells)
        rm, rl = [], []
        for d in add_doubles:
            gc = P["dbl_cells_raw"][d]
            if gc.shape[0] > N_CELLS:
                gc = gc[rng.choice(gc.shape[0], N_CELLS, replace=False)]
            rm.append(gc); rl += [d] * gc.shape[0]
        conds["real_doubles"] = build_cb_input(ctrl_X, rm, rl, genes)

        # B. real_means_overlay — real double MEAN through the SAME overlay
        bm, bl = [], []
        for d in add_doubles:
            real_mean = P["dbl_cells_raw"][d].mean(0)            # real double per-gene mean
            gs = seed + int(hashlib.md5((d + "B").encode()).hexdigest()[:8], 16)
            bm.append(sample_perturbed_cells_overlay(ctrl_X, real_mean, N_CELLS, seed=gs))
            bl += [d] * N_CELLS
        conds["real_means_overlay"] = build_cb_input(ctrl_X, bm, bl, genes)

        # C. additive_singles — additive mean through the overlay (identical to A3)
        am, al = [], []
        for d in add_doubles:
            A, B = d.split("_")
            crm = P["ctrl_raw_mean"]
            mean_pred = np.maximum(crm + (P["single_mean_raw"][A] - crm)
                                   + (P["single_mean_raw"][B] - crm), 0.0)
            gs = seed + int(hashlib.md5(d.encode()).hexdigest()[:8], 16)
            am.append(sample_perturbed_cells_overlay(ctrl_X, mean_pred, N_CELLS, seed=gs))
            al += [d] * N_CELLS
        conds["additive_singles"] = build_cb_input(ctrl_X, am, al, genes)

        gies_edges = {c: set(run_gies(cb, seed=seed, partition_size=PARTITION))
                      for c, cb in conds.items()}
        primary_gt = gies_edges["real_doubles"]  # within-regime secondary GT
        for cname in conds:
            edges = gies_edges[cname]
            row = results["conditions"].setdefault(
                cname, {"anchored": [], "within": [], "n_edges": []})
            row["n_edges"].append(len(edges))
            row["anchored"].append(score(edges, anch))
            row["within"].append(score(edges, primary_gt))
        print(f"seed {seed}: " + ", ".join(
            f"{c[:4]}={results['conditions'][c]['anchored'][-1]['f1']:.4f}" for c in conds))

    for c, row in results["conditions"].items():
        f1 = [x["f1"] for x in row["anchored"]]
        row["summary"] = dict(anchored_f1_mean=float(np.mean(f1)),
                              anchored_f1_sd=float(np.std(f1, ddof=1)),
                              anchored_per_seed=[round(x, 4) for x in f1],
                              n_edges_mean=float(np.mean(row["n_edges"])))

    A = np.array(results["conditions"]["real_doubles"]["summary"]["anchored_per_seed"])
    Bc = np.array(results["conditions"]["real_means_overlay"]["summary"]["anchored_per_seed"])
    C = np.array(results["conditions"]["additive_singles"]["summary"]["anchored_per_seed"])

    # PRIMARY GATE: B vs C (marginal-mean channel)
    dBC = Bc - C
    dAB = A - Bc
    gate = dict(
        primary_contrast="real_means_overlay (B) vs additive_singles (C) on anchored GT",
        B_mean=float(Bc.mean()), C_mean=float(C.mean()),
        BC_gain=float(dBC.mean()), BC_sd=float(dBC.std(ddof=1)), BC_nwin=int((dBC > 0).sum()),
        CONFIRM=bool((dBC > 0).sum() >= 4 and dBC.mean() > dBC.std(ddof=1)),
        secondary_AB_gain=float(dAB.mean()), secondary_AB_sd=float(dAB.std(ddof=1)),
        secondary_AB_nwin=int((dAB > 0).sum()),
    )
    if gate["CONFIRM"]:
        gate["verdict"] = "CONFIRM: A3 advantage is marginal-mean non-additivity (within Wall-1)"
    elif (dBC.mean() <= dBC.std(ddof=1)) or (dBC > 0).sum() < 3:
        gate["verdict"] = "KILL: B~=C; real advantage came from cell-path joint structure (investigate)"
    else:
        gate["verdict"] = "AMBIGUOUS: partial mean channel"
    results["gate"] = gate
    print(json.dumps(gate, indent=2))

    # ---- embedded test suite (sanity asserts; infra bug if these fail) ----
    tests = {}
    tests["nonempty_edges"] = all(
        max(results["conditions"][c]["n_edges"]) > 0 for c in results["conditions"])
    tests["C_reproduces_A3_additive"] = bool(
        abs(C.mean() - A3_ADD_REF[0]) <= 2 * A3_ADD_REF[1] + C.std(ddof=1))
    tests["A_reproduces_A3_real"] = bool(
        abs(A.mean() - A3_REAL_REF[0]) <= 2 * A3_REAL_REF[1] + A.std(ddof=1))
    results["tests"] = tests
    print("TESTS:", json.dumps(tests))

    json.dump(results, open(OUT, "w"), indent=2)
    print("wrote", OUT)
    # gate script exit contract: fail (nonzero) only on infra-test failure
    if not all(tests.values()):
        print("GATE-SCRIPT FAIL: sanity tests did not pass — infrastructure bug, not a result")
        sys.exit(1)
    print("GATE-SCRIPT PASS")


if __name__ == "__main__":
    main()
