#!/usr/bin/env python3
"""J1 multi-seed: CI on the GIES-on-real ground truth vs TRRUST lift.

Hardens the single-seed J1 (scripts/j1_trrust_groundtruth_validity.py) into a
multi-seed estimate. The TF-rich subset is deterministic; across seeds the
control subsample, per-perturbation cell subsample, and GIES jitter vary, giving
a CI on precision / recall / lift-over-chance of the real-data ground truth
against the curated TRRUST network. Fully local, no GPU.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from causalcellbench.data.trrust import load_trrust  # noqa: E402
from scripts.run_eval_local import build_cb_input, run_gies  # noqa: E402

K562 = ROOT / "data" / "k562_real.h5ad"
OUT = ROOT / "results" / "track_b" / "j1_trrust_multiseed.json"
SEEDS = [42, 123, 456, 789, 1024]
MAX_GENES, MIN_CELLS, N_CELLS, N_CTRL, PARTITION = 70, 50, 150, 200, 20
POOL_PER_PERT = 400
CONTROL = "non-targeting"


def main():
    trrust = load_trrust()
    tr_edges = {(e[0], e[1]) for e in trrust}
    tr_tfs = {a for a, _ in tr_edges}

    A = ad.read_h5ad(K562, backed="r")
    sym = A.var["gene_name"].astype(str).values
    measurable = set(sym)
    pert = A.obs["gene"].astype(str)
    counts = Counter(pert)
    eligible = {
        g
        for g, n in counts.items()
        if n >= MIN_CELLS and g != CONTROL and g in measurable
    }

    seed_tfs = sorted(
        [t for t in tr_tfs if t in eligible],
        key=lambda t: -sum(1 for (a, b) in tr_edges if a == t and b in measurable),
    )
    sset, subset = set(), []
    for tf in seed_tfs:
        if len(sset) >= MAX_GENES:
            break
        for g in [tf] + [b for (a, b) in tr_edges if a == tf and b in measurable]:
            if g not in sset and len(sset) < MAX_GENES:
                sset.add(g)
                subset.append(g)
    internal = {(a, b) for (a, b) in tr_edges if a in sset and b in sset}
    keep = {a for a, _ in internal} | {b for _, b in internal}
    subset = [g for g in subset if g in keep]
    internal = {(a, b) for (a, b) in internal if a in subset and b in subset}
    perturbed = [g for g in subset if g in eligible]
    print(
        f"subset {len(subset)} genes, {len(internal)} TRRUST edges, {len(perturbed)} perturbed"
    )

    sym_to_idx = {}
    for i, s in enumerate(sym):
        sym_to_idx.setdefault(s, i)
    gene_idx = [sym_to_idx[g] for g in subset]

    # load a generous cell pool ONCE
    rng0 = np.random.default_rng(0)
    ctrl_all = np.where((pert == CONTROL).values)[0]
    ctrl_pool = np.sort(rng0.choice(ctrl_all, min(2000, len(ctrl_all)), replace=False))
    pool_by_g = {}
    for g in perturbed:
        pos = np.where((pert == g).values)[0]
        if len(pos) > POOL_PER_PERT:
            pos = rng0.choice(pos, POOL_PER_PERT, replace=False)
        pool_by_g[g] = np.sort(pos)
    allpos = np.sort(np.unique(np.concatenate([ctrl_pool, *pool_by_g.values()])))
    mem = A[allpos].to_memory().X
    mem = np.asarray(
        mem.todense() if hasattr(mem, "todense") else mem, dtype=np.float64
    )[:, gene_idx]
    rowmap = {int(p): i for i, p in enumerate(allpos)}
    ctrl_pool_X = mem[[rowmap[int(p)] for p in ctrl_pool]]
    pool_X = {g: mem[[rowmap[int(p)] for p in pool_by_g[g]]] for g in perturbed}

    G = len(subset)
    n_pairs = G * (G - 1)

    def score(edges, rng):
        tp = len(edges & internal)
        p = tp / len(edges) if edges else 0.0
        r = tp / len(internal) if internal else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        rp = []
        nodes = subset
        for _ in range(200):
            re = set()
            while len(re) < len(edges):
                a, b = rng.choice(len(nodes), 2, replace=False)
                re.add((nodes[a], nodes[b]))
            rp.append(len(re & internal) / max(1, len(re)))
        chance = float(np.mean(rp))
        return p, r, f, chance, (p / chance if chance > 0 else None), len(edges)

    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        ci = rng.choice(
            ctrl_pool_X.shape[0], min(N_CTRL, ctrl_pool_X.shape[0]), replace=False
        )
        ctrl_X = ctrl_pool_X[ci]
        mats, labs = [], []
        for g in perturbed:
            P = pool_X[g]
            idx = rng.choice(P.shape[0], min(N_CELLS, P.shape[0]), replace=False)
            mats.append(P[idx])
            labs.extend([g] * len(idx))
        cb = build_cb_input(ctrl_X, mats, labs, subset)
        edges = {(a, b) for a, b in run_gies(cb, seed=seed, partition_size=PARTITION)}
        p, r, f, chance, lift, ne = score(edges, rng)
        rows.append(
            {
                "seed": seed,
                "precision": p,
                "recall": r,
                "f1": f,
                "chance": chance,
                "lift": lift,
                "n_edges": ne,
            }
        )
        print(f"seed {seed}: prec={p:.3f} rec={r:.3f} lift={lift:.2f} edges={ne}")

    def agg(k):
        v = np.array([row[k] for row in rows], dtype=float)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}

    res = {
        "subset_size": G,
        "trrust_internal_edges": len(internal),
        "n_perturbed": len(perturbed),
        "per_seed": rows,
        "precision": agg("precision"),
        "recall": agg("recall"),
        "lift": agg("lift"),
        "chance": agg("chance"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print("\n=== J1 MULTI-SEED ===")
    print(
        f"precision {res['precision']['mean']:.3f} ± {res['precision']['sd']:.3f}  "
        f"recall {res['recall']['mean']:.3f} ± {res['recall']['sd']:.3f}  "
        f"lift {res['lift']['mean']:.2f} ± {res['lift']['sd']:.2f}x over chance"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
