#!/usr/bin/env python3
"""J1: Does the GIES-on-real ground truth encode real biology?

The circularity critique of Track A is that the ground truth is GIES applied to
real data, validated against nothing external. The original Track B was vacuous
because the essential-gene N=200 subset contained zero TRRUST edges. Here we
build a *TF-rich* subset that maximizes curated-edge coverage, run GIES on the
REAL K562 perturbation cells over that subset, and measure the recovered
ground-truth graph against the literature-curated TRRUST v2 network with a
random-edge chance baseline. A ground truth that recovers TRRUST edges above
chance has demonstrable biological content; one that does not is an honest
limitation. Either way the circularity defense stops being empty.

Fully local, no GPU. Reuses the project's GIES runner and TRRUST loader.
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
OUT = ROOT / "results" / "track_b" / "j1_trrust_groundtruth.json"
MAX_GENES = 70
MIN_CELLS_PER_PERT = 50
N_CELLS_PER_PERT = 150
PARTITION = 20
SEED = 42
CONTROL_LABELS = {"non-targeting", "non_targeting", "control", "NTC", "neg"}


def directed_set(edges):
    return {(a, b) for a, b in edges}


def main():
    rng = np.random.default_rng(SEED)
    trrust = load_trrust()  # list of (tf, target, sign) or (tf, target)
    tr_edges = directed_set((e[0], e[1]) for e in trrust)
    tr_tfs = {a for a, _ in tr_edges}
    tr_targets = {b for _, b in tr_edges}
    print(f"TRRUST: {len(tr_edges)} directed edges, {len(tr_tfs)} TFs")

    A = ad.read_h5ad(K562, backed="r")
    sym = A.var["gene_name"].astype(str).values
    measurable = set(sym)
    pert = A.obs["gene"].astype(str)
    counts = Counter(pert)
    ctrl_label = next(
        (c for c in pert.unique() if str(c).lower() in CONTROL_LABELS), None
    )
    if ctrl_label is None:  # fall back to most common label
        ctrl_label = counts.most_common(1)[0][0]
    print(f"control label = {ctrl_label!r} ({counts[ctrl_label]} cells)")

    eligible = {
        g
        for g, n in counts.items()
        if n >= MIN_CELLS_PER_PERT and g != ctrl_label and g in measurable
    }
    # TF-rich subset: TRRUST TFs that are perturbed + eligible, plus measurable targets
    seed_tfs = [t for t in tr_tfs if t in eligible]
    # rank TFs by number of measurable targets to maximise internal edges
    seed_tfs.sort(
        key=lambda t: -sum(1 for (a, b) in tr_edges if a == t and b in measurable)
    )
    subset = []
    sset = set()
    for tf in seed_tfs:
        if len(sset) >= MAX_GENES:
            break
        tgts = [b for (a, b) in tr_edges if a == tf and b in measurable]
        add = [tf] + tgts
        for g in add:
            if g not in sset and len(sset) < MAX_GENES:
                sset.add(g)
                subset.append(g)
    # keep only genes in >=1 within-subset TRRUST edge
    internal = {(a, b) for (a, b) in tr_edges if a in sset and b in sset}
    keep = {a for a, _ in internal} | {b for _, b in internal}
    subset = [g for g in subset if g in keep]
    internal = {(a, b) for (a, b) in internal if a in subset and b in subset}
    perturbed_in_subset = [g for g in subset if g in eligible]
    print(
        f"TF-rich subset: {len(subset)} genes, {len(internal)} internal TRRUST edges, "
        f"{len(perturbed_in_subset)} perturbed"
    )
    if len(internal) < 5:
        print("ABORT: too few curated edges in subset")
        return

    # symbol -> var index
    sym_to_idx = {}
    for i, s in enumerate(sym):
        sym_to_idx.setdefault(s, i)
    gene_idx = [sym_to_idx[g] for g in subset]

    # cells: controls + perturbed cells for perturbed-in-subset genes.
    # Gather all needed row positions, do ONE backed slice to memory, then
    # subset genes + split by label in memory (avoids backed view-of-view).
    ctrl_pos = np.where((pert == ctrl_label).values)[0]
    if len(ctrl_pos) > 2000:
        ctrl_pos = rng.choice(ctrl_pos, 2000, replace=False)
    pert_pos_by_gene = {}
    for g in perturbed_in_subset:
        pos = np.where((pert == g).values)[0]
        if len(pos) > N_CELLS_PER_PERT:
            pos = rng.choice(pos, N_CELLS_PER_PERT, replace=False)
        pert_pos_by_gene[g] = pos

    all_pos = np.sort(np.unique(np.concatenate([ctrl_pos, *pert_pos_by_gene.values()])))
    mem = A[all_pos].to_memory()
    Xfull = mem.X
    Xfull = np.asarray(
        Xfull.todense() if hasattr(Xfull, "todense") else Xfull, dtype=np.float64
    )
    Xfull = Xfull[:, gene_idx]  # subset genes in memory
    rowmap = {int(p): i for i, p in enumerate(all_pos)}

    ctrl_X = Xfull[[rowmap[int(p)] for p in ctrl_pos]]
    pert_mats, pert_labels = [], []
    for g in perturbed_in_subset:
        rows = [rowmap[int(p)] for p in pert_pos_by_gene[g]]
        pert_mats.append(Xfull[rows])
        pert_labels.extend([g] * len(rows))

    cb = build_cb_input(ctrl_X, pert_mats, pert_labels, subset)
    print(
        f"GIES input: {ctrl_X.shape[0]} ctrl + {sum(m.shape[0] for m in pert_mats)} pert cells, "
        f"{len(subset)} genes"
    )
    gies_edges = directed_set(run_gies(cb, seed=SEED, partition_size=PARTITION))
    print(f"GIES-on-real recovered {len(gies_edges)} directed edges")

    # Score GIES-real vs TRRUST (directed + undirected)
    def prf(pred, gold):
        tp = len(pred & gold)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(gold) if gold else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return dict(
            precision=p, recall=r, f1=f, tp=tp, n_pred=len(pred), n_gold=len(gold)
        )

    undir = lambda S: {frozenset(e) for e in S if e[0] != e[1]}
    G = len(subset)
    n_pairs = G * (G - 1)
    chance_prec = (
        len(internal) / n_pairs
    )  # expected precision of a random directed edge
    # empirical random-edge baseline matched to GIES edge count
    rand_p = []
    nodes = subset
    for _ in range(200):
        re = set()
        while len(re) < len(gies_edges):
            a, b = rng.choice(len(nodes), 2, replace=False)
            re.add((nodes[a], nodes[b]))
        rand_p.append(len(re & internal) / max(1, len(re)))
    rand_prec_emp = float(np.mean(rand_p))

    res = {
        "subset_size": G,
        "n_perturbed": len(perturbed_in_subset),
        "trrust_internal_edges": len(internal),
        "gies_directed": prf(gies_edges, internal),
        "gies_undirected": prf(undir(gies_edges), undir(internal)),
        "chance_precision_analytic": chance_prec,
        "chance_precision_empirical_matched": rand_prec_emp,
        "precision_lift_over_chance": (
            prf(gies_edges, internal)["precision"] / rand_prec_emp
        )
        if rand_prec_emp > 0
        else None,
        "subset_genes": subset,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print("\n=== J1 RESULT ===")
    print(
        f"GIES-real vs TRRUST (directed): precision={res['gies_directed']['precision']:.3f} "
        f"recall={res['gies_directed']['recall']:.3f} f1={res['gies_directed']['f1']:.3f} "
        f"(tp={res['gies_directed']['tp']}/{len(internal)})"
    )
    print(
        f"chance precision (matched random): {rand_prec_emp:.4f}  "
        f"=> lift {res['precision_lift_over_chance']}"
    )
    print(
        f"undirected precision={res['gies_undirected']['precision']:.3f} "
        f"recall={res['gies_undirected']['recall']:.3f}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
