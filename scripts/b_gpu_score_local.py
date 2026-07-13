#!/usr/bin/env python3
"""Score the GPU CVAE predicted means (preds.json) through overlay->GIES vs the
A1 anchored GT. Runs LOCALLY in .venv_gears (GIES/R stack).

For each (term, seed): take predicted per-perturbation means, overlay-sample cells,
run GIES, score recovered edges vs the anchored GT. Compares each joint-structure
objective (precision/mmd/sinkhorn) against recon-only. Honest keep/discard:
term beats recon on >=3/5... here 3 seeds, so >=2/3 AND mean gain > SD.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.run_eval_local import (  # noqa: E402
    build_cb_input, run_gies, sample_perturbed_cells_overlay,
)
import anndata as ad

PREDS = ROOT / "results" / "phaseB" / "gpu_preds.json"
ANCHORED = ROOT / "data" / "ground_truth" / "perturbation_anchored_k562_n200.json"
SLIM = ROOT / "data" / "k562_subset_n200_slim.h5ad"
OUT = ROOT / "results" / "phaseB" / "b_gpu_definitive.json"
CTRL = "non-targeting"
N_CELLS, N_CTRL, PARTITION = 100, 200, 20


def score(pred, gt):
    pred, gt = set(map(tuple, pred)), set(map(tuple, gt))
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt); prec = tp/len(pred); rec = tp/len(gt)
    return 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0


def main():
    preds = json.load(open(PREDS))
    genes = preds["meta"]["genes"]
    anchored = json.load(open(ANCHORED))
    pk = anchored["meta"]["primary_key"]
    gt = set(tuple(e) for e in anchored["edge_sets"][pk])

    slim = ad.read_h5ad(SLIM)
    sym = list(slim.var["gene_name"].astype(str).values)
    cc = slim.X[slim.obs["gene"].astype(str).values == CTRL]
    cc = np.asarray(cc.todense() if hasattr(cc, "todense") else cc, dtype=np.float64)[:, [sym.index(g) for g in genes]]
    ctrl = cc[np.random.default_rng(0).choice(cc.shape[0], min(N_CTRL, cc.shape[0]), replace=False)]
    # predicted means are in log1p-normalized space; overlay sampler expects count-like
    # means. Convert back with expm1 (matches training normalization target scale).
    def to_counts(mu):
        return np.maximum(np.expm1(np.asarray(mu)), 0.0)

    res = {"meta": preds["meta"], "gt_key": pk, "n_gt": len(gt), "conditions": {}}
    for term, byseed in preds["predicted_means"].items():
        f1s = []
        for seed, means in byseed.items():
            mats, labs = [], []
            for p, mu in means.items():
                s = int(hashlib.md5((term+seed+p).encode()).hexdigest()[:8], 16)
                mats.append(sample_perturbed_cells_overlay(ctrl, to_counts(mu), N_CELLS, seed=s))
                labs += [p]*N_CELLS
            edges = list(run_gies(build_cb_input(ctrl, mats, labs, genes), seed=int(seed), partition_size=PARTITION))
            f1 = score(edges, gt); f1s.append(f1)
            print(f"{term} seed{seed}: F1={f1:.4f} ({len(edges)} edges)", flush=True)
        res["conditions"][term] = dict(f1_mean=float(np.mean(f1s)), f1_sd=float(np.std(f1s, ddof=1)),
                                       per_seed=[round(x,4) for x in f1s])
    # gates vs recon
    recon = np.array(res["conditions"]["recon"]["per_seed"])
    res["gates"] = {}
    for term in [t for t in res["conditions"] if t != "recon"]:
        d = np.array(res["conditions"][term]["per_seed"]) - recon
        res["gates"][term] = dict(mean_gain=float(d.mean()), sd=float(d.std(ddof=1)),
                                  n_win=int((d > 0).sum()),
                                  keep=bool((d > 0).sum() >= 2 and d.mean() > d.std(ddof=1)))
    json.dump(res, open(OUT, "w"), indent=2)
    print("\n=== B DEFINITIVE (GPU CVAE) vs anchored GT ===")
    for t, r in res["conditions"].items():
        print(f"{t:12s} F1={r['f1_mean']:.4f}±{r['f1_sd']:.4f} {r['per_seed']}")
    for t, g in res["gates"].items():
        print(f"gate[{t}]: {g['n_win']}/3 win, gain {g['mean_gain']:+.4f}±{g['sd']:.4f}, KEEP={g['keep']}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
