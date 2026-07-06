#!/usr/bin/env python3
"""Score the B2 Sinkhorn lambda-sweep predictions (b2_preds.json) through
overlay->GIES vs the A1 anchored GT. Runs LOCALLY in .venv_gears.

Primary (confirmatory): sinkhorn_lam1 vs recon, 5 seeds, gate = >=4/5 AND mean>SD.
Secondary (exploratory): dose-response over lambda in {0.3,1,3,10}.
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

PREDS = ROOT / "results" / "phaseB" / "b2_preds.json"
ANCHORED = ROOT / "data" / "ground_truth" / "perturbation_anchored_k562_n200.json"
SLIM = ROOT / "data" / "k562_subset_n200_slim.h5ad"
OUT = ROOT / "results" / "phaseB" / "b2_ot_definitive.json"
CTRL = "non-targeting"
N_CELLS, N_CTRL, PARTITION = 100, 200, 20


def f1(pred, gt):
    pred, gt = set(map(tuple, pred)), set(map(tuple, gt))
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt); p = tp/len(pred); r = tp/len(gt)
    return 2*p*r/(p+r) if p+r > 0 else 0.0


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

    def counts(mu):
        return np.maximum(np.expm1(np.asarray(mu)), 0.0)

    res = {"meta": preds["meta"], "gt_key": pk, "n_gt": len(gt), "conditions": {}}
    for cond, byseed in preds["predicted_means"].items():
        f1s = []
        for seed, means in byseed.items():
            mats, labs = [], []
            for p, mu in means.items():
                s = int(hashlib.md5((cond+seed+p).encode()).hexdigest()[:8], 16)
                mats.append(sample_perturbed_cells_overlay(ctrl, counts(mu), N_CELLS, seed=s))
                labs += [p]*N_CELLS
            edges = list(run_gies(build_cb_input(ctrl, mats, labs, genes), seed=int(seed), partition_size=PARTITION))
            v = f1(edges, gt); f1s.append(v)
            print(f"{cond} seed{seed}: F1={v:.4f}", flush=True)
        res["conditions"][cond] = dict(f1_mean=float(np.mean(f1s)), f1_sd=float(np.std(f1s, ddof=1)),
                                       per_seed={k: round(x, 4) for k, x in zip(byseed.keys(), f1s)})
    # confirmatory gate: sinkhorn_lam1 vs recon
    recon = np.array(list(res["conditions"]["recon"]["per_seed"].values()))
    star_key = next((k for k in res["conditions"] if k.endswith("lam1") or k.endswith("lam1.0")), None)
    if star_key:
        d = np.array(list(res["conditions"][star_key]["per_seed"].values())) - recon
        res["confirmatory_gate"] = dict(condition=star_key, mean_gain=float(d.mean()),
                                        sd=float(d.std(ddof=1)), n_win=int((d > 0).sum()), n_seeds=len(d),
                                        keep=bool((d > 0).sum() >= 4 and d.mean() > d.std(ddof=1)))
    # dose-response
    res["dose_response"] = {c: res["conditions"][c]["f1_mean"] for c in res["conditions"]}
    json.dump(res, open(OUT, "w"), indent=2)
    print("\n=== B2 OT confirmatory + dose-response vs anchored GT ===")
    for c, r in res["conditions"].items():
        print(f"{c:16s} F1={r['f1_mean']:.4f}±{r['f1_sd']:.4f}")
    g = res.get("confirmatory_gate", {})
    print(f"\nCONFIRMATORY [{g.get('condition')}]: {g.get('n_win')}/{g.get('n_seeds')} win, "
          f"gain {g.get('mean_gain'):+.4f}±{g.get('sd'):.4f}, KEEP={g.get('keep')}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
