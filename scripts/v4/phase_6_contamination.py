"""Phase 6: Geneformer pretraining-contamination probe via cell-line holdout.

Logic: Geneformer is pretrained on Genecorpus-30M, which includes K562 but not
the Replogle RPE1 arm at the same depth. If a foundation model's apparent
substitutability is inflated by having "seen" the in-corpus cell type, its
GIES-F1 should drop MORE from K562 -> RPE1 than a from-scratch model (GEARS/CPA,
which have no pretraining corpus and serve as the contamination-free control).

We report, per model, F1(K562) - F1(RPE1) and compare the foundation-model drop
to the from-scratch drop. A larger foundation-model drop is suggestive (not
proof) of contamination.

CAVEAT (must be stated in the manuscript): this Geneformer "prediction" is a
similarity-heuristic overlay on the control mean, not learned ISP expression,
so the contamination signal is confounded by the heuristic. Treat as
exploratory, not confirmatory (V4 audit H16: no clean post-Genecorpus public
Perturb-seq atlas exists for a fully held-out human cell type).

Usage:
    python3 scripts/v4/phase_6_contamination.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "results"

# F1 sources
K562_JSON = RES / "eval_results_n200.json"
RPE1_JSON = RES / "phase2_eval_rpe1.json"

FOUNDATION = {"geneformer"}          # pretrained on Genecorpus-30M
FROM_SCRATCH = {"gears", "cpa", "elastic_net"}  # no pretraining corpus (controls)


def _f1s(path):
    if not path.exists():
        return {}
    d = json.load(open(path))
    conds = d.get("conditions", {})
    return {k.lower(): v.get("f1") for k, v in conds.items()
            if isinstance(v, dict) and v.get("f1") is not None}


def main():
    k = _f1s(K562_JSON)
    r = _f1s(RPE1_JSON)
    if not k or not r:
        print("Need both K562 + RPE1 eval JSONs. Run phase_2_eval.sh first.")
        return

    rows = []
    for m in sorted(set(k) & set(r)):
        if m in ("real_data", "grnboost2_real", "overlay_resampled_real"):
            continue
        drop = k[m] - r[m]
        kind = ("foundation" if m in FOUNDATION
                else "from-scratch" if m in FROM_SCRATCH else "other")
        rows.append((m, kind, k[m], r[m], drop))

    print("\n%-16s %-13s %8s %8s %8s" % ("Model", "Type", "F1_K562", "F1_RPE1", "Drop"))
    print("-" * 60)
    for m, kind, fk, fr, drop in rows:
        print("%-16s %-13s %8.3f %8.3f %+8.3f" % (m, kind, fk, fr, drop))

    fdn = [d for _, kind, _, _, d in rows if kind == "foundation"]
    fsc = [d for _, kind, _, _, d in rows if kind == "from-scratch"]
    verdict = None
    if fdn and fsc:
        import numpy as np
        mf, ms = float(np.mean(fdn)), float(np.mean(fsc))
        verdict = {"foundation_mean_drop": mf, "from_scratch_mean_drop": ms,
                   "excess_drop": mf - ms,
                   "interpretation": ("foundation drops MORE (suggestive of "
                                      "in-corpus advantage / contamination)"
                                      if mf > ms else
                                      "no excess foundation drop (no contamination signal)")}
        print(f"\nFoundation mean drop={mf:+.3f}  From-scratch mean drop={ms:+.3f}  "
              f"excess={mf-ms:+.3f}")
        print(f"  -> {verdict['interpretation']}")
        print("  CAVEAT: Geneformer prediction is a similarity-heuristic, not learned "
              "ISP; treat as exploratory.")

    out = RES / "geneformer_contamination" / "phase6_holdout_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": [dict(model=m, type=kind, f1_k562=fk, f1_rpe1=fr, drop=d)
                        for m, kind, fk, fr, d in rows],
               "verdict": verdict,
               "caveat": "Geneformer prediction is a similarity-heuristic overlay, "
                         "not learned ISP expression; signal is confounded."},
              open(out, "w"), indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
