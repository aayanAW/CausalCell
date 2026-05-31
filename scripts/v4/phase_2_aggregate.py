"""Aggregate Phase 2 cross-dataset substitutability results.

Reads the per-dataset eval JSONs (K562 from the existing multi-seed/eval, plus
the new RPE1 cross-context and Norman cross-paradigm runs) and emits a single
table of GIES-F1 substitutability per model per dataset, plus the substitutability
GAP (model F1 vs real-data upper bound) that is the paper's core quantity.

Usage:
    python3 scripts/v4/phase_2_aggregate.py
"""
from __future__ import annotations
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "results"

# dataset -> (eval json path, human label)
SOURCES = {
    "K562 (in-domain, CRISPRi)":    RES / "eval_results_n200.json",
    "RPE1 (cross-context, CRISPRi)": RES / "phase2_eval_rpe1.json",
    "Norman (cross-paradigm, CRISPRa)": RES / "phase2_eval_norman.json",
}

# Deep-model condition keys are written as "{model}_real" by run_eval_local;
# baselines keep their bare names. We normalize "{model}_real" -> "{model}".
MODEL_KEYS = ["gears", "cpa", "geneformer", "elastic_net", "overlay_resampled_real", "random"]


def _extract(d):
    """Pull per-condition F1 from a run_eval_local results json.

    Prefer the bootstrap-CI mean (what the manuscript reports) over the single-run
    point estimate, and normalize '{model}_real' keys to '{model}'.
    """
    out = {}
    conds = d.get("conditions") or {}
    boot = d.get("bootstrap_ci") or {}

    def f1_of(key):
        if key in boot and isinstance(boot[key], dict) and "f1" in boot[key]:
            return boot[key]["f1"].get("mean")
        if key in conds and isinstance(conds[key], dict):
            return conds[key].get("f1")
        return None

    for k in conds:
        kl = k.lower()
        norm = kl[:-5] if kl.endswith("_real") and kl != "overlay_resampled_real" else kl
        v = f1_of(k)
        if v is not None:
            out[norm] = v
    return out


def main():
    rows = []
    real_ub = {}
    for label, path in SOURCES.items():
        if not path.exists():
            print(f"  [missing] {label}: {path.name}")
            continue
        d = json.load(open(path))
        f1s = _extract(d)
        ub = f1s.get("real_data", 1.0)
        real_ub[label] = ub
        for mk in MODEL_KEYS:
            if mk in f1s:
                rows.append((label, mk, f1s[mk], ub, ub - f1s[mk]))

    print("\n%-34s %-22s %7s %7s %7s" % ("Dataset", "Model", "F1", "UB", "Gap"))
    print("-" * 80)
    for label, mk, f1, ub, gap in rows:
        print("%-34s %-22s %7.3f %7.3f %7.3f" % (label, mk, f1, ub, gap))

    out = RES / "phase2_cross_dataset_summary.json"
    json.dump({"rows": [dict(dataset=r[0], model=r[1], f1=r[2], ub=r[3], gap=r[4])
                        for r in rows], "real_ub": real_ub}, open(out, "w"), indent=2)
    print(f"\nWrote {out}")
    if not rows:
        print("No eval JSONs found yet — run phase_2_eval.sh after predictions download.")


if __name__ == "__main__":
    main()
