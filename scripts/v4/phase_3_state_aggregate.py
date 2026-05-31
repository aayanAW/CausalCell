"""Aggregate STATE multi-seed GIES F1 into a mean + bootstrap CI.

Reads results/state/state_f1_summary*.json (one per seed) and reports
mean F1 with a normal-approx 95% CI across seeds, matching how the other
models' multi-seed numbers are reported in the manuscript.

Usage:
    python3 scripts/v4/phase_3_state_aggregate.py
"""
from __future__ import annotations
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
STATE_DIR = ROOT / "results" / "state"


def main():
    files = sorted(glob.glob(str(STATE_DIR / "state_f1_summary*.json")))
    f1s, seeds = [], []
    for fp in files:
        d = json.load(open(fp))
        f1 = d.get("state", {}).get("f1")
        if f1 is not None:
            f1s.append(f1)
            seeds.append(d.get("seed"))
    if not f1s:
        print("No STATE summaries found.")
        return
    f1s = np.array(f1s, dtype=float)
    n = len(f1s)
    mean = float(f1s.mean())
    if n >= 2:
        se = float(f1s.std(ddof=1) / np.sqrt(n))
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
    else:
        lo = hi = mean
    out = {
        "model": "state",
        "n_seeds": n,
        "seeds": seeds,
        "f1_per_seed": f1s.tolist(),
        "f1_mean": mean,
        "f1_ci95": [lo, hi],
    }
    summary = STATE_DIR / "state_multiseed_summary.json"
    json.dump(out, open(summary, "w"), indent=2)
    print(f"STATE multi-seed: n={n} seeds={seeds}")
    print(f"  F1 = {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  per-seed: {[round(x,4) for x in f1s.tolist()]}")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
