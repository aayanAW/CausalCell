"""Per-seed F1 strip plot supporting the multi-seed equivalence (Appendix)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

agg = json.loads(
    (ROOT / "results" / "multi_seed" / "aggregated_results.json").read_text()
)
ps = agg["per_seed"]
SEEDS = ["42", "123", "456", "789", "1024"]

WONG = {
    "elastic_net": "#0072B2",
    "geneformer_real": "#009E73",
    "cpa_real": "#D55E00",
    "gears_real": "#CC79A7",
}
LABELS = {
    "elastic_net": "ElasticNet",
    "geneformer_real": "Geneformer",
    "cpa_real": "CPA",
    "gears_real": "GEARS",
}
ORDER = ["elastic_net", "geneformer_real", "cpa_real", "gears_real"]


def f1_series(cond: str) -> list[float]:
    vals = []
    for s in SEEDS:
        conds = ps[s].get("conditions", ps[s])
        if isinstance(conds, dict) and cond in conds:
            vals.append(conds[cond]["f1"])
    return vals


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

fig, ax = plt.subplots(figsize=(3.4, 2.5))
rng = np.random.default_rng(0)
for i, cond in enumerate(ORDER):
    ys = f1_series(cond)
    xs = i + rng.uniform(-0.12, 0.12, size=len(ys))
    ax.scatter(
        xs, ys, s=18, color=WONG[cond], edgecolor="black", linewidth=0.3, zorder=3
    )
    m = float(np.mean(ys))
    ax.hlines(m, i - 0.25, i + 0.25, color="black", lw=1.3, zorder=4)

ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([LABELS[c] for c in ORDER], fontsize=7)
ax.set_ylabel("Causal-recovery $F_1$ (per seed)")
ax.set_ylim(0.36, 0.47)
ax.set_title("Per-seed $F_1$ across 5 seeds ($N{=}200$ K562)", fontsize=8)
ax.grid(axis="y", lw=0.3, alpha=0.4)

fig.tight_layout()
fig.savefig(OUT / "fig_seeds.pdf")
fig.savefig(OUT / "fig_seeds.png")
plt.close(fig)
print(f"Wrote {OUT / 'fig_seeds.pdf'}")
