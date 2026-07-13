"""Regenerate Figure 1 (causal recovery) with multi-seed 95% CIs."""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

agg = json.loads((RES / "multi_seed" / "aggregated_results.json").read_text())
techdup = json.loads((RES / "technical_duplicate" / "summary.json").read_text())

WONG = {
    "real": "#000000",
    "elastic": "#0072B2",
    "overlay": "#56B4E9",
    "geneformer": "#009E73",
    "cpa": "#D55E00",
    "gears": "#CC79A7",
    "grnboost": "#999999",
}

ORDER = [
    ("Real Data\n(ceiling)", "real_data", WONG["real"]),
    ("Overlay Real\n(UB)", "overlay_resampled_real", WONG["overlay"]),
    ("ElasticNet", "elastic_net", WONG["elastic"]),
    ("Geneformer", "geneformer_real", WONG["geneformer"]),
    ("CPA", "cpa_real", WONG["cpa"]),
    ("GEARS", "gears_real", WONG["gears"]),
    ("GRNBoost2\n(observational)", "grnboost2_real", WONG["grnboost"]),
]

names, means, lows, highs, colors = [], [], [], [], []
for nm, key, col in ORDER:
    cond = agg["aggregated"].get(key)
    if cond is None:
        continue
    f = cond["f1"]
    means.append(f["mean"])
    lows.append(f["mean"] - f["ci_low"])
    highs.append(f["ci_high"] - f["mean"])
    names.append(nm)
    colors.append(col)

td_f1 = techdup["f1_mean"]
td_lo = td_f1 - techdup["f1_std"]
td_hi = td_f1 + techdup["f1_std"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

fig, ax = plt.subplots(figsize=(3.5, 2.7))
xs = np.arange(len(names))
ax.bar(
    xs,
    means,
    color=colors,
    edgecolor="black",
    linewidth=0.4,
    yerr=[lows, highs],
    capsize=2.5,
    error_kw=dict(elinewidth=0.7),
)

for x, m in zip(xs, means):
    ax.text(x, m + 0.025, f"{m:.3f}", ha="center", va="bottom", fontsize=6.4)

# Technical-duplicate noise floor
ax.axhline(td_f1, color="darkred", lw=0.7, ls="--")
ax.text(
    1.5,
    td_f1 + 0.02,
    f"tech.-duplicate floor F1={td_f1:.3f}",
    fontsize=5.5,
    color="darkred",
    ha="center",
    va="bottom",
)

# Random floor
floor = 0.04
ax.axhline(floor, color="grey", lw=0.5, ls=":")
ax.text(0, floor + 0.005, "random floor", fontsize=5.5, color="grey", ha="left")

ax.set_xticks(xs)
ax.set_xticklabels(
    [n.replace("\n", " ") for n in names],
    fontsize=6.0,
    rotation=22,
    ha="right",
    rotation_mode="anchor",
)
ax.set_ylabel("F1 vs. real-data GIES")
ax.set_ylim(0, 1.12)
ax.set_title("Causal recovery on K562 ($N{=}200$)", fontsize=8.5)

fig.tight_layout()
fig.savefig(OUT / "fig1.pdf")
fig.savefig(OUT / "fig1.png")
plt.close(fig)
print(f"Wrote {OUT / 'fig1.pdf'}")
