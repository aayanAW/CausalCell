"""Generate CausalCellBench evaluation figures.

Reads eval_results_n*.json and produces publication-quality figures.

Usage:
    python3 scripts/generate_figures.py --results results/eval_results_n100.json
    python3 scripts/generate_figures.py --results results/eval_results_n50.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DISPLAY_NAMES = {
    "real_data": "Real Data\n(upper bound)",
    "elastic_net": "ElasticNet",
    "cpa": "CPA",
    "gears": "GEARS",
    "scgpt": "scGPT",
    "geneformer": "Geneformer",
    "grnboost2_real": "GRNBoost2\n(observational)",
    "random": "Random\n(floor)",
}

COLORS = {
    "real_data": "#2ecc71",
    "elastic_net": "#3498db",
    "cpa": "#e74c3c",
    "gears": "#e67e22",
    "scgpt": "#9b59b6",
    "geneformer": "#1abc9c",
    "grnboost2_real": "#95a5a6",
    "random": "#bdc3c7",
}

CONDITION_ORDER = [
    "real_data", "elastic_net", "cpa", "gears",
    "scgpt", "geneformer", "grnboost2_real", "random",
]


def load_results(path):
    with open(path) as f:
        return json.load(f)


def get_ordered_conditions(results):
    """Get conditions in display order, filtering to those present."""
    return [c for c in CONDITION_ORDER if c in results["conditions"]]


def figure1_auprc_bar(results, outdir):
    """Figure 1: AUPRC bar chart across all conditions."""
    conditions = get_ordered_conditions(results)
    names = [DISPLAY_NAMES.get(c, c) for c in conditions]
    auprcs = [results["conditions"][c]["auprc"] for c in conditions]
    colors = [COLORS.get(c, "#7f8c8d") for c in conditions]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(conditions)), auprcs, color=colors, edgecolor="white", linewidth=0.5)

    # Random floor line
    rf = results.get("random_floor_auprc", 0)
    ax.axhline(y=rf, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7, label=f"Random floor ({rf:.3f})")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(names, ha="center")
    ax.set_ylabel("AUPRC")
    ax.set_title(f"CausalCellBench: Causal Graph Recovery (AUPRC)\n{results['n_genes']} genes, {results['n_control_cells']} control cells")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)

    # Add value labels on bars
    for bar, val in zip(bars, auprcs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "figure1_auprc_bars.png")
    fig.savefig(outdir / "figure1_auprc_bars.pdf")
    plt.close(fig)
    print(f"  Saved figure1_auprc_bars.png/pdf")


def figure2_metrics_heatmap(results, outdir):
    """Figure 2: Heatmap of all 5 metrics across conditions."""
    conditions = get_ordered_conditions(results)
    # Skip real_data (it's the ceiling)
    conditions = [c for c in conditions if c != "real_data"]
    names = [DISPLAY_NAMES.get(c, c).replace("\n", " ") for c in conditions]

    metrics = ["auprc", "precision_at_100", "shd", "wasserstein", "false_omission"]
    metric_labels = ["AUPRC", "P@100", "SHD", "Wasserstein", "FOR"]

    data = np.zeros((len(conditions), len(metrics)))
    for i, c in enumerate(conditions):
        cond = results["conditions"][c]
        data[i, 0] = cond.get("auprc", 0)
        data[i, 1] = cond.get("precision_at_100", 0)
        shd_val = cond.get("shd", {})
        data[i, 2] = shd_val.get("shd", 0) if isinstance(shd_val, dict) else shd_val
        ws_val = cond.get("wasserstein", {})
        data[i, 3] = ws_val.get("mean_wasserstein", 0) if isinstance(ws_val, dict) else ws_val
        for_val = cond.get("false_omission", {})
        data[i, 4] = for_val.get("for_rate", 0) if isinstance(for_val, dict) else for_val

    # Normalize each column to [0, 1] for heatmap
    data_norm = data.copy()
    for j in range(data.shape[1]):
        col = data[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            data_norm[:, j] = (col - cmin) / (cmax - cmin)
        else:
            data_norm[:, j] = 0.5

    fig, ax = plt.subplots(figsize=(8, max(4, len(conditions) * 0.6 + 1)))
    im = ax.imshow(data_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(names)

    # Annotate with actual values
    for i in range(len(conditions)):
        for j in range(len(metrics)):
            val = data[i, j]
            text = f"{val:.3f}" if val < 100 else f"{val:.0f}"
            color = "white" if data_norm[i, j] < 0.3 or data_norm[i, j] > 0.7 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    ax.set_title(f"CausalCellBench Metrics Heatmap ({results['n_genes']} genes)")
    fig.colorbar(im, ax=ax, label="Normalized (0=worst, 1=best)")
    fig.tight_layout()
    fig.savefig(outdir / "figure2_metrics_heatmap.png")
    fig.savefig(outdir / "figure2_metrics_heatmap.pdf")
    plt.close(fig)
    print(f"  Saved figure2_metrics_heatmap.png/pdf")


def figure3_fraction_ceiling(results, outdir):
    """Figure 3: Fraction of real-data ceiling per model."""
    conditions = get_ordered_conditions(results)
    conditions = [c for c in conditions if c not in ("real_data", "random")]
    names = [DISPLAY_NAMES.get(c, c).replace("\n", " ") for c in conditions]

    real_edges = results["conditions"]["real_data"]["n_edges"]
    real_auprc = 1.0  # by definition

    fractions = []
    for c in conditions:
        cond = results["conditions"][c]
        # Fraction = model AUPRC / real AUPRC (which is 1.0)
        fractions.append(cond["auprc"])

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS.get(c, "#7f8c8d") for c in conditions]
    bars = ax.barh(range(len(conditions)), fractions, color=colors, edgecolor="white")

    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Fraction of Real-Data Ceiling (AUPRC)")
    ax.set_title(f"Model Performance as Fraction of Real-Data Ceiling\n{results['n_genes']} genes")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=1.0, color="#2ecc71", linestyle="--", linewidth=1, alpha=0.5, label="Real data ceiling")

    for bar, val in zip(bars, fractions):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", ha="left", va="center", fontsize=9)

    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "figure3_fraction_ceiling.png")
    fig.savefig(outdir / "figure3_fraction_ceiling.pdf")
    plt.close(fig)
    print(f"  Saved figure3_fraction_ceiling.png/pdf")


def figure4_shd_breakdown(results, outdir):
    """Figure 4: SHD breakdown (missing, extra, reversed) stacked bar."""
    conditions = get_ordered_conditions(results)
    conditions = [c for c in conditions if c != "real_data"]
    names = [DISPLAY_NAMES.get(c, c).replace("\n", " ") for c in conditions]

    missing_vals, extra_vals, reversed_vals = [], [], []
    for c in conditions:
        shd = results["conditions"][c].get("shd", {})
        if isinstance(shd, dict):
            missing_vals.append(shd.get("missing", 0))
            extra_vals.append(shd.get("extra", 0))
            reversed_vals.append(shd.get("reversed", 0))
        else:
            missing_vals.append(0)
            extra_vals.append(0)
            reversed_vals.append(0)

    x = np.arange(len(conditions))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 5))
    p1 = ax.bar(x, missing_vals, width, label="Missing edges", color="#e74c3c", alpha=0.8)
    p2 = ax.bar(x, extra_vals, width, bottom=missing_vals, label="Extra edges", color="#f39c12", alpha=0.8)
    bottoms = [m + e for m, e in zip(missing_vals, extra_vals)]
    p3 = ax.bar(x, reversed_vals, width, bottom=bottoms, label="Reversed edges", color="#3498db", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, ha="center")
    ax.set_ylabel("Count")
    ax.set_title(f"Structural Hamming Distance Breakdown ({results['n_genes']} genes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "figure4_shd_breakdown.png")
    fig.savefig(outdir / "figure4_shd_breakdown.pdf")
    plt.close(fig)
    print(f"  Saved figure4_shd_breakdown.png/pdf")


def figure5_prediction_vs_causal(results, outdir):
    """Figure 5: Prediction accuracy vs causal graph quality (THE killer figure).

    X-axis: Wasserstein distance to real perturbation data (lower = better prediction)
    Y-axis: AUPRC of causal graph (higher = better causal recovery)

    If these are uncorrelated, it proves current prediction benchmarks
    measure the wrong thing.
    """
    conditions = get_ordered_conditions(results)
    conditions = [c for c in conditions if c not in ("real_data", "random", "grnboost2_real")]

    fig, ax = plt.subplots(figsize=(7, 6))

    for c in conditions:
        cond = results["conditions"][c]
        ws = cond.get("wasserstein", {})
        ws_val = ws.get("mean_wasserstein", 0) if isinstance(ws, dict) else ws
        auprc_val = cond["auprc"]

        color = COLORS.get(c, "#7f8c8d")
        name = DISPLAY_NAMES.get(c, c).replace("\n", " ")
        ax.scatter(ws_val, auprc_val, s=150, c=color, edgecolors="black",
                   linewidths=0.5, zorder=3, label=name)
        ax.annotate(name, (ws_val, auprc_val), textcoords="offset points",
                    xytext=(8, 8), fontsize=8)

    # Add random floor
    rf = results.get("random_floor_auprc", 0)
    ax.axhline(y=rf, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.5,
               label=f"Random floor ({rf:.3f})")

    ax.set_xlabel("Mean Wasserstein Distance\n(prediction accuracy: lower = better)")
    ax.set_ylabel("AUPRC\n(causal graph quality: higher = better)")
    ax.set_title("Figure 5: Prediction Accuracy vs Causal Graph Quality\n"
                 "Dissociation = current benchmarks measure the wrong thing")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "figure5_prediction_vs_causal.png")
    fig.savefig(outdir / "figure5_prediction_vs_causal.pdf")
    plt.close(fig)
    print(f"  Saved figure5_prediction_vs_causal.png/pdf")


def figure6_edge_overlap(results, outdir):
    """Figure 6: Edge count and true positive overlap."""
    conditions = get_ordered_conditions(results)
    conditions = [c for c in conditions if c != "real_data"]
    names = [DISPLAY_NAMES.get(c, c).replace("\n", " ") for c in conditions]

    n_edges = [results["conditions"][c]["n_edges"] for c in conditions]
    n_tp = [results["conditions"][c].get("n_true_positive", 0) for c in conditions]
    gt_edges = results.get("n_gt_edges", results["conditions"]["real_data"]["n_edges"])

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, n_edges, width, label="Total predicted edges", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, n_tp, width, label="True positive edges", color="#2ecc71", alpha=0.8)
    ax.axhline(y=gt_edges, color="#e74c3c", linestyle="--", linewidth=1, label=f"Ground truth edges ({gt_edges})")

    ax.set_xticks(x)
    ax.set_xticklabels(names, ha="center")
    ax.set_ylabel("Edge count")
    ax.set_title(f"Predicted vs True Positive Edges ({results['n_genes']} genes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "figure6_edge_overlap.png")
    fig.savefig(outdir / "figure6_edge_overlap.pdf")
    plt.close(fig)
    print(f"  Saved figure6_edge_overlap.png/pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to eval_results JSON")
    parser.add_argument("--outdir", default="figures", help="Output directory")
    args = parser.parse_args()

    results = load_results(args.results)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    print(f"Generating figures from {args.results}")
    print(f"  {results['n_genes']} genes, {len(results['conditions'])} conditions")
    print()

    figure1_auprc_bar(results, outdir)
    figure2_metrics_heatmap(results, outdir)
    figure3_fraction_ceiling(results, outdir)
    figure4_shd_breakdown(results, outdir)
    figure5_prediction_vs_causal(results, outdir)
    figure6_edge_overlap(results, outdir)

    print()
    print("All figures generated.")


if __name__ == "__main__":
    main()
