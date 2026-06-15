"""Generate CausalCellBench publication figures (Figures 5-8) for the n=50 evaluation.

Reads eval_results_n50.json and produces four publication-quality figures:
  - Figure 5: AUPRC horizontal bar chart
  - Figure 6: Multi-metric heatmap (normalized)
  - Figure 7: Edge count comparison (stacked TP + FP)
  - Figure 8: Precision@100 comparison

Usage:
    python3 scripts/generate_figures_n50.py
    python3 scripts/generate_figures_n50.py --results results/eval_results_n50.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Publication style -- Helvetica / Arial, clean axes, no grid
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "pdf.fonttype": 42,       # TrueType fonts in PDF (journal requirement)
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------------
# Condition ordering and display names (matches request)
# ---------------------------------------------------------------------------
CONDITION_ORDER = [
    "real_data",
    "elastic_net",
    "gears_real",
    "overlay_resampled_real",
    "grnboost2_real",
    "cpa_real",
    "geneformer_real",
    "random",
]

DISPLAY_NAMES = {
    "real_data": "Real Data (ceiling)",
    "overlay_resampled_real": "Overlay Real (UB)",
    "elastic_net": "ElasticNet",
    "cpa_real": "CPA",
    "gears_real": "GEARS",
    "geneformer_real": "Geneformer",
    "grnboost2_real": "GRNBoost2 (obs.)",
    "random": "Random (floor)",
}

# Color scheme:
# real=green, overlay=light green, ElasticNet=blue,
# models=warm shades, GRNBoost2=purple, random=gray
COLORS = {
    "real_data": "#27ae60",
    "overlay_resampled_real": "#82e0aa",
    "elastic_net": "#2980b9",
    "cpa_real": "#e67e22",
    "gears_real": "#d35400",
    "geneformer_real": "#e74c3c",
    "grnboost2_real": "#8e44ad",
    "random": "#95a5a6",
}


def load_results(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_ordered(results: dict) -> list[str]:
    """Return conditions in display order, filtering to those present."""
    return [c for c in CONDITION_ORDER if c in results["conditions"]]


def _extract_scalar(cond: dict, field: str) -> float:
    """Extract a scalar value from a condition dict, handling nested dicts."""
    val = cond.get(field, 0)
    if isinstance(val, dict):
        # known nested fields
        if "shd" in val:
            return float(val["shd"])
        if "mean_wasserstein" in val:
            return float(val["mean_wasserstein"])
        if "for_rate" in val:
            return float(val["for_rate"])
        return 0.0
    return float(val)


# ===================================================================
# Figure 5: AUPRC horizontal bar chart
# ===================================================================
def figure5_auprc_bars(results: dict, outdir: Path) -> None:
    conditions = get_ordered(results)
    names = [DISPLAY_NAMES[c] for c in conditions]
    auprcs = [results["conditions"][c]["auprc"] for c in conditions]
    colors = [COLORS[c] for c in conditions]

    # Reverse so highest (Real Data) is at top
    names = names[::-1]
    auprcs = auprcs[::-1]
    colors = colors[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = np.arange(len(conditions))
    bars = ax.barh(y_pos, auprcs, color=colors, edgecolor="white", linewidth=0.5,
                   height=0.7)

    # Random floor vertical dashed line
    rf = results.get("random_floor_auprc", 0)
    ax.axvline(x=rf, color="#555555", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Random floor ({rf:.3f})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("AUPRC")
    ax.set_title(
        f"CausalCellBench: AUPRC by Simulation Method (K562, {results['n_genes']} genes)"
    )
    ax.set_xlim(0, 1.08)

    # Value labels
    for bar, val in zip(bars, auprcs):
        ax.text(
            val + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            ha="left", va="center", fontsize=8,
        )

    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "fig5_auprc_bars.pdf")
    fig.savefig(outdir / "fig5_auprc_bars.png")
    plt.close(fig)
    print("  Saved fig5_auprc_bars.pdf / .png")


# ===================================================================
# Figure 6: Multi-metric heatmap
# ===================================================================
def figure6_metrics_heatmap(results: dict, outdir: Path) -> None:
    conditions = get_ordered(results)
    names = [DISPLAY_NAMES[c] for c in conditions]

    # Metrics to show
    metric_keys = ["auprc", "precision_at_100", "shd", "wasserstein", "false_omission"]
    metric_labels = ["AUPRC", "P@100", "SHD (inv.)", "Wasserstein", "FOR"]
    # higher_is_better flags -- used for normalization direction
    higher_is_better = [True, True, False, True, False]

    n_cond = len(conditions)
    n_met = len(metric_keys)
    raw = np.zeros((n_cond, n_met))

    for i, c in enumerate(conditions):
        cond = results["conditions"][c]
        raw[i, 0] = cond.get("auprc", 0)
        raw[i, 1] = cond.get("precision_at_100", 0)
        shd_val = cond.get("shd", {})
        raw[i, 2] = float(shd_val["shd"]) if isinstance(shd_val, dict) else float(shd_val)
        ws_val = cond.get("wasserstein", {})
        raw[i, 3] = float(ws_val["mean_wasserstein"]) if isinstance(ws_val, dict) else float(ws_val)
        for_val = cond.get("false_omission", {})
        raw[i, 4] = float(for_val["for_rate"]) if isinstance(for_val, dict) else float(for_val)

    # Normalize each column to [0, 1].  For columns where lower is better
    # (SHD, FOR) we invert so that green = good, red = bad.
    norm = np.zeros_like(raw)
    for j in range(n_met):
        col = raw[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            norm[:, j] = (col - cmin) / (cmax - cmin)
        else:
            norm[:, j] = 0.5
        if not higher_is_better[j]:
            norm[:, j] = 1.0 - norm[:, j]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(n_met))
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_yticks(range(n_cond))
    ax.set_yticklabels(names, fontsize=9)

    # Annotate cells with raw values
    for i in range(n_cond):
        for j in range(n_met):
            val = raw[i, j]
            txt = f"{val:.0f}" if val >= 10 else f"{val:.3f}"
            color = "white" if norm[i, j] < 0.3 or norm[i, j] > 0.7 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=color)

    ax.set_title(
        f"CausalCellBench: Multi-Metric Comparison ({results['n_genes']} genes)"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Normalized score (1 = best)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(outdir / "fig6_metrics_heatmap.pdf")
    fig.savefig(outdir / "fig6_metrics_heatmap.png")
    plt.close(fig)
    print("  Saved fig6_metrics_heatmap.pdf / .png")


# ===================================================================
# Figure 7: Edge count comparison (stacked TP / FP)
# ===================================================================
def figure7_edge_counts(results: dict, outdir: Path) -> None:
    conditions = get_ordered(results)
    # Remove real_data -- it is the ground truth, shown as horizontal line
    conditions_plot = [c for c in conditions if c != "real_data"]
    names = [DISPLAY_NAMES[c] for c in conditions_plot]
    colors = [COLORS[c] for c in conditions_plot]

    tp_vals = []
    fp_vals = []
    for c in conditions_plot:
        cond = results["conditions"][c]
        n_tp = cond.get("n_true_positive", 0)
        n_edges = cond.get("n_edges", 0)
        tp_vals.append(n_tp)
        fp_vals.append(n_edges - n_tp)

    gt_edges = results.get("n_gt_edges", results["conditions"]["real_data"]["n_edges"])

    x = np.arange(len(conditions_plot))
    width = 0.55

    fig, ax = plt.subplots(figsize=(7, 4.5))
    p1 = ax.bar(x, tp_vals, width, label="True positives", color="#27ae60", alpha=0.9)
    p2 = ax.bar(x, fp_vals, width, bottom=tp_vals, label="False positives",
                color="#e74c3c", alpha=0.7)

    ax.axhline(y=gt_edges, color="#2c3e50", linestyle="--", linewidth=1.2,
               label=f"Ground truth ({gt_edges} edges)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Number of edges")
    ax.set_title(
        f"Edge Count Comparison ({results['n_genes']} genes)"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    # Label total on top of each stacked bar
    for i, (tp, fp) in enumerate(zip(tp_vals, fp_vals)):
        total = tp + fp
        ax.text(i, total + 5, str(total), ha="center", va="bottom", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(outdir / "fig7_edge_counts.pdf")
    fig.savefig(outdir / "fig7_edge_counts.png")
    plt.close(fig)
    print("  Saved fig7_edge_counts.pdf / .png")


# ===================================================================
# Figure 8: Precision@100 comparison
# ===================================================================
def figure8_precision_at_k(results: dict, outdir: Path) -> None:
    conditions = get_ordered(results)
    names = [DISPLAY_NAMES[c] for c in conditions]
    p100 = [results["conditions"][c].get("precision_at_100", 0) for c in conditions]
    colors = [COLORS[c] for c in conditions]

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(range(len(conditions)), p100, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.65)

    # Random P@100 as floor line
    rand_p100 = results["conditions"].get("random", {}).get("precision_at_100", 0)
    if rand_p100 > 0:
        ax.axhline(y=rand_p100, color="#555555", linestyle="--", linewidth=1,
                   alpha=0.7, label=f"Random ({rand_p100:.2f})")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Precision @ 100")
    ax.set_title(
        f"CausalCellBench: Precision@100 ({results['n_genes']} genes)"
    )
    ax.set_ylim(0, 1.1)

    for bar, val in zip(bars, p100):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=8,
        )

    if rand_p100 > 0:
        ax.legend(loc="upper right", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "fig8_precision_at_k.pdf")
    fig.savefig(outdir / "fig8_precision_at_k.png")
    plt.close(fig)
    print("  Saved fig8_precision_at_k.pdf / .png")


# ===================================================================
# Figure 9: Fraction of real-data ceiling recovered
# ===================================================================
def figure9_fraction_ceiling(results: dict, outdir: Path) -> None:
    # Exclude real_data (it's the ceiling = 100%) and random
    conditions = [c for c in get_ordered(results) if c not in ("real_data", "random")]
    names = [DISPLAY_NAMES[c] for c in conditions]
    colors = [COLORS[c] for c in conditions]

    ceiling_auprc = results["conditions"]["real_data"]["auprc"]  # 1.0
    random_auprc = results.get("random_floor_auprc", 0.093)

    # Fraction = (model - random_floor) / (ceiling - random_floor)
    fractions = []
    for c in conditions:
        model_auprc = results["conditions"][c]["auprc"]
        frac = (model_auprc - random_auprc) / (ceiling_auprc - random_auprc)
        fractions.append(frac)

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = np.arange(len(conditions))
    bars = ax.barh(y_pos[::-1], fractions, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.7)

    ax.set_yticks(y_pos[::-1])
    ax.set_yticklabels(names)
    ax.set_xlabel("Fraction of Real-Data Ceiling Recovered")
    ax.set_title(
        f"CausalCellBench: Model Performance vs. Real Data ({results['n_genes']} genes)"
    )
    ax.set_xlim(0, 0.45)

    # Value labels
    for bar, val in zip(bars, fractions):
        ax.text(
            val + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            ha="left", va="center", fontsize=8,
        )

    # Annotation
    ax.text(0.98, 0.02, f"Ceiling: Real Data AUPRC = {ceiling_auprc:.3f}\n"
            f"Floor: Random AUPRC = {random_auprc:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8))

    fig.tight_layout()
    fig.savefig(outdir / "fig9_fraction_ceiling.pdf")
    fig.savefig(outdir / "fig9_fraction_ceiling.png")
    plt.close(fig)
    print("  Saved fig9_fraction_ceiling.pdf / .png")


# ===================================================================
# Figure 10: Model radar chart (multi-metric comparison)
# ===================================================================
def figure10_radar(results: dict, outdir: Path) -> None:
    # Compare only the 3 real models + ElasticNet + GRNBoost2
    model_keys = ["elastic_net", "gears_real", "cpa_real", "geneformer_real", "grnboost2_real"]
    model_keys = [k for k in model_keys if k in results["conditions"]]
    model_names = [DISPLAY_NAMES[k] for k in model_keys]
    model_colors = [COLORS[k] for k in model_keys]

    # Metrics (all normalized 0-1, higher=better)
    metrics = ["AUPRC", "P@100", "Precision", "Low SHD", "Low FOR"]
    n_metrics = len(metrics)

    # Compute normalized values
    values = []
    for k in model_keys:
        c = results["conditions"][k]
        n_edges = c["n_edges"]
        n_tp = c["n_true_positive"]
        precision = n_tp / n_edges if n_edges > 0 else 0

        shd_val = c["shd"]["shd"] if isinstance(c["shd"], dict) else c["shd"]
        for_val = c["false_omission"]["for_rate"] if isinstance(c["false_omission"], dict) else c["false_omission"]

        values.append([
            c["auprc"],
            c["precision_at_100"],
            precision,
            1.0 - (shd_val / 400),  # normalize: 0 SHD = 1.0, 400 SHD = 0.0
            1.0 - for_val,           # low FOR = good
        ])

    # Radar plot
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for i, (name, vals, color) in enumerate(zip(model_names, values, model_colors)):
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, 'o-', linewidth=1.5, color=color, label=name,
                markersize=4)
        ax.fill(angles, vals_plot, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title(f"Model Comparison Radar ({results['n_genes']} genes)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "fig10_radar.pdf")
    fig.savefig(outdir / "fig10_radar.png")
    plt.close(fig)
    print("  Saved fig10_radar.pdf / .png")


# ===================================================================
# Figure 11: AUPRC gap analysis (waterfall-style)
# ===================================================================
def figure11_gap_analysis(results: dict, outdir: Path) -> None:
    """Shows the AUPRC gap between each method and real data."""
    conditions = [c for c in get_ordered(results) if c != "real_data"]
    names = [DISPLAY_NAMES[c] for c in conditions]
    colors = [COLORS[c] for c in conditions]

    ceiling = results["conditions"]["real_data"]["auprc"]
    gaps = [ceiling - results["conditions"][c]["auprc"] for c in conditions]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(conditions))
    bars = ax.bar(x, gaps, color=colors, edgecolor="white", linewidth=0.5, width=0.65)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUPRC Gap from Real Data")
    ax.set_title(f"Gap to Real-Data Ceiling ({results['n_genes']} genes)")

    for bar, val in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, max(gaps) * 1.15)
    fig.tight_layout()
    fig.savefig(outdir / "fig11_gap_analysis.pdf")
    fig.savefig(outdir / "fig11_gap_analysis.png")
    plt.close(fig)
    print("  Saved fig11_gap_analysis.pdf / .png")


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figures 5-8 for CausalCellBench (n=50)")
    parser.add_argument(
        "--results",
        default="results/eval_results_n50.json",
        help="Path to eval_results JSON",
    )
    parser.add_argument("--outdir", default="figures", help="Output directory")
    args = parser.parse_args()

    results = load_results(args.results)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_cond = len(results["conditions"])
    print(f"Generating figures 5-8 from {args.results}")
    print(f"  {results['n_genes']} genes, {n_cond} conditions")
    print()

    figure5_auprc_bars(results, outdir)
    figure6_metrics_heatmap(results, outdir)
    figure7_edge_counts(results, outdir)
    figure8_precision_at_k(results, outdir)
    figure9_fraction_ceiling(results, outdir)
    figure10_radar(results, outdir)
    figure11_gap_analysis(results, outdir)

    print()
    print("All figures (5-11) generated.")


if __name__ == "__main__":
    main()
