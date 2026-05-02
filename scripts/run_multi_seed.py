"""Run CausalCellBench evaluation across multiple random seeds and aggregate results.

Each seed changes gene subset selection, control cell subsampling, and GIES
partition order. Results are aggregated with mean, std, 95% CI (t-distribution),
and pairwise permutation tests for condition comparisons.

Usage:
    # Run all seeds sequentially
    python3 scripts/run_multi_seed.py --n-genes 200 --seeds 42 123 456 789 1024

    # Run a single seed (for parallel execution in separate terminals)
    python3 scripts/run_multi_seed.py --n-genes 200 --seeds 42

    # Aggregate existing results without re-running
    python3 scripts/run_multi_seed.py --n-genes 200 --seeds 42 123 456 789 1024 --aggregate-only

    # Pass extra flags through to run_eval_local.py
    python3 scripts/run_multi_seed.py --n-genes 200 --seeds 42 123 --extra-args="--skip-random --skip-stubs"
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [multi_seed] %(message)s",
)
logger = logging.getLogger("multi_seed")

# Project root is two levels up from this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "run_eval_local.py"
RESULTS_DIR = PROJECT_ROOT / "results"
MULTI_SEED_DIR = RESULTS_DIR / "multi_seed"

DEFAULT_SEEDS: list[int] = [42, 123, 456, 789, 1024]

# Metrics to aggregate (flat scalars extracted from the per-condition results)
SCALAR_METRICS = ["auprc", "precision_at_100", "n_edges", "n_true_positive"]
SHD_METRICS = ["shd", "missing", "extra", "reversed"]
DERIVED_METRICS = ["precision", "recall", "f1"]
ALL_METRICS = SCALAR_METRICS + SHD_METRICS + DERIVED_METRICS


# =====================================================================
# Per-seed file management
# =====================================================================

def seed_result_path(n_genes: int, seed: int) -> Path:
    """Path where per-seed results are stored."""
    return MULTI_SEED_DIR / f"eval_results_n{n_genes}_s{seed}.json"


def canonical_result_path(n_genes: int) -> Path:
    """Path the eval script writes to (no seed in name)."""
    return RESULTS_DIR / f"eval_results_n{n_genes}.json"


# =====================================================================
# Run evaluation for a single seed
# =====================================================================

def run_single_seed(
    n_genes: int,
    seed: int,
    extra_args: list[str] | None = None,
    dry_run: bool = False,
) -> Path:
    """Run run_eval_local.py for a single seed and archive the result.

    Parameters
    ----------
    n_genes : Number of genes.
    seed : Random seed.
    extra_args : Additional CLI args forwarded to the eval script.
    dry_run : If True, print the command but don't execute.

    Returns
    -------
    Path to the per-seed result JSON.
    """
    out_path = seed_result_path(n_genes, seed)
    if out_path.exists():
        logger.info("Seed %d already has results at %s -- skipping", seed, out_path)
        return out_path

    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--n-genes", str(n_genes),
        "--seed", str(seed),
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Running seed %d: %s", seed, " ".join(cmd))

    if dry_run:
        logger.info("[dry-run] Would execute: %s", " ".join(cmd))
        return out_path

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False)
    if result.returncode != 0:
        logger.error("Seed %d failed with return code %d", seed, result.returncode)
        raise RuntimeError(f"Eval script failed for seed {seed} (rc={result.returncode})")

    # Move the canonical output to the per-seed location
    canonical = canonical_result_path(n_genes)
    if not canonical.exists():
        raise FileNotFoundError(
            f"Expected eval output at {canonical} after running seed {seed}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(canonical), str(out_path))
    logger.info("Seed %d results archived to %s", seed, out_path)

    return out_path


# =====================================================================
# Parse per-seed results
# =====================================================================

def load_seed_result(path: Path) -> dict[str, Any]:
    """Load and validate a per-seed result JSON."""
    with open(path) as f:
        data = json.load(f)
    if "conditions" not in data:
        raise ValueError(f"No 'conditions' key in {path}")
    return data


def extract_condition_metrics(
    cond: dict[str, Any],
    n_gt_edges: int,
) -> dict[str, float]:
    """Extract flat metric dict from a single condition result.

    Derives precision, recall, F1 from n_true_positive / n_edges / n_gt_edges.
    """
    n_edges = cond.get("n_edges", 0)
    tp = cond.get("n_true_positive", 0)

    precision = tp / n_edges if n_edges > 0 else 0.0
    recall = tp / n_gt_edges if n_gt_edges > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    shd_dict = cond.get("shd", {})

    return {
        "auprc": cond.get("auprc", 0.0),
        "precision_at_100": cond.get("precision_at_100", 0.0),
        "n_edges": float(n_edges),
        "n_true_positive": float(tp),
        "shd": float(shd_dict.get("shd", 0)),
        "missing": float(shd_dict.get("missing", 0)),
        "extra": float(shd_dict.get("extra", 0)),
        "reversed": float(shd_dict.get("reversed", 0)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =====================================================================
# Aggregation statistics
# =====================================================================

def compute_ci_t(
    values: np.ndarray,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Compute mean, std, and confidence interval using t-distribution.

    Parameters
    ----------
    values : 1-D array of metric values across seeds.
    confidence : Confidence level (default 0.95 for 95% CI).

    Returns
    -------
    Dict with mean, std, ci_low, ci_high, n.
    """
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0

    if n <= 1:
        return {"mean": mean, "std": std, "ci_low": mean, "ci_high": mean, "n": n}

    se = std / np.sqrt(n)
    t_crit = sp_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * se

    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
        "n": n,
    }


def paired_permutation_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_permutations: int = 10000,
    rng_seed: int = 0,
) -> float:
    """Two-sided paired permutation test.

    Tests H0: mean(A) == mean(B) using the paired differences.

    Parameters
    ----------
    values_a, values_b : Metric values for two conditions across seeds.
    n_permutations : Number of random sign-flips.
    rng_seed : Seed for reproducibility.

    Returns
    -------
    Two-sided p-value.
    """
    diffs = values_a - values_b
    observed = np.abs(np.mean(diffs))
    n = len(diffs)

    rng = np.random.default_rng(rng_seed)
    count_extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        perm_stat = np.abs(np.mean(diffs * signs))
        if perm_stat >= observed:
            count_extreme += 1

    return (count_extreme + 1) / (n_permutations + 1)


# =====================================================================
# Main aggregation
# =====================================================================

def aggregate_results(
    n_genes: int,
    seeds: list[int],
    n_permutations: int = 10000,
) -> dict[str, Any]:
    """Load all per-seed results, compute aggregate stats and pairwise tests.

    Parameters
    ----------
    n_genes : Number of genes used in the evaluation.
    seeds : List of seeds to aggregate.
    n_permutations : Permutations for pairwise tests.

    Returns
    -------
    Dict with per_seed, aggregated, and pairwise_comparisons sections.
    """
    # ------------------------------------------------------------------
    # Load per-seed data
    # ------------------------------------------------------------------
    per_seed: dict[int, dict[str, Any]] = {}
    missing_seeds: list[int] = []

    for seed in seeds:
        path = seed_result_path(n_genes, seed)
        if not path.exists():
            logger.warning("Missing results for seed %d at %s", seed, path)
            missing_seeds.append(seed)
            continue
        per_seed[seed] = load_seed_result(path)

    if missing_seeds:
        logger.warning(
            "Missing %d/%d seeds: %s. Aggregating available seeds only.",
            len(missing_seeds), len(seeds), missing_seeds,
        )

    available_seeds = sorted(per_seed.keys())
    if len(available_seeds) == 0:
        raise RuntimeError("No per-seed results found. Run evaluation first.")

    logger.info("Aggregating %d seeds: %s", len(available_seeds), available_seeds)

    # ------------------------------------------------------------------
    # Collect metrics per condition per seed
    # ------------------------------------------------------------------
    # condition -> metric -> list of values (one per seed, aligned)
    all_conditions: set[str] = set()
    for data in per_seed.values():
        all_conditions.update(data["conditions"].keys())
    all_conditions_sorted = sorted(all_conditions)

    # Build: condition -> metric -> np.array(n_seeds)
    cond_metric_values: dict[str, dict[str, np.ndarray]] = {}

    for cond in all_conditions_sorted:
        metric_lists: dict[str, list[float]] = {m: [] for m in ALL_METRICS}
        seeds_with_cond: list[int] = []

        for seed in available_seeds:
            cond_data = per_seed[seed]["conditions"].get(cond)
            if cond_data is None:
                continue
            n_gt = per_seed[seed].get("n_gt_edges", 0)
            metrics = extract_condition_metrics(cond_data, n_gt)
            for m in ALL_METRICS:
                metric_lists[m].append(metrics[m])
            seeds_with_cond.append(seed)

        cond_metric_values[cond] = {
            m: np.array(metric_lists[m]) for m in ALL_METRICS
        }
        cond_metric_values[cond]["_seeds"] = np.array(seeds_with_cond)  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for cond in all_conditions_sorted:
        aggregated[cond] = {}
        for metric in ALL_METRICS:
            vals = cond_metric_values[cond][metric]
            aggregated[cond][metric] = compute_ci_t(vals)

    # ------------------------------------------------------------------
    # Pairwise permutation tests
    # ------------------------------------------------------------------
    comparison_metrics = ["auprc", "f1", "precision_at_100", "shd"]
    pairwise_rows: list[dict[str, Any]] = []

    # Only compare non-trivial conditions (skip real_data upper bound)
    comparable = [c for c in all_conditions_sorted if c != "real_data"]

    for metric in comparison_metrics:
        for cond_a, cond_b in itertools.combinations(comparable, 2):
            vals_a = cond_metric_values[cond_a][metric]
            vals_b = cond_metric_values[cond_b][metric]
            # Only test if both have values for the same seeds
            seeds_a = set(cond_metric_values[cond_a]["_seeds"].tolist())
            seeds_b = set(cond_metric_values[cond_b]["_seeds"].tolist())
            common_seeds = sorted(seeds_a & seeds_b)

            if len(common_seeds) < 2:
                continue

            # Align to common seeds
            idx_a = [i for i, s in enumerate(cond_metric_values[cond_a]["_seeds"].tolist())
                     if s in common_seeds]
            idx_b = [i for i, s in enumerate(cond_metric_values[cond_b]["_seeds"].tolist())
                     if s in common_seeds]
            va = vals_a[idx_a]
            vb = vals_b[idx_b]

            p_val = paired_permutation_test(va, vb, n_permutations=n_permutations)

            pairwise_rows.append({
                "metric": metric,
                "condition_a": cond_a,
                "condition_b": cond_b,
                "mean_a": float(np.mean(va)),
                "mean_b": float(np.mean(vb)),
                "diff": float(np.mean(va) - np.mean(vb)),
                "p_value": p_val,
                "n_seeds": len(common_seeds),
                "significant_0.05": p_val < 0.05,
            })

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    per_seed_serializable: dict[str, Any] = {}
    for seed in available_seeds:
        raw = per_seed[seed]
        per_seed_entry: dict[str, Any] = {
            "seed": seed,
            "n_genes": raw.get("n_genes"),
            "n_gt_edges": raw.get("n_gt_edges"),
            "random_floor_auprc": raw.get("random_floor_auprc"),
            "conditions": {},
        }
        for cond, cond_data in raw["conditions"].items():
            n_gt = raw.get("n_gt_edges", 0)
            per_seed_entry["conditions"][cond] = extract_condition_metrics(cond_data, n_gt)
        per_seed_serializable[str(seed)] = per_seed_entry

    output = {
        "pipeline": "CausalCellBench Multi-Seed Aggregation",
        "n_genes": n_genes,
        "seeds": available_seeds,
        "n_seeds": len(available_seeds),
        "missing_seeds": missing_seeds,
        "per_seed": per_seed_serializable,
        "aggregated": aggregated,
        "pairwise_comparisons": pairwise_rows,
    }

    return output


# =====================================================================
# Output writing
# =====================================================================

def write_aggregated_json(output: dict[str, Any]) -> Path:
    """Write aggregated results to JSON."""
    MULTI_SEED_DIR.mkdir(parents=True, exist_ok=True)
    path = MULTI_SEED_DIR / "aggregated_results.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Aggregated results written to %s", path)
    return path


def write_pairwise_csv(pairwise: list[dict[str, Any]]) -> Path:
    """Write pairwise comparisons to CSV."""
    MULTI_SEED_DIR.mkdir(parents=True, exist_ok=True)
    path = MULTI_SEED_DIR / "pairwise_comparisons.csv"

    if not pairwise:
        logger.warning("No pairwise comparisons to write.")
        path.write_text("metric,condition_a,condition_b,mean_a,mean_b,diff,p_value,n_seeds,significant_0.05\n")
        return path

    fieldnames = list(pairwise[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairwise)

    logger.info("Pairwise comparisons written to %s", path)
    return path


def print_summary_table(output: dict[str, Any]) -> None:
    """Print a formatted summary table with CIs to stdout."""
    aggregated = output["aggregated"]
    n_seeds = output["n_seeds"]
    seeds = output["seeds"]

    print()
    print("=" * 110)
    print(f"CAUSALCELLBENCH MULTI-SEED RESULTS  (n_seeds={n_seeds}, seeds={seeds})")
    print("=" * 110)

    display_names: dict[str, str] = {
        "real_data": "Real Data (UB)",
        "overlay_resampled_real": "Overlay Real",
        "elastic_net": "ElasticNet",
        "cpa_real": "CPA (real)",
        "gears_real": "GEARS (real)",
        "scgpt_real": "scGPT (real)",
        "geneformer_real": "Geneformer (real)",
        "cpa": "CPA (stub)",
        "gears": "GEARS (stub)",
        "scgpt": "scGPT (stub)",
        "geneformer": "Geneformer (stub)",
        "grnboost2_real": "GRNBoost2 (obs)",
        "dcdi_real": "DCDI-G (soft)",
        "random": "Random (floor)",
    }

    display_order = [
        "real_data", "overlay_resampled_real", "elastic_net",
        "cpa_real", "gears_real", "scgpt_real", "geneformer_real",
        "grnboost2_real", "dcdi_real", "random",
    ]

    def fmt_ci(stats: dict[str, float], decimals: int = 4) -> str:
        """Format mean +/- CI as string."""
        m = stats["mean"]
        lo = stats["ci_low"]
        hi = stats["ci_high"]
        return f"{m:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"

    # Header
    header = (
        f"  {'Condition':<20} {'AUPRC (95% CI)':<30} {'F1 (95% CI)':<30} "
        f"{'P@100 (95% CI)':<30} {'SHD (95% CI)':<25}"
    )
    print(header)
    print(f"  {'-'*20} {'-'*30} {'-'*30} {'-'*30} {'-'*25}")

    # Rows in display order, then any remaining
    printed: set[str] = set()
    for key in display_order:
        if key not in aggregated:
            continue
        printed.add(key)
        cond = aggregated[key]
        name = display_names.get(key, key)
        print(
            f"  {name:<20} {fmt_ci(cond['auprc']):<30} {fmt_ci(cond['f1']):<30} "
            f"{fmt_ci(cond['precision_at_100']):<30} {fmt_ci(cond['shd'], 1):<25}"
        )

    # Any remaining conditions not in display_order
    for key in sorted(aggregated.keys()):
        if key in printed:
            continue
        cond = aggregated[key]
        name = display_names.get(key, key)
        print(
            f"  {name:<20} {fmt_ci(cond['auprc']):<30} {fmt_ci(cond['f1']):<30} "
            f"{fmt_ci(cond['precision_at_100']):<30} {fmt_ci(cond['shd'], 1):<25}"
        )

    # Precision / Recall detail
    print()
    print("  Precision and Recall:")
    print(f"  {'Condition':<20} {'Precision (95% CI)':<30} {'Recall (95% CI)':<30}")
    print(f"  {'-'*20} {'-'*30} {'-'*30}")
    for key in display_order:
        if key not in aggregated:
            continue
        cond = aggregated[key]
        name = display_names.get(key, key)
        print(
            f"  {name:<20} {fmt_ci(cond['precision']):<30} {fmt_ci(cond['recall']):<30}"
        )

    # Significant pairwise comparisons
    pairwise = output.get("pairwise_comparisons", [])
    sig = [r for r in pairwise if r["significant_0.05"]]
    if sig:
        print()
        print(f"  Significant pairwise comparisons (p < 0.05, {len(sig)} of {len(pairwise)}):")
        for r in sorted(sig, key=lambda x: x["p_value"]):
            direction = ">" if r["diff"] > 0 else "<"
            print(
                f"    {r['metric']:<18} {r['condition_a']:<25} {direction} "
                f"{r['condition_b']:<25} (diff={r['diff']:+.4f}, p={r['p_value']:.4f})"
            )
    else:
        print()
        print("  No significant pairwise differences found (p < 0.05).")

    print()
    print("=" * 110)
    print()


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run CausalCellBench across multiple seeds and aggregate results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n-genes", type=int, default=200,
        help="Number of genes to evaluate (passed through to eval script). Default: 200.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help=f"Random seeds to evaluate. Default: {DEFAULT_SEEDS}",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip evaluation runs; only aggregate existing per-seed results.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=10000,
        help="Number of permutations for pairwise tests. Default: 10000.",
    )
    parser.add_argument(
        "--extra-args", type=str, default="",
        help='Extra arguments to pass to run_eval_local.py (e.g., "--skip-random --skip-stubs").',
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run seeds even if results already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: run seeds (optionally) then aggregate."""
    args = parse_args()

    n_genes: int = args.n_genes
    seeds: list[int] = sorted(set(args.seeds))
    extra_args = args.extra_args.split() if args.extra_args.strip() else None

    logger.info("n_genes=%d, seeds=%s, aggregate_only=%s", n_genes, seeds, args.aggregate_only)

    # ------------------------------------------------------------------
    # Optionally clear existing per-seed results
    # ------------------------------------------------------------------
    if args.force and not args.aggregate_only:
        for seed in seeds:
            p = seed_result_path(n_genes, seed)
            if p.exists():
                p.unlink()
                logger.info("Removed existing result for seed %d", seed)

    # ------------------------------------------------------------------
    # Run evaluations
    # ------------------------------------------------------------------
    if not args.aggregate_only:
        for i, seed in enumerate(seeds):
            logger.info("--- Seed %d/%d: %d ---", i + 1, len(seeds), seed)
            try:
                run_single_seed(n_genes, seed, extra_args=extra_args, dry_run=args.dry_run)
            except (RuntimeError, FileNotFoundError) as e:
                logger.error("Failed on seed %d: %s", seed, e)
                logger.error("Continuing with remaining seeds.")
                continue

    if args.dry_run:
        logger.info("[dry-run] Skipping aggregation.")
        return

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    logger.info("Aggregating results...")
    output = aggregate_results(n_genes, seeds, n_permutations=args.n_permutations)

    # Write outputs
    write_aggregated_json(output)
    write_pairwise_csv(output["pairwise_comparisons"])

    # Print formatted table
    print_summary_table(output)

    logger.info("Done. Outputs in %s", MULTI_SEED_DIR)


if __name__ == "__main__":
    main()
