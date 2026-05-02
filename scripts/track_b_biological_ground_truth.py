#!/usr/bin/env python3
"""Track B: Evaluate causal graph predictions against biological ground truth.

Breaks the circularity of Track A (GIES-on-real vs GIES-on-simulated) by
evaluating all conditions against experimentally validated TF-target
interactions from TRRUST v2 (literature-curated, PMID-backed).

TRRUST v2: Han et al., Nucleic Acids Res 2018.
    - ~8,444 human TF-target regulatory interactions
    - Curated from PubMed abstracts via NLP + manual review
    - Format: TF  Target  Regulation(+/-)  PMID

Metrics computed:
    - Precision, Recall, F1 against TRRUST edges (Track B)
    - Precision, Recall, F1 against GIES-on-real edges (Track A)
    - Concordance between Track A and Track B condition rankings

Usage:
    python3 scripts/track_b_biological_ground_truth.py
    python3 scripts/track_b_biological_ground_truth.py --gies-dir results/gies_cache/n200_s42_p20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("track_b")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRRUST_URL = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"
TRRUST_CACHE_DIR = PROJECT_ROOT / "data" / "ground_truth"
TRRUST_CACHE_PATH = TRRUST_CACHE_DIR / "trrust_rawdata.human.tsv"

DEFAULT_GIES_DIR = PROJECT_ROOT / "results" / "gies_cache" / "n200_s42_p20"
DEFAULT_EVAL_RESULTS = PROJECT_ROOT / "results" / "eval_results_n200.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "track_b"

# Conditions to evaluate (filename stem -> display name)
CONDITIONS = {
    "real_data": "Real Data (GIES)",
    "elastic_net": "Elastic Net",
    "cpa_real": "CPA",
    "gears_real": "GEARS",
    "geneformer_real": "Geneformer",
    "overlay_resampled_real": "Overlay Resampled",
}


# ---------------------------------------------------------------------------
# TRRUST download and parsing
# ---------------------------------------------------------------------------


def download_trrust(
    url: str = TRRUST_URL,
    cache_path: Path = TRRUST_CACHE_PATH,
    max_retries: int = 3,
    timeout: int = 30,
) -> Path:
    """Download TRRUST v2 human TF-target interactions with retry logic.

    Parameters
    ----------
    url : str
        URL to the TRRUST raw data TSV.
    cache_path : Path
        Local path to cache the downloaded file.
    max_retries : int
        Number of download attempts before giving up.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    Path to the cached TSV file.

    Raises
    ------
    RuntimeError
        If download fails after all retries.
    """
    if cache_path.exists():
        logger.info("TRRUST data already cached at %s", cache_path)
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Downloading TRRUST v2 (attempt %d/%d) from %s",
                attempt, max_retries, url,
            )
            req = Request(url, headers={"User-Agent": "CausalCellBench/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()

            cache_path.write_bytes(data)
            n_lines = data.decode("utf-8", errors="replace").count("\n")
            logger.info("  Downloaded %d bytes (%d lines)", len(data), n_lines)
            return cache_path

        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            logger.warning("  Download attempt %d failed: %s", attempt, exc)
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to download TRRUST after {max_retries} attempts: {exc}"
                ) from exc
            time.sleep(2 ** attempt)  # exponential backoff

    # unreachable, but keeps mypy happy
    raise RuntimeError("Download loop exited unexpectedly")


def parse_trrust(
    tsv_path: Path,
    gene_subset: Optional[set[str]] = None,
) -> tuple[set[tuple[str, str]], dict[str, str]]:
    """Parse TRRUST v2 TSV into a set of (TF, Target) edges.

    Parameters
    ----------
    tsv_path : Path
        Path to the TRRUST raw data TSV.
    gene_subset : set of str, optional
        If provided, only keep edges where BOTH TF and target are in this set.

    Returns
    -------
    edges : set of (TF, Target) tuples
    regulation_types : dict mapping "TF->Target" to regulation type (+/-/?)
    """
    edges: set[tuple[str, str]] = set()
    regulation_types: dict[str, str] = {}

    text = tsv_path.read_text(encoding="utf-8", errors="replace")
    n_total = 0
    n_filtered = 0

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 3:
            continue

        tf, target, regulation = parts[0], parts[1], parts[2]
        n_total += 1

        if gene_subset is not None:
            if tf not in gene_subset or target not in gene_subset:
                n_filtered += 1
                continue

        edge = (tf, target)
        edges.add(edge)
        regulation_types[f"{tf}->{target}"] = regulation

    logger.info(
        "TRRUST: %d total interactions, %d after filtering to gene subset (%d dropped)",
        n_total, len(edges), n_filtered,
    )
    return edges, regulation_types


# ---------------------------------------------------------------------------
# Edge loading
# ---------------------------------------------------------------------------


def load_gies_edges(json_path: Path) -> set[tuple[str, str]]:
    """Load GIES edge list from a cached JSON file.

    Parameters
    ----------
    json_path : Path
        Path to JSON file with format {"edges": [["GENE1", "GENE2"], ...], ...}.

    Returns
    -------
    Set of (source, target) edge tuples.
    """
    with open(json_path) as f:
        data = json.load(f)

    edges_raw = data.get("edges", data) if isinstance(data, dict) else data
    return {(e[0], e[1]) for e in edges_raw}


def load_gene_subset(eval_results_path: Path) -> list[str]:
    """Load gene subset from the evaluation results JSON.

    Parameters
    ----------
    eval_results_path : Path
        Path to eval_results_n200.json (or similar).

    Returns
    -------
    Sorted list of gene names.
    """
    with open(eval_results_path) as f:
        data = json.load(f)
    genes = data["gene_subset"]
    logger.info("Loaded gene subset: %d genes from %s", len(genes), eval_results_path.name)
    return genes


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_precision_recall_f1(
    predicted: set[tuple[str, str]],
    ground_truth: set[tuple[str, str]],
) -> dict[str, float]:
    """Compute precision, recall, and F1 for edge prediction.

    For undirected comparison, we consider edge (A, B) equivalent to (B, A).
    TRRUST edges are directed (TF -> target), but GIES edges from CPDAGs
    include undirected edges. We evaluate both directed and undirected.

    Parameters
    ----------
    predicted : set of (str, str) tuples
    ground_truth : set of (str, str) tuples

    Returns
    -------
    Dict with directed and undirected precision, recall, F1, and counts.
    """
    # --- Directed evaluation ---
    tp_directed = predicted & ground_truth
    n_tp_directed = len(tp_directed)
    precision_directed = n_tp_directed / len(predicted) if predicted else 0.0
    recall_directed = n_tp_directed / len(ground_truth) if ground_truth else 0.0
    f1_directed = (
        2 * precision_directed * recall_directed / (precision_directed + recall_directed)
        if (precision_directed + recall_directed) > 0
        else 0.0
    )

    # --- Undirected evaluation ---
    # Canonicalize edges: always (min, max) alphabetically
    def _undirected(edges: set[tuple[str, str]]) -> set[tuple[str, str]]:
        return {(min(a, b), max(a, b)) for a, b in edges}

    pred_undir = _undirected(predicted)
    gt_undir = _undirected(ground_truth)

    tp_undirected = pred_undir & gt_undir
    n_tp_undirected = len(tp_undirected)
    precision_undirected = n_tp_undirected / len(pred_undir) if pred_undir else 0.0
    recall_undirected = n_tp_undirected / len(gt_undir) if gt_undir else 0.0
    f1_undirected = (
        2 * precision_undirected * recall_undirected / (precision_undirected + recall_undirected)
        if (precision_undirected + recall_undirected) > 0
        else 0.0
    )

    return {
        "directed": {
            "precision": precision_directed,
            "recall": recall_directed,
            "f1": f1_directed,
            "n_true_positive": n_tp_directed,
            "n_predicted": len(predicted),
            "n_ground_truth": len(ground_truth),
        },
        "undirected": {
            "precision": precision_undirected,
            "recall": recall_undirected,
            "f1": f1_undirected,
            "n_true_positive": n_tp_undirected,
            "n_predicted": len(pred_undir),
            "n_ground_truth": len(gt_undir),
        },
    }


def compute_jaccard(
    set_a: set[tuple[str, str]],
    set_b: set[tuple[str, str]],
) -> float:
    """Compute Jaccard similarity between two edge sets.

    Parameters
    ----------
    set_a, set_b : sets of edge tuples

    Returns
    -------
    Jaccard index (intersection / union), or 0 if both empty.
    """
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def spearman_rank_correlation(ranks_a: list[float], ranks_b: list[float]) -> float:
    """Compute Spearman rank correlation without scipy dependency.

    Parameters
    ----------
    ranks_a, ranks_b : lists of numeric ranks (same length).

    Returns
    -------
    Spearman rho.
    """
    n = len(ranks_a)
    if n < 3:
        return float("nan")

    d_sq = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
    rho = 1.0 - (6 * d_sq) / (n * (n ** 2 - 1))
    return rho


def rank_conditions(
    metrics: dict[str, dict],
    metric_key: str = "f1",
    sub_key: str = "undirected",
) -> dict[str, int]:
    """Rank conditions by a given metric (1 = best).

    Parameters
    ----------
    metrics : dict mapping condition name -> metrics dict
    metric_key : which metric to rank by
    sub_key : "directed" or "undirected"

    Returns
    -------
    Dict mapping condition name -> rank (1-indexed).
    """
    items = [
        (name, m[sub_key][metric_key])
        for name, m in metrics.items()
        if sub_key in m and metric_key in m[sub_key]
    ]
    # Sort descending by metric value
    items.sort(key=lambda x: -x[1])
    return {name: rank + 1 for rank, (name, _) in enumerate(items)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track B: Evaluate GRN predictions against TRRUST v2 biological ground truth.",
    )
    parser.add_argument(
        "--gies-dir",
        type=str,
        default=str(DEFAULT_GIES_DIR),
        help="Directory containing *_edges.json GIES cache files.",
    )
    parser.add_argument(
        "--eval-results",
        type=str,
        default=str(DEFAULT_EVAL_RESULTS),
        help="Path to eval_results_n200.json (for gene subset).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory to save Track B results.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip TRRUST download (use cached file only).",
    )
    args = parser.parse_args()

    gies_dir = Path(args.gies_dir)
    eval_results_path = Path(args.eval_results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load gene subset
    # ------------------------------------------------------------------
    if not eval_results_path.exists():
        logger.error("Eval results not found at %s", eval_results_path)
        sys.exit(1)

    gene_subset = load_gene_subset(eval_results_path)
    gene_set = set(gene_subset)
    n_possible_edges = len(gene_subset) * (len(gene_subset) - 1)
    logger.info("Possible directed edges: %d", n_possible_edges)

    # ------------------------------------------------------------------
    # Step 2: Download and parse TRRUST v2
    # ------------------------------------------------------------------
    if args.skip_download and not TRRUST_CACHE_PATH.exists():
        logger.error(
            "--skip-download set but no cached TRRUST file at %s", TRRUST_CACHE_PATH,
        )
        sys.exit(1)

    trrust_path = download_trrust() if not args.skip_download else TRRUST_CACHE_PATH
    trrust_edges, trrust_regulation = parse_trrust(trrust_path, gene_subset=gene_set)

    if not trrust_edges:
        logger.warning(
            "No TRRUST edges found for the %d-gene subset. "
            "This gene subset may contain few known TFs.",
            len(gene_subset),
        )

    # Report TRRUST summary stats
    tfs_in_subset = {tf for tf, _ in trrust_edges}
    targets_in_subset = {tgt for _, tgt in trrust_edges}
    activation_count = sum(1 for v in trrust_regulation.values() if v == "Activation")
    repression_count = sum(1 for v in trrust_regulation.values() if v == "Repression")
    unknown_count = sum(1 for v in trrust_regulation.values() if v == "Unknown")

    logger.info("TRRUST edges in subset: %d", len(trrust_edges))
    logger.info("  TFs: %d, Targets: %d", len(tfs_in_subset), len(targets_in_subset))
    logger.info(
        "  Activation: %d, Repression: %d, Unknown: %d",
        activation_count, repression_count, unknown_count,
    )

    # ------------------------------------------------------------------
    # Step 3: Load GIES edges for all conditions
    # ------------------------------------------------------------------
    if not gies_dir.exists():
        logger.error("GIES cache directory not found: %s", gies_dir)
        sys.exit(1)

    condition_edges: dict[str, set[tuple[str, str]]] = {}
    for cond_stem, cond_display in CONDITIONS.items():
        json_path = gies_dir / f"{cond_stem}_edges.json"
        if not json_path.exists():
            logger.warning("  Missing edge file for %s: %s", cond_display, json_path)
            continue
        edges = load_gies_edges(json_path)
        condition_edges[cond_stem] = edges
        logger.info("  %s: %d edges loaded", cond_display, len(edges))

    if not condition_edges:
        logger.error("No condition edge files found in %s", gies_dir)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4: Evaluate each condition against both tracks
    # ------------------------------------------------------------------
    # Track A ground truth = GIES on real data
    track_a_gt = condition_edges.get("real_data", set())
    if not track_a_gt:
        logger.warning("No real_data edges — Track A comparison will be empty.")

    results: dict[str, dict] = {}

    logger.info("\n=== Evaluating conditions ===")
    for cond_stem, edges in condition_edges.items():
        cond_display = CONDITIONS.get(cond_stem, cond_stem)
        logger.info("Evaluating: %s (%d edges)", cond_display, len(edges))

        # Track B: vs TRRUST
        track_b_metrics = compute_precision_recall_f1(edges, trrust_edges)

        # Track A: vs GIES-on-real (skip self-comparison for real_data)
        if cond_stem == "real_data":
            track_a_metrics = {
                "directed": {
                    "precision": 1.0, "recall": 1.0, "f1": 1.0,
                    "n_true_positive": len(edges),
                    "n_predicted": len(edges),
                    "n_ground_truth": len(track_a_gt),
                },
                "undirected": {
                    "precision": 1.0, "recall": 1.0, "f1": 1.0,
                    "n_true_positive": len(edges),
                    "n_predicted": len(edges),
                    "n_ground_truth": len(track_a_gt),
                },
            }
        else:
            track_a_metrics = compute_precision_recall_f1(edges, track_a_gt)

        # Jaccard with TRRUST (undirected)
        jaccard_trrust = compute_jaccard(
            {(min(a, b), max(a, b)) for a, b in edges},
            {(min(a, b), max(a, b)) for a, b in trrust_edges},
        )

        results[cond_stem] = {
            "display_name": cond_display,
            "n_edges": len(edges),
            "track_b_vs_trrust": track_b_metrics,
            "track_a_vs_gies_real": track_a_metrics,
            "jaccard_with_trrust_undirected": jaccard_trrust,
        }

    # ------------------------------------------------------------------
    # Step 5: Compute concordance between Track A and Track B rankings
    # ------------------------------------------------------------------
    # Exclude real_data from ranking (it's the Track A GT, always rank 1)
    model_conditions = [c for c in results if c != "real_data"]

    track_a_ranking = rank_conditions(
        {c: results[c]["track_a_vs_gies_real"] for c in model_conditions},
        metric_key="f1",
        sub_key="undirected",
    )
    track_b_ranking = rank_conditions(
        {c: results[c]["track_b_vs_trrust"] for c in model_conditions},
        metric_key="f1",
        sub_key="undirected",
    )

    # Spearman correlation between Track A and Track B ranks
    shared_conds = sorted(set(track_a_ranking) & set(track_b_ranking))
    if len(shared_conds) >= 3:
        ranks_a = [track_a_ranking[c] for c in shared_conds]
        ranks_b = [track_b_ranking[c] for c in shared_conds]
        spearman_rho = spearman_rank_correlation(ranks_a, ranks_b)
    else:
        spearman_rho = float("nan")

    concordance = {
        "track_a_ranking_by_f1_undirected": track_a_ranking,
        "track_b_ranking_by_f1_undirected": track_b_ranking,
        "spearman_rho": spearman_rho,
        "n_conditions_compared": len(shared_conds),
        "conditions_compared": shared_conds,
    }

    # ------------------------------------------------------------------
    # Step 6: Build and save final report
    # ------------------------------------------------------------------
    report = {
        "track": "B",
        "description": (
            "Evaluation of GIES causal graph predictions against TRRUST v2 "
            "literature-curated TF-target interactions (biological ground truth)."
        ),
        "gene_subset": gene_subset,
        "n_genes": len(gene_subset),
        "trrust_summary": {
            "total_interactions_in_db": sum(
                1 for line in trrust_path.read_text().strip().split("\n")
                if line.strip() and not line.startswith("#")
            ),
            "edges_in_gene_subset": len(trrust_edges),
            "n_tfs_in_subset": len(tfs_in_subset),
            "n_targets_in_subset": len(targets_in_subset),
            "regulation_breakdown": {
                "activation": activation_count,
                "repression": repression_count,
                "unknown": unknown_count,
            },
        },
        "trrust_edges": sorted([list(e) for e in trrust_edges]),
        "conditions": results,
        "concordance": concordance,
    }

    output_path = output_dir / "track_b_results.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Results saved to %s", output_path)

    # ------------------------------------------------------------------
    # Print human-readable summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TRACK B: Biological Ground Truth Evaluation (TRRUST v2)")
    print("=" * 72)
    print(f"Gene subset: {len(gene_subset)} genes")
    print(f"TRRUST edges in subset: {len(trrust_edges)} "
          f"({len(tfs_in_subset)} TFs, {len(targets_in_subset)} targets)")
    print()

    # Table header
    header = f"{'Condition':<25} {'Edges':>6}  {'Track B (vs TRRUST)':^30}  {'Track A (vs GIES-real)':^30}"
    sub_header = f"{'':25} {'':>6}  {'P':>8} {'R':>8} {'F1':>8} {'TP':>5}  {'P':>8} {'R':>8} {'F1':>8} {'TP':>5}"
    print(header)
    print(sub_header)
    print("-" * len(sub_header))

    for cond_stem in ["real_data", "elastic_net", "cpa_real", "gears_real",
                       "geneformer_real", "overlay_resampled_real"]:
        if cond_stem not in results:
            continue
        r = results[cond_stem]
        tb = r["track_b_vs_trrust"]["undirected"]
        ta = r["track_a_vs_gies_real"]["undirected"]
        print(
            f"{r['display_name']:<25} {r['n_edges']:>6}  "
            f"{tb['precision']:>8.4f} {tb['recall']:>8.4f} {tb['f1']:>8.4f} {tb['n_true_positive']:>5}  "
            f"{ta['precision']:>8.4f} {ta['recall']:>8.4f} {ta['f1']:>8.4f} {ta['n_true_positive']:>5}"
        )

    print()
    print("Rankings (excluding real_data, by undirected F1):")
    print(f"  Track A: {track_a_ranking}")
    print(f"  Track B: {track_b_ranking}")
    print(f"  Spearman rho: {spearman_rho:.4f}" if not (spearman_rho != spearman_rho) else "  Spearman rho: N/A (too few conditions)")
    print()

    # Interpretation
    if len(trrust_edges) < 10:
        print("WARNING: Very few TRRUST edges overlap with the gene subset.")
        print("  This is expected if the subset contains few transcription factors.")
        print("  Consider using a larger/TF-enriched gene subset for Track B.")
    elif spearman_rho == spearman_rho:  # not NaN
        if spearman_rho > 0.8:
            print("STRONG concordance between Track A and Track B rankings.")
            print("  Track A (GIES-on-real) is a reasonable proxy for biological truth.")
        elif spearman_rho > 0.4:
            print("MODERATE concordance between Track A and Track B rankings.")
            print("  Track A captures some biological signal but has limitations.")
        else:
            print("WEAK concordance between Track A and Track B rankings.")
            print("  Track A may not reliably reflect biological ground truth.")


if __name__ == "__main__":
    main()
