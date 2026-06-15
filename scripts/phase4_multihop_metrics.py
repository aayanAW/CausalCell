"""PHASE 4: Multi-hop causal evaluation.

Standard AUPRC penalizes models for predicting indirect causal effects
(A->C when ground truth is A->B->C). This script computes multi-hop
adjusted metrics to test if DL models capture longer-range pathways.
"""

import json
import logging
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path("/Users/aayanalwani/virtual cells")
GIES_CACHE = BASE / "results" / "gies_cache" / "n200_s42_p20"
OUTDIR = BASE / "results" / "phase4"
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_edges(name):
    """Load edges from GIES cache."""
    path = GIES_CACHE / f"{name}_edges.json"
    with open(path) as f:
        data = json.load(f)
    return set(tuple(e) for e in data["edges"])


def build_dag(edges):
    """Build a directed graph from edges."""
    G = nx.DiGraph()
    for src, tgt in edges:
        G.add_edge(src, tgt)
    return G


def compute_reachability(G, max_hops=5):
    """Compute all reachable pairs within k hops."""
    reachable = {}
    for k in range(1, max_hops + 1):
        reachable[k] = set()
        for node in G.nodes():
            # BFS from node, track distances
            descendants = nx.single_source_shortest_path_length(G, node, cutoff=k)
            for target, dist in descendants.items():
                if target != node and dist <= k:
                    reachable[k].add((node, target))
    return reachable


def multihop_evaluation(predicted_edges, gt_edges, reachable, max_hops=5):
    """Evaluate predicted edges with multi-hop credit.

    Classification:
    - direct_tp: edge is in ground truth (hop=1)
    - khop_tp: edge connects nodes reachable within k hops in GT (biologically correct, indirect)
    - true_fp: no causal path exists in GT between the predicted nodes
    """
    pred_set = set(predicted_edges)
    gt_set = set(gt_edges)

    results = {}

    for max_k in range(1, max_hops + 1):
        direct_tp = pred_set & gt_set
        indirect_tp = set()
        true_fp = set()

        for edge in pred_set:
            if edge in gt_set:
                continue  # already counted as direct TP
            if edge in reachable.get(max_k, set()):
                indirect_tp.add(edge)
            else:
                true_fp.add(edge)

        total_tp = len(direct_tp) + len(indirect_tp)
        total_pred = len(pred_set)
        total_gt_extended = len(reachable.get(max_k, set()))

        precision = total_tp / total_pred if total_pred > 0 else 0
        recall_direct = len(direct_tp) / len(gt_set) if gt_set else 0
        recall_extended = total_tp / total_gt_extended if total_gt_extended > 0 else 0

        results[max_k] = {
            "direct_tp": len(direct_tp),
            "indirect_tp": len(indirect_tp),
            "true_fp": len(true_fp),
            "total_tp": total_tp,
            "total_predicted": total_pred,
            "precision": float(precision),
            "recall_direct": float(recall_direct),
            "recall_extended": float(recall_extended),
            "gt_edges_direct": len(gt_set),
            "gt_edges_extended": total_gt_extended,
            "frac_indirect": float(len(indirect_tp) / total_tp) if total_tp > 0 else 0,
        }

    return results


def analyze_path_lengths(G, predicted_edges, gt_edges):
    """For each predicted edge, find the shortest path in the GT graph."""
    gt_set = set(gt_edges)
    path_lengths = []

    for src, tgt in predicted_edges:
        if (src, tgt) in gt_set:
            path_lengths.append({"src": src, "tgt": tgt, "path_length": 1, "type": "direct"})
        elif G.has_node(src) and G.has_node(tgt):
            try:
                length = nx.shortest_path_length(G, src, tgt)
                path_lengths.append({"src": src, "tgt": tgt, "path_length": length, "type": f"hop-{length}"})
            except nx.NetworkXNoPath:
                # Try reverse
                try:
                    length = nx.shortest_path_length(G, tgt, src)
                    path_lengths.append({"src": src, "tgt": tgt, "path_length": -length, "type": f"reversed-{length}"})
                except nx.NetworkXNoPath:
                    path_lengths.append({"src": src, "tgt": tgt, "path_length": 0, "type": "no_path"})
        else:
            path_lengths.append({"src": src, "tgt": tgt, "path_length": 0, "type": "node_missing"})

    return pd.DataFrame(path_lengths)


def main():
    logger.info("=" * 70)
    logger.info("PHASE 4: Multi-Hop Causal Evaluation")
    logger.info("=" * 70)

    # Load all edge sets
    gt_edges = load_edges("real_data")
    logger.info(f"Ground truth: {len(gt_edges)} edges")

    model_names = {
        "elastic_net": "ElasticNet",
        "gears_real": "GEARS",
        "cpa_real": "CPA",
        "geneformer_real": "Geneformer",
        "overlay_resampled_real": "Overlay Real",
    }

    model_edges = {}
    for key, display in model_names.items():
        try:
            model_edges[key] = load_edges(key)
            logger.info(f"  {display}: {len(model_edges[key])} edges")
        except FileNotFoundError:
            logger.warning(f"  {display}: not found")

    # Build GT graph and compute reachability
    logger.info("\nBuilding ground truth DAG...")
    G_gt = build_dag(gt_edges)
    logger.info(f"  Nodes: {G_gt.number_of_nodes()}, Edges: {G_gt.number_of_edges()}")
    logger.info(f"  Connected components: {nx.number_weakly_connected_components(G_gt)}")

    # Check for cycles
    if nx.is_directed_acyclic_graph(G_gt):
        logger.info("  Graph is a DAG")
        longest_path = nx.dag_longest_path_length(G_gt)
        logger.info(f"  Longest path: {longest_path} edges")
    else:
        logger.info(f"  Graph has cycles ({len(list(nx.simple_cycles(G_gt)))} cycles)")
        longest_path = "N/A (cyclic)"

    logger.info("\nComputing reachability (up to 5 hops)...")
    reachable = compute_reachability(G_gt, max_hops=5)
    for k in range(1, 6):
        logger.info(f"  {k}-hop reachable pairs: {len(reachable[k])}")

    # Multi-hop evaluation for each model
    logger.info("\n" + "=" * 70)
    logger.info("MULTI-HOP ADJUSTED METRICS")
    logger.info("=" * 70)

    all_results = {}
    for key, display in model_names.items():
        if key not in model_edges:
            continue
        logger.info(f"\n--- {display} ---")
        mh = multihop_evaluation(model_edges[key], gt_edges, reachable, max_hops=5)
        all_results[key] = mh

        for k in [1, 2, 3]:
            r = mh[k]
            logger.info(f"  k={k}: DirectTP={r['direct_tp']}, IndirectTP={r['indirect_tp']}, "
                        f"TrueFP={r['true_fp']}, Precision={r['precision']:.4f} "
                        f"(+{r['indirect_tp']} indirect)")

    # Summary comparison table
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON: Standard vs Multi-Hop Precision")
    logger.info("=" * 70)

    header = f"{'Model':<15} {'Std(k=1)':>10} {'k=2':>10} {'k=3':>10} {'k=2 boost':>10} {'k=3 boost':>10}"
    logger.info(header)
    logger.info("-" * 65)

    summary_rows = []
    for key, display in model_names.items():
        if key not in all_results:
            continue
        p1 = all_results[key][1]["precision"]
        p2 = all_results[key][2]["precision"]
        p3 = all_results[key][3]["precision"]
        boost2 = p2 - p1
        boost3 = p3 - p1
        logger.info(f"{display:<15} {p1:>10.4f} {p2:>10.4f} {p3:>10.4f} {boost2:>+10.4f} {boost3:>+10.4f}")
        summary_rows.append({
            "model": display, "key": key,
            "precision_k1": p1, "precision_k2": p2, "precision_k3": p3,
            "boost_k2": boost2, "boost_k3": boost3,
        })

    # Path length analysis for each model
    logger.info("\n" + "=" * 70)
    logger.info("PATH LENGTH DISTRIBUTION OF PREDICTED EDGES")
    logger.info("=" * 70)

    for key, display in model_names.items():
        if key not in model_edges:
            continue
        path_df = analyze_path_lengths(G_gt, model_edges[key], gt_edges)
        path_df.to_csv(OUTDIR / f"path_lengths_{key}.csv", index=False)

        type_counts = path_df["type"].value_counts()
        logger.info(f"\n  {display}:")
        for t, c in type_counts.items():
            logger.info(f"    {t}: {c} ({c/len(path_df):.1%})")

    # Save all results
    with open(OUTDIR / "multihop_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    pd.DataFrame(summary_rows).to_csv(OUTDIR / "multihop_summary.csv", index=False)

    logger.info(f"\nAll results saved to {OUTDIR}")

    # Key finding
    logger.info("\n" + "=" * 70)
    logger.info("KEY FINDING:")
    if summary_rows:
        # Which model benefits most from multi-hop credit?
        best_boost = max(summary_rows, key=lambda x: x["boost_k2"])
        logger.info(f"  {best_boost['model']} benefits most from 2-hop credit "
                     f"(+{best_boost['boost_k2']:.4f} precision)")

        # Does multi-hop change rankings?
        rank_k1 = sorted(summary_rows, key=lambda x: -x["precision_k1"])
        rank_k2 = sorted(summary_rows, key=lambda x: -x["precision_k2"])
        logger.info(f"  Standard ranking: {', '.join(r['model'] for r in rank_k1)}")
        logger.info(f"  2-hop ranking:    {', '.join(r['model'] for r in rank_k2)}")


if __name__ == "__main__":
    main()
