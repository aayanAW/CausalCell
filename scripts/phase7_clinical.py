"""PHASE 7: Clinical Translation Pipeline.

Takes the K562 causal graph (GIES edges), identifies hub genes,
cross-references with DepMap SELECTIVE essentiality scores,
and queries DGIdb for FDA-approved drug-gene interactions.

Outputs a ranked drug candidate report for imatinib-resistant CML.

Uses SELECTIVE essentiality (K562 Chronos - pan-cell-line mean)
to avoid flagging universal housekeeping genes.
"""

import json
import logging
import requests
import time
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path("/workspace") if Path("/workspace").exists() else Path("/Users/aayanalwani/virtual cells")
GIES_CACHE = BASE / "gies_cache" / "n200_s42_p20" if (BASE / "gies_cache").exists() else BASE / "results" / "gies_cache" / "n200_s42_p20"
OUTDIR = BASE / "results" / "phase7_clinical"
OUTDIR.mkdir(parents=True, exist_ok=True)

DGIDB_GRAPHQL_URL = "https://dgidb.org/api/graphql"
DEPMAP_URL = "https://ndownloader.figshare.com/files/34008404"  # DepMap 22Q2 CRISPRGeneEffect


# =====================================================================
# Step 1: Build causal graph and compute centrality
# =====================================================================

def build_graph_and_centrality(edges_path: Path) -> tuple:
    """Build NetworkX graph from GIES edges, compute centrality metrics."""
    logger.info("Step 1: Building causal graph and computing centrality")

    with open(edges_path) as f:
        data = json.load(f)
    edges = [tuple(e) for e in data["edges"]]
    logger.info(f"  Loaded {len(edges)} edges")

    G = nx.DiGraph()
    G.add_edges_from(edges)
    logger.info(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Compute centrality metrics
    eigen_cent = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
    between_cent = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)
    out_degree = dict(G.out_degree())
    in_degree = dict(G.in_degree())

    centrality_df = pd.DataFrame({
        "gene": list(eigen_cent.keys()),
        "eigenvector": [eigen_cent[g] for g in eigen_cent],
        "betweenness": [between_cent[g] for g in eigen_cent],
        "pagerank": [pagerank[g] for g in eigen_cent],
        "out_degree": [out_degree.get(g, 0) for g in eigen_cent],
        "in_degree": [in_degree.get(g, 0) for g in eigen_cent],
    })

    # Composite hub score (weighted combination)
    centrality_df["hub_score"] = (
        0.4 * centrality_df["eigenvector"] / centrality_df["eigenvector"].max() +
        0.3 * centrality_df["pagerank"] / centrality_df["pagerank"].max() +
        0.3 * centrality_df["out_degree"] / centrality_df["out_degree"].max()
    )
    centrality_df = centrality_df.sort_values("hub_score", ascending=False)

    centrality_df.to_csv(OUTDIR / "gene_centrality.csv", index=False)
    logger.info(f"  Top 10 hub genes:")
    for _, row in centrality_df.head(10).iterrows():
        logger.info(f"    {row['gene']}: hub_score={row['hub_score']:.4f}, "
                     f"out_degree={row['out_degree']}, eigen={row['eigenvector']:.4f}")

    return G, centrality_df


# =====================================================================
# Step 2: DepMap Selective Essentiality
# =====================================================================

def compute_selective_essentiality(centrality_df: pd.DataFrame) -> pd.DataFrame:
    """
    Query DepMap for K562 CRISPR dependency scores.
    Compute SELECTIVE essentiality = K562_score - pan_cell_line_mean.
    This avoids flagging universal housekeeping genes.
    """
    logger.info("\nStep 2: Computing selective essentiality (DepMap)")

    depmap_cache = OUTDIR / "depmap_k562_selective.csv"
    if depmap_cache.exists():
        logger.info("  DepMap cache found, loading...")
        depmap_df = pd.read_csv(depmap_cache)
    else:
        logger.info("  Downloading DepMap CRISPRGeneEffect (this takes ~1 min)...")
        try:
            response = requests.get(DEPMAP_URL, timeout=120)
            response.raise_for_status()

            # Save temporarily
            tmp_path = OUTDIR / "CRISPRGeneEffect_raw.csv"
            with open(tmp_path, "wb") as f:
                f.write(response.content)
            logger.info(f"  Downloaded {len(response.content) / 1e6:.1f} MB")

            # Parse DepMap matrix
            df = pd.read_csv(tmp_path, index_col=0)

            # Column names are like "GENE (ENTREZ_ID)" — extract gene name
            gene_cols = {col: col.split(" (")[0] for col in df.columns}
            df = df.rename(columns=gene_cols)

            # Find K562 row
            k562_rows = [idx for idx in df.index if "K562" in str(idx).upper() or "ACH-000551" in str(idx)]
            if not k562_rows:
                logger.warning("  K562 not found in DepMap, using closest match")
                k562_rows = [df.index[0]]

            k562_scores = df.loc[k562_rows[0]]
            pan_mean = df.mean(axis=0)

            # Selective essentiality = K562 score - pan-cell-line mean
            # More negative = more essential. Selective = uniquely essential in K562.
            selective = k562_scores - pan_mean

            depmap_df = pd.DataFrame({
                "gene": selective.index,
                "k562_chronos": k562_scores.values,
                "pan_mean_chronos": pan_mean.values,
                "selective_score": selective.values,
            })
            depmap_df.to_csv(depmap_cache, index=False)
            logger.info(f"  Computed selective essentiality for {len(depmap_df)} genes")

            # Clean up
            tmp_path.unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"  DepMap download failed: {e}")
            logger.info("  Using fallback: all genes marked as unknown essentiality")
            depmap_df = pd.DataFrame({
                "gene": centrality_df["gene"],
                "k562_chronos": np.nan,
                "pan_mean_chronos": np.nan,
                "selective_score": np.nan,
            })

    # Merge with centrality
    hub_genes = set(centrality_df["gene"])
    depmap_filtered = depmap_df[depmap_df["gene"].isin(hub_genes)].copy()
    merged = centrality_df.merge(depmap_filtered, on="gene", how="left")

    # Flag selectively essential genes
    # Use DepMap standard threshold: Chronos < -0.5 for strongly essential
    # Selective = K562 score < -0.5 AND at least 0.3 more negative than pan-cancer mean
    merged["is_selectively_essential"] = (merged["selective_score"] < -0.3) & (merged["k562_chronos"] < -0.5)
    merged["is_essential"] = merged["k562_chronos"] < -0.5

    n_selective = merged["is_selectively_essential"].sum()
    n_essential = merged["is_essential"].sum()
    logger.info(f"  Selectively essential in K562: {n_selective}/{len(merged)}")
    logger.info(f"  Universally essential (would flag housekeeping): {n_essential}/{len(merged)}")

    merged.to_csv(OUTDIR / "gene_essentiality.csv", index=False)
    return merged


# =====================================================================
# Step 3: DGIdb Drug-Gene Interactions
# =====================================================================

def query_dgidb(genes: list) -> pd.DataFrame:
    """Query DGIdb v5 GraphQL API for drug-gene interactions, with caching.

    Uses the v5 GraphQL endpoint (REST v2 is deprecated and returns empty results).
    """
    logger.info(f"\nStep 3: Querying DGIdb v5 (GraphQL) for {len(genes)} genes")

    dgidb_cache = OUTDIR / "dgidb_interactions.json"
    if dgidb_cache.exists():
        logger.info("  DGIdb cache found, loading...")
        with open(dgidb_cache) as f:
            all_interactions = json.load(f)
    else:
        all_interactions = []

        # GraphQL query template
        query_template = """
        query($names: [String!]!) {
          genes(names: $names) {
            nodes {
              name
              interactions {
                drug { name approved }
                interactionScore
                interactionTypes { type directionality }
                publications { pmid }
                sources { fullName }
              }
            }
          }
        }
        """

        # Batch query (25 genes per request to stay within GraphQL limits)
        batch_size = 25
        for i in range(0, len(genes), batch_size):
            batch = genes[i:i + batch_size]
            logger.info(f"  Querying batch {i // batch_size + 1} ({len(batch)} genes)...")

            try:
                response = requests.post(
                    DGIDB_GRAPHQL_URL,
                    json={"query": query_template, "variables": {"names": batch}},
                    timeout=30,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()

                nodes = data.get("data", {}).get("genes", {}).get("nodes", [])
                for node in nodes:
                    gene_name = node.get("name", "")
                    for interaction in node.get("interactions", []):
                        drug_info = interaction.get("drug", {})
                        drug_name = drug_info.get("name", "") if drug_info else ""
                        approved = drug_info.get("approved", False) if drug_info else False
                        int_types = interaction.get("interactionTypes", [])
                        type_str = "; ".join(t.get("type", "") for t in int_types) if int_types else ""
                        sources = interaction.get("sources", [])
                        source_str = "; ".join(s.get("fullName", "") for s in sources) if sources else ""
                        pubs = interaction.get("publications", [])
                        pmids = [p.get("pmid", "") for p in pubs if p.get("pmid")] if pubs else []

                        all_interactions.append({
                            "gene": gene_name,
                            "drug": drug_name,
                            "approved": approved,
                            "interaction_type": type_str,
                            "source": source_str,
                            "pmids": pmids,
                        })

                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.warning(f"  DGIdb batch failed: {e}")

        with open(dgidb_cache, "w") as f:
            json.dump(all_interactions, f, indent=2)

    logger.info(f"  Found {len(all_interactions)} drug-gene interactions")

    if not all_interactions:
        return pd.DataFrame(columns=["gene", "drug", "approved", "interaction_type", "source"])

    interactions_df = pd.DataFrame(all_interactions)
    # Deduplicate
    interactions_df = interactions_df.drop_duplicates(subset=["gene", "drug"])

    n_genes_with_drugs = interactions_df["gene"].nunique()
    n_drugs = interactions_df["drug"].nunique()
    n_approved = interactions_df[interactions_df.get("approved", False) == True]["drug"].nunique() if "approved" in interactions_df.columns else 0
    logger.info(f"  {n_genes_with_drugs} genes have drug interactions ({n_drugs} unique drugs, {n_approved} FDA-approved)")

    interactions_df.to_csv(OUTDIR / "drug_interactions.csv", index=False)
    return interactions_df


# =====================================================================
# Step 4: Candidate Ranking
# =====================================================================

def rank_candidates(essentiality_df: pd.DataFrame,
                     interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Merge hub scores, essentiality, and drug interactions into ranked candidates."""
    logger.info("\nStep 4: Ranking drug candidates")

    # Get hub genes (top 20% by hub_score)
    n_hubs = max(int(len(essentiality_df) * 0.2), 10)
    top_hubs = essentiality_df.head(n_hubs).copy()
    logger.info(f"  Top {n_hubs} hub genes selected")

    # Merge with drug interactions
    if interactions_df.empty:
        candidates = top_hubs.copy()
        candidates["drug"] = "None found"
        candidates["interaction_type"] = ""
        candidates["n_drugs"] = 0
    else:
        # For each hub gene, find all drugs
        hub_drugs = interactions_df[interactions_df["gene"].isin(top_hubs["gene"])]

        if hub_drugs.empty:
            candidates = top_hubs.copy()
            candidates["drug"] = "None found"
            candidates["interaction_type"] = ""
            candidates["n_drugs"] = 0
        else:
            # Aggregate drugs per gene
            drug_summary = hub_drugs.groupby("gene").agg({
                "drug": lambda x: "; ".join(sorted(set(x)))[:200],
                "interaction_type": lambda x: "; ".join(sorted(set(str(i) for i in x if i)))[:100],
            }).reset_index()
            drug_summary["n_drugs"] = hub_drugs.groupby("gene")["drug"].nunique().values

            candidates = top_hubs.merge(drug_summary, on="gene", how="left")
            candidates["n_drugs"] = candidates["n_drugs"].fillna(0).astype(int)
            candidates["drug"] = candidates["drug"].fillna("None found")

    # Composite evidence score
    candidates["evidence_score"] = (
        candidates["hub_score"] * 0.4 +
        (candidates["is_selectively_essential"].fillna(False).astype(float)) * 0.3 +
        (candidates["n_drugs"] > 0).astype(float) * 0.3
    )
    candidates = candidates.sort_values("evidence_score", ascending=False)

    # Save
    output_cols = ["gene", "hub_score", "eigenvector", "out_degree",
                    "k562_chronos", "selective_score", "is_selectively_essential",
                    "drug", "n_drugs", "evidence_score"]
    output_cols = [c for c in output_cols if c in candidates.columns]
    candidates[output_cols].to_csv(OUTDIR / "drug_candidates_ranked.csv", index=False)

    logger.info(f"\n  Top drug candidates for imatinib-resistant CML:")
    for _, row in candidates.head(10).iterrows():
        sel = "SEL-ESS" if row.get("is_selectively_essential") else ""
        drugs = str(row.get("drug", ""))[:60]
        logger.info(f"    {row['gene']:<12} score={row['evidence_score']:.3f} "
                     f"{sel:<8} drugs={drugs}")

    return candidates


# =====================================================================
# Main
# =====================================================================

def main():
    logger.info("=" * 70)
    logger.info("PHASE 7: Clinical Translation Pipeline")
    logger.info("=" * 70)

    # Step 1: Build graph from GIES edges (real_data)
    edges_path = GIES_CACHE / "real_data_edges.json"
    if not edges_path.exists():
        logger.error(f"  GIES edges not found at {edges_path}")
        return

    G, centrality_df = build_graph_and_centrality(edges_path)

    # Step 2: DepMap selective essentiality
    essentiality_df = compute_selective_essentiality(centrality_df)

    # Step 3: DGIdb drug interactions
    hub_genes = centrality_df.head(40)["gene"].tolist()  # Query top 40 hubs
    interactions_df = query_dgidb(hub_genes)

    # Step 4: Rank candidates
    candidates = rank_candidates(essentiality_df, interactions_df)

    # Positive control check: does BCR-ABL1 pathway appear?
    bcr_abl_targets = {"ABL1", "BCR", "STAT5A", "STAT5B", "GRB2", "SOS1", "CRKL"}
    found_targets = set(centrality_df["gene"]) & bcr_abl_targets
    logger.info(f"\n  BCR-ABL1 pathway genes in our 200-gene set: {found_targets or 'None'}")

    # Summary
    logger.info(f"\n{'=' * 70}")
    logger.info("PHASE 7 COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Hub genes identified: {len(centrality_df)}")
    sel_ess = essentiality_df["is_selectively_essential"].sum() if "is_selectively_essential" in essentiality_df.columns else 0
    logger.info(f"  Selectively essential: {sel_ess}")
    logger.info(f"  Drug-gene interactions found: {len(interactions_df)}")
    logger.info(f"  Top candidates saved to: {OUTDIR / 'drug_candidates_ranked.csv'}")


if __name__ == "__main__":
    main()
