"""Phase 3 — inspre lambda sweep.

The original CausalCellBench reports `inspre recovers 12,280 edges from real
data, ... zero from CPA and Geneformer at the test-error-optimal lambda`.
Reviewers will (correctly) ask whether zero edges is universal or specific
to one lambda. This script traces recovered edge count across the full
lambda path for each condition and reports the smallest lambda at which any
edge is recovered for each.

Output: results/phase7/inspre_lambda_sweep.json with per-condition arrays
        of (lambda, n_edges).

This script depends on R + the inspre package being available; if R is
absent it writes a placeholder and exits gracefully so downstream pipelines
do not break.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"
OUT = RES / "phase7"
OUT.mkdir(parents=True, exist_ok=True)

K562 = DATA / "k562_subset_n200_slim.h5ad"
PRED_DIR_VAN = DATA / "model_outputs_real_n200"
PRED_DIR_CAL = DATA / "model_outputs_real_n200_calibrated"
PERT_COL = "gene"
CTRL = "non-targeting"


def find_rscript() -> str | None:
    for candidate in [
        "/opt/mamba/envs/renv/bin/Rscript",
        "/opt/homebrew/bin/Rscript",
        "/usr/local/bin/Rscript",
        shutil.which("Rscript"),
    ]:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


R_PATH_SCRIPT = """
library(inspre)
args <- commandArgs(trailingOnly=TRUE)
ace_path <- args[1]; out_path <- args[2]
R_hat <- as.matrix(read.csv(ace_path, header=FALSE))
result <- fit_inspre_from_R(R_hat=R_hat, its=200, verbose=0, train_prop=0.8)
G_hat <- result$G_hat
lambdas <- result$lambda
edge_counts <- sapply(1:dim(G_hat)[3], function(i) sum(abs(G_hat[,,i]) > 1e-6))
write.csv(data.frame(lambda=lambdas, n_edges=edge_counts), out_path, row.names=FALSE)
"""


def build_ace(adata: ad.AnnData, gene_names: list[str], pred_means: dict[str, np.ndarray] | None) -> np.ndarray:
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float64)
    labels = adata.obs[PERT_COL].values
    is_ctrl = labels == CTRL
    ctrl_mean = X[is_ctrl].mean(axis=0)
    n = len(gene_names)
    ace = np.zeros((n, n), dtype=np.float64)
    if pred_means is not None:
        for i, g in enumerate(gene_names):
            if g in pred_means:
                ace[i] = pred_means[g] - ctrl_mean
        return ace
    # Real-data ACE
    for i, g in enumerate(gene_names):
        idx = np.where(labels == g)[0]
        if len(idx) >= 5:
            ace[i] = X[idx].mean(axis=0) - ctrl_mean
    return ace


def load_pred_means(path: Path, gene_names: list[str]) -> dict[str, np.ndarray]:
    p = ad.read_h5ad(path)
    p_genes = list(p.var_names)
    pred_to_idx = {g: i for i, g in enumerate(p_genes)}
    cols = [pred_to_idx[g] for g in gene_names if g in pred_to_idx]
    pmat = p.X.toarray() if hasattr(p.X, "toarray") else np.asarray(p.X)
    pmat = np.asarray(pmat[:, cols], dtype=np.float64)
    out = {}
    for i in range(pmat.shape[0]):
        gene = str(p.obs[PERT_COL].iloc[i]) if PERT_COL in p.obs.columns else \
               str(p.obs.iloc[i, 0])
        out[gene] = pmat[i]
    return out


def run_lambda_sweep(ace: np.ndarray, rscript: str) -> dict | None:
    with tempfile.TemporaryDirectory() as tmp:
        ap = os.path.join(tmp, "ace.csv"); op = os.path.join(tmp, "out.csv"); rp = os.path.join(tmp, "r.R")
        np.savetxt(ap, ace, delimiter=",")
        with open(rp, "w") as f:
            f.write(R_PATH_SCRIPT)
        try:
            res = subprocess.run([rscript, rp, ap, op], capture_output=True, text=True, timeout=900)
        except Exception as e:
            return {"error": str(e)}
        if res.returncode != 0 or not os.path.exists(op):
            return {"error": (res.stderr or "")[-500:]}
        import csv
        rows = []
        with open(op) as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    lam = float(r["lambda"])
                    ne = int(float(r["n_edges"]))
                except (ValueError, TypeError):
                    continue
                if not (np.isnan(lam) or np.isinf(lam)):
                    rows.append((lam, ne))
        if not rows:
            return {"error": "no finite lambda values returned"}
        return {"lambdas": [r[0] for r in rows], "n_edges": [r[1] for r in rows]}


def main() -> int:
    rscript = find_rscript()
    out: dict[str, dict] = {"r_available": rscript is not None}
    if rscript is None:
        print("Rscript not found on PATH — writing placeholder JSON.")
        with open(OUT / "inspre_lambda_sweep.json", "w") as f:
            json.dump({"r_available": False, "note": "skipped — Rscript missing"}, f, indent=2)
        return 0
    print(f"Using Rscript: {rscript}")
    adata = ad.read_h5ad(K562)
    adata.var_names = [str(g) for g in adata.var_names]
    gene_names = list(adata.var_names)

    # Real data
    print("Real-data lambda sweep ...")
    ace_real = build_ace(adata, gene_names, None)
    out["real"] = run_lambda_sweep(ace_real, rscript) or {}

    # ElasticNet (use cached predictions if present)
    enet_path = PRED_DIR_VAN / "elastic_net_predictions.h5ad"
    if enet_path.exists():
        means = load_pred_means(enet_path, gene_names)
        ace = build_ace(adata, gene_names, means)
        print("ElasticNet lambda sweep ...")
        out["elastic_net"] = run_lambda_sweep(ace, rscript) or {}

    # Deep models
    for m in ["gears", "cpa", "geneformer"]:
        for tag, d in [("vanilla", PRED_DIR_VAN), ("calibrated", PRED_DIR_CAL)]:
            p = d / f"{m}_predictions.h5ad"
            if not p.exists():
                continue
            means = load_pred_means(p, gene_names)
            ace = build_ace(adata, gene_names, means)
            print(f"{m} ({tag}) lambda sweep ...")
            out[f"{m}_{tag}"] = run_lambda_sweep(ace, rscript) or {}

    # Summary
    summary = {}
    for k, v in out.items():
        if not isinstance(v, dict) or "n_edges" not in v:
            continue
        ne = np.array(v["n_edges"])
        ll = np.array(v["lambdas"])
        any_idx = np.where(ne > 0)[0]
        summary[k] = {
            "max_edges_along_path": int(ne.max() if len(ne) else 0),
            "smallest_lambda_with_any_edge": float(ll[any_idx[0]]) if len(any_idx) else None,
            "edges_at_smallest_lambda": int(ne[any_idx[0]]) if len(any_idx) else 0,
            "any_edge_along_path": bool(len(any_idx) > 0),
        }
    out["summary"] = summary
    print(json.dumps(summary, indent=2))

    with open(OUT / "inspre_lambda_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT/'inspre_lambda_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
