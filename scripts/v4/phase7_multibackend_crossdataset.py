"""Cross-dataset multi-backend causal-recovery robustness check.

Tests whether the K562 finding — that trained deep VCMs (GEARS, CPA) recover
no more of the real causal graph than chance — is GIES-specific, by re-running
the substitutability evaluation on RPE1 (cross-context) and Norman
(cross-paradigm) under THREE additional causal-discovery backends:

  - PC       (constraint-based, observational)        via causal-learn
  - NOTEARS  (continuous-optimization, gradient)      via notears.linear
  - inspre   (sparse-inverse of the ACE matrix)       via R subprocess

Methodology mirrors the existing K562 backend scripts (phase5_inspre_subprocess,
phase7_pc_robustness, phase7_notears_robustness): every backend operates on the
square ACE matrix (perturbation x gene shift-from-control) restricted to the
canonical perturbed-gene set, and each backend defines its OWN real-data ground
truth (backend run on the real ACE). A condition's F1 is its recovered edge set
vs that backend's real-data edge set.

Chance floor: a column-permutation null. Each column of the REAL ACE is
independently shuffled across rows, destroying cross-gene joint structure while
preserving each gene's marginal shift magnitude. The backend is run on K such
nulls; the resulting F1 distribution is the chance band. A model is "no better
than chance" for that backend if its F1 falls inside the null's [2.5, 97.5]
percentile band.

All compute is LOCAL and CPU-only (predictions are cached h5ads; no GPU, no
Modal). R + the `inspre` package must be installed locally (they are).

Usage:
    python scripts/v4/phase7_multibackend_crossdataset.py \
        --datasets rpe1 norman --backends pc notears inspre --n-perm 20
    # smoke: one backend pass on real+gears for the smaller dataset
    python scripts/v4/phase7_multibackend_crossdataset.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import anndata as ad
from sklearn.linear_model import ElasticNet

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
OUT = ROOT / "results" / "phase7_crossdataset"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def _cache_get(key):
    p = CACHE / f"{key}.json"
    if p.exists():
        try:
            return json.load(open(p))
        except Exception:
            return None
    return None


def _cache_set(key, val):
    json.dump(val, open(CACHE / f"{key}.json", "w"))

RSCRIPT = os.environ.get("RSCRIPT_BIN", "/usr/local/bin/Rscript")

# (slim file, real-data perturbation column, control label, prediction dir)
DATASETS = {
    "rpe1": {
        "slim": DATA / "rpe1_subset_n200_slim.h5ad",
        "pert_col": "gene",
        "ctrl": "non-targeting",
        "pred_dir": DATA / "model_outputs_real_rpe1",
        "gene_list": DATA / "gene_lists" / "rpe1_genes.txt",
    },
    "norman": {
        "slim": DATA / "norman_subset_n200_slim.h5ad",
        "pert_col": "perturbation",
        "ctrl": "control",
        "pred_dir": DATA / "model_outputs_real_norman",
        "gene_list": DATA / "gene_lists" / "norman_genes.txt",
    },
}

MODELS = ["gears", "cpa"]  # Geneformer omitted (heuristic, not learned ISP)
MIN_CELLS_PER_PERT = 5

# inspre R driver (square ACE in, adjacency CSV out). Mirrors phase5.
R_INSPRE_SCRIPT = r'''
library(inspre)
args <- commandArgs(trailingOnly = TRUE)
ace_path <- args[1]; out_path <- args[2]; target_edges <- as.integer(args[3])
R_hat <- as.matrix(read.csv(ace_path, header=FALSE))
result <- fit_inspre_from_R(R_hat = R_hat, its = 200, verbose = 0, train_prop = 0.8)
G_hat <- result$G_hat; lambdas <- result$lambda; test_err <- result$test_error
valid_te <- !is.nan(test_err) & !is.na(test_err)
# Finite-guarded edge counts: at n=196 inspre's ADMM can leave NaN entries in
# some G slices AND return all-NaN CV test_error. Without is.finite(), the sum
# is NA -> which.min(all-NA) -> integer(0) -> empty G_best -> empty CSV.
edge_counts <- sapply(1:dim(G_hat)[3], function(i) {
    sl <- G_hat[,,i]; sum(abs(sl) > 1e-6 & is.finite(sl))
})
if (any(valid_te)) {
    best_idx <- which(valid_te)[which.min(test_err[valid_te])]
} else {
    best_idx <- which.min(abs(edge_counts - target_edges))
}
if (length(best_idx) == 0 || is.na(best_idx)) best_idx <- which.max(edge_counts)
G_best <- G_hat[,,best_idx]
G_best[!is.finite(G_best)] <- 0
write.csv(G_best, out_path, row.names=FALSE)
'''


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def load_gene_names(cfg) -> list[str]:
    genes = [g.strip() for g in open(cfg["gene_list"]) if g.strip()]
    return sorted(genes)


def load_real(cfg, gene_names):
    """Return (ctrl_mean, {gene: real_mean_expr}) over gene_names, from the slim."""
    adata = ad.read_h5ad(cfg["slim"])
    adata.var_names = [str(v) for v in adata.var_names]
    v2i = {v: i for i, v in enumerate(adata.var_names)}
    cols = [v2i[g] for g in gene_names]  # gene_names verified subset of var_names
    X = _dense(adata.X).astype(np.float64)
    labels = adata.obs[cfg["pert_col"]].astype(str).values

    ctrl_mask = labels == cfg["ctrl"]
    if ctrl_mask.sum() == 0:
        raise ValueError(f"no control cells (label={cfg['ctrl']!r})")
    ctrl_mean = X[ctrl_mask][:, cols].mean(axis=0)

    real_means = {}
    for g in gene_names:
        idx = np.where(labels == g)[0]
        if len(idx) >= MIN_CELLS_PER_PERT:
            real_means[g] = X[idx][:, cols].mean(axis=0)
    return ctrl_mean, real_means, X[ctrl_mask][:, cols]


def load_pred_means(pred_path: Path, gene_names) -> dict[str, np.ndarray]:
    """Mean predicted expression per perturbation over gene_names.

    Correctly averages over cells when the prediction h5ad is cell-level
    (CPA: many rows per perturbation). GEARS has one row per perturbation,
    so the mean is a no-op there.
    """
    p = ad.read_h5ad(pred_path)
    p.var_names = [str(v) for v in p.var_names]
    v2i = {v: i for i, v in enumerate(p.var_names)}
    cols, present = [], []
    for g in gene_names:
        if g in v2i:
            cols.append(v2i[g]); present.append(g)
    pmat = _dense(p.X).astype(np.float64)[:, cols]
    pert_col = next((c for c in ["perturbation", "condition", "gene", "pert"]
                     if c in p.obs.columns), None)
    if pert_col is None:
        raise ValueError(f"no perturbation column in {pred_path.name}")
    labels = p.obs[pert_col].astype(str).values

    out = {}
    for g in gene_names:
        m = labels == g
        if m.sum() == 0:
            continue
        vec = np.zeros(len(gene_names), dtype=np.float64)
        # fill only the columns present in the prediction var set
        sub = pmat[m].mean(axis=0)  # mean over cells for this perturbation
        for k, gp in enumerate(present):
            vec[gene_names.index(gp)] = sub[k]
        out[g] = vec
    return out


def elasticnet_means(ctrl_X, gene_names, seed=0) -> dict[str, np.ndarray]:
    """Trained linear baseline: per-gene ElasticNet on control cells, predict
    the ~90% CRISPRi-knockdown mean for each perturbation. Matches
    run_eval_local.run_elasticnet_baseline (returns means, not sampled cells)."""
    n = len(gene_names)
    control_means = ctrl_X.mean(axis=0)
    models = {}
    for g_idx in range(n):
        feat = [i for i in range(n) if i != g_idx]
        m = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=0)
        m.fit(ctrl_X[:, feat], ctrl_X[:, g_idx])
        models[g_idx] = m
    out = {}
    for g_idx, gene in enumerate(gene_names):
        state = control_means.copy()
        state[g_idx] *= 0.1
        pred = state.copy()
        for t_idx, mdl in models.items():
            if t_idx == g_idx:
                continue
            feat = [i for i in range(n) if i != t_idx]
            pred[t_idx] = max(float(mdl.predict(state[feat].reshape(1, -1))[0]), 0.0)
        out[gene] = pred
    return out


def build_ace(means: dict, ctrl_mean, gene_names) -> np.ndarray:
    """Square ACE: row i = mean(X | do(gene_i)) - ctrl_mean, over gene_names."""
    n = len(gene_names)
    ace = np.zeros((n, n), dtype=np.float64)
    for i, g in enumerate(gene_names):
        if g in means:
            ace[i] = means[g] - ctrl_mean
    return np.nan_to_num(ace, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------
# Backends — each returns a directed edge set over gene_names
# --------------------------------------------------------------------------
def pc_edges(ace, gene_names, alpha=0.05) -> set:
    from causallearn.search.ConstraintBased.PC import pc
    data = np.nan_to_num(np.asarray(ace, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    rng = np.random.default_rng(0)
    zero_cols = data.std(axis=0) < 1e-12
    if zero_cols.any():
        data = data + rng.normal(0, 1e-8, size=data.shape) * zero_cols[None, :]
    cg = pc(data, alpha=alpha, indep_test="fisherz", verbose=False, show_progress=False)
    mat = cg.G.graph
    n = len(gene_names)
    return {(gene_names[i], gene_names[j]) for i in range(n) for j in range(n)
            if i != j and mat[i, j] != 0}


def notears_edges(ace, gene_names, lambda1=0.05, w_thresh=0.10, max_iter=100) -> set:
    from notears.linear import notears_linear
    W = notears_linear(np.asarray(ace, dtype=np.float64), lambda1=lambda1,
                       loss_type="l2", max_iter=max_iter, h_tol=1e-8, w_threshold=0.0)
    n = len(gene_names)
    return {(gene_names[i], gene_names[j]) for i in range(n) for j in range(n)
            if i != j and abs(W[i, j]) >= w_thresh}


def inspre_edges(ace, gene_names, target_edges=1200) -> set:
    with tempfile.TemporaryDirectory() as td:
        ace_path = os.path.join(td, "ace.csv")
        out_path = os.path.join(td, "adj.csv")
        r_path = os.path.join(td, "run.R")
        np.savetxt(ace_path, ace, delimiter=",")
        with open(r_path, "w") as f:
            f.write(R_INSPRE_SCRIPT)
        res = subprocess.run(f"{RSCRIPT} {r_path} {ace_path} {out_path} {target_edges}",
                             shell=True, capture_output=True, text=True, timeout=1800)
        if res.returncode != 0:
            raise RuntimeError(f"inspre R failed: {res.stderr[-400:]}")
        import pandas as pd
        adj = pd.read_csv(out_path).values
    n = len(gene_names)
    if adj.shape != (n, n):
        raise RuntimeError(f"inspre adjacency malformed: shape {adj.shape}, expected {(n, n)}")
    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    return {(gene_names[i], gene_names[j]) for i in range(n) for j in range(n)
            if i != j and abs(adj[i, j]) > 1e-6}


BACKENDS = {"pc": pc_edges, "notears": notears_edges, "inspre": inspre_edges}


def f1(pred: set, gt: set) -> tuple[float, float, float]:
    pred, gt = set(pred), set(gt)
    tp = len(pred & gt)
    if tp == 0:
        return 0.0, 0.0, 0.0
    p = tp / max(1, len(pred)); r = tp / max(1, len(gt))
    return 2 * p * r / (p + r), p, r


def column_permute(ace, rng) -> np.ndarray:
    """Independently shuffle each column across rows: destroys cross-gene joint
    structure, preserves per-gene marginal shift magnitudes."""
    out = ace.copy()
    n_rows = out.shape[0]
    for j in range(out.shape[1]):
        out[:, j] = out[rng.permutation(n_rows), j]
    return out


def _null_worker(args):
    """One permutation-null draw (module-level for multiprocessing). Runs the
    backend on a column-permuted real ACE and returns (perm_seed, F1 vs GT)."""
    bk_name, ace_real, gene_names, gt, target, perm_seed = args
    perm = column_permute(ace_real, np.random.default_rng(perm_seed))
    return perm_seed, f1(BACKENDS[bk_name](perm, gene_names, **target), gt)[0]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run_dataset(ds_name, backends, n_perm, seed, smoke=False, perm_null=False,
                only_conds=None):
    cfg = DATASETS[ds_name]
    gene_names = load_gene_names(cfg)
    print(f"\n{'='*72}\n{ds_name}: {len(gene_names)} canonical genes\n{'='*72}", flush=True)

    ctrl_mean, real_means, ctrl_X = load_real(cfg, gene_names)
    print(f"  real perturbations with >={MIN_CELLS_PER_PERT} cells: {len(real_means)}", flush=True)

    # Condition means
    cond_means = {"real_data": real_means}
    for m in MODELS:
        pp = cfg["pred_dir"] / f"{m}_predictions.h5ad"
        if pp.exists():
            cond_means[m] = load_pred_means(pp, gene_names)
            print(f"  {m}: {len(cond_means[m])} perturbations", flush=True)
    if not smoke:
        t0 = time.time()
        cond_means["elastic_net"] = elasticnet_means(ctrl_X, gene_names, seed=seed)
        print(f"  elastic_net: trained ({time.time()-t0:.1f}s)", flush=True)

    aces = {c: build_ace(m, ctrl_mean, gene_names) for c, m in cond_means.items()}
    eval_conds = [c for c in ["gears", "cpa", "elastic_net"] if c in aces]
    if only_conds:
        eval_conds = [c for c in eval_conds if c in only_conds]
    if smoke:
        eval_conds = [c for c in ["gears"] if c in aces]

    results = {"dataset": ds_name, "n_genes": len(gene_names),
               "n_real_perts": len(real_means), "n_perm": (0 if smoke else n_perm),
               "seed": seed, "backends": {}}

    for bk in backends:
        fn = BACKENDS[bk]
        print(f"\n  --- backend: {bk} ---", flush=True)
        ck = f"{ds_name}_{bk}"
        # GT (cached): edges stored as list of [g1, g2]
        gt_c = _cache_get(f"{ck}_gt")
        if gt_c is not None:
            gt = set(tuple(e) for e in gt_c)
            gt_secs = 0.0
            print(f"  {bk} real GT: {len(gt)} edges (CACHED)", flush=True)
        else:
            t0 = time.time()
            gt = fn(aces["real_data"], gene_names)
            gt_secs = time.time() - t0
            _cache_set(f"{ck}_gt", [list(e) for e in gt])
            print(f"  {bk} real GT: {len(gt)} edges ({gt_secs:.1f}s)", flush=True)
        # inspre conditions match the GT edge budget
        target = {"inspre": {"target_edges": len(gt) if len(gt) else 1200}}.get(bk, {})

        n_universe = len(gene_names) * (len(gene_names) - 1)  # directed, no self-loops

        bk_res = {"gt_edges": len(gt), "gt_secs": round(gt_secs, 1),
                  "n_universe": n_universe, "conditions": {}}
        for c in eval_conds:
            cached = _cache_get(f"{ck}_cond_{c}")
            if cached is not None:
                bk_res["conditions"][c] = cached
                print(f"    {c:<12} F1={cached['f1']:.4f} (CACHED)", flush=True)
                continue
            t0 = time.time()
            edges = fn(aces[c], gene_names, **target)
            fv, pv, rv = f1(edges, gt)
            # Analytic random-edge-set floor: expected F1 of a random predictor
            # emitting the SAME number of edges as this condition, vs this
            # backend's GT. E[TP] = n_pred * n_gt / U. Free, density-matched,
            # directly comparable to the GIES eval's Random baseline.
            n_pred, n_gt = len(edges), len(gt)
            if n_pred > 0 and n_gt > 0:
                etp = n_pred * n_gt / n_universe
                rand_f1 = round(2 * etp / (n_pred + n_gt), 4)
            else:
                rand_f1 = 0.0
            bk_res["conditions"][c] = {"f1": round(fv, 4), "precision": round(pv, 4),
                                       "recall": round(rv, 4), "n_edges": n_pred,
                                       "random_edge_f1": rand_f1,
                                       "f1_over_random": round(fv / rand_f1, 2) if rand_f1 else None,
                                       "secs": round(time.time()-t0, 1)}
            _cache_set(f"{ck}_cond_{c}", bk_res["conditions"][c])
            print(f"    {c:<12} F1={fv:.4f} (rand={rand_f1:.4f}, x{(fv/rand_f1 if rand_f1 else 0):.2f}) "
                  f"P={pv:.4f} R={rv:.4f} edges={n_pred} ({time.time()-t0:.1f}s)", flush=True)

        # Permutation-null chance floor (EXPENSIVE). Each permutation's F1 is
        # cached by seed, so a kill mid-null only loses the in-flight perm —
        # relaunching resumes from the cache (the run keeps getting reaped on
        # session boundaries, so this is essential).
        if not smoke and perm_null and n_perm > 0:
            from multiprocessing import Pool
            t0 = time.time()
            ace_real = aces["real_data"]
            seeds = [seed * 100003 + k for k in range(n_perm)]
            cached_f1 = {}
            todo = []
            for s in seeds:
                cv = _cache_get(f"{ck}_null_{s}")
                if cv is not None:
                    cached_f1[s] = cv
                else:
                    todo.append(s)
            print(f"    null: {len(cached_f1)}/{n_perm} cached, {len(todo)} to run",
                  flush=True)
            if todo:
                jobs = [(bk, ace_real, gene_names, gt, target, s) for s in todo]
                n_workers = max(1, min(len(todo), (os.cpu_count() or 2) - 1))
                with Pool(n_workers) as pool:
                    for s, fv in pool.imap_unordered(_null_worker, jobs):
                        cached_f1[s] = fv
                        _cache_set(f"{ck}_null_{s}", fv)
            null_f1 = np.array([cached_f1[s] for s in seeds])
            print(f"    null: {n_perm} perms done ({time.time()-t0:.1f}s)", flush=True)
            bk_res["chance_floor"] = {
                "mean": round(float(null_f1.mean()), 4),
                "p2.5": round(float(np.percentile(null_f1, 2.5)), 4),
                "p97.5": round(float(np.percentile(null_f1, 97.5)), 4),
                "n_perm": n_perm,
            }
            cf = bk_res["chance_floor"]
            print(f"    chance_floor  mean={cf['mean']:.4f} "
                  f"[{cf['p2.5']:.4f}, {cf['p97.5']:.4f}] (n={n_perm})", flush=True)
            # verdict per condition
            for c in eval_conds:
                f1c = bk_res["conditions"][c]["f1"]
                bk_res["conditions"][c]["within_chance"] = bool(
                    cf["p2.5"] <= f1c <= cf["p97.5"])

        results["backends"][bk] = bk_res
        # Write a per-backend partial immediately so parallel-by-backend jobs
        # never collide on one combined file, and progress survives a crash.
        if not smoke:
            bpath = OUT / f"{ds_name}_{bk}.json"
            with open(bpath, "w") as f:
                json.dump({**{k: results[k] for k in results if k != "backends"},
                           "backend": bk, **bk_res}, f, indent=2)
            print(f"  wrote {bpath}", flush=True)

    tag = "smoke" if smoke else "multibackend"
    out_path = OUT / f"{ds_name}_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  wrote {out_path}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["rpe1", "norman"],
                    choices=list(DATASETS))
    ap.add_argument("--backends", nargs="+", default=["pc", "notears", "inspre"],
                    choices=list(BACKENDS))
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perm-null", action="store_true",
                    help="also compute the EXPENSIVE column-permutation null "
                         "(reruns backend n_perm times; default off)")
    ap.add_argument("--conditions", nargs="+", default=None,
                    choices=["gears", "cpa", "elastic_net"],
                    help="restrict evaluated conditions (e.g. skip a backend-"
                         "intractable one); GT + null still computed")
    ap.add_argument("--smoke", action="store_true",
                    help="one backend pass on real+gears for one dataset, no null")
    args = ap.parse_args()

    # Fail fast if inspre's R toolchain is missing (don't crash 20 min into a run).
    if "inspre" in args.backends:
        chk = subprocess.run(f'{RSCRIPT} -e \'cat(requireNamespace("inspre",quietly=TRUE))\'',
                             shell=True, capture_output=True, text=True)
        if chk.returncode != 0 or "TRUE" not in chk.stdout:
            sys.exit(f"inspre/R not available via {RSCRIPT!r} "
                     f"(rc={chk.returncode}, out={chk.stdout!r}); set RSCRIPT_BIN or drop --backends inspre")

    if args.smoke:
        run_dataset("norman", args.backends, 0, args.seed, smoke=True)
        return
    for ds in args.datasets:
        run_dataset(ds, args.backends, args.n_perm, args.seed, smoke=False,
                    perm_null=args.perm_null, only_conds=args.conditions)


if __name__ == "__main__":
    main()
