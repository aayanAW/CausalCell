#!/usr/bin/env python3
"""B (definitive) — GPU CPA-style conditional-VAE retrain with joint-structure loss.

The pre-registered DEFINITIVE Track B test (prereg_B_precision_objective.md names
the GPU CPA retrain as the real experiment; the CPU head sweep was only a proxy).

Trains a conditional VAE (CPA-style: shared encoder, per-perturbation latent
shift, shared decoder) on REAL K562 single cells, once per condition:
  recon        : ELBO only (the incumbent per-gene objective).
  precision    : ELBO + lam * precision_support_penalty on decoded pert shifts.
  mmd          : ELBO + lam * mmd_penalty.
  sinkhorn     : ELBO + lam * sinkhorn_penalty.
Outputs predicted per-perturbation MEAN expression for every condition to a JSON
(means only -- small). Causal scoring (overlay -> GIES vs the A1 anchored GT) is
done LOCALLY in .venv_gears afterward, because the GIES/R stack is not on the GPU
box. This cleanly separates GPU training from CPU causal eval.

Runs inside the ROCm PyTorch container. Device auto-detected (cuda/hip or cpu).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
try:
    from causalcellbench.training.joint_structure_objectives import (
        mmd_penalty, sinkhorn_penalty, precision_support_penalty, fit_real_precision,
    )
except Exception:
    # allow running standalone on the GPU box: inline-import from a sibling copy
    from joint_structure_objectives import (  # type: ignore
        mmd_penalty, sinkhorn_penalty, precision_support_penalty, fit_real_precision,
    )

CTRL = "non-targeting"


class CVAE(nn.Module):
    """CPA-style: encoder to a basal latent, additive per-perturbation latent
    shift, decoder back to gene space. Log1p-normalized inputs."""

    def __init__(self, n_genes, n_perts, latent=64, hidden=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_genes, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.lv = nn.Linear(hidden, latent)
        self.pert_emb = nn.Embedding(n_perts, latent)  # additive latent shift
        nn.init.zeros_(self.pert_emb.weight)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_genes))

    def encode(self, x):
        h = self.enc(x); return self.mu(h), self.lv(h)

    def forward(self, x, pid):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        z = z + self.pert_emb(pid)
        return self.dec(z), mu, lv

    @torch.no_grad()
    def predict_mean(self, basal_x, pid, device):
        """Predicted mean expression: decode basal latent + pert shift, no noise."""
        mu, _ = self.encode(basal_x)
        z = mu + self.pert_emb(torch.full((basal_x.shape[0],), pid, device=device))
        return self.dec(z).mean(0)


def log1p_norm(X):
    lib = X.sum(1, keepdims=True); med = np.median(lib[lib > 0])
    return np.log1p(X / np.maximum(lib, 1e-9) * med)


def train_condition(term, lam, Xt, pid_t, ctrl_idx, real_shift, pmap, device,
                    genes, epochs, seed, batch=512):
    torch.manual_seed(seed); np.random.seed(seed)
    n_genes, n_perts = Xt.shape[1], len(pmap)
    net = CVAE(n_genes, n_perts).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    real_prec = fit_real_precision(real_shift).to(device) if term == "precision" else None
    real_shift_d = real_shift.to(device)
    N = Xt.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            x = Xt[idx]; pid = pid_t[idx]
            out, mu, lv = net(x, pid)
            recon = ((out - x) ** 2).mean()
            kld = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()) * 1e-3
            loss = recon + kld
            if lam > 0:
                # decoded per-perturbation shift on this batch (pred - ctrl-decoded)
                pen = torch.tensor(0.0, device=device)
                if term == "precision":
                    # build predicted shift matrix over the batch's perts
                    pen = precision_support_penalty(out - x.mean(0, keepdim=True), real_prec)
                elif term == "mmd":
                    pen = mmd_penalty(out, real_shift_d)
                elif term == "sinkhorn":
                    pen = sinkhorn_penalty(out, real_shift_d)
                loss = loss + lam * pen
            opt.zero_grad(); loss.backward(); opt.step()
    # predicted means per perturbation, using control cells as basal
    basal = Xt[ctrl_idx]
    means = {}
    for p, pid in pmap.items():
        means[p] = net.predict_mean(basal, pid, device).cpu().numpy().tolist()
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--terms", nargs="+", default=["recon", "precision", "mmd", "sinkhorn"])
    ap.add_argument("--min-cells", type=int, default=50)
    ap.add_argument("--lam-grid", type=float, nargs="+", default=None,
                    help="if set, sweep these lambda values for a SINGLE term "
                         "(--terms must name exactly one non-recon term); "
                         "lam=0 is trained once as recon. Keys results by <term>_lam<lam>.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = ad.read_h5ad(args.slim)
    genes = list(a.var["gene_name"].astype(str).values) if "gene_name" in a.var \
        else list(a.var_names.astype(str).values)
    X = np.asarray(a.X.todense() if hasattr(a.X, "todense") else a.X, dtype=np.float64)
    Xln = log1p_norm(X)
    labels = a.obs["gene"].astype(str).values

    from collections import Counter
    cnt = Counter(labels)
    perts = [g for g in genes if cnt.get(g, 0) >= args.min_cells and g != CTRL]
    pmap = {p: i for i, p in enumerate(perts)}
    # cells belonging to a kept perturbation (+ control get pid via nearest? no:
    # controls are used only as basal; training uses perturbed cells with their pid)
    keep = np.isin(labels, perts)
    Xt_all = torch.tensor(Xln[keep], dtype=torch.float32)
    pid_all = torch.tensor([pmap[l] for l in labels[keep]], dtype=torch.long)
    ctrl_idx_full = np.where(labels == CTRL)[0]
    # control basal in the SAME normalized space, appended so predict uses them
    ctrl_ln = torch.tensor(Xln[labels == CTRL], dtype=torch.float32)

    # real per-perturbation shift (mean pert - mean ctrl) for the joint target
    ctrl_mean = Xln[labels == CTRL].mean(0)
    real_shift = torch.tensor(np.stack([Xln[labels == p].mean(0) - ctrl_mean for p in perts]),
                              dtype=torch.float32)

    device_t = torch.device(device)
    Xt_all = Xt_all.to(device_t); pid_all = pid_all.to(device_t)
    ctrl_ln_d = ctrl_ln.to(device_t)
    # ctrl_idx into the ctrl tensor: we pass ctrl_ln_d directly as basal
    results = {"meta": {"device": device, "n_genes": len(genes), "n_perts": len(perts),
                        "epochs": args.epochs, "lam": args.lam, "seeds": args.seeds,
                        "terms": args.terms, "genes": genes, "ctrl_mean": ctrl_mean.tolist()},
               "predicted_means": {}}

    # monkeypatch: train_condition expects Xt indexed by ctrl_idx for basal;
    # simpler: temporarily concat ctrl cells and pass their indices
    Xcat = torch.cat([Xt_all, ctrl_ln_d], 0)
    pid_cat = torch.cat([pid_all, torch.zeros(ctrl_ln_d.shape[0], dtype=torch.long, device=device_t)], 0)
    ctrl_idx = torch.arange(Xt_all.shape[0], Xcat.shape[0], device=device_t)

    if args.lam_grid is not None:
        # lambda-sweep mode: one non-recon term, keyed <term>_lam<lam> (+ recon once)
        nonrecon = [t for t in args.terms if t != "recon"]
        assert len(nonrecon) == 1, "lam-grid mode needs exactly one non-recon term"
        term = nonrecon[0]
        conditions = [("recon", 0.0)] + [(f"{term}_lam{lam:g}", lam)
                                         for lam in args.lam_grid if lam > 0]
        results["meta"]["sweep_term"] = term
        results["meta"]["lam_grid"] = args.lam_grid
        for cond_name, lam in conditions:
            results["predicted_means"][cond_name] = {}
            train_term = "recon" if lam == 0 else term
            for seed in args.seeds:
                means = train_condition(train_term, lam, Xcat, pid_cat, ctrl_idx,
                                        real_shift, pmap, device_t, genes, args.epochs, seed)
                results["predicted_means"][cond_name][str(seed)] = means
                print(f"{cond_name} seed{seed} done", flush=True)
    else:
        for term in args.terms:
            results["predicted_means"][term] = {}
            for seed in args.seeds:
                means = train_condition(term, (0.0 if term == "recon" else args.lam),
                                        Xcat, pid_cat, ctrl_idx, real_shift, pmap, device_t,
                                        genes, args.epochs, seed)
                results["predicted_means"][term][str(seed)] = means
                print(f"{term} seed{seed} done", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"))
    print("SAVED", args.out, flush=True)


if __name__ == "__main__":
    main()
