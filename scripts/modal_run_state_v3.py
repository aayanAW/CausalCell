"""STATE inference v3 — fixes the LlamaConfig validation bug.

Root cause discovered in this session:
- The released arcinstitute/ST-SE-Replogle checkpoint specifies
  hidden_size=328, num_attention_heads=12, head_dim=64 (decoupled).
- transformers >= 5.x added a strict validator that rejects
  hidden_size % num_heads != 0 even when head_dim is explicit.
- The arc-state package pins transformers >= 4.52.3 (which would pull
  in the strict 5.x), so `pip install -e state` breaks with the
  released checkpoint.
- transformers 4.45.2 supports decoupled head_dim and accepts the
  config without complaint.

Fix in this script:
1. Install state package
2. Force-downgrade transformers to 4.45.2 (pre-strict-validator)
3. Run state tx infer

Usage:
    modal run --detach scripts/modal_run_state_v3.py
"""
from __future__ import annotations

import modal

app = modal.App("ccbench-state-v3")
vol = modal.Volume.from_name("ccbench-data", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.4.1",
        "anndata==0.10.9",
        "scanpy==1.10.3",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "h5py",
        "huggingface_hub",
        "tqdm",
        "scipy==1.13.1",
        "pyyaml",
        "scikit-misc",
        "hatchling",
        "hatch-vcs",
    )
    .run_commands(
        "git clone https://github.com/ArcInstitute/state.git /opt/state",
        "cd /opt/state && pip install -e .",
        # Pin transformers to 4.52.3: this version BOTH accepts decoupled
        # head_dim (released ST-SE-Replogle ckpt has hidden=328, heads=12,
        # head_dim=64 -- the strict validator that rejects this was added in
        # transformers >= 5.x) AND has the **flash_attn_kwargs catch-all in
        # LlamaModel.forward that the state package's LlamaBidirectionalModel
        # passes is_causal=False through. 4.45.2 lacked the catch-all and
        # blew up at inference time. Pre-flight verified locally before launch.
        "pip install transformers==4.52.3 'tokenizers>=0.21,<0.22' "
        "'huggingface_hub>=0.30,<1.0' 'accelerate>=0.30' "
        "'safetensors>=0.4' 'regex' 'packaging' "
        "--force-reinstall --no-deps",
        # Patch transformers' modeling_utils.py source directly: replace
        # 'if v not in ALL_PARALLEL_STYLES:' with a None-tolerant check.
        # Reproducible bug in 4.52.3 (verified locally): ALL_PARALLEL_STYLES
        # is None on Modal containers -> 'v not in None' raises TypeError.
        # Patching source ensures the fix is in effect for ALL Python
        # processes including state CLI subprocess (sitecustomize did NOT
        # auto-fire in the subprocess in attempt 14).
        "python3 -c \"import re; "
        "p='/usr/local/lib/python3.11/site-packages/transformers/modeling_utils.py'; "
        "s=open(p).read(); "
        "old='if v not in ALL_PARALLEL_STYLES:'; "
        "new='if ALL_PARALLEL_STYLES is not None and v not in ALL_PARALLEL_STYLES:'; "
        "n=s.count(old); "
        "assert n==1, f'expected 1 match, found {n}'; "
        "open(p,'w').write(s.replace(old, new)); "
        "print(f'patched modeling_utils.py: {n} replacement(s)')\"",
        # CPU-only torch since Modal GPU exposure was unreliable for this
        # image. SE-600M on CPU is slower but completes reliably.
        "pip install torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps",
        # Fix `ImportError: cannot import name 'triton_key'`: a stale triton 2.x
        # was on the path; torch 2.4.1 needs triton 3.0.0 (which defines
        # triton_key). Pin it explicitly.
        "pip install triton==3.0.0 --force-reinstall --no-deps || true",
    )
    # Belt-and-suspenders: CPU SE-600M inference never needs torch.compile/
    # inductor, so disable it to avoid importing triton at all.
    .env({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
          "TORCH_COMPILE_DISABLE": "1", "TORCHINDUCTOR_DISABLE": "1"})
)


@app.function(
    image=image,
    cpu=16,
    timeout=21600,  # 6 hours
    volumes={"/data": vol},
    memory=131072,  # 128 GB to head off OOM kills (SE-600M peak with state pkg overhead unknown)
)
def run_state_infer(n_cells_per_pert: int = 1, n_ctrl_keep: int = 120, seed: int = 42):
    import os, subprocess, json, time, sys
    from pathlib import Path
    import torch
    import anndata as ad
    import numpy as np
    import scanpy as sc

    # Patch transformers BEFORE importing LlamaConfig/LlamaModel: in trans
    # 4.52.3 on Modal, ALL_PARALLEL_STYLES is None at LlamaModel.__init__
    # time (causes 'NoneType not iterable' in post_init's tp_plan check).
    # Sitecustomize doesn't reliably fire here so do it inline.
    import transformers.modeling_utils as _mu
    if _mu.ALL_PARALLEL_STYLES is None:
        class _AllPass:
            def __contains__(self, x):
                return True
        _mu.ALL_PARALLEL_STYLES = _AllPass()
        print("[runtime] patched transformers ALL_PARALLEL_STYLES (was None)")
    else:
        print(f"[runtime] ALL_PARALLEL_STYLES already set: {type(_mu.ALL_PARALLEL_STYLES).__name__}")
    import transformers
    from transformers import LlamaConfig, LlamaModel

    print(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    print(f"Transformers: {transformers.__version__}")

    # Sanity-check the LlamaConfig fix AND the is_causal pass-through (the
    # latter caused STATE 9th to fail at Stage 2 with transformers==4.45.2).
    test_cfg = LlamaConfig(
        hidden_size=328, num_attention_heads=12, head_dim=64,
        num_hidden_layers=8, intermediate_size=3072,
        num_key_value_heads=12, max_position_embeddings=64,
    )
    test_model = LlamaModel(test_cfg)
    n_params = sum(p.numel() for p in test_model.parameters())
    print(f"Sanity 1/2: LlamaModel built with hidden=328, heads=12, head_dim=64 "
          f"({n_params:,} params)")
    test_in = torch.randn(1, 8, 328)
    test_mask = torch.ones((1, 1, 8, 8), dtype=torch.float)
    try:
        _ = test_model.forward(inputs_embeds=test_in, attention_mask=test_mask,
                                is_causal=False)
        print("Sanity 2/2: LlamaModel.forward(is_causal=False) accepted (via flash_attn_kwargs)")
    except TypeError as e:
        print(f"FATAL: forward(is_causal=False) rejected: {e}")
        return {"error": "transformers does not support is_causal pass-through; pin different version"}
    del test_model

    # Locate cached ST-SE-Replogle checkpoint (downloaded by prior script)
    st_root = Path("/data/state_cache/models--arcinstitute--ST-SE-Replogle/snapshots")
    if not st_root.exists():
        # Download fresh
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            "arcinstitute/ST-SE-Replogle",
            cache_dir="/data/state_cache",
            allow_patterns=["zeroshot/k562/**", "*.json", "*.yaml"],
        )
        print(f"Downloaded to: {local}")
        st_root = Path(local).parent.parent
    st_dir = next(st_root.iterdir())
    print(f"ST-SE-Replogle dir: {st_dir}")

    k562_zs_dir = st_dir / "zeroshot" / "k562"
    if not k562_zs_dir.exists():
        return {"error": f"no zeroshot/k562 in {st_dir}",
                "files": [str(p) for p in st_dir.iterdir()]}

    print(f"Model dir: {k562_zs_dir}")
    print(f"  contents: {os.listdir(k562_zs_dir)}")

    # Find the checkpoint file
    ckpt_candidates = []
    for name in ("eval_best.ckpt", "eval_last.ckpt"):
        p = k562_zs_dir / name
        if p.exists():
            if p.is_file():
                ckpt_candidates.append(p)
            else:
                # Lightning checkpoint directory
                for inner in p.rglob("*"):
                    if inner.is_file() and inner.suffix in (".bin", ".pt", ".safetensors", ".pth"):
                        ckpt_candidates.append(inner)
                        break
    if not ckpt_candidates:
        # Look in checkpoints/ subdir
        ck_dir = k562_zs_dir / "checkpoints"
        if ck_dir.exists():
            for p in ck_dir.iterdir():
                if p.is_file() and p.suffix in (".ckpt", ".pt", ".bin"):
                    ckpt_candidates.append(p)
    if not ckpt_candidates:
        return {"error": "no checkpoint found",
                "files": [str(p) for p in k562_zs_dir.rglob("*")][:30]}
    ckpt = ckpt_candidates[0]
    print(f"Using checkpoint: {ckpt} ({ckpt.stat().st_size / 1e6:.1f} MB)")

    # Preprocess K562 N=200 subset (matches our other evals)
    print("\nLoading K562 ...")
    adata = ad.read_h5ad("/data/raw/k562_subset_n200_slim.h5ad")
    adata.var_names = [str(g) for g in adata.var_names]
    print(f"K562 subset: {adata.shape}")

    # Build X_hvg embedding (STATE expects this)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=min(200, adata.n_vars - 1),
                                     flavor="seurat_v3", inplace=True)
        hvg = adata.var.get("highly_variable")
        if hvg is not None:
            hvg_idx = np.where(hvg.values)[0]
            X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
            adata.obsm["X_hvg"] = X[:, hvg_idx].astype(np.float32)
        else:
            X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
            adata.obsm["X_hvg"] = X.astype(np.float32)
    except Exception as exc:
        print(f"HVG failed ({exc}); using full X")
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        adata.obsm["X_hvg"] = X.astype(np.float32)
    print(f"X_hvg shape: {adata.obsm['X_hvg'].shape}")

    # Subset adata before SE-600M embedding. 7th attempt confirmed CPU
    # embedding is ~28 sec/cell, so full 38k cells is infeasible (12 days).
    # For STATE GIES eval we only need one predicted profile per perturbation
    # plus enough controls for overlay sampling -> a few hundred cells suffices.
    print("\nSubsetting cells for tractable CPU embedding...")
    pert_col = "gene"
    rng_sub = np.random.default_rng(seed)
    is_ctrl_mask = (adata.obs[pert_col].values == "non-targeting")
    ctrl_idx_all = np.where(is_ctrl_mask)[0]
    pert_idx_all = np.where(~is_ctrl_mask)[0]
    n_ctrl_keep = min(n_ctrl_keep, len(ctrl_idx_all))
    ctrl_keep = rng_sub.choice(ctrl_idx_all, size=n_ctrl_keep, replace=False)
    # n_cells_per_pert cells per perturbation. seed=42/n_cells_per_pert=1 ->
    # the original ~320-cell run; multi-seed uses n_cells_per_pert=5,
    # n_ctrl_keep=200 -> ~1200 cells (V4 Phase 3, ~9.3h CPU per seed).
    pert_labels = adata.obs[pert_col].values
    pert_keep = []
    for g in sorted(set(pert_labels[pert_idx_all])):
        gidx = np.where(pert_labels == g)[0]
        pert_keep.extend(rng_sub.choice(gidx, size=min(n_cells_per_pert, len(gidx)), replace=False).tolist())
    keep = np.concatenate([ctrl_keep, np.array(pert_keep, dtype=int)])
    adata = adata[keep].copy()
    print(f"  subset: {adata.shape}  (controls={n_ctrl_keep}, perturbed={len(pert_keep)})")

    pre = "/tmp/k562_preprocessed.h5ad"
    adata.write_h5ad(pre)

    # STAGE 1: SE-600M embedding step. The ST-SE-Replogle checkpoint expects
    # 2058-dim X_state embeddings as input (not raw expression / X_hvg).
    # Locate or download SE-600M.
    from huggingface_hub import snapshot_download
    se_root = Path("/data/state_cache/models--arcinstitute--SE-600M/snapshots")
    if se_root.exists() and any(se_root.iterdir()):
        se_dir = next(se_root.iterdir())
    else:
        local = snapshot_download(
            "arcinstitute/SE-600M",
            cache_dir="/data/state_cache",
        )
        se_dir = Path(local)
    print(f"\nSE-600M dir: {se_dir}")
    print(f"  contents: {os.listdir(se_dir)}")

    # Memory diagnostic
    import resource, gc
    def mem_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    pre_emb = "/tmp/k562_with_xstate.h5ad"
    print(f"\n[Stage 1/2] Memory before SE-600M load: {mem_mb():.0f} MB")

    # Stream subprocess output line-by-line so Modal sees progress in real time.
    # The previous capture_output=True version buffered everything in memory and
    # if the container was OOM-killed mid-run, all output was lost.
    cmd_emb = ["state", "emb", "transform",
               "--model-folder", str(se_dir),
               "--input", pre,
               "--output", pre_emb,
               "--embed-key", "X_state",
               "--batch-size", "8"]   # smaller batch to reduce peak memory
    print(f"  cmd: {' '.join(cmd_emb)}")

    import subprocess as sp
    proc = sp.Popen(cmd_emb, stdout=sp.PIPE, stderr=sp.STDOUT,
                    text=True, bufsize=1)
    last_mem_log = time.time()
    for line in iter(proc.stdout.readline, ""):
        print(f"    [emb] {line.rstrip()}", flush=True)
        if time.time() - last_mem_log > 60:
            print(f"    [emb-mem] {mem_mb():.0f} MB", flush=True)
            last_mem_log = time.time()
    proc.wait()
    print(f"  rc={proc.returncode}, peak mem={mem_mb():.0f} MB")
    if proc.returncode != 0:
        return {"error": "state emb transform failed", "rc": proc.returncode}

    gc.collect()

    if not os.path.exists(pre_emb):
        return {"error": f"embedding step did not produce {pre_emb}"}
    print(f"  embedded h5ad: {os.path.getsize(pre_emb) / 1e6:.1f} MB")

    # STAGE 2: STATE inference using the X_state embeddings.
    out_path = f"/data/results/model_outputs_real_n200_state/state_predictions_seed{seed}.h5ad"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd_infer = ["state", "tx", "infer",
                 "--model-dir", str(k562_zs_dir),
                 "--checkpoint", str(ckpt),
                 "--adata", pre_emb,
                 "--pert-col", "gene",
                 "--embed-key", "X_state",
                 "--output", out_path]
    print(f"\n[Stage 2/2] Memory before inference: {mem_mb():.0f} MB")
    print(f"  cmd: {' '.join(cmd_infer)}")

    proc2 = sp.Popen(cmd_infer, stdout=sp.PIPE, stderr=sp.STDOUT,
                     text=True, bufsize=1)
    last_mem_log = time.time()
    for line in iter(proc2.stdout.readline, ""):
        print(f"    [infer] {line.rstrip()}", flush=True)
        if time.time() - last_mem_log > 60:
            print(f"    [infer-mem] {mem_mb():.0f} MB", flush=True)
            last_mem_log = time.time()
    proc2.wait()
    print(f"  rc={proc2.returncode}, peak mem={mem_mb():.0f} MB")
    if proc2.returncode == 0 and os.path.exists(out_path):
        sz = os.path.getsize(out_path)
        return {"success": True, "out_path": out_path, "size_mb": sz / 1e6}
    return {"error": "state tx infer failed", "rc": proc2.returncode}


@app.function(
    image=image,
    cpu=16,
    timeout=21600,  # 6 hours; must be >= run_state_infer's timeout, since
                    # .local() runs inside this container.
    volumes={"/data": vol},
    memory=131072,
)
def chained_pipeline(n_cells_per_pert: int = 1, n_ctrl_keep: int = 120, seed: int = 42) -> dict:
    """Detach-safe wrapper."""
    return run_state_infer.local(n_cells_per_pert=n_cells_per_pert,
                                 n_ctrl_keep=n_ctrl_keep, seed=seed)


@app.local_entrypoint()
def main(n_cells_per_pert: int = 1, n_ctrl_keep: int = 120, seed: int = 42):
    handle = chained_pipeline.spawn(n_cells_per_pert=n_cells_per_pert,
                                    n_ctrl_keep=n_ctrl_keep, seed=seed)
    print(f"Spawned function call: {handle.object_id} "
          f"(seed={seed}, n_cells_per_pert={n_cells_per_pert}, n_ctrl_keep={n_ctrl_keep})")
    print("STATE inference will run on Modal CPU regardless of local CLI state.")
