# HANDOFF — PerturbCausal (2026-07-03)

Everything needed to continue this project in a fresh session.

---

## 1. TL;DR — current state

**PerturbCausal** is a benchmark testing whether virtual-cell-model (VCM) predicted Perturb-seq can substitute for real Perturb-seq in downstream causal gene-regulatory-network (GRN) recovery.

- **Core finding:** No deep VCM (GEARS, CPA, Geneformer, STATE) beats a sparse linear baseline (ElasticNet) on causal-recovery F1, across K562/RPE1/Norman × GIES/PC/NOTEARS/inspre. None reaches real-data quality. The failure is in the **joint (covariance / conditional-independence) structure**, not the per-gene means — post-hoc calibration fixes the marginal but not the causal gap.
- **Status:** **Accepted to the ICML 2026 FAGEN workshop** (Failure Modes in Agentic AI — takes rigorous negative-results / failure analyses). Camera-ready is built and compliant.
- **Honest self-assessment:** solid, honest, *modest* negative-results benchmark + mechanism diagnosis (~7/10 novelty). It confirms the field's "VCMs underperform" direction but on a genuinely new axis (causal recovery ≠ prediction) with a mechanism. To become high-impact it needs the extensions in §5.

---

## 2. What the project is (1-paragraph)

VCMs are marketed as in-silico replacements for million-dollar CRISPR screens. Prior work (Ahlmann-Eltze & Huber 2025, Nat Methods) showed they don't beat linear baselines at **per-gene prediction**. This project asks the downstream question that actually matters for biology: can their predicted data substitute for real data when building a **causal regulatory network**? Method: hold the causal-discovery algorithm fixed, vary only the data source (real vs each model's predictions), compare recovered edge sets. Answer: no — and the mechanism is joint-structure distortion driven by the per-gene training objective.

---

## 3. Where things stand

### Git
- Branch: **`fix/camera-ready-corrections`** (off `main`). Root repo: `/Users/aayanalwani/virtual cells`.
- Remote `origin` → the CausalCell repo. Commits use conventional prefixes, **NO Co-Authored-By / AI attribution**.
- **submissions/ and HANDOFF_* are gitignored but TRACKED where already added** — new files there need `git add -f`.

### Submission artifacts (submissions/icml/)
| File | Venue | State |
|---|---|---|
| `CausalCellBench_FAGEN_CameraReady.tex` / `.pdf` | **FAGEN (accepted)** | camera-ready, `[accepted]`, authors **Aayan Alwani + Ethan Wang** (Independent Researcher), **7-page body** (limit 8 excl refs/appendix), 3 full-width `table*` + 3 inline figures, verified clean |
| `CausalCellBench_FAGEN.tex` / `.pdf` | FAGEN | anonymous review version |
| `CausalCellBench_FM4LS_CameraReady.tex` / `.pdf` | FM4LS | 4–5pp camera-ready variant (backup venue) |
| `CausalCellBench_ICML.tex` / `.pdf` | (source) | full 14-page-body manuscript — the master; all others derive from it |

FAGEN page rule: **max 8 pages excluding references + appendix**; camera-ready = `[accepted]` + revealed authors.

### Key package / harness
- Package: `causalcellbench/` (data loaders, calibration/, causal/ engines, eval/ metrics, models/ wrappers).
- Eval harness: `scripts/run_eval_local.py` (overlay sampling → GIES; also PC/NOTEARS). Cross-dataset backends: `scripts/v4/phase7_multibackend_crossdataset.py`.
- Results: `results/*.json` (every manuscript number traces here).

---

## 4. This session's work (already done + committed to the branch)

- **Number-integrity audit + corrections** (9-dim multi-agent area-chair audit → `AUDIT_10of10_ROADMAP.md`): fixed fabricated/stale numbers (GRNBoost2 edges, Table 1 edges column), inverted RPE1 bootstrap means → true point estimates, deprecated an invalid pairwise permutation test, cosine-direction fix, etc.
- **J7 identifiability proposition + TOST equivalence test + per-seed strip plot** added to the manuscript.
- **J1 (TRRUST external validation):** multi-seed → **NULL** (0.83 ± 0.70× chance; the single-seed 2.4× did NOT replicate). Circularity remains an OPEN limitation. `scripts/j1_multiseed.py`, `results/track_b/j1_trrust_multiseed.json`.
- **J2 (covariance-regularizer λ-sweep):** pre-registered (`preregistration_j2_lambda.md`), val/test split, locked test → **DISCARD** (best-val λ=100 gains only +0.003 ± 0.035; the +0.111 J2 gain is the *learned joint map*, not the covariance penalty). `scripts/j2_lambda_sweep.py`, `results/phase4/j2_lambda_sweep.json`.
- **FAGEN + FM4LS camera-ready builds** produced and verified (page limits, overlap, numbers, self-checked).

---

## 5. NEXT STEPS — the extension plan (this is the roadmap)

Two independent problems; the strongest paper does BOTH, **hardening first** (see `AUDIT_10of10_ROADMAP.md` for full costing).

### A. HARDENING (safe, near-guaranteed win — do first; SAME expanded paper)
Fixes the #1 reviewer objection (circular ground truth). No model training.
1. **Perturbation-anchored ground truth** *(highest priority, cheap, uses existing data)*: build ground-truth edges directly from real perturbation effects (knock out A → B changes ⇒ real causal edge A→B), instead of GIES-on-real. Breaks the circularity because truth comes from the interventions, not an algorithm. Then score predicted data against these real edges.
2. **External K562 GRN** (Gasperini 2019 CRISPRi enhancer screen / ENCODE-rE2G) — re-pick the gene subset to maximize overlap (the J1 null happened because the essential-gene subset had ~0 curated edges).
3. **Combination perturbations** (re-include the ~135 dropped Norman doubles as held-out) + **genome-scale** (N=500→1000+ via inspre) — widen conditions; combinations are where a linear baseline *can't* extrapolate, so the most likely place to find a *surprising positive*.

### B. THE FIX (high-risk, high-reward — the "tool"; likely a SEPARATE new paper, or the climax of the expanded one if it works)
Turn the diagnosis into a cure: retrain a small VCM (**CPA**, ~45 min, scvi-tools) with a **joint-structure-preserving training objective**, evaluated against the *hardened* ground truth from A.
- The naive version (raw covariance Frobenius penalty) already **failed** (J2 null). The untried, better-motivated objective is **conditional-independence / precision-matrix (graphical-lasso-style) aware**, or **distribution matching** (optimal transport / MMD / adversarial).
- **Honest odds: ~25–40%** it cleanly beats the linear baseline. It's a real bet. Frame it as *"we test whether a training-time intervention closes the gap"* so BOTH outcomes are publishable.
- Base = deep learning (conditional VAE = CPA); the novelty is the **loss**, blending DL with graphical-model / OT ideas. Stack: PyTorch + scvi-tools + the existing eval harness.

### Why the fix is *possible* despite billion-dollar labs "failing"
They optimized for **prediction**, never for joint structure — you can't fail a target you don't aim at. The paper *proved* scale/architecture didn't matter (identical inflation across GNN/VAE/104M transformer) → it's a **wrong-objective** problem, not a compute problem. A small model with the *right* objective can beat a huge one with the wrong objective. Your edge = the diagnosis, not compute.

**Venue for the expanded version:** journal / main conference (NeurIPS/ICML/ICLR or Nature-family) — the complete "found it → explained it → fixed it → validated it" paper needs the room. FAGEN/FM4LS are non-archival, so expansion + resubmission elsewhere is allowed.

---

## 6. Technical details / gotchas (do NOT relearn)

- **Eval env:** Python 3.9 at `.venv_gears`. Use the ABSOLUTE path: `"/Users/aayanalwani/virtual cells/.venv_gears/bin/python"` (bare relative fails when cwd is `submissions/icml/`).
- **R + inspre** installed locally (`/usr/local/bin/Rscript`). Tectonic for PDF builds (`cd submissions/icml && tectonic <file>.tex`).
- **macOS:** no GNU `timeout`/`setsid`/`xargs -r`. Use `nohup` + `caffeinate`. Run long jobs tracked in background (harness `run_in_background`), not detached via `nohup &` (harness can't notify on the latter).
- **Tables:** `tab:main` and `tab:backend` must be `\begin{table*}` (full-width) — downgrading to single-column `table` causes text OVERLAP (was a real bug).
- **ICML `[accepted]` mode requires `\icmlcorrespondingauthor`** or it prints "AUTHORERR" in the footnote.
- **Every manuscript number must trace to a `results/*.json`.** Report multi-seed with CIs; single-seed = "suggestive," not significant. Report nulls straight — no inflation.
- **Self-check agents mandatory** after non-trivial code/number edits (citation-honesty + reviewer-2, Opus). This session's self-checks caught real defects (duplicate figure, dangling ref, 2.4× vs 1.84× mismatch).
- **PDF preview for the user:** serve via `python -m http.server 8742 --bind 127.0.0.1` from `submissions/icml/` (they open `http://localhost:8742/<file>.pdf`), or `open <file>.pdf`. The claude-app inline attachment tool was intermittently disabled.

---

## 7. How to resume

1. Read `~/.claude/CLAUDE.md` (autonomous mode, research-grade honesty, self-check agents, novelty protocol).
2. Read this handoff + `AUDIT_10of10_ROADMAP.md` (full flaw list + costed extension plan) + `V4_EXECUTION_STATUS.md` (older execution history).
3. Confirm branch `fix/camera-ready-corrections`; `git add -f` the submission files if continuing paper work.
4. Pick the track: **hardening (A) for a safe stronger paper**, or **the fix (B) for the high-risk tool**. Recommended: A1 (perturbation-anchored ground truth) first — cheapest, biggest credibility win, and it's the right evaluation target for B.

## 8. Reference files
- `AUDIT_10of10_ROADMAP.md` — audit findings + camera-ready vs journal bucket + J1/J2 outcomes.
- `CausalCellBench_Explanation.md` — plain-English project explanation.
- `submissions/icml/CausalCellBench_ICML.tex` — master manuscript (full body + appendix + bib).
- `preregistration_j2_lambda.md` — the J2 pre-registration (template for future keep/discard experiments).
- Motivating lit: Ahlmann-Eltze & Huber 2025 (Nat Methods); CausalBench/Chevalley 2025; Mejia 2025 (mode collapse, arXiv 2506.22641); Kernfeld 2024 (transcriptome insufficient for FDR); Gasperini 2019 (K562 enhancer screen — for hardening A2).
