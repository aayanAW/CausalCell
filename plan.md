# Plan — phased, gated (Phase C)

All model/training-side candidates are **killed by Wall 1** (see architecture.md); no compute is
spent searching for a positive that the metric provably cannot see. The plan builds only changes with
a mechanism to move — or correctly interpret — the reported result.

## Phase 1 — Feasibility gate (runs FIRST, per D-Build Step 1)
- **A3 mechanism discriminator** (feasibility.md). CONFIRM → A3 stays as the constructive
  non-additivity result in the reframe. KILL → A3 leaves the positive column. AMBIGUOUS → downgraded
  to "partial non-additive structure."
- Go/no-go recorded in `decision-ledger.jsonl`.

## Phase 2 — Wall-1 invariance lemma (parallel; no dependency on Phase 1)
- Formal statement + corollary (CPDAG is a function of `mean_pred` support only, invariant to joint
  reshaping).
- **Numerical verification against the actual `sample_perturbed_cells_overlay`** — two divergent mean
  predictions → correlation `max|Δ|` at machine zero on the deterministic rescale; precision-support
  Jaccard = 1.0; Poisson residual quantified separately.
- Deliverables: lemma text block, `verify_wall1_invariance.py` (green gate script), verification figure.

## Phase 3 — Manuscript reframe patch
- Fold the Wall-1 lemma into the declared theorem environment; state Wall 2 as **empirical** (list the
  instrument-fix null + genome-scale evidence, do NOT call it a theorem); insert A3 with the Phase-1
  verdict. Lead with the methodology contribution (evaluation under-determination), demote the model
  ranking. Produced as a diff/patch of `CausalCellBench_ICML.tex`, not an overwrite.

## Phase 4 — Deferred / off critical path
- **Wall-2 inspre-ADMM confirmation:** robustness of the genome-scale negative under full ADMM (local
  used closed-form). Needs R + free CPU (the GPU box has no R and is 2× oversubscribed). Confirmation of
  a negative — scheduled only if a reviewer demands it. Recorded as DEFER in the ledger.
- **Causally-structured predictor:** only meaningful paired with an eval that bypasses the overlay; the
  bypass (instrument-fix) is already null (Wall 2). Not scheduled; recorded as KILL-under-current-eval.

## Gates & discipline
- Pre-registered multi-seed keep/discard (seeds 42/123/456/789/1024; KEEP iff ≥4/5 and gain > run SD).
  Do not relax to rescue a result — this discipline correctly nulled J1/J2/B/B2/instrument/genome-scale.
- Every phase appends its go/no-go to `decision-ledger.jsonl`.
- Git-first: existing untracked hardening work (A1/A2/A3/B/instrument/genome-scale scripts, figures,
  results, preregs — ~29 files) is committed as the FIRST action of Phase D Step 0 before new code; the
  working tree is NOT clean at the start (corrected from an earlier misstatement).

## Compute / fit
- No training on the critical path → 192 GB VRAM ceiling not binding. A3 discriminator and lemma
  verification are CPU-only in `.venv_gears` (Py3.9; gies/sklearn/torch/anndata/scipy). Minutes each.

## Oral odds (honest)
- Lemma: near-certain to hold (verified 3 ways already; formalization is low-risk).
- A3 discriminator: genuinely uncertain — this is why it is the feasibility gate. Prior is ~60% CONFIRM
  (Norman doubles are known to carry genetic interactions), but a marginal-mean artifact is live.
- Reframe lands the paper at the "~7.5 methodology-first" assessment regardless of A3's outcome; A3
  CONFIRM adds the one clean positive.
