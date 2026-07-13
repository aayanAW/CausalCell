# Handoff — PLAN→VALIDATE→BUILD hardening pass (2026-07-05)

Executed the "claude ultimate workflow" A→H prompts (PLAN/VALIDATE/BUILD) on the PerturbCausal
hardening. All work committed to `fix/camera-ready-corrections`. Nothing on the critical path needed the
GPU box — by design (see below).

## What changed (one line each)
- **Phase C** — architecture.md / feasibility.md / plan.md written. BUILD: Wall-1 lemma, A3
  discriminator, reframe. KILL: Track-B retrain, causal-architecture. DEFER: inspre-ADMM.
- **Phase B gate** — novelty.md with per-change verdicts + a Claude-only adversarial council (3
  personas). Council forced 3 corrections that shaped the builds (below).
- **Phase D0** — git-first: ~29 untracked hardening files committed in logical units. Large prediction
  checkpoints gitignored (saved as artifacts).
- **Wall-1 lemma** — BUILT + verified against the ACTUAL overlay function. GATE PASS.
- **A3 discriminator** — BUILT + run. Verdict KILL-as-mean-positive, RECAST as signal-localization.
- **Validation** — TOST + mean-fidelity + floor closed all 3 council objections.
- **Reframe** — lemma folded into the .tex, +0.111 overclaim corrected, A3 control added, honest Wall-2
  preserved. Delivered as artifact (submissions/ is gitignored).

## The three results (numbers)

**Wall-1 invariance lemma** (`wall1_lemma.md`, `scripts/verify_wall1_invariance.py`):
- Deterministic rescale: precision-support Jaccard **1.000**, corr max|Δ| **2.1e-08** (float32 storage
  epsilon — the function casts to float32; the old handoff's 1e-15 was a float64 reimplementation).
- Poisson residual: edge-flip **0.496 → 0.347** over depth ×{0.25..8}. Substantial at realistic depth,
  shrinks with depth. Honest: NOT machine-zero once noise is on.
- Same-mean/different-joint: identical means (Δ 6.8e-13), 14.8× different covariance → **identical**
  overlay support (Jaccard 1.0, max|Δcorr| 0.0). The airtight Track-B kill.

**A3 mechanism discriminator** (`RESULTS_A3_mechanism_discriminator.md`,
`scripts/a3_mechanism_discriminator.py`) — anchored GT 2092 edges, 5 seeds:
- A real_doubles (cells) **0.0552**, B real-means-overlay **0.0345**, C additive-overlay **0.0359**.
- **B vs C = −0.0014, TOST-equivalent (p=0.001)** — non-additive MEANS give ~0 recovery advantage.
- **A vs B = +0.0207, 5/5 (p=2e-4)** — the entire A3 gap is real-CELL joint structure.
- Sanity: C reproduces locked A3-additive 0.036, A reproduces locked A3-real 0.055.
- **Recast:** A3 is NOT a "VCMs help" positive. It's a positive control localizing recoverable signal
  to cell-level joint structure = the channel Wall-1 metric and mean-output VCMs both cannot access.
  It REINFORCES the negative.

## Council corrections (all adopted — the honest reframe)
1. Methodologist: "exactly invariant" overclaimed → lemma states exact-deterministic +
   asymptotic-under-noise, with the empirical Poisson sweep.
2. ML reviewer: killing Track-B "by proof" over-reaches; test the bypass path → primary kill reason is
   the 5-seed empirical DISCARD, Wall-1 is the scoped mechanistic explanation; bypass (real cells) is
   separately null (instrument-fix).
3. Biologist: A3's decisive control is a **mean-matched null** → built as condition B
   (real-means-through-overlay). This is what flipped A3 from "positive" to "signal-localization."

## Validation (Phase D3) — 3 objections closed
- **TOST**: B≈C is equivalence-certified (p=0.001; 95% CI [−0.0049,+0.0022] inside ±0.0104 margin), MDE
  at n=5 is 0.0047 < the 0.021 A−B gap → powered, not just non-detected.
- **Mean-fidelity**: B's realized means track A's real means to 1.2% (median 0.35%); real-vs-additive
  means differ 13.9% yet B≈C → metric doesn't convert mean diff to edges.
- **Floor**: label-permutation chance from committed A3 run; primary-GT floor is high (compressed range,
  20-gene partitions cap co-partition recall ~9.6%) → decisive claim rests on within-metric B-vs-C
  contrast, disclosed honestly.

## Manuscript reframe (`CausalCellBench_ICML_reframed.tex`, `REFRAME_PATCH.md`)
- New **Lemma~lem:overlay-invariance** + proof, sharpening the existing prop:ci-invariance from
  per-gene-monotone to **mean-only**.
- **Corrected the +0.111 proxy overclaim**: was "first direct confirmation the gap is reachable by a
  joint operation" → now "a better MEAN prediction from a learned cross-gene re-mapping" (the metric
  reads means; the covariance term itself gave +0.003, the predicted null). This removed a claim that
  contradicted the Track-B null.
- New **§sec:a3-localization** positive control + new **appendix app:overlay-invariance** with full
  numerics. All refs resolve, all envs balanced (no local LaTeX to compile).
- **Wall-2 NOT upgraded to a theorem** — remains the empirical limitation (TRRUST null,
  technical-duplicate floor, genome-scale sink). Explicitly preserved.

## Still open / next
- **inspre-ADMM confirmation** DEFERRED (needs R + free CPU, off critical path).
- **End-to-end GPU CPA retrain** with the released regularizer — the definitive Track-B confirmation.
  Blocked locally on ROCm/cpa-tools packaging; would use `gpu-box-callcommand-relay` skill on
  ssh:research-box (shared box — namespaced workdir, VRAM check, no disturbing other users). It will
  NOT change the conclusion (Wall-1 bounds its upside to the mean channel); it adds one independent
  nail. Low priority.
- **LaTeX compile** of the reframed .tex on a machine with a TeX toolchain (structure verified, but not
  compiled here).

## Ledger
`decision-ledger.jsonl` — 11 entries, every go/no-go with numbers.
