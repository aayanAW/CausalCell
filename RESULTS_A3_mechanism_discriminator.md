# A3 mechanism discriminator — result (feasibility experiment)

**Pre-registered** in `prereg_A3_mechanism_discriminator.md`. Gate script:
`scripts/a3_mechanism_discriminator.py` (its own test suite; all sanity tests passed). Result:
`results/hardening/a3_mechanism_discriminator.json`. Figure: `figures/a3_mechanism_discriminator.png`.
Norman held-out doubles, 71-gene panel, 129 doubles, anchored GT = 2,092 edges, seeds
42/123/456/789/1024.

## Verdict: KILL — A3 is NOT a marginal-mean (non-additivity) result

| Condition | What it carries | anchored-GT F1 (mean ± sd) | per-seed |
|-----------|-----------------|----------------------------|----------|
| **A. real_doubles** (cells) | real covariance + real means | **0.0552 ± 0.0056** | 0.057, 0.049, 0.052, 0.055, 0.063 |
| **B. real_means_overlay** | real non-additive means, covariance destroyed | **0.0345 ± 0.0039** | 0.035, 0.034, 0.031, 0.032, 0.041 |
| **C. additive_singles** (overlay) | additive-of-singles means | **0.0359 ± 0.0037** | 0.032, 0.038, 0.036, 0.033, 0.041 |

**Primary contrast B vs C** (isolates the marginal-mean channel — both go through the identical
overlay): **−0.0014 ± 0.0028, 1/5 seeds. CONFIRM = false.** Real doubles' *non-additive means* give
essentially **no** causal-recovery advantage over additive-of-singles means.

**Secondary contrast A vs B** (isolates the cell-path joint structure): **+0.0207 ± 0.0036, 5/5
seeds.** The **entire** A3 gap comes from the joint (covariance) structure that real *cells* carry and
that is destroyed when you keep only their means.

**Sanity tests passed:** C reproduces the locked A3-additive number (0.036) and A reproduces the locked
A3-real number (0.055) — the reused machinery is unchanged, so the split is a real decomposition, not a
harness artifact.

## Why this is the honest, and stronger, reading

The prior handoff framed A3 as the project's "one positive — a regime where VCMs could help." The
discriminator overturns that framing and, importantly, makes the whole paper **more coherent**:

1. A3's real-vs-additive advantage is **not** in the means. It is in real-cell **covariance** — the
   `output_type=cells` path, which bypasses the overlay and keeps gene-gene structure.
2. That covariance channel is **exactly** the one the mean-overlay evaluation is blind to (Wall-1
   lemma) **and** the one no mean-output VCM produces (all K562 VCM predictions are (200,200) means).
3. So A3 does not identify a regime where VCMs help; it identifies, one more time, that the recoverable
   causal signal lives in a channel that (a) VCMs don't emit and (b) the metric can't read. It
   **reinforces the negative** rather than qualifying it.

## What A3 becomes in the manuscript

Not "when do VCMs help." Instead: *a positive control that localizes the signal.* Feeding real
double-perturbation cells (with their covariance) recovers measurably more anchored interventional
structure than any mean-based predictor — additive or real-mean alike — establishing that (i) the
anchored GT is recoverable in principle from the right data channel, and (ii) that channel is joint
cell-level structure, not per-gene means. This is the constructive complement to Wall-1: it shows the
missing information is real and locates it precisely where the mean-overlay metric and mean-output
models both cannot reach.

## Consistency with the rest of the project
- Coheres with **Wall-1** (metric is mean-only): B (real means) ≈ C (additive means) because the metric
  sees only means, and real vs additive means barely differ *for edge recovery*.
- Coheres with the **instrument-fix** (Wall-2): even the real-cell channel (A) recovers only F1≈0.055 —
  above the mean conditions but still low in absolute terms, consistent with the task being
  under-determined even from real cells.
- Coheres with **Track-B / B2 nulls**: a mean-output model retrained with a joint-structure loss cannot
  reach A's advantage, because its output collapses to a mean at overlay time (Wall-1 Part 3).

**Ledger:** A3 verdict updated from "POSITIVE — VCMs help" to "KILL as a mean/VCM positive; RECAST as a
signal-localization positive control (advantage is real-cell covariance, 5/5)."
