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

## Robustness (Phase-D validate cross-check — closing three council objections)

An adversarial cross-check (statistician / causal-theorist / PI) raised three objections; all now closed:

**1. "B≈C is an underpowered null, not evidence of absence" (statistician).** Replaced the seed-vote
with a **TOST equivalence test**. Equivalence margin = half the A−B gap (±0.0104). Result: B vs C is
**statistically equivalent** (TOST p = 0.0010; 95% CI on B−C = [−0.0049, +0.0022], inside the margin).
The **minimum detectable effect at n=5 is 0.0047**, well below the 0.0207 A−B gap — the design is
powered to detect a mean-channel effect of the size that matters; there is none. A vs B is significant
(paired t = 12.8, p = 0.0002). So "non-additive means give no recovery advantage" is now an
equivalence-certified claim, not an accepted null.

**2. "A−B conflates covariance with residual mean error / higher moments" (causal-theorist).** Verified
that condition B's overlay-realized mean **tracks A's real mean to 1.2% (median 0.35%) relative
deviation** — the clip barely bites, so A−B is not contaminated by mean mismatch. Meanwhile real-double
means differ from additive means by **13.9%** — a large mean difference that nonetheless produces B≈C.
Honest scope correction: B destroys covariance **and** cell-level discreteness/higher moments together,
so the licensed claim is **"the A−B advantage lives in the cell-level joint distribution (covariance +
higher moments), not in the per-gene means"** — not "covariance" exclusively. Wording updated
accordingly below.

**3. "Does A−B clear the noise floor?" (PI/statistician).** Label-permutation chance from the committed
A3 run (`results/hardening/a3_floor.json`; shuffle GT gene labels, rescore, 5 seeds). On the **primary
within-regime GT** (GIES-on-real-doubles), chance is high — additive F1 0.036 vs chance 0.053,
real_doubles 0.055 vs chance 0.080 — a direct consequence of the compressed dynamic range (20-gene
partitions cap co-partition recall at ~9.6%; every F1 in the paper lives in [0, ~0.1]). **This is why
the discriminator's decisive contrast is B-vs-C on the *anchored* GT, not raw F1 vs a permutation
floor.** The equivalence result (TOST p=0.001, an *unconstructed* comparison of two overlay conditions)
does not depend on beating a chance floor — it is a within-metric contrast at matched scale, so the
high permutation chance does not weaken it. Honest disclosure: absolute anchored-F1 values are low and
near their permutation floor; the claim rests on the *relative* B≈C vs A≫B decomposition, not on
absolute recovery.

**Honest residual caveat (PI's circularity point).** Condition B is *designed* to remove joint
structure, so A−B measuring a joint-structure effect is partly confirmatory-by-construction. What the
experiment legitimately establishes is the **direction of the decomposition**: the real-vs-additive A3
gap is NOT in the means (B vs C equivalence, the non-constructed contrast), therefore it must be in the
cell-level channel. The claim is "A3's advantage is not a mean/non-additivity effect," which is carried
by the B-vs-C equivalence — an unconstructed, powered comparison — not by A−B alone.

**Ledger:** A3 verdict updated from "POSITIVE — VCMs help" to "KILL as a mean/VCM positive; RECAST as a
signal-localization positive control — the real-vs-additive advantage is equivalence-certified NOT in
the means (TOST p=0.001), and lives in the cell-level joint distribution (covariance + higher moments),
the channel the overlay metric and mean-output VCMs cannot access."
