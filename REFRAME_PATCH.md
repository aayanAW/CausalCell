# Manuscript reframe patch — CausalCellBench_ICML.tex

Folds the Wall-1 invariance lemma and the A3 signal-localization control into the paper, and corrects
one latent overclaim. All refs resolve; all theorem/proof environments balanced (checked). No local
LaTeX available to compile, but structure verified by ref/env audit. `submissions/` is gitignored, so
the reframed source is delivered as an artifact (`CausalCellBench_ICML_reframed.tex`) rather than a git
commit.

## Change blocks

**1. Abstract (line ~100).** Added two sentences: (a) the harness invariance — recovered graph is
provably a function of predicted means alone, so a mean-output model can't close the gap by reshaping
covariance and a covariance-regularized objective produces the predicted null; (b) the A3 positive
control localizing the missing signal to cell-level joint structure. Added "invariance lemma" to the
released-artifacts list.

**2. New Lemma~\ref{lem:overlay-invariance} + proof (Discussion §identifiability, after
Prop~\ref{prop:ci-invariance}).** Formal statement: under the mean-overlay sampler a prediction enters
discovery only through the per-gene ratio vector; deterministic rescale leaves correlation and
precision-support exactly invariant (positive diagonal congruence). Proof included. Followed by the
numerical verification summary (14× fold-change → Jaccard 1.0; same-mean/14.8×-different-joint →
identical support) and the explicit statement that the lemma **sharpens** Corollary~\ref{cor:calib-kernel}:
the covariance objective that corollary motivates cannot register under a mean-only metric.

**3. Corrected the +0.111 proxy overclaim (Results §calibration, and appendix §j2).** OLD: "the first
direct confirmation that the gap is reachable by a joint operation." NEW: by Lemma, the overlay reads
only the corrected mean, so the +0.111 is a **better-mean-prediction** gain (a non-monotone cross-gene
re-mapping that escapes the per-gene monotone invariance of Prop~\ref{prop:ci-invariance}), NOT evidence
that shaping joint structure helps — consistent with the covariance-Frobenius term itself giving +0.003
(null). This removes a claim that directly contradicted the Track-B null.

**4. New §\ref{sec:a3-localization} "Localizing the recoverable signal: a positive control" (end of
Discussion).** The A3 mean-matched-null result: real doubles' non-additive means (B) ≈ additive means
(C), TOST-certified equivalent (p=0.001); real cells (A) beat B by +0.021, 5/5. Conclusion: the A3
advantage lives in cell-level joint structure, not means — the channel the metric and mean-output VCMs
cannot access. Honest caveat that absolute F1 is near the permutation floor and the claim rests on the
equivalence-vs-gap decomposition.

**5. New appendix §\ref{app:overlay-invariance}.** Full numerical verification of both the lemma
(deterministic float32-exact support; Poisson residual sweep; same-mean/diff-joint) and the A3 control
(sanity checks reproduce locked numbers; TOST; mean-fidelity 1.2% vs 13.9%). Points to released scripts
and JSON.

## Honest Wall-2 status preserved
The reframe does NOT upgrade Wall-2 (real cells also don't recover the anchored graph above chance at
genome scale) to a theorem. Wall-1 is the proved invariance (harness property); Wall-2 remains the
empirical limitation, explicitly retained in §limitations (TRRUST null, technical-duplicate floor,
genome-scale sink). The A3 control shows real cells recover *more* than means, but absolute F1 stays low
— consistent with the task being under-determined even from real cells. No claim that the joint-structure
problem is solved; the open problem is scoped to the evaluation channel.

## Files
- `CausalCellBench_ICML_reframed.tex` — full reframed source (1087 lines)
- `wall1_lemma.md`, `wall1_invariance.png`, `wall1_invariance.json` — the lemma + verification
- `RESULTS_A3_mechanism_discriminator.md`, `a3_mechanism_discriminator.png` — the positive control
- `results/hardening/a3_robustness.json`, `a3_floor.json` — TOST / mean-fidelity / floor numbers
