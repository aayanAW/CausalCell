# Pre-registration — B2: OT-objective confirmatory + dose-response (the stronger claim)

**Date:** 2026-07-05 · **Base:** branch `fix/camera-ready-corrections`, GPU box
`ssh:research-box` (AMD MI300, ROCm). Follows the marginal 3-seed KEEP for
Sinkhorn-OT (`b_gpu_definitive.json`).

## Motivation

The definitive 3-seed GPU test gave Sinkhorn-OT a KEEP that only just cleared the
run-to-run SD (+0.0038, 3/3 seeds). Sinkhorn λ=1.0 was pre-selected as the winner
TWICE independently (the no-GPU proxy's validation pass, then the 3-seed GPU run),
so it is now a PRE-SPECIFIED hypothesis, not a selected one. This experiment powers
it up.

## Primary (confirmatory) hypothesis — PRE-SPECIFIED, no selection

Sinkhorn-OT at **λ=1.0** improves held-out causal F1 over recon-only (λ=0) when a
CPA-style conditional VAE is trained on real K562 and scored via overlay→GIES vs the
A1 perturbation-anchored GT.

- **Seeds:** 42, 123, 456, 789, 1024 (5, up from 3).
- **Keep/discard gate (locked):** KEEP iff Sinkhorn λ=1.0 beats recon on **≥4/5 seeds
  AND mean gain > run-to-run SD**. This is the single primary comparison; because
  λ=1.0 was pre-specified, there is no multiple-comparisons correction on it.

## Secondary (exploratory) — dose-response, clearly labeled

λ ∈ {0, 0.3, 1.0, 3.0, 10.0} for the Sinkhorn term, 5 seeds. Reported as a
characterization curve only; no keep/discard verdict is drawn from the exploratory
λ values, and their multiple-comparison exposure (4 non-primary λ × 5 seeds) is
stated. MMD and precision-support are NOT re-run (already discarded at 3 seeds).

## Frozen (everything except seed and λ)

CVAE architecture (enc 256-256 → latent 64 → +pert-embedding → dec 256-256-200),
epochs 150, lr 1e-3, Adam, KLD weight 1e-3, overlay N_cells=100/N_ctrl=200,
GIES partition 20, anchored GT primary key q0.05/fc0.5, 183 perturbations.

## Released-CPA arm (attempted, honestly gated)

We additionally ATTEMPT to retrain the released CPA (scvi-tools / cpa-tools) on the
GPU box with the Sinkhorn-OT objective, as a check that the effect is not an artifact
of the from-scratch VAE. This is ROCm-compatibility-gated: if the install/run fails
within a reasonable time-box, it is reported as "attempted, blocked by ROCm
compatibility," NOT silently dropped, and the from-scratch-VAE result stands as the
primary Track-B evidence with that stated caveat.

## Reporting

Report per-seed F1 for every (λ, seed), the primary λ=1.0-vs-recon gate result, the
dose-response curve, the exploratory-trial count, and the released-CPA arm outcome
(success or the specific blocker). Nulls/marginals reported straight.
Decision ledger: `results/phaseB/b2_ot_decision_ledger.jsonl`.

## Released-CPA arm — OUTCOME (2026-07-05): attempted, blocked at install

`pip install cpa-tools` in the ROCm container FAILS: its build chain pins an old
PyYAML that cannot build a wheel against the container's modern setuptools
("Failed to build 'PyYAML' when getting requirements to build wheel"). `scvi-tools`
1.4.3 (CPA's backbone) DOES install and import cleanly on ROCm. Because the from-
scratch conditional VAE used for the primary arm already replicates the CPA design
(shared encoder → per-perturbation additive latent shift → shared decoder), the
released-CPA retrain is reported as **attempted, blocked by cpa-tools packaging
incompatibility on ROCm**; the from-scratch-VAE result stands as the primary Track-B
evidence with this caveat, exactly as this pre-registration specified. A follow-up on
a CUDA box (where cpa-tools wheels are routinely built) is the clean path to close it.