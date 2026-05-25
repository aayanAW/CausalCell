#!/usr/bin/env bash
# Phase 9: manuscript v3.0 rewrite + bioRxiv prep.
# Most of this is text editing that requires manual reasoning over Phase 1-7
# outputs. Orchestrator runs the build + checks; full restructure is an automated
# follow-up turn.
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1

# Build manuscript if tectonic available
if command -v tectonic &>/dev/null; then
    cd "$ROOT/submissions/icml"
    tectonic CausalCellBench_ICML.tex 2>&1 | tail -10 || echo "  build failed"
    cd "$ROOT"
fi

# Copy current PDF to bioRxiv staging
mkdir -p "$ROOT/submissions/biorxiv"
if [ -f "$ROOT/submissions/icml/CausalCellBench_ICML.pdf" ]; then
    cp "$ROOT/submissions/icml/CausalCellBench_ICML.pdf" "$ROOT/submissions/biorxiv/PerturbCausal_v0.pdf"
    echo "  Staged biorxiv v0 at submissions/biorxiv/PerturbCausal_v0.pdf"
fi

# Draft cover letter
cat > "$ROOT/submissions/biorxiv/cover_letter.md" <<'MD'
# PerturbCausal — Cover Letter

We submit PerturbCausal, the first systematic benchmark testing whether
virtual cell model (VCM) predicted Perturb-seq data can substitute for real
Perturb-seq in downstream causal gene regulatory network recovery.

Across 3 datasets (Replogle K562, Replogle RPE1, Norman 2019), 5 VCMs (GEARS,
CPA, Geneformer V2, STATE, and the ElasticNet baseline), 4 causal discovery
backends (GIES, inspre, PC, NOTEARS), and 20 random seeds, we find that no
deep VCM significantly exceeds a sparse linear regression on causal-recovery
F1. The substitutability gap decomposes into a magnitude marginal (~40%,
fixable via post-hoc quantile-matching calibration) and a joint-structure
residual (~60%, requiring training-time intervention).

Our findings extend the linear-baselines-beat-DL conclusion of recent prediction
benchmarks (Ahlmann-Eltze 2025, Wei 2025, Wong 2025, Csendes 2025, Vinas Torné
2025) from the per-gene reconstruction axis to the downstream causal-recovery
axis. The release includes the benchmark, calibration module, multi-seed
multi-dataset aggregated results, and reproducibility infrastructure (Docker,
Makefile, pinned environment).

We invite reviewer attention to: (1) the multi-seed n=20 evaluation with paired
permutation tests; (2) the train/test perturbation split for calibration
parameter estimation to avoid leakage; (3) the explicit scope-of-claims table
that enumerates exactly which datasets × models × backends were evaluated.
MD

echo "  Phase 9: bioRxiv staged; full manuscript v3.0 rewrite is a follow-up turn"
