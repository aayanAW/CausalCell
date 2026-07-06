# The Wall-1 Invariance Lemma (formal statement + numerical verification)

**Verified numerically against the actual `sample_perturbed_cells_overlay`** (not a reimplementation),
`scripts/verify_wall1_invariance.py` → `results/hardening/wall1_invariance.json`,
`figures/wall1_invariance.png`. K562 panel, 200 genes, 400 cells/condition, seed 42.

## Setup
The benchmark evaluates a model prediction through the mean-overlay sampler. For a mean prediction
vector `μ ∈ ℝ^G`, the sampler forms per-gene ratios `r_g = clip(μ_g / c_g, 0.01, 10)` (with `c` the
control mean), bootstraps control cells `x`, and returns

  `y = Poisson( diag(r) · x )`,   i.e. `y_g = Poisson(r_g x_g)`.

Let `D = diag(r)`, a positive diagonal matrix. In the deterministic (noise-free) limit `y = D x`.

## Lemma (mean-only dependence of the causal metric)
Let a causal-discovery algorithm depend on the data only through the **support of the precision
matrix** `supp(Σ⁻¹)` (equivalently, the pattern of nonzero partial correlations) — true for GIES, PC,
and inspre. Then:

**(i) Exact, deterministic.** For the deterministic overlay `y = D x`, the sample correlation matrix
and `supp(Σ_y⁻¹)` are **invariant to `D`**. Proof: `Σ_y = D Σ_x D`; `corr(Σ_y) = corr(Σ_x)` because a
positive diagonal congruence divides out in the correlation normalization; and `Σ_y⁻¹ = D⁻¹ Σ_x⁻¹ D⁻¹`
is `Σ_x⁻¹` scaled by a positive diagonal, so its zero pattern — the support — is unchanged. ∎

**(ii) Corollary — the metric is a function of `μ` only.** Because `D` is the *only* place the
prediction enters the deterministic pipeline, the recovered CPDAG depends on `μ` **only through the
clipped ratio vector `r`**. Any two predictions with the same `r` yield the same graph; in particular
the metric is **blind to the prediction's joint (covariance) structure** — a model may reshape its
predicted gene-gene covariance arbitrarily without changing a single recovered edge.

**(iii) With Poisson noise — approximate, count-dependent.** Once the mean-dependent Poisson step is
added (`var = mean`), `D` perturbs off-diagonal sample correlations gene-by-gene. This residual is
**not zero**; it decreases as sequencing depth grows and vanishes as counts→∞. Crucially, even with
noise the pipeline still depends on the prediction **only through `μ`** — the noise injects no
cross-gene structure from the prediction (Poisson is per-gene independent).

## Numerical verification (from the real function)

| Test | Result | Meaning |
|------|--------|---------|
| **(i) deterministic**: two very different means `μ₁,μ₂` (14× different fold-changes), overlay with Poisson disabled | correlation `max\|Δ\|` = **2.07e-08** (float32 storage epsilon; the function casts to float32); precision-support **Jaccard = 1.000** (21,926 edges each, identical) | Deterministic invariance is exact in support and correlation-to-storage-precision. |
| **(iii) Poisson sweep** over UMI depth ×{0.25…8} | edge-flip fraction **0.496 → 0.347** as mean counts rise 582 → 18,552; `max\|Δcorr\|` 0.377 → 0.154 | Residual is **substantial at realistic depth** and shrinks with depth — honest: NOT machine-zero once noise is on. |
| **(Track-B kill) same mean, different joint** — two predicted-cell distributions with identical per-gene mean (max diff **6.8e-13**) but **14.8× different** off-diagonal correlation, each reduced to its mean and overlaid (the mean-path the pipeline uses) | overlay support **Jaccard = 1.000**, `max\|Δcorr\| = 0.0` | A model that reshapes joint structure while holding means fixed produces an **identical** metric. This is why every Track-B objective nulled. |

## Scope (honest boundaries)
- The invariance is a property of **this mean-overlay evaluation path**, not of causal recovery in
  general. It binds Track-B (a mean-predictor trained with a joint-structure loss) and any
  mean-output VCM under this metric.
- The **only** eval that bypasses `D` is feeding real/predicted **cells** directly (`output_type=cells`).
  That path was tested (the instrument-fix) and is **null** on RPE1+Norman — so the bypass does not
  rescue joint structure either. This is the empirical answer to "just evaluate cells," and it is a
  *separate* limitation (Wall 2), not covered by this lemma.
- Part (iii)'s non-negligible finite-depth residual means the manuscript must NOT claim "exactly
  invariant" unqualified; the correct claim is **"depends on the prediction only through per-gene
  means"** (exact) plus **"support-invariant to the diagonal rescale, up to a count-dependent Poisson
  residual"** (finite-sample).

## One-line statement for the manuscript
> Under the mean-overlay evaluation, a prediction enters causal discovery only as a per-gene diagonal
> rescale of control cells; the recovered graph is therefore a function of the predicted means alone
> and is invariant to the prediction's joint structure (exactly so for the deterministic rescale;
> up to a depth-dependent Poisson residual with counts). A model cannot improve this metric by
> reshaping covariance — verified against the evaluation code, including two predictions with identical
> means and 15× different joint structure that yield identical recovered edges.
