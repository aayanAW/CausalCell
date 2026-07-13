# BUILD: ICML 2026 FM4LS Workshop Poster — "PerturbCausal" (portrait, figure-first)

You are Claude Design. Produce a **single, self-contained, print-ready conference poster** as one HTML/CSS artifact. It must render correctly offline. This is a **reformat** of an existing ISMB poster (`REFERENCE_ISMB_poster.pdf` in this folder) into an **ICML workshop** poster: different size, portrait, far more figure-first, and with the **novel contribution and impact made unmistakable**.

**Asset rule:**

- **Data charts** (bar / forest / dot / line): rebuild as **native inline SVG/CSS** from the locked numbers below. Match the versions in `REFERENCE_ISMB_poster.pdf`. No raster.
- **Illustrative figures**: embed the provided PNGs from `figures/` inline as `data:` URIs. No external URLs, no CDN fonts, everything embedded.

---

## Physical spec (ICML 2026 workshop — HARD CAP)

- **Size: 24 in wide × 36 in tall (61 × 91 cm), PORTRAIT.** This is the ICML workshop maximum and portrait is required. Design portrait from the first pixel; do NOT adapt a square/landscape layout.
- Root container `aspect-ratio: 2 / 3`, fixed canvas (e.g. `width: 1440px` → `height: 2160px`, 60 px/in). `@page { size: 24in 36in; margin: 0 }` inside `@media print` so Save-as-PDF yields a true 24×36 PDF. On-screen note outside the canvas: "Print / Save-as-PDF at 24×36 in, portrait, 100% scale."
- ~1 in quiet inner margin all sides.

## Venue / identity

- **Workshop:** FM4LS @ ICML 2026 (Foundation Models for the Life Sciences).
- **Title (verbatim):** "PerturbCausal: Virtual Cell Model Predictions Do Not Substitute for Real Perturb-seq in Causal Network Recovery"
- **Author:** Aayan Alwani · William A. Shine Great Neck South High School.
- Two QR placeholders (Paper, Code · PerturbCausal package) in the header corners.

---

## The overriding priorities (read first)

1. **FIGURE-FIRST.** ICML workshop posters are skimmed in seconds at a busy session. Figures must carry the story; text is captions + a few tight bullets. Target **~65–70% figure area**. No paragraph blocks. Every section is led by a figure with ≤3 short bullets.
2. **NOVEL CONTRIBUTION + IMPACT MUST BE UNMISTAKABLE (highest priority).** A viewer must grasp, within 10 seconds, _what is new_ and _why it matters / who benefits_. Give this its own dedicated, visually dominant real estate (a takeaway banner + a boxed Contribution/Impact panel), not buried in prose. Details in the "Contribution & Impact" section below.

## Layout (portrait, top→bottom, figure-led)

**TOP BAND (~top 14%)** — identity + the hook:

- Title, author, FM4LS @ ICML 2026 tag, two QR placeholders.
- **TAKEAWAY BANNER** (largest-effect text on the poster, reserved vermillion `#D55E00` accent):
  > **Predicted Perturb-seq does NOT substitute for real Perturb-seq in causal-network recovery: four deep virtual-cell models tie a sparse linear baseline, and per-gene calibration provably cannot fix it.**

**CONTRIBUTION & IMPACT PANEL (~next 12%, full width, visually distinct — boxed, light-tint background).** This is the most important block on the poster. Two side-by-side halves:

- **What's new (contribution):** three tight bullets —
  1. **First benchmark** to test whether VCM _predictions_ can substitute for _real_ Perturb-seq **downstream in causal-network recovery**, not just per-gene prediction.
  2. **A mechanism:** the failure is joint-structure distortion (mode collapse), objective-driven, identical across architectures.
  3. **An identifiability proof + a reachable fix:** per-gene calibration provably cannot close the gap; a training-time joint operation can.
- **Why it matters (impact):** two tight bullets —
  1. **For anyone using VCMs:** before trusting a model to replace a real screen for network/target discovery, you must test _causal substitutability_, not prediction accuracy — otherwise you build on wiring that is wrong.
  2. **For anyone building VCMs:** stop optimizing per-gene loss and post-hoc fixes; put joint structure into the _training objective_. We show it pays off (CPA causal F1 0.42 → 0.54). Benchmark + code open-source.

**BODY — four sections, each figure-led, in this order.** Use a clean grid (a 2-column band for Background+Methods to save vertical space, then full-width Results, then full-width Discussion/Conclusion), or a single vertical spine — whichever keeps Results dominant and the poster readable at ~1.5 m. Each section header is bold with a small ordinal.

### 1. Background / Rationale (compact — this is setup, keep it short)

- Lead figure: **`figures/illus_pipeline.png`** (real vs virtual Perturb-seq → one fixed causal engine → network).
- Bullets (≤3): VCMs are marketed as in-silico substitutes for CRISPR screens costing **millions per cell line**. Prior work (Ahlmann-Eltze & Huber 2025, _Nat Methods_) shows they don't beat a linear baseline at **per-gene** prediction — a **marginal** property. The applications that motivate VCMs depend on the **joint** distribution across genes. Nobody had tested downstream substitutability.

### 2. Methods (compact)

- Lead figure: a native **fixed-algorithm pipeline** diagram (rebuild from the ISMB reference): REAL and VCM-PREDICTED data → **SAME fixed causal-discovery algorithm (GIES primary)** → two networks → compare → **edge-set F1**. Only the data source varies.
- Bullets (≤3): Predictors = GEARS (GNN), CPA (cVAE), Geneformer (104M transformer), STATE (Arc 2025) + sparse-linear **ElasticNet**. Backends held fixed = GIES, inspre, PC, NOTEARS. Datasets = K562 (primary), RPE1, Norman 2019 (CRISPRi + CRISPRa). Chance = per-backend permutation null [2.5, 97.5].

### 3. Results (DOMINANT — biggest figures)

- **Hero chart (native SVG):** K562 edge-set F1 bar chart. Real **1.000** | Overlay-Real 0.429 | ElasticNet 0.426 | Geneformer 0.425 | CPA 0.412 | GEARS 0.406 | GRNBoost2 0.098; dashed vermillion floor at **0.045**; 95% CI whiskers; bracket the ~0.40 cluster: "deep VCMs tie the linear baseline — no pairwise diff at p<0.05; TOST-equivalent at Δ=0.045." Keep the "technical-duplicate noise floor" label clear of the bars.
- **`figures/real_vs_predicted_grn.png`** (real K562 network vs CPA, same genes/layout, teal=correct / vermillion=spurious / grey dashed=missed, **full-network edge-set F1 = 0.42**). Caption ties it to the "wiring is wrong" point.
- **Backend forest plot (native SVG):** {RPE1, Norman} × {PC, NOTEARS, inspre}, grey [2.5, 97.5] chance band per row, markers GEARS=circle / CPA=square / ElasticNet=triangle. Fact: largest deep F1 = **0.440** (CPA/inspre/RPE1), inside the chance band; ElasticNet strongest in **4 of 6**.
- **`figures/illus_modecollapse.png`** (mode collapse). Bullets: real **2.7%** of effects strong → deep **~63%** (GEARS 63.2, CPA 63.4, Geneformer 63.5) = **23× inflation, identical to within 0.3 pp** across GNN/VAE/transformer → objective-driven. CPA inter-perturbation cosine **0.84 vs 0.08**. inspre probe: CPA & Geneformer recover **ZERO** edges where real & ElasticNet recover **~12,000**. STATE single-seed **0.462** does not replicate ([0.462, 0.041, 0.048]).

### 4. Discussion / Conclusion

- Lead figure: **`figures/illus_calibration.png`** (calibration ✓/✗ fork: magnitude corrected exactly, wiring unchanged).
- Supporting native chart (optional if space): calibration F1 barely moves on held-out 160/40 (GEARS 0.303→0.290, CPA 0.309→0.289, Geneformer 0.319→0.287; onto real = 0.114; Geneformer ends below ElasticNet 0.425).
- **Identifiability result (boxed):** conditional independence is invariant under per-gene monotone calibration, so marginal rescaling **provably cannot** change the recovered graph ⇒ the residual is off-diagonal joint (precision-matrix) structure ⇒ **needs a training-time fix**.
- **The gap is reachable:** a no-GPU learned joint-transform lifts held-out CPA causal F1 **0.424 → 0.535 (+0.111)**; the naive covariance-Frobenius penalty alone fails a pre-registered multi-seed gate (**+0.003**).
- **Conclusions (≤3 bullets):** no deep VCM beats a sparse linear baseline on causal recovery, none reaches real quality; ElasticNet ties a 104M transformer by inductive-bias matching (L1 sparsity ≈ real-response sparsity, **56.8%** of variance in the leading singular vector); closing the gap is a training-time, not post-hoc, operation.
- **Limitations (1 line):** ground truth internally consistent but not yet externally biologically validated; cross-dataset cells single-seed; N=200.

---

## INTEGRITY — locked numbers (render verbatim; see `LOCKED_DATA.md`)

Do not invent, recompute, or re-round. In particular print **Geneformer +0.0002** exactly (not 0.001). All F1 values, 2.7%→~63% (63.2/63.4/63.5), 23×, 0.3 pp, STATE [0.462, 0.041, 0.048], calibration deltas, real=0.114, +0.111, +0.003, 0.440, 4 of 6, ~12,000, 56.8% must appear exactly.

## Visual system

- Clean Swiss / editorial, flat (no gradients, no drop-shadows, no clip-art). White/near-white background, near-black text. Generous whitespace; ICML posters read as competent when minimal.
- **Okabe–Ito, colorblind-safe, semantic:** Real = black `#000000`; deep VCMs = blue `#0072B2`; baselines (ElasticNet/GRNBoost2) = bluish-green `#009E73`; chance band = grey `#BBBBBB`; reserved attention accent = vermillion `#D55E00` (banner, noise-floor line, identifiability box only). Never encode a series by color alone — redundant-encode with shape/line-style/label.
- One clean sans via a system-font stack (Inter/Source Sans/Helvetica/Arial), no external fonts. Portrait 24×36 type scale (translate to px): Title 60–72 pt; **takeaway banner the largest visual weight**; contribution/impact panel headers ~40 pt; section headers 36–44 pt; body/bullets 26–32 pt (never below 24); captions 22–26 pt. Left-align body; center only title + banner.

## Anti-template / quality bar

- Must look like an intentional, figure-first ICML workshop poster, not a dense text wall or a default template. Strong hierarchy (title ≫ banner ≫ contribution panel ≫ section headers ≫ body), deliberate spacing, ≥30% whitespace.
- Avoid: paragraph blocks; body <24 pt; too many competing charts; color-only encoding; the contribution/impact hidden or buried; landscape/square canvas.

## Self-check (satisfy ALL)

1. Portrait **24×36 in**; exports to a true 24×36 PDF at 100%.
2. **Novel contribution AND impact are unmistakable within 10 s** — takeaway banner + a dedicated, visually dominant Contribution & Impact panel naming _what's new_ and _who benefits_.
3. Genuinely **figure-first** (~65–70% figure area; every section figure-led; ≤3 bullets each; no paragraphs).
4. Exactly four sections in order: **Background/Rationale · Methods · Results · Discussion/Conclusion**; Results is the dominant band.
5. All four provided illustrations embedded (`illus_pipeline`, `real_vs_predicted_grn`, `illus_modecollapse`, `illus_calibration`); data charts rebuilt native (K562 bar, backend forest, methods pipeline, optional calibration panel).
6. Hero K562 bar shows Real=1.000 vs ~0.40 cluster + 0.045 floor + TOST bracket + CI whiskers; no label/bar overlap.
7. Okabe–Ito semantic palette, redundant encoding; vermillion only on banner + floor + identifiability box.
8. Two labeled QR placeholders (Paper / Code); title + author + FM4LS@ICML2026 correct.
9. **Every locked number verbatim** — incl. Geneformer +0.0002. No fabricated, re-rounded, or dropped values.

Output the poster as one HTML artifact. Title: "PerturbCausal — FM4LS @ ICML 2026 Poster". Then stop.
