# BUILD: ISMB 2026 Poster — "PerturbCausal" (modeled on the SPARK reference poster)

You are Claude Design. Produce a **single, self-contained, print-ready conference poster** as one HTML/CSS artifact. It must render correctly offline.

**Asset rule (two kinds of figures):**

- **Data charts** (bar / forest / dot / line plots) → **native inline SVG/CSS** built from the locked numbers. No raster.
- **Illustrative / biological schematics** → create them with **BioRender, Figma, and any available illustration/diagram plugins** (see the Illustrative-figures section), then **embed each as a high-resolution asset inlined as a `data:` URI** so the single HTML file stays self-contained and renders offline. No live/external URLs, no CDN fonts, no web requests at render time — everything embedded.

**Model the design after the SPARK reference poster** (`SAMPLE_reference_SPARK_poster.pdf` in this folder). Match its layout system, color language, card structure, and typography closely — but with PerturbCausal's content. The SPARK design is described in full below so you can reproduce it without the file. This is a **traditional, dense, multi-column academic poster done cleanly** — NOT a billboard/one-takeaway poster.

---

## Physical Spec (ISMB 2026)

- **Venue:** ISMB 2026 (ISCB), Washington, D.C., 13–16 July 2026 — poster session.
- **Size:** **46 in × 46 in (117 × 117 cm) — ISMB hard cap.** Design a near-**square** board (like SPARK). A 4-column layout on a 46-in-wide board ≈ 11 in per column.
- **Build:** root container `aspect-ratio: 1 / 1`, fixed square canvas (e.g. `width: 2208px` → `height: 2208px`, 48 px/in). Add `@page { size: 46in 46in; margin: 0; }` inside `@media print` so **Save-as-PDF** yields a true 46×46 in PDF. Add an on-screen note outside the canvas: "Print / Save-as-PDF at 46×46 in, 100% scale. No onsite printing at ISMB — print in advance."
- **Quiet margin:** ~1 in inner safe margin on all sides.

---

## SPARK reference design system (reproduce this exactly, with PerturbCausal content)

**Overall:** white background; a full-width header strip on top; below it, **four equal vertical columns**, each a stack of **white content cards**. Clean, flat, grid-aligned, generous but efficient. No drop shadows beyond a hairline card border; no gradients; no clip-art.

**Header strip (full width, top ~9%):**

- **Top-left wordmark:** the project short-name in bold accent-teal, with a small gray two-line tagline immediately to its right. SPARK does: **"SPARK"** + _"Simulation Platform for Adaptive Resistance Kinetics."_ For this poster use **"PerturbCausal"** (bold teal) + tagline _"A causal-recovery substitutability benchmark for virtual cell models."_
- **Title:** large bold near-black sans, up to 2 lines, left-aligned, directly under the wordmark:
  **"PerturbCausal: Virtual Cell Model Predictions Do Not Substitute for Real Perturb-seq in Causal Network Recovery"**
- **Author line (gray, smaller):** **"Aayan Alwani · William A. Shine Great Neck South High School"** (same format as SPARK's "Ethan Y. Wang, Aayan Alwani · William A. Shine Great Neck South High School").
- A thin horizontal rule separates the header from the columns.

**Column section-header bars:** each column begins with a **filled rounded-rectangle bar** spanning the column width, in a **dark teal/slate** fill with **white text**: a lighter/outlined **two-digit ordinal** ("01", "02", "03", "04") followed by the bold section name. The four bars, in order:

- **01 Background** (= Background / Rationale)
- **02 Methods**
- **03 Results**
- **04 Findings & Conclusions** (= Discussion / Conclusion)

**Content cards (inside each column):** white fill, hairline light-gray border, rounded corners, comfortable padding, stacked with even gaps. Each card has:

- A **sub-heading** in **bold accent-teal** (e.g. SPARK's "Melanoma is a MAPK-driven disease", "Degrading BRAF with PROTACs").
- **Bullets** led by a small **teal ► triangle marker**, dark-gray body text, **key terms bolded inline**, and **italic gray citations** in parentheses — e.g. _(Ahlmann-Eltze & Huber 2025)_.
- Usually **one figure or schematic** per card, with a tiny **italic gray caption** beneath: "Fig 1 · …".
- Optional small **pill chips** for term-to-role mappings (SPARK uses light-lavender chips: "PROTAC → Bulk tumor").

**Color language (match SPARK):**

- **Accent teal/green** for the wordmark, sub-headings, ► markers, and section-bar fill (a dark teal for bars, brighter teal for text). Pick one coherent teal family.
- **Background** white; **cards** white with a `#E6EAED`-ish hairline border; page gutters very light gray if needed.
- **Body text** dark slate-gray `#2B2F33`; **citations** medium gray italic.
- **Figures** carry their own data palette — use **Okabe–Ito colorblind-safe** encodings there: **Real data = black `#000000`**, **deep VCMs (GEARS/CPA/Geneformer/STATE) = blue `#0072B2`**, **baselines (ElasticNet/GRNBoost2) = bluish-green `#009E73`**, **chance band = grey `#BBBBBB`**, and a single **reserved vermillion `#D55E00`** accent for the noise-floor line and the one identifiability callout. Never encode a series by color alone — redundant-encode with **shape (circle/square/triangle)**, **line style**, and **direct labels**.

**Typography (match SPARK's hierarchy; one clean sans via a system-font stack — Inter/Source Sans/Helvetica/Arial, no external fonts).** On-poster scale for a 46-in board (translate proportionally to px):

- **Title:** ~64–80 pt bold. **Wordmark:** ~48–56 pt bold teal.
- **Section-bar label:** ~40–48 pt (ordinal lighter, name bold white).
- **Card sub-heading:** ~30–36 pt bold teal.
- **Body / bullets:** ~24–30 pt (this is a dense poster; keep readable at ~1.5 m — do not go below ~22 pt).
- **Captions / citations / references:** ~18–22 pt italic gray.
- Left-align everything except the header wordmark/title block. Hierarchy from scale + weight + the teal accent, exactly as SPARK does.

**Fidelity to SPARK:** column rhythm, the teal numbered bars, white bordered cards, ► bullets, italic captions, bold-teal sub-headings, and the top-left wordmark+tagline are the signature elements. Reproduce all of them. Do not invent a different visual language.

---

## Figure-first within the SPARK card system

SPARK is figure-rich — most cards carry a figure. Keep that, but **lead each card with its figure/diagram and keep bullets tight (≤3–4, short)**. **Data charts** are **native inline SVG/CSS built from the locked numbers below** — do not raster them. **Illustrative/biological schematics** come from BioRender/Figma/illustration plugins, embedded as inline `data:` URIs (see the Illustrative-figures section). Reference PNGs of the real paper figures are in `figures/` (fig1 headline bars, fig2 mechanism, fig3 inspre probe, fig4 calibration, fig_backend forest, fig_seeds) — match their **content**, redraw in this poster's system.

### Charts to build natively

**(A) HERO — K562 edge-set F1 bar chart** (Column 03, first card). Vertical bars, y = "Edge-set F1 vs real-data GIES network (K562, N=200, 5 seeds)", 0→1.0, with 95% CI whiskers. Bars in order: **Real 1.000 (black)** · Overlay-Real 0.429 · **ElasticNet 0.426 (green)** · Geneformer 0.425 (blue) · CPA 0.412 (blue) · GEARS 0.406 (blue) · GRNBoost2 0.098 (muted green) · dashed **vermillion** line at **0.045** "technical-duplicate noise floor." Bracket the ~0.40 cluster: **"deep VCMs tie a linear baseline — no pairwise diff at p<0.05; TOST-equivalent at Δ=0.045 F1."** Visual beat: the Real=1.000 bar towers over a tight ~0.40 cluster.

**(B) BACKEND / GENERALIZATION — forest / dot plot** (Column 03). 6 rows = {RPE1, Norman} × {PC, NOTEARS, inspre}; per-row grey [2.5, 97.5] permutation-null band; markers GEARS=circle, CPA=square, ElasticNet=triangle. Deep markers land inside/below the band. Caption: **largest deep-model F1 across the 6 cells = 0.440 (CPA/inspre/RPE1), inside the chance band; ElasticNet strongest in 4 of 6; no deep VCM exceeds ElasticNet on any backend/dataset.**

**(C) MECHANISM panel** (Column 03). Effect-size inflation bar: real **2.7%** of (perturbation,gene) effects over |log2FC|=0.5 → deep models **~63% (GEARS 63.2, CPA 63.4, Geneformer 63.5) = 23× inflation, identical to within 0.3 pp** → objective-driven, not architecture-driven. Plus mode-collapse marker: **CPA inter-perturbation cosine 0.84 vs real 0.08.**

**(D) CALIBRATION-NULL panel** (Column 04). Panel 1 — magnitude marginal: vanilla (hatched ~1.0) vs calibrated (solid, drops onto real = **0.114** line). Panel 2 — causal F1 barely moves: **GEARS 0.303→0.290, CPA 0.309→0.289, Geneformer 0.319→0.287** (held-out 160/40 split); Geneformer ends **below** ElasticNet = 0.425. Header: **"Calibration corrects magnitude EXACTLY — the causal gap does not move."**

### Two ORIGINAL schematic diagrams (design natively; keep the SPARK schematic-with-caption look)

**(a) "Two different axes" concept** (Column 01). Two orthogonal axes: X = "matches per-gene means (marginal)", Y = "preserves gene–gene joint / conditional-independence structure". Deep VCMs sit **high-X, low-Y**; real data high on both. Caption: **"A model can match every per-gene mean yet distort the gene–gene structure causal discovery exploits. Prior work tested only the marginal axis."**

**(b) Fixed-algorithm pipeline** (Column 02, top card — the core design). Left: two data sources — **"REAL Perturb-seq"** and **"VCM-PREDICTED Perturb-seq"** — both flow into one boxed **"SAME fixed causal-discovery algorithm (GIES primary)"** → two recovered networks → a **compare/diff** node → **"edge-set F1."** Message: **only the data source varies; the algorithm is fixed; compare recovered edge sets.**

---

## Illustrative figures (make these RICH — use BioRender / Figma / illustration plugins)

The SPARK reference earns its clarity from **illustrative biological schematics** (BioRender cell membranes, signaling cascades, molecular graphics — not just boxes-and-arrows). **Do the same here.** Beyond the data charts and the two schematics above, **create the following illustrative figures using BioRender for the biology and Figma for polished vector schematics/icons** (or whatever illustration/diagram plugins are available). These should be genuinely illustrative — cells, DNA, CRISPR, sequencers, neural-net motifs, gene-network graphs, molecular icons — **not** plain rectangles pointing at rectangles. If a specific plugin is unavailable, produce an equivalently rich, detailed illustrative SVG natively rather than falling back to a plain block diagram. Embed each as an inlined `data:` URI. Keep a consistent illustration style across all of them (line weight, palette tied to the poster's teal + Okabe–Ito accents), and give each a SPARK-style italic-gray caption + a small "Illustration" credit where a tool is used.

**Priority illustrations to create:**

1. **"Real vs virtual Perturb-seq" hero illustration** (Column 01 or the top of Column 02 — the concept that anchors the whole poster). Two contrasting lanes feeding a shared pipeline:
   - **REAL lane:** a CRISPR/Cas9 editing a gene in a cell → a dish/pool of perturbed cells → a sequencer → a real expression matrix. Tag it with cost/effort ("CRISPR screen — millions per cell line").
   - **VIRTUAL lane:** a stylized **neural-network / in-silico "virtual cell"** predicting an expression matrix from a perturbation token (cheap, in-silico).
   - Both matrices flow into the **same causal-discovery engine** → a recovered **gene-regulatory-network graph**. Load-bearing message: _can the cheap virtual lane replace the expensive real lane, downstream?_

2. **"Marginal vs joint" — REAL-DATA network comparison** (Column 03 or 01, pairs with the two-axes concept). **Embed the provided figure `figures/real_vs_predicted_grn.png`** — a real, data-backed two-panel gene-regulatory-network comparison generated from this project's own results (`gies_cache/n200_s123_p20`, K562 seed 123): **left = network recovered from REAL Perturb-seq; right = network recovered from CPA (deep VCM) predictions**, same genes, same layout. Edges colored: **teal = correct (in real), vermillion = spurious (wrong), grey dashed = missed real edge.** Full-network **edge-set F1 = 0.42** (precision 0.42, recall 0.43). This is the headline "matching per-gene levels ≠ getting the wiring right" beat, on real data — prefer it over a cartoon. (A stylized SVG version is an acceptable fallback only if the PNG cannot be embedded; regenerate via `make_real_vs_predicted_grn.py`.)

3. **"Mode collapse / effect-size inflation" illustration** (Column 03, alongside chart C). A biological cartoon: **real cells respond sparsely and diversely** (a few genes light up, different genes per perturbation) vs **the VCM inflates everything** (nearly all genes light up, near-identical across perturbations). A stylized cell-by-gene response heat-grid or a set of cells with lit gene markers conveys "23× inflation + near-identical responses" more vividly than the bar alone.

4. **"Calibration fixes the ruler, not the wiring" illustration** (Column 04, pairs with chart D + the identifiability card). A small illustrative metaphor: a per-gene **quantile-matching "dial"** rescales each gene's magnitude to match real (the marginal), while the **network wiring underneath stays scrambled** — visually proving calibration cannot touch the joint/edge structure (the identifiability point).

Use these to _explain_ the project, not decorate it: each illustration must carry a specific idea from the paper. Keep them clean and on-brand (flat, teal-accented, colorblind-safe), matching SPARK's illustrative density and polish.

---

## Column-by-column content (map to the four section bars; SPARK-style cards)

**INTEGRITY:** render every number verbatim. In particular print **Geneformer +0.0002** exactly as **+0.0002** — do NOT recompute from bar heights (0.426−0.425) or round to 0.001.

### Column 01 — Background (bar "01 Background")

- **Card: "VCMs promise in-silico CRISPR screens."** VCMs are marketed as substitutes for CRISPR screens costing **millions per cell line**. ► set up the promise.
- **Card: "Prior work: they fail at per-gene prediction."** _(Ahlmann-Eltze & Huber 2025, Nat Methods)_ — VCMs don't beat simple **linear baselines** at per-gene prediction — a property of the **marginal** distribution.
- **Card: "But the payoff is downstream — in the joint structure."** Concept diagram **(a)**. Drug-target ID, combination prediction, GRN inference depend on the **joint** distribution across genes.
- **Card: "The open question (our contribution)."** Nobody had tested whether **predicted** data can substitute for **real** data **downstream in causal-network recovery**. **This is the first benchmark that does — holding the causal algorithm fixed and varying only the data source.**

### Column 02 — Methods (bar "02 Methods")

- **Card: "Fixed-algorithm, vary-data-source design."** Pipeline diagram **(b)**. Edge-set **F1** of predicted-data network vs the network the **same** algorithm recovers from **real** data.
- **Card: "Predictors (4 deep VCMs + 1 linear)."** GEARS (graph NN), CPA (conditional VAE), **Geneformer (104M-param transformer)**, STATE (Arc Institute 2025 foundation model); comparator = sparse-linear **ElasticNet** on held-out controls.
- **Card: "Causal-discovery backends (held FIXED)."** **GIES** (score-based, interventional — _primary_), **inspre** (sparse inverse of the ACE matrix), **PC** (constraint-based), **NOTEARS** (continuous optimization).
- **Card: "Datasets & chance reference."** **Replogle K562** (primary, 200-gene subset), **RPE1** (cross-context), **Norman 2019** (cross-paradigm CRISPRa); CRISPRi + CRISPRa. Chance = per-backend **permutation null** (shuffle each ACE column across perturbations) → **[2.5, 97.5]** band.

### Column 03 — Results (bar "03 Results")

- **Card (HERO): "Deep VCMs tie a sparse linear baseline."** Chart **A**. K562, N=200, 5 seeds: Real 1.000 | Overlay-Real 0.429 | ElasticNet 0.426 | Geneformer 0.425 | CPA 0.412 | GEARS 0.406 | GRNBoost2 0.098 | floor 0.045. **No pairwise diff at p<0.05; TOST-equivalent at Δ=0.045** (CPA +0.014, GEARS +0.020, Geneformer +0.0002). A 104M transformer, a GO-prior GNN, and a conditional VAE all tie a linear regression.
- **Card: "Holds across backends and datasets."** Chart **B**. Largest deep-model F1 across 6 cross-dataset cells = **0.440 (CPA/inspre/RPE1)**, inside the chance band; ElasticNet strongest in **4 of 6**.
- **Card: "Mechanism: effect-size inflation = mode collapse."** Chart **C**. 2.7% → ~63% (**23×**, identical to 0.3 pp across architectures); CPA cosine **0.84 vs 0.08**. **inspre signal probe:** CPA & Geneformer ACE fall below detection → **ZERO edges** where real & ElasticNet recover **~12,000**.
- **Card: "The 2025 foundation model too."** **STATE** single-seed **0.462** beats seed-42 ElasticNet but **does not replicate** — per-seed **[0.462, 0.041, 0.048]**.

### Column 04 — Findings & Conclusions (bar "04 Findings & Conclusions")

- **Card: "Calibration corrects magnitude, not causality."** Chart **D**. Quantile-matching fixes the magnitude marginal **exactly** (onto real = 0.114), yet held-out **160/40** causal F1 does not close: GEARS 0.303→0.290, CPA 0.309→0.289, Geneformer 0.319→0.287.
- **Card (ACCENT callout): "Why — an identifiability result."** Conditional independence is **invariant under per-gene monotone (rank-preserving) calibration**, so marginal rescaling **provably cannot change the recovered graph** ⇒ the residual is the **off-diagonal joint (precision-matrix support)** structure → closing the gap needs a **TRAINING-TIME** intervention, not post-hoc rescaling. Give this card the reserved **vermillion** accent (the one attention beat).
- **Card: "The gap is reachable."** A no-GPU **learned joint-transform proxy** raises held-out CPA causal F1 **0.424 → 0.535 (+0.111)** — first evidence the gap is reachable by a joint operation. A naive covariance-Frobenius penalty alone **fails a pre-registered multi-seed gate (+0.003)** → it's the learned joint map, not the raw penalty.
- **Card: "Conclusions."** ► The substitutability gap **decomposes** into a magnitude marginal (calibration-correctable) + a **joint-distribution residual** calibration cannot reach. ► No deep VCM beats a sparse linear baseline on causal recovery; none reaches real-data quality. ► **Why ElasticNet ties a 104M transformer:** inductive-bias matching — L1 sparsity matches real-response sparsity (**56.8%** of variance in the leading singular vector). ► Closing the gap is a **training-time**, not post-hoc, operation.
- **Card: "Limitations."** Ground truth internally consistent but **not yet externally biologically validated**; cross-dataset cells single-seed; N=200 scale.
- **Card: "Selected references"** (tiny 2-column gray list, SPARK-style): Ahlmann-Eltze & Huber 2025, Nat Methods · Replogle 2022, Cell · Norman 2019, Science · Hauser & Bühlmann 2012, JMLR (GIES) · Brown 2025, Nat Commun (inspre) · Lotfollahi 2023, Mol Syst Biol (CPA) · Roohani 2024, Nat Biotech (GEARS) · Theodoris 2023, Nature (Geneformer) · Adduri 2025, bioRxiv (STATE) · Zou & Hastie 2005, JRSS-B (ElasticNet).

---

## Two beats that must land (as in SPARK's clear takeaways)

1. **The novel contribution** — fixed-algorithm / vary-data-source (pipeline b) + the "two axes" concept (a). A viewer grasps in a few seconds that this tests **downstream causal substitutability**, not per-gene prediction.
2. **"Calibration provably cannot fix it → needs a training-time fix"** — the identifiability card, in the reserved vermillion accent: magnitude corrected exactly while causal F1 does not move, with a Proposition proving why.

---

## Anti-template / quality bar

- Must look like SPARK: an intentional, professional, dense-but-clean 4-column academic poster — not a generic template or default slide deck.
- Consistent card rhythm, aligned grid, disciplined teal accent, real hierarchy (title ≫ section bars ≫ sub-headings ≫ body).
- Avoid: walls of text (keep bullets ≤3–4, short); body below ~22 pt; color-only encoding; rainbow palettes; heavy borders/shadows; the core finding hidden behind a QR.

## Self-check (satisfy ALL)

1. **46×46 in square**, ~1 in margin, exports to a true 46×46 PDF at 100%.
2. Visual system clearly **modeled on SPARK**: top-left wordmark + tagline; big left-aligned title; author line; four teal **numbered section bars** (01 Background · 02 Methods · 03 Results · 04 Findings & Conclusions); white bordered cards with **teal sub-headings**, **► teal bullets**, **italic gray captions/citations**.
3. **Exactly four columns = the four sections in order**; each column's bar carries its ordinal.
4. **Novel contribution unmistakable** (downstream causal substitutability; fixed-algorithm/vary-data-source) via cards + diagrams (a) and (b).
5. Hero K562 chart shows **Real=1.000 vs ~0.40 cluster** + **0.045** floor + the "tie / TOST-equivalent" bracket + 95% CI whiskers.
6. The **identifiability beat** has its own **vermillion** accent card.
7. **Null is credible:** 5 seeds, TOST Δ=0.045, CIs / chance bands visible on figures.
8. Figure-led cards; bullets tight; **data charts** = native inline SVG/CSS from locked numbers; **illustrative schematics** = BioRender/Figma illustrations embedded as inline `data:` URIs; single self-contained file; renders offline.
   8b. **≥4 rich illustrative figures present** (real-vs-virtual hero, marginal-vs-joint, mode-collapse/inflation, calibration-fixes-the-ruler) — genuinely illustrative (cells/CRISPR/DNA/network/neural-net motifs), not plain box-and-arrow, on-brand and colorblind-safe.
9. Okabe–Ito semantic palette in figures with **redundant shape/linestyle/label** encoding; vermillion only on noise-floor + identifiability callout.
10. Title matches the paper exactly; author = **Aayan Alwani · William A. Shine Great Neck South High School**; venue vibe = **ISMB 2026**.
11. Two original schematics (two-axes concept + fixed-algorithm pipeline) present and legible at ~1.5–2 m.
12. **Every locked number verbatim** — incl. **Geneformer +0.0002** (not 0.001). No fabricated, re-rounded, or dropped values.

Output the poster as one HTML artifact. Set a stable `<title>` ("PerturbCausal — ISMB 2026 Poster"). Then stop.
