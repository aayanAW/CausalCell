# REVISE: ISMB 2026 PerturbCausal poster — targeted edits

You previously built the PerturbCausal ISMB 2026 poster (4-column SPARK-style: 01 Background · 02 Methods · 03 Results · 04 Findings & Conclusions, blue numbered section bars, white cards, italic figure captions). Keep that design, all text, and all numbers. Make ONLY the changes below. `CURRENT_poster_render.png` shows the current state; the swap-in figures are in `figures/`.

**Do not change any number, claim, or citation.** In particular keep **Geneformer +0.0002** exactly (not 0.001), and all F1 values, 2.7%→~63%, 23×, STATE [0.462, 0.041, 0.048], calibration deltas, +0.111, 56.8% verbatim (see `LOCKED_DATA.md`).

---

## Change 1 — Physical size: make it a true 46×46 in square

The current canvas is 23×23 in. ISMB allows up to **46×46 in (117×117 cm)**. Rebuild at **46×46 in** (root `aspect-ratio: 1/1`, `@page { size: 46in 46in }`), scaling all type and figures proportionally so nothing shrinks. Keep ~1 in quiet margin. Add an on-screen note: "Print / Save-as-PDF at 46×46 in, 100% scale."

## Change 2 — Add a TAKEAWAY BANNER under the title (currently missing)

Between the author/venue line and the four columns, add a full-width banner strip carrying the one-line finding as the **largest-effect text on the poster** (bigger visual weight than the title). Use the reserved vermillion `#D55E00` accent (a thin left rule or the key words in vermillion; keep the rest near-black). Wording (verbatim):

> **Predicted Perturb-seq does NOT substitute for real Perturb-seq in causal-network recovery — four deep virtual-cell models tie a sparse linear baseline, and per-gene calibration provably cannot fix it.**

## Change 3 — Add two QR-code placeholders (currently missing)

Top corners of the header: left = **"Paper"**, right = **"Code · PerturbCausal package"**. Render each as a crisp SVG placeholder (bordered square with a QR-grid motif + label), ~2–2.5 in. Do not route any finding through them.

## Change 4 — Swap four figures for the polished/real versions in `figures/`

Replace these native-SVG diagrams with the provided PNGs (embed each inline as a `data:` URI; keep the italic "Fig N." caption text, just change the image). **Delete the old diagram each replaces.**

| Current figure                                                   | Replace with (in `figures/`)    | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------------------------------------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Fig 3** — "real vs scrambled wiring" 5-node cartoon (Column 1) | **`real_vs_predicted_grn.png`** | REAL-DATA figure (real K562 network vs CPA, F1 0.42). **MOVE it into Column 3 (Results)** near Fig 6/Fig 8 — it is a results figure, and it is a wide 2-panel image that needs a full-column-width slot, not the narrow Column-1 slot. Update its caption to: "Fig. Real K562 recovered network vs CPA (deep VCM), same genes and layout; teal = correct edge, vermillion = spurious, grey dashed = missed. Full-network edge-set F1 = 0.42." |
| **Fig 1** — concept pipeline "substitute downstream?" (Column 1) | **`illus_pipeline.png`**        | BioRender real-vs-virtual pipeline. Keep the Column-1 slot + caption intent (real and virtual Perturb-seq feed one fixed causal engine).                                                                                                                                                                                                                                                                                                      |
| **Fig 8** — mode-collapse grid (Column 3)                        | **`illus_modecollapse.png`**    | BioRender version. Keep caption.                                                                                                                                                                                                                                                                                                                                                                                                              |
| **Fig 10** — calibration "dial" (Column 4)                       | **`illus_calibration.png`**     | BioRender ✓/✗ fork version. Keep caption.                                                                                                                                                                                                                                                                                                                                                                                                     |

Keep ALL other figures unchanged (Fig 2 two-axes, Fig 4 methods pipeline, Fig 5 chance band, Fig 6 K562 bar, Fig 7 backend forest, Fig 9 calibration panels, STATE strip). Renumber captions if needed so they stay sequential after the Fig 3 move.

## Change 5 — Fix the Fig 6 label collision

On the K562 F1 bar chart, the red "technical-duplicate floor" label overlaps the GRNBoost2 bar / 0.045 line region. Move the floor label clear of the bars (e.g. left end of the dashed line, above it) so nothing overlaps.

## Change 6 — Rebalance Column 4 whitespace

Column 4 (Findings & Conclusions) has large empty gaps (after "Why: an identifiability result" and after "The gap is reachable"). Even out the vertical rhythm so the column reads as consistent cards with balanced spacing — no big dead zones, no cramped blocks. Do not delete content.

---

## Keep unchanged

- All four section bars, all body text, all numbers, all citations, the reference list.
- The visual system: white background, blue numbered section bars, teal/blue + vermillion accent, Okabe–Ito figure palette, italic gray captions, the clean SPARK-style layout.

## Self-check before finishing

1. True 46×46 in square; exports to a 46×46 PDF at 100%.
2. Takeaway banner present, largest visual weight, vermillion accent, verbatim wording.
3. Two labeled QR placeholders in the header.
4. All four figure swaps done (`real_vs_predicted_grn` moved to Results; `illus_pipeline`, `illus_modecollapse`, `illus_calibration` embedded); old cartoons deleted; captions still sequential.
5. Fig 6 has no label/bar overlap.
6. Column 4 spacing balanced.
7. Every number still verbatim — **Geneformer +0.0002**, all F1s, 2.7/63/23×, STATE [0.462, 0.041, 0.048], calibration deltas, +0.111, 56.8%. No value changed.
8. Everything else identical to the current poster.

Output the revised poster as one self-contained HTML artifact. Title: "PerturbCausal — ISMB 2026 Poster (rev)".
