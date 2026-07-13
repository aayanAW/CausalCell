# ISMB 2026 Poster — PerturbCausal

Everything needed to generate the poster with **Claude Design** (or any capable design agent).

## How to use

1. Open a Claude Design session (or an artifact-capable Claude).
2. Paste the **entire contents of `POSTER_PROMPT.md`** as the prompt.
3. It produces a self-contained HTML poster artifact. Open it, then **Print → Save as PDF at 46×46 in, 100% scale**.
4. Sanity-check against `LOCKED_DATA.md` — every number on the poster must match verbatim.

**Design is modeled on the SPARK poster** (`SAMPLE_reference_SPARK_poster.pdf`): 4 teal numbered columns (01 Background · 02 Methods · 03 Results · 04 Findings & Conclusions), white cards, teal ► bullets, italic gray captions, top-left wordmark. The prompt describes that system in full so the designer reproduces it.

## Files

| File                                        | What it is                                                                                                                                                                                                                                                |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POSTER_PROMPT.md`                          | **The prompt.** Paste this into Claude Design. Self-sufficient — has all locked numbers, layout, figure specs, visual system, self-check.                                                                                                                 |
| `LOCKED_DATA.md`                            | Source-of-truth numbers + the novel-contribution statement. Use to verify the output.                                                                                                                                                                     |
| `figures/`                                  | Reference PNGs of the paper's real figures (fig1 headline, fig2 mechanism, fig3 inspre probe, fig4 calibration, fig_backend, fig_seeds). The prompt tells the designer to **redraw these natively** — these show intended _content_, not assets to embed. |
| `PerturbCausal_paper_FM4LS_cameraready.pdf` | The full paper (source of every number).                                                                                                                                                                                                                  |
| `SAMPLE_reference_SPARK_poster.pdf`         | The reference poster the design is modeled on (a different project; ISMB 2026 4-column style).                                                                                                                                                            |

## Poster spec (ISMB 2026)

- **Size:** max **46 in × 46 in (117 × 117 cm), square.** Orientation not mandated — design fills the square.
- **Printing:** self-arranged (no onsite printing at ISMB). Washington, D.C., 13–16 July 2026.
- **Digital submission:** ISMB requires a PDF of the poster **+ a ≤5-min video** to the virtual platform (deadline **July 3, 2026** — verify current date against the ISMB site).

## The one thing the poster must nail

> First benchmark of whether VCM **predicted** Perturb-seq substitutes for **real** Perturb-seq **downstream in causal-network recovery** (not per-gene prediction). Fixed algorithm, vary only the data source. Four deep VCMs tie a linear baseline; per-gene calibration **provably** can't fix it (identifiability result) → a training-time fix is required.

## Sections (fixed order)

1. Background / Rationale — 2. Methods — 3. Results (hero) — 4. Discussion / Conclusion.
