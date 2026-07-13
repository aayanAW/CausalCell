# ICML FM4LS Poster — build pack

Reformats the ISMB poster (`REFERENCE_ISMB_poster.pdf`) into an **ICML FM4LS workshop** poster: portrait, figure-first, contribution + impact front-and-center.

## Before you run it — add 3 figures

`figures/real_vs_predicted_grn.png` (real K562 network) is already here. **Save your 3 BioRender exports into `figures/` with these exact names:**

- `figures/illus_pipeline.png` — real-vs-virtual Perturb-seq pipeline
- `figures/illus_modecollapse.png` — REAL sparse vs VCM inflated grid
- `figures/illus_calibration.png` — calibration ✓/✗ fork

(The native data charts — K562 bar, backend forest, methods pipeline — are rebuilt by Claude Design from the numbers; it matches them to `REFERENCE_ISMB_poster.pdf`.)

## How to run

Paste the full contents of `POSTER_PROMPT.md` into Claude Design. Attach `REFERENCE_ISMB_poster.pdf` so it can match the existing charts.

## Key differences from the ISMB poster

|          | ISMB            | ICML FM4LS                                     |
| -------- | --------------- | ---------------------------------------------- |
| Size     | 46×46 in square | **24×36 in portrait** (hard cap)               |
| Style    | dense 4-column  | **figure-first, minimal text**                 |
| Emphasis | balanced        | **contribution + impact dominant** (own panel) |
| Venue    | ISMB 2026       | FM4LS @ ICML 2026                              |

## Files

| File                        | Purpose                                            |
| --------------------------- | -------------------------------------------------- |
| `POSTER_PROMPT.md`          | Paste into Claude Design                           |
| `REFERENCE_ISMB_poster.pdf` | Source poster (charts to match, content reference) |
| `LOCKED_DATA.md`            | Source-of-truth numbers to verify against          |
| `figures/`                  | Illustrations (add the 3 BioRender PNGs)           |

## Sources on ICML workshop poster spec

- ICML 2026 Poster Instructions: https://icml.cc/Conferences/2026/PosterInstructions — workshop posters max **61×91 cm (24×36 in), portrait**.
