# ISMB Poster — Revision pack

Edits for the current PerturbCausal poster (`CURRENT_poster.pptx` / `CURRENT_poster_render.png`).

## Before you run it — add 3 figures

I could not pull your BioRender PNGs from the chat images. **Save these 3 into `figures/` with these exact names:**

- `figures/illus_pipeline.png` — BioRender real-vs-virtual Perturb-seq pipeline
- `figures/illus_modecollapse.png` — BioRender "REAL sparse vs VCM inflated" grid
- `figures/illus_calibration.png` — BioRender calibration ✓/✗ fork

`figures/real_vs_predicted_grn.png` (real K562 network vs CPA) is already here.
Then delete `figures/DROP_3_BIORENDER_PNGS_HERE.txt`.

## How to run

Paste the full contents of `REVISION_PROMPT.md` into Claude Design — ideally the **same** Claude Design conversation that built the poster (so it still has the artifact). If it's a fresh session, also attach `CURRENT_poster_render.png` for reference.

## What the revision does

1. Resize to true **46×46 in** (ISMB max; current is 23×23).
2. Add the missing **takeaway banner** (one-line finding, vermillion).
3. Add **QR placeholders** (Paper + Code).
4. **Swap 4 figures** → the polished BioRender + real-data versions.
5. Fix the **Fig 6** label collision.
6. Rebalance **Column 4** whitespace.
   All text + numbers stay identical (verify against `LOCKED_DATA.md`).

## Files

| File                                                | Purpose                                    |
| --------------------------------------------------- | ------------------------------------------ |
| `REVISION_PROMPT.md`                                | Paste into Claude Design                   |
| `CURRENT_poster.pptx` / `CURRENT_poster_render.png` | The current poster (reference)             |
| `figures/`                                          | Swap-in figures (add the 3 BioRender PNGs) |
| `LOCKED_DATA.md`                                    | Source-of-truth numbers to verify against  |
