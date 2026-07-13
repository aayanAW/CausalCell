# BioRender figure prompts — for Claude Science (or BioRender AI / manual reference)

Paste each prompt into Claude Science (which may have BioRender-AI entitlement your Claude Code session lacks). Export each as PNG (transparent background if possible) and drop into `figures/`. Illustration #2 (real-vs-predicted network) is already done from real data — these are the remaining three.

**Shared style (state this once, applies to all):** flat modern scientific-schematic style; white/transparent background; palette = teal `#0F766E` + slate `#334155`, with a single vermillion `#D55E00` accent for the "wrong/attention" element; colorblind-safe; minimal text, large legible labels; no drop-shadows, no clip-art, no gradients. Square-ish or 2:1 aspect. Match a clean SPARK-poster look.

---

## Figure 1 — "Real vs Virtual Perturb-seq" (the hero / anchor)

> Make a clean scientific schematic comparing two ways to obtain Perturb-seq data, flat modern style, teal + slate palette, white background, minimal text.
> **Top-left lane, labeled "REAL Perturb-seq":** a CRISPR-Cas9 complex editing a gene inside a cell → arrow → a dish of pooled perturbed cells → arrow → a sequencer producing a gene-expression matrix. Small caption under the lane: "costly CRISPR screen — millions per cell line."
> **Bottom-left lane, labeled "VIRTUAL cell model (VCM)":** a stylized neural-network / in-silico cell icon predicting a gene-expression matrix from a perturbation token. Small caption: "cheap in-silico prediction."
> **Both** expression matrices flow with arrows into a single shared box labeled **"SAME causal-discovery algorithm (GIES)"**, which outputs a **gene-regulatory-network graph** (nodes + directed edges) on the right.
> The message the layout must convey: _only the data source differs; the algorithm is held fixed; can the cheap virtual lane replace the expensive real lane downstream?_ Put a subtle question mark or "substitute?" label between the two lanes.

---

## Figure 3 — "Mode collapse / effect-size inflation"

> Make a clean two-panel biological schematic, flat style, teal + slate + one vermillion accent, white background.
> **Left panel, labeled "REAL responses — sparse & diverse":** a small grid of cells (or a gene × perturbation heat-grid) where only a FEW genes light up, and DIFFERENT genes light up for different perturbations. Caption: "only 2.7% of effects are strong."
> **Right panel, labeled "VCM prediction — inflated & collapsed":** the same grid but NEARLY ALL genes light up (in vermillion), and the pattern is NEARLY IDENTICAL across every perturbation. Caption: "~63% of effects strong — 23× inflation, near-identical across perturbations."
> Add a small callout: "identical to within 0.3 percentage points across GNN, VAE, and transformer → the objective, not the architecture." The visual contrast (sparse/varied vs saturated/uniform) is the whole point.

---

## Figure 4 — "Calibration fixes the ruler, not the wiring"

> Make a clean conceptual schematic, flat style, teal + slate + vermillion accent, white background, minimal text.
> Show a per-gene **calibration "dial/ruler"** rescaling each gene's magnitude so the predicted magnitude distribution now MATCHES the real one (show two small distributions snapping into alignment, labeled "magnitude marginal — corrected exactly").
> BELOW or BEHIND the dial, show the **gene-regulatory-network wiring UNCHANGED and still wrong** (a small node-edge graph with several vermillion "wrong" edges that the dial does NOT fix).
> Big takeaway label: **"Calibration corrects the magnitude ruler — the joint wiring stays wrong."** Small note: "conditional independence is invariant under per-gene monotone calibration (proven) → needs a training-time fix, not post-hoc rescaling."

---

### If Claude Science also returns "access denied"

That means the BioRender AI entitlement is missing from the account, not the surface. Then either (a) build these in the BioRender web app manually using its icon library, or (b) ask Claude Code to render them as rich native SVG (no BioRender needed).
