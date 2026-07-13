# Figure reference — real captions + what to say (axes / panels)

Paper: Johnson, Bergman et al., "Human interpretable grammar encodes multicellular systems
biology models to democratize virtual cell laboratories." Cell 188(17):4711–4733, 2025.
DOI 10.1016/j.cell.2025.06.048. Open access, CC BY 4.0.

All figure images are in `figures/`. These are the NIH/PMC open-access versions (~1000 px wide).

---

## GraphicalAbstract.jpg

Overview cartoon. Plain-language hypothesis statements (cell behavior rules) → automatically parsed
into Hill functions → stochastic agent-based timecourse simulations (0h/12h/24h). Lower half: a
mechanistic-study path and an informatics path (flow cytometry, cell-type proportions, gene/SNP data)
both feed the rules; bottom row shows the same engine running dish, mouse, pancreas, and brain models.
→ Use on the title/overview slide.

## Figure 1 — "Using agent-based models to digitize cell knowledge"

The grammar concept + workflow. Contains the rule template and the response-function machinery.

- Rule template (say it verbatim): **"In [cell type], [signal] increases/decreases [behavior]."**
- behavior = f(signal), where f is a Hill (dose-response) curve; up-signals and down-signals combine.
- Workflow: plain-text rules → CSV → PhysiCell agent-based model; omics data sets cell identities/positions.
  → Slides S3, S5, S6, S7. (Crop different panels for the math slide vs the workflow slide.)

## Figure 2 — "The hypoxia model captures the post-hypoxic dynamics"

2D tumor, 5 days, 38 mmHg oxygen, 2,000 cells seeded in a 400-µm disk.
Rules: "oxygen decreases necrosis," "oxygen decreases transformation to motile tumor cells."

- Panel A: the rules. Panel B: tumor snapshots + oxygen field over time → oxygen-poor necrotic core,
  motile cells disseminating, denser near the peri-necrotic boundary.
- Population plots: X = time, Y = number of cells (motile vs non-motile); QoI = area under the curve.
- Radial-distribution panels: X = radius from center, Y = cell density.
- Sensitivity panels (C–E): half-max values (necrosis onset, motile transitions) most influential;
  model most sensitive to cell motility; medians stable under small parameter perturbations.
  → Slide S8. Box the necrotic core and the invasive front.

## Figure 3 — "CAFs support the epithelial-to-mesenchymal transition in simulated pancreatic

## epithelial cells, which promotes invasive growth and the establishment of new epithelial-like foci"

PDAC co-culture. Two neoplastic states: proliferative EPITHELIAL vs migrating MESENCHYMAL (EMT).
Fibroblasts (CAFs) secrete a factor that drives EMT; motility vs collagen/ECM is biphasic (two Hill curves).

- Virtual co-cultures, 1,000 cells, varying PANC:CAF ratios, 7 days.
- Read invasiveness/invasion-distance vs PANC:CAF ratio over time; peak invasion at 10:1 and 5:1,
  significant vs tumor-only by ~24 h; very low CAF ratios match monoculture only at late times.
  → Slide S9. Box the 10:1 / 5:1 peak. Define CAF and EMT first.

## Figure 4 — "A simulated tumor evades cytotoxic killing by manipulating its immune microenvironment"

Adds CD8+ T cells (contact killing; IFN-γ promotes, IL-10 inhibits activation) and M0/M1/M2 macrophages
(polarization tied to local oxygen; phagocytose dead cells). 2,000 tumor + 400 M0 + 400 naïve T cells.

- Tumor-volume-vs-time curve: X = time (days), Y = tumor volume/size. Tumor shrinks **>50%**, then
  **rebounds to ~original volume by day 5**.
- Companion panels: T-cell states shifting toward EXHAUSTED; intratumoral macrophages becoming all M2-like.
- Sensitivity: hypoxia half-max for macrophage polarization is the most sensitive parameter.
  → Slide S10. Arrow the dip-then-rebound. Meaning: immunosuppression EMERGED, was not coded in.

## Figure 5 — "Tumor-associated macrophages in M2-like polarization state assist simulated invasive

## breast cancer spheroids through EGF signaling"

Recreates DeNardo et al. MMTV-PyMT system. Tests three hypotheses about what EGF does to tumor cells:
GROW (enter cell cycle), GO (become motile), or GO+GROW.

- In-silico panels: grow-only FAILS to reproduce invasive structures; go and go+grow REPRODUCE invasion.
- Wet-lab validation panels: EGFR inhibitor **gefitinib reduces invasive protrusions** in real organoids
  (only reduces colony formation at very high doses); **EGF increases MCF10A proliferation AND motility**
  (live-cell imaging quantification).
  → Slides S11 (in-silico) + S12 (validation). This is the predict→confirm highlight.

## Figure 6 — "Combination immunotherapies simulated for a cohort of untreated pancreatic adenocarcinomas

## based on immune cell proportions estimated from scRNA-seq data"

Virtual clinical trial of GVAX + Nivolumab (anti-PD-1) + Urelumab (anti-CD137).
T cells subtyped by PD-1 (PDCD1) and CD137 (TNFRSF9); tumor by PD-L1 (CD274). Virtual patients built from
untreated PDAC scRNA-seq (GSE155698), 1,000 tumor cells each. Drugs simulated as mechanism-of-action
shifts in cell-state proportions (GVAX doubles T cells; Nivolumab PD-1 hi→lo; Urelumab CD137 lo→hi).

- Big inter-patient variation: outcomes range from near-elimination to uncontrolled growth.
- Some single/double combos beat the triple combo in several tissues.
- Responders to the triple combo had **significantly higher macrophage abundance** (Pearson correlation).
- Endpoint lymphoid aggregates mirror clinical biospecimens.
- NEW hypothesis: macrophages must clear tumor cells first to enable T-cell trafficking and killing.
  → Slide S13. Highlight the responder-vs-macrophage correlation.

## Figure 7 — "Region-specific laminarization of the cortex"

Generalization beyond cancer. Adds asymmetric cell division (stem → 1 stem + 1 differentiating daughter);
differentiated cells migrate to the pial surface; simulation TIME used as a proxy signal to sequence layers.

- Calibrated to the Allen Brain Atlas for two adult-mouse regions: somatosensory (SOM) and auditory (AUD).
- Read layer-thickness model-vs-data per region (fit by minimizing residual sum of squares).
- Reproduces region-specific laminar (layered) structure for both regions.
  → Slide S14. Message: same grammar, totally different tissue.
