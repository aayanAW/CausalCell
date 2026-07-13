WHAT'S IN THIS FOLDER
=====================
PROMPT.txt           <- the full prompt. Paste this into Claude.
figure-legends.md    <- real captions + axis/panel notes for each figure (reference for your talk).
figures/             <- 9 image files to upload to Claude (the actual paper figures, CC BY 4.0):
                          GraphicalAbstract.jpg
                          Figure1_grammar.jpg
                          Figure2_hypoxia.jpg
                          Figure3_PDAC_fibroblasts.jpg
                          Figure4_tumor_immune_evasion.jpg
                          Figure5_go_vs_grow_validation.jpg
                          Figure6_virtual_clinical_trial.jpg
                          Figure7_brain_cortex.jpg

HOW TO USE
==========
1. Open Claude (claude.ai). Start a new chat.
2. Attach ALL 9 images from the figures/ folder (drag-and-drop or the paperclip).
3. Paste the entire contents of PROMPT.txt as your message. Send.
4. Claude returns a .pptx (download it) + a slide-by-slide transcript.
5. Check its self-report: 15-20 slides, 15-20 min, every body <=20 words. If any slide is over
   20 body words, tell Claude the slide number and say "trim to <=20 words."

NOTES
=====
- Figures are the open-access PMC versions (~1000 px wide). Fine for slides; add arrows/boxes/zoom
  callouts as the prompt instructs. If a figure looks soft when projected full-screen, ask Claude to
  crop to the single panel you're discussing instead of showing the whole multi-panel figure.
- Paper: Johnson, Bergman et al., Cell 188(17):4711-4733, 2025. DOI 10.1016/j.cell.2025.06.048.
- Rubric reminders baked into the prompt: titles must be specific (not "Methods"); teach AP-bio level;
  each experiment = method -> figure walk -> result -> meaning -> transition; do NOT read off slides.
