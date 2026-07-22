# JaxCont engine-consolidation slides

This directory contains a beginner-friendly 16:9 Beamer presentation about
JaxCont's continuation methods and the merged v0.2 engine consolidation.

Files:

- `engine_consolidation.tex` — editable Beamer source.
- `engine_consolidation.pdf` — compiled presentation.
- `SPEAKER_NOTES.md` — suggested explanations and a short/long presentation route.
- `METHOD.md` — how the presentation was researched and constructed.

Build from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error engine_consolidation.tex
```

The deck uses `\documentclass[aspectratio=169]{beamer}`. It intentionally uses
standard Beamer/TikZ packages instead of external images, so every mathematical
diagram remains editable and reproducible from the repository.

The inspected implementation is merged into `main`; this edition follows the
history through commit `b105417`. The design and implementation plan are under
`docs/superpowers/`, and final verification is recorded in `notes/ROADMAP.md`.
No JaxCont source file was changed to produce or update these slides.
