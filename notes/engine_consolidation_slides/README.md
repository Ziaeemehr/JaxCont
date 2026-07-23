# JaxCont technical presentation

This directory contains a beginner-friendly 16:9 Beamer presentation about
JaxCont's continuation methods, implementation, research applications, and
practical validation workflow.

Files:

- `jaxcont_technical_presentation.tex` — editable Beamer source.
- `jaxcont_technical_presentation.pdf` — compiled presentation.
- `Makefile` — build and cleanup commands.
- `SPEAKER_NOTES.md` — suggested explanations and a short/long presentation route.
- `METHOD.md` — how the presentation was researched and constructed.

Build from this directory with:

```bash
make
```

Use `make clean` to remove generated LaTeX intermediates while keeping the
PDF, `make distclean` to remove the PDF as well, or `make rebuild` for a clean
recompilation.

The deck uses `\documentclass[aspectratio=169]{beamer}`. It intentionally uses
standard Beamer/TikZ packages instead of external images, so every mathematical
diagram remains editable and reproducible from the repository.

## Documentation strategy

Use this deck for visual explanation, teaching, and talks. Use
`notes/BEGINNERS_GUIDE_TO_BIFURCATION_ANALYSIS.md` as the detailed,
searchable prose guide and eventual documentation source. Avoid turning the
slides into full-page prose or maintaining a second Markdown copy of the same
material.

Update the deck when a change affects the public workflow, numerical method,
result interpretation, research applications, or a stated limitation. Keep
release history and implementation-migration details in plans, changelogs, and
architecture notes rather than in the teaching narrative.
