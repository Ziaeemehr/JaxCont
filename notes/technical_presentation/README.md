# JaxCont technical presentation

This directory builds the beginner-friendly, 16:9 Beamer deck labeled
**JaxCont v0.3.1+**. The final artifact is one PDF:
[`jaxcont_technical_presentation.pdf`](jaxcont_technical_presentation.pdf).

The label is intentionally not a package release number. “JaxCont v0.3.1+”
means the published v0.3.1 release plus selected features on current `main`
that are intended for v0.4. The deck distinguishes three source statuses:

- **Available in v0.3.1**;
- **Current main — planned for v0.4**;
- **Future work — not implemented**.

In particular, PRC/dPRC material is current-main/planned-for-v0.4. The deck
does not claim that v0.4 has been released.

## Read or present the deck

Open the generated PDF in any PDF viewer. From this directory, common desktop
commands are:

```bash
xdg-open jaxcont_technical_presentation.pdf  # Linux
open jaxcont_technical_presentation.pdf      # macOS
```

The current clean build has 112 pages. Page numbers are navigation aids for
this revision; frame titles and chapter names are the durable references.

| Pages | Source | Chapter |
|---:|---|---|
| 1 | master | Title |
| 2–6 | [`01_orientation.tex`](chapters/01_orientation.tex) | Orientation and application map |
| 7–13 | [`02_continuation_fundamentals.tex`](chapters/02_continuation_fundamentals.tex) | Continuation fundamentals |
| 14–25 | [`03_pseudo_arclength.tex`](chapters/03_pseudo_arclength.tex) | Natural and pseudo-arclength continuation |
| 26–37 | [`04_api_and_jax.tex`](chapters/04_api_and_jax.tex) | Public API and JAX execution model |
| 38–50 | [`05_periodic_orbits.tex`](chapters/05_periodic_orbits.tex) | Periodic orbits and Floquet stability |
| 51–60 | [`06_hopf_classification.tex`](chapters/06_hopf_classification.tex) | Hopf refinement and criticality |
| 61–67 | [`07_codim2_solvers.tex`](chapters/07_codim2_solvers.tex) | Direct codimension-two point solvers |
| 68–73 | [`08_visualization.tex`](chapters/08_visualization.tex) | Visualization: parameter space and state space |
| 74–85 | [`09_prc_dprc.tex`](chapters/09_prc_dprc.tex) | Phase-response curves and parameter sensitivity |
| 86–94 | [`10_validation.tex`](chapters/10_validation.tex) | Validation: from residuals to independent evidence |
| 95–100 | [`11_guided_workflows.tex`](chapters/11_guided_workflows.tex) | Guided application workflows |
| 101–112 | [`12_scope_and_appendix.tex`](chapters/12_scope_and_appendix.tex) | Scope, glossary, and technical appendix |

Pages 101–105 close the main narrative. Pages 106–112 are optional technical
appendix material.

Suggested overview, methods, and complete presentation routes, including
timing and live-demo guidance, are in
[`SPEAKER_NOTES.md`](SPEAKER_NOTES.md).

## Source layout

- [`jaxcont_technical_presentation.tex`](jaxcont_technical_presentation.tex)
  owns metadata, the title frame, chapter order, and the document boundary.
- [`presentation_setup.tex`](presentation_setup.tex) owns packages, the Madrid
  theme, JaxCont colors, listings, TikZ libraries, status badges, and reusable
  chapter-divider macros.
- [`chapters/`](chapters/) contains the twelve ordered chapter sources. Chapter
  files own their section declaration and frames.
- [`assets/`](assets/) contains tracked, reviewed snapshots used by the deck.
  Their exact sources, commands, revisions, and hashes are recorded in
  [`assets/README.md`](assets/README.md).
- [`METHOD.md`](METHOD.md) records the evidence hierarchy, status conventions,
  teaching pattern, figure policy, and validation standard.

Do not reorder chapters by renaming files alone: the master document’s
explicit `\input{chapters/01_orientation}` through
`\input{chapters/12_scope_and_appendix}` sequence is authoritative.

## Build and verification

Run commands from this directory:

```bash
make
```

This incrementally builds `jaxcont_technical_presentation.pdf`, regenerating
the ignored `jaxcont_logo.png` from `../../docs/images/logo/logo.svg` when
needed.

```bash
make rebuild
```

This removes the PDF and LaTeX intermediates, then performs a clean build.

```bash
make verify
```

This performs the clean rebuild plus the automated PDF page-count, log, and
required-text checks. It does not replace manual review of overfull-box
warnings or rendered pages.

```bash
make clean
make distclean
```

`make clean` removes LaTeX intermediates while preserving the PDF.
`make distclean` also removes the generated PDF. Neither command removes the
reviewed snapshots under `assets/`.

The build requires `latexmk`, pdfTeX, and ImageMagick’s `convert`. The verify
target additionally uses Poppler’s `pdfinfo`/`pdftotext` and `rg`.

## Tracked and generated images

Tracked presentation inputs are source-controlled evidence:

- `assets/example_08_period_doubling.png`;
- `assets/example_09_neimark_sacker.png`;
- `assets/example_13_phase_response_curve.png`;
- `assets/example_14_prc_shooting_validation.png`;
- `../../examples/images/example_12_fitzhugh_nagumo_phase_plane.jpg`, reused
  directly from the example gallery.

The PDF, `jaxcont_logo.png`, LaTeX intermediates, and ordinary example-gallery
outputs are generated artifacts and are ignored by Git. Never hand-edit a
reviewed snapshot. When its source example changes:

1. run the command recorded in `assets/README.md` from `examples/`;
2. inspect the fresh gallery output and its printed diagnostics;
3. copy the reviewed output byte-for-byte into `assets/`;
4. update the provenance revision/date and SHA-256 in `assets/README.md`;
5. rebuild and inspect every frame that uses the figure.

## Chapter-by-chapter authoring workflow

1. Confirm the claim against the evidence order in `METHOD.md`, including the
   release/current-main/future status.
2. Edit only the chapter that owns the teaching point. Change the master or
   setup file only for truly deck-wide structure or styling.
3. Refresh every affected repository-generated figure from its source example;
   do not copy an older plot merely because the filename matches.
4. Run `make`, scan the LaTeX log, and locate the changed frame by its title in
   extracted PDF text.
5. Render the affected page range and inspect titles, equations, code, figure
   labels, badges, margins, and footer at presentation zoom.
6. Run `make verify` before handing off a complete deck change, then update the
   page table above and `SPEAKER_NOTES.md` if pagination or routes changed.

Keep one primary teaching message on each normal frame. Explain the scientific
or geometric intuition before the equations, and move interruptive engine
detail to the Chapter 12 appendix.
