# JaxCont technical presentation

This directory builds the beginner-friendly, 16:9 Beamer deck labeled
**JaxCont v0.4.0**. The final artifact is one PDF:
[`jaxcont_technical_presentation.pdf`](jaxcont_technical_presentation.pdf).

The deck distinguishes two implementation statuses:

- **Available in v0.4.0**;
- **Future work — not implemented**.

PRC/dPRC and fold/Hopf two-parameter continuation are part of the v0.4.0
capability surface. An additional **experimental in v0.4.0** maturity label
keeps LPC/PD/NS detection available while the strict `MC-LC-002` comparison
remains failing.

## Read or present the deck

Open the generated PDF in any PDF viewer. From this directory, common desktop
commands are:

```bash
xdg-open jaxcont_technical_presentation.pdf  # Linux
open jaxcont_technical_presentation.pdf      # macOS
```

The current clean build has 142 pages. Page numbers are navigation aids for
this revision; frame titles and chapter names are the durable references.

| Pages | Source | Chapter |
|---:|---|---|
| 1 | master | Title |
| 2–6 | [`01_orientation.tex`](chapters/01_orientation.tex) | Orientation and application map |
| 7–13 | [`02_continuation_fundamentals.tex`](chapters/02_continuation_fundamentals.tex) | Continuation fundamentals |
| 14–25 | [`03_pseudo_arclength.tex`](chapters/03_pseudo_arclength.tex) | Natural and pseudo-arclength continuation |
| 26–38 | [`04_api_and_jax.tex`](chapters/04_api_and_jax.tex) | Public API and JAX execution model |
| 39–52 | [`05_periodic_orbits.tex`](chapters/05_periodic_orbits.tex) | Periodic orbits and Floquet stability |
| 53–62 | [`06_hopf_classification.tex`](chapters/06_hopf_classification.tex) | Hopf refinement and criticality |
| 63–69 | [`07_codim2_solvers.tex`](chapters/07_codim2_solvers.tex) | Direct codimension-two point solvers |
| 70–83 | [`08_two_parameter_curves.tex`](chapters/08_two_parameter_curves.tex) | Two-parameter fold and Hopf curves |
| 84–90 | [`09_visualization.tex`](chapters/09_visualization.tex) | Visualization: parameter space and state space |
| 91–102 | [`10_prc_dprc.tex`](chapters/10_prc_dprc.tex) | Phase-response curves and parameter sensitivity |
| 103–117 | [`11_validation.tex`](chapters/11_validation.tex) | Validation: from residuals to independent evidence |
| 118–124 | [`12_guided_workflows.tex`](chapters/12_guided_workflows.tex) | Guided application workflows |
| 125–142 | [`13_scope_and_appendix.tex`](chapters/13_scope_and_appendix.tex) | Scope, release integrity, glossary, and technical appendix |

Pages 125–134 close the main narrative. Pages 135–142 are optional technical
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
- [`chapters/`](chapters/) contains the thirteen ordered chapter sources. Chapter
  files own their section declaration and frames.
- [`assets/`](assets/) contains tracked, reviewed snapshots used by the deck.
  Their exact sources, commands, revisions, and hashes are recorded in
  [`assets/README.md`](assets/README.md).
- [`METHOD.md`](METHOD.md) records the evidence hierarchy, status conventions,
  teaching pattern, figure policy, and validation standard.

Do not reorder chapters by renaming files alone: the master document’s
explicit `\input{chapters/01_orientation}` through
`\input{chapters/13_scope_and_appendix}` sequence is authoritative.

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

Tracked presentation inputs use two provenance classes.

The deck-specific snapshots below are governed by
[`assets/README.md`](assets/README.md):

- `assets/example_08_period_doubling.png`;
- `assets/example_09_neimark_sacker.png`;
- `assets/example_13_phase_response_curve.png`;
- `assets/example_14_prc_shooting_validation.png`;
- `assets/example_16_matcont_cubic_overlay.png`;
- `assets/example_17_matcont_vanderpol_overlay.png`;
- `assets/example_18_matcont_adaptive_control_overlay.png`;
- `assets/example_19_matcont_radial_cycle_overlay.png`;
- `assets/example_20_matcont_torbpc_overlay.png`.

The PDF, `jaxcont_logo.png`, LaTeX intermediates, and ordinary example-gallery
outputs are generated artifacts and are ignored by Git. Never hand-edit a
reviewed snapshot. When a deck-specific snapshot's source example changes:

1. run the command recorded in `assets/README.md` from `examples/`;
2. inspect the fresh gallery output and its printed diagnostics;
3. copy the reviewed output byte-for-byte into `assets/`;
4. update the provenance revision/date and SHA-256 in `assets/README.md`;
5. rebuild and inspect every frame that uses the figure.

The tracked
`../../examples/images/example_12_fitzhugh_nagumo_phase_plane.jpg` is a
truthful exception: the deck reuses this gallery JPEG in place, but the current
`example_12_fitzhugh_nagumo_phase_plane.py` only displays figures with
`plt.show()` and does not regenerate that file. The retained JPEG was added in
commit `bd7937a` and has SHA-256
`5bef74c41825727169dac2bfed33591916ab7c3d16ce2a73dd174aae06792d88`.

Verify this direct gallery input from the repository root with:

```bash
git ls-files --error-unmatch examples/images/example_12_fitzhugh_nagumo_phase_plane.jpg
sha256sum examples/images/example_12_fitzhugh_nagumo_phase_plane.jpg
MPLBACKEND=Agg python examples/example_12_fitzhugh_nagumo_phase_plane.py
```

The headless run checks current model/plot execution and the expected Hopf
stdout; it does not reproduce the JPEG bytes. If the Example 12 visual must be
refreshed, first add a deterministic save/export workflow and record its exact
command, revision, and hash, or create a deck-specific reviewed snapshot under
`assets/`. Until then, do not overwrite the retained JPEG or claim that a
display-only run regenerated it.

## Chapter-by-chapter authoring workflow

1. Confirm the claim against the evidence order in `METHOD.md`, including the
   release/future status.
2. Edit only the chapter that owns the teaching point. Change the master or
   setup file only for truly deck-wide structure or styling.
3. Refresh each affected `assets/` snapshot from its recorded command. For the
   direct Example 12 gallery exception, apply the tracked-file/hash/runtime
   checks above and add a deterministic export before replacing the image.
4. Run `make`, scan the LaTeX log, and locate the changed frame by its title in
   extracted PDF text.
5. Render the affected page range and inspect titles, equations, code, figure
   labels, badges, margins, and footer at presentation zoom.
6. Run `make verify` before handing off a complete deck change, then update the
   page table above and `SPEAKER_NOTES.md` if pagination or routes changed.

Keep one primary teaching message on each normal frame. Explain the scientific
or geometric intuition before the equations, and move interruptive engine
detail to the Chapter 13 appendix.
