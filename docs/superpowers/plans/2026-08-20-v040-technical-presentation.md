# JaxCont v0.4.0 Technical Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand and reorganize the Beamer presentation so it accurately teaches all material JaxCont v0.4.0 capabilities, behavioral contracts, validation evidence, and remaining boundaries without a slide-count limit.

**Architecture:** Insert a dedicated two-parameter-continuation chapter after direct codimension-two point solvers, renumber later chapter sources, and integrate v0.4.0 changes into the chapters that own their concepts. Rebuild evidence images from executable examples, then derive all page references from the final verified PDF.

**Tech Stack:** LaTeX/Beamer, TikZ, Python/JAX examples, Matplotlib PNG assets, `latexmk`, `pdfinfo`, `pdftotext`, Sphinx.

**Spec:** `docs/superpowers/specs/2026-08-20-v040-technical-presentation-design.md`

## Global Constraints

- The deck describes JaxCont v0.4.0 and has no target or maximum slide count.
- Green badges mean released v0.4.0; red badges mean unimplemented or explicitly experimental boundaries.
- Fold/Hopf equilibrium curves are supported; periodic-orbit PD/LPC/NS curves are not.
- `MC-LC-002` remains a visible failure and its tolerances must not be weakened.
- Public names/signatures come from `src/jaxcont`; behavior comes from tests/examples; validation claims come from a fresh CPU validation run.
- Preserve the user's pre-existing `docs/source/index.rst` edit and unrelated untracked files.
- Do not update the archived v0.3.1 DOI until a v0.4.0 Zenodo archive exists.

---

### Task 1: Renumber the chapter source structure

**Files:**
- Rename: `notes/technical_presentation/chapters/08_visualization.tex` → `09_visualization.tex`
- Rename: `notes/technical_presentation/chapters/09_prc_dprc.tex` → `10_prc_dprc.tex`
- Rename: `notes/technical_presentation/chapters/10_validation.tex` → `11_validation.tex`
- Rename: `notes/technical_presentation/chapters/11_guided_workflows.tex` → `12_guided_workflows.tex`
- Rename: `notes/technical_presentation/chapters/12_scope_and_appendix.tex` → `13_scope_and_appendix.tex`
- Modify: `notes/technical_presentation/jaxcont_technical_presentation.tex`
- Modify: the `\chapterdivider` number at the top of each renamed source

**Interfaces:**
- Consumes: the existing ordered `\input{chapters/...}` list.
- Produces: chapter-number/file-number agreement and an insertion point for `08_two_parameter_curves.tex`.

- [ ] **Step 1: Record the current chapter inputs and visible numbers**

Run:

```bash
rg -n '^\\input|^\\chapterdivider' notes/technical_presentation/jaxcont_technical_presentation.tex notes/technical_presentation/chapters
```

Expected: chapters 1–12, with visualization currently numbered 8.

- [ ] **Step 2: Rename the five chapter files and update their divider numbers**

Use filesystem renames for the files and `apply_patch` for the first-line
divider changes: 8→9, 9→10, 10→11, 11→12, and 12→13.

- [ ] **Step 3: Update the master input sequence**

Insert:

```tex
\input{chapters/08_two_parameter_curves}
```

between chapters 7 and 9, and point the later inputs at their renamed files.
Create a temporary minimal chapter 8 containing only its divider and one scope
frame so the structure can compile before its content is expanded.

- [ ] **Step 4: Verify structural consistency**

Run the Step 1 command again. Expected: one ordered chapter for every number
1–13 and no master references to the old filenames.

- [ ] **Step 5: Commit the structural change**

```bash
git add notes/technical_presentation/chapters notes/technical_presentation/jaxcont_technical_presentation.tex
git commit -m "docs(presentation): add two-parameter chapter structure"
```

---

### Task 2: Teach two-parameter fold and Hopf continuation

**Files:**
- Modify: `notes/technical_presentation/chapters/08_two_parameter_curves.tex`

**Interfaces:**
- Consumes: `fold_curve_problem`, `hopf_curve_problem`, CP/BT/GH/ZH/HH event classes, `plot_two_parameter_diagram`, and Example 15.
- Produces: a self-contained chapter that hands off from direct point refinement to curve tracing.

- [ ] **Step 1: Add the scientific distinction and reduction**

Create frames that contrast an isolated codimension-two solve with a
codimension-one curve in a two-parameter plane, then show

```tex
X_{\rm fold}=(u,p_{\rm fixed},v),\qquad q=p_{\rm free}
```

and

```tex
X_{\rm Hopf}=(u,p_{\rm fixed},q_1,q_2,\omega),\qquad q=p_{\rm free}.
```

Explain that the existing scalar-parameter continuation engine advances `q`
while the extended residual solves the other physical parameter.

- [ ] **Step 2: Add the public factory contract**

Include a fragile code frame with exact calls to `jc.fold_curve_problem` and
`jc.hopf_curve_problem`, `p_guess` of shape `(2,)`, `free=1`, and a matching
`p_span=(p_guess[1], end)` continuation call. State that eager construction
refines the supplied fold/Hopf seed and rejects nonconvergence.

- [ ] **Step 3: Add the curve-event map**

Use a table or TikZ map showing:

- Fold curve: `Cusp`, `BogdanovTakens`, `ZeroHopf`.
- Hopf curve: `BogdanovTakens`, `GeneralizedHopf`, `ZeroHopf`, `DoubleHopf`.

Explain removal of the pinned zero eigenvalue or pinned Hopf pair before
testing for an additional degeneracy, and that every event carries `raw_f`,
`free`, and an explicit `curve` kind.

- [ ] **Step 4: Add Example 15 end to end**

Show the shifted Bogdanov–Takens system, its exact fold parabola, the detected
BT point, and the `plot_two_parameter_diagram([(sol, "fold")], free=1)` call.
Use the generated Example 15 figure if it is reproducible in the verification
environment; otherwise use a TikZ parabola with explicitly labeled analytic
and detected points.

- [ ] **Step 5: Add JAX composition and limitations**

Teach that the factory is built once outside `vmap`, swept values travel
through `problem.at(args=...)`, traced endpoints use `Branch.valid`, and
codimension-two parameter sensitivities use the parameter-only implicit-root
wrappers. End with the Hopf eigenvector-anchor restart limitation and the
unsupported PD/LPC/NS curve boundary.

- [ ] **Step 6: Compile the chapter in the full deck**

Run:

```bash
make -C notes/technical_presentation
```

Expected: exit 0 with no LaTeX error or undefined control sequence.

- [ ] **Step 7: Commit the chapter**

```bash
git add notes/technical_presentation/chapters/08_two_parameter_curves.tex
git commit -m "docs(presentation): teach two-parameter continuation"
```

---

### Task 3: Integrate v0.4 capabilities into the conceptual chapters

**Files:**
- Modify: `notes/technical_presentation/chapters/01_orientation.tex`
- Modify: `notes/technical_presentation/chapters/04_api_and_jax.tex`
- Modify: `notes/technical_presentation/chapters/05_periodic_orbits.tex`
- Modify: `notes/technical_presentation/chapters/07_codim2_solvers.tex`
- Modify: `notes/technical_presentation/chapters/09_visualization.tex`

**Interfaces:**
- Consumes: the new chapter 8 terminology and v0.4 public contracts.
- Produces: consistent introductions, handoffs, API diagrams, and scope claims.

- [ ] **Step 1: Update orientation and route selection**

Add “How do local events move in a two-parameter plane?” to the scientific
question map, route two-parameter users through chapters 7–9, and include
curve tracing in the outcomes frame.

- [ ] **Step 2: Expand the public API tree**

Add curve factories, curve events, `plot_two_parameter_diagram`, PRC helpers,
and `ContinuationSolution.save/load`. Add a seed-integrity frame stating that
`p_span[0]` must equal `problem.p0` and the seed is Newton-corrected/validated
before branch construction.

- [ ] **Step 3: Update periodic-event refinement**

Add a frame or focused block showing interpolation → periodic Newton
correction → Floquet evaluation → event test. Preserve the experimental label
for LPC/PD/NS detection because `MC-LC-002` remains failing.

- [ ] **Step 4: Correct the codimension-two handoff**

Replace “does not trace curves” as a library-wide limitation with the narrower
statement that direct point solvers themselves refine one point, while chapter
8's curve factories trace fold/Hopf families through such organizers.

- [ ] **Step 5: Expand visualization taxonomy**

Add the physical parameter-plane view and exact
`plot_two_parameter_diagram([(solution, "fold"), ...], free=...)` interface.
Keep it distinct from a one-parameter bifurcation diagram, state projection,
and frozen-parameter phase plane.

- [ ] **Step 6: Run stale-claim and compile checks**

```bash
rg -n -i 'no.*two-parameter|does not.*two-parameter|two-parameter.*future|cannot.*curve' notes/technical_presentation/chapters
make -C notes/technical_presentation
```

Expected: only accurately scoped periodic-curve or direct-solver limitations;
build exit 0.

- [ ] **Step 7: Commit the integration**

```bash
git add notes/technical_presentation/chapters
git commit -m "docs(presentation): integrate v0.4 capability contracts"
```

---

### Task 4: Add the MatCont visual comparison gallery

**Files:**
- Create: `notes/technical_presentation/assets/example_16_matcont_cubic_overlay.png`
- Create: `notes/technical_presentation/assets/example_17_matcont_vanderpol_overlay.png`
- Create: `notes/technical_presentation/assets/example_18_matcont_adaptive_control_overlay.png`
- Create: `notes/technical_presentation/assets/example_19_matcont_radial_cycle_overlay.png`
- Create: `notes/technical_presentation/assets/example_20_matcont_torbpc_overlay.png`
- Modify: `notes/technical_presentation/assets/README.md`
- Modify: `notes/technical_presentation/chapters/11_validation.tex`

**Interfaces:**
- Consumes: Examples 16–20 and reviewed MatCont CSV/JSON references.
- Produces: five provenance-recorded visual assets and slides that distinguish diagnostic overlays from pass/fail validation.

- [ ] **Step 1: Regenerate the five overlays in a temporary output directory**

Run each example with CPU JAX, a writable Matplotlib cache, and a temporary
working directory. Capture the exact revision and commands used.

- [ ] **Step 2: Copy the generated PNGs into presentation assets**

Use the canonical filenames listed above, then record SHA-256 hashes with:

```bash
sha256sum notes/technical_presentation/assets/example_{16,17,18,19,20}_*.png
```

- [ ] **Step 3: Record provenance**

Extend `assets/README.md` with source module, command, revision, date, and hash
for every new image. State that Example 20 is a known-failing diagnostic.

- [ ] **Step 4: Add gallery interpretation slides**

Add frames for equilibrium folds/Hopf comparisons, periodic-branch
comparisons, and the torBPC failure. State that visual overlap is supportive
but the interpolation/event/spectrum tolerance CLI is authoritative.

- [ ] **Step 5: Update the validation matrix**

Show eight supported cases: seven PASS and `MC-LC-002` FAIL, including
`MC-PRC-001`. Preserve the approximately `1.13e-2` multiplier discrepancy and
missing correctly located LPC/PD labels.

- [ ] **Step 6: Build and visually inspect the PDF pages**

Run `make -C notes/technical_presentation`, locate the gallery pages using
`pdftotext`, render those pages to images if needed, and check that plots,
legends, and failure labels are legible.

- [ ] **Step 7: Commit gallery sources and assets**

```bash
git add notes/technical_presentation/assets notes/technical_presentation/chapters/11_validation.tex
git commit -m "docs(presentation): add MatCont visual comparison gallery"
```

---

### Task 5: Teach release integrity and changed contracts

**Files:**
- Modify: `notes/technical_presentation/chapters/12_guided_workflows.tex`
- Modify: `notes/technical_presentation/chapters/13_scope_and_appendix.tex`

**Interfaces:**
- Consumes: v0.4 changelog contracts and the new two-parameter workflow.
- Produces: actionable migration, persistence, diagnostics, and compatibility slides.

- [ ] **Step 1: Add a two-parameter guided workflow**

Cover model/parameter-pair definition, checked fold/Hopf seed, curve factory,
continuation with curve events, physical-plane plot, and independent checks.

- [ ] **Step 2: Add return-shape migration slides**

Show exact v0.4 signatures:

```python
u, p, v, converged = jc.fold_point(...)
u, p, q1, q2, omega0, converged = jc.hopf_point(...)
```

State that callers must check `converged` before interpreting the iterate.

- [ ] **Step 3: Add continuation-integrity slides**

Teach corrected/validated seeds, universal start-parameter equality, true
`adaptive=False` behavior with failed-step backoff, real Newton iteration
counts, and `verbose=True` one-line summaries.

- [ ] **Step 4: Add persistence and security slides**

Explain the `format_version=1` pickle-free NPZ schema, `allow_pickle=False`,
presence flags for optional arrays, preserved convergence/label metadata, and
the incompatibility of broken pre-v0.4 archives.

- [ ] **Step 5: Update capability boundaries and installation**

List Python 3.11+, JAX/JAXlib 0.9+, Python 3.11–3.12 CI, the
`plot_phase_portrait` v0.5 removal schedule, supported fold/Hopf curves, and
unsupported branch switching/general BVP/connecting or periodic codim-two
curves.

- [ ] **Step 6: Compile and scan migration language**

```bash
make -C notes/technical_presentation
pdftotext notes/technical_presentation/jaxcont_technical_presentation.pdf - | rg 'format_version=1|Python 3.11|adaptive=False|converged'
```

Expected: all four contracts appear in extracted text and the build exits 0.

- [ ] **Step 7: Commit release-integrity teaching**

```bash
git add notes/technical_presentation/chapters/12_guided_workflows.tex notes/technical_presentation/chapters/13_scope_and_appendix.tex
git commit -m "docs(presentation): teach v0.4 release integrity"
```

---

### Task 6: Synchronize presentation documentation and speaker guidance

**Files:**
- Modify: `notes/technical_presentation/README.md`
- Modify: `notes/technical_presentation/METHOD.md`
- Modify: `notes/technical_presentation/SPEAKER_NOTES.md`
- Modify: `notes/technical_presentation/jaxcont_technical_presentation.tex`
- Modify: `docs/source/presentation.md`

**Interfaces:**
- Consumes: the final chapter contents and final PDF pagination.
- Produces: accurate routes, chapter ranges, demo cues, status language, and Sphinx-facing description.

- [ ] **Step 1: Update title and subtitle**

Make the subtitle include two-parameter curves and release-integrity evidence
without turning the title page into a feature list.

- [ ] **Step 2: Rewrite chapter and route summaries**

Update the README and speaker notes for 13 chapters, add Example 15 and
Examples 16–20 cues, and remove all old chapter-number references.

- [ ] **Step 3: Rebuild and derive pagination**

Run:

```bash
make -C notes/technical_presentation verify
pdfinfo notes/technical_presentation/jaxcont_technical_presentation.pdf | rg '^Pages:'
```

Use the PDF's table of contents and `pdftotext` output to determine each
chapter's exact page range; do not estimate.

- [ ] **Step 4: Apply exact page ranges**

Update every total-page and page-range reference in README and speaker notes,
including presentation routes and optional appendix pages.

- [ ] **Step 5: Run a stale-reference scan**

```bash
rg -n '(112|113|v0\.3\.1\+|current.main|planned.for.v0\.4|chapters/(08_visualization|09_prc_dprc|10_validation|11_guided_workflows|12_scope))' notes/technical_presentation docs/source/presentation.md
```

Expected: no stale matches.

- [ ] **Step 6: Commit synchronized guidance**

```bash
git add notes/technical_presentation docs/source/presentation.md
git commit -m "docs(presentation): synchronize v0.4 deck guidance"
```

---

### Task 7: Final release verification

**Files:**
- Verify: all files modified by Tasks 1–6

**Interfaces:**
- Consumes: the complete deck, assets, and documentation.
- Produces: evidence that the presentation is buildable, internally consistent, and visible in Sphinx.

- [ ] **Step 1: Run the presentation's clean verification target**

```bash
make -C notes/technical_presentation verify
```

Expected: exit 0, positive page count, no LaTeX errors, and required v0.4 text.

- [ ] **Step 2: Verify required PDF content**

```bash
pdftotext notes/technical_presentation/jaxcont_technical_presentation.pdf - | rg 'Two-parameter fold and Hopf curves|fold_curve_problem|plot_two_parameter_diagram|MC-LC-002|format_version=1|JaxCont v0.4.0'
```

Expected: every required phrase appears.

- [ ] **Step 3: Run the authoritative validation command**

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl-jaxcont-validation python3 -m examples.MatCont.run_validation
```

Expected: seven PASS lines and the documented `MC-LC-002` FAIL; overall exit
status 1 is expected specifically because that case remains failing.

- [ ] **Step 4: Build Sphinx with warnings as errors**

Use a compatible isolated environment and run:

```bash
sphinx-build -M html docs/source docs/build -W --keep-going
```

Expected: `build succeeded` and exit 0.

- [ ] **Step 5: Run final consistency checks**

```bash
git diff --check
rg -n -i 'current main|planned for v0\.4|v0\.4 is not released|two-parameter continuation.*unsupported' notes/technical_presentation docs/source/presentation.md
git status --short
```

Expected: no whitespace errors, no stale status claims, and only intentional
working-tree changes plus the user's pre-existing files.

- [ ] **Step 6: Commit final verification-only corrections if needed**

If verification required wording or layout corrections, stage only those
presentation files and commit:

```bash
git commit -m "docs(presentation): finalize v0.4 technical deck"
```
