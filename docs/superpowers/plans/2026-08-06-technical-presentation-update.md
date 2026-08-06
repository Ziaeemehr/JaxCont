# JaxCont v0.3.1+ Technical Presentation Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the JaxCont technical presentation as one newcomer-friendly v0.3.1+ PDF assembled from modular Beamer chapters, with accurate coverage of released and current-main features, real application figures, reproducible demos, and matching speaker/maintenance documentation.

**Architecture:** Keep one Beamer master document and one shared setup file, then load twelve ordered chapter files with `\input`. Preserve the existing Madrid/JaxCont visual system, reuse the strongest current slides, add focused v0.3.1+ chapters, and track every non-TikZ figure required for a clean build. Verify the deck through example execution, clean LaTeX builds, log checks, PDF metadata, text extraction, and representative rendered-page review.

**Tech Stack:** LaTeX Beamer, TikZ, `listings`, Matplotlib-generated repository examples, GNU Make, `latexmk`/pdfTeX, ImageMagick, Poppler (`pdfinfo`, `pdftotext`, `pdftoppm`), Python/JAX/JaxCont

## Global Constraints

- The final output is one PDF named `notes/technical_presentation/jaxcont_technical_presentation.pdf`.
- Use the deck-wide label **“JaxCont v0.3.1+”**.
- Define “+” as the published v0.3.1 release plus current-main features intended for v0.4; never present v0.4 as released.
- Distinguish **Available in v0.3.1**, **Current main — planned for v0.4**, and **Future work — not implemented**.
- Preserve the existing 16:9 Madrid theme, JaxCont palette, title treatment, footer, block styles, Python listings, and TikZ conventions.
- Each normal slide has one primary teaching message.
- Explain visual/scientific intuition before equations and move interruptive implementation detail to the appendix.
- Use real repository outputs for application claims and keep their provenance reproducible.
- Do not imply support for branch switching, two-parameter curve continuation, automatic cycle discovery from a Hopf point, general BVP continuation, or connecting-orbit continuation.
- PRC/dPRC stays marked as current-main/planned-for-v0.4 unless `src/jaxcont/_version.py` and `CHANGELOG.md` change before final verification.
- Do not edit JaxCont numerical behavior or public APIs as part of this presentation task.

---

## File map

**Create**

- `notes/technical_presentation/presentation_setup.tex` — packages, theme, colors, macros, listings, paths, and shared chapter/status components.
- `notes/technical_presentation/chapters/01_orientation.tex` — title-adjacent orientation, application map, status legend, and routes.
- `notes/technical_presentation/chapters/02_continuation_fundamentals.tex` — roots, branches, simulation distinction, and local stability.
- `notes/technical_presentation/chapters/03_pseudo_arclength.tex` — natural continuation, fold failure, pseudo-arclength geometry, equations, and recap.
- `notes/technical_presentation/chapters/04_api_and_jax.tex` — public API, results, events/solvers, functional engine, JIT, batching, and differentiation.
- `notes/technical_presentation/chapters/05_periodic_orbits.tex` — collocation, phase condition, Floquet stability, PD/NS events, demos, and limits.
- `notes/technical_presentation/chapters/06_hopf_classification.tex` — Hopf refinement, frequency, first Lyapunov coefficient, API, interpretation, and limits.
- `notes/technical_presentation/chapters/07_codim2_solvers.tex` — CP/BT/GH/ZH/HH taxonomy, direct-solver contract, API, results, and boundaries.
- `notes/technical_presentation/chapters/08_visualization.tex` — visualization surface, three complementary views, phase-plane architecture, and example.
- `notes/technical_presentation/chapters/09_prc_dprc.tex` — phase sensitivity, adjoint method, iPRC/dPRC/branch APIs, plots, and limitations.
- `notes/technical_presentation/chapters/10_validation.tex` — validation ladder, oracle types, MatCont/BifurcationKit/shooting evidence, and convention alignment.
- `notes/technical_presentation/chapters/11_guided_workflows.tex` — example learning path, six demo cards, research workflow, and reading results.
- `notes/technical_presentation/chapters/12_scope_and_appendix.tex` — capability boundaries, glossary, takeaways, derivations, engine pseudocode, and sources.
- `notes/technical_presentation/assets/README.md` — source command, source file, date, and intended slide for each committed figure.
- `notes/technical_presentation/assets/example_08_period_doubling.png` — committed periodic-event output.
- `notes/technical_presentation/assets/example_09_neimark_sacker.png` — committed periodic-event output.
- `notes/technical_presentation/assets/example_13_phase_response_curve.png` — committed iPRC/dPRC output.
- `notes/technical_presentation/assets/example_14_prc_shooting_validation.png` — committed independent-validation output.

**Modify**

- `notes/technical_presentation/jaxcont_technical_presentation.tex:1-1666` — replace the monolith with metadata, document boundary, title frame, and ordered chapter inputs.
- `notes/technical_presentation/Makefile:1-30` — declare chapter/setup/image dependencies and add a reproducible `verify` target.
- `notes/technical_presentation/README.md:1-44` — document modular authoring, assets, commands, and output.
- `notes/technical_presentation/METHOD.md:1-74` — record the new evidence hierarchy, chapter pattern, status semantics, and figure/validation policy.
- `notes/technical_presentation/SPEAKER_NOTES.md:1-165` — reorganize notes by chapter and add overview/methods/complete routes plus demo cautions.
- `examples/example_13_phase_response_curve.py:53-65` — give the dPRC axis an exact parameter name and save the reviewed two-panel figure.
- `examples/example_14_prc_shooting_validation.py:281-316` — save the reviewed four-panel comparison figure.
- `.gitignore:102-104` — allow presentation PNG assets while leaving other generated plots ignored.

**Evidence consulted**

- `src/jaxcont/api.py` and `src/jaxcont/__init__.py` — equilibrium public API and exports.
- `src/jaxcont/problems/periodic.py` — periodic-problem factory.
- `src/jaxcont/bifurcations/hopf_normal_form.py` — Hopf point and `l1` contracts.
- `src/jaxcont/bifurcations/codim2.py` — CP/BT/GH/ZH/HH contracts.
- `src/jaxcont/stability/prc.py` — iPRC, branch iPRC, and dPRC contracts.
- `src/jaxcont/viz/` — plotting contracts.
- `examples/example_08_period_doubling.py`, `example_09_neimark_sacker.py`, and `example_12_fitzhugh_nagumo_phase_plane.py` — demo behavior and figures.
- `examples/MatCont/README.md` and `docs/source/validation.md` — validation claims and limitations.
- `CHANGELOG.md`, `notes/ROADMAP.md`, and `src/jaxcont/_version.py` — release status.

---

### Task 1: Capture the baseline and make presentation assets trackable

**Files:**

- Modify: `.gitignore:102-104`
- Create: `notes/technical_presentation/assets/README.md`

**Interfaces:**

- Consumes: the current 72-page PDF and the existing ignored `examples/images/*.png` outputs.
- Produces: a tracked `assets/` location accepted by Git and a provenance contract used by Tasks 6, 10, and 11.

- [ ] **Step 1: Record the current baseline**

Run:

```bash
make -C notes/technical_presentation rebuild
pdfinfo notes/technical_presentation/jaxcont_technical_presentation.pdf | rg 'Pages|Page size'
pdftotext notes/technical_presentation/jaxcont_technical_presentation.pdf - | sed -n '1,20p'
```

Expected: build succeeds, `Pages: 72`, page size is 16:9, and the extracted title is “JaxCont Technical Presentation”.

- [ ] **Step 2: Add the narrow Git ignore exception**

Append directly after `*.png`:

```gitignore
!notes/technical_presentation/assets/*.png
```

Do not unignore PNG files elsewhere.

- [ ] **Step 3: Add the asset provenance policy**

Create `assets/README.md` with a table containing these exact rows:

| Asset | Produced from | Regeneration command | Used in |
|---|---|---|---|
| `example_08_period_doubling.png` | `examples/example_08_period_doubling.py` | `MPLBACKEND=Agg python example_08_period_doubling.py` from `examples/` | periodic-orbit chapter |
| `example_09_neimark_sacker.png` | `examples/example_09_neimark_sacker.py` | `MPLBACKEND=Agg python example_09_neimark_sacker.py` from `examples/` | periodic-orbit chapter |
| `example_13_phase_response_curve.png` | `examples/example_13_phase_response_curve.py` | `MPLBACKEND=Agg python example_13_phase_response_curve.py` from `examples/` | PRC chapter |
| `example_14_prc_shooting_validation.png` | `examples/example_14_prc_shooting_validation.py` | `MPLBACKEND=Agg python example_14_prc_shooting_validation.py` from `examples/` | validation chapter |

State that assets are reviewed snapshots, must be refreshed when their source example changes, and must not be edited by hand.

- [ ] **Step 4: Verify the ignore rule**

Run:

```bash
touch notes/technical_presentation/assets/ignore-check.png
git check-ignore notes/technical_presentation/assets/ignore-check.png
```

Expected: `git check-ignore` exits nonzero because the path is no longer ignored. Remove only `ignore-check.png` after the check.

- [ ] **Step 5: Commit**

```bash
git add .gitignore notes/technical_presentation/assets/README.md
git commit -m "docs(presentation): define tracked figure assets"
```

---

### Task 2: Split the monolithic deck without changing its teaching content

**Files:**

- Create: `notes/technical_presentation/presentation_setup.tex`
- Create: all twelve `notes/technical_presentation/chapters/*.tex` files
- Modify: `notes/technical_presentation/jaxcont_technical_presentation.tex:1-1666`
- Modify: `notes/technical_presentation/Makefile:1-30`

**Interfaces:**

- Consumes: the current preamble and 72 frames.
- Produces: a master → setup/chapters dependency graph. Every later task edits only its chapter plus shared documentation when necessary.

- [ ] **Step 1: Add a structural regression check**

Before splitting, save these expected invariants in the work log:

```text
PDF filename: jaxcont_technical_presentation.pdf
Page count: 72
First title: JaxCont Technical Presentation
Last title: Appendix: sources used for this deck
```

- [ ] **Step 2: Extract the shared setup**

Move the existing package imports, TikZ libraries, colors, Beamer colors/templates, listing settings, and mathematical macros from lines 1–57 into `presentation_setup.tex`. Add:

```tex
\graphicspath{{assets/}{../../examples/images/}}

\newcommand{\releasedbadge}{%
  \colorbox{JaxGreen!16}{\strut\textcolor{JaxGreen!55!black}{\scriptsize\bfseries Available in v0.3.1}}}
\newcommand{\mainbadge}{%
  \colorbox{JaxOrange!16}{\strut\textcolor{JaxOrange!80!black}{\scriptsize\bfseries Current main --- planned for v0.4}}}
\newcommand{\futurebadge}{%
  \colorbox{JaxRed!12}{\strut\textcolor{JaxRed}{\scriptsize\bfseries Future work --- not implemented}}}

\newcommand{\chapterdivider}[4]{%
  \section{#2}
  \begin{frame}[plain]
    \vfill
    {\color{JaxBlue}\Large\bfseries Chapter #1\par}
    \vspace{2mm}
    {\huge\bfseries #2\par}
    \vspace{5mm}
    \begin{beamercolorbox}[sep=8pt,rounded=true]{block body}
      \textbf{Question:} #3
    \end{beamercolorbox}
    \vspace{3mm}
    {\small\textcolor{JaxTeal}{#4}\par}
    \vfill
  \end{frame}
}
```

- [ ] **Step 3: Replace the monolith with the master**

Keep title metadata in the master, change the subtitle to `From continuation fundamentals to codimension-two points and phase response curves`, and change the date line to:

```tex
\date{JaxCont v0.3.1+}
```

After the title frame, load the chapters in this exact order:

```tex
\input{chapters/01_orientation}
\input{chapters/02_continuation_fundamentals}
\input{chapters/03_pseudo_arclength}
\input{chapters/04_api_and_jax}
\input{chapters/05_periodic_orbits}
\input{chapters/06_hopf_classification}
\input{chapters/07_codim2_solvers}
\input{chapters/08_visualization}
\input{chapters/09_prc_dprc}
\input{chapters/10_validation}
\input{chapters/11_guided_workflows}
\input{chapters/12_scope_and_appendix}
```

- [ ] **Step 4: Move existing frames by responsibility**

Use this exact mapping from the old source:

- lines 88–203 → `01_orientation.tex`;
- lines 204–306 → `02_continuation_fundamentals.tex`;
- lines 307–503 → `03_pseudo_arclength.tex`;
- lines 504–728 and 1007–1278 → `04_api_and_jax.tex`;
- lines 729–1006 → `05_periodic_orbits.tex`;
- create `06_hopf_classification.tex` and `07_codim2_solvers.tex` with only their chapter-divider calls;
- lines 1279–1367 → `08_visualization.tex`;
- create `09_prc_dprc.tex` and `10_validation.tex` with only their chapter-divider calls;
- lines 1368–1500 → `11_guided_workflows.tex`;
- lines 1500–1664 → `12_scope_and_appendix.tex`.

Remove old `\section` commands where `\chapterdivider` supplies the section.

- [ ] **Step 5: Declare all source dependencies in Make**

Define:

```make
SETUP := presentation_setup.tex
CHAPTERS := $(wildcard chapters/*.tex)
ASSETS := $(wildcard assets/*)

$(PDF): $(TEX) $(SETUP) $(CHAPTERS) $(ASSETS) $(LOGO_PNG)
```

- [ ] **Step 6: Verify the structural refactor**

Run:

```bash
make -C notes/technical_presentation rebuild
pdfinfo notes/technical_presentation/jaxcont_technical_presentation.pdf | rg 'Pages:'
pdftotext notes/technical_presentation/jaxcont_technical_presentation.pdf - | rg 'What you should be able to explain|Appendix: sources used'
```

Expected: compilation succeeds; the old 72 content frames plus the new chapter dividers are present; all original titled frames are still discoverable. Record the new page count as the modular baseline.

- [ ] **Step 7: Commit**

```bash
git add notes/technical_presentation
git commit -m "refactor(presentation): split deck into modular chapters"
```

---

### Task 3: Rebuild the orientation around JaxCont v0.3.1+

**Files:**

- Modify: `notes/technical_presentation/chapters/01_orientation.tex`

**Interfaces:**

- Consumes: `\releasedbadge`, `\mainbadge`, `\futurebadge`, and `\chapterdivider`.
- Produces: the vocabulary and status semantics used by every later chapter.

- [ ] **Step 1: Add the chapter divider**

Use:

```tex
\chapterdivider{1}{Orientation and application map}
  {What can JaxCont compute, and where should a newcomer begin?}
  {Featured application: one model viewed as branches, cycles, phase geometry, and phase sensitivity}
```

- [ ] **Step 2: Replace release-history-first slides with a status legend**

Create a frame titled `How to read “v0.3.1+”` with three rows:

- `\releasedbadge` — continuation, periodic orbits, Floquet events, Hopf classification, direct codim-2 solvers, phase planes, and validation suite;
- `\mainbadge` — `prc_curve`, `branch_prc`, `dprc_curve`, `plot_prc`, and Examples 13–14;
- `\futurebadge` — branch switching, two-parameter curve continuation, general BVP/connecting-orbit workflows.

The closing sentence must say that “+” is a source-status marker, not a released version number.

- [ ] **Step 3: Add an application-first capability map**

Use a TikZ flow with these five questions:

```text
Where do steady states exist? → equilibrium continuation
Where do oscillations exist? → periodic-orbit continuation
Where does stability change? → eigenvalues, Floquet multipliers, events
What kind of local organizing point is this? → Hopf l1 and direct codim-2 refinement
How does an oscillator respond to perturbations? → iPRC and dPRC
```

- [ ] **Step 4: Add the three presentation routes**

Create one frame with:

- Overview route: Chapters 1, 2, selected 3, 5, 8, 11, 12;
- Methods route: Chapters 2–7 and 10;
- Complete route: Chapters 1–12.

Do not promise exact talk duration until the final page count is known.

- [ ] **Step 5: Rewrite the learning outcomes**

Use the ten outcomes from the approved design specification, split across two columns. Keep each outcome to one line where possible and move explanatory wording to speaker notes.

- [ ] **Step 6: Compile and inspect the orientation pages**

Run:

```bash
make -C notes/technical_presentation
pdftotext notes/technical_presentation/jaxcont_technical_presentation.pdf - | rg 'How to read|planned for v0.4|application map'
```

Expected: all three status phrases and all five application questions appear in extracted text.

- [ ] **Step 7: Commit**

```bash
git add notes/technical_presentation/chapters/01_orientation.tex
git commit -m "docs(presentation): orient newcomers to v0.3.1+"
```

---

### Task 4: Tighten the continuation fundamentals and pseudo-arclength chapters

**Files:**

- Modify: `notes/technical_presentation/chapters/02_continuation_fundamentals.tex`
- Modify: `notes/technical_presentation/chapters/03_pseudo_arclength.tex`

**Interfaces:**

- Consumes: the scalar cubic `F(u,p)=p+u-u^3/3` and existing TikZ branch geometry.
- Produces: the conceptual foundation assumed by periodic, Hopf, codim-2, and workflow chapters.

- [ ] **Step 1: Add chapter dividers**

Use:

```tex
\chapterdivider{2}{Continuation fundamentals}
  {How do we follow a connected family of solutions instead of solving isolated points?}
  {Featured application: the scalar cubic branch and its two folds}

\chapterdivider{3}{Natural and pseudo-arclength continuation}
  {How can a branch follower pass a point where the physical parameter turns around?}
  {Featured application: predictor--corrector continuation through a saddle-node fold}
```

- [ ] **Step 2: Enforce the visual-before-equation sequence**

Keep this order in Chapter 2:

1. continuation versus simulation versus one root;
2. branch as a geometric curve;
3. why nearby solutions help;
4. Jacobian and local sensitivity;
5. fold picture;
6. “What to remember”.

Keep this order in Chapter 3:

1. natural continuation;
2. fixed-parameter slice failure;
3. pseudo-arclength idea;
4. predictor/corrector geometry;
5. augmented equation;
6. tangent bordered system;
7. Newton bordered system;
8. one-step state machine;
9. adaptive step size;
10. method-selection table;
11. “What to remember”.

- [ ] **Step 3: Reduce dense text**

For every frame in these chapters, cap the visible structure at one equation block plus one diagram/block, or two short columns. Move the “failed attempt consumes an outer iteration” detail to Chapter 12’s implementation appendix.

- [ ] **Step 4: Add explicit transition statements**

End Chapter 2 with “A fold breaks the parameter coordinate, not the branch.” End Chapter 3 with “The engine can now follow any finite-dimensional residual whose input and output shapes match.”

- [ ] **Step 5: Compile and inspect**

Run:

```bash
make -C notes/technical_presentation
rg -n 'Overfull \\\\vbox|LaTeX Error' notes/technical_presentation/jaxcont_technical_presentation.log
```

Expected: no LaTeX errors and no overfull frame boxes attributable to Chapters 2–3.

- [ ] **Step 6: Commit**

```bash
git add notes/technical_presentation/chapters/02_continuation_fundamentals.tex notes/technical_presentation/chapters/03_pseudo_arclength.tex
git commit -m "docs(presentation): streamline continuation foundations"
```

---

### Task 5: Reframe the public API and JAX execution chapter

**Files:**

- Modify: `notes/technical_presentation/chapters/04_api_and_jax.tex`

**Interfaces:**

- Consumes: `jc.bif_problem`, `jc.continuation`, `ContinuationResult`, `ContinuationPar`, algorithms, events, solver protocols, and fixed-buffer engine concepts.
- Produces: the API/data-flow model used by every demo chapter.

- [ ] **Step 1: Add the chapter divider**

```tex
\chapterdivider{4}{Public API and JAX execution model}
  {How does a scientific model become a transformable branch computation?}
  {Featured application: one functional API used eagerly, under JIT, and across batches}
```

- [ ] **Step 2: Lead with the public call contract**

Show the exact stable front-door shape:

```python
problem = jc.bif_problem(rhs, u0, p0, args=args)
result = jc.continuation(
    problem,
    jc.PseudoArclength(),
    p_span=(p_min, p_max),
    settings=jc.ContinuationPar(ds=0.02, max_steps=300),
    events=[jc.Fold(), jc.Hopf()],
)
```

Do not use removed object-oriented APIs.

- [ ] **Step 3: Add a result-reading diagram**

Draw:

```text
ContinuationResult
├── branch.states / params / stability / eigenvalues
├── branch.valid / n_valid
└── events → kind / state / parameter / diagnostics
```

State that eager results may be trimmed while traced results retain fixed buffers plus a validity mask.

- [ ] **Step 4: Consolidate implementation slides**

Retain:

- “scan” naming clarification: outer loop is `jax.lax.while_loop`;
- whole-loop JIT and cold/warm distinction;
- fixed buffers and validity masks;
- `vmap` across independent branches;
- autodiff roles and implicit roots;
- solver protocols and bounded termination.

Move detailed carry-field names, branch-free control mechanics, and eager/traced reassembly internals to Chapter 12.

- [ ] **Step 5: Add a benefit-versus-nonbenefit table**

Rows:

| JAX mechanism | Enables | Does not guarantee |
|---|---|---|
| autodiff | Jacobians and sensitivities | good conditioning |
| JIT + `while_loop` | compiled branch execution | a more accurate method |
| fixed shapes | transformation compatibility | every event path is batchable |
| `vmap` | independent ensemble branches | parallel steps within one branch |
| custom VJP root | implicit reverse-mode gradients | convergence from a poor seed |

- [ ] **Step 6: Verify code names against source**

Run:

```bash
python - <<'PY'
import jaxcont as jc
for name in ["bif_problem", "continuation", "PseudoArclength",
             "ContinuationPar", "Fold", "Hopf"]:
    assert hasattr(jc, name), name
print("public API names verified")
PY
```

Expected: `public API names verified`.

- [ ] **Step 7: Commit**

```bash
git add notes/technical_presentation/chapters/04_api_and_jax.tex
git commit -m "docs(presentation): clarify the public API and JAX model"
```

---

### Task 6: Upgrade the periodic-orbit chapter with real PD/NS figures

**Files:**

- Modify: `notes/technical_presentation/chapters/05_periodic_orbits.tex`
- Create: `notes/technical_presentation/assets/example_08_period_doubling.png`
- Create: `notes/technical_presentation/assets/example_09_neimark_sacker.png`
- Modify: `notes/technical_presentation/assets/README.md`

**Interfaces:**

- Consumes: `periodic_orbit_problem`, `Collocation`, Floquet multipliers, `PeriodDoubling`, `NeimarkSacker`, and Examples 08–10.
- Produces: a complete cycle workflow and two tracked event visuals.

- [ ] **Step 1: Regenerate the analytic event examples**

From `examples/` run:

```bash
MPLBACKEND=Agg python example_08_period_doubling.py
MPLBACKEND=Agg python example_09_neimark_sacker.py
```

Expected: each script exits zero, reports its event near `alpha=0`, and writes its PNG under `examples/images/`.

- [ ] **Step 2: Refresh the tracked snapshots**

Copy the two regenerated PNGs byte-for-byte into `notes/technical_presentation/assets/`. Record the refresh date and commit hash in `assets/README.md`.

- [ ] **Step 3: Add the chapter divider**

```tex
\chapterdivider{5}{Periodic orbits and Floquet stability}
  {How can the same continuation engine follow an entire oscillation?}
  {Featured applications: Van der Pol cycles, period doubling, and Neimark--Sacker crossings}
```

- [ ] **Step 4: Preserve the collocation learning sequence**

Use this frame order:

1. equilibrium unknown versus entire-cycle unknown;
2. collocation residual blocks;
3. why the phase condition is necessary;
4. user responsibility for a coarse trajectory and period;
5. minimal periodic-problem API;
6. packed branch state;
7. Floquet multiplier stability rule;
8. PD and NS event geometry;
9. Example 08 real figure;
10. Example 09 real figure;
11. practical numerical boundaries;
12. “What to remember”.

- [ ] **Step 5: Use exact status and scope language**

Mark the chapter `\releasedbadge`. State that the factory refines a supplied coarse cycle but does not discover a cycle by integrating from a Hopf point.

- [ ] **Step 6: Verify assets and compile**

Run:

```bash
test -s notes/technical_presentation/assets/example_08_period_doubling.png
test -s notes/technical_presentation/assets/example_09_neimark_sacker.png
make -C notes/technical_presentation
```

Expected: both files are nonempty and LaTeX finds both images.

- [ ] **Step 7: Commit**

```bash
git add notes/technical_presentation/chapters/05_periodic_orbits.tex notes/technical_presentation/assets
git commit -m "docs(presentation): add periodic event demonstrations"
```

---

### Task 7: Add Hopf refinement and criticality classification

**Files:**

- Modify: `notes/technical_presentation/chapters/06_hopf_classification.tex`

**Interfaces:**

- Consumes: `jc.hopf_point`, `jc.hopf_parameter`, `jc.lyapunov_coefficient`, `Hopf.refine()` diagnostics, and BifurcationKit validation evidence.
- Produces: the `l1` and criticality concepts needed by the GH explanation.

- [ ] **Step 1: Add the chapter divider and release badge**

```tex
\chapterdivider{6}{Hopf refinement and criticality}
  {A Hopf crossing was detected; where is it exactly, and what local oscillation does it organize?}
  {Featured application: refine a Hopf point, recover its frequency, and classify its criticality}
```

Place `\releasedbadge` on the first content frame.

- [ ] **Step 2: Build the conceptual sequence**

Create these frames:

1. “Detection is not refinement” — bracketed eigenvalue crossing versus extended-system root;
2. “What the Hopf solve returns” — `u*`, `p*`, `q_1`, `q_2`, `omega_0`;
3. “The extended-system picture” — equilibrium block, two eigenspace blocks, normalization/phase blocks;
4. “What the first Lyapunov coefficient answers”;
5. “Supercritical versus subcritical” — mirrored schematic amplitude diagrams;
6. “Minimal Hopf classification API”;
7. “What `Hopf.refine()` reports” — `omega0`, `l1`, `criticality`;
8. “What this still does not do”;
9. “What to remember”.

- [ ] **Step 3: Use the exact API example**

```python
u_h, p_h, q1, q2, omega0 = jc.hopf_point(
    rhs, u_guess, p_guess, args
)
l1 = jc.lyapunov_coefficient(
    rhs, u_h, p_h, q1, q2, omega0, args
)
criticality = "supercritical" if l1 < 0 else "subcritical"
```

Add a note that a near-zero `l1` is a generalized-Hopf candidate and requires tolerance-aware interpretation.

- [ ] **Step 4: State the interpretation carefully**

Use:

```text
l1 < 0 → supercritical local Hopf normal form; stable small cycle on the appropriate side
l1 > 0 → subcritical local Hopf normal form; unstable small cycle on the appropriate side
l1 ≈ 0 → degenerate/GH candidate; refine with the two-parameter GH solver
```

Avoid claiming the sign alone determines global cycle behavior.

- [ ] **Step 5: Verify exports and compile**

```bash
python - <<'PY'
import jaxcont as jc
for name in ["hopf_point", "hopf_parameter", "lyapunov_coefficient"]:
    assert hasattr(jc, name), name
print("Hopf API names verified")
PY
make -C notes/technical_presentation
```

Expected: API check and LaTeX build succeed.

- [ ] **Step 6: Commit**

```bash
git add notes/technical_presentation/chapters/06_hopf_classification.tex
git commit -m "docs(presentation): teach Hopf criticality"
```

---

### Task 8: Add the direct codimension-two solver chapter

**Files:**

- Modify: `notes/technical_presentation/chapters/07_codim2_solvers.tex`

**Interfaces:**

- Consumes: `cusp_point`, `bogdanov_takens_point`, `generalized_hopf_point`, `zero_hopf_point`, `double_hopf_point`, their parameter-only companions, and `fold_coefficient`.
- Produces: a correct local-refinement taxonomy with explicit two-parameter-curve limitations.

- [ ] **Step 1: Add the chapter divider and release badge**

```tex
\chapterdivider{7}{Direct codimension-two point solvers}
  {How can two local degeneracy conditions organize a two-parameter model?}
  {Featured application: refine an approximate Bogdanov--Takens point}
```

- [ ] **Step 2: Draw the taxonomy**

Use a TikZ map with these exact relationships:

```text
Fold + vanishing quadratic fold coefficient → CP
Double zero eigenvalue / fold–Hopf organizing center → BT
Hopf + l1 = 0 → GH
Zero eigenvalue + imaginary pair → ZH
Two distinct imaginary pairs → HH
```

- [ ] **Step 3: Add a comparison table**

Columns: label/name, spectral or normal-form condition, minimum state dimension, extra returned vectors/frequencies, and special seed requirement. Include `n >= 3` for ZH, `n >= 4` and required keyword-only `seed_b` for HH.

- [ ] **Step 4: Show the direct-solver contract**

Use:

```text
approximate state + parameter pair + optional model args
                    ↓
          extended-system Newton solve
                    ↓
refined state + parameter pair + null/eigenvectors + converged flag
```

State on the same frame: “This is point refinement, not continuation of a fold/Hopf curve.”

- [ ] **Step 5: Add the BT API demo**

```python
u_bt, p_bt, v0, v1, converged = jc.bogdanov_takens_point(
    rhs,
    u_guess,
    jnp.array([p1_guess, p2_guess]),
    args,
)
assert converged
```

Explain `J v0 = 0` and `J v1 = v0` beside the code.

- [ ] **Step 6: Add a parameter-only differentiation frame**

Explain that `*_parameters(...)` returns the refined parameter pair without the convergence flag and is intended for grad-ready composition. Do not suggest ignoring convergence in exploratory use; point solvers should be checked first.

- [ ] **Step 7: Verify exports and compile**

```bash
python - <<'PY'
import jaxcont as jc
names = [
    "cusp_point", "cusp_parameters",
    "bogdanov_takens_point", "bogdanov_takens_parameters",
    "generalized_hopf_point", "generalized_hopf_parameters",
    "zero_hopf_point", "zero_hopf_parameters",
    "double_hopf_point", "double_hopf_parameters",
]
for name in names:
    assert hasattr(jc, name), name
print("codim-2 API names verified")
PY
make -C notes/technical_presentation
```

Expected: all ten exports exist and the chapter compiles.

- [ ] **Step 8: Commit**

```bash
git add notes/technical_presentation/chapters/07_codim2_solvers.tex
git commit -m "docs(presentation): add direct codim-two solvers"
```

---

### Task 9: Update the visualization chapter around real questions

**Files:**

- Modify: `notes/technical_presentation/chapters/08_visualization.tex`

**Interfaces:**

- Consumes: `plot_continuation`, `plot_all_states`, `plot_eigenvalues`, `plot_branch_states`, `plot_phase_plane`, and the tracked `examples/images/example_12_fitzhugh_nagumo_phase_plane.jpg`.
- Produces: a question-to-view guide reused by newcomer workflows.

- [ ] **Step 1: Add the chapter divider and release badge**

```tex
\chapterdivider{8}{Visualization: parameter space and state space}
  {Which plot answers the scientific question in front of me?}
  {Featured application: FitzHugh--Nagumo before and after a Hopf transition}
```

- [ ] **Step 2: Keep and expand the “three complementary views” table**

Use four rows:

| View | Answers | Does not establish |
|---|---|---|
| branch diagram | how solutions vary with parameter | basins or full flow geometry |
| eigenvalue/Floquet view | how local stability changes | nonlinear long-time behavior |
| branch-state projection | how stored branch states relate | a true phase portrait |
| 2D phase plane | nullclines, vector field, equilibria, trajectories at one parameter | higher-dimensional geometry or automatic branch discovery |

- [ ] **Step 3: Present visualization as a composable surface**

Show exact names grouped by purpose:

```text
Branches: plot_continuation, plot_all_states
Spectra: plot_eigenvalues
State projections: plot_branch_states
2D geometry: plot_phase_plane, plot_nullclines,
             plot_vector_field, plot_streamlines,
             plot_equilibria, plot_trajectory
Phase sensitivity: plot_prc
```

Mark `plot_prc` with `\mainbadge` and the other named v0.3 visualization features with `\releasedbadge`.

- [ ] **Step 4: Keep the FitzHugh–Nagumo code and real figure**

Retain the compact `plot_phase_plane(...)` example and reference:

```tex
\includegraphics[width=.96\textwidth]{example_12_fitzhugh_nagumo_phase_plane.jpg}
```

Add three callouts on interpretation: branch location, equilibrium stability marker, and trajectory relative to nullclines.

- [ ] **Step 5: Compile and visually inspect the real figure**

Render the deck to a temporary directory and inspect the Chapter 8 divider, API, and FitzHugh–Nagumo pages:

```bash
make -C notes/technical_presentation
review_dir=$(mktemp -d /tmp/jaxcont-viz-review.XXXXXX)
pdftoppm -png -r 120 \
  notes/technical_presentation/jaxcont_technical_presentation.pdf \
  "$review_dir/page"
```

Expected: axis text remains legible and the image is not stretched.

- [ ] **Step 6: Commit**

```bash
git add notes/technical_presentation/chapters/08_visualization.tex
git commit -m "docs(presentation): connect visualizations to questions"
```

---

### Task 10: Add the iPRC and dPRC chapter with tracked output

**Files:**

- Modify: `notes/technical_presentation/chapters/09_prc_dprc.tex`
- Modify: `examples/example_13_phase_response_curve.py:53-65`
- Create: `notes/technical_presentation/assets/example_13_phase_response_curve.png`
- Modify: `notes/technical_presentation/assets/README.md`

**Interfaces:**

- Consumes: `prc_curve(raw_f, mesh, U, p)`, `branch_prc(raw_f, mesh, states, params)`, `dprc_curve(problem)`, and `plot_prc(curve, ...)`.
- Produces: the phase-sensitivity concepts required by the PRC validation material.

- [ ] **Step 1: Make Example 13 save its reviewed figure**

Change the dPRC title and add the save immediately before `plt.show()`:

```python
plot_prc(dZ, ax=ax_dprc, labels=["x", "y"], title="dPRC (d/dρ)")
plt.tight_layout()
plt.savefig("images/example_13_phase_response_curve.png", dpi=180, bbox_inches="tight")
plt.show()
```

Then run from `examples/`:

```bash
MPLBACKEND=Agg python example_13_phase_response_curve.py
```

Expected: the script exits zero and writes a two-panel figure with `iPRC` on the left and `dPRC (d/dρ)` on the right.

- [ ] **Step 2: Commit the reviewed snapshot**

Copy `examples/images/example_13_phase_response_curve.png` byte-for-byte to `notes/technical_presentation/assets/example_13_phase_response_curve.png` and record the generation command and commit hash in `assets/README.md`.

- [ ] **Step 3: Add the chapter divider and current-main badge**

```tex
\chapterdivider{9}{Phase-response curves and parameter sensitivity}
  {When an oscillator is perturbed, how much does its phase shift, and how does that sensitivity change with a parameter?}
  {Featured application: the circle oscillator's iPRC and dPRC}
```

Place `\mainbadge` on the first content frame.

- [ ] **Step 4: Build the visual-first sequence**

Create these frames:

1. “Why phase sensitivity matters” — same impulse at three phases, three phase shifts;
2. “The iPRC is a gradient of asymptotic phase”;
3. “Adjoint propagation around one period” — backward arrows and periodic closure;
4. “Normalization fixes the scale” — `Z(0)·f(x_0,p)=2π/T`;
5. “One orbit, a branch, or a parameter derivative?” — compare three APIs;
6. “Minimal iPRC/dPRC API”;
7. “Plot the curves”;
8. real Example 13 figure;
9. “How to read sign, component, phase, and magnitude”;
10. “What these curves do not establish”;
11. “What to remember”.

- [ ] **Step 5: Use exact API calls**

```python
from jaxcont.stability.prc import branch_prc, dprc_curve, prc_curve
from jaxcont.viz import plot_prc

Z = prc_curve(rhs, mesh, problem.u0, problem.p0)
dZ_dp = dprc_curve(problem)
plot_prc(Z, labels=["x", "y"], title="iPRC")
```

Add a separate one-line `branch_prc` example using branch states/parameters, and state that compatible fixed shapes are required for batching.

- [ ] **Step 6: State dPRC semantics exactly**

Use the sentence: “JaxCont’s `dprc_curve` is `d(PRC)/dp` after reconverging the periodic orbit; it is not the time derivative `d(PRC)/dt` exported under the dPRC name by MatCont.”

- [ ] **Step 7: Verify names and compile**

```bash
python - <<'PY'
from jaxcont.stability.prc import branch_prc, dprc_curve, prc_curve
from jaxcont.viz import plot_prc
print("PRC API names verified")
PY
test -s notes/technical_presentation/assets/example_13_phase_response_curve.png
make -C notes/technical_presentation
```

Expected: imports succeed, asset is nonempty, and the chapter compiles.

- [ ] **Step 8: Commit**

```bash
git add examples/example_13_phase_response_curve.py notes/technical_presentation/chapters/09_prc_dprc.tex notes/technical_presentation/assets
git commit -m "docs(presentation): add PRC and dPRC chapter"
```

---

### Task 11: Add the validation chapter and independent shooting figure

**Files:**

- Modify: `notes/technical_presentation/chapters/10_validation.tex`
- Modify: `examples/example_14_prc_shooting_validation.py:281-316`
- Create: `notes/technical_presentation/assets/example_14_prc_shooting_validation.png`
- Modify: `notes/technical_presentation/assets/README.md`

**Interfaces:**

- Consumes: reviewed validation registry/results, BifurcationKit references, analytic examples, and Example 14.
- Produces: the evidence vocabulary and warnings used in final workflow slides.

- [ ] **Step 1: Make Example 14 save its reviewed figure**

Add the save immediately before the existing `plt.show()`:

```python
plt.tight_layout()
plt.savefig(
    "images/example_14_prc_shooting_validation.png",
    dpi=180,
    bbox_inches="tight",
)
plt.show()
```

Then run the independent shooting validation.

From `examples/` run:

```bash
MPLBACKEND=Agg python example_14_prc_shooting_validation.py
```

Expected: the script exits zero and prints finite maximum errors for the sheared-circle and Van der Pol PRC/dPRC comparisons.

- [ ] **Step 2: Track the four-panel figure**

Copy `examples/images/example_14_prc_shooting_validation.png` byte-for-byte to `assets/example_14_prc_shooting_validation.png` and record the exact command/commit in `assets/README.md`.

- [ ] **Step 3: Add the chapter divider**

```tex
\chapterdivider{10}{Validation: from residuals to independent evidence}
  {What evidence makes a continuation or sensitivity result trustworthy?}
  {Featured application: collocation PRC/dPRC checked by analytic formulas and independent shooting}
```

- [ ] **Step 4: Draw the validation ladder**

Use five ascending levels:

1. convergence/residual/termination checks;
2. repeat with changed mesh, tolerances, step size, and direction;
3. analytic oracle;
4. independent algorithm;
5. independent package/reference artifacts with aligned conventions.

State that higher levels supplement rather than replace lower levels.

- [ ] **Step 5: Add an evidence matrix**

Rows:

| Capability | Evidence shown | Important qualification |
|---|---|---|
| folds/branch topology | analytic + MatCont/BifurcationKit cases | align traversal/order before pointwise comparison |
| periodic orbits/Floquet | analytic + MatCont cases | remove the trivial multiplier consistently |
| Hopf `l1` | BifurcationKit cross-check + gradient tests | local normal-form result |
| selected codim-2 points | analytic systems + BifurcationKit BT | point refinement, not curve continuation |
| iPRC | analytic + MatCont + shooting | align phase origin and units |
| dPRC | analytic/reconverged finite difference + shooting | MatCont's “dPRC” is a different derivative |

- [ ] **Step 6: Add the real shooting comparison**

Use the tracked four-panel figure across one full frame or two readable crop frames. Call out agreement markers without inventing rounded error values; if numerical values are printed, copy them from the fresh execution log.

- [ ] **Step 7: Add a conventions checklist**

Include: state ordering, parameter units, phase origin, phase units, eigenvalue/multiplier matching, trivial multiplier removal, branch orientation, and interpolation grid.

- [ ] **Step 8: Compile and commit**

```bash
test -s notes/technical_presentation/assets/example_14_prc_shooting_validation.png
make -C notes/technical_presentation
git add examples/example_14_prc_shooting_validation.py notes/technical_presentation/chapters/10_validation.tex notes/technical_presentation/assets
git commit -m "docs(presentation): add evidence-based validation chapter"
```

---

### Task 12: Build guided workflows, boundaries, and appendices

**Files:**

- Modify: `notes/technical_presentation/chapters/11_guided_workflows.tex`
- Modify: `notes/technical_presentation/chapters/12_scope_and_appendix.tex`

**Interfaces:**

- Consumes: concepts, APIs, status labels, examples, and evidence from Chapters 1–10.
- Produces: newcomer execution paths, final capability boundaries, glossary, and maintainer appendix.

- [ ] **Step 1: Add both chapter dividers**

```tex
\chapterdivider{11}{Guided application workflows}
  {How should a newcomer plan, run, inspect, and validate a JaxCont study?}
  {Featured applications: six reproducible paths through the example gallery}

\chapterdivider{12}{Scope, glossary, and technical appendix}
  {What should you remember, and where are the implementation boundaries?}
  {Reference material: terminology, derivations, engine details, and sources}
```

- [ ] **Step 2: Turn the research workflow into a six-stage diagram**

Use:

```text
Question → model/residual → trustworthy seed → small continuation
         → inspect/visualize → repeat and validate
```

Under “inspect”, list residuals, `n_valid`/mask, termination, step size, stability, and event diagnostics.

- [ ] **Step 3: Add six demo cards**

Each card must show objective, file, command, expected artifact/result, and one caution:

1. README saddle-node quick start;
2. `examples/example_03_van_der_pol.py` — Hopf;
3. `examples/example_10_van_der_pol_limit_cycle.py` — periodic orbit;
4. `examples/example_12_fitzhugh_nagumo_phase_plane.py` — phase plane;
5. `examples/example_13_phase_response_curve.py` — iPRC/dPRC;
6. `examples/example_14_prc_shooting_validation.py` — independent validation.

Use two or three cards per frame so commands remain readable.

- [ ] **Step 4: Add a “How to read a result” decision tree**

Branches:

```text
Did continuation stop early?
├── yes → inspect termination, residuals, finite values, ds_min, attempt budget
└── no  → inspect stability/event evidence
          ├── important conclusion → rerun with changed settings/direction
          └── publication claim → add analytic or independent validation
```

- [ ] **Step 5: Replace stale capability boundaries**

Use a three-column status matrix matching Chapter 1. Explicitly list capabilities available in v0.3.1, current-main PRC/dPRC capabilities, and future unsupported workflows.

- [ ] **Step 6: Move deep implementation detail to the appendix**

Move or preserve:

- natural versus pseudo-arclength comparison;
- cubic fold derivation;
- simplified engine pseudocode;
- fixed carry fields and failed-attempt semantics;
- eager versus traced result reassembly;
- “scan” naming clarification;
- source/evidence list.

- [ ] **Step 7: Update the glossary and takeaways**

Glossary must define: residual, branch, fold, Hopf, pseudo-arclength, collocation, phase condition, Floquet multiplier, PD, NS, `l1`, codimension, CP/BT/GH/ZH/HH, iPRC, dPRC, validity mask, and implicit differentiation.

Final takeaways must fit on one frame and end with: “A computed diagram is evidence about the supplied mathematical model, not automatic validation of the model or a causal claim.”

- [ ] **Step 8: Compile and commit**

```bash
make -C notes/technical_presentation
git add notes/technical_presentation/chapters/11_guided_workflows.tex notes/technical_presentation/chapters/12_scope_and_appendix.tex
git commit -m "docs(presentation): add guided workflows and scope"
```

---

### Task 13: Synchronize README, method, and speaker notes

**Files:**

- Modify: `notes/technical_presentation/README.md:1-44`
- Modify: `notes/technical_presentation/METHOD.md:1-74`
- Modify: `notes/technical_presentation/SPEAKER_NOTES.md:1-165`

**Interfaces:**

- Consumes: final chapter titles, demo commands, asset provenance, and verified release status.
- Produces: authoring instructions and presenter guidance that match the finished deck.

- [ ] **Step 1: Rewrite the presentation README**

Document:

- “JaxCont v0.3.1+” meaning;
- master/setup/chapter layout;
- tracked versus generated assets;
- `make`, `make rebuild`, `make verify`, `make clean`, and `make distclean`;
- final PDF filename;
- chapter-by-chapter review workflow;
- requirement to refresh figures from their source examples.

- [ ] **Step 2: Update the maintenance method**

Record the evidence hierarchy:

```text
current public API and implementation
→ tests and runnable examples
→ validation artifacts and independent references
→ current docs/changelog/roadmap
→ approved specs/plans for rationale only
```

Add the scientific-question → visual → method → API → output → interpretation → limitation chapter pattern and the v0.3.1+ status rule.

- [ ] **Step 3: Reorganize speaker notes by chapter**

Provide:

- overview route;
- methods route;
- complete route;
- one section for each of the twelve chapters;
- live-demo reliability guidance;
- interpretation cautions;
- likely questions for Hopf `l1`, codim-2 point solvers, PRC/dPRC, and validation conventions.

- [ ] **Step 4: Check documentation consistency**

Run:

```bash
rg -n 'v0\.2|current-main visualization|72 slides|55–70|55-70' \
  notes/technical_presentation
rg -n '01_orientation|12_scope_and_appendix|make verify|v0\.3\.1\+' \
  notes/technical_presentation/README.md \
  notes/technical_presentation/METHOD.md \
  notes/technical_presentation/SPEAKER_NOTES.md
```

Expected: stale duration/version claims are removed or explicitly historical; all new structural terms appear.

- [ ] **Step 5: Commit**

```bash
git add notes/technical_presentation/README.md notes/technical_presentation/METHOD.md notes/technical_presentation/SPEAKER_NOTES.md
git commit -m "docs(presentation): synchronize author and speaker guidance"
```

---

### Task 14: Add automated deck verification and perform final visual QA

**Files:**

- Modify: `notes/technical_presentation/Makefile`
- Modify: presentation chapters/setup only when verification reveals a concrete defect
- Update: `notes/technical_presentation/jaxcont_technical_presentation.pdf` build artifact in the workspace

**Interfaces:**

- Consumes: the complete modular source and committed assets.
- Produces: a clean final PDF plus repeatable build/log/text/visual checks.

- [ ] **Step 1: Add the Make verify target**

Add:

```make
.PHONY: verify

verify: rebuild
	@test "$$(pdfinfo $(PDF) | awk '/^Pages:/ {print $$2}')" -gt 0
	@! rg -n "LaTeX Error|Undefined control sequence|File .* not found|There were undefined references" $(JOB).log
	@pdftotext $(PDF) - | rg -q "JaxCont v0.3.1"
	@pdftotext $(PDF) - | rg -q "Current main"
	@pdftotext $(PDF) - | rg -q "Phase-response"
```

Keep overfull-box review visible but manual: text or TikZ can intentionally exceed a TeX box by tiny rounding amounts, so the automated target must not hide the warnings or treat every sub-point warning as fatal.

- [ ] **Step 2: Run targeted source/API checks**

```bash
python -m pytest \
  tests/test_hopf_normal_form.py \
  tests/test_codim2.py \
  tests/test_prc.py \
  tests/test_viz.py
```

Expected: all selected tests pass. Record the exact test count.

- [ ] **Step 3: Run the final clean build**

```bash
make -C notes/technical_presentation verify
pdfinfo notes/technical_presentation/jaxcont_technical_presentation.pdf
rg -n 'Overfull|Underfull|LaTeX Warning' \
  notes/technical_presentation/jaxcont_technical_presentation.log
```

Expected: verify succeeds, the PDF is nonempty and 16:9, there are no undefined references/missing files, and every overfull frame warning is either fixed or proven visually harmless.

- [ ] **Step 4: Validate chapter presence and status wording**

```bash
pdftotext notes/technical_presentation/jaxcont_technical_presentation.pdf /tmp/jaxcont-deck.txt
for phrase in \
  "Orientation and application map" \
  "Continuation fundamentals" \
  "Natural and pseudo-arclength continuation" \
  "Public API and JAX execution model" \
  "Periodic orbits and Floquet stability" \
  "Hopf refinement and criticality" \
  "Direct codimension-two point solvers" \
  "Visualization: parameter space and state space" \
  "Phase-response curves and parameter sensitivity" \
  "Validation: from residuals to independent evidence" \
  "Guided application workflows" \
  "Scope, glossary, and technical appendix"; do
  rg -F "$phrase" /tmp/jaxcont-deck.txt
done
```

Expected: all twelve phrases are found.

- [ ] **Step 5: Render a contact sheet for every chapter**

Use `pdftoppm` to render at least the divider, one equation/code frame, and one real-figure frame from every chapter. Build contact sheets with ImageMagick and inspect:

- no cropped titles, code, equations, legends, or footer;
- readable labels at normal presentation zoom;
- consistent margins and color use;
- no low-resolution or stretched figures;
- no slide dominated by prose;
- correct badge on released/current-main/future material.

- [ ] **Step 6: Perform a claim audit**

Compare extracted slide text with:

```bash
python - <<'PY'
import jaxcont
print(jaxcont.__version__)
PY
git log -1 --oneline
```

Then recheck `CHANGELOG.md`, `examples/MatCont/README.md`, and the PRC shooting example. Fix any stale version, derivative, comparison, or unsupported-feature claim.

- [ ] **Step 7: Run final repository hygiene checks**

```bash
git diff --check
git status --short
git ls-files notes/technical_presentation/assets
```

Expected: no whitespace errors; only intentional presentation changes remain; all four required PNG assets and `assets/README.md` are tracked.

- [ ] **Step 8: Commit**

```bash
git add notes/technical_presentation/Makefile \
  notes/technical_presentation/presentation_setup.tex \
  notes/technical_presentation/chapters \
  notes/technical_presentation/assets \
  notes/technical_presentation/README.md \
  notes/technical_presentation/METHOD.md \
  notes/technical_presentation/SPEAKER_NOTES.md
git commit -m "docs(presentation): verify the v0.3.1+ technical deck"
```

Do not force-add LaTeX intermediates. The PDF remains a generated delivery artifact unless repository policy is intentionally changed.

---

## Final self-review checklist

- [ ] Every requirement in `docs/superpowers/specs/2026-08-06-technical-presentation-update-design.md` maps to a task above.
- [ ] No slide calls v0.4 released.
- [ ] PRC/dPRC carries the current-main badge.
- [ ] Hopf, codim-2, phase-plane, periodic-orbit, and validation-suite content carries the released badge.
- [ ] All public API snippets import and use current signatures.
- [ ] The HH slide mentions `seed_b`, `n >= 4`, and the frequency-separation guard.
- [ ] The dPRC slide distinguishes parameter and time derivatives.
- [ ] Every real application figure has provenance and a reproducible refresh path.
- [ ] All twelve chapter files compile into the one required PDF.
- [ ] Speaker notes, method, README, and final PDF agree on naming and scope.
