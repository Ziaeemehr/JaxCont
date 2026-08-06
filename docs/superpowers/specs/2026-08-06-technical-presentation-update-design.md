# Technical Presentation Update — Design Specification

**Date:** 2026-08-06

**Status:** Approved for implementation planning

**Deliverable:** One beginner-friendly “JaxCont v0.3.1+” technical presentation PDF assembled from modular Beamer chapters

## Purpose

Update the JaxCont technical presentation so it accurately teaches the current application surface to newcomers. Preserve the established visual language, add the features merged since the presentation's 2026-07-28 update, and use diagrams, real example figures, and reproducible demos wherever they improve understanding.

The finished artifact remains one PDF. The editable source is split into chapters so authors and reviewers can work on one coherent topic without repeatedly reading a monolithic source file.

## Audience and learning goals

The primary audience is a technically literate newcomer who may know differential equations or JAX but does not yet know numerical continuation or JaxCont.

After the presentation, the audience should be able to:

1. distinguish continuation from time integration and isolated root solving;
2. explain why natural continuation stalls at a fold and how pseudo-arclength continuation passes it;
3. create equilibrium and periodic-orbit problems through the public API;
4. interpret equilibrium eigenvalues, Floquet multipliers, and the supported codimension-one events;
5. explain what Hopf refinement and the first Lyapunov coefficient add to a detected Hopf point;
6. recognize the purpose and limitations of the CP, BT, GH, ZH, and HH direct point solvers;
7. relate a bifurcation diagram to a two-dimensional phase-plane view;
8. explain the scientific meaning of an infinitesimal PRC and its parameter derivative;
9. run the curated examples, inspect their outputs, and choose an appropriate validation strategy;
10. state the package's current boundaries without implying unsupported branch switching or two-parameter continuation.

## Scope baseline

The presentation was last updated at commit `a969e31` on 2026-07-28. The update covers the supported release surface through JaxCont 0.3.1 and the PRC/dPRC work present on current `main` as of 2026-08-06.

The new or substantially revised material includes:

- Hopf extended-system refinement through `hopf_point` and `hopf_parameter`;
- Hopf frequency and criticality classification through `lyapunov_coefficient`;
- direct codimension-two point solvers for cusp (CP), Bogdanov–Takens (BT), generalized Hopf (GH), zero-Hopf (ZH), and double-Hopf (HH) points;
- two-dimensional phase-plane visualization as a released v0.3 capability;
- infinitesimal phase-response curves through `prc_curve` and branch batching through `branch_prc`;
- PRC parameter sensitivity through `dprc_curve`;
- PRC visualization through `plot_prc`;
- analytic, MatCont, BifurcationKit, and independent shooting validation workflows;
- the v0.3.1 detector-margin and documentation-build fixes where they affect user expectations.

The deck-wide version label is **“JaxCont v0.3.1+”**. The plus sign means “the published v0.3.1 release plus current-main features intended for v0.4”; it must not imply that v0.4 has already been released. A short legend near the beginning distinguishes three statuses:

- **Available in v0.3.1**;
- **Current main — planned for v0.4**;
- **Future work — not implemented**.

PRC/dPRC features must use the current-main/planned-for-v0.4 status unless the package version is advanced before the deck is finalized. Release claims must be checked against `src/jaxcont/_version.py` and `CHANGELOG.md` during implementation. After v0.4 is published, maintainers should be able to update the deck by changing the global version label and removing obsolete current-main markers without rewriting the teaching narrative.

## Source architecture

The presentation will use this layout:

```text
notes/technical_presentation/
├── jaxcont_technical_presentation.tex
├── presentation_setup.tex
├── chapters/
│   ├── 01_orientation.tex
│   ├── 02_continuation_fundamentals.tex
│   ├── 03_pseudo_arclength.tex
│   ├── 04_api_and_jax.tex
│   ├── 05_periodic_orbits.tex
│   ├── 06_hopf_classification.tex
│   ├── 07_codim2_solvers.tex
│   ├── 08_visualization.tex
│   ├── 09_prc_dprc.tex
│   ├── 10_validation.tex
│   ├── 11_guided_workflows.tex
│   └── 12_scope_and_appendix.tex
├── assets/
├── SPEAKER_NOTES.md
├── METHOD.md
├── README.md
└── Makefile
```

`jaxcont_technical_presentation.tex` owns the document metadata, chapter order, and final document boundary. `presentation_setup.tex` owns packages, Beamer configuration, colors, listings, TikZ libraries, and reusable macros. Chapter files own only their section declarations and frames. Deck-specific generated or copied figures live under `assets/`; reusable gallery figures remain under `examples/images/` and are referenced without duplication when practical.

The default `make` target continues to produce `jaxcont_technical_presentation.pdf`. The source split must not change the public output filename.

## Visual system

The established presentation style is retained:

- 16:9 Beamer layout;
- Madrid theme;
- JaxBlue, JaxTeal, JaxOrange, JaxGreen, and JaxRed palette;
- existing title treatment, footer, block styles, and Python listing style;
- TikZ diagrams with consistent arrow, node, and annotation conventions.

New chapter-divider slides use the same palette and show the chapter number, chapter question, prerequisite chapters when needed, and featured application. Long chapters end with a concise “What to remember” frame.

Each normal slide has one primary teaching message. Dense paragraphs are replaced with diagrams, short labels, progressive conceptual structure, or speaker notes. Equations appear after the relevant geometric or scientific motivation. Deep implementation details that are useful to maintainers but interrupt the newcomer narrative move to the appendix.

## Teaching structure

The final deck uses twelve chapters:

1. **Orientation and application map** — what JaxCont answers, current version boundaries, and the complete learning path.
2. **Continuation fundamentals** — roots, branches, stability, events, and the distinction from simulation.
3. **Natural and pseudo-arclength continuation** — fold failure, tangent prediction, bordered correction, and adaptive steps.
4. **Public API and JAX execution model** — problem construction, continuation results, fixed buffers, JIT, `vmap`, autodiff, and implicit roots.
5. **Periodic-orbit continuation and Floquet stability** — collocation, phase condition, packed orbit state, multiplier interpretation, PD, and NS events.
6. **Hopf refinement and criticality classification** — extended-system solve, oscillation frequency, first Lyapunov coefficient, and criticality.
7. **Direct codimension-two point solvers** — CP, BT, GH, ZH, and HH as local refinements from supplied guesses.
8. **Phase planes and visualization** — branch diagrams, eigenvalue views, state-space geometry, trajectories, and composed phase planes.
9. **Phase-response curves and parameter sensitivity** — iPRC meaning, adjoint construction, `prc_curve`, `branch_prc`, `dprc_curve`, and `plot_prc`.
10. **Validation** — analytic oracles, independent implementations, MatCont, BifurcationKit, shooting, tolerances, and honest partial comparisons.
11. **Guided application workflows** — reproducible examples, diagnostic checks, interpretation prompts, and live-demo routes.
12. **Scope, limitations, glossary, and appendix** — unsupported workflows, vocabulary, derivations, implementation details, and sources.

Feature chapters follow a consistent sequence:

1. scientific question;
2. representative visual result;
3. mathematical intuition;
4. minimum useful API example;
5. interpretation of output;
6. limitations and common misuse.

## New feature explanations

### Hopf refinement and criticality

The deck must distinguish detecting a candidate crossing from refining the Hopf point. It explains that the extended system solves for the equilibrium, parameter, critical eigenspace, and oscillation frequency. The first Lyapunov coefficient `l1` classifies the local Hopf normal form: negative for supercritical, positive for subcritical, and near zero for a degenerate/generalized Hopf candidate under the package's tolerance conventions.

The presentation must not imply that Hopf refinement automatically constructs or branch-switches onto a periodic orbit.

### Direct codimension-two solvers

A visual taxonomy shows which degeneracies combine at CP, BT, GH, ZH, and HH points. Each solver is presented as an extended-system point refinement that consumes an approximate state and two-parameter guess. The deck must explicitly distinguish direct point solving from two-parameter bifurcation-curve continuation.

At least one compact API example and one solver-result interpretation slide are included. The remaining point types may share a comparison table if that improves clarity.

### Phase-response curves

The PRC chapter introduces phase as position along a stable oscillation and the iPRC as first-order phase sensitivity to a perturbation. The adjoint construction is explained visually before the periodic boundary and normalization equations are shown.

`prc_curve` is presented as the iPRC for one collocated orbit, `branch_prc` as batched evaluation along compatible branch states, and `dprc_curve` as the derivative of the PRC with respect to the model parameter after reconverging the periodic orbit. The deck must explicitly distinguish JaxCont's parameter derivative from MatCont's exported time derivative convention.

## Figures and demos

Three visual categories are used:

1. **TikZ teaching figures** for branch geometry, predictor–corrector steps, eigenvalue crossings, codimension-two relationships, adjoint propagation, and validation flow.
2. **Repository-generated application figures** for phase planes, periodic events, PRC/dPRC curves, and shooting comparisons.
3. **Compact workflow diagrams** for model-to-result and validation data flow.

The primary guided demo path is:

1. saddle-node equilibrium continuation from the README quick start;
2. Van der Pol Hopf detection/refinement and limit-cycle continuation;
3. FitzHugh–Nagumo phase-plane composition;
4. Hopf criticality plus a direct codimension-two point refinement;
5. circle-system PRC/dPRC;
6. independent PRC shooting validation.

Each demo slide or short sequence states the scientific objective, exact repository example or command, essential API lines, expected output, interpretation prompts, and a visible warning when a common misuse is likely. Speaker notes identify which demos are reliable for live execution and which should use pre-generated output.

## Validation model

The deck presents validation as a ladder rather than a single pass/fail claim:

1. inspect convergence, residuals, validity masks, termination, and step sizes;
2. repeat with changed numerical settings or continuation direction;
3. compare with an analytic reference when one exists;
4. compare with an independent algorithm or package;
5. verify that conventions, phase origins, units, and branch topology were aligned before comparing numbers.

Repository software tests are described as evidence for implementation behavior, not proof that a research model or interpretation is correct.

Validation claims must match the available evidence:

- equilibrium, periodic, and PRC comparisons may use MatCont where the repository contains reviewed reference artifacts;
- Hopf normal-form and selected codimension-two comparisons may use BifurcationKit;
- dPRC uses analytic or independently reconstructed shooting/finite-difference evidence rather than a misleading MatCont dPRC comparison;
- partial, unsupported, or convention-sensitive comparisons are labeled honestly.

## Navigation and presentation routes

The deck will provide three suggested routes in the opening material and speaker notes:

- **Overview route:** application map, core continuation intuition, representative outputs, workflow, and limitations.
- **Methods route:** continuation mathematics, periodic collocation, stability, Hopf refinement, and codimension-two systems.
- **Complete route:** all chapters, demos, validation, and selected appendix material.

Section dividers and chapter recaps make it possible to skip a chapter without losing the global narrative. Cross-references should point to concepts by descriptive name, not only by slide number.

## Documentation updates

`README.md` will describe the modular source layout, final build artifact, figure provenance, and build/verification commands. `METHOD.md` will record the updated source hierarchy, teaching strategy, terminology decisions, feature-status rules, and validation standards. `SPEAKER_NOTES.md` will be reorganized by chapter and include short/complete routes, demo guidance, interpretation cautions, and likely newcomer questions.

## Verification requirements

Before delivery:

1. run every repository example used to generate or substantiate a demo figure;
2. confirm referenced figure artifacts exist and have current timestamps/content;
3. compile the complete deck from a clean LaTeX build;
4. confirm the expected PDF exists and has a plausible nonzero page count;
5. scan the LaTeX log for errors, undefined references, missing files, overfull frames, and clipped content;
6. render representative pages from every chapter and inspect them visually;
7. check all displayed API names and code snippets against current source;
8. review every frame for excessive text, unexplained terminology, stale release language, and unsupported claims;
9. confirm `README.md`, `METHOD.md`, and `SPEAKER_NOTES.md` match the final deck structure;
10. preserve the final filename `jaxcont_technical_presentation.pdf`.

## Out of scope

This update does not add new JaxCont numerical capabilities, redesign the package API, create an unrelated visual identity, or promise unsupported workflows. It does not implement branch switching, two-parameter curve continuation, automatic cycle discovery from a Hopf point, general BVP continuation, or connecting-orbit continuation.

The presentation may explain those boundaries but must not present them as available features.

## Acceptance criteria

The work is complete when:

- the final presentation builds as one PDF from the modular source;
- the existing style is recognizably preserved;
- the title and status legend use “JaxCont v0.3.1+” without presenting v0.4 as released;
- all post-2026-07-28 supported features in scope are accurately represented;
- every major application feature has a visual explanation or real output figure;
- the guided demos are reproducible from repository commands;
- newcomers can follow a clear path from scientific question to API, result, interpretation, and validation;
- speaker notes and maintenance documentation match the new chapter structure;
- the build and visual checks pass without unresolved errors or materially clipped frames;
- a detailed implementation plan exists under `docs/superpowers/plans/` before presentation files are edited.
