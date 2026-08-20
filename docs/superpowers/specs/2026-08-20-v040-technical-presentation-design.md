# JaxCont v0.4.0 Technical Presentation Expansion

## Purpose

Update the technical Beamer presentation so it teaches every material public
capability and release-integrity change added after the deck's last substantive
update on 6 August 2026. The deck has no slide-count limit; conceptual clarity,
accurate scope, and reproducible evidence take priority over preserving its
current length.

## Audience and teaching approach

The audience includes researchers who know dynamical systems but may be new to
continuation, and JAX users who may be new to numerical bifurcation analysis.
Each capability must therefore progress through scientific motivation,
mathematical construction, public API, interpretation, limitations, and
evidence. Existing visual conventions remain: green denotes released v0.4.0
capabilities and red denotes unimplemented or experimental boundaries.

## Structural design

Insert a dedicated chapter, **Two-parameter fold and Hopf curves**, immediately
after **Direct codimension-two point solvers**. Renumber the later chapters so
the sequence remains conceptually ordered:

1. Orientation and application map
2. Continuation fundamentals
3. Natural and pseudo-arclength continuation
4. Public API and JAX execution model
5. Periodic orbits and Floquet stability
6. Hopf refinement and criticality
7. Direct codimension-two point solvers
8. Two-parameter fold and Hopf curves
9. Visualization: parameter space and state space
10. Phase-response curves and parameter sensitivity
11. Validation: from residuals to independent evidence
12. Guided application workflows
13. Scope, release integrity, glossary, and technical appendix

Chapter source filenames will be renumbered to match the visible chapter
numbers. The master file, technical-presentation README, speaker notes, and all
page references will be updated after a clean PDF build establishes final
pagination.

## New two-parameter chapter

The chapter will explain:

- the distinction between refining one codimension-two point and tracing a
  codimension-one curve in a two-parameter plane;
- how a fold or Hopf extended system becomes an ordinary one-parameter
  continuation problem when one physical parameter is solved inside the
  packed state and the other is selected by `free`;
- the public factories `fold_curve_problem` and `hopf_curve_problem`, including
  the required `p_guess.shape == (2,)`, `free in {0, 1}`, eager seed refinement,
  and the `p_span[0] == p_guess[free]` contract;
- packed state layouts and how `plot_two_parameter_diagram` reconstructs the
  physical parameter plane;
- the curve-event protocol and where CP, BT, GH, ZH, and HH can be detected on
  fold and Hopf curves;
- the need to discard the curve's pinned zero eigenvalue or pinned imaginary
  pair before testing for an additional degeneracy;
- Example 15's shifted Bogdanov--Takens normal form, exact fold parabola,
  detected BT point, `jax.vmap` batch, and `jax.grad` sensitivity;
- traced-result validity masks, passing swept values through `args`, and the
  Hopf eigenvector-anchor restart limitation.

The chapter will end with an explicit boundary: JaxCont v0.4.0 traces
equilibrium fold and Hopf curves, but not periodic-orbit PD/LPC/NS curves,
branch switching, or connecting-orbit families.

## Updates to existing chapters

### Orientation and public API

Add two-parameter questions and routes to the application map. Expand the
public-surface tree with curve factories, curve events, two-parameter plotting,
PRC helpers, and persistence. Explain the v0.4.0 rule that every continuation
seed is corrected or rejected and every problem kind must start at `p0`.

### Direct codimension-two solvers

Replace claims that JaxCont cannot continue fold/Hopf curves. Preserve the
distinction between direct local point refinement and curve continuation, then
hand off explicitly to the new chapter.

### Periodic orbits

Explain that PD/NS refinement now Newton-corrects an interpolated periodic
orbit before evaluating its Floquet multipliers. Retain the experimental
status of fold-of-cycles, PD, and NS detection until `MC-LC-002` closes.

### Visualization

Add `plot_two_parameter_diagram`, its accepted `(solution, curve_kind)` inputs,
physical-axis reconstruction, and codimension-two annotations. Keep parameter
planes distinct from branch-state projections and frozen-parameter phase
planes.

### Validation

Add the MatCont visual comparison gallery from Examples 16--20. Teach that
overlaid geometry is diagnostic evidence while the tolerance-based CLI is
authoritative. Show four successful comparison families and retain the torBPC
failure without weakening its tolerances. Update the supported-case count to
seven passing of eight, including `MC-PRC-001`.

### Workflows and appendix

Add a two-parameter workflow beginning from a checked fold/Hopf seed. Add a
release-integrity section covering:

- Python 3.11+, JAX/JAXlib 0.9+, and the supported Python 3.11--3.12 matrix;
- changed `fold_point` and `hopf_point` return tuples with convergence flags;
- corrected/validated continuation seeds and universal `p_span` matching;
- real `adaptive=False` semantics and the failed-step backoff boundary;
- real Newton iteration diagnostics and `verbose=True` summaries;
- the versioned, pickle-free `ContinuationSolution` NPZ schema and preserved
  optional fields/metadata;
- the deprecated `plot_phase_portrait` alias remaining until v0.5.0.

Update the capability boundary, main takeaways, glossary, evidence sources,
and installation slide accordingly.

## Visual assets

Prefer native Beamer/TikZ diagrams for algorithms, state layouts, event maps,
and API flows. Reuse reviewed raster assets already tracked by the presentation.
For Examples 16--20, use reproducible overlay images generated by the example
scripts and record their provenance and hashes in the presentation assets
README before relying on them as evidence. Do not present the torBPC overlay as
a passing validation result.

## Source-of-truth rules

- Public names and signatures come from `src/jaxcont` and exported APIs.
- Behavioral contracts come from tests and executable examples.
- Release claims come from `src/jaxcont/_version.py` and `CHANGELOG.md`.
- Validation counts and limitations come from the MatCont registry and a fresh
  CPU validation run.
- Historical plans and the August review may explain changes but cannot
  override the current implementation.

## Verification

Completion requires all of the following:

1. A clean `make verify` in `notes/technical_presentation`.
2. A non-empty PDF containing `JaxCont v0.4.0`, the new chapter title, and the
   released v0.4.0 badge.
3. No stale claims that fold/Hopf two-parameter continuation is unavailable or
   still planned for v0.4.
4. Updated page counts and chapter ranges in the presentation README and
   speaker notes based on `pdfinfo`, not estimates.
5. A warning-as-error Sphinx HTML build after the PDF is refreshed.
6. `git diff --check` with no whitespace errors.

## Explicit non-goals

- Do not claim branch switching, automatic cycle discovery, general BVP,
  homoclinic/heteroclinic continuation, or periodic-orbit codimension-two curve
  continuation.
- Do not hide or relabel the known `MC-LC-002` failure.
- Do not update the archived v0.3.1 citation DOI before a v0.4.0 Zenodo archive
  exists.
- Do not impose a target or maximum number of slides.
