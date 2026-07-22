# Method used to prepare the presentation

## Source hierarchy

The deck was constructed from the implementation rather than from a generic
description of continuation theory. Sources were used in this order:

1. The approved design specification:
   `docs/superpowers/specs/2026-07-21-engine-consolidation-design.md`.
2. The detailed execution plan:
   `docs/superpowers/plans/2026-07-21-engine-consolidation.md`.
3. The merged implementation, particularly:
   `src/jaxcont/core/scan_continuation.py` and `src/jaxcont/api.py`.
4. Tests and examples that state intended behavior.
5. `notes/ARCHITECTURE.md`, `notes/ROADMAP.md`, and
   `notes/BEGINNERS_GUIDE_TO_BIFURCATION_ANALYSIS.md` for motivation and
   historical performance evidence.
6. Git history and completion records to distinguish implemented work from
   future roadmap work.

The original `worktree-v0.2-engine-consolidation` work was merged into `main`.
This revision was inspected through commit `b105417`. Engine consolidation is
complete; the slides still avoid claiming that every planned v0.2 feature is
complete, because periodic-orbit and related roadmap work is separate.

## Teaching strategy

The presentation follows a layered explanation:

1. Start from the familiar problem “solve `F(u,p)=0`.”
2. Explain a branch geometrically before introducing algorithms.
3. Show natural continuation and let its failure at a fold motivate the next
   method.
4. Introduce pseudo-arclength through geometry, then equations, then the
   bordered linear system.
5. Separate the *numerical method* from the *JAX implementation strategy*.
   This is where the ambiguous word “scan” is resolved.
6. Explain performance from the dispatch/compilation model, not from slogans.
7. Finish with limitations, testing, and an accurate two-minute summary.

All branch diagrams are TikZ drawings of the scalar cubic
`F(u,p)=p+u-u^3/3`; the same example has analytic folds at `p=+/-2/3`. Using
one example repeatedly reduces the amount of new notation a beginner must
hold in memory.

## Terminology decision: “scan”

The source function is named `pseudo_arclength_scan`, but its outer iteration
uses `jax.lax.while_loop`. The presentation uses “scan engine” as the project's
name for a whole-sweep, fixed-buffer functional engine. It explicitly does not
claim that the implementation calls `jax.lax.scan`.

This distinction is central because three different ideas are easy to mix up:

- pseudo-arclength is the numerical continuation method;
- predictor-corrector is the repeated algorithmic pattern;
- `lax.while_loop` plus JIT/fixed buffers is the JAX execution strategy.

## Performance claims

The deck repeats two measurements recorded in the repository roadmap:

- approximately `0.74 ms` warm scan versus `250 ms` for the prior Python loop
  on a pitchfork experiment (about `340x`);
- a recorded `256`-diagram `vmap` experiment at about `163x` versus a Python
  loop.

They are labeled as experiment-specific. The slides emphasize the transferable
architectural claim—whole-loop compilation reduces dispatch and enables
batching—rather than promising those speedups for every model.

## Verification performed

- Final consolidation verification recorded 75 default tests and 12
  slow-marked tests passing with zero failures.
- All six affected gallery examples ran headlessly; the BifurcationKit.jl
  cross-validation results in examples 02 and 05 remained intact.
- The documentation repair commit records a clean Sphinx `-W` build with zero
  warnings.
- The LaTeX source is compiled with `latexmk -pdf`.
- The PDF page count and build log are checked after compilation.
- Only presentation files under `notes/engine_consolidation_slides/` are
  modified; no JaxCont source code is changed.
