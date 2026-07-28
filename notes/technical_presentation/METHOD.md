# Method used to maintain the technical presentation

## Source hierarchy

The deck is maintained from the current implementation and user-facing
documentation rather than from release-history notes. Sources are used in this
order:

1. The current implementation, particularly
   `src/jaxcont/core/scan_continuation.py` and `src/jaxcont/api.py`.
2. Tests and examples that state intended behavior.
3. `notes/ARCHITECTURE.md`, `notes/ROADMAP.md`, and
   `notes/BEGINNERS_GUIDE_TO_BIFURCATION_ANALYSIS.md` for motivation and
   research workflow.
4. Design plans and Git history only when they are needed to explain why a
   current behavior or limitation exists.

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
6. Connect implementation details to concrete research workloads.
7. Finish with a reproducible workflow, validation, interpretation, and scope.

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

The deck explains the reusable architectural ideas—whole-loop compilation,
fixed shapes, and batching—without retaining historical old-versus-new timing
comparisons. Any future benchmark must report cold and warm timing, hardware,
precision, model size, branch settings, and batch size.

## Verification performed

- The LaTeX source is compiled with `latexmk -pdf`.
- The PDF page count and build log are checked after compilation.
- Claims about the public API and numerical behavior are checked against the
  current implementation, tests, and examples.
- Research advice is kept distinct from repository software tests.
