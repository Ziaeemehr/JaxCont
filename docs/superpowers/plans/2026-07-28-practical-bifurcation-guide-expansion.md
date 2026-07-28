# Practical Bifurcation Guide Expansion Plan

**Goal:** Turn `notes/doc_plan/jaxcont_practical_bifurcation_guide.tex` from a
short equilibrium-oriented handbook into a beginner-friendly but technically
detailed, step-by-step v0.2 tutorial. Every supported workflow should include
an executable package example, interpretation guidance, diagnostics, and
failure modes. Unsupported future work must remain visible as clearly labeled
placeholders rather than being presented as current API.

## Editorial rules

1. Explain the scientific object before the software call.
2. Pair every code block with expected output or an explicit inspection task.
3. Distinguish equilibrium eigenvalues from periodic-orbit Floquet multipliers.
4. Treat event detection as bracket/refine evidence, not automatic scientific
   classification.
5. Show common incorrect workflows and explain why they fail.
6. Keep package-version boundaries explicit.
7. Prefer repository examples and analytic checks over invented demonstrations.
8. End each practical chapter with a checklist or exercise.

## Chapter-by-chapter expansion

### Front matter and navigation

- Update the edition statement from equilibrium-only to v0.2.
- Add a capability table: supported now, partially supported, future.
- Provide short, full, equilibrium-only, and periodic-orbit reading routes.
- Replace overlapping left/right running headers with one bounded chapter mark.

### Part I — Foundations

- Expand the distinction between simulation, root solving, continuation, and
  event refinement.
- Add a solution-object map: equilibria, periodic orbits, and future global
  objects.
- Add worked stability calculations for scalar and two-dimensional systems.
- Separate equilibrium bifurcations from multiplier-based periodic events.
- Use Example 04 to demonstrate why a parameter sweep and natural continuation
  cannot pass a fold, while pseudo-arclength can.

### Part II — Practical equilibrium workflow

- Add a model-encoding checklist and examples of safe/unsafe JAX residuals.
- Show analytic starts, Newton-refined starts, multiple-root searches, residual
  checks, scaling, and start-parameter consistency.
- Build a complete minimal equilibrium run from a verified analytic root.
- Expand result inspection, settings, termination diagnosis, event semantics,
  and visualization.
- Add systematic residual, resolution, direction, simulation, and independent
  software validation.
- Expand JAX batching and timing with Example 06.
- Add differentiable fold sensitivity and inverse design from Example 07.

### Part III — Case studies

- Scalar fold: Natural versus PseudoArclength from Example 04.
- Van der Pol equilibrium: analytic eigenvalues and the degenerate crossing.
- Lorenz-84: multi-state continuation, annotated visualization, and
  BifurcationKit.jl cross-validation.
- Neural mass: Newton refinement, stiffness/tolerance choice, all-state plots,
  event comparison, and diagnosing an early stop.

### Part IV — Periodic orbits in v0.2

- Explain fixed-mesh orthogonal collocation, phase conditions, packed orbit
  states, periods, monodromy matrices, and Floquet multipliers.
- Give a complete cycle-initialization workflow using simulation, peak finding,
  time rebasing, `Collocation`, and `periodic_orbit_problem`.
- Use Example 10 (Van der Pol) and Example 11 (Brusselator) to show different
  observables: period/waveform versus amplitude growth.
- Use Examples 08–09 as analytic tests for PeriodDoubling and NeimarkSacker.
- Document mesh, float32 tolerance, event-kind, and construction-transformation
  pitfalls.

### Part V — Future-version placeholders

- Branch switching and automatic Hopf-to-cycle initialization.
- Adaptive collocation mesh redistribution.
- Two-parameter continuation and codimension-two classification.
- Homoclinic, heteroclinic, SNIC, and invariant-manifold workflows.
- Iterative sparse/Krylov solver implementations.
- Transformation-safe event orchestration.

Each placeholder must state what exists now, what is missing, and what evidence
should be required before converting it into a user tutorial.

### Appendices

- Expand Newton and pseudo-arclength pseudocode.
- Add periodic-state packing formulas and a troubleshooting matrix.
- Expand the glossary and trust checklist for both equilibrium and periodic
  studies.

## Verification

- Compile with `make rebuild`.
- Remove overfull boxes and material underfull table warnings.
- Render contact sheets and inspect title, table of contents, chapter openings,
  running headers, code-heavy pages, figures, and final appendices.
- Check referenced public names against JaxCont v0.2 imports.
- Confirm all repository image paths resolve.
- Run `git diff --check`.
