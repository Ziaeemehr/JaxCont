# Changelog

All notable changes to JaxCont will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-05

Adds normal-form classification for Hopf points and five direct codim-2 bifurcation solvers on
top of v0.2.0's periodic-orbit continuation, plus 2D phase-plane visualization.

### Added
- 2D phase-plane visualization in `jaxcont.viz`: `plot_phase_plane`, `plot_nullclines`,
  `plot_vector_field`, `plot_streamlines`, `plot_equilibria`, and `plot_trajectory`.
  Supports 2D autonomous systems only; trajectories integrate with `scipy.integrate.solve_ivp`
  (already a dependency) or accept a precomputed `(n_steps, 2)` array from any solver.
- Example: FitzHugh–Nagumo phase plane (`example_12`).
- `hopf_point`/`hopf_parameter`: a differentiable, extended-system Hopf-point solver (implicit
  reverse-mode gradients via the implicit function theorem), mirroring `fold_point`/
  `fold_parameter`.
- `lyapunov_coefficient`: Kuznetsov's first Lyapunov coefficient `l1` — classifies a Hopf point as
  supercritical/subcritical/degenerate. `Hopf.refine()` now uses `hopf_point`/
  `lyapunov_coefficient` internally (an extended-system Newton solve replacing bisection) and
  reports `omega0`/`l1`/`criticality` on every detected Hopf point.
- Five direct codim-2 bifurcation point solvers, all differentiable in `args` via the same
  extended-system machinery: `cusp_point` (`CP`), `bogdanov_takens_point` (`BT`),
  `generalized_hopf_point` (`GH`), `zero_hopf_point` (`ZH`), `double_hopf_point` (`HH`) — plus
  `fold_coefficient`, the fold's own quadratic normal-form coefficient (needed by `CP`). Each has
  a `*_parameters` companion returning a bare, grad-ready parameter array. These refine a
  codim-2 point you can already guess at; they are not two-parameter continuation, which remains
  a separate, unstarted item.
- Independent cross-validation of `lyapunov_coefficient` and `bogdanov_takens_point` against
  BifurcationKit.jl v0.5.2 (`examples/BifurcationKit/04_hopf_normal_form.jl`,
  `examples/BifurcationKit/05_codim2.jl`).

### Changed
- `plot_phase_portrait` is renamed to `plot_branch_states`, which describes what it does:
  scatter branch points in state space. The old name remains as a deprecated alias and will be
  removed in v0.4.0.

### Fixed
- `bifurcations/taxonomy.py`'s `CP`/`BT`/`ZH`/`HH`/`GH` labels now have real implementations
  behind their reserved `"v0.3"` status.

## [0.2.0] - 2026-07-24

First PyPI release: `pip install jaxcont`. Adds periodic-orbit (limit-cycle) continuation,
Floquet multipliers, and period-doubling/Neimark–Sacker detection on top of v0.1.0's equilibrium
continuation.

### Added
- Periodic-orbit continuation via `periodic_orbit_problem`, using Gauss–Legendre orthogonal
  collocation.
- Floquet multipliers via a collocation monodromy matrix, replacing the old scipy stub;
  `compute_stability` now dispatches to Floquet analysis for periodic problems.
- `Event` protocol for bifurcation detection — `Fold`, `Hopf`, `PeriodDoubling`, `NeimarkSacker`
  — passed as `events=[...]` to `continuation()`.
- `LinearSolver`/`EigenSolver` protocols with `Dense`/`DenseEigen` defaults, exposed as a
  `Solvers` bundle on `continuation()`.
- Examples: Van der Pol and Brusselator limit cycles (`example_10`, `example_11`),
  period-doubling and Neimark–Sacker bifurcations (`example_08`, `example_09`).
- Practical bifurcation-analysis guide.

### Changed
- `scan_continuation`'s internal solves now route through the `LinearSolver`/`EigenSolver`
  protocols instead of a single hardcoded dense implementation.

### Fixed
- Duplicate/spurious fold-vs-Hopf flags near closely-spaced or lower-quality crossings
  (issue #7).
- Period-doubling/Neimark–Sacker near-unit-circle filter reverted to strict `<` after a boundary
  bug surfaced during verification.
- Solver-wiring baseline regression test now uses a float32-tolerant comparison.

### Removed
- `BifurcationDetector`, `FoldBifurcation`, `HopfBifurcation` — superseded by the `Event`
  protocol (`events=[Fold(), Hopf(), ...]`).

## [0.1.0] - Unreleased

### Added
- Initial supported release of JaxCont's equilibrium-continuation API.
- Functional `bif_problem` / `continuation` interface.
- Whole-loop compiled pseudo-arclength engine, used by default.
- Batched continuation sweeps with `jax.vmap`.
- Differentiable fold locations with implicit reverse-mode gradients.
- Core continuation framework
  - Natural parameter continuation
  - Pseudo-arclength continuation
  - Adaptive step size control
  - Predictor-corrector base class
- Problem definitions
  - Equilibrium problems
- Numerical solvers
  - Newton solver with JAX autodiff
  - Corrector methods
- Bifurcation detection
  - Fold bifurcation detector
  - Hopf bifurcation detector
  - Bifurcation point framework
- Stability analysis
  - Eigenvalue computation
  - Stability classification
- Utilities
  - Configuration system
  - Plotting functions
  - Bifurcation diagram generation
- Examples
  - Pitchfork bifurcation
  - Lorenz system
  - Neural-mass model
  - Batched imperfect-pitchfork sweep
  - Differentiable fold inverse design
- Testing
  - Test framework with pytest
  - Core functionality tests
  - Example validation
- Documentation
  - README with quick start
  - Installation guide
  - Development guide
  - Contributing guidelines
  - MIT License

### Known Issues
- Periodic orbits, Floquet multipliers, boundary-value problems, normal forms,
  branch switching, and two-parameter continuation are outside the supported
  v0.1 scope.

## [0.0.1] - 2025-11-10

### Added
- Project initialization
- Package structure
- Basic framework design

[0.3.0]: https://github.com/Ziaeemehr/JaxCont/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Ziaeemehr/JaxCont/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Ziaeemehr/JaxCont/compare/v0.0.1...v0.1.0
