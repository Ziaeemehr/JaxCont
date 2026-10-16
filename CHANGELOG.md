# Changelog

All notable changes to JaxCont will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - Unreleased

### Known limitations
- Near a Hopf seed, periodic branches can report spurious folds at very small amplitude.

### Breaking
- PD/NS refinement rejects unfinished or unresolved event brackets with
  `reason="not_converged"`. The default event refinement budget is now 50
  iterations to resolve float64 brackets. Direct refinement requires `PeriodicArgs` and
  raises `ValueError` when it is missing.
- Shared-seed events in `bothside` continuation belong to the forward leg.
  Spectral scans preserve distinct crossing modes within a single bracket;
  Hopf refinement follows each candidate's mode.
- Both continuation engines report `nonfinite` for nonfinite corrector
  trials while retaining only finite accepted branch points.
- `floquet_multipliers(...)` now returns `(multipliers, trivial_index)`;
  `branch_floquet_multipliers(...)` returns the corresponding batched arrays.
  The trivial multiplier is selected by eigenvector alignment with the flow.
  `floquet_stable(multipliers, trivial_index)` now requires those indices.
  Periodic `Branch` objects expose `trivial_index` when spectra are computed.
  Direct PD/NS scans require `BranchPoint.trivial_index`; `detect_events`
  accepts the per-point `trivial_index` array and continuation supplies it.
  PD/NS detection and refinement exclude this flow mode.
- Eager `periodic_orbit_problem` construction now raises `ValueError` when
  `period0` is nonpositive, the trajectory spans less than `0.95 * period0`,
  refinement fails to converge, the refined period is nonpositive, or the
  refined orbit amplitude is at most `10 * tol * (1 + norm(mean(u)))`. These checks are skipped for
  traced values.
  Trajectory times are rebased to their first sample; only `[0, period0)` of the
  trajectory is resampled (later laps are not folded in), and a gap between the
  last sample and `period0` is closed linearly back to the first sample. The new
  `max_iter` keyword defaults to 50.
- A periodic problem's `args` is now `jaxcont.problems.periodic_args.PeriodicArgs` (an
  `eqx.Module`: dynamic `u_ref_coll`, `uref_prime_coll`, `user_args`; static `raw_f`, `mesh`)
  instead of the tuple `(u_ref_coll, uref_prime_coll, raw_f, mesh)`, and the problem's `f` is the
  module-level `periodic_residual`, so repeated continuations of the same `raw_f` and mesh share one
  compiled kernel. `raw_f` is always called as `raw_f(u, p, user_args)`; the internal `None` is
  gone.
- `PeriodDoubling` and `NeimarkSacker` no longer take `raw_f`/`mesh`: write `jc.PeriodDoubling()`.
  `continuation()` hands them the problem's `PeriodicArgs` through `detect_events(..., args=...)`.
- `BP.rank_tolerance` now thresholds a scale-invariant ratio: the smallest singular
  value of `[f_u | f_p]` at the refined point divided by the largest singular value of
  that matrix over the two bracket endpoints and the refined point, replacing the
  absolute smallest-singular-value criterion. Its default is now `None`, resolved to
  `1e3 * default_tol(dtype)` (strict comparison). Scaling the right-hand side no longer changes
  classification, scalar systems work, and an all-zero reference scale counts as rank
  deficient.
- `ContinuationResult.events` now holds only validated hits. A refinement that did not
  converge, landed outside its bracket (widened by one bracket width per side) or failed a
  spectral check is no longer reported as a located event at the right bracket end; it is
  `EventHit(status="failed")` with `info["reason"]` and `p = nan`, listed in
  `rejected_events`. `Hopf`/`Fold`/`BP`/codim-2/`PeriodDoubling`/`NeimarkSacker` `refine()`
  results lose the old `info["converged"]: False` / `criticality="unknown"` form. Plots show
  validated hits only.
- `differentiable_root` returns a seed that already meets the residual tolerance unchanged
  (previously a singular Jacobian at the root, e.g. `G = x**2` at `0`, produced NaN).
- `ContinuationPar.newton_tol` and the `tol` arguments of the solvers, fold/Hopf/codim-2
  point finders, curve factories, `branch_switch` and `dprc_curve` now default to `None`,
  resolved from the state dtype (`1e-9` for float64, `1e-5` for float32). Previous fixed
  defaults (`1e-6`/`1e-8`) are gone; `detect_events(tolerance=...)` likewise. Periodic-orbit
  problems no longer need an explicit `newton_tol=1e-5`.
- `pseudo_arclength_scan` / `natural_scan` now take `args` as a dynamic input right after
  `p_end`, and `f` is called as `f(u, p, args)`; `branch_eigenvalues(f, states, params, args,
  eigen_solver)` likewise. `continuation()` reuses its compiled kernel across calls and
  across changed `args` values (a fresh lambda `f` per call still recompiles). Only array
  leaves of `args` are dynamic: pass arrays to get compile-cache reuse. Python scalar and
  callable leaves stay concrete (usable as shapes or flags) and recompile when changed.
- `continuation()` no longer lets rejected steps consume `max_steps`: total corrector
  attempts are bounded by the new `ContinuationPar.max_attempts` (default `4 * max_steps`),
  the initial `ds` is clipped to `[ds_min, ds_max]`, and a run stalls only when a step already
  attempted at `ds <= ds_min` fails. A seed point that fails to converge now raises
  `ValueError` in eager mode (under `jit`/`vmap` nothing is marked valid). `pseudo_arclength_scan`
  / `natural_scan` take a trailing `max_attempts` argument and `ScanResult.n_valid` is `0` on
  seed failure.
- Spectral events detect on tracked modes. `Hopf`, `BP`, `PeriodDoubling`, `NeimarkSacker`,
  `BogdanovTakens`, `ZeroHopf` and `DoubleHopf` reorder each point's spectrum into continuous
  modes (`jaxcont.bifurcations.modes.track_modes`) and report a bracket only where one mode
  meets its condition at both ends (Hopf: `Re` crosses 0 on a complex mode; BP: a real mode
  crosses 0; PD: a real multiplier crosses -1; NS: a complex multiplier crosses the unit
  circle). Previously Hopf/BP/BT/ZH/HH re-selected the candidate nearest critical at every
  point, inventing crossings when two modes traded places, and PD/NS followed one multiplier
  forever, missing crossings by any other. Each refined point is re-checked against the
  spectrum (`reason="spectral_check_failed"` when the eigenvalue nearest the critical value
  has the wrong real/complex type). These events now expose `scan(points)` instead of
  `test_function`; `refine()` takes the bracket's `mode`. Removed: `select_candidate`,
  `test_value`, `near_unit_circle` (PD/NS) and `near_critical` (BT/ZH/HH). Their `tolerance`
  (the `|Im|` threshold between real and complex) defaults to `None`, resolved to
  `sqrt(default_tol)` of the spectrum's dtype. PD/NS refinement re-corrects the final
  bracket midpoint onto the collocation residual and stores the critical multiplier in
  `info["multiplier"]`. A Hopf curve that ends with `|omega| < 10 * default_tol` and no
  sign change records a `BogdanovTakens` entry with `reason="curve_terminated"` in
  `rejected_events`.
- Event deduplication no longer merges on parameter distance alone. Two validated hits of
  one kind merge only when their brackets are adjacent and their joint `(u, p)` distance is
  at most `2 * max(ds_i, ds_j)` (per-point `ds`), so hits on different sheets of a folded
  branch at the same `p` are both kept, and a nearby but distinct event is no longer
  swallowed. An exact zero of a test value at a sample is reported once (it owns the
  preceding bracket), including on the `test_function` path, which previously ignored it.
  `Event.test_function` values are evaluated in one `jax.vmap` call where the function is
  array-pure, per point otherwise; `Fold.test_function` now returns an array.
  `detect_events` takes an optional `ds_values`.
- `pseudo_arclength_scan` / `natural_scan` take a static `step_update` (or `None`) as their
  second positional argument, right after `f`; positional calls must insert it.
- A Hopf-curve problem's `args` is now `jaxcont.bifurcations.curves.HopfCurveArgs`
  (`q1_anchor`, `q2_anchor`, `user_args`) instead of the caller's `args`, which moves to
  `user_args`. The phase condition is anchored to that eigenvector instead of one recomputed
  from the seed point, and is re-anchored on every accepted point, so a curve whose critical
  eigenvector rotates far from the seed's no longer stalls or jumps.
- Periodic-orbit continuation re-anchors the integral phase condition on each accepted orbit
  (`periodic_step_update`), so the residual of a branch point is zero with the phase reference
  of the point before it, not with the problem's original `args`. A cycle that turns far from
  its starting orbit (e.g. reverses its sense of rotation) no longer stalls.

### Added
- `BifProblem.step_update(args, u, p) -> args` (and `bif_problem(..., step_update=...)`):
  per-step refresh of auxiliary `args` data such as a phase-condition reference. The scan
  engines carry `args` and apply it to the corrected seed and after every accepted step;
  `detect_events(..., step_update=, f=)` refines each bracket with
  `step_update(args, u_left, p_left)`. `jaxcont.problems.periodic.periodic_step_update` and
  `jaxcont.bifurcations.curves.hopf_curve_step_update` are the periodic and Hopf-curve hooks.
- `ContinuationResult.rejected_events` and `EventHit.status` (`"validated"` or `"failed"`),
  with `info["reason"]` in `"not_converged"`, `"outside_bracket"`, `"spectral_check_failed"`.
- `ContinuationResult.status` (`"reached_end"`, `"max_steps"`, `"stalled"`, `"nonfinite"`,
  `"seed_failed"`, `"max_attempts"`) and `stats["n_attempts"]` / `stats["status"]`; under
  `jit`/`vmap` `status` is `None` and `stats["status"]` is the integer
  `jaxcont.core.scan_continuation.Status` code. `ScanResult` gains `status` and `n_attempts`.
- `periodic_orbit_problem(..., args=...)`: user `args` reach the right-hand side at every
  evaluation and are differentiable (`jax.grad` of the converged period with respect to them).
  `floquet_multipliers`, `branch_floquet_multipliers`, `prc_curve`, `branch_prc` and
  `branch_switch_pd` take a keyword `args=` for the same purpose; `detect_events` takes
  `args=` (the problem's `args`).
- `branch_switch_pd(..., refine_tol=...)`: the Newton tolerance for each candidate's
  correction (default `None`, the `default_tol` that `periodic_orbit_problem` uses).
  Systems with a continuous symmetry that leaves the doubled orbit unpinned need a
  looser explicit value.
- Top-level exports `periodic_orbit_problem`, `Collocation`, `prc_curve`, `branch_prc`,
  `dprc_curve`, and `PeriodicArgs` in `jaxcont.problems.periodic_args`.

## [Unreleased]

### Added
- `ContinuationPar(bothside=True)`: explore both directions from `problem.p0` in one
  `continuation()` call (issue #8), matching BifurcationKit.jl's `bothside=true`. Runs the
  existing single-direction scan engine twice and stitches the two legs into one branch; no
  changes to `core/scan_continuation.py`. Eager-only (raises `NotImplementedError` under
  `jax.vmap`/`jax.jit`, matching the existing `events=[...]` restriction).
- MatCont cross-validation for two-parameter fold/Hopf curves: `examples/MatCont/cases.json`'s
  `US-C2-LP-001`/`US-C2-H-001` promoted to `supported`, cross-checked against a live MatCont 7.6
  run of the catalytic-oscillator LP/Hopf curves (`examples/MatCont/python_cases/codim2_curves.py`).
- `jc.BP`/`jc.branch_switch`: detect equilibrium branch points (pitchfork/transcritical
  crossings) and compute the crossing branch's tangent, closing the last v0.3.0+ roadmap
  item. Period-doubling switching remains a separate future item.
- `jc.branch_switch_pd`: switch from a detected period-doubling bifurcation onto the
  period-doubled cycle, completing the branch-switching feature. Verifies the switched
  orbit is genuinely period-doubled rather than the original orbit traversed twice, and
  raises instead of returning an unverified branch. Eager-only.

### Fixed
- `periodic_orbit_problem(...)` construction is now safe under `jax.grad`, including when
  trajectory and period guesses depend on the differentiated parameter. All parameter-dependent
  phase-reference and seed values are passed explicitly through the implicit solve instead of
  leaking captured tracers across its Newton `lax.while_loop`; existing `jax.vmap` and `jax.jit`
  construction behavior is preserved.

## [0.4.0] - 2026-08-20

### Added
- Two-parameter continuation of equilibrium fold and Hopf curves via
  `fold_curve_problem` and `hopf_curve_problem`, including codimension-two
  events detected along those curves and `plot_two_parameter_diagram`.
- Infinitesimal phase-response curves (`prc_curve` and `branch_prc`) and
  parameter-derivative sensitivity (`dprc_curve`), plus `plot_prc` and
  independent/MatCont validation coverage.
- MatCont visual comparison gallery: five new example scripts (`example_16`-`example_20`)
  overlaying JaxCont branches directly on reviewed MatCont 7.6 reference data for the cubic
  fold, Van der Pol Hopf, adaptive-control Hopf, radial periodic orbit, and torBPC cases, plus
  a featured comparison figure in the README.

### Changed
- The deprecated `plot_phase_portrait` compatibility alias is retained for one
  more release and is now scheduled for removal in v0.5.0.
- **Breaking:** `requires-python` narrowed from `>=3.9` to `>=3.11`. `lyapunov_coefficient`
  relies on JAX's SVD-JVP rule for `full_matrices=True`, which was only implemented around
  jax 0.9.0; `jaxlib` never published a `cp310` or `cp39` wheel at or above that version, so
  Python 3.9/3.10 cannot install a working `jax`/`jaxlib` pair for this codebase at all.
  Confirmed via CI: both versions failed with `NotImplementedError: Singular value
  decomposition JVP not implemented for full matrices` on every jax release available to
  them. Declared `jax`/`jaxlib` floors corrected accordingly, from an untested `>=0.3.0`
  placeholder to the empirically bisected `>=0.9.0`; `numpy` floor corrected to `>=1.22.0`.
  CI now tests the full declared Python 3.11-3.12 matrix instead of only 3.11.
- **Breaking:** `fold_point()` now returns a 4-tuple `(u, p, v, converged)` (was a 3-tuple);
  `hopf_point()` now returns a 6-tuple `(u, p, q1, q2, omega0, converged)` (was a 5-tuple).
  Existing code unpacking the old shape (e.g. `u, p, v = jc.fold_point(...)`) will now raise
  `ValueError: too many values to unpack`.
- `continuation()`'s `p_span[0] == problem.p0` validation now applies to every problem kind
  (previously only checked for `fold_curve`/`hopf_curve` kinds) -- a mismatch that previously
  silently started the branch at the wrong point now raises `ValueError` instead.
- **Breaking:** `ContinuationSolution.save()`/`.load()` now use a versioned (`format_version=1`),
  pickle-free `.npz` schema. Archives written by the previous implementation cannot be loaded
  (the previous format was broken for any solution with an optional field left as `None` -- see
  Fixed, below -- so no working archives from it exist to migrate).

### Fixed
- `BoundaryValueProblem`'s class docstring now states up front that it is an unimplemented
  placeholder (`solve_collocation`/`solve_shooting` always raise `NotImplementedError`),
  instead of only saying so in an inline comment inside each method body.
- The continuation seed (`u0`) is now Newton-corrected/validated before entering the branch,
  instead of being accepted unconditionally.
- `PeriodDoubling`/`NeimarkSacker` event refinement now Newton-corrects interpolated periodic
  orbits before evaluating Floquet multipliers, instead of trusting a linear interpolation.
- `ContinuationPar(adaptive=False)` now actually disables step-size adaptation: a
  successful step no longer grows `ds`, instead of silently growing it; the effective step
  size can still be smaller than the originally requested `ds` if an earlier step in the run
  failed and shrank it (there is no recovery back up). A failed fixed-size step still backs
  off (and the run still terminates via the existing `ds <= ds_min` stall condition) rather
  than retrying forever.
- `convergence_info` entries now report the real Newton iteration count for each accepted
  step instead of a hardcoded `0`, and `continuation(..., verbose=True)` now prints a
  one-line summary instead of doing nothing.
- `ContinuationSolution.save()`/`.load()` no longer raises `TypeError` when `eigenvalues`,
  `stability`, or `tangent_vectors` is `None`; these fields now round-trip correctly, along with
  `convergence_info`, `state_names`, and `param_name`, none of which the previous format
  preserved at all.
- `ContinuationSolution.load()` now defaults to `allow_pickle=False`, so loading an untrusted
  `.npz` file can no longer trigger arbitrary pickle execution.

## [0.3.1] - 2026-08-05

### Fixed
- `PeriodDoubling`/`NeimarkSacker`'s `near_unit_circle` filter margin widened (0.5 -> 0.9): the
  old value left as little as a 0.02 margin between a genuine candidate multiplier and the
  window boundary in some continuations, thin enough that ordinary hardware/collocation
  floating-point differences could push it outside the window and silently drop a real
  bifurcation detection. Reproduced deterministically on CI, not on all hardware.
- Read the Docs build: `examples/MatCont`'s validation-suite modules
  (`artifacts.py`/`compare.py`/`registry.py`/`run_validation.py`) were being swept into the
  Sphinx-Gallery scan as title-less example pages, breaking the `fail_on_warning` docs build
  introduced alongside v0.3.0's MatCont validation suite. Excluded via `ignore_pattern` and
  given a proper `GALLERY_HEADER.txt`.

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
  removed in v0.5.0 (its original v0.4.0 removal schedule was deferred).

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

## [0.1.0] - 2026-07-19

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

[0.4.0]: https://github.com/Ziaeemehr/JaxCont/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/Ziaeemehr/JaxCont/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Ziaeemehr/JaxCont/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Ziaeemehr/JaxCont/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Ziaeemehr/JaxCont/compare/v0.0.1...v0.1.0
