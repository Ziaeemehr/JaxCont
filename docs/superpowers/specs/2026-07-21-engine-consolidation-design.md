# v0.2 engine consolidation — design spec

**Date:** 2026-07-21
**Roadmap reference:** `notes/ROADMAP.md`, "Engineering / architecture recommendations for v0.2",
item (i): *"Retire the three-implementations-of-one-algorithm pattern before adding a fourth
(periodic orbits)."*
**Related fix this builds on:** issue #13 (`jc.continuation()` `vmap`-safety, fixed 2026-07-21) —
the `Branch.valid` mask and the `_run_scan`/`_run_scan_traced` split introduced there are reused
here, not replaced.

## Problem

Three independent implementations of predictor-corrector continuation exist:

1. `core/natural_continuation.py` — `NaturalContinuation(PredictorCorrector)`, a Python-loop OO
   class. Has a known bug (issue #10): `compute_tangent()` uses central finite differences
   instead of `jacfwd`, and wraps `jnp.linalg.solve` in a bare `except:` that silently returns a
   zero tangent on any failure.
2. `core/pseudo_arclength.py` — `PseudoArclengthContinuation(PredictorCorrector)`, the sibling
   OO class. Got the `jacfwd`/bordered-solve fix that (1) never received.
3. `core/scan_continuation.py` — `pseudo_arclength_scan()`, the whole-loop `lax.while_loop`
   engine. Fully `jit`/`vmap`-safe (issue #13 made the *public* `jc.continuation()` wrapper
   around it `vmap`-safe too). Has no natural-continuation equivalent.

Consequence: a fix landed in (2) never propagated to (1) — direct, observed evidence that
three parallel implementations of one algorithm is an actively unsafe pattern, not just
theoretical duplication. Adding a fourth (periodic-orbit collocation, v0.2's actual feature work)
on top of this pattern would compound the problem.

## Decision: full removal, not deprecation

`NaturalContinuation`, `PseudoArclengthContinuation`, and `PredictorCorrector` are public,
**already published on PyPI as v0.1.0** exports, with real dependents:

- 32 tests across `test_adaptive_stepsize.py` (16), `test_bifurcation_workflow.py` (2),
  `test_bordered_newton.py` (5), `test_pseudo_arclength.py` (9) instantiate the classes directly.
- `examples/example_04_continuation_methods.py` and `examples/profile_continuation.py` use them
  directly.
- `examples/example_01/02/03/05` use `core/continuation.py`'s `equilibrium_continuation()` free
  function, which wraps `PseudoArclengthContinuation` internally (not exposed as a class, but the
  same dependency chain).

Despite this being a breaking change against a published release, the explicit decision (made
with the size of the blast radius stated up front) is: **remove all of it now** — the OO classes,
`PredictorCorrector`, `equilibrium_continuation()`, and `periodic_continuation()` (the latter is
already dead code: zero callers anywhere in the repo) — rather than freezing them as a permanent
deprecated shim. No new `NaturalContinuation`/`PseudoArclengthContinuation`-shaped API surfaces
should be reintroduced as a compatibility layer.

## Architecture

### New engine: `natural_scan()`

Added to `core/scan_continuation.py`, structurally mirroring `pseudo_arclength_scan()`:

- Reuses the exact `ScanResult` NamedTuple (`params`, `states`, `tangents`, `converged`,
  `n_valid`) rather than introducing a parallel `NaturalScanResult`. Natural continuation has no
  tangent concept, so `tangents` is zero-filled. This is a deliberate trade: a few unused zeros
  per buffer slot, in exchange for `_run_scan`/`_run_scan_traced` (and `Branch`'s pytree
  registration from issue #13) working unchanged for both engines with zero branching on result
  shape. Revisit only if a third engine (periodic-orbit collocation, v0.2 feature work) needs a
  result shape `ScanResult` genuinely can't express (e.g. per-point mesh data) — don't
  speculatively generalize before that's a real, observed need.
- Predictor: `p_pred = p + ds` (direction-signed), `u_pred = u` (state held fixed — this is
  natural continuation's defining property and also why it stalls at folds: no tangent to
  reorient with).
- Corrector: plain Newton on `f(u, p_pred) = 0` with `p_pred` fixed, using `jacfwd(f, argnums=0)`
  — no bordered `(n+1)×(n+1)` system, no arclength constraint row. Written fresh; does not import
  or adapt any code from the deleted `natural_continuation.py`.
- Same jit/vmap contract as `pseudo_arclength_scan`: `f` and `max_steps` static, everything else
  traced-array-compatible, bounded by `max_steps` × `max_iter` with an `isfinite` guard.

### `api.py` changes

- `PseudoArclength` drops its `engine: Literal["legacy", "scan"]` field — only one engine exists,
  so there is nothing left to select. (Breaking change to `PseudoArclength`'s constructor if
  anyone passed `engine=...`; acceptable per the "remove now" decision above.)
- `_run_scan` and `_run_scan_traced` (introduced for issue #13) gain a `scan_fn` parameter so both
  `Natural` and `PseudoArclength` share one reassembly/detection/traced-fallback code path,
  calling `natural_scan` or `pseudo_arclength_scan` respectively. No duplicated logic between the
  two algorithm branches beyond the engine call itself.
- `continuation()`'s dispatch collapses from three branches (scan / legacy PseudoArclength /
  legacy Natural) to two (`Natural` / `PseudoArclength`), both hitting `_run_scan`.
- `Branch.valid`/pytree registration from issue #13 needs no changes — both engines produce the
  same `ScanResult`-shaped buffers, so the traced-path logic is unaffected.

### Deletions

- `src/jaxcont/core/natural_continuation.py` (whole file)
- `src/jaxcont/core/pseudo_arclength.py` (whole file — the class and its `_correct_jit` helper;
  nothing else in the file has other callers, confirmed by grep)
- `src/jaxcont/core/predictor_corrector.py` (whole file — `PredictorCorrector` has exactly two
  subclasses, both being deleted; confirmed by grep, no third subclass exists)
- `equilibrium_continuation()` and `periodic_continuation()` in `src/jaxcont/core/continuation.py`
  (free functions only; `ContinuationProblem`/`ContinuationSolution` classes in the same file stay
  — still used by `api.py`'s `_to_legacy_problem`/`_to_result` adapters and `BifurcationDetector`)
- Exports removed from `src/jaxcont/__init__.py` (`NaturalContinuation`,
  `PseudoArclengthContinuation`, `PredictorCorrector`, `equilibrium_continuation`) and
  `src/jaxcont/core/__init__.py` (same, plus `periodic_continuation`)

### Out of scope for this project (explicitly)

- `BifurcationDetector` / `Event` protocol rewrite (roadmap item iv) — untouched here. The
  detector keeps working exactly as it does today, fed by whichever engine produced the branch.
- `equinox` adoption for periodic-orbit types (item ii), `fold_solve.py` pattern generalization
  (item iii), `LinearSolver`/`EigenSolver` protocols (item v) — separate projects.
- Actually building periodic-orbit collocation (v0.2's feature work) — this project is the
  prerequisite cleanup, not the feature itself.

## Test migration

Two distinct patterns, not one mechanical find-replace:

1. **High-level `.run(problem, param_range)` callers** — `test_adaptive_stepsize.py` (16 tests),
   `test_bifurcation_workflow.py` (2 tests): rewrite to
   `jc.continuation(jc.bif_problem(...), jc.PseudoArclength(), p_span=..., settings=...)`,
   adapting `sol.states`/`sol.parameters` → `sol.branch.states`/`sol.branch.params`.
2. **Granular `.predict()`/`.correct()`/`.compute_tangent()` callers** — `test_bordered_newton.py`
   (5 tests) and part of `test_pseudo_arclength.py` (9 tests): these are white-box tests of the
   bordered-Newton corrector itself (where issue #1's fix is actually verified step-by-step).
   Rewrite against `scan_continuation.py`'s private `_tangent`/`_newton_correct` functions,
   following the same white-box import pattern `test_functional_api.py::TestScanEngine` already
   uses for `pseudo_arclength_scan`. Each test's *intent* (not just its call shape) needs
   preserving — e.g. a test asserting `correct()` reports non-convergence after a bad step needs
   its equivalent assertion reformulated against `_newton_correct`'s return signature, which
   differs from the deleted class's `.correct()` signature.

Coverage on engine-path files (currently ≥85%, per `notes/ROADMAP.md`'s release-engineering
section) must not regress as a result of this migration.

## Example migration

All 6 non-`example_06`/`07` gallery examples are touched:

- `example_01_pitchfork.py`, `example_02_lorenz.py`, `example_03_van_der_pol.py`,
  `example_05_neural_mass.py` — currently call `equilibrium_continuation()`; rewrite to
  `jc.continuation()`.
- `example_04_continuation_methods.py` — currently instantiates both OO classes directly to
  demonstrate the natural-vs-pseudo-arclength contrast; rewrite to compare
  `jc.continuation(prob, jc.Natural(), ...)` vs `jc.continuation(prob, jc.PseudoArclength(), ...)`.
- `examples/profile_continuation.py` — currently instantiates `PseudoArclengthContinuation`
  directly for profiling; rewrite to profile `jc.continuation()` (or `pseudo_arclength_scan`
  directly, matching how `example_06`/`07` already profile at the engine level).

Each migrated example must be re-run headless (`MPLBACKEND=Agg`) after migration to confirm it
still produces its expected output/plot, matching the verification already done for `example_06`
this session — cross-validated examples (`example_01/02/03/05`, per `notes/ROADMAP.md`) must
still match their BifurcationKit.jl reference values after the rewrite, not just "run without
crashing."

## Verification plan

- Full test suite green (`pytest tests/`), no regressions outside the intentionally-migrated
  files.
- Coverage on engine-path files (`scan_continuation.py`, `api.py`, `pseudo_arclength.py`'s
  successor code, `natural_scan`) stays ≥85%.
- Each of the 6 migrated examples re-run headless; cross-validated ones re-checked against their
  hardcoded BifurcationKit.jl reference values.
- `grep` sweep confirming zero remaining references to `NaturalContinuation`,
  `PseudoArclengthContinuation`, `PredictorCorrector`, `equilibrium_continuation`,
  `periodic_continuation` anywhere in `src/`, `tests/`, `examples/` (excluding this spec and the
  roadmap's historical notes).

## Risks

- **Silent behavior drift in migrated tests.** Rewriting 32 tests from one API shape to another
  risks accidentally weakening an assertion rather than faithfully porting it. Mitigate by
  reviewing each test's original intent (docstring, what bug/behavior it guards against) before
  rewriting, not just mechanically swapping call syntax.
- **`natural_scan()` is new code with no prior art in this codebase to adapt from** (unlike
  `pseudo_arclength_scan`, which came from hardening an existing corrector). Higher first-draft
  risk; needs its own direct unit tests (fold-stalling behavior, agreement with the deleted
  `NaturalContinuation` class's *correct* runs on simple systems, before that class is deleted) —
  not just coverage-by-example.
- **Example cross-validation regressions.** `example_01/02/03/05` are checked against hardcoded
  BifurcationKit.jl values; a subtle change in default settings during the `equilibrium_continuation()`
  → `jc.continuation()` rewrite could silently shift results within tolerance but away from the
  cross-validated behavior. Mitigate by diffing default `ContinuationPar` values against
  `equilibrium_continuation()`'s current `**kwargs` defaults before rewriting each example.
