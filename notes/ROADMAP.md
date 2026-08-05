# JaxCont Roadmap — Single Source of Truth

**Last updated:** 2026-08-05
**Current version:** 0.3.0 — tagged `v0.3.0`, GitHub release cut. Adds phase-plane visualization,
the Hopf normal form / `l₁` criticality classifier, and five direct codim-2 bifurcation point
solvers (`CP`/`BT`/`GH`/`ZH`/`HH`) on top of v0.2.0's periodic-orbit continuation. **Zenodo DOI:
not yet minted** — repo-side prep (`CITATION.cff`, README badge slot) is ready, but archival
requires enabling the GitHub-Zenodo integration at zenodo.org/account/settings/github/ (an
account-level action, not something automatable from here); once enabled, this release (or the
next one) will be the one archived. PyPI publish for v0.3.0 is a separate, independent step, not
yet done as of this edit.
**Scope decision:** Ship a focused **equilibrium continuation** library first. See [PROJECT_REVIEW_2026-07.md](PROJECT_REVIEW_2026-07.md) for the full rationale.
**API design:** Committed to a functional, diffrax-style surface (`continuation(problem, alg, ...)`).
See [ARCHITECTURE.md](ARCHITECTURE.md) for the full spine contract and provisional v0.2+ API.

**2026-07-19 pass:** re-verified the state below against a real test/coverage run and the MatCont
7.1 manual (the closest thing to a canonical taxonomy of what a "complete" continuation toolbox
covers). Added the two new sections at the bottom — **Strategic direction beyond v0.1** and
**Engineering recommendations for v0.2** — answering "should we match MatCont's feature list, or
go further?" The short answer: match MatCont's *most-used* subset (equilibria + limit cycles +
their codim-1 bifurcations), skip its most expensive corners (homoclinic/heteroclinic orbits, PRC,
GUI) indefinitely, and spend the saved effort on the differentiability/`vmap`/GPU story MatCont
cannot offer at all. See below for the reasoning.

> This is the *only* file that tracks status and next steps. Older planning/status
> notes are archived in [old_plans/](old_plans/) for reference (profiling data, etc.)
> but are **superseded** by this file. If they disagree, this file wins.

---

## Status at a glance

| Area | State | Notes |
|------|-------|-------|
| Pseudo-arclength continuation | ✅ Works, tested | Passes fold points |
| Natural-parameter continuation | ✅ Works, tested | |
| Newton solver (autodiff) | ✅ Works, 100% cov | Not yet JIT'd on hot loop |
| Fold + Hopf detection | ✅ Works, tested | Both refine via the extended-system Newton solve (implicit diff), not bisection |
| Stability (eigenvalues) | ✅ Works, 98% cov | Fixed 2026-07-19 (was 51%) |
| Naming/abbreviation reference | ✅ Works, tested | [bifurcations/taxonomy.py](../src/jaxcont/bifurcations/taxonomy.py) |
| Plotting | ✅ Works, tested | Consolidated into `jaxcont/viz/` (2026-07-22) |
| Periodic orbits | ⚠️ Stub | Untested, hidden from v0.1.0 |
| Floquet multipliers | ❌ Stub, 22% cov | Hidden from v0.1.0 |
| BVP (collocation/shooting) | ❌ NotImplementedError | Hidden from v0.1.0 |
| Normal forms / Lyapunov coeff | ❌ Returns placeholders | Hidden from v0.1.0 |

**Test suite (re-verified 2026-07-19, after this session's fixes, CPU):** 68 passed, 21
deselected (18 `slow` + 3 `gpu`), 0 failed. **Same 68, real GPU backend (no `JAX_PLATFORMS`
override):** also 68 passed, 0 failed (verified 4+ repeated runs — see issues #11/#14).
**GPU-marked suite** (`pytest -m gpu`, real GPU only): 3 passed — [tests/test_gpu_smoke.py](../tests/test_gpu_smoke.py).
**Coverage (re-verified 2026-07-19):** 73% overall. Engine-path files are ≥85%:
`scan_continuation.py`/`newton.py`/`fold_solve.py` 100%, `pseudo_arclength.py` 97%, `api.py` 95%,
`hopf.py` 93%, `fold.py` 91%, `taxonomy.py` 86%, `detector.py` 87%. **`stability/eigenvalue.py` is
now 98%** (was 51% — the one gap the roadmap named explicitly; fixed this session, see
"Remaining v0.1.0 work" below, now essentially closed).

---

## Known issues to fix before release

1. ✅ **Broken singular-matrix handling.** *(fixed 2026-07-18)* The corrector now solves the
   full `(n+1)×(n+1)` bordered system instead of eliminating through `df/du`, so it no longer
   inverts a matrix that is singular at folds. The dead `try/except` is gone.
2. ✅ **Finite-difference `df/dp`.** *(fixed 2026-07-18)* Replaced with `jacfwd(f, argnums=1)`
   in both `correct()` and `compute_tangent()`.
3. ✅ **JIT on the hot loop.** *(solved 2026-07-18 — engine validated, wiring pending)* First
   attempt (JIT the corrector alone) gave ~no speedup — the cost was the Python outer loop.
   The real fix is the **whole-loop `lax.while_loop`** engine in
   [`core/scan_continuation.py`](../src/jaxcont/core/scan_continuation.py): the entire sweep is
   one compiled program over fixed-size buffers. Validated on pitchfork: residual 5e-8, **0.74 ms
   warmed vs ~250 ms** for the Python loop (~340×), `vmap`-batches 64 runs in one kernel, and runs
   through `jax.grad`. Remaining: wire it in behind `continuation()` and port detection.
4. ✅ **README placeholders.** *(fixed 2026-07-18)* Author, repository, and
   citation metadata are now real; DOI text explicitly waits for the Zenodo archive.
5. ✅ **Saturating-branch hang.** *(fixed 2026-07-18 by the scan engine)* The whole-loop engine is
   structurally bounded (≤ `max_steps` × ≤ `max_iter` iterations) with an explicit `isfinite`
   guard in the Newton loop, so degenerate branches (`r − tanh(x)` into saturation) terminate
   cleanly instead of hanging — verified: 0.30 s, clean stop. The `slow`-marked
   `tests/test_adaptive_stepsize.py` can rejoin the fast suite once the engine is wired in and the
   tests are pointed at it.
6. ✅ **`ds_min` break condition off-by-boundary.** *(fixed 2026-07-18)* `predictor_corrector.py`'s
   stall check used `abs(ds) < self.ds_min`, but `adapt_stepsize` clamps a shrinking `ds` to
   *exactly* `ds_min` on failure — so once pinned there the loop never satisfied the strict `<`
   and could spin forever if the corrector kept failing at the floor step size. Changed to `<=`.
   Found while cross-validating `examples/example_05_neural_mass.py` against BifurcationKit.jl.
7. ⚠️ **Bifurcation detector produces duplicate/spurious fold-vs-Hopf flags near closely-spaced
   or lower-quality crossings.** *(found 2026-07-18, cross-validating `example_02_lorenz.py` and
   `example_05_neural_mass.py` against real BifurcationKit.jl v0.5.2 runs on the identical
   equations)* Where detections land close to a true bifurcation (within ~0.005 in the parameter,
   about one continuation step), the *locations* JaxCont finds are accurate — but the detector
   sometimes (a) flags the same true Hopf point twice, once correctly as `hopf` and once
   mislabeled as `fold`, and (b) emits one clearly spurious `fold` with no BifurcationKit.jl
   counterpart at all (`example_05`, E0≈-1.550). Both examples now print an explicit comparison
   table against hardcoded BifurcationKit.jl reference values so this is visible rather than
   hidden. → needs a real fix in `bifurcations/detector.py` (likely: de-duplicate near-coincident
   fold/Hopf flags, and tighten the fold test function to reduce false positives) before the
   detector can be trusted unsupervised; not done here, flagged for v0.1.0 hardening.
8. ℹ️ **`run()` only continues in one direction per call.** BifurcationKit.jl's `bothside=true`
   explores both directions from the initial point in one call; JaxCont's `run()` picks a single
   direction from `param_range` vs. the starting parameter. Not a bug — pseudo-arclength still
   passes folds and can reach the region of interest with a well-chosen range (see
   `example_02_lorenz.py`) — but it's a real ergonomics gap vs. BifurcationKit.jl worth a
   `bothside` option in the functional API (`ContinuationPar` or `continuation()`) later.
10. ⚠️ **`natural_continuation.py` still has both bugs that #1/#2 supposedly fixed — only the
   pseudo-arclength path was fixed.** *(found 2026-07-19, re-reading the source)*
   `NaturalContinuation.compute_tangent()` computes `df/dparam` by **central finite difference**
   (`eps=1e-6`) instead of `jacfwd`, and wraps the tangent's `jnp.linalg.solve` in a **bare
   `except:`** that silently returns a zero tangent on any failure (not just `LinAlgError`).
   `pseudo_arclength.py` got the `jacfwd` + bordered-solve fix; this sibling class, which
   implements the exact same predictor-corrector interface, did not. Root cause: three parallel
   implementations of the same algorithm (`natural_continuation.py`,
   `pseudo_arclength.py`'s legacy class, `scan_continuation.py`) mean a fix in one doesn't
   propagate to the others. Low severity today (natural continuation is a teaching/comparison
   path, not the default engine, and is exercised only by
   `examples/example_04_continuation_methods.py`), but worth fixing or deleting before v0.2 adds
   more algorithm variants on top of this pattern — see "Engineering recommendations" below.
11. ℹ️ **Local dev-machine GPU: usable, but cuDNN is broken and noisy — verified, not just
   suspected, 2026-07-19.** *(revised after actually running real GPU workloads — see the v0.1.0
   GPU-smoke-test entry below)* This machine has a real GPU (`nvidia-smi`: RTX A5000, driver
   535.183); `jax.devices()` lists a `CudaDevice`, and the installed jaxlib's bundled cuDNN
   refuses to initialize against that driver (`CUDNN_STATUS_NOT_INITIALIZED`, logged repeatedly).
   **However**, cuDNN is only needed for convolution-style ops JaxCont never uses — a battery of
   real GPU tests (`tests/test_gpu_smoke.py`: a dense linear solve, a full `jc.continuation()`
   run, and a `vmap`-batched sweep) all **pass correctly on this GPU**, cuDNN noise
   notwithstanding. The one thing that *did* fail under the real GPU backend was a pre-existing
   test bug, not a GPU/driver issue — see issue #14. `JAX_PLATFORMS=cpu` is still worth knowing as
   a way to silence the cuDNN log noise, but "no environment exercises the GPU story" (the
   original, hastier version of this note) was wrong — corrected here rather than left standing.
12. ⚠️ **Recurring pattern: `newton_tol`/`NewtonSolver(tol=...)` set below float32 machine epsilon
   (~1.2×10⁻⁷) silently reports `converged=False` forever, even at points where the true residual
   is already at the numeric floor.** Found independently three times this session — issue #5/#6
   (`example_05`, triggered the `ds_min` hang), and again in the (now-removed) old
   `example_04`/`example_06`, where `newton_tol=1e-8`/`1e-10` caused most steps to silently report
   non-convergence despite a correct numeric answer (only visible because the printed error stayed
   ~0 regardless — the convergence *flag* was wrong, not the computed value). → worth either (a) a
   one-line doc note on `newton_tol`'s float32 floor, or (b) `NewtonSolver`/the correctors warning
   when constructed with `tol < ~1e-6` in float32. All shipped examples now use `tol >= 1e-6`; as
   of issue #14, `tests/test_pseudo_arclength.py` does too.
13. ✅ **`jc.continuation()` branch/stability data is now `vmap`-safe.** *(found 2026-07-19, fixed
   2026-07-21)* `api.py`'s `_run_scan()` did `n = int(res.n_valid)` to trim the fixed-size buffer
   to a Python-level ragged length — a bare `int()` on a traced value, raising
   `jax.errors.ConcretizationTypeError` under `jax.vmap`/`jax.jit`. **Fix:** `_run_scan()` now
   catches that error and falls back to `_run_scan_traced()`, which returns the fixed-size engine
   buffers as-is plus a new `Branch.valid` boolean mask (mirrors the existing
   `examples/example_06_vmap_sweep.py` pattern: trim per-batch-element with `n_valid` after the
   trace exits) instead of trying to concretely trim. Eager (non-traced) calls are byte-for-byte
   unchanged. **A second, previously-undiscovered blocker surfaced and was fixed alongside this**:
   `Branch`/`ContinuationResult` were never registered as JAX pytrees, so even with `n_valid`
   fixed, returning either from a `jax.vmap`-traced function raised `TypeError: ... is not a
   valid JAX type` — found by direct reproduction with a minimal dataclass, not mentioned in the
   original issue. Both are now registered via `jax.tree_util.register_pytree_node`, matching
   `BifProblem`'s existing registration.
   **Scope, deliberately narrow:** event detection (`events=[Fold()]`/`Hopf()`) still isn't
   traceable — `BifurcationDetector` is Python-level control flow (loops, `float()`,
   `list.sort()`) with no fixed-size/trace-safe rewrite yet — so `_run_scan_traced()` raises a
   clear `NotImplementedError` immediately if `events` is non-empty, rather than silently
   dropping them or crashing confusingly. Making event detection itself trace-safe is unchanged
   from before: still tracked under "Engineering recommendations for v0.2" item 4 (`Event`
   protocol rewrite) — this fix is forward-compatible groundwork for that (the `valid` mask is
   exactly what a trace-safe `Event` implementation would consume), not a competing design.
   See `tests/test_functional_api.py::TestVmapSafety` for the new coverage (fixed-size buffers +
   mask under `vmap`, the `NotImplementedError` on `events` under `vmap`, eager path unaffected).
   `tests/test_gpu_smoke.py`'s `vmap` test still exercises `pseudo_arclength_scan` directly (that
   test predates this fix and is still valid; a follow-up could switch it to `jc.continuation()`
   now that the public API supports the same pattern).
14. ✅ **Two latent test flakes from the issue #9 pattern, found and fixed 2026-07-19 while running
   the suite on a real GPU backend (not `JAX_PLATFORMS=cpu`).** `tests/test_pseudo_arclength.py`
   had five separate `PseudoArclengthContinuation(newton_tol=1e-8, ...)` instantiations — below
   the float32 epsilon floor issue #9 already documents. On CPU these happened to still pass
   (CPU/GPU XLA reductions aren't bit-identical, so which side of the epsilon floor a residual
   lands on isn't backend-invariant); on GPU, `test_quadratic_system` failed outright
   (`assert step >= 1` — zero steps because Newton never reported convergence). Fixed by raising
   all five to `newton_tol=1e-6` (now consistent with the shipped examples). That fix then
   uncovered a *second*, independent bug: `test_different_step_sizes`'s
   `assert param_range < 0.5` was only ever passing because the sub-epsilon tolerance was
   truncating some runs early; for this test's actual linear system (`rhs = r - x`, tangent
   `(1,1)/√2`), 5 fully-converged pseudo-arclength steps at `ds ∈ {0.05, 0.1, 0.2}` genuinely
   produce a parameter range of `≈0.53` — the bound itself was simply tighter than the correct
   converged answer. Loosened to `< 0.6` with the derivation left as a comment. Both fixes verified
   stable across 4+ repeated runs on the real GPU backend (previously: reliably failing).

---

## v0.1.0 — "Equilibria, done well" (target: next release)

Public surface is the functional API — `bif_problem` / `continuation` / `Fold`/`Hopf` — per
[ARCHITECTURE.md](ARCHITECTURE.md). The OO classes remain as a deprecated internal shim.

**Core — done:**
- [x] Natural + pseudo-arclength equilibrium continuation
- [x] Fold + Hopf detection with refinement (legacy detector; port to `Event` protocol)
- [x] Stability along the branch
- [x] Bifurcation-diagram plotting
- [x] Examples: 7 curated gallery scripts (pitchfork, Lorenz-84, Van der Pol, natural-vs-
      pseudo-arclength, neural-mass, `vmap` sweep, differentiable fold). Pitchfork, Lorenz-84,
      Van der Pol, and neural-mass are cross-validated against BifurcationKit.jl v0.5.2
      (independent Julia runs, offline); the rest are self-verified against closed-form theory.
      Consolidated from 9 files: dropped one redundant plotting demo and merged two overlapping
      manual-stepping tutorials into one that actually demonstrates the fold-passing contrast.
- [x] Autodiff `df/dp` (issue #2)
- [x] Robust bordered solve — no singular-`df/du` inversion (issue #1)
- [x] Functional spine: `BifProblem` + `continuation()` over the loop ([api.py](../src/jaxcont/api.py))
- [x] Whole-loop `lax.while_loop` engine — validated (issue #3, [scan_continuation.py](../src/jaxcont/core/scan_continuation.py))

**Core — done (scan path):**
- [x] Scan engine wired behind `continuation()` and made the default
- [x] Fold/Hopf events detected and refined on scan results
- [x] Stability computed by the vectorized `branch_eigenvalues` post-pass

**JAX differentiators — the reason to exist (ARCHITECTURE §3); must ship as first-class:**
- [x] `vmap` parameter-sweep example — [example_06_vmap_sweep.py](../examples/example_06_vmap_sweep.py)
      (256 diagrams, one kernel, **163× vs a Python loop**)
- [x] Differentiable-bifurcation example — [example_07_differentiable.py](../examples/example_07_differentiable.py)
      (reverse-mode `jax.grad` inverse design on a differentiable equilibrium; forward-mode
      `jacfwd` through the engine). Both cross-checked vs finite differences.
- [x] Reverse-mode `jax.grad` of a fold location — [fold_solve.py](../src/jaxcont/bifurcations/fold_solve.py)
      (`jc.fold_parameter`/`fold_point`: extended system + `custom_vjp` implicit diff; exact to
      analytic incl. vector-θ Jacobians). Hopf/codim-2 extended-system solvers are follow-ups.
- [x] These are the headline of the README and docs quickstart

**Release engineering:**
- [x] Core modules >85% coverage (on the engine path) — **done 2026-07-19.** Was already true for
  `scan_continuation.py`/`newton.py`/`fold_solve.py` (100%), `pseudo_arclength.py` (97%), `api.py`
  (95%), `hopf.py` (93%), `fold.py` (91%), `detector.py` (87%). The one gap the roadmap named
  explicitly, `stability/eigenvalue.py`, was raised **51% → 98%** by adding tests for the unstable-
  node/unstable-focus/center classification branches and for
  `compute_eigenvalues_along_branch`/`compute_stability_along_branch` (previously entirely
  untested) — see `tests/test_stability.py`. Box checked.
- [x] GPU smoke test — **done 2026-07-19.** [tests/test_gpu_smoke.py](../tests/test_gpu_smoke.py)
  (marked `gpu`, excluded from the default run via `pyproject.toml`'s `addopts`, run explicitly
  with `pytest -m gpu`) asserts a GPU device is present and usable, then runs a real
  `jc.continuation()` call and a `vmap`-batched sweep on it — passing on this project's own dev
  GPU (RTX A5000). Writing this test is also what surfaced issue #13 (`jc.continuation()` isn't
  actually `vmap`-safe) and issue #14 (two latent test flakes) — real value beyond "checks a box".
  No GPU runner exists in CI yet (`.github/workflows/tests.yml` is `ubuntu-latest` CPU-only), so
  this only runs when someone with GPU hardware runs `pytest -m gpu` manually; a CI job is a
  follow-up, not a blocker, now that the test itself is real and passing.
- [x] Honest README led by the vmap/grad story + stated scope + fixed placeholders
- [x] Sphinx docs: install, quickstart, Sphinx-Gallery examples, API reference
- [x] Clean sdist/wheel build + Twine metadata validation
- [x] TestPyPI → PyPI — **done 2026-07-21.** Tagged `v0.1.0`, published to PyPI
  (https://pypi.org/project/jaxcont/) via `publish.yml`.
- [x] GitHub release — **confirmed 2026-08-05**, this line was stale: a release exists for both
  `v0.1.0` and `v0.2.0` (`gh release list` shows both, the latter titled `v0.2`), just never
  checked off here.
- [ ] Zenodo DOI — **deliberately deferred**, by decision, until a more mature release with more
  results; not a v0.1.0 blocker. `CITATION.cff` metadata is ready. As of v0.3.0 (2026-08-05) this
  is no longer deferred by choice — repo-side prep is done, blocked only on enabling the
  GitHub-Zenodo integration (an account-level action) — see the header note above.

**Out of scope (hidden / marked experimental):** periodic orbits, Floquet, BVP,
normal forms, codim-2, branch switching, two-parameter continuation.

### Remaining v0.1.0 work, concretely (updated 2026-07-19 — engineering items now done)

The two engineering items originally listed here are **done** (see "Release engineering" above):
`stability/eigenvalue.py` coverage 51%→98%, and a real, passing `tests/test_gpu_smoke.py`. Along
the way, that work also fixed two latent test flakes (issue #14) and surfaced one important new
finding, issue #13 (`jc.continuation()` wasn't actually `vmap`-safe — **now fixed 2026-07-21**,
see issue #13 above; event detection under `vmap` remains future work, tracked separately under
v0.2 engineering recommendation #4).

**v0.1.0 is published.** Remaining loose ends, non-blocking:

1. ✅ GitHub release confirmed to exist for `v0.1.0` (2026-08-05).
2. Zenodo DOI archival — no longer deferred by choice as of v0.3.0; blocked only on enabling the
   GitHub-Zenodo integration (see the header note at the top of this file).

Issues #10 (legacy natural-continuation FD/bare-except) and #8/#9 (bothside, sub-epsilon tol) are
real but non-blocking for v0.1.0 — they don't affect the default `scan`/`PseudoArclength` path.
Fix opportunistically or fold into the v0.2 engine consolidation (see below).

## Visualization module consolidation (done 2026-07-22)

Found while extending `example_01`'s plot labels: `plot_continuation()` (in the then-
`jaxcont/utils/plotting.py`) only ever plots one state variable vs. the parameter, on one axis,
with plain fold/Hopf markers. Two examples worked around this by hand-rolling their own plotting
instead of extending the shared function: `example_02_lorenz.py` has a 40-line
`plot_lorenz84_diagram()` for annotated (text-box+arrow) bifurcation labels, and
`example_05_neural_mass.py` has a 20-line manual per-state-variable subplot loop. Both duplicate
(and drift from) the marker/color styling already in `plot_continuation`.

- [x] Move `plot_continuation`/`plot_bifurcation_diagram`/`plot_phase_portrait`/`plot_eigenvalues`
  into a new `jaxcont/viz/` subpackage (`core.py`/`styles.py`/`portraits.py`), delete
  `jaxcont/utils/plotting.py` outright (matches this project's "remove, don't deprecate" pre-1.0
  practice — see the engine-consolidation entry above). Top-level `jc.plot_continuation`/
  `jc.plot_bifurcation_diagram` names are unaffected.
- [x] Add a single shared `BIFURCATION_STYLES` table (`viz/styles.py`), replacing the three
  independently-hardcoded marker/color dicts in `plotting.py`, `example_02`, and `example_05`.
- [x] Add `annotate: bool = False` to `plot_continuation` (the `example_02` text-box+arrow style,
  opt-in — existing plots unaffected by default) and a new `plot_all_states()` (the `example_05`
  multi-panel style), both consuming the shared style table.
- [x] Migrate `example_02`/`example_05` onto the shared functions; `example_03` gets an import-path
  update only.
- [x] Add `tests/test_viz.py` — closes part of the "Plotting ... Under-tested" gap above (currently
  zero dedicated plotting tests exist). 19 new tests, full suite 95 passed / 15 deselected, 0 failed.
- [x] Update this table's "Plotting" row once done.

**Found along the way (2026-07-22):**
- While moving `plot_phase_portrait` into `viz/portraits.py`: the original function had no `ax`
  parameter, so `example_03_van_der_pol.py`'s `ax=ax2` call silently dropped into an unused
  `**kwargs`, and the function always built its own standalone figure — meaning the script's
  intended two-panel image (bifurcation diagram + phase portrait) never actually saved correctly;
  only the phase-portrait panel did, because `plt.savefig()` grabs the most-recently-created
  figure. Fixed alongside this consolidation — `viz/portraits.py` now has a real `ax` parameter.
- Two further fixes landed as direct controller edits mid-plan (not formal plan tasks, so the
  checkboxes above don't cover them):
  - Between the shared-styles task and the example migrations, `BIFURCATION_STYLES`' fold/Hopf
    labels were changed from the full words `"Fold"`/`"Hopf"` to
    [`bifurcations/taxonomy.py`](../src/jaxcont/bifurcations/taxonomy.py)'s standard abbreviations
    `"LP"`/`"H"`, matching the `"PD"`/`"BP"` entries that were already correct
    (`597a762`, "fix: use taxonomy.py's LP/H acronyms for fold/hopf bifurcation labels") — a
    user-requested consistency fix, not something the plan called for. Every bifurcation-diagram
    legend now reads "LP"/"H" rather than "Fold"/"Hopf".
  - The plan assumed `example_01_pitchfork.py` needed no import change (on the theory that it only
    used the top-level `jc.plot_continuation`), but it actually had its own direct
    `from jaxcont.utils.plotting import plot_continuation` import, which broke as soon as
    `jaxcont/utils/plotting.py` was deleted. Fixed and re-verified by running the example
    (`82acfae`, "fix: repoint example_01_pitchfork's plot_continuation import to jaxcont.viz").

Verification (final sweep, 2026-07-22): full suite 95 passed / 15 deselected (`slow`), 0 failed;
`example_01`/`02`/`03`/`05`/`06` all re-run headless (`MPLBACKEND=Agg`), all exit 0, and
`example_02`/`05`'s BifurcationKit.jl comparison tables still match as before (data/detection
unchanged — only presentation changed); grep sweep for `jaxcont.utils.plotting` across
`src/`/`tests/`/`examples/`/`docs/source/` (excluding Sphinx-Gallery's `auto_examples/`) is empty.

Design spec: [docs/superpowers/specs/2026-07-22-viz-module-design.md](../docs/superpowers/specs/2026-07-22-viz-module-design.md).
Implementation plan: [docs/superpowers/plans/2026-07-22-viz-module.md](../docs/superpowers/plans/2026-07-22-viz-module.md).

## v0.2.0 — Periodic orbits
- [x] Periodic-orbit continuation (collocation preferred over shooting) — *(done 2026-07-24, see
      [plan](../docs/superpowers/plans/2026-07-24-periodic-orbit-collocation.md) and its
      [design spec](../docs/superpowers/specs/2026-07-24-periodic-orbit-collocation-design.md))*.
      Fixed-mesh Gauss-Legendre orthogonal collocation, reusing the equilibrium scan engine
      completely unchanged — a periodic orbit's collocation system (mesh/collocation-point states +
      period + phase condition) is just a large residual `F(U, p) = 0`, exactly what
      `pseudo_arclength_scan`/`natural_scan` already solve generically. `periodic_orbit_problem(...)`
      is a pure factory: resamples a caller-supplied trajectory guess onto the mesh, refines to
      convergence via `differentiable_root`, returns an ordinary `BifProblem`. One new guard clause
      in `api.py` (`compute_stability=True` raises for periodic problems — that pass would
      eigendecompose the entire collocation Jacobian, not a meaningful quantity; Floquet
      multipliers are next). Verified against a closed-form exact answer (not just design
      reasoning): `r' = r(ρ-r²), θ'=1` has an exact circular limit cycle at ρ=1, and both the
      initial refinement and full continuation reproduce it to float32-achievable precision.
      Found and fixed two real numerical bugs mid-implementation, both via independent
      re-verification refusing to trust an initially-wrong "fix": (1) the plan's original
      verification numbers were contaminated by a stray `jax_enable_x64=True` left in a prototype
      script; (2) once genuinely re-verified under this machine's real float32/GPU, `jnp.einsum`'s
      default reduced (TensorFloat32) matmul precision on GPU turned out to corrupt the collocation
      Jacobian badly enough to stall Newton convergence — fixed with
      `jax.default_matmul_precision("float32")` around the residual's two einsum calls, plus a
      recalibrated `newton_tol=1e-5` for periodic continuation (the default `1e-6` is tighter than
      the achievable float32 floor and silently stalls continuation otherwise, no error raised).
      Fixed mesh only — no adaptive mesh redistribution (would break the fixed-shape-buffer
      `jit`/`vmap` discipline the whole scan engine relies on); explicitly deferred, not an
      oversight.
- [x] Floquet multipliers from monodromy matrix — *(done 2026-07-24, see
      [plan](../docs/superpowers/plans/2026-07-24-floquet-multipliers.md) and its
      [design spec](../docs/superpowers/specs/2026-07-24-floquet-multipliers-design.md))*. The
      guard clause above is gone: `settings.compute_stability=True` now works for periodic
      problems. `Φ(T)` (the monodromy matrix) is built as a block linear recursion across the
      collocation mesh's `ntst` intervals — reusing the existing Lagrange differentiation matrix
      `D` and the raw right-hand side's `df/du` Jacobian at each collocation point — not by
      re-integrating a separate variational-equation IVP (the old pre-v0.1 `scipy.integrate.
      solve_ivp`-based stub in `stability/floquet.py` was architecturally incompatible with the
      collocation representation and is now deleted). Floquet multipliers are `Φ(T)`'s
      eigenvalues, dispatched into `Branch.eigenvalues`/`Branch.stable` the same way equilibrium
      stability already was — `Branch.stable` uses the periodic-orbit-appropriate magnitude
      condition (all *non-trivial* multipliers inside the unit circle; the trivial multiplier,
      always exactly `1`, is identified as `argmin(|multiplier - 1|)` and excluded), not
      equilibria's real-part condition. Verified against a closed-form exact answer: `r' =
      r(ρ-r²), θ'=1` has exact Floquet multipliers `{1, exp(-4πρ)}`; both a plain-NumPy and a
      JAX/`jit` prototype matched it to float32-achievable precision before the plan was written.
      One notable negative finding (checked, not assumed): unlike the periodic-orbit residual's
      big einsum contraction, this recursion's small per-interval linear solves needed **no**
      `jax.default_matmul_precision("float32")` fix for the same GPU TensorFloat32 issue found in
      the collocation sub-project.
- [x] Period-doubling detection — *(done 2026-07-24, see
      [plan](../docs/superpowers/plans/2026-07-24-period-doubling-neimark-sacker.md) and its
      [design spec](../docs/superpowers/specs/2026-07-24-period-doubling-neimark-sacker-design.md))*.
      Shipped alongside Neimark–Sacker detection (not in the checklist item's literal name, but the
      Floquet design spec's own scope-cut section grouped them as "the natural next `Event`
      implementations" and the added verification cost turned out to be nearly free — see below).
      `PeriodDoubling`/`NeimarkSacker` are two new `Event` implementations in
      `bifurcations/events.py`, alongside `Fold`/`Hopf`, using the existing `Event` protocol/
      `detect_events` machinery unchanged. Both consume `BranchPoint.eigenvalues` (Floquet
      multipliers, for periodic branches) exactly the way `Hopf` consumes equilibrium eigenvalues,
      excluding the trivial multiplier via the same `argmin(|multiplier-1|)` rule
      `stability.floquet.floquet_stable` uses, then picking the real (`PeriodDoubling`) or complex
      (`NeimarkSacker`) candidate closest to `-1`/the unit circle. Deleted the dead, pre-`Event`-
      protocol `bifurcations/period_doubling.py` stub. Each event carries its own `raw_f`/`mesh`
      fields (mirroring `Hopf`'s `tolerance` field) so `refine()` can call
      `stability.floquet.floquet_multipliers` directly — `detect_events`'s generic `rhs` parameter is
      the assembled collocation residual, not the raw ODE, so reusing it would repeat `Hopf`'s
      equilibrium-only footgun in reverse.
      Mathematical grounding: for a **2D** autonomous periodic orbit the single non-trivial Floquet
      multiplier is `exp(∫div(f)dt)`, always positive real — period-doubling (`-1` crossing) and
      Neimark–Sacker (complex-pair unit-circle crossing) are only possible for **3+**-dimensional
      systems, so a new verification system was needed (the existing 2D circle system can't exhibit
      either). Built one: the verified circle system plus a decoupled linear "transverse" block
      (`w1'=αw1-βw2, w2'=βw1+αw2`), whose exact monodromy contribution is the closed-form
      `exp((α±iβ)T)` — `β=π/T` gives a real multiplier `=-exp(αT)` crossing `-1` at `α=0` (PD ground
      truth); any other `β` gives a genuine complex pair whose magnitude crosses `1` at `α=0` (NS
      ground truth). `w≡0` is an exact periodic solution for any `α`/`β`, so the test fixture needed
      no new simulation.
      Found and fixed a real bug via end-to-end verification (not just the closed-form math): the
      naive "closest multiplier to -1/unit-circle" `argmin` selection could silently switch which
      *physical* multiplier it tracks once the true one moved far enough away, latching onto an
      unrelated, always-far multiplier instead and producing a false-positive detection. Fixed with a
      `near_unit_circle` pre-filter (`|magnitude-1| < 0.5`, a new field on both classes) excluding
      candidates that were never near the unit circle to begin with — required, not optional; an
      implementer's attempt to weaken it to `<=` during execution was reverted (the real bug was in
      one of the plan's own hand-built test cases landing exactly on the filter boundary, not the
      filter itself).
- [x] Limit-cycle examples (Van der Pol, Brusselator) — *(done 2026-07-24, see
      [plan](../docs/superpowers/plans/2026-07-24-limit-cycle-examples.md) and its
      [design spec](../docs/superpowers/specs/2026-07-24-limit-cycle-examples-design.md))*. This
      closes the v0.2.0 "Periodic orbits" epic. Two new example scripts
      (`examples/example_10_van_der_pol_limit_cycle.py`,
      `examples/example_11_brusselator_limit_cycle.py`) demonstrate real limit-cycle continuation —
      not just an equilibrium's Hopf crossing — via `periodic_orbit_problem`. Since JaxCont doesn't
      integrate ODEs itself, each script simulates a short trajectory with `scipy.integrate.solve_ivp`
      (the user's own simulation, matching the architecture's intended usage pattern — the library
      still doesn't call it), extracts one period from the tail via `scipy.signal.find_peaks`, and
      hands `(u_trajectory, t_trajectory, period0)` to `periodic_orbit_problem` for collocation
      refinement before continuing. Also fixed `example_03_van_der_pol.py`'s stale docstring claim
      ("periodic-orbit continuation is outside JaxCont's current scope").
      Two real issues found via end-to-end verification (not assumed from design reasoning alone):
      (1) `t_trajectory` must be re-based to start at `0` before `periodic_orbit_problem` — its
      internal resampling computes `t = τ·period0` for `τ∈[0,1]`, so a raw (non-zero-based) time
      array falls outside `jnp.interp`'s domain and silently clamps to a constant, producing a
      degenerate `T≈0` "solution" with a deceptively small residual rather than an error; (2) the
      Brusselator's achievable float32 residual floor at the same mesh size (`ntst=20`) needed a
      looser `newton_tol=1e-4` than Van der Pol's `1e-5` — confirming (again) that this floor is
      system-specific, not something to assume transfers from a previously-verified value.
      Verified results: Van der Pol (`μ: 1→4`) shows near-constant amplitude (`≈2.0`) with period
      growing `6.66→10.25` — the relaxation-oscillator signature here is period growth and waveform
      sharpening, not amplitude growth (a known fact about this normalization); Brusselator
      (`a=1`, `b: 2.5→4.0`) shows amplitude growing `2.02→4.70`, a deliberately different pair of
      qualitative behaviors. Both stable throughout (`compute_stability=True`, asserted via `raise`,
      not a silent print).

## v0.3.0+ — Advanced (demand-driven)
- [x] Phase-plane visualization for 2D autonomous systems: nullclines, vector fields,
      streamlines, continuation equilibria, and trajectories.
- [ ] Branch switching
- [ ] Two-parameter continuation
- [x] Normal forms / Lyapunov **coefficient** `l₁` (Hopf criticality — a *bifurcation* invariant,
      NOT the Lyapunov exponent spectrum; see below) — *(done 2026-08-04, see
      [plan](../docs/superpowers/plans/2026-08-04-hopf-normal-form.md) and its
      [design spec](../docs/superpowers/specs/2026-08-04-hopf-normal-form-design.md))*. New
      `bifurcations/hopf_normal_form.py`: `hopf_point`/`hopf_parameter` (a differentiable
      extended-system Hopf-point solver, mirroring `fold_solve.py`'s `fold_point`/`fold_parameter`
      via the same `solvers/implicit.py:differentiable_root` primitive) plus `lyapunov_coefficient`
      (Kuznetsov's first-Lyapunov-coefficient formula, pure algebra via `jax.jvp` directional
      derivatives — never calls `jnp.linalg.eig` itself, so gradients compose through ordinary
      chain rule). All three exported top-level (`jc.hopf_point`/`jc.hopf_parameter`/
      `jc.lyapunov_coefficient`). `Hopf.refine()` (`bifurcations/events.py`) is rewired onto this:
      it used to bisect and return a linearly-interpolated (not actually converged) `u`; now it
      solves the real extended system and classifies every detected Hopf point's `EventHit.info`
      with `omega0`/`l1`/`criticality` (`"supercritical"`/`"subcritical"`/`"degenerate"`, plus a
      `"unknown"` fallback — see below). `example_02_lorenz.py`/`example_05_neural_mass.py`'s
      BifurcationKit.jl comparison tables now show criticality/`l1` alongside location.
      Verified against a closed-form textbook Hopf example (exact `l1`) and independently
      cross-checked against a real BifurcationKit.jl v0.5.2 run, not just design reasoning.
      Three real bugs found and fixed during final whole-branch review, before merge (not
      hypothetical — each had a concrete failure mode):
      (1) `_seed()`'s eigenvector pick used a bare `argmin |Re(eigenvalue)|` over *all* eigenvalues,
      so a slow real mode near the origin could out-rank the genuine Hopf pair and seed the solver
      with `omega=0` — now masks to `|Im| > 1e-8` first (matching `Hopf.test_function`'s own
      selection rule) and prefers `Im > 0` for a consistent `+iω` orientation;
      (2) `Hopf.refine()` classified criticality from the sign of `l1` with no finiteness check —
      a non-convergent `hopf_point` solve (a known occurrence when a detector bracket's sign change
      isn't a real Hopf point) can return non-finite `p`/`l1`/`omega0`, and `nan < 0` is `False` in
      Python/NumPy/JAX, so a failed solve silently mislabeled itself `"subcritical"`; fixed with an
      explicit `isfinite`+`omega0>0` guard reporting `"unknown"` instead;
      (3) no test previously verified `l1`'s sign actually drives the supercritical/subcritical
      label end-to-end through `Hopf().refine()` — the feature's headline output — added directly
      (`tests/test_bifurcations.py`), plus seed-masking and non-finite-guard coverage in
      `tests/test_hopf_normal_form.py`.
      `l1_tolerance` (default `1e-6`, the `"degenerate"` cutoff) is documented as scale-dependent —
      tune per-system, not a universal constant. `lyapunov_coefficient` requires `f` complex-
      analytic (holomorphic) in `u` near the Hopf point; documented, not runtime-checked (an
      `abs`/`clip`/active `jnp.where` RHS would silently produce a wrong `l1`).
      **Deliberately deferred, per the design spec's explicit scope cut:** the fold's own
      normal-form coefficient `a`, a `jc.normal_form(sol, event)` dispatcher, GH/codim-2 detection
      (the `GH` taxonomy entry this unblocks), branch switching, two-parameter continuation — each
      its own future roadmap item.
- [x] Codim-2 bifurcations: cusp, Bogdanov-Takens, generalized Hopf, zero-Hopf, double-Hopf —
      *(done 2026-08-05, see
      [plan](../docs/superpowers/plans/2026-08-05-codim2-direct-solvers.md) and its
      [design spec](../docs/superpowers/specs/2026-08-05-codim2-direct-solvers-design.md))*. New
      `bifurcations/codim2.py` (`cusp_point`/`cusp_parameters`,
      `bogdanov_takens_point`/`bogdanov_takens_parameters`,
      `generalized_hopf_point`/`generalized_hopf_parameters`,
      `zero_hopf_point`/`zero_hopf_parameters`, `double_hopf_point`/`double_hopf_parameters`) plus
      `bifurcations/fold_normal_form.py: fold_coefficient` (the fold's quadratic normal-form
      coefficient — needed by CP, also a standalone useful quantity). All eleven functions exported
      top-level (`jc.cusp_point`, etc.). Each solver is a square extended system `G(x,θ)=0` solved
      via the same `solvers/implicit.py:differentiable_root` implicit-function-theorem Newton
      primitive `fold_point`/`hopf_point` already use — no new engine. `p` is generalized from
      scalar to shape `(2,)` throughout this feature (codim-2 needs two free parameters); every
      other function is untouched. **Deliberate scope choice:** these are direct point *solves* —
      refine a point you can already guess at — not the `jc.codim2(prob, event=...)` two-parameter-
      *continuation* sketch in ARCHITECTURE.md (finding points you can't yet approximate), which
      remains its own, unstarted roadmap item below.
      Four real design findings, verified numerically during planning (not just reasoned about) and
      confirmed correct through implementation:
      (1) ω's sign is unconstrained — the Hopf block of every extended system (GH, ZH, HH) is
      exactly invariant under `(ω, q2) → (-ω, -q2)`, and Newton reliably converged to *negative* ω
      from positive seeds during planning; fixed with a post-solve `_normalize_omega` helper (flip
      both if ω<0) rather than an extra equation, which would break squareness — this matters
      because `events.py`'s `Hopf.refine()` already treats `omega0<=0` as a failed solve;
      (2) HH (double Hopf) is genuinely degenerate when both Hopf blocks get seeded onto the same
      physical pair — verified during planning to produce `nan`, not a plausible-looking wrong
      answer — so `double_hopf_point` requires a caller-supplied `seed_b` (keyword-only, no
      default, unlike every other solver here) and a post-solve pair-separation check
      (`abs(|omega_a|-|omega_b|) > separation_tolerance`, default `1e-3`) that reports
      `converged=False` on collapse rather than a bare `nan`;
      (3) BT's textbook left/right-null-vector formulation (`f=0, Jv=0, Jᵀw=0, |v|=1, |w|=1, w·v=0`)
      is overdetermined — `3n+3` equations for `3n+2` unknowns, confirmed by direct count — so the
      Jordan-chain formulation is used instead (`f=0, Jv0=0, Jv1=v0, |v0|=1, v0·v1=0`), which is
      exactly square;
      (4) origin-centered test systems have no discriminating power — every standard textbook normal
      form (cusp, BT, GH, ZH, HH) places its codim-2 point at `u=0, p=(0,0)`, so a stub that simply
      returns zeros would pass all of them; every test in `tests/test_codim2.py` instead uses an
      affinely *shifted* system with a non-trivial known answer (verified during planning: a shifted
      BT recovered `u*=(5,2), p*=(3,-1)` to `6.5e-13` with conditioning unchanged from the
      origin-centered version).
      One implementation-time finding, discovered independently three times by three different task
      implementers (Tasks 4, 5, 6) and each time correctly root-caused rather than worked around:
      the GH/ZH/HH residuals recompute a Hopf phase-condition seed on every Newton iteration via
      `_hopf_seed` (from `hopf_normal_form.py`), and that seed's `jnp.linalg.eig` call has no
      gradient rule for non-symmetric eigenvectors. Wrapping the seed's `args`/`theta` input in
      `jax.lax.stop_gradient(...)` (matching the pattern `hopf_normal_form.py`'s own
      `_extended_residual` already used) fixes it — each fix was verified against the actual
      gradient-vs-finite-difference test failing without it and passing with it, not assumed. Fixed
      in the shipped code for all three; the plan's own text was also corrected retroactively
      (commits `fb66b17`, `5213d4d`) so this doesn't recur if the plan is ever reused as a reference.
      **float32 tolerance calibration:** measured residual floors during planning: BT `0.0` (exact
      recovery), CP `≈4.4e-8`, GH `≈5.96e-08` (the tightest). All five solvers default to
      `tol=1e-6` — tight enough to be meaningful, loose enough to be achievable; `1e-8` would make
      `converged` report `False` forever even on an exact answer (issue #12's pattern, again);
      `1e-4` stops measurably early.
      **Independent cross-validation:** `examples/BifurcationKit/05_codim2.jl` runs
      BifurcationKit.jl v0.5.2's own codim-2 detection on the Lorenz-84 atmospheric model (already
      used elsewhere in this repo, `examples/BifurcationKit/02_lorenz84.jl` — not a system tuned to
      fit the answer) and finds a genuine BT point. `bogdanov_takens_point` reproduces it; the test
      (`tests/test_codim2.py`) asserts agreement against Julia's actual printed output, independently
      re-verified byte-for-byte during code review, and cross-corroborated against BifurcationKit's
      own internal test suite values (`nf.a`/`nf.b` in its `test/lorenz84.jl`, matching to 10
      significant figures). A review-time check deliberately injected a sign bug into the test's
      model copy and confirmed the test then fails to converge or lands measurably off — real
      discriminating power, not a tautology. An earlier attempt on a different applied model
      (Bazykin's predator-prey) found no codim-2 point in the parameter window tried and was
      correctly abandoned rather than forced.
      **Explicitly deferred, not forgotten (each its own future roadmap item):** GH-specific
      BifurcationKit cross-validation (only BT was cross-validated — the design spec's Step 3
      template was BT-only by scope, not an oversight); codim-2 normal-form coefficients beyond
      CP/GH's defining conditions (BT's own `(a,b)` pair, GH's second Lyapunov coefficient `l2`);
      `Event`/`events=[...]` integration (a codim-2 point cannot be detected along a
      single-parameter branch — it needs a two-parameter curve, which doesn't exist); branch
      switching; two-parameter continuation itself.
      Verified: full test suite green throughout implementation — 211 (baseline) → 239 passed by
      the end (28 new tests across all nine tasks, zero regressions, verified after every single
      task via `JAX_PLATFORMS=cpu pytest tests/ -n auto`).

> **Lyapunov exponents** (trajectory/chaos spectrum) are out of scope — they live in the sibling
> package **lyapax** (`~/git/lyapunov`). JaxCont interops via a thin `as_rhs(p)` bridge rather
> than reimplementing them. See [ARCHITECTURE.md §8](ARCHITECTURE.md).

## v0.4.0+ — Explicitly out of scope (won't do unless someone asks)

See "Strategic direction" below for the reasoning. Listed here so it's a deliberate decision, not
an oversight, and so a future contributor doesn't rediscover the same MatCont chapter and assume
it was simply forgotten:

- **Homoclinic/heteroclinic orbit continuation.** MatCont devotes ~10 of its ~120 pages and 4
  dedicated global structures (`homds`, `hetds`, invariant-subspace continuation) to this; it is
  its own subfield with its own toolbox lineage (HomCont, DDE-BIFTOOL-adjacent techniques).
- **Phase response curves (PRC/dPRC).** A MatCont specialty (§7.8 of the manual) with real
  neuroscience value, but a self-contained feature nobody has asked for here yet.
- **A GUI.** Actively contradicts this project's own stance (`ARCHITECTURE.md` §1.6: "mine MATCONT
  for its taxonomy, not its API") — JaxCont's whole value proposition is being embedded in a JAX
  script/notebook, not a standalone application.
- **Poincaré maps / general event-triggered integration.** Would be a real feature but is
  currently better served by composing `jax`/`diffrax` directly; revisit only if periodic-orbit
  work in v0.2 creates a natural implementation for free.

## Strategic direction beyond v0.1 (recommendation, 2026-07-19)

**Question this answers:** should JaxCont eventually match everything MatCont's manual covers, or
go further, or stay narrower? MatCont's manual is the closest thing to a canonical taxonomy of a
"complete" continuation/bifurcation toolbox (equilibria → limit cycles → codim-1 → codim-2 →
homoclinic/heteroclinic → PRC → GUI), so it's a fair yardstick.

**Recommendation: match MatCont's most-used subset, do not chase its full breadth, and spend the
difference on differentiability/`vmap`/GPU — the one axis MatCont cannot compete on at all.**

- **Match (v0.2, already scheduled above):** periodic-orbit continuation + Floquet multipliers +
  period-doubling/fold-of-cycles/Neimark–Sacker detection. This is the single biggest gap right
  now and the one most users of *any* continuation tool actually need — MatCont Ch. 7-8 spends the
  bulk of its pages here for the same reason. It's also architecturally in-scope: collocation with
  a fixed mesh is exactly the "fixed-shape buffers" discipline the scan engine already requires
  (§4.3 of ARCHITECTURE.md), so it composes with the existing whole-loop-JIT/`vmap` story instead
  of fighting it.
- **Demand-driven (v0.3, already scheduled above):** branch switching, two-parameter/codim-2
  continuation (cusp, Bogdanov–Takens, generalized Hopf), and *real* normal-form coefficients
  (the current `fold.compute_normal_form()`/`hopf.compute_first_lyapunov_coefficient()` are
  literal `return {"a": 0.0, ...}` / `return 0.0` placeholders — worth fixing even before v0.3
  proper, since they're currently silently wrong rather than absent). These matter to a
  minority of users but are moderate, well-understood effort once periodic orbits exist.
- **Don't chase (v0.4+, listed above):** homoclinic/heteroclinic orbits, PRC, GUI. MatCont's own
  page count shows how expensive these are relative to how many users need them; replicating them
  would take years and mostly duplicate an already-excellent, free, actively-maintained tool.
- **Go further than MatCont, on purpose, everywhere:** every new curve/event type added in v0.2/v0.3
  should ship with a `vmap`-batched example and, where the extended-system pattern applies (see
  `fold_solve.py`), a differentiable variant — because that combination (batched *and*
  differentiable bifurcation analysis) is the actual reason to prefer JaxCont over MatCont/
  BifurcationKit.jl for a given problem, not feature-count parity.

**Main tradeoff, stated plainly:** matching MatCont feature-for-feature would take years and
mostly reproduce a tool that already exists and is good; going narrow-but-differentiable means
JaxCont will *never* be a drop-in MatCont replacement for e.g. a homoclinic-bifurcation study, but
it becomes the only tool that can do gradient-based bifurcation design or GPU-batched sweeps of
thousands of parameter settings in one kernel — a different, smaller, but currently-unserved
niche.

## Engineering / architecture recommendations for v0.2 (2026-07-19)

`ARCHITECTURE.md` already specifies the target shape well (pluggable `LinearSolver`/`EigenSolver`,
the `Event` protocol, fixed-shape `Branch` buffers). These five items are what the *current* code
needs to actually get there cleanly, surfaced by re-reading the source while updating this file —
worth resolving before, not during, the v0.2 periodic-orbit push:

1. ✅ **Retire the three-implementations-of-one-algorithm pattern before adding a fourth (periodic
   orbits).** *(done 2026-07-22 — see
   [docs/superpowers/plans/2026-07-21-engine-consolidation.md](../docs/superpowers/plans/2026-07-21-engine-consolidation.md)
   and its [design spec](../docs/superpowers/specs/2026-07-21-engine-consolidation-design.md))*
   `natural_continuation.py`, `pseudo_arclength.py`'s legacy OO class, and `PredictorCorrector`
   deleted outright (not deprecated — already-published v0.1.0 API, removed anyway per explicit
   decision, since it's a pre-1.0 project). `equilibrium_continuation()`/`periodic_continuation()`
   free functions removed too. `Natural`/`PseudoArclength` in `api.py` are now thin config objects
   dispatching to `core/scan_continuation.py`'s two engines (`pseudo_arclength_scan`, and a new
   `natural_scan()` built for this — same fixed-buffer/`ds`-tracking/jit/`vmap` contract, predictor
   swapped). All 32 dependent tests (4 files) and all 6 non-`example_06`/`07` gallery examples
   migrated onto `jc.continuation()` or the engines' private `_tangent`/`_newton_correct` directly.
   Full suite green (75 passed + 12 `slow`-marked, 0 failures); all cross-validated BifurcationKit.jl
   matches in `example_02`/`05` still hold.
   **Found along the way, worth knowing:** `jc.continuation()`'s `p_span[0]` is the *literal*
   starting parameter value (paired with `u0` directly) — not `problem.p0` — a pre-existing `api.py`
   design point (predates this cleanup) that differs from the deleted OO engine's semantics (which
   started at `problem.p0` and used its range argument only for direction/stop-bound). Every
   migrated example needing a `p_span` fix for this is now commented in-file explaining why.
   **Not fixed here, still open:** `docs/source/quickstart.rst` still shows the removed
   `PseudoArclength(engine=...)` kwarg; `src/jaxcont/utils/config.py`'s `test_package_imports()`
   still lists the 3 deleted module paths (caught by `except ImportError`, non-breaking, just
   stale). `natural_scan`/`pseudo_arclength_scan` duplicate most of their `lax.while_loop` body
   (predict/correct/write/adapt/stop) — a candidate for a shared helper once a third predictor
   (periodic-orbit collocation) actually needs it, not before (YAGNI).
2. ✅ **Resolve the `eqx.Module` "open decision" (ARCHITECTURE.md §4, line ~170) now, before
   periodic orbits land.** *(done 2026-07-22 — see
   [docs/superpowers/plans/2026-07-22-equinox-adoption.md](../docs/superpowers/plans/2026-07-22-equinox-adoption.md)
   and its [design spec](../docs/superpowers/specs/2026-07-22-equinox-adoption-design.md))*
   `equinox` is now a runtime dependency; a throwaway `CollocationMeshScaffold`
   (`core/_periodic_eqx_scaffold.py`, not exported from `jaxcont.__init__`) proves the
   static-vs-traced field split (`eqx.field(static=True)` for `ntst`/`ncol`, traced `mesh` array)
   works end-to-end under `jit` and `vmap` — see `tests/test_equinox_scaffold.py`. The real
   periodic-orbit types (`Collocation` predictor, `PeriodDoubling`/`LPC`/`NS` events) are still not
   built; this only removes the open decision and gives that future work a proven pattern. v0.1's
   `BifProblem`/`Branch` are untouched, per the original recommendation.
3. ✅ **Generalize the `fold_solve.py` pattern into one reusable primitive before hand-writing it
   again for Hopf/LPC/PD/NS.** *(done 2026-07-23 — see
   [docs/superpowers/plans/2026-07-23-differentiable-root-primitive.md](../docs/superpowers/plans/2026-07-23-differentiable-root-primitive.md)
   and its [design spec](../docs/superpowers/specs/2026-07-23-differentiable-root-primitive-design.md))*
   The genuinely novel piece of this project — Newton-in-`lax.while_loop` over an extended system
   `G(x,θ)=0`, wrapped in `jax.custom_vjp` implementing the implicit function theorem so `jax.grad`
   skips the iteration — was bespoke to folds; it's now `solvers/implicit.py: differentiable_root(G,
   x0, theta) -> x*`, and `fold_solve.py` delegates to it. Each new differentiable event (Hopf, then
   LPC/PD/NS in v0.2/v0.3) can now be just a new `G`, not a new `custom_vjp` implementation.
   **Found along the way, worth knowing:** `x0` (the Newton seed) must be passed as a callable
   `theta -> Array`, not a precomputed `Array`, whenever the seed genuinely depends on `theta` (as
   fold's SVD-based null-vector guess does) — a precomputed theta-dependent seed closed over from
   outside leaks a JAX tracer under `jax.grad`. `differentiable_root` supports both forms; the
   callable form resolves the seed inside the traced primal instead.
4. ✅ **Replace `BifurcationDetector` with the sketched `Event` protocol (ARCHITECTURE.md §4.7) as
   part of fixing issue #7 (duplicate/spurious fold-vs-Hopf flags), not instead of it.** *(done
   2026-07-23 — see
   [docs/superpowers/plans/2026-07-23-event-protocol-rewrite.md](../docs/superpowers/plans/2026-07-23-event-protocol-rewrite.md)
   and its [design spec](../docs/superpowers/specs/2026-07-23-event-protocol-rewrite-design.md))*
   `BifurcationDetector`/`FoldBifurcation`/`HopfBifurcation` are gone, replaced by small,
   independently-testable `Event` implementations (`Fold`, `Hopf`) in
   `bifurcations/events.py`. Root-caused issue #7 to `Fold`'s eigenvalue-based test function
   picking up a Hopf pair's real part; fixed by switching `Fold` to the pseudo-arclength tangent's
   `dp` sign change (no eigenvalues at all), plus a same-kind-only dedup pass and a `nan` (not
   `inf`) "no complex eigenvalues" sentinel — both found and fixed during design by actually
   running the change against the two real BifurcationKit.jl-cross-validated examples, not just
   reasoning about it. Verified end to end: `example_02_lorenz.py` goes from 6 raw detections
   (with duplicates) to exactly 4 clean ones; `example_05_neural_mass.py` from 5 to exactly 3;
   zero spurious/unmatched rows in either. Still eager-only — trace-safe (`vmap`/`jit`) event
   detection remains unimplemented and is now its own future item, not bundled into this one.
5. ✅ **Make `LinearSolver`/`EigenSolver` (ARCHITECTURE.md §4.6) real protocols with a `Dense()`
   implementation now, even though nothing else exists yet.** *(done 2026-07-24 — see
   [plan](../docs/superpowers/plans/2026-07-23-linear-eigen-solver-protocols.md) and its
   [design spec](../docs/superpowers/specs/2026-07-23-linear-eigen-solver-protocols-design.md))*
   Every hardcoded `jnp.linalg.solve`/`jnp.linalg.eigvals` call in the live jitted scan engine
   (`core/scan_continuation.py`) now routes through a `LinearSolver`/`EigenSolver` parameter,
   defaulting to `Dense()`/`DenseEigen()` — two zero-field frozen dataclasses, exposed on
   `continuation()` via a new `Solvers` bundle. `Dense`/`DenseEigen` (two classes, not one
   `Dense()` reused for both, which the design spec found isn't viable — different `__call__`
   signatures) are safe `jax.jit` static arguments by construction (value-based `__eq__`/`__hash__`
   from having no fields), verified with a test that calls the engine twice with two
   independently-constructed `Dense()` instances and confirms no spurious recompile. Custom-solver
   routing was proven real, not decorative, with counting-solver test doubles at both the
   low-level scan-function layer and the public `continuation()` layer. Numerically a no-op:
   full suite green (134 passed) and `example_02`/`example_05`'s BifurcationKit.jl comparison
   tables unchanged. Scoped to the live path only — `solvers/implicit.py` (fold refinement) and
   the not-yet-wired `stability/`/`bifurcations/period_doubling.py` periodic-orbit code were left
   untouched, per the design spec's explicit scope cut. This was the last unresolved v0.2
   engineering-prep item; periodic-orbit feature work (collocation, Floquet multipliers,
   period-doubling detection) is next.

---

## Do this next (in order)

1. ✅ **Tidy notes** — done: this roadmap + archive.
2. ✅ **JIT the Newton loop** and re-profile — done. Key learning: JIT-ing the corrector alone
   gives ~no speedup at small sizes; the real win requires whole-loop JIT / `vmap`. See issue #3
   and [ARCHITECTURE.md §2](ARCHITECTURE.md).
3. ✅ **Fix the two correctness bugs** (issues #1, #2) — done (bordered solve + autodiff `df/dp`).
4. ✅ **Commit the API design** — done: functional `continuation(...)` surface, see
   [ARCHITECTURE.md](ARCHITECTURE.md).
5. ✅ **Implement the functional spine** — done: `BifProblem` + `continuation()` over the loop
   ([api.py](../src/jaxcont/api.py)); OO classes kept as internal shim.
6. ✅ **Whole-loop engine** — done & validated: [scan_continuation.py](../src/jaxcont/core/scan_continuation.py)
   (~340× warmed, vmap-batches, no hang). Proves the performance/vmap/grad thesis.
7. ✅ **Wire the engine into `continuation()`** — done: `PseudoArclength(engine="scan")` is the
   default; detection, refinement, and vectorized stability are reused ([api.py](../src/jaxcont/api.py)).
8. ✅ **Ship the differentiators as examples** — done: [example_06](../examples/example_06_vmap_sweep.py)
   (`vmap`, 163×) + [example_07](../examples/example_07_differentiable.py) (`grad` of a fold via
   [fold_solve.py](../src/jaxcont/bifurcations/fold_solve.py), + forward-mode `jacfwd`).
9. ✅ **Trim `__init__.py`** — done: top-level surface is the equilibrium spine; periodic/BVP/
   Floquet/period-doubling stubs are importable only from their submodules.
10. ✅ **Docs + packaging → ship v0.1.0.** — **done 2026-07-21.** Tagged and published to PyPI
    (https://pypi.org/project/jaxcont/). GitHub release confirmed to exist (2026-08-05); Zenodo DOI
    no longer deferred by choice as of v0.3.0, blocked only on the GitHub-Zenodo integration.
11. **v0.2 kickoff — do the engineering cleanup *before* the periodic-orbit feature work**, per
    "Engineering / architecture recommendations for v0.2" above, in this order: (i) consolidate
    the three continuation-engine implementations onto the scan engine (`jc.continuation()`'s
    branch/stability data is already `vmap`-safe as of issue #13's 2026-07-21 fix — this item is
    now about removing the duplication itself, not vmap-safety); (ii) decide `equinox` for the new
    periodic-orbit types; (iii) extract `fold_solve.py`'s differentiable-root pattern into a
    reusable primitive; (iv) replace `BifurcationDetector` with real `Event` implementations,
    fixing issue #7 **and making event detection itself `vmap`-safe** as part of the rewrite
    (`Branch.valid` from issue #13 is the mask a trace-safe `Event` would consume); (v) introduce
    `LinearSolver`/`EigenSolver` as real (if currently single-implementation) protocols. Then
    build periodic-orbit collocation with a static
    (non-traced) `ntst`/`ncol` mesh on top of the cleaned-up spine, matching MatCont's own
    `ntst`/`ncol` discretization discipline (manual §7.2) and the fixed-shape-buffer requirement
    the whole-loop-JIT/`vmap` story already depends on (ARCHITECTURE.md §3.1, §4.3).
12. ✅ **v0.2.0 feature work** — done 2026-07-24: periodic-orbit collocation, Floquet multipliers,
    period-doubling/Neimark–Sacker detection, limit-cycle examples. Tagged and published to PyPI.
13. ✅ **Phase-plane visualization** (v0.3.0) — nullclines, vector fields, streamlines,
    equilibria, trajectories for 2D autonomous systems. See
    [plan](../docs/superpowers/plans/2026-07-28-phase-plane-visualization.md).
14. ✅ **Hopf normal form / `l₁` criticality** (v0.3.0, done 2026-08-04) — see the v0.3.0+ section
    above for the full writeup.
15. ✅ **Codim-2 direct point solvers** (v0.3.0, done 2026-08-05) — cusp, Bogdanov-Takens,
    generalized Hopf, zero-Hopf, double-Hopf, plus the fold's own normal-form coefficient
    `fold_coefficient` — see the v0.3.0+ section above for the full writeup. This closes the
    "Codim-2 groundwork" item this list previously pointed at next.
16. ✅ **v0.3.0 release cut** (2026-08-05) — version bumped, `CHANGELOG.md`/`CITATION.cff`/README
    updated, tagged `v0.3.0`, GitHub release created. PyPI publish not yet done as a separate step
    (see below). Zenodo DOI still pending the GitHub-Zenodo integration being enabled — see the
    header note at the top of this file.
17. **Next up** — no single item is blocking; pick by what's wanted:
    - **Ergonomics:** `bothside` continuation (issue #8) and the legacy `natural_continuation.py`
      FD/bare-except cleanup (issue #10) are still open, low-severity, non-blocking items.
    - **PyPI publish** for v0.3.0 via the existing `publish.yml` workflow (matches v0.1.0/v0.2.0's
      release process) — a separate, independent step from the GitHub release/Zenodo archive.
    - **Larger v0.3.0+ epics** (bigger, less demand-driven urgency so far): branch switching,
      two-parameter continuation (which the codim-2 *direct solvers* just shipped are explicitly
      not a substitute for — see the v0.3.0+ writeup above).

---

## Reference / learning
- Kuznetsov, *Elements of Applied Bifurcation Theory*
- BifurcationKit.jl · MATCONT · AUTO-07p · PyDSTool
- JAX docs: JIT, vmap, `lax.while_loop`, `lax.scan`
