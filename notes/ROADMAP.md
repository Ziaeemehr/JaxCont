# JaxCont Roadmap — Single Source of Truth

**Last updated:** 2026-07-21
**Current version:** 0.1.0 — **published to PyPI** (https://pypi.org/project/jaxcont/), tagged
`v0.1.0`. Zenodo DOI archival deliberately deferred until a more mature release (not a v0.1.0
blocker — `CITATION.cff` metadata is ready whenever it happens).
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
| Fold + Hopf detection | ✅ Works, tested | With bisection refinement |
| Stability (eigenvalues) | ✅ Works, 98% cov | Fixed 2026-07-19 (was 51%) |
| Naming/abbreviation reference | ✅ Works, tested | [bifurcations/taxonomy.py](../src/jaxcont/bifurcations/taxonomy.py) |
| Plotting | ⚠️ Works, 9% cov | Under-tested |
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
13. 🔴 **`jc.continuation()` (the public, "blessed" API) is not `vmap`-safe — only the lower-level
   `pseudo_arclength_scan` engine it wraps is.** *(found 2026-07-19, while writing a GPU `vmap`
   smoke test)* `api.py`'s `_run_scan()` does `n = int(res.n_valid)` to trim the fixed-size
   buffer to a Python-level ragged length before building the legacy `ContinuationSolution`. That
   bare `int()` on a traced value raises `jax.errors.ConcretizationTypeError` the moment
   `jc.continuation(...)` is called inside `jax.vmap(...)` — confirmed by direct reproduction (a
   plain call succeeds; wrapping the identical call in `jax.vmap` fails on that exact line). This
   is why `examples/example_06_vmap_sweep.py` calls `pseudo_arclength_scan` directly instead of
   `jc.continuation()` — it has to, silently, and doesn't say why. **This matters more than a
   normal bug**: `vmap`-batched continuation is the flagship capability (ARCHITECTURE.md §3.1,
   the README's headline), and the public entry point that's supposed to deliver it doesn't. Not
   a quick patch — trimming to `n_valid` happens because downstream event-detection/stability
   code (`BifurcationDetector`, Python `for i in range(n)` loops) also assumes concrete shapes, so
   a real fix means making that whole downstream path trace-safe, not just the one `int()` call.
   → tracked as the top item under "Engineering recommendations for v0.2" (item 1, engine
   consolidation) rather than patched piecemeal here. `tests/test_gpu_smoke.py`'s `vmap` test
   exercises `pseudo_arclength_scan` directly and documents this exact issue inline, so the smoke
   test passes for an honest reason rather than masking the gap.
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
- [ ] GitHub release — not yet confirmed created from the `v0.1.0` tag.
- [ ] Zenodo DOI — **deliberately deferred**, by decision, until a more mature release with more
  results; not a v0.1.0 blocker. `CITATION.cff` metadata is ready for whenever this happens.

**Out of scope (hidden / marked experimental):** periodic orbits, Floquet, BVP,
normal forms, codim-2, branch switching, two-parameter continuation.

### Remaining v0.1.0 work, concretely (updated 2026-07-19 — engineering items now done)

The two engineering items originally listed here are **done** (see "Release engineering" above):
`stability/eigenvalue.py` coverage 51%→98%, and a real, passing `tests/test_gpu_smoke.py`. Along
the way, that work also fixed two latent test flakes (issue #14) and surfaced one important new
finding, issue #13 (`jc.continuation()` isn't actually `vmap`-safe — tracked under v0.2 engine
consolidation, not a v0.1.0 blocker since the *underlying* `pseudo_arclength_scan` engine that
`example_06` uses genuinely is `vmap`-safe, and that's what the shipped example/README claims).

**v0.1.0 is published.** Remaining loose ends, non-blocking:

1. Confirm/cut a GitHub release from the `v0.1.0` tag if not already done.
2. Zenodo DOI archival — intentionally deferred until a more mature release with more results.

Issues #10 (legacy natural-continuation FD/bare-except) and #8/#9 (bothside, sub-epsilon tol) are
real but non-blocking for v0.1.0 — they don't affect the default `scan`/`PseudoArclength` path.
Fix opportunistically or fold into the v0.2 engine consolidation (see below), alongside issue #13.

## v0.2.0 — Periodic orbits
- [ ] Periodic-orbit continuation (collocation preferred over shooting)
- [ ] Floquet multipliers from monodromy matrix
- [ ] Period-doubling detection
- [ ] Limit-cycle examples (Van der Pol, Brusselator)

## v0.3.0+ — Advanced (demand-driven)
- [ ] Branch switching
- [ ] Two-parameter continuation
- [ ] Normal forms / Lyapunov **coefficient** `l₁` (Hopf criticality — a *bifurcation* invariant,
      NOT the Lyapunov exponent spectrum; see below)
- [ ] Codim-2 bifurcations (cusp, Bogdanov-Takens, ...)

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

1. **Retire the three-implementations-of-one-algorithm pattern before adding a fourth (periodic
   orbits).** `natural_continuation.py`, `pseudo_arclength.py`'s legacy OO class, and
   `scan_continuation.py` all implement predictor-corrector continuation; issue #10 above is a
   direct, observed consequence (a fix landed in one, not the others). Recommendation: make
   `Natural`/`PseudoArclength` in the functional API (`api.py`) thin configuration objects that
   *both* dispatch to the scan engine (predictor swapped, not a whole second code path), and
   either delete the legacy OO classes or explicitly mark them `deprecated`/frozen-as-is in v0.1.0
   docs. Do this before v0.2 adds `Collocation` as a third predictor/mesh strategy on top of the
   current duplication.
2. **Resolve the `eqx.Module` "open decision" (ARCHITECTURE.md §4, line ~170) now, before periodic
   orbits land.** v0.1's `BifProblem`/`Branch` are flat enough that hand-rolled
   `register_pytree_node` dataclasses work fine (and that's what's shipped, zero new deps — good
   call for v0.1). Periodic-orbit problems add real structure (mesh, `ntst`/`ncol`, a phase
   condition, static vs. traced fields) and v0.2/v0.3 add several more pluggable protocol
   implementations (`Collocation` predictor, `PeriodDoubling`/`LPC`/`NS` events, more solver
   variants) — exactly the case equinox's `Module`/`field(static=...)` idiom exists for.
   Recommendation: adopt `equinox` starting with the v0.2 periodic-orbit types, leave the already-
   shipped v0.1 equilibrium types as-is (not worth churning a working, tested surface).
3. **Generalize the `fold_solve.py` pattern into one reusable primitive before hand-writing it
   again for Hopf/LPC/PD/NS.** The genuinely novel piece of this project — Newton-in-
   `lax.while_loop` over an extended system `G(x,θ)=0`, wrapped in `jax.custom_vjp` implementing
   the implicit function theorem so `jax.grad` skips the iteration — is currently bespoke to folds.
   ARCHITECTURE.md §3.2 already calls Hopf/codim-2 versions "natural follow-ups"; before writing
   the second one, extract the shared scaffolding (e.g.
   `solvers/implicit.py: differentiable_root(G, x0, theta) -> (x*, custom_vjp)`) so each new
   differentiable event (Hopf, then LPC/PD/NS in v0.2/v0.3) is just a new `G`, not a new custom_vjp
   implementation. This is the highest-leverage single refactor for the strategic direction above.
4. **Replace `BifurcationDetector` with the sketched `Event` protocol (ARCHITECTURE.md §4.7) as
   part of fixing issue #7 (duplicate/spurious fold-vs-Hopf flags), not instead of it.** The
   current detector is one class doing sign-change scanning, bisection, and dedup for multiple
   bifurcation types at once — a plausible root cause of the duplicate-flag bug. Rewriting `Fold`/
   `Hopf` as small, independently-testable `Event` implementations (as already designed) is a
   natural place to fix the dedup logic once, cleanly, rather than patching the monolithic
   detector and inheriting the same fragility when `PeriodDoubling`/`LPC`/`NS` events are added in
   v0.2.
5. **Make `LinearSolver`/`EigenSolver` (ARCHITECTURE.md §4.6) real protocols with a `Dense()`
   implementation now, even though nothing else exists yet.** Right now the "GPU-ready" and
   "matrix-free/iterative" claims in ARCHITECTURE.md §3's comparison table are aspirational — every
   solve is hardcoded `jnp.linalg.solve`/`jnp.linalg.eigvals`. Introducing the protocol boundary
   (with `Dense()` as the only implementation for now) costs little and means a future
   `GMRES()`/`Arnoldi()` swap — needed for the DDE eigensolver seam in §10.2 and for genuinely
   large systems — doesn't require touching the continuation loop. Doing this before the v0.1 GPU
   smoke test also makes that smoke test mean something: "runs on GPU" vs. the honest "runs on GPU
   because everything is dense and small, scaling claims unverified."

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
    (https://pypi.org/project/jaxcont/). GitHub release still to be confirmed; Zenodo DOI
    deliberately deferred to a more mature release.
11. **v0.2 kickoff — do the engineering cleanup *before* the periodic-orbit feature work**, per
    "Engineering / architecture recommendations for v0.2" above, in this order: (i) consolidate
    the three continuation-engine implementations onto the scan engine **and make the result
    actually `vmap`-safe** (issue #13 — the `int(res.n_valid)` concretization in `_run_scan` is the
    same kind of fix as (i), so do them together); (ii) decide `equinox` for the new periodic-orbit
    types; (iii) extract `fold_solve.py`'s differentiable-root pattern into a reusable primitive;
    (iv) replace `BifurcationDetector` with real `Event` implementations, fixing issue #7 as part
    of the rewrite; (v) introduce `LinearSolver`/`EigenSolver` as real (if currently
    single-implementation) protocols. Then build periodic-orbit collocation with a static
    (non-traced) `ntst`/`ncol` mesh on top of the cleaned-up spine, matching MatCont's own
    `ntst`/`ncol` discretization discipline (manual §7.2) and the fixed-shape-buffer requirement
    the whole-loop-JIT/`vmap` story already depends on (ARCHITECTURE.md §3.1, §4.3).

---

## Reference / learning
- Kuznetsov, *Elements of Applied Bifurcation Theory*
- BifurcationKit.jl · MATCONT · AUTO-07p · PyDSTool
- JAX docs: JIT, vmap, `lax.while_loop`, `lax.scan`
