# JaxCont Roadmap — Single Source of Truth

**Last updated:** 2026-07-18
**Current version:** 0.1.0-dev (unreleased)
**Scope decision:** Ship a focused **equilibrium continuation** library first. See [PROJECT_REVIEW_2026-07.md](PROJECT_REVIEW_2026-07.md) for the full rationale.
**API design:** Committed to a functional, diffrax-style surface (`continuation(problem, alg, ...)`).
See [ARCHITECTURE.md](ARCHITECTURE.md) for the full spine contract and provisional v0.2+ API.

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
| Stability (eigenvalues) | ✅ Works | |
| Plotting | ⚠️ Works, 9% cov | Under-tested |
| Periodic orbits | ⚠️ Stub | Untested, hidden from v0.1.0 |
| Floquet multipliers | ❌ Stub, 22% cov | Hidden from v0.1.0 |
| BVP (collocation/shooting) | ❌ NotImplementedError | Hidden from v0.1.0 |
| Normal forms / Lyapunov coeff | ❌ Returns placeholders | Hidden from v0.1.0 |

**Test suite:** 52 passed in the default fast suite (18 slow tests deselected).

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

---

## v0.1.0 — "Equilibria, done well" (target: next release)

Public surface is the functional API — `bif_problem` / `continuation` / `Fold`/`Hopf` — per
[ARCHITECTURE.md](ARCHITECTURE.md). The OO classes remain as a deprecated internal shim.

**Core — done:**
- [x] Natural + pseudo-arclength equilibrium continuation
- [x] Fold + Hopf detection with refinement (legacy detector; port to `Event` protocol)
- [x] Stability along the branch
- [x] Bifurcation-diagram plotting
- [x] Examples: pitchfork, Lorenz, neural-mass
- [x] Autodiff `df/dp` (issue #2)
- [x] Robust bordered solve — no singular-`df/du` inversion (issue #1)
- [x] Functional spine: `BifProblem` + `continuation()` over the loop ([api.py](../src/jaxcont/api.py))
- [x] Whole-loop `lax.while_loop` engine — validated (issue #3, [scan_continuation.py](../src/jaxcont/core/scan_continuation.py))

**Core — done (scan path):**
- [x] Scan engine wired behind `continuation()` and made the default
- [x] Fold/Hopf events detected and refined on scan results
- [x] Stability computed by the vectorized `branch_eigenvalues` post-pass

**JAX differentiators — the reason to exist (ARCHITECTURE §3); must ship as first-class:**
- [x] `vmap` parameter-sweep example — [example_08_vmap_sweep.py](../examples/example_08_vmap_sweep.py)
      (256 diagrams, one kernel, **163× vs a Python loop**)
- [x] Differentiable-bifurcation example — [example_09_differentiable.py](../examples/example_09_differentiable.py)
      (reverse-mode `jax.grad` inverse design on a differentiable equilibrium; forward-mode
      `jacfwd` through the engine). Both cross-checked vs finite differences.
- [x] Reverse-mode `jax.grad` of a fold location — [fold_solve.py](../src/jaxcont/bifurcations/fold_solve.py)
      (`jc.fold_parameter`/`fold_point`: extended system + `custom_vjp` implicit diff; exact to
      analytic incl. vector-θ Jacobians). Hopf/codim-2 extended-system solvers are follow-ups.
- [x] These are the headline of the README and docs quickstart

**Release engineering:**
- [ ] Core modules (`core/`, `stability/eigenvalue.py`) >85% coverage (on the engine path)
- [ ] GPU smoke test
- [x] Honest README led by the vmap/grad story + stated scope + fixed placeholders
- [x] Sphinx docs: install, quickstart, Sphinx-Gallery examples, API reference
- [x] Clean sdist/wheel build + Twine metadata validation
- [ ] TestPyPI → PyPI → GitHub release/Zenodo DOI

**Out of scope (hidden / marked experimental):** periodic orbits, Floquet, BVP,
normal forms, codim-2, branch switching, two-parameter continuation.

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
8. ✅ **Ship the differentiators as examples** — done: [example_08](../examples/example_08_vmap_sweep.py)
   (`vmap`, 163×) + [example_09](../examples/example_09_differentiable.py) (`grad` of a fold via
   [fold_solve.py](../src/jaxcont/bifurcations/fold_solve.py), + forward-mode `jacfwd`).
9. ✅ **Trim `__init__.py`** — done: top-level surface is the equilibrium spine; periodic/BVP/
   Floquet/period-doubling stubs are importable only from their submodules.
10. **Docs + packaging → ship v0.1.0.** ← **IN PROGRESS** README and Sphinx quickstart now lead
    with the `vmap`/gradient story; scan is the default with fold/Hopf refinement; author and
    citation placeholders are fixed. Remaining external release sequence: TestPyPI → PyPI →
    GitHub release/Zenodo DOI.

---

## Reference / learning
- Kuznetsov, *Elements of Applied Bifurcation Theory*
- BifurcationKit.jl · MATCONT · AUTO-07p · PyDSTool
- JAX docs: JIT, vmap, `lax.while_loop`, `lax.scan`
