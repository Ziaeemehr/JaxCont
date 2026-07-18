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

**Test suite:** 49 passed, 4 skipped, ~60% coverage (run in `~/envs/jaxcont`).

---

## Known issues to fix before release

1. ✅ **Broken singular-matrix handling.** *(fixed 2026-07-18)* The corrector now solves the
   full `(n+1)×(n+1)` bordered system instead of eliminating through `df/du`, so it no longer
   inverts a matrix that is singular at folds. The dead `try/except` is gone.
2. ✅ **Finite-difference `df/dp`.** *(fixed 2026-07-18)* Replaced with `jacfwd(f, argnums=1)`
   in both `correct()` and `compute_tangent()`.
3. ⚠️ **JIT on the hot loop.** *(partial 2026-07-18)* The corrector inner loop is now a JIT'd
   `lax.while_loop` (`_correct_jit`). **But profiling showed ~no speedup** (~21 ms/point before
   and after, n=1 and n=3): the cost is the *Python outer loop* (per-step dispatch for tangent +
   eigenvalues, host syncs, bookkeeping), not the corrector. The "high-performance JAX" claim
   needs **whole-loop JIT (`lax.scan`)** and/or **`vmap` batching** — see
   [ARCHITECTURE.md §2](ARCHITECTURE.md). This is now an API-design-dependent task, not a
   local fix.
4. **README placeholders.** `Your Name`, `yourusername`, stub citation/DOI.
5. ⚠️ **Interim corrector hangs on saturating branches.** *(found 2026-07-18)* The new JIT
   bordered corrector stalls when a branch runs into a degenerate-Jacobian regime — e.g.
   `smooth_rhs = r − tanh(x)` pushed to `x ≳ 8`, where `sech²(x)` underflows and the bordered
   system degenerates. The *original* block-elimination corrector happened to sidestep this. The
   new functional API (`jc.continuation`) calls the same corrector, so it inherits the gap.
   → fix during the `lax.scan` whole-loop rewrite (issue #3): NaN/inf guard + early termination
   in the Newton `while_loop`, and detect the degenerate step to shrink `ds` instead of stalling.
   The tests that expose it (`tests/test_adaptive_stepsize.py`) are marked `slow` and excluded
   from the default `make test` run, **not deleted** — they are the canary for this fix.

---

## v0.1.0 — "Equilibria, done well" (target: next release)

**In scope (public API):**
- [x] Natural + pseudo-arclength equilibrium continuation
- [x] Fold + Hopf detection with refinement
- [x] Stability along the branch
- [x] Bifurcation-diagram plotting
- [x] Examples: pitchfork, Lorenz, neural-mass
- [ ] **JIT'd Newton/corrector inner loop** (issue #3)
- [ ] Fix singular-solve bug (issue #1)
- [ ] Autodiff `df/dp` (issue #2)
- [ ] Core modules (`core/`, `solvers/`, `stability/eigenvalue.py`) >85% coverage
- [ ] GPU smoke test
- [ ] Honest README with stated scope + fixed placeholders
- [ ] Sphinx docs: install, quickstart, one tutorial, API reference
- [ ] Clean wheel build; TestPyPI → PyPI; Zenodo DOI

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
5. **Implement the functional spine** (`BifProblem`, `continuation()`, `Event`, solver protocols)
   over the existing loop; keep the OO class as a deprecated wrapper for one release.
6. **Whole-loop `lax.scan`** behind the new API → makes the performance claim real; re-profile.
7. **Trim `__init__.py`** to the equilibrium spine (hide periodic/BVP/Floquet stubs).
8. Then: docs + packaging → ship v0.1.0.

---

## Reference / learning
- Kuznetsov, *Elements of Applied Bifurcation Theory*
- BifurcationKit.jl · MATCONT · AUTO-07p · PyDSTool
- JAX docs: JIT, vmap, `lax.while_loop`, `lax.scan`
