# JaxCont Roadmap — Single Source of Truth

**Last updated:** 2026-07-17
**Current version:** 0.1.0-dev (unreleased)
**Scope decision:** Ship a focused **equilibrium continuation** library first. See [PROJECT_REVIEW_2026-07.md](PROJECT_REVIEW_2026-07.md) for the full rationale.

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

1. **Broken singular-matrix handling.** `try/except` around `jnp.linalg.solve` in
   `core/pseudo_arclength.py` never fires — JAX returns NaN, doesn't raise.
   → check conditioning explicitly or use `lstsq`.
2. **Finite-difference `df/dp`.** `core/pseudo_arclength.py` uses manual FD for the
   parameter derivative while everything else is autodiff. → replace with `jacfwd`.
3. **No JIT on the hot loop.** Newton/corrector is plain Python calling small JAX ops
   (~50 ms/point, dispatch-bound). → rewrite inner solve as `jax.lax.while_loop`.
   This is the make-or-break item for the "high-performance JAX" claim.
4. **README placeholders.** `Your Name`, `yourusername`, stub citation/DOI.

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
- [ ] Normal forms / Lyapunov coefficient (Hopf criticality)
- [ ] Codim-2 bifurcations (cusp, Bogdanov-Takens, ...)

---

## Do this next (in order)

1. ✅ **Tidy notes** — done: this roadmap + archive.
2. **JIT the Newton loop** and re-run `examples/profile_continuation.py`. Proves the
   core thesis. If JAX gives no real speedup here, that's critical to learn now.
3. **Fix the two correctness bugs** (issues #1, #2) — small, high-trust payoff.
4. Then: docs + packaging → ship v0.1.0.

---

## Reference / learning
- Kuznetsov, *Elements of Applied Bifurcation Theory*
- BifurcationKit.jl · MATCONT · AUTO-07p · PyDSTool
- JAX docs: JIT, vmap, `lax.while_loop`, `lax.scan`
