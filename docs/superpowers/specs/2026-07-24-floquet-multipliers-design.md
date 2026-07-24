# Floquet Multipliers via Collocation Monodromy — Design Spec

**Status:** Approved for implementation planning.
**Roadmap item:** v0.2.0 "Periodic orbits" (`notes/ROADMAP.md`), second checklist item —
"Floquet multipliers from the monodromy matrix." Second sub-project of the same epic as
periodic-orbit collocation continuation (first sub-project, already shipped — see
[its design spec](2026-07-24-periodic-orbit-collocation-design.md) and
[plan](../plans/2026-07-24-periodic-orbit-collocation.md)).

## Motivation

`api.py`'s `_run_scan` currently raises `ValueError` if `settings.compute_stability=True` is
passed for a `kind="periodic"` `BifProblem` — a deliberate placeholder, because the equilibrium
stability pass eigendecomposes `df/du`, which for a periodic problem's assembled residual is the
entire collocation Jacobian, not a meaningful dynamical quantity. The error message explicitly
says Floquet multipliers are a planned future feature. This spec is that feature: it replaces the
guard clause with real Floquet-multiplier computation, and enables `Branch.eigenvalues`/
`Branch.stable` for periodic branches, matching ARCHITECTURE.md §6's original (explicitly
provisional) sketch — "Floquet multipliers land in `Branch.eigenvalues` for `kind="periodic"`."

## Scope

**In scope:** computing the monodromy matrix `Φ(T)` and its eigenvalues (the Floquet multipliers)
at each point along a periodic branch, wired into `continuation()`'s existing
`settings.compute_stability` pass exactly the way equilibrium stability already works.

**Out of scope (explicit, future sub-projects of the same epic):**
- **Period-doubling / Neimark–Sacker event detection.** Needs this sub-project's multipliers to
  exist first — a multiplier crossing `-1` (period-doubling) or a complex-conjugate pair crossing
  the unit circle (Neimark–Sacker) are the natural next `Event` implementations, not built here.
- **Limit-cycle example scripts.** Need the above to be a meaningful demonstration.
- **Adaptive mesh, DDE eigensolvers, or any other periodic-orbit-continuation-sub-project scope
  cuts already made** — unchanged, not revisited here.

## Architecture

**Core decision: reuse the collocation structure already built, not a second numerical pathway.**
The existing (currently dead, pre-v0.1, `scipy.integrate.solve_ivp`-based) `stability/floquet.py`
stub computes the monodromy matrix by re-integrating the variational equation
`dΦ/dt = J(u(t))·Φ` as a separate IVP. That is architecturally incompatible with the collocation
representation (no continuous, re-integrable trajectory exists — the orbit lives as discrete
mesh/collocation-point states) and would reintroduce the "JaxCont integrates ODEs itself" pattern
the collocation design spec deliberately avoided.

Instead, `Φ(T)` is built as a byproduct of the same collocation machinery
(`core/collocation.py`'s Lagrange differentiation matrix `D`, the raw right-hand side's
`df/du` Jacobian blocks evaluated along the already-converged orbit): a block linear recursion
across the `ntst` mesh intervals, seeded with the `n×n` identity (one column per basis
perturbation direction), composes to `Φ(T)`. This is fully JAX-native, differentiable, reuses
existing infrastructure, and needs zero new dependencies. `Φ(T)`'s eigenvalues (via the
`EigenSolver` protocol — the consumer anticipated when that protocol was built) are the Floquet
multipliers.

**Data plumbing:** `_run_scan` (generic, doesn't know collocation internals) needs the raw
right-hand side `f` and the `Collocation` mesh config to build `Φ(T)` at a branch point — neither
is directly visible from a plain `BifProblem` (only the assembled residual is). `periodic_orbit_problem`
extends what it packs into `BifProblem.args` — currently `(u_ref_coll, uref_prime_coll)` for the
phase condition — to `(u_ref_coll, uref_prime_coll, raw_f, mesh)`. `args` was documented as "extra
parameters — the axis you vmap/grad over"; for periodic problems it already meant "collocation
bookkeeping" (the phase-condition data isn't a dynamics parameter either), so this extends an
existing pattern rather than introducing a new one. `BifProblem`'s fields/signature and
`continuation()`'s signature are unchanged.

`_run_scan`'s stability-computation branch dispatches on `problem.kind`: `"equilibrium"` keeps
calling `branch_eigenvalues` exactly as today; `"periodic"` unpacks `args` and calls the new
Floquet path instead. The `compute_stability=True` guard clause for periodic problems is removed
entirely — replaced by correct computation, not merely un-blocked.

## Math

At a converged branch point (state `U`, parameter `p`), unpack `U` into the mesh-point states
`u_0, ..., u_{ntst-1}` (the same layout `periodic_orbit_problem` already uses). For each mesh
interval `i`, evaluate the raw right-hand side's Jacobian `J_i = jacfwd(raw_f, argnums=0)(u_i, p)`
at that interval's collocation points (reusing the same evaluation points the residual's defect
equations already use). The block recursion carries a perturbation matrix `M` (initialized to the
`n×n` identity) across each interval using the local Lagrange differentiation matrix `D` and the
interval's Jacobian blocks — the same linear structure the collocation defect equations encode,
solved here for the *sensitivity* of the endpoint map rather than for the state itself. After all
`ntst` intervals, `M = Φ(T)`. Floquet multipliers are `eigen_solver(Φ(T))`.

**Not yet verified against running code** — this is a derivation from standard collocation-BVP
sensitivity theory, not confirmed-working. Per this project's established discipline (the
collocation scheme itself needed a numerical surprise fixed before its plan was finalized), the
exact recursion is prototyped and numerically verified against the closed-form answer below
*before* being written into an implementation plan — not decided by reasoning alone.

**Verification target:** `r' = r·(ρ - r²), θ' = 1` (the same system periodic-orbit continuation's
own tests use) has, at `ρ=1`, the exact circular limit cycle `x=cos(t), y=sin(t)`, `T=2π`.
Linearizing radially: `dr'/dr = ρ - 3r²`, evaluated at `r=√ρ` gives `ρ - 3ρ = -2ρ`. The
non-trivial Floquet multiplier is `exp(-2ρ·T) = exp(-4πρ)` (at `ρ=1`: `exp(-4π) ≈ 3.49e-6`,
strongly attracting). Floquet's theorem guarantees an autonomous periodic orbit's *other*
multiplier is exactly `1` (the trivial one, tangent to the flow direction) — independent of any
external reference tool.

## Stability boolean for periodic branches

Equilibrium stability is "all eigenvalues have negative real part" (a half-plane condition).
Floquet stability is "all *non-trivial* multipliers lie inside the unit circle" (a magnitude
condition) — the trivial multiplier is always exactly `1`, on the unit circle by construction, and
must be excluded or every periodic branch would read as marginally unstable. Identify the trivial
multiplier as the one closest to `1` (`argmin(|multiplier - 1|)` — the same heuristic the old dead
stub's `analyze_periodic_orbit_stability` already used, worth reusing the idea even though its
surrounding scipy-integration code is not), then `stable = all(|other multipliers| < 1)`.

For the verification system above, this must read `stable=True` for every `ρ>0` in the tested
range, since `exp(-4πρ) < 1` always.

## File layout

- **`src/jaxcont/core/collocation.py`**: add the monodromy block-recursion builder, reusing the
  existing `D` matrix / mesh conventions already there.
- **Rewrite `src/jaxcont/stability/floquet.py`** (currently the dead, `scipy`-based,
  pre-v0.1 stub — deleted outright, not incrementally edited): new
  `floquet_multipliers(raw_f, mesh, U, p, eigen_solver=DenseEigen()) -> Array`, built on the new
  collocation-monodromy machinery. The old `analyze_periodic_orbit_stability`'s trivial-multiplier
  heuristic is reused conceptually inside `_run_scan`'s stability-boolean computation (see above),
  not as a standalone ported function — it's a two-line piece of logic, not worth its own module
  boundary given exactly one caller.
- **Modify `src/jaxcont/problems/periodic.py`**: `args` grows to
  `(u_ref_coll, uref_prime_coll, raw_f, mesh)`.
- **Modify `src/jaxcont/api.py`**: `_run_scan`'s stability branch dispatches on `problem.kind`;
  the `compute_stability=True` guard clause for periodic problems is removed.
- **Untouched:** `BifProblem`'s fields/signature, `continuation()`'s signature,
  `core/scan_continuation.py`, `bifurcations/events.py`, `solvers/protocols.py`,
  `bifurcations/period_doubling.py` (still a stub — the next sub-project).

## Testing

Per this project's established standard: empirical verification against a known answer.

1. **Monodromy/multiplier unit test:** at a converged branch point for the circle system at
   `ρ=1`, the computed Floquet multipliers match `{1, exp(-4π)}` (order-independent) to
   float32-achievable tolerance (calibrated the same way periodic-orbit continuation's own tests
   were — this project runs float32 by default, and the collocation Jacobian's einsum-heavy
   internals may need the same `jax.default_matmul_precision("float32")` treatment found necessary
   there; re-verify, don't assume it transfers automatically).
2. **Stability-boolean test:** for the same branch point, `stable=True`, and the identified trivial
   multiplier is within tolerance of `1`.
3. **Full `continuation()` integration test:** sweep `ρ` over a range with `compute_stability=True`
   (now permitted); confirm `Branch.eigenvalues` has shape `(n_valid, n)` matching equilibrium's
   existing convention, and `Branch.stable` is `True` at every point (consistent with
   `exp(-4πρ) < 1` for all `ρ>0` in the tested range).
4. **Regression guard:** the previous sub-project's `test_compute_stability_true_raises_for_periodic_problem`
   test (asserting the guard clause raises) is now obsolete and must be replaced/removed —
   `compute_stability=True` is the whole point of this sub-project, not an error case anymore.
5. Full existing suite green, confirming zero regression to equilibrium continuation's
   `compute_stability` behavior (the dispatch-by-kind must leave the equilibrium path provably
   untouched).

## Global Constraints

- `BifProblem`'s fields/signature and `continuation()`'s signature are unchanged.
- No re-integration of a separate variational-equation IVP — Floquet multipliers are computed from
  the existing collocation structure only.
- The monodromy math must be numerically prototyped and verified against the closed-form circle
  example before being written into an implementation plan — not decided by derivation alone.
- The `compute_stability=True` guard clause for periodic problems is removed, not merely
  loosened — it's replaced by correct computation.
- No period-doubling/Neimark–Sacker event detection, no new example scripts, no changes to
  `core/scan_continuation.py`, `bifurcations/events.py`, or `solvers/protocols.py` in this
  sub-project.
