# Periodic-Orbit Continuation via Orthogonal Collocation — Design Spec

**Status:** Approved for implementation planning.
**Roadmap item:** v0.2.0 "Periodic orbits" (`notes/ROADMAP.md`), first checklist item —
"Periodic-orbit continuation (collocation preferred over shooting)". First sub-project of a
4-part epic (this / Floquet multipliers / period-doubling detection / limit-cycle examples), each
of which gets its own spec once its prerequisite lands.

## Motivation

JaxCont's v0.1 spine handles equilibria only. Periodic-orbit (limit-cycle) continuation is, per
`ROADMAP.md`'s strategic-direction section, "the single biggest gap right now" relative to
MatCont/AUTO/BifurcationKit.jl, and the one most continuation-tool users actually need. All five
v0.2 engineering-prep items (engine consolidation, `equinox` adoption, `differentiable_root`
extraction, the `Event` protocol rewrite, `LinearSolver`/`EigenSolver` protocols) were done
specifically to make this land cleanly; this spec is where that groundwork gets used.

## Scope

**In scope:** representing a periodic orbit via fixed-mesh orthogonal collocation, building the
extended nonlinear system, resampling a user-supplied trajectory guess onto the mesh, and
continuing the resulting `BifProblem` with the *existing, unmodified* continuation engine.

**Out of scope (explicit, each a future sub-project of the same epic):**
- **Floquet multipliers** (monodromy-matrix eigenvalues) — needs this sub-project's periodic-orbit
  representation to exist first.
- **Period-doubling / fold-of-cycles / Neimark–Sacker event detection** — period-doubling and
  Neimark–Sacker need Floquet multipliers; fold-of-cycles is already free (see "Interop" below).
- **Limit-cycle example scripts** (Van der Pol, Brusselator) — need the above to be meaningful
  demonstrations, not just a correctness check against a synthetic system.
- **Adaptive mesh redistribution** (AUTO/MatCont-style relocation of mesh/collocation points to
  concentrate resolution where the orbit is steep, re-triggered periodically during continuation).
  This is not an oversight — it would change `U`'s shape mid-run, breaking the fixed-shape-buffer
  discipline the whole scan-engine architecture relies on for `jit`/`vmap`, and is a genuinely
  separate, nontrivial feature deserving its own design pass if ever needed. `ntst`/`ncol` are
  fixed for the lifetime of a continuation run, chosen once by the caller.
- **JaxCont-native ODE integration.** The user supplies the initial trajectory guess from
  whatever integrator they already use (`diffrax`, `scipy`, hand-rolled); JaxCont only resamples
  it onto the collocation mesh. Matches `ARCHITECTURE.md` §9's existing stance that general ODE
  integration belongs to `lyapax`/`diffrax`, not JaxCont, and matches BifurcationKit.jl's
  `PeriodicOrbitOCollProblem`, which takes the same approach (leans on `DifferentialEquations.jl`
  for the initial trajectory rather than shipping its own integrator).

## Architecture

**Core decision: zero changes to `core/scan_continuation.py`, `BifProblem`, or `continuation()`.**
`pseudo_arclength_scan`/`natural_scan` are already generic over the dimension and meaning of the
continued state `u` — they only need `f(u, p) -> residual` and use `jacfwd` + a bordered Newton
solve with no assumption about what `u` represents. A periodic orbit's full collocation system
(every mesh/collocation-point state, the period, one phase-condition equation) is just a very
large `u`. So "periodic-orbit continuation" reduces entirely to: build the right `f` and the right
initial `U0`, then hand both to the existing engine unchanged.

```python
mesh = jc.Collocation(ntst=20, ncol=4)
prob = jc.periodic_orbit_problem(f, u_trajectory, t_trajectory, period0, p0, mesh)
sol = jc.continuation(prob, p_span=(...), settings=jc.ContinuationPar(compute_stability=False))
```

`periodic_orbit_problem` is a *factory*: it returns an ordinary `BifProblem` whose `u0` is the
resampled, flattened collocation unknown vector `U0`, and whose `f` is the assembled residual
(closing over `mesh`'s static `ntst`/`ncol` and the phase-condition reference trajectory).
`BifProblem.kind` stays the inert `Literal["equilibrium", "periodic", "bvp"]` it already is — no
new special-casing is added to `continuation()`'s dispatch, **except** the one guardrail described
under "Interop constraints" below.

Today's dense `Dense()`/`DenseEigen()` (`solvers/protocols.py`) treat the collocation Jacobian as
one big dense matrix via `jacfwd` — correct, but not exploiting its near-block-bidiagonal
structure the way AUTO's specialized linear algebra does. That's an explicitly deferred
optimization: a future `LinearSolver` implementation (e.g. `BlockBorderedSolver`) can slot in
later without revisiting this design or the engine, exactly the seam `LinearSolver`/`EigenSolver`
were built for.

## Collocation scheme

Standard Gauss–Legendre orthogonal collocation (AUTO/COLSYS/BifurcationKit.jl
`PeriodicOrbitOCollProblem` convention), fully discretized (collocation-point values are unknowns,
not eliminated via an implicit per-interval solve).

**Time normalization:** actual time `t = T·τ`, `τ ∈ [0, 1]`, `T` the (unknown, continued) period.
This is how the period enters the residual as a scale factor rather than as an integration bound.

**Mesh:** `ntst` equally-spaced subintervals partition `[0, 1]`: `τ_i = i/ntst` for
`i = 0, ..., ntst`, with `τ_ntst ≡ τ_0` (periodicity closes the ring). `ntst`, `ncol` are
`eqx.field(static=True)` on `Collocation` — compile-time constants fixing `U`'s shape for `jit`,
following `CollocationMeshScaffold`'s already-proven static/traced split.

**Per-interval representation:** on subinterval `i` (from `τ_i` to `τ_{i+1}`, length `h = 1/ntst`),
place `ncol` interior collocation points at the Gauss–Legendre nodes of degree `ncol` (roots of the
degree-`ncol` Legendre polynomial on `[-1,1]`, affinely mapped into `(τ_i, τ_{i+1})`). Together with
the left mesh point `τ_i`, these `ncol + 1` local nodes uniquely determine a degree-`ncol` vector
Lagrange polynomial per interval. The corresponding `(ncol+1) × (ncol+1)` Lagrange differentiation
matrix (built once, since the local node layout relative to each interval is identical up to
affine scaling — only `h` differs, and `h` is constant since the mesh is uniform) gives the
polynomial's derivative at each local node as a linear combination of the nodal values.

**Unknowns** (packed into the flat vector `U`):
- `u_i` for `i = 0, ..., ntst - 1`: state at each mesh point (`n`-dimensional). `u_ntst` is not
  separately stored — periodicity identifies it with `u_0`.
- `u_{i,j}` for `i = 0, ..., ntst - 1`, `j = 1, ..., ncol`: state at each interior collocation
  point (`n`-dimensional).
- `T`: the scalar period.

Total: `dim(U) = n · ntst · (ncol + 1) + 1`.

**Residual `F(U, p)`** (same total size, square system):
1. **Defect equations** (`n · ntst · ncol` scalar equations): for each interval `i` and interior
   node `j = 1, ..., ncol`,
   `(1/h) · Σ_k D_{jk} · u_{i,k} = T · f(u_{i,j}, p)`,
   where `D` is the local Lagrange differentiation matrix (index `k=0` is the left mesh point
   `u_i`, indices `k=1..ncol` are the interior points `u_{i,1..ncol}`), and `f` is the user's
   right-hand side.
2. **Continuity / periodicity equations** (`n · ntst` scalar equations): for each interval
   `i = 0, ..., ntst - 1`, evaluate that interval's Lagrange interpolant at its right endpoint
   (`τ_{i+1}`, an extrapolation beyond the interior nodes using the same local basis) and require
   it equal the next interval's left mesh point — `u_{(i+1) mod ntst} = p_i(τ_{i+1})`. For
   `i < ntst - 1` this is ordinary continuity; for `i = ntst - 1` this *is* the periodicity
   condition (`u_0 = p_{ntst-1}(1)`), handled by the same formula via the `mod`.
3. **Integral phase condition** (1 scalar equation):
   `Σ_i Σ_j w_j · h · ⟨u_{i,j} - u_ref_{i,j}, u_ref'_{i,j}⟩ = 0`,
   where `w_j` are the Gauss–Legendre quadrature weights for degree `ncol`, `u_ref` is the
   resampled *initial* guess (fixed for the whole continuation run — the reference orbit whose
   phase we're pinning to), and `u_ref'` is `u_ref`'s derivative at the same collocation points
   (available directly from `f(u_ref, p0)`, since `u_ref` approximately satisfies the dynamics).

`dim(F) = n·ntst·ncol + n·ntst + 1 = n·ntst·(ncol+1) + 1 = dim(U)`. Square.

## Initial guess construction

`periodic_orbit_problem(f, u_trajectory, t_trajectory, period0, p0, mesh)`:
1. Rescale `t_trajectory` to normalized `τ ∈ [0, 1]` via `τ = t_trajectory / period0`.
2. For each mesh point `τ_i` and each collocation point in each interval, interpolate
   `u_trajectory` (given at the user's own, generally irregular, `t_trajectory` samples) via
   piecewise-linear interpolation (`jnp.interp`, per state component) to get the initial nodal
   values `u_i^{(0)}`, `u_{i,j}^{(0)}`. This need only be a *reasonable* starting point for
   Newton — collocation's accuracy comes from the residual, not from a highly accurate initial
   guess.
3. Flatten into `U_guess` in the same layout `F` expects.
4. `u_ref`/`u_ref'` for the phase condition (see above) are captured from this same initial
   resampling — they do not change as continuation proceeds, even though `U` (and thus the
   *current* orbit) does. This is what "phase condition" means: pinning the time-shift relative to
   the *original* guess, not to a moving target.
5. **Refine `U_guess` to convergence before returning it as `u0`.** `pseudo_arclength_scan`/
   `natural_scan` do not Newton-correct their starting point — the engine takes `(u0, p0)` as
   already satisfying `f(u0, p0) ≈ 0` and marks slot 0 `converged=True` unconditionally (verified
   by reading `core/scan_continuation.py`: `_tangent` is computed directly from `(u0, p0)`, and the
   `body` loop only ever corrects *predicted* points from step 1 onward). That's a reasonable
   contract for equilibrium problems, where callers typically already supply an exact or
   near-exact point, but directly contradicts this factory's premise of a *coarse* trajectory
   guess. So `periodic_orbit_problem` calls `differentiable_root` (`solvers/implicit.py`, built
   earlier this v0.2 cycle for exactly this — implicit-root problems with IFT-based
   differentiability) on `F(·, p0)` starting from `U_guess`, and uses the converged result as the
   returned `BifProblem`'s `u0`. This was verified end-to-end in prototyping (see "Verification
   performed during design," below): from a deliberately wrong guess (radius 0.8 instead of the
   true 1.0, a phase offset, and a period guess of 5.5 instead of the true `2π`), the refined `U0`
   matches the exact circle to residual norm `~5e-15` under `jax.jacfwd` + `jit`.
6. `u_ref`/`u_ref'` (step 4) are captured from the *pre-refinement* `U_guess`, not the refined
   `U0` — the phase condition pins time-shift relative to what the caller supplied, independent of
   how the refinement converges within that phase.

## Verification performed during design

Before writing this into an implementation plan, the full scheme (differentiation matrix,
defect/continuity/periodicity/phase-condition residual, and initial refinement) was prototyped and
numerically verified against the closed-form circle example from "Testing," below — first in plain
NumPy/`scipy.optimize.fsolve` (residual norm `3.5e-14`, period `6.283185313` vs. exact
`6.283185307179586`), then re-verified in JAX under `jax.jacfwd` with a manual Newton loop using
`jnp.linalg.solve` (matching to all reported digits, residual norm `4.7e-15`), and confirmed to
trace cleanly under `jax.jit`. The Lagrange differentiation matrix was separately checked against
a degree-`ncol` polynomial's exact derivative (max error `5.6e-15`, machine precision). This is not
a design-only exercise — the implementation plan's code is transcribed from what was actually run.

## Interop constraints with existing machinery

- **`Fold()` events work unmodified and are meaningful.** A fold-of-cycles (a periodic-orbit
  turning point in the continuation parameter) is detected by the exact same tangent
  `dp`-component sign change `Fold.test_function` already implements — it's dimension-agnostic,
  no special-casing needed.
- **`Hopf()` events are not applicable and must not silently produce garbage.** `Hopf.test_function`
  eigendecomposes `df/du` at each branch point; for a periodic `f`, that's the entire
  `n·ntst·(ncol+1) × n·ntst·(ncol+1)` collocation Jacobian, not a meaningful dynamical quantity (a
  Hopf bifurcation is an *equilibrium* concept — the periodic-orbit analogues are period-doubling
  and Neimark–Sacker, both future items needing Floquet multipliers). This is a documentation-only
  constraint for now (don't pass `events=[Hopf()]` to a periodic `continuation()` call) — adding
  enforcement would require `continuation()`/`events` machinery to know about `BifProblem.kind`,
  which is a larger change than this sub-project's "zero changes to existing machinery" goal
  justifies for a single footgun. Revisit if it proves to be a real trap in practice.
- **`settings.compute_stability=True` must raise a clear error for periodic problems, not silently
  return garbage.** `_run_scan`'s stability pass eigendecomposes `df/du` for the same reason
  described above — meaningless for a periodic `f`, and wastefully large. Unlike the `Hopf()`
  case, this is enforced: `_run_scan` gains a guard clause —
  `if problem.kind == "periodic" and settings.compute_stability: raise ValueError(...)` — pointing
  the caller at passing `settings=ContinuationPar(compute_stability=False)` and noting Floquet
  multipliers are a future item. **This is the one deliberate, narrow exception to "zero changes
  to `api.py`"**: a guard clause, not new solver logic, and it only fires for `kind="periodic"`,
  so equilibrium continuation is provably unaffected.

## File layout

- **New: `src/jaxcont/core/collocation.py`** — pure numerics: Gauss–Legendre nodes/weights for
  degree `ncol`, the local Lagrange differentiation matrix, mesh construction, and the
  `Collocation` config type (`ntst`/`ncol` as `eqx.field(static=True)`). Mirrors
  `core/scan_continuation.py`'s role as the pure-numerics engine layer — no `BifProblem`/API
  concerns here.
- **Rewrite: `src/jaxcont/problems/periodic.py`** — the existing `PeriodicOrbitProblem`
  (`scipy.integrate.solve_ivp`-based shooting stub, pre-v0.1, non-jittable) is deleted outright
  and replaced with `periodic_orbit_problem(...) -> BifProblem` (assembles `F`/`U0` using
  `core.collocation`'s building blocks). Mirrors `problems/equilibrium.py`'s existing style (plain
  functions, no OO ceremony).
- **Delete: `src/jaxcont/core/_periodic_eqx_scaffold.py`** and **`tests/test_equinox_scaffold.py`**
  — the scaffold's own docstring instructs deleting it once real periodic-orbit continuation
  exists; `core/collocation.py`'s `Collocation` type supersedes it with real tests.
- **Modify: `src/jaxcont/api.py`** — the one guard clause in `_run_scan` described above.
- **Uses (imports only, no modification): `src/jaxcont/solvers/implicit.py`'s
  `differentiable_root`** — for the initial-refinement step described above.
- **Untouched:** `BifProblem`, `continuation()`'s signature, `core/scan_continuation.py`,
  `bifurcations/events.py`, `solvers/protocols.py`, `stability/floquet.py`,
  `bifurcations/period_doubling.py` (still stubs — next sub-projects).

## Testing

Per this session's established standard: empirical verification against a known answer, not just
design reasoning.

**Primary test system:** `r' = r·(ρ - r²), θ' = 1` in polar coordinates (converted to Cartesian
`(x, y) = (r·cos θ, r·sin θ)` for use as `f(u, p)` with `p = ρ`). At `ρ = 1` this has an **exact,
closed-form, hyperbolic limit cycle**: `x(t) = cos(t), y(t) = sin(t)`, period `T = 2π` —
independent of any external reference tool (BifurcationKit.jl, etc.), matching the standard this
session set for the `differentiable_root`/Event-protocol work.

1. **Convergence to the exact cycle:** given a deliberately coarse/perturbed initial trajectory
   guess (not the exact circle) and period guess (not exactly `2π`), `periodic_orbit_problem`'s
   internal refinement (step 5 above) converges `u0` to the exact cycle at `p0 = ρ = 1` (states
   within solver tolerance of `(cos τ·T, sin τ·T)` at each mesh/collocation point) and the
   refined period matches `2π` to solver tolerance — checked directly on the returned
   `BifProblem.u0`, no call to `continuation()` needed for this test.
2. **Continuation + `Fold()` false-positive check:** continue in `ρ` over a range with no actual
   fold-of-cycles (this system's limit cycle radius `√ρ` varies smoothly, no turning point) with
   `events=[Fold()]`; assert zero detections — a real, meaningful check (not just "does it not
   crash").
3. **`compute_stability` guard:** `continuation()` on a `periodic_orbit_problem`-built `BifProblem`
   with `settings.compute_stability=True` (the default) raises `ValueError` with a message
   pointing at `compute_stability=False`; with `compute_stability=False` it runs cleanly.
4. **Mesh-size scaling sanity:** the same system solved at two different `(ntst, ncol)` pairs
   converges to matching cycles (states agree to a tolerance consistent with the coarser mesh's
   expected discretization error) — a basic correctness check on the mesh-size parameterization,
   not a convergence-order study (out of scope).

## Global Constraints

- Zero changes to `BifProblem`'s fields/signature, `continuation()`'s signature, or
  `core/scan_continuation.py` — the whole point of the engine-reuse architecture.
- The one exception: a single guard clause in `_run_scan` (`api.py`) rejecting
  `compute_stability=True` for `kind="periodic"` problems with a clear `ValueError`.
- `ntst`/`ncol` are fixed for the lifetime of a continuation run (no adaptive mesh redistribution
  in this sub-project — a deliberate, documented scope cut, not an oversight).
- JaxCont does not integrate ODEs itself; `periodic_orbit_problem` only resamples a caller-supplied
  trajectory guess onto the collocation mesh.
- `periodic_orbit_problem` must return a `BifProblem` whose `u0` already satisfies `F(u0, p0) ≈ 0`
  (refined via `differentiable_root`) — `pseudo_arclength_scan`/`natural_scan` do not correct their
  starting point, so an unrefined `u0` would silently mark an unconverged point `converged=True`.
- `events=[Hopf()]` on a periodic problem is a documented footgun (not enforced) — out of scope to
  prevent in this sub-project.
- Floquet multipliers, period-doubling/Neimark–Sacker detection, and limit-cycle example scripts
  are explicitly out of scope — separate future sub-projects of the same v0.2.0 epic.
