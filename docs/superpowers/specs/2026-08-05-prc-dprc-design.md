# Phase Response Curves (iPRC/dPRC) via Adjoint Method — Design Spec

**Status:** Approved for implementation planning.
**Roadmap item:** `## v0.3.0+ — Advanced (demand-driven)` (`notes/ROADMAP.md`) — new item, promoted from
`## v0.4.0+ — Explicitly out of scope`, where it previously read "a self-contained feature nobody
has asked for here yet." Someone asked; this spec moves it. Depends on periodic-orbit collocation
and the Floquet-multiplier monodromy machinery (both already shipped — see
[periodic-orbit collocation design](2026-07-24-periodic-orbit-collocation-design.md) and
[Floquet-multiplier design](2026-07-24-floquet-multipliers-design.md)).

## Motivation

The infinitesimal phase response curve (iPRC) — the gradient of asymptotic phase with respect to a
state perturbation, `Z(t) = ∇φ(x(t))` — is standard machinery in weakly-coupled-oscillator theory
and a MatCont specialty (manual §7.8). JaxCont already builds everything the adjoint method needs
as a side effect of Floquet-multiplier computation (`stability/floquet.py`): the monodromy matrix
`Φ(T)` and the per-interval Jacobian blocks it's assembled from. This spec computes `Z(t)` (iPRC)
and its parameter derivative `∂Z/∂p` (dPRC) as a small addition on top of that existing structure,
not a second numerical pathway.

## Scope

**In scope:**
- `prc_curve`: the iPRC `Z(t)` at the `ntst` mesh points of one converged periodic-orbit branch
  point, via the bordered-linear-system adjoint method (see Math below).
- `branch_prc`: the `vmap`-batched analogue across a stored periodic branch, mirroring
  `floquet.py`'s `branch_floquet_multipliers`.
- `dprc_curve`: `∂(prc_curve)/∂p`, via `jax.jacfwd` — free once `prc_curve` is built on
  differentiable primitives only (see Architecture).
- `plot_prc`: a `viz/portraits.py` plot of the curve, alongside `plot_eigenvalues`.
- One new example script demonstrating iPRC on an oscillator already used elsewhere in the repo.
- A refactor of `core/collocation.py:monodromy_matrix` to expose its per-interval propagator
  blocks (`interval_propagators`) as a standalone, reusable helper (behavior-preserving).
- **A MatCont cross-validation phase**, promoting `examples/MatCont`'s existing `US-PRC-001`
  registry entry from unsupported to a real, numerically-checked case — see "Validation phase"
  below. This is in scope, not a follow-up: the codim-2 spec's precedent
  ([2026-08-05-codim2-direct-solvers-design.md](2026-08-05-codim2-direct-solvers-design.md)) shows
  this project treats independent cross-validation as part of shipping a feature, not an optional
  extra.

**Out of scope (explicit):**
- **Collocation-point resolution.** `Z(t)` is returned at the `ntst` mesh points only, not at the
  internal Gauss-Legendre collocation points within each interval. No existing plotting or branch
  storage path in this codebase reconstructs periodic-orbit curves at collocation-point resolution
  either (checked `viz/phase_plane.py`, `viz/portraits.py` — neither does), so this is not a new
  scope cut relative to the rest of the codebase, just consistency with it.
- **MatCont's *direct* (non-adjoint) PRC method.** The adjoint method is the one this spec's
  existing infrastructure (monodromy matrix, per-interval Jacobian blocks) directly supports.
- **Homoclinic/heteroclinic PRC variants, coupling-function (interaction function) reduction,
  second-order phase reduction.** Genuinely separate, larger features.
- **`Event`/`events=[...]` integration, top-level export.** `prc_curve`/`branch_prc`/`dprc_curve`
  are importable from `jaxcont.stability.prc`, matching the deliberate non-export of
  `floquet_multipliers`/`branch_floquet_multipliers` (`__init__.py`'s existing note: periodic
  orbits overall aren't top-level yet).

## Architecture

**Core decision: reuse the monodromy-matrix machinery, add an adjoint pass on top — not a second
numerical pathway.** `stability/floquet.py`'s `floquet_multipliers` only needs `Φ(T)`'s
*eigenvalues*, so `core/collocation.py:monodromy_matrix` currently computes the per-interval
propagator blocks (`M_all`, shape `(ntst, n, n)`) purely as an internal step and immediately
`lax.scan`s them down to the single `(n, n)` matrix `Φ(T)`. The PRC curve needs those per-interval
blocks individually (to propagate an adjoint vector across each interval, not just to know the
composed endpoint map), so this spec factors them out:

```python
def interval_propagators(raw_f, D, E, h, mesh_states, coll_states, T, p) -> Array:
    """(ntst, n, n) per-interval propagator blocks M_i, s.t. Phi(T) = M_{ntst-1} @ ... @ M_0."""
    # body = monodromy_matrix's existing interval_map, vmapped — unchanged math, extracted.

def monodromy_matrix(raw_f, D, E, h, mesh_states, coll_states, T, p) -> Array:
    M_all = interval_propagators(raw_f, D, E, h, mesh_states, coll_states, T, p)
    Phi, _ = jax.lax.scan(lambda carry, M: (M @ carry, None), jnp.eye(M_all.shape[-1]), M_all)
    return Phi
```

This is a pure refactor — `monodromy_matrix`'s inputs, outputs, and numerics are unchanged; a
regression test (byte-identical Floquet multipliers before/after) guards this.

`stability/prc.py` (new module, sibling to `floquet.py`) then:
1. Calls `interval_propagators` to get `M_all` and chains them (same `lax.scan` composition
   `floquet.py` already uses via `monodromy_matrix`) to get `Φ(T)`.
2. Solves the bordered linear system for `Z(0)` (see Math) via the existing `LinearSolver`
   protocol (`solvers/protocols.py`) — no new solver abstraction.
3. Adjoint-propagates `Z(0)` backward across `M_all` (transposed) to fill in `Z` at every mesh
   point over the period.
4. `dprc_curve` differentiates through a **re-solve of `U(p)`**, not `prc_curve` at frozen `U` —
   see "Design findings from prototyping" below for why the originally-proposed
   `jax.jacfwd(prc_curve, argnums=3)` at fixed `U` was wrong, and the corrected approach:
   `dprc_curve(problem, ...)` takes the periodic orbit's `BifProblem` (residual `problem.f`, seed
   `problem.u0`, `problem.args`), reconverges `U(p)` via `differentiable_root` — the same primitive
   `problems/periodic.py:periodic_orbit_problem` already uses to build `U0` from a coarse guess —
   then differentiates the whole re-solve-plus-`prc_curve` composition. Two corrections found
   during implementation (Task 4's review independently reproduced both numerically, not just
   accepted the implementer's claim):
   - **`jax.jacrev`, not `jax.jacfwd`.** `differentiable_root` is built on `jax.custom_vjp`
     (reverse-mode only, no paired forward-mode rule) — `jax.jacfwd` raises `TypeError: can't apply
     forward-mode autodiff (jvp) to a custom_vjp function`. `jax.jacrev` is the only usable choice,
     not a stylistic swap.
   - **The phase-condition anchor must be recomputed from `p`, not frozen at `problem.args`.**
     `problem.args`'s `uref_prime_coll` is itself `f(u_ref_coll, p0, None)` — a function of
     whichever `p0` the original `periodic_orbit_problem` call used (`problems/periodic.py:136`).
     Reusing `problem.args` unchanged inside the re-solve silently keeps that anchor pinned to the
     *original* `p0` regardless of the perturbed `p`, which measurably disagrees with "what a fresh
     `periodic_orbit_problem` call would build at the perturbed `p`" (the finite-difference
     verification target below) by ~0.2 absolute — 20x the test's tolerance. Recomputing
     `uref_prime_coll = f(u_ref_coll, p, None)` inside the re-solve (mirroring
     `periodic_orbit_problem`'s own formula exactly) matches to ~1e-4. This differs from how
     `jc.continuation()`'s branch-stepping treats `args` (held fixed across an entire scan,
     `core/scan_continuation.py`/`api.py`) — that is a different code path answering a different
     question ("sensitivity along one fixed-anchor branch"), not what re-deriving `U(p)` from
     scratch at a perturbed `p` needs.

   ```python
   def dprc_curve(problem, linear_solver=Dense(), newton_tol=1e-5):
       u_ref_coll, _uref_prime_coll0, raw_f, mesh = problem.args

       def anchor_at(p):
           uref_prime_coll_p = jax.vmap(jax.vmap(lambda u: raw_f(u, p, None)))(u_ref_coll)
           return (u_ref_coll, uref_prime_coll_p, raw_f, mesh)

       def prc_at(p):
           U_p = differentiable_root(
               lambda U, pp: problem.f(U, pp, anchor_at(pp)), problem.u0, p, tol=newton_tol
           )
           return prc_curve(raw_f, mesh, U_p, p, linear_solver)

       return jax.jacrev(prc_at)(problem.p0)
   ```

   This still needs no eigendecomposition and still relies on the bordered-linear-system seed
   being differentiable — only the AD mode and the re-solve's own phase-anchor formula changed
   from the original draft above.

## Design findings from prototyping

Per this project's established discipline (and this spec's own Global Constraint requiring
numerical verification before finalizing an implementation plan — the same discipline the Floquet,
Hopf-normal-form, and codim-2 specs already applied), the recursion above was prototyped against
the closed-form circle system before this plan was written. One real, initially-wrong assumption
was found and corrected:

**`jax.jacfwd(prc_curve, argnums=p)` at a fixed collocation state `U` does not compute a
physically meaningful dPRC.** The original Architecture draft (directly above, now corrected)
assumed differentiating `prc_curve` alone — holding `U` fixed at its convergence for `p=p₀` — would
give `∂Z/∂p`, "free" because every primitive involved is differentiable. It *is* differentiable, and
matched a float64 central finite difference of the exact same (frozen-`U`) function to `1.4e-5` —
so the code was doing exactly what it was told. But the *value* it computed was wrong by roughly an
order of magnitude versus the true sensitivity (entries of magnitude `~1-6` where the real answer is
`~0.1-0.6`), because freezing `U` while perturbing `p` pushes the collocation state off the manifold
of genuine periodic orbits for that `p` — `Φ(T)` built there is a Jacobian-chain computation, but not
a classical monodromy matrix of any real trajectory, and its "eigenvalue-1" sensitivity is an
artifact, not a phase-response quantity. Confirmed two ways: (1) re-solving fresh, genuinely
converged periodic orbits at `ρ₀±ε` via independent `periodic_orbit_problem` calls and taking a
finite difference of `prc_curve` across *those* gave values matching the closed-form
`-Z(θ)/(2ρ) + (∂Z/∂θ)(dθ/dρ)` — including the circle system's own phase condition's small,
correctly-signed `θ`-drift with `ρ` — to `<0.01` absolute, versus `>6` disagreement for the
frozen-`U` version; (2) the fix (differentiate through a `differentiable_root`-based re-solve of
`U(p)`, not around it) is not new infrastructure — it's the exact primitive
`periodic_orbit_problem` already uses to build `u0` in the first place, applied one level up.
**Consequence for the API:** `dprc_curve` cannot share `prc_curve`/`branch_prc`'s
`(raw_f, mesh, U, p)` signature — it needs the residual function and Newton-seed `periodic_orbit_problem`
already assembles, so it takes the `BifProblem` itself (see API below).

## Math

At a converged branch point (state `U`, parameter `p`), unpack `U` exactly as `floquet_multipliers`
does: mesh-point states `u_0, ..., u_{ntst-1}`, collocation states, period `T`. Reuse
`interval_propagators` to get `M_0, ..., M_{ntst-1}`, each `(n, n)`, satisfying
`Φ(T) = M_{ntst-1} @ ... @ M_0` (forward composition — a state perturbation `δu` at mesh point `0`
propagates to `M_i @ ... @ M_0 @ δu` at mesh point `i+1`).

**Step 1 — seed `Z(0)`.** `Z(0)` is the left-eigenvector of `Φ(T)` for eigenvalue `1`
(equivalently, `(Φ(T)ᵀ − I) Z(0) = 0`), normalized by the standard adjoint-PRC condition
`Z(0) · f(x_0, p) = ω`, `ω = 2π / T`. Solved as one `(n+1) × (n+1)` bordered linear system (avoids
eigendecomposition entirely, and is well-posed because `1` is generically a simple eigenvalue of
`Φ(T)` for a hyperbolic-transverse periodic orbit — same genericity assumption Floquet-multiplier
stability classification already relies on):

```
[[Φ(T)ᵀ − I,  f(x_0, p)],   [Z(0)]     [0]
 [f(x_0, p)ᵀ,      0    ]] · [ λ  ]  =  [ω]
```

**Step 2 — fill in the rest of the period.** The adjoint/costate propagates backward relative to
the forward state-perturbation map: `Z_i = M_i^T @ Z_{i+1}`. Seed `Z_{ntst} := Z(0)` (mesh point
`ntst` wraps to mesh point `0`, i.e. `t = T`) and `lax.scan` backward through
`M_{ntst-1}^T, ..., M_0^T` to get `Z_{ntst-1}, ..., Z_0`. By construction, `Z_0` must equal `Z(0)`
again — periodicity, exactly what `(Φ(T)ᵀ − I) Z(0) = 0` encodes — which doubles as a numerical
sanity check during implementation.

**Verification target.** Reuses the Floquet spec's own system, no new reference model needed:
`r' = r·(ρ − r²), θ' = 1` has, at `ρ = 1`, the closed-form circular limit cycle `x = cos(t)`,
`y = sin(t)`, `T = 2π`, `ω = 1`. Because `θ' = 1` independent of `r` (no shear/isochron distortion —
isochrons are exactly the radial lines `θ = const`), the asymptotic phase is `φ = θ` exactly, so
`Z = ∇φ` in Cartesian coordinates is closed-form:

```
Z(θ) = (−sin θ, cos θ) / √ρ
```

(check: `Z(θ) · f(x,y)` at the limit cycle radius `r = √ρ` — `f = (−r sin θ, r cos θ)` since
`ṙ = 0` there — gives `sin²θ + cos²θ = 1 = ω`. ✓.) This also gives a closed-form dPRC check for
free: `∂Z/∂ρ = −Z / (2ρ)`.

**Verified against running code** (not merely derived): the bordered-solve seed and backward-chain
recursion above were prototyped directly against JaxCont's own `periodic_orbit_problem`/
`Collocation` machinery for the closed-form circle system at `ρ=1`. `Z` at every mesh point matched
`Z(θ) = (−sinθ, cosθ)/√ρ` (using each mesh point's *actual* converged phase `θᵢ = atan2(yᵢ, xᵢ)`,
not an assumed uniform grid — the phase condition anchors the branch's phase arbitrarily, so mesh
points are not at `θ = 2πi/ntst`) to `5.4e-7` max absolute error, and the periodicity identity
`Z₀ == Z(0)` (the backward chain landing back on its own seed) held to the same tolerance. See
"Design findings from prototyping" below for what this prototyping caught on the dPRC side.

## Validation phase (MatCont cross-check)

`examples/MatCont/cases.json` already registers this exact gap as `US-PRC-001` ("PRC and dPRC
calculations", `support: "unsupported"`, `manual_source: "MatCont 7.6 Testruns/testadaptPRC.m"`),
with a matching stub producer at `examples/MatCont/matlab/unsupported/run_prc_dprc.m` that only
asserts the MatCont output is finite, never exports it. This spec promotes that entry to a real,
numerically-compared case — `MC-PRC-001` — rather than leaving PRC/dPRC as the one shipped feature
in this codebase with no MatCont cross-check, which would be inconsistent with how every other
periodic-orbit/codim-2 feature here was validated.

**Reference system.** MatCont's own upstream test script
(`Testruns/testadaptPRC.m`, confirmed by reading the installed MatCont 7.6 tree at
`~/prog/MatCont/MatCont7p6`) exercises `Testruns/TestSystems/adaptx.m`:

```
x' = y
y' = z
z' = -alpha*z - beta*y - x + x²      (beta = 1 fixed, alpha the continuation parameter)
```

started from the origin, continued to a Hopf point, then a limit cycle (`ntst=20, ncol=4`), then a
second limit-cycle continuation with MatCont's `PRC`/`dPRC`/`Input` options enabled. MatCont packs
both curves into one "processor data" column: entries `22:102` are PRC, `103:183` are dPRC — 81
values each, i.e. `ntst*ncol + 1 = 81` (mesh **and** internal collocation points, MatCont's native
resolution).

**The resolution mismatch this surfaces.** This spec's `prc_curve` deliberately returns only the
`ntst` mesh points (see Scope), while MatCont's reference is at full `ntst*ncol+1` collocation
resolution. Matching every MatCont point exactly would require the collocation-point reconstruction
this spec explicitly scopes out. Instead, the validator resolves the mismatch the same way this
suite's shared comparison engine (`artifacts.py`'s `_compare_branch_rows`/`_interpolate_spectrum_roots`,
not per-case code) already resolves resolution mismatches for branches and spectra elsewhere:
`compare.py`'s `interpolate_observable` interpolates MatCont's finer reference curve onto JaxCont's
`ntst` mesh-point phase fractions `t/T` before comparing with `scaled_close`. `prc.py`'s
`run_adaptx_prc_dprc` produces CSV rows in the same shape `artifacts.py` already knows how to
compare (coordinate column + value columns), reusing that engine rather than writing a new
comparison path — confirmed by reading `artifacts.py` directly (`interpolate_observable`/
`scaled_close` are consumed there, not in any individual `python_cases/*.py` file today).

**What changes:**
- `matlab/unsupported/run_prc_dprc.m` moves to `matlab/run_prc_dprc.m` and stops being a stub: it
  writes `MC-PRC-001_prc.csv` (columns: mesh index, phase fraction `t/T`, PRC components, dPRC
  components) and `MC-PRC-001_metadata.json`, matching the `<case>_branch.csv`/`_metadata.json`
  convention every other `matlab/` producer already follows, instead of only asserting finiteness.
- New `examples/MatCont/python_cases/prc.py` (own file — mirrors `codim2.py` being its own file
  rather than folded into a topically-adjacent module): builds `adaptx` as a
  `jc.bif_problem`/`periodic_orbit_problem`, reaches the same Hopf → limit-cycle → parameter point
  MatCont's script reaches, computes `prc_curve`/`dprc_curve`, and compares against the committed
  reference CSV.
- `cases.json`: `US-PRC-001` becomes `MC-PRC-001`; `support` changes `"unsupported"` →
  `"supported"`; `unsupported_execution` is removed (only valid on unsupported cases, per
  `registry.py`'s own validation rule); `python` gains
  `"examples.MatCont.python_cases.prc:run_adaptx_prc_dprc"`; `references` gains the new CSV/JSON
  paths.
- `README.md`: the row moves from "Unsupported matrix" to "Supported coverage".

**Constraint:** the committed MatCont reference must come from an actual run against the real
MATLAB R2020a / MatCont 7.6 installation (`--regenerate-matcont`, per this suite's existing
workflow), not hand-edited — per this suite's documented policy, "numerical disagreements are
failures, not a reason to relax the registry tolerances."

## File layout

- **`src/jaxcont/core/collocation.py`**: extract `interval_propagators` out of the existing
  `monodromy_matrix` (behavior-preserving refactor — see Architecture).
- **New `src/jaxcont/stability/prc.py`**: `prc_curve(raw_f, mesh, U, p, linear_solver=Dense()) ->
  Array` (shape `(ntst, n)`), `branch_prc(raw_f, mesh, states, params, linear_solver=Dense()) ->
  Array` (shape `(n_valid, ntst, n)`, `vmap`-batched, mirroring `floquet.py`'s
  `branch_floquet_multipliers`). `dprc_curve(problem: BifProblem, linear_solver=Dense(),
  newton_tol=1e-5) -> Array` (shape `(ntst, n) + p.shape`) — deliberately **not**
  `(raw_f, mesh, U, p)`-shaped like its siblings; see "Design findings from prototyping": it needs
  `problem.f` (the residual) and `problem.u0` (the Newton seed) to reconverge `U(p)` via
  `differentiable_root` before differentiating, not just a frozen point. None exported from
  top-level `jaxcont/__init__.py` — matches the existing deliberate non-export of
  `floquet_multipliers`/`branch_floquet_multipliers`.
- **Modify `src/jaxcont/viz/portraits.py`**: add `plot_prc(curve, ...)`, alongside the existing
  `plot_eigenvalues`; add to `viz/__init__.py`'s exports.
- **New `examples/example_13_phase_response_curve.py`**: iPRC on an oscillator already used
  elsewhere in the repo (Van der Pol or the circle system above).
- **`examples/MatCont/`**: `matlab/unsupported/run_prc_dprc.m` → `matlab/run_prc_dprc.m` (rewritten
  to export data, not just assert); new `python_cases/prc.py`; `cases.json`'s `US-PRC-001` →
  `MC-PRC-001`; `README.md`'s coverage tables updated — see "Validation phase" above.
- **Untouched:** `BifProblem`'s fields/signature, `continuation()`'s signature,
  `core/scan_continuation.py`, `bifurcations/events.py`, `solvers/protocols.py` (consumed, not
  modified), `stability/floquet.py` (consumed via the shared `interval_propagators`, not modified
  beyond the internal refactor already covered above).

## Testing

Per this project's established standard: empirical verification against a known answer.

1. **Refactor regression test:** `interval_propagators`-based `monodromy_matrix` produces
   byte-identical (or float32-tolerance-identical) Floquet multipliers to the pre-refactor version,
   on the existing circle-system Floquet test case — confirms the extraction changed nothing.
2. **iPRC unit test:** at a converged branch point for the circle system at `ρ=1`, `prc_curve`
   matches `Z(θ) = (−sinθ, cosθ)` (mesh points sampled at their respective `θ` values) to
   float32-achievable tolerance (calibrated the same way the Floquet suite was — re-verify, don't
   assume the same tolerance transfers automatically).
3. **dPRC unit test:** at the same branch point, `dprc_curve` matches a finite difference built
   from two *independently re-converged* `periodic_orbit_problem` solves at `ρ±ε` (not the naive
   closed form `−Z/(2ρ)` alone — that omits the phase condition's own small `θ`-drift with `ρ`,
   confirmed during prototyping; see "Design findings from prototyping"). Tolerance `<0.01`
   absolute, matching what prototyping achieved with `ε=0.01`.
4. **Branch test:** `branch_prc` over a `ρ` sweep has shape `(n_valid, ntst, n)` and matches the
   closed form at every sampled point.
5. **Periodicity sanity check:** the backward-adjoint-chain `Z_0` (computed by propagating all the
   way around) matches the bordered-solve `Z(0)` seed to solver tolerance — the numerical
   expression of `(Φ(T)ᵀ − I)Z(0) = 0`.
6. Full existing suite green, confirming zero regression from the `monodromy_matrix` refactor.
7. **MatCont cross-validation (`MC-PRC-001`):** `python3 -m examples.MatCont.run_validation --case
   MC-PRC-001` passes — JaxCont's `prc_curve`/`dprc_curve` on the `adaptx` system, interpolated
   onto MatCont's reference phase points, agree within the case's registered tolerance. See
   "Validation phase" above.

## Global Constraints

- `BifProblem`'s fields/signature and `continuation()`'s signature are unchanged.
- No new solver abstraction — the bordered-system solve uses the existing `LinearSolver` protocol.
- No eigendecomposition (`jnp.linalg.eig`) anywhere in this feature — it has no JAX gradient rule
  for general non-symmetric matrices and would block dPRC, the entire reason the bordered-linear-
  system approach was chosen.
- `interval_propagators`'s extraction from `monodromy_matrix` must be verified as behavior-
  preserving (regression test) before any PRC-specific code is written on top of it.
- `Z(t)` is returned at `ntst` mesh-point resolution only — no collocation-point-resolution
  reconstruction in this sub-project.
- `prc_curve`/`branch_prc`/`dprc_curve` are not exported from top-level `jaxcont/__init__.py`,
  matching `floquet_multipliers`'s existing precedent.
- The adjoint recursion must be numerically prototyped and verified against the closed-form circle
  example (iPRC *and* dPRC) before being written into an implementation plan — not decided by
  derivation alone. **Done** — see "Design findings from prototyping"; iPRC matched closed-form
  to `5e-7`, and it caught the frozen-`U` dPRC mistake before it reached the plan.
- `dprc_curve` must differentiate through a re-solve of `U(p)` (via `differentiable_root`, the same
  primitive `periodic_orbit_problem` already uses), never through `prc_curve` alone at a fixed `U`
  — the latter is differentiable but not meaningful (see "Design findings from prototyping").
- `US-PRC-001` must be promoted to a real `MC-PRC-001` validation case (MatCont cross-check), not
  left unsupported — this feature does not ship without it. The MatCont reference data must come
  from an actual MATLAB/MatCont 7.6 run, never hand-edited.
