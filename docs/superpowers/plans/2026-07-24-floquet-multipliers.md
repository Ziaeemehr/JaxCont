# Floquet Multipliers via Collocation Monodromy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute Floquet multipliers from the collocation monodromy matrix for periodic-orbit
branches, replacing the `compute_stability=True` guard clause with real computation wired into
`continuation()`'s existing stability pass.

**Architecture:** At a converged collocation branch point, the monodromy matrix `Φ(T)` is built as
a block linear recursion across the `ntst` mesh intervals — reusing the existing Lagrange
differentiation matrix `D` and right-endpoint weights `E` from `core/collocation.py`, and the raw
right-hand side's `df/du` Jacobian at each collocation point (`jax.jacfwd`). No re-integration of a
separate variational-equation IVP. `Φ(T)`'s eigenvalues (via the existing `EigenSolver` protocol)
are the Floquet multipliers. `api.py`'s `_run_scan`/`_run_scan_traced` dispatch their existing
stability-computation branch on `problem.kind`: `"equilibrium"` is untouched; `"periodic"` calls
the new Floquet path.

**Tech Stack:** JAX (`jacfwd`, `vmap`, `lax.scan`, `jnp.linalg.solve`/`eigvals` via the
`EigenSolver` protocol), reusing `core/collocation.py`'s existing `D`/`E` matrices.

## Global Constraints

- `BifProblem`'s fields/signature and `continuation()`'s signature are unchanged.
- No re-integration of a separate variational-equation IVP — Floquet multipliers are computed from
  the existing collocation structure only (numerically verified below).
- The `compute_stability=True` guard clause for periodic problems is removed entirely, not merely
  loosened — replaced by correct computation.
- No period-doubling/Neimark–Sacker event detection, no new example scripts, no changes to
  `core/scan_continuation.py`, `bifurcations/events.py`, or `solvers/protocols.py` in this plan.
- The monodromy block recursion needs **no** `jax.default_matmul_precision("float32")` fix. This
  was checked explicitly (unlike `periodic_orbit_problem`'s residual assembly, which needed one for
  its large `(ntst, ncol+1, n)` einsum) because the periodic-orbit-collocation sub-project found a
  real GPU TensorFloat32 bug there. The monodromy recursion's linear algebra is small per-interval
  blocks (`ncol*n x ncol*n`, e.g. 8×8 for `ntst=10`/`ncol=4`/`n=2`) and matched the closed-form
  answer cleanly with no fix, verified during design — do not add one speculatively.
- All code in this plan was prototyped and numerically verified before being written here, both in
  plain NumPy and in JAX (`jacfwd`/`vmap`/`lax.scan`/`jit`), against the closed-form circle example
  `r' = r(ρ-r²), θ'=1` at `ρ=1` (exact circle `x=cos(t), y=sin(t)`, `T=2π`, Floquet multipliers
  `{1, exp(-4π)} = {1, 3.4873423562089973e-06}`). NumPy prototype result: `[3.4873e-06, 0.999998]`.
  JAX/`jit` result: `[3.4570694e-06, 1.0000001]`. A `ρ` sweep (0.5, 1.0, 2.0, 4.0) confirmed
  `stable=True` at every point and the non-trivial multiplier tracking `exp(-4πρ)` down to the
  float32 underflow floor. Exact values from that verification appear in this plan's tests.

---

## Background: reading the existing code

Before starting, the engineer should know:

- `src/jaxcont/core/collocation.py` has `Collocation` (an `eqx.Module` with static `ntst`/`ncol`),
  `gauss_legendre_01`, `lagrange_diff_matrix`, `lagrange_eval_weights`, `collocation_matrices(ncol)
  -> (D, E, gauss, gw)`. `D` is the local `(ncol+1, ncol+1)` Lagrange differentiation matrix (local
  node 0 = the mesh point at the interval's left end, nodes 1..ncol = the interior Gauss-Legendre
  collocation points). `E` is the `(ncol+1,)` right-endpoint extrapolation weight vector.
- `src/jaxcont/problems/periodic.py`'s `periodic_orbit_problem(f, u_trajectory, t_trajectory,
  period0, p0, mesh) -> BifProblem` packs a periodic orbit's unknowns as a flat vector `U`: the
  first `ntst*n` entries are mesh-point states (`(ntst, n)` reshaped), the next `ntst*ncol*n`
  entries are collocation-point states (`(ntst, ncol, n)` reshaped), and the last entry is the
  period `T`. Currently, `BifProblem.args` is `(u_ref_coll, uref_prime_coll)` — the phase-condition
  reference data. This plan extends it to `(u_ref_coll, uref_prime_coll, raw_f, mesh)`.
- `src/jaxcont/stability/floquet.py` is currently a **dead, pre-v0.1 stub**:
  `compute_floquet_multipliers` integrates the variational equation via
  `scipy.integrate.solve_ivp` (non-jittable, and architecturally incompatible with the collocation
  representation — no continuous re-integrable trajectory exists there). This plan deletes it
  entirely and replaces it.
- `src/jaxcont/core/scan_continuation.py`'s `branch_eigenvalues(f, states, params,
  eigen_solver=DenseEigen())` is the existing equilibrium pattern this plan's periodic analogue
  mirrors: `jax.vmap(lambda u, p: eigen_solver(jacfwd(f, argnums=0)(u, p)))(states, params)`.
- `src/jaxcont/api.py`'s `_run_scan` (eager path) and `_run_scan_traced` (the `jax.vmap`/`jax.jit`
  path) each compute `eigenvalues`/`stability` from a branch's `states`/`params` — currently always
  via `branch_eigenvalues` and the equilibrium real-part condition
  (`jnp.all(jnp.real(eigenvalues) < 0.0, axis=1)`). This plan adds a `problem.kind` dispatch to
  both.

---

### Task 1: Monodromy matrix builder in `core/collocation.py`

**Files:**
- Modify: `src/jaxcont/core/collocation.py`
- Test: `tests/test_collocation.py`

**Interfaces:**
- Consumes: nothing new (uses `jax`, `jax.numpy` — not yet imported in this file — plus the
  existing `collocation_matrices`).
- Produces: `monodromy_matrix(raw_f, D, E, h, mesh_states, coll_states, T, p) -> Array` — an
  `(n, n)` array `Φ(T)`. `raw_f(u, p, args)` is the ODE right-hand side (same convention as
  `BifProblem.f`, called internally with `args=None`); `D`/`E` are `collocation_matrices(ncol)`'s
  first two outputs (as `jnp.ndarray`); `h = 1/ntst`; `mesh_states` is `(ntst, n)`; `coll_states`
  is `(ntst, ncol, n)`; `T`/`p` are scalars. Later tasks (Task 2) call this.

This is the block linear recursion: for each mesh interval `i`, the local collocation defect
equations (`D @ v - T*h*f(v) = 0` at each interior collocation point, where `v` stacks the
interval's mesh-point state and its `ncol` collocation-point states) are linearized around the
converged solution. That gives a linear map from a perturbation of the interval's mesh-point state
to perturbations of its `ncol` collocation-point states (solving one `(ncol*n, ncol*n)` linear
system per interval), and from there — via the same `E` extrapolation weights the continuity
equations use — a linear map `M_i` from this interval's mesh-point perturbation to the *next*
interval's. `Φ(T)` is the product of all `ntst` `M_i`'s.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_collocation.py` (add `import jax`, `import jax.numpy as jnp`, and the two new
imports from `jaxcont.problems.periodic`/`jaxcont.core.collocation` at the top alongside the
existing imports):

```python
import jax
import jax.numpy as jnp

from jaxcont.core.collocation import monodromy_matrix
from jaxcont.problems.periodic import periodic_orbit_problem


def _circle_rhs(u, p, args):
    x, y = u[0], u[1]
    r2 = x * x + y * y
    rho = p
    return jnp.array([(rho - r2) * x - y, (rho - r2) * y + x])


def _circle_periodic_problem():
    import numpy as np

    rng = np.random.default_rng(0)
    t_traj = np.sort(rng.uniform(0, 5.5, size=40))
    t_traj[0] = 0.0
    theta = lambda t: 2 * np.pi * t / 5.5 + 0.3
    u_traj = np.stack(
        [0.8 * np.cos(theta(t_traj)), 0.8 * np.sin(theta(t_traj))], axis=1
    )
    mesh = Collocation(ntst=10, ncol=4)
    return periodic_orbit_problem(
        _circle_rhs, jnp.asarray(u_traj), jnp.asarray(t_traj), 5.5, 1.0, mesh
    ), mesh


def test_monodromy_matrix_matches_closed_form_circle_multipliers():
    # r' = r*(rho - r^2), theta' = 1 at rho=1: exact circle, T=2*pi.
    # Closed-form Floquet multipliers {1, exp(-4*pi)} -- see
    # docs/superpowers/specs/2026-07-24-floquet-multipliers-design.md.
    # Verified during design: NumPy prototype gave [3.4873e-06, 0.999998],
    # JAX/jit gave [3.4570694e-06, 1.0000001] -- both float32-level
    # agreement with the exact exp(-4*pi) = 3.4873423562089973e-06.
    prob, mesh = _circle_periodic_problem()
    ntst, ncol, n = mesh.ntst, mesh.ncol, 2
    h = 1.0 / ntst

    D_np, E_np, _, _ = collocation_matrices(ncol)
    D, E = jnp.asarray(D_np), jnp.asarray(E_np)

    mesh_states = prob.u0[: ntst * n].reshape(ntst, n)
    coll_states = prob.u0[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
    T = prob.u0[-1]
    p = prob.p0

    Phi = monodromy_matrix(_circle_rhs, D, E, h, mesh_states, coll_states, T, p)
    assert Phi.shape == (n, n)

    multipliers = jnp.sort(jnp.abs(jnp.linalg.eigvals(Phi)))
    expected = jnp.sort(jnp.array([np.exp(-4 * np.pi), 1.0]))
    assert float(jnp.max(jnp.abs(multipliers - expected))) < 1e-5


def test_monodromy_matrix_is_jit_compatible():
    prob, mesh = _circle_periodic_problem()
    ntst, ncol, n = mesh.ntst, mesh.ncol, 2
    h = 1.0 / ntst
    D_np, E_np, _, _ = collocation_matrices(ncol)
    D, E = jnp.asarray(D_np), jnp.asarray(E_np)
    mesh_states = prob.u0[: ntst * n].reshape(ntst, n)
    coll_states = prob.u0[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
    T, p = prob.u0[-1], prob.p0

    jitted = jax.jit(monodromy_matrix, static_argnums=(0,))
    Phi = jitted(_circle_rhs, D, E, h, mesh_states, coll_states, T, p)
    assert Phi.shape == (n, n)
```

`import numpy as np` is already present at the top of `tests/test_collocation.py` — no need to
duplicate it, only add the `import jax`/`import jax.numpy as jnp` lines and the two new `from
jaxcont...` imports shown above.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_collocation.py -k monodromy -v`
Expected: FAIL with `ImportError: cannot import name 'monodromy_matrix'`

- [ ] **Step 3: Implement `monodromy_matrix` in `core/collocation.py`**

Add `import jax` and `import jax.numpy as jnp` to the top of `src/jaxcont/core/collocation.py`
(alongside the existing `import equinox as eqx` / `import numpy as np`), and append this function
at the end of the file:

```python
def monodromy_matrix(raw_f, D: "jnp.ndarray", E: "jnp.ndarray", h: float, mesh_states, coll_states, T, p):
    """``(n, n)`` monodromy matrix ``Phi(T)`` via a block linear recursion
    across the ``ntst`` mesh intervals, reusing the local differentiation
    matrix ``D`` and extrapolation weights ``E`` already built for the
    collocation residual (see ``collocation_matrices``). ``raw_f(u, p, args)``
    is the ODE right-hand side (``args=None`` internally) -- NOT the
    assembled collocation residual. No re-integration of a separate
    variational-equation IVP: this is a linearization of the same defect/
    continuity equations the residual already encodes, solved for
    sensitivity instead of state. Verified during design against the
    closed-form circle system's Floquet multipliers -- see
    docs/superpowers/specs/2026-07-24-floquet-multipliers-design.md."""
    ntst, n = mesh_states.shape
    ncol = coll_states.shape[1]
    eye_n = jnp.eye(n)

    def interval_map(mesh_state_i, coll_states_i):
        # Jacobian df/du at each of this interval's ncol collocation points.
        Jm = jax.vmap(jax.jacfwd(lambda u: raw_f(u, p, None)))(coll_states_i)  # (ncol, n, n)

        def build_A_row(m):
            def build_block(k):
                coeff = D[m + 1, k + 1]
                block = coeff * eye_n
                return jnp.where(k == m, block - T * h * Jm[m], block)

            return jax.vmap(build_block)(jnp.arange(ncol))

        A_blocks = jax.vmap(build_A_row)(jnp.arange(ncol))  # (ncol, ncol, n, n)
        A = jnp.transpose(A_blocks, (0, 2, 1, 3)).reshape(ncol * n, ncol * n)
        b0 = (-D[1:, 0][:, None, None] * eye_n[None, :, :]).reshape(ncol * n, n)
        S = jnp.linalg.solve(A, b0).reshape(ncol, n, n)
        return E[0] * eye_n + jnp.sum(E[1:][:, None, None] * S, axis=0)

    M_all = jax.vmap(interval_map)(mesh_states, coll_states)  # (ntst, n, n)
    Phi, _ = jax.lax.scan(lambda carry, M: (M @ carry, None), eye_n, M_all)
    return Phi
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_collocation.py -k monodromy -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full collocation test file**

Run: `python -m pytest tests/test_collocation.py -v`
Expected: all tests PASS (no regression to the existing Gauss-Legendre/Lagrange-matrix tests)

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/core/collocation.py tests/test_collocation.py
git commit -m "feat: add collocation monodromy matrix builder for Floquet multipliers"
```

---

### Task 2: `floquet_multipliers` and branch-batched variants in `stability/floquet.py`

**Files:**
- Modify (full rewrite): `src/jaxcont/stability/floquet.py`
- Test: `tests/test_floquet.py` (new)

**Interfaces:**
- Consumes: `monodromy_matrix` from `jaxcont.core.collocation` (Task 1), `DenseEigen`/`EigenSolver`
  from `jaxcont.solvers.protocols` (existing, unmodified), `collocation_matrices` from
  `jaxcont.core.collocation` (existing, unmodified).
- Produces:
  - `floquet_multipliers(raw_f, mesh, U, p, eigen_solver=DenseEigen()) -> Array` — `(n,)` Floquet
    multipliers at one branch point. `mesh: Collocation`; `U` is the flat collocation unknown
    vector (same layout as `BifProblem.u0` for a periodic problem).
  - `branch_floquet_multipliers(raw_f, mesh, states, params, eigen_solver=DenseEigen()) -> Array`
    — `(n_valid, n)`, `jax.vmap`'d over a branch's `states`/`params`. Mirrors
    `core/scan_continuation.branch_eigenvalues`'s batching convention. Task 4 calls this.
  - `floquet_stable(multipliers: Array) -> Array` — takes `(n_valid, n)` multipliers, returns
    `(n_valid,)` booleans: excludes the trivial multiplier (closest to `1`,
    `argmin(|multiplier - 1|)`) per row, then requires all *other* multipliers in that row to have
    magnitude `< 1`. This is the periodic-orbit analogue of the equilibrium real-part condition —
    a magnitude condition, not a half-plane condition. Task 4 calls this.

Current `src/jaxcont/stability/floquet.py` is a dead, pre-v0.1 stub (`compute_floquet_multipliers`
via `scipy.integrate.solve_ivp`, `analyze_periodic_orbit_stability`) — read it first to confirm,
then delete it entirely and replace with the content below.

- [ ] **Step 1: Write the failing test**

Create `tests/test_floquet.py`:

```python
"""
Tests for jaxcont.stability.floquet: Floquet multipliers via the collocation
monodromy matrix (core/collocation.py's monodromy_matrix), for the periodic
orbit r' = r*(rho - r^2), theta' = 1, which has an exact closed-form limit
cycle x=cos(t), y=sin(t), T=2*pi at rho=1, with exact Floquet multipliers
{1, exp(-4*pi)}. See
docs/superpowers/specs/2026-07-24-floquet-multipliers-design.md.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem
from jaxcont.stability.floquet import (
    branch_floquet_multipliers,
    floquet_multipliers,
    floquet_stable,
)


def _rhs(u, p, args):
    x, y = u[0], u[1]
    r2 = x * x + y * y
    rho = p
    return jnp.array([(rho - r2) * x - y, (rho - r2) * y + x])


def _coarse_wrong_trajectory():
    rng = np.random.default_rng(0)
    t_traj = np.sort(rng.uniform(0, 5.5, size=40))
    t_traj[0] = 0.0
    theta = lambda t: 2 * np.pi * t / 5.5 + 0.3
    u_traj = np.stack(
        [0.8 * np.cos(theta(t_traj)), 0.8 * np.sin(theta(t_traj))], axis=1
    )
    return jnp.asarray(u_traj), jnp.asarray(t_traj)


def _circle_problem(rho=1.0):
    u_traj, t_traj = _coarse_wrong_trajectory()
    mesh = Collocation(ntst=10, ncol=4)
    prob = periodic_orbit_problem(_rhs, u_traj, t_traj, 5.5, rho, mesh)
    return prob, mesh


def test_floquet_multipliers_matches_closed_form_at_rho_1():
    # Verified during design: JAX result [3.4570694e-06, 1.0000001] vs
    # exact {1, exp(-4*pi)} = {1, 3.4873423562089973e-06}.
    prob, mesh = _circle_problem(rho=1.0)
    multipliers = floquet_multipliers(_rhs, mesh, prob.u0, prob.p0)
    got = jnp.sort(jnp.abs(multipliers))
    expected = jnp.sort(jnp.array([np.exp(-4 * np.pi), 1.0]))
    assert float(jnp.max(jnp.abs(got - expected))) < 1e-5


def test_floquet_stable_true_at_rho_1_with_correct_trivial_multiplier():
    prob, mesh = _circle_problem(rho=1.0)
    multipliers = floquet_multipliers(_rhs, mesh, prob.u0, prob.p0)
    stable = floquet_stable(multipliers[None, :])
    assert bool(stable[0]) is True

    trivial_idx = int(jnp.argmin(jnp.abs(multipliers - 1.0)))
    assert abs(float(multipliers[trivial_idx].real) - 1.0) < 1e-4


def test_branch_floquet_multipliers_matches_per_point_calls():
    prob1, mesh = _circle_problem(rho=1.0)
    prob2, _ = _circle_problem(rho=1.5)
    states = jnp.stack([prob1.u0, prob2.u0])
    params = jnp.stack([prob1.p0, prob2.p0])

    batched = branch_floquet_multipliers(_rhs, mesh, states, params)
    individual = jnp.stack(
        [
            floquet_multipliers(_rhs, mesh, prob1.u0, prob1.p0),
            floquet_multipliers(_rhs, mesh, prob2.u0, prob2.p0),
        ]
    )
    assert batched.shape == (2, 2)
    assert jnp.allclose(jnp.sort(jnp.abs(batched), axis=1), jnp.sort(jnp.abs(individual), axis=1), atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_floquet.py -v`
Expected: FAIL with `ImportError: cannot import name 'floquet_multipliers'` (module still has the
old `compute_floquet_multipliers` stub)

- [ ] **Step 3: Rewrite `stability/floquet.py`**

Replace the entire content of `src/jaxcont/stability/floquet.py` with:

```python
"""
Floquet multipliers for periodic-orbit stability, via the collocation
monodromy matrix -- see
docs/superpowers/specs/2026-07-24-floquet-multipliers-design.md.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from jaxcont.core.collocation import Collocation, collocation_matrices, monodromy_matrix
from jaxcont.solvers.protocols import DenseEigen, EigenSolver

PyTree = Any


def floquet_multipliers(
    raw_f: Callable[[Array, Array, PyTree], Array],
    mesh: Collocation,
    U: Array,
    p: Array,
    eigen_solver: EigenSolver = DenseEigen(),
) -> Array:
    """Floquet multipliers (eigenvalues of the monodromy matrix ``Phi(T)``)
    at one converged periodic-orbit branch point. ``U`` is the flat
    collocation unknown vector (same layout as a periodic ``BifProblem``'s
    ``u0``: ``ntst`` mesh-point states, then ``ntst*ncol`` collocation-point
    states, then the period ``T``). ``raw_f`` is the ODE right-hand side
    (``args=None`` internally), not the assembled collocation residual.
    ``mesh`` has no ``n`` (state dimension) field, so it is derived
    algebraically from ``U``'s length."""
    ntst, ncol = mesh.ntst, mesh.ncol
    n = (U.shape[-1] - 1) // (ntst * (1 + ncol))
    h = 1.0 / ntst

    D_np, E_np, _, _ = collocation_matrices(ncol)
    D, E = jnp.asarray(D_np), jnp.asarray(E_np)

    mesh_states = U[: ntst * n].reshape(ntst, n)
    coll_states = U[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
    T = U[-1]

    Phi = monodromy_matrix(raw_f, D, E, h, mesh_states, coll_states, T, p)
    return eigen_solver(Phi)


def branch_floquet_multipliers(
    raw_f: Callable[[Array, Array, PyTree], Array],
    mesh: Collocation,
    states: Array,
    params: Array,
    eigen_solver: EigenSolver = DenseEigen(),
) -> Array:
    """Vectorized (vmap) Floquet multipliers along a stored periodic branch
    -- the periodic-orbit analogue of
    ``core.scan_continuation.branch_eigenvalues``."""
    def at(U, p):
        return floquet_multipliers(raw_f, mesh, U, p, eigen_solver)

    return jax.vmap(at)(states, params)


def floquet_stable(multipliers: Array) -> Array:
    """``(n_valid,)`` stability booleans from ``(n_valid, n)`` Floquet
    multipliers. A periodic orbit always has exactly one trivial multiplier
    equal to ``1`` (tangent to the flow) -- identified per-point as the one
    closest to ``1`` (``argmin(|multiplier - 1|)``) and excluded. Stability
    is a magnitude condition on the remaining multipliers (inside the unit
    circle), unlike equilibria's real-part condition."""
    def stable_at(row: Array) -> Array:
        trivial_idx = jnp.argmin(jnp.abs(row - 1.0))
        is_trivial = jnp.arange(row.shape[0]) == trivial_idx
        return jnp.all(jnp.where(is_trivial, True, jnp.abs(row) < 1.0))

    return jax.vmap(stable_at)(multipliers)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_floquet.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/stability/floquet.py tests/test_floquet.py
git commit -m "feat: replace scipy Floquet stub with collocation-monodromy floquet_multipliers"
```

---

### Task 3: Extend `periodic_orbit_problem`'s `args` with `raw_f`/`mesh`

**Files:**
- Modify: `src/jaxcont/problems/periodic.py`
- Test: `tests/test_periodic_orbit_problem.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `periodic_orbit_problem(...).args` is now the 4-tuple `(u_ref_coll, uref_prime_coll,
  raw_f, mesh)` (was `(u_ref_coll, uref_prime_coll)`). Task 4 unpacks this 4-tuple in `api.py`.

Two edits to `src/jaxcont/problems/periodic.py`, both inside `periodic_orbit_problem`:

1. The `residual` closure currently destructures `args` as a 2-tuple
   (`u_ref_coll, uref_prime_coll = args`) — this **must** change to a 4-tuple destructure, or every
   call to `residual` (including inside `differentiable_root`'s Newton loop, and every
   `continuation()` step) breaks with a `ValueError: too many values to unpack` once `args` grows.
   `residual` itself does not need `raw_f`/`mesh` — it already has `f`/`ntst`/`ncol` closed over
   directly as the factory's own parameters — so the extra two elements are ignored with `_`.
2. The `args` tuple built near the end of the function grows to include `f` and `mesh` (the
   factory's own parameters, already in scope — no new computation needed).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_periodic_orbit_problem.py` (below the existing tests, using the same
`_coarse_wrong_trajectory`/`_rhs` helpers already defined in that file):

```python
def test_periodic_orbit_problem_args_carries_raw_f_and_mesh():
    u_trajectory, t_trajectory = _coarse_wrong_trajectory()
    mesh = Collocation(ntst=10, ncol=4)
    prob = periodic_orbit_problem(_rhs, u_trajectory, t_trajectory, 5.5, 1.0, mesh)

    assert len(prob.args) == 4
    u_ref_coll, uref_prime_coll, raw_f, returned_mesh = prob.args
    assert raw_f is _rhs
    assert returned_mesh is mesh
    # Residual must still evaluate correctly with the extended args (this
    # exercises the fixed destructuring line inside residual()).
    r = prob.f(prob.u0, prob.p0, prob.args)
    assert float(jnp.linalg.norm(r)) < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_periodic_orbit_problem.py -k args_carries -v`
Expected: FAIL — either `ValueError: not enough values to unpack` (args is still a 2-tuple) when
`prob.args` is unpacked into 4 names, or (once Step 3's tuple-literal edit alone is made without
also fixing `residual`'s destructure) a `ValueError: too many values to unpack` from inside
`prob.f(...)`. Confirms both edits below are necessary, not just one.

- [ ] **Step 3: Make both edits in `periodic.py`**

In `src/jaxcont/problems/periodic.py`, change:

```python
    def residual(U: Array, p: Array, args: PyTree) -> Array:
        u_ref_coll, uref_prime_coll = args
```

to:

```python
    def residual(U: Array, p: Array, args: PyTree) -> Array:
        u_ref_coll, uref_prime_coll, _raw_f, _mesh = args
```

and change:

```python
    args: PyTree = (coll_guess, uref_prime_coll)
```

to:

```python
    args: PyTree = (coll_guess, uref_prime_coll, f, mesh)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_periodic_orbit_problem.py -v`
Expected: all PASS, including the new test and the three pre-existing ones (confirms the extended
`args` didn't break refinement/residual behavior)

- [ ] **Step 5: Run the periodic-orbit continuation integration tests too**

Run: `python -m pytest tests/test_periodic_orbit_continuation.py -v`
Expected: `test_compute_stability_false_runs_cleanly_for_periodic_problem` and
`test_fold_events_zero_false_positives_on_periodic_branch` still PASS (they don't inspect `args`
directly, but exercise `residual` through real `continuation()` runs — confirms the destructuring
fix didn't break the corrector). `test_compute_stability_true_raises_for_periodic_problem` still
PASSES too at this point (Task 4 removes the guard clause it tests; that test is replaced there,
not here).

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/problems/periodic.py tests/test_periodic_orbit_problem.py
git commit -m "feat: carry raw_f/mesh in periodic BifProblem.args for Floquet dispatch"
```

---

### Task 4: Wire Floquet dispatch into `api.py`, remove the guard clause

**Files:**
- Modify: `src/jaxcont/api.py`
- Modify: `tests/test_periodic_orbit_continuation.py`

**Interfaces:**
- Consumes: `branch_floquet_multipliers`, `floquet_stable` from `jaxcont.stability.floquet` (Task
  2); `problem.args == (u_ref_coll, uref_prime_coll, raw_f, mesh)` for `kind="periodic"` (Task 3).
- Produces: `continuation()` with `settings.compute_stability=True` now works for periodic
  problems: `Branch.eigenvalues` holds Floquet multipliers (shape `(n_valid, n)`, matching
  equilibrium's convention), `Branch.stable` uses the magnitude condition.

Read `src/jaxcont/api.py` in full first — it was modified in two prior sub-projects this session,
so line numbers below are illustrative anchors (match by the surrounding code shown, not by
assuming these exact line numbers still hold).

- [ ] **Step 1: Write the failing tests**

Replace `test_compute_stability_true_raises_for_periodic_problem` in
`tests/test_periodic_orbit_continuation.py` (the guard clause it tests is removed in Step 3 below)
with:

```python
def test_compute_stability_true_computes_floquet_multipliers_for_periodic_problem():
    # Regression for the removed guard clause: compute_stability=True is now
    # the whole point for periodic problems, not an error case. r' = r*(rho -
    # r^2), theta' = 1 has limit-cycle radius sqrt(rho) for all rho > 0, so
    # every branch point should read stable=True (exp(-4*pi*rho) < 1
    # always) -- see
    # docs/superpowers/specs/2026-07-24-floquet-multipliers-design.md.
    prob = _periodic_problem()
    sol = jc.continuation(
        prob, p_span=(1.0, 2.0),
        settings=jc.ContinuationPar(
            compute_stability=True, ds=0.05, max_steps=50, newton_tol=1e-5
        ),
    )
    assert sol.branch.n_valid > 1
    assert sol.branch.eigenvalues is not None
    assert sol.branch.eigenvalues.shape == (sol.branch.n_valid, 2)
    assert sol.branch.stable is not None
    assert bool(jnp.all(sol.branch.stable))


def test_compute_stability_true_default_still_works_for_equilibrium_problem_after_dispatch():
    # Same as the existing equilibrium regression test, run again after the
    # problem.kind dispatch is added, to confirm the equilibrium path
    # (real-part condition via branch_eigenvalues) is provably untouched.
    def pitchfork(u, p, args):
        return jnp.array([p * u[0] - u[0] ** 3])

    prob = jc.bif_problem(pitchfork, u0=jnp.array([0.1]), p0=0.5)
    sol = jc.continuation(prob, p_span=(0.5, 1.5))
    assert sol.branch.n_valid > 1
    assert sol.branch.eigenvalues is not None
    assert sol.branch.stable is not None
```

Add `import jax.numpy as jnp` to the top of `tests/test_periodic_orbit_continuation.py` if not
already present (it is — the file already imports `jax.numpy as jnp`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_periodic_orbit_continuation.py -v`
Expected: `test_compute_stability_true_computes_floquet_multipliers_for_periodic_problem` FAILS
(the old guard clause still raises `ValueError`); the other new test currently passes already
(dispatch not yet added, but equilibrium path is unaffected either way) — that's fine, it becomes a
real regression check once Step 3 lands.

- [ ] **Step 3: Edit `api.py`**

**3a. Remove the guard clause.** Find (near the top of `_run_scan`'s body):

```python
    if problem.kind == "periodic" and settings.compute_stability:
        raise ValueError(
            "settings.compute_stability=True is not supported for "
            "kind=\"periodic\" problems: the equilibrium stability pass "
            "eigendecomposes df/du, which for a periodic problem's f is "
            "the entire collocation Jacobian, not a meaningful dynamical "
            "quantity. Pass settings=ContinuationPar(compute_stability=False) "
            "instead. Floquet multipliers (the periodic-orbit analogue of "
            "stability) are a planned future feature, not yet implemented."
        )

    from jaxcont.core.scan_continuation import branch_eigenvalues
```

Replace with:

```python
    from jaxcont.core.scan_continuation import branch_eigenvalues
    from jaxcont.stability.floquet import branch_floquet_multipliers, floquet_stable
```

**3b. Dispatch the eager path's stability computation by `problem.kind`.** Find (inside
`_run_scan`, after `res = scan_fn(...)`):

```python
    eigenvalues = None
    stability = None
    want_eigs = settings.compute_stability or len(events) > 0
    if want_eigs and states.shape[0] > 0:
        eigenvalues = branch_eigenvalues(rhs2, states, params, eigen_solver=solvers.eigen)
        stability = jnp.all(jnp.real(eigenvalues) < 0.0, axis=1)
```

Replace with:

```python
    eigenvalues = None
    stability = None
    want_eigs = settings.compute_stability or len(events) > 0
    if want_eigs and states.shape[0] > 0:
        if problem.kind == "periodic":
            _, _, raw_f, mesh = problem.args
            eigenvalues = branch_floquet_multipliers(
                raw_f, mesh, states, params, eigen_solver=solvers.eigen
            )
            stability = floquet_stable(eigenvalues)
        else:
            eigenvalues = branch_eigenvalues(rhs2, states, params, eigen_solver=solvers.eigen)
            stability = jnp.all(jnp.real(eigenvalues) < 0.0, axis=1)
```

**3c. Thread `problem` into `_run_scan_traced`.** Find the call site inside `_run_scan`:

```python
        return _run_scan_traced(res, rhs2, settings, events, solvers)
```

Replace with:

```python
        return _run_scan_traced(res, problem, rhs2, settings, events, solvers)
```

Find `_run_scan_traced`'s signature:

```python
def _run_scan_traced(
    res,
    rhs2: Callable[[Array, Array], Array],
    settings: ContinuationPar,
    events: Sequence[Event],
    solvers: Solvers,
) -> ContinuationResult:
```

Replace with:

```python
def _run_scan_traced(
    res,
    problem: BifProblem,
    rhs2: Callable[[Array, Array], Array],
    settings: ContinuationPar,
    events: Sequence[Event],
    solvers: Solvers,
) -> ContinuationResult:
```

**3d. Dispatch `_run_scan_traced`'s stability computation by `problem.kind`.** Find (inside
`_run_scan_traced`):

```python
    eigenvalues = None
    stability = None
    if settings.compute_stability:
        eigenvalues = branch_eigenvalues(rhs2, states, params, eigen_solver=solvers.eigen)
        stability = jnp.all(jnp.real(eigenvalues) < 0.0, axis=1)
```

Replace with:

```python
    eigenvalues = None
    stability = None
    if settings.compute_stability:
        if problem.kind == "periodic":
            _, _, raw_f, mesh = problem.args
            eigenvalues = branch_floquet_multipliers(
                raw_f, mesh, states, params, eigen_solver=solvers.eigen
            )
            stability = floquet_stable(eigenvalues)
        else:
            eigenvalues = branch_eigenvalues(rhs2, states, params, eigen_solver=solvers.eigen)
            stability = jnp.all(jnp.real(eigenvalues) < 0.0, axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_periodic_orbit_continuation.py -v`
Expected: all PASS (5 tests: the 2 new ones from Step 1, plus
`test_compute_stability_false_runs_cleanly_for_periodic_problem`,
`test_compute_stability_true_default_still_works_for_equilibrium_problem`,
`test_fold_events_zero_false_positives_on_periodic_branch`)

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest -v`
Expected: all PASS, zero regressions — this confirms the `problem.kind` dispatch leaves the
equilibrium `compute_stability` path (the majority of the existing suite: fold/Hopf detection,
plain continuation, vmap sweeps) provably untouched.

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/api.py tests/test_periodic_orbit_continuation.py
git commit -m "feat: dispatch compute_stability to Floquet multipliers for periodic problems"
```

---

## Post-plan state

After this plan: `jc.continuation(periodic_prob, settings=jc.ContinuationPar(compute_stability=True,
...))` returns a `Branch` with real Floquet multipliers in `.eigenvalues` and the correct
magnitude-based `.stable` — matching `ARCHITECTURE.md` §6's original provisional sketch. Remaining
v0.2.0 epic items (not in scope here): period-doubling/Neimark–Sacker event detection (needs this
plan's multipliers to exist first), and limit-cycle example scripts.
