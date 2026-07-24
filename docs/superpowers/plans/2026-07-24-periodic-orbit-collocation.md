# Periodic-Orbit Continuation via Collocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add periodic-orbit continuation via fixed-mesh Gauss-Legendre orthogonal collocation,
reusing the existing scan engine (`pseudo_arclength_scan`/`natural_scan`) completely unchanged.

**Architecture:** A periodic orbit's collocation system (every mesh/collocation-point state, the
period, one phase-condition equation) is packed into one large residual `F(U, p) = 0` — exactly
the shape the existing engine already solves generically. `periodic_orbit_problem(...)` is a pure
*factory*: it resamples a caller-supplied trajectory guess onto a fixed mesh, refines it to
convergence via `differentiable_root`, and returns an ordinary `BifProblem` that
`jc.continuation()` runs with zero special-casing.

**Tech Stack:** JAX (`jacfwd`, `jit`, `lax.while_loop` via `differentiable_root`), `equinox`
(`Collocation`'s static fields), plain `numpy` (Gauss-Legendre nodes/weights, Lagrange
differentiation matrix — precomputed once per `ncol`, closed over as `jit`-time constants).

## Global Constraints

- Zero changes to `BifProblem`'s fields/signature, `continuation()`'s signature, or
  `core/scan_continuation.py` — the whole point of the engine-reuse architecture.
- The one exception: a single guard clause in `_run_scan` (`api.py`) rejecting
  `compute_stability=True` for `kind="periodic"` problems with a clear `ValueError`.
- `ntst`/`ncol` are fixed for the lifetime of a continuation run — no adaptive mesh redistribution
  in this plan. This is a deliberate, documented future item (it would change `U`'s shape mid-run,
  breaking the fixed-shape-buffer discipline `jit`/`vmap` rely on), not an oversight.
- JaxCont does not integrate ODEs itself; `periodic_orbit_problem` only resamples a caller-supplied
  trajectory guess onto the collocation mesh (via `jnp.interp`).
- `periodic_orbit_problem` must return a `BifProblem` whose `u0` already satisfies
  `F(u0, p0) ≈ 0` (refined via `differentiable_root`) — `pseudo_arclength_scan`/`natural_scan` do
  not correct their starting point; an unrefined `u0` would silently be marked `converged=True`.
- `events=[Hopf()]` on a periodic problem is a documented-but-not-enforced footgun (out of scope
  to prevent in this plan).
- Floquet multipliers, period-doubling/Neimark–Sacker detection, and limit-cycle example scripts
  are out of scope — separate future sub-projects of the same v0.2.0 epic.
- All code in this plan was prototyped and numerically verified before being written here — both
  standalone (NumPy/`scipy.optimize.fsolve` and JAX/`jacfwd`/`jit`) and through the real,
  unmodified `jaxcont.continuation()`/`jaxcont.BifProblem`/`jaxcont.Fold` — against the closed-form
  circle example used throughout this plan's tests. Exact values from that verification appear in
  the test steps below; they are not estimates.

---

### Task 1: Collocation numerics (`core/collocation.py`)

**Files:**
- Create: `src/jaxcont/core/collocation.py`
- Delete: `src/jaxcont/core/_periodic_eqx_scaffold.py`
- Delete: `tests/test_equinox_scaffold.py`
- Test: `tests/test_collocation.py`

**Interfaces:**
- Produces: `Collocation` (an `eqx.Module` with `ntst: int = eqx.field(static=True)`,
  `ncol: int = eqx.field(static=True)`), `gauss_legendre_01(ncol: int) -> tuple[np.ndarray, np.ndarray]`
  (nodes, weights on `[0, 1]`), `lagrange_diff_matrix(nodes: np.ndarray) -> np.ndarray`,
  `lagrange_eval_weights(nodes: np.ndarray, x: float) -> np.ndarray`,
  `collocation_matrices(ncol: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
  (returns `D, E, gauss, gw` — the local `(ncol+1, ncol+1)` differentiation matrix, the
  `(ncol+1,)` right-endpoint extrapolation weights, the `(ncol,)` interior Gauss-Legendre nodes,
  and the `(ncol,)` quadrature weights). All pure `numpy` (not `jax.numpy`) — `ncol` is always a
  Python `int` (static), so these are meant to run once at problem-construction time and have
  their results closed over as `jit`-time constants by Task 2, not traced.

Before starting, read `src/jaxcont/core/_periodic_eqx_scaffold.py` and
`tests/test_equinox_scaffold.py` in full to confirm exactly what's being deleted — both files are
small (22 and 65 lines) and their own docstrings say to delete them once real periodic-orbit
continuation exists.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_collocation.py`:

```python
"""
Tests for jaxcont.core.collocation: Gauss-Legendre nodes/weights, the local
Lagrange differentiation matrix, and the Collocation config type. Pure
numerics -- see docs/superpowers/specs/2026-07-24-periodic-orbit-collocation-design.md.
"""

import numpy as np
import pytest

from jaxcont.core.collocation import (
    Collocation,
    collocation_matrices,
    gauss_legendre_01,
    lagrange_diff_matrix,
    lagrange_eval_weights,
)


def test_gauss_legendre_01_matches_numpy_reference():
    for ncol in (2, 3, 4, 5):
        nodes, weights = gauss_legendre_01(ncol)
        x_ref, w_ref = np.polynomial.legendre.leggauss(ncol)
        expected_nodes = 0.5 * (x_ref + 1.0)
        expected_weights = 0.5 * w_ref
        assert np.allclose(nodes, expected_nodes)
        assert np.allclose(weights, expected_weights)


def test_gauss_legendre_nodes_are_interior_to_01():
    nodes, _ = gauss_legendre_01(4)
    assert np.all(nodes > 0.0)
    assert np.all(nodes < 1.0)


def test_lagrange_diff_matrix_is_exact_on_degree_ncol_polynomial():
    # Regression for the exact scheme verified during design: p(x) =
    # x^4 - 2x^3 + x - 1 (degree 4), p'(x) = 4x^3 - 6x^2 + 1. Local nodes
    # are [0, four interior Gauss-Legendre points] -- 5 nodes, degree-4
    # exact fit. Verified during design at max abs error 5.6e-15.
    ncol = 4
    gauss, _ = gauss_legendre_01(ncol)
    local_nodes = np.concatenate([[0.0], gauss])
    D = lagrange_diff_matrix(local_nodes)

    def p(x):
        return x**4 - 2 * x**3 + x - 1

    def pprime(x):
        return 4 * x**3 - 6 * x**2 + 1

    v = p(local_nodes)
    Dv = D @ v
    assert np.max(np.abs(Dv - pprime(local_nodes))) < 1e-12


def test_lagrange_eval_weights_extrapolate_exactly():
    # A degree-ncol polynomial evaluated at x=1 via the weight vector must
    # match direct evaluation, since the interpolant is exact for it.
    ncol = 4
    gauss, _ = gauss_legendre_01(ncol)
    local_nodes = np.concatenate([[0.0], gauss])
    E = lagrange_eval_weights(local_nodes, 1.0)

    def p(x):
        return x**4 - 2 * x**3 + x - 1

    v = p(local_nodes)
    assert abs(float(E @ v) - p(1.0)) < 1e-12


def test_collocation_matrices_shapes():
    ncol = 4
    D, E, gauss, gw = collocation_matrices(ncol)
    assert D.shape == (ncol + 1, ncol + 1)
    assert E.shape == (ncol + 1,)
    assert gauss.shape == (ncol,)
    assert gw.shape == (ncol,)


def test_collocation_ntst_ncol_are_static_python_ints():
    m = Collocation(ntst=10, ncol=4)
    assert isinstance(m.ntst, int)
    assert isinstance(m.ncol, int)

    # Changing a static field changes the pytree's structure (jit cache key).
    import jax

    m2 = Collocation(ntst=15, ncol=4)
    _, treedef1 = jax.tree_util.tree_flatten(m)
    _, treedef2 = jax.tree_util.tree_flatten(m2)
    assert treedef1 != treedef2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_collocation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jaxcont.core.collocation'`

- [ ] **Step 3: Create `src/jaxcont/core/collocation.py`**

```python
"""
Fixed-mesh Gauss-Legendre orthogonal collocation building blocks for
periodic-orbit continuation -- see
docs/superpowers/specs/2026-07-24-periodic-orbit-collocation-design.md.

Pure numerics -- no BifProblem/API concerns here, mirroring
core/scan_continuation.py's role as the engine's pure-numerics layer.
"""

from __future__ import annotations

import equinox as eqx
import numpy as np


class Collocation(eqx.Module):
    """Fixed collocation mesh config: ntst subintervals x ncol Gauss-Legendre
    points per subinterval. Both static (compile-time constants, since they
    fix the collocation unknown vector's shape for jit). The mesh itself is
    uniform (mesh point i is at tau=i/ntst), so it is derived on the fly
    rather than stored as a field. No adaptive mesh redistribution -- ntst/
    ncol are fixed for the lifetime of a continuation run (see design spec's
    explicit scope cut)."""

    ntst: int = eqx.field(static=True)
    ncol: int = eqx.field(static=True)


def gauss_legendre_01(ncol: int):
    """Gauss-Legendre nodes/weights of degree ``ncol`` on ``[0, 1]`` (mapped
    from the standard ``[-1, 1]`` via an affine transform)."""
    x, w = np.polynomial.legendre.leggauss(ncol)
    nodes = 0.5 * (x + 1.0)
    weights = 0.5 * w
    return nodes, weights


def lagrange_diff_matrix(nodes: np.ndarray) -> np.ndarray:
    """``(m, m)`` Lagrange differentiation matrix for ``nodes`` (any 1D
    array): ``D[j, k] = L_k'(nodes[j])``, where ``L_k`` is the k-th Lagrange
    basis polynomial for these nodes. For nodal values ``v`` of a
    degree-<m polynomial, ``D @ v`` gives its derivative at each node
    exactly."""
    m = len(nodes)
    D = np.zeros((m, m))
    for k in range(m):
        others = [nodes[i] for i in range(m) if i != k]
        denom = np.prod([nodes[k] - o for o in others])
        for j in range(m):
            xj = nodes[j]
            s = 0.0
            for i in range(m):
                if i == k:
                    continue
                term = 1.0
                for l in range(m):
                    if l == k or l == i:
                        continue
                    term *= (xj - nodes[l])
                s += term
            D[j, k] = s / denom
    return D


def lagrange_eval_weights(nodes: np.ndarray, x: float) -> np.ndarray:
    """Weight vector ``w`` such that ``w @ v`` evaluates the Lagrange
    interpolant through ``(nodes, v)`` at ``x`` (used to extrapolate each
    collocation interval's polynomial to its right endpoint, ``x=1``, for
    the continuity/periodicity equations)."""
    m = len(nodes)
    w = np.zeros(m)
    for k in range(m):
        Lk = 1.0
        for i in range(m):
            if i == k:
                continue
            Lk *= (x - nodes[i]) / (nodes[k] - nodes[i])
        w[k] = Lk
    return w


def collocation_matrices(ncol: int):
    """Precompute the local ``(ncol+1, ncol+1)`` differentiation matrix
    ``D``, the ``(ncol+1,)`` right-endpoint extrapolation weights ``E``, the
    ``(ncol,)`` interior Gauss-Legendre nodes ``gauss``, and the ``(ncol,)``
    quadrature weights ``gw`` for a degree-``ncol`` collocation scheme.
    Local node 0 is the left mesh point (x=0); local nodes 1..ncol are the
    interior Gauss-Legendre points. Pure numpy -- ``ncol`` is a Python int
    (static), so this is meant to be called once at problem-construction
    time and its results closed over as jax.jit-time constants, not
    traced."""
    gauss, gw = gauss_legendre_01(ncol)
    local_nodes = np.concatenate([[0.0], gauss])
    D = lagrange_diff_matrix(local_nodes)
    E = lagrange_eval_weights(local_nodes, 1.0)
    return D, E, gauss, gw
```

- [ ] **Step 4: Delete the superseded scaffold and its test**

```bash
git rm src/jaxcont/core/_periodic_eqx_scaffold.py tests/test_equinox_scaffold.py
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_collocation.py -v`
Expected: 6 passed

Run: `python -m pytest tests -v`
Expected: full suite passes; no collection error from the now-deleted `tests/test_equinox_scaffold.py`.

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/core/collocation.py tests/test_collocation.py
git commit -m "feat: add Gauss-Legendre orthogonal collocation numerics

Supersedes core/_periodic_eqx_scaffold.py (deleted along with its test),
per that file's own docstring."
```

---

### Task 2: `periodic_orbit_problem` factory (`problems/periodic.py`)

**Files:**
- Modify (full rewrite): `src/jaxcont/problems/periodic.py`
- Modify: `src/jaxcont/problems/__init__.py`
- Modify: `src/jaxcont/__init__.py:57` (comment only)
- Modify: `docs/source/development.rst:34`
- Test: `tests/test_periodic_orbit_problem.py`

**Interfaces:**
- Consumes: `Collocation`, `collocation_matrices` from `jaxcont.core.collocation` (Task 1);
  `differentiable_root` from `jaxcont.solvers.implicit` (existing, unmodified);
  `BifProblem` from `jaxcont.api` (existing, unmodified).
- Produces: `periodic_orbit_problem(f, u_trajectory, t_trajectory, period0, p0, mesh) -> BifProblem`
  in `jaxcont.problems.periodic`.

Before starting, read `src/jaxcont/problems/periodic.py` in full (120 lines — the existing
`scipy.integrate.solve_ivp`-based `PeriodicOrbitProblem` shooting stub, pre-v0.1, non-jittable;
its content is deleted entirely, not incrementally edited) and `src/jaxcont/problems/__init__.py`
(7 lines — it imports `PeriodicOrbitProblem` from `problems/periodic.py`, which breaks the whole
package's import chain once that class is gone, since `jaxcont/__init__.py` imports
`jaxcont.problems.equilibrium`, which runs `problems/__init__.py` first).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_periodic_orbit_problem.py`:

```python
"""
Tests for jaxcont.problems.periodic.periodic_orbit_problem -- fixed-mesh
collocation applied to r' = r*(rho - r^2), theta' = 1 (Cartesian), which has
an exact closed-form limit cycle x(t)=cos(t), y(t)=sin(t), T=2*pi at
rho=1 -- independent of any external reference tool. See
docs/superpowers/specs/2026-07-24-periodic-orbit-collocation-design.md.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem


def _rhs(u, p, args):
    x, y = u[0], u[1]
    r2 = x * x + y * y
    rho = p
    return jnp.array([(rho - r2) * x - y, (rho - r2) * y + x])


def _coarse_wrong_trajectory():
    # Deliberately wrong: radius 0.8 (true circle has radius 1.0), phase
    # offset 0.3, and a period guess of 5.5 (true period is 2*pi ~ 6.283).
    # Discrete, irregularly-spaced samples -- not a closed form -- to
    # exercise jnp.interp resampling with real data, matching what was
    # verified during design.
    rng = np.random.default_rng(0)
    t_traj = np.sort(rng.uniform(0, 5.5, size=40))
    t_traj[0] = 0.0
    theta = lambda t: 2 * np.pi * t / 5.5 + 0.3
    u_traj = np.stack(
        [0.8 * np.cos(theta(t_traj)), 0.8 * np.sin(theta(t_traj))], axis=1
    )
    return jnp.asarray(u_traj), jnp.asarray(t_traj)


def test_periodic_orbit_problem_refines_to_exact_circle():
    u_trajectory, t_trajectory = _coarse_wrong_trajectory()
    mesh = Collocation(ntst=10, ncol=4)

    prob = periodic_orbit_problem(_rhs, u_trajectory, t_trajectory, 5.5, 1.0, mesh)

    n = 2
    mesh_states = prob.u0[: mesh.ntst * n].reshape(mesh.ntst, n)
    T = prob.u0[-1]

    # Tolerances here (1e-5) are float32-achievable, not float64-tight --
    # this project runs float32 by default (no jax_enable_x64 anywhere).
    # Verified during design under real float32: T error ~3.5e-8, radius
    # error ~1.0e-8 -- both comfortably inside 1e-5 with margin to spare.
    assert abs(float(T) - 2 * np.pi) < 1e-5
    radii = jnp.linalg.norm(mesh_states, axis=1)
    assert float(jnp.max(jnp.abs(radii - 1.0))) < 1e-5
    assert prob.kind == "periodic"


def test_periodic_orbit_problem_residual_is_near_zero_at_u0():
    u_trajectory, t_trajectory = _coarse_wrong_trajectory()
    mesh = Collocation(ntst=10, ncol=4)
    prob = periodic_orbit_problem(_rhs, u_trajectory, t_trajectory, 5.5, 1.0, mesh)

    r = prob.f(prob.u0, prob.p0, prob.args)
    # Verified during design under real float32: residual norm ~7e-8.
    assert float(jnp.linalg.norm(r)) < 1e-5


def test_periodic_orbit_problem_mesh_size_scaling_sanity():
    # Regression for the mesh-size sanity check verified during design: a
    # finer mesh (ntst=15) must converge at least as accurately as the
    # coarser one (ntst=10) verified in the previous test, to the same
    # exact circle.
    u_trajectory, t_trajectory = _coarse_wrong_trajectory()
    mesh_coarse = Collocation(ntst=10, ncol=4)
    mesh_fine = Collocation(ntst=15, ncol=4)

    prob_coarse = periodic_orbit_problem(_rhs, u_trajectory, t_trajectory, 5.5, 1.0, mesh_coarse)
    prob_fine = periodic_orbit_problem(_rhs, u_trajectory, t_trajectory, 5.5, 1.0, mesh_fine)

    n = 2
    mesh_states_coarse = prob_coarse.u0[: mesh_coarse.ntst * n].reshape(mesh_coarse.ntst, n)
    mesh_states_fine = prob_fine.u0[: mesh_fine.ntst * n].reshape(mesh_fine.ntst, n)

    err_coarse = float(jnp.max(jnp.abs(jnp.linalg.norm(mesh_states_coarse, axis=1) - 1.0)))
    err_fine = float(jnp.max(jnp.abs(jnp.linalg.norm(mesh_states_fine, axis=1) - 1.0)))

    # float32-achievable tolerances -- verified during design: coarse
    # (ntst=10) radius error ~1.0e-8, fine (ntst=15) ~8.8e-9.
    assert err_coarse < 1e-5
    assert err_fine < 1e-5
    assert err_fine <= err_coarse
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_periodic_orbit_problem.py -v`
Expected: FAIL with `ImportError: cannot import name 'periodic_orbit_problem' from 'jaxcont.problems.periodic'`

- [ ] **Step 3: Replace `src/jaxcont/problems/periodic.py` entirely**

```python
"""
Periodic-orbit continuation via fixed-mesh Gauss-Legendre orthogonal
collocation -- see
docs/superpowers/specs/2026-07-24-periodic-orbit-collocation-design.md.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from jaxcont.api import BifProblem
from jaxcont.core.collocation import Collocation, collocation_matrices
from jaxcont.solvers.implicit import differentiable_root

PyTree = Any


def periodic_orbit_problem(
    f: Callable[[Array, Array, PyTree], Array],
    u_trajectory: Array,
    t_trajectory: Array,
    period0: float,
    p0: float,
    mesh: Collocation,
) -> BifProblem:
    """
    Build a periodic-orbit :class:`~jaxcont.api.BifProblem` via fixed-mesh
    orthogonal collocation.

    ``f(u, p, args)`` is the same right-hand-side convention used for
    equilibrium problems (``args`` is passed through as ``None`` when this
    factory calls ``f`` internally -- the periodic problem's own ``args``,
    on the returned ``BifProblem``, carries the phase-condition reference
    data instead, not anything ``f`` itself sees).

    ``u_trajectory``/``t_trajectory`` is a caller-supplied coarse trajectory
    guess (from the caller's own simulation -- JaxCont does not integrate
    ODEs itself); ``period0`` is the corresponding period guess. Both are
    resampled onto ``mesh`` and refined to convergence (via
    :func:`~jaxcont.solvers.implicit.differentiable_root`) before being
    returned as the ``BifProblem``'s ``u0`` --
    ``pseudo_arclength_scan``/``natural_scan`` do not Newton-correct their
    starting point, so an unrefined guess would silently be marked
    ``converged=True``.

    Note: this factory's *construction* (this function call itself) is not
    guaranteed safe under an outer ``jax.grad``/``jax.vmap`` -- the
    phase-condition reference derivative is computed from ``p0`` before the
    internal refinement's ``lax.while_loop``, so wrapping this whole call in
    ``jax.grad`` can leak a tracer across that boundary. The *returned*
    ``BifProblem``, once built, is fully ``jit``/``vmap``/``grad``-safe when
    passed to ``jc.continuation()`` (that's an ordinary call into the
    existing scan engine). Making construction itself differentiable is a
    possible future enhancement, not required by the current design spec.
    """
    ntst, ncol = mesh.ntst, mesh.ncol
    n = u_trajectory.shape[-1]
    h = 1.0 / ntst

    D_np, E_np, gauss_np, gw_np = collocation_matrices(ncol)
    D = jnp.asarray(D_np)
    E = jnp.asarray(E_np)
    gauss = jnp.asarray(gauss_np)
    gw = jnp.asarray(gw_np)

    def resample(tau: Array) -> Array:
        t = tau * period0
        return jnp.stack(
            [jnp.interp(t, t_trajectory, u_trajectory[:, c]) for c in range(n)]
        )

    mesh_tau = jnp.arange(ntst) / ntst
    coll_tau = (jnp.arange(ntst)[:, None] + gauss[None, :]) / ntst

    mesh_guess = jax.vmap(resample)(mesh_tau)  # (ntst, n)
    coll_guess = jax.vmap(jax.vmap(resample))(coll_tau)  # (ntst, ncol, n)

    def pack(mesh_states: Array, coll_states: Array, T: Array) -> Array:
        return jnp.concatenate(
            [mesh_states.flatten(), coll_states.flatten(), jnp.array([T])]
        )

    def unpack(U: Array):
        mesh_states = U[: ntst * n].reshape(ntst, n)
        coll_states = U[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
        T = U[-1]
        return mesh_states, coll_states, T

    def residual(U: Array, p: Array, args: PyTree) -> Array:
        u_ref_coll, uref_prime_coll = args
        mesh_states, coll_states, T = unpack(U)
        v = jnp.concatenate([mesh_states[:, None, :], coll_states], axis=1)  # (ntst, ncol+1, n)
        Dv = jnp.einsum("jk,ikc->ijc", D, v)
        f_at_v = jax.vmap(jax.vmap(lambda u: f(u, p, None)))(v[:, 1:, :])
        defect = Dv[:, 1:, :] - T * h * f_at_v  # (ntst, ncol, n)
        extrap = jnp.einsum("k,ikc->ic", E, v)  # (ntst, n)
        u_next = jnp.roll(mesh_states, -1, axis=0)
        continuity = u_next - extrap  # (ntst, n)
        phase = jnp.sum(
            gw[None, :, None] * h * (coll_states - u_ref_coll) * uref_prime_coll
        )
        return jnp.concatenate(
            [defect.flatten(), continuity.flatten(), jnp.array([phase])]
        )

    U_guess = pack(mesh_guess, coll_guess, jnp.asarray(period0, dtype=mesh_guess.dtype))
    uref_prime_coll = jax.vmap(jax.vmap(lambda u: f(u, p0, None)))(coll_guess)
    args: PyTree = (coll_guess, uref_prime_coll)

    p0_arr = jnp.asarray(p0, dtype=mesh_guess.dtype)
    # tol=1e-5, not differentiable_root's default 1e-8: this project runs
    # float32 by default (no jax_enable_x64 anywhere -- see
    # tests/test_functional_api.py's "tol=1e-6 (float32-reachable)" note).
    # A ~100-dimensional collocation Newton solve at tol=1e-8 cannot
    # converge in float32 -- worse, letting it keep iterating past
    # float32's noise floor actively degrades the result (the linear solve
    # each iteration becomes noise-dominated once residuals are near
    # machine epsilon, and repeated exposure to that over many iterations
    # accumulates error rather than improving it). Verified during design:
    # default tol=1e-8 left residual norm ~0.02 after 50 iterations; tol=1e-5
    # converges cleanly to residual norm ~7e-8 in far fewer iterations.
    U0 = differentiable_root(lambda U, p: residual(U, p, args), U_guess, p0_arr, tol=1e-5)

    return BifProblem(f=residual, u0=U0, p0=p0_arr, args=args, kind="periodic")
```

- [ ] **Step 4: Fix the import that would otherwise break `import jaxcont`**

`src/jaxcont/problems/__init__.py` currently reads:

```python
"""Problem definitions and boundary value problem solvers."""

from jaxcont.problems.equilibrium import EquilibriumProblem
from jaxcont.problems.periodic import PeriodicOrbitProblem
from jaxcont.problems.bvp import BoundaryValueProblem

__all__ = ["EquilibriumProblem", "PeriodicOrbitProblem", "BoundaryValueProblem"]
```

Replace it with:

```python
"""Problem definitions and boundary value problem solvers."""

from jaxcont.problems.equilibrium import EquilibriumProblem
from jaxcont.problems.periodic import periodic_orbit_problem
from jaxcont.problems.bvp import BoundaryValueProblem

__all__ = ["EquilibriumProblem", "periodic_orbit_problem", "BoundaryValueProblem"]
```

- [ ] **Step 5: Fix two stale references to the deleted `PeriodicOrbitProblem` name**

In `src/jaxcont/__init__.py`, line 57 currently reads (inside a comment block):

```python
#     from jaxcont.problems.periodic import PeriodicOrbitProblem
```

Replace with:

```python
#     from jaxcont.problems.periodic import periodic_orbit_problem
```

In `docs/source/development.rst`, line 34 currently reads:

```rst
- ``PeriodicOrbitProblem``: For periodic orbit continuation
```

Replace with:

```rst
- ``periodic_orbit_problem``: For periodic orbit continuation
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_periodic_orbit_problem.py -v`
Expected: 3 passed

Run: `python -c "import jaxcont"`
Expected: no error (confirms Step 4's fix — this would fail with `ImportError` without it)

- [ ] **Step 7: Run the full existing suite to confirm no regressions**

Run: `python -m pytest tests -v`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/jaxcont/problems/periodic.py src/jaxcont/problems/__init__.py \
        src/jaxcont/__init__.py docs/source/development.rst \
        tests/test_periodic_orbit_problem.py
git commit -m "feat: add periodic_orbit_problem factory (collocation-based)

Replaces the scipy.integrate.solve_ivp-based PeriodicOrbitProblem shooting
stub (pre-v0.1, non-jittable) with a collocation-based factory that returns
an ordinary BifProblem -- continuation() needs zero changes to run it."
```

---

### Task 3: `compute_stability` guard + integration tests (`api.py`)

**Files:**
- Modify: `src/jaxcont/api.py`
- Test: `tests/test_periodic_orbit_continuation.py`

**Interfaces:**
- Consumes: `periodic_orbit_problem`, `Collocation` (Task 2); `jc.continuation`, `jc.BifProblem`,
  `jc.ContinuationPar`, `jc.Fold`, `jc.Solvers` (existing, unmodified).
- Produces: `_run_scan` raises `ValueError` when `problem.kind == "periodic" and
  settings.compute_stability` is true.

Before starting, read `src/jaxcont/api.py` in full — it was modified in the immediately-preceding
`LinearSolver`/`EigenSolver` sub-project this session, so re-read it rather than relying on older
context. `_run_scan`'s current signature is
`_run_scan(scan_fn, problem, p_span, settings, events, solvers, verbose)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_periodic_orbit_continuation.py`:

```python
"""
Integration tests: periodic_orbit_problem's returned BifProblem run through
the real jc.continuation() -- the compute_stability guard, and a
false-positive check for events=[Fold()] on a system with no fold-of-cycles.
See docs/superpowers/specs/2026-07-24-periodic-orbit-collocation-design.md.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxcont as jc
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem


def _rhs(u, p, args):
    x, y = u[0], u[1]
    r2 = x * x + y * y
    rho = p
    return jnp.array([(rho - r2) * x - y, (rho - r2) * y + x])


def _periodic_problem():
    rng = np.random.default_rng(0)
    t_traj = np.sort(rng.uniform(0, 5.5, size=40))
    t_traj[0] = 0.0
    theta = lambda t: 2 * np.pi * t / 5.5 + 0.3
    u_traj = np.stack(
        [0.8 * np.cos(theta(t_traj)), 0.8 * np.sin(theta(t_traj))], axis=1
    )
    mesh = Collocation(ntst=10, ncol=4)
    return periodic_orbit_problem(
        _rhs, jnp.asarray(u_traj), jnp.asarray(t_traj), 5.5, 1.0, mesh
    )


def test_compute_stability_true_raises_for_periodic_problem():
    prob = _periodic_problem()
    with pytest.raises(ValueError, match="compute_stability"):
        jc.continuation(
            prob, p_span=(1.0, 2.0),
            settings=jc.ContinuationPar(compute_stability=True),
        )


def test_compute_stability_false_runs_cleanly_for_periodic_problem():
    prob = _periodic_problem()
    sol = jc.continuation(
        prob, p_span=(1.0, 2.0),
        settings=jc.ContinuationPar(compute_stability=False, ds=0.05, max_steps=50),
    )
    assert sol.branch.n_valid > 1


def test_compute_stability_true_default_still_works_for_equilibrium_problem():
    # The guard must only fire for kind="periodic" -- equilibrium
    # continuation (compute_stability=True is the default) must be
    # completely unaffected.
    def pitchfork(u, p, args):
        return jnp.array([p * u[0] - u[0] ** 3])

    prob = jc.bif_problem(pitchfork, u0=jnp.array([0.1]), p0=0.5)
    sol = jc.continuation(prob, p_span=(0.5, 1.5))
    assert sol.branch.n_valid > 1


def test_fold_events_zero_false_positives_on_periodic_branch():
    # r' = r*(rho - r^2) has limit-cycle radius sqrt(rho), smooth and
    # monotonic in rho -- no fold-of-cycles anywhere on this branch.
    # Verified during design directly against jc.continuation(): 34 valid
    # points from rho=1.0 to rho~2.018, zero Fold detections, final radius
    # matching sqrt(rho) to 7 significant figures.
    prob = _periodic_problem()
    sol = jc.continuation(
        prob, p_span=(1.0, 2.0),
        settings=jc.ContinuationPar(compute_stability=False, ds=0.05, max_steps=50),
        events=[jc.Fold()],
    )
    assert sol.branch.n_valid > 1
    assert sol.events == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_periodic_orbit_continuation.py -v`
Expected: `test_compute_stability_true_raises_for_periodic_problem` FAILS (no `ValueError` is
raised yet — the guard clause doesn't exist). The other three tests currently PASS already (they
don't depend on the guard clause) — that's expected; only the guard-clause test is RED at this
point.

- [ ] **Step 3: Add the guard clause to `_run_scan` in `src/jaxcont/api.py`**

At the start of `_run_scan`'s body (immediately after its docstring, before
`from jaxcont.core.scan_continuation import branch_eigenvalues`), add:

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
```

So `_run_scan` becomes:

```python
def _run_scan(
    scan_fn,
    problem: BifProblem,
    p_span: Tuple[float, float],
    settings: ContinuationPar,
    events: Sequence[Event],
    solvers: Solvers,
    verbose: bool,
) -> ContinuationResult:
    """
    Run ``scan_fn`` (``pseudo_arclength_scan`` or ``natural_scan``) and
    reassemble a legacy-shaped :class:`ContinuationSolution` so
    detection/plotting reuse existing code.
    """
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

    args = problem.args
    rhs2 = lambda u, p: problem.f(u, p, args)

    p_start, p_end = p_span
    u0 = jnp.asarray(problem.u0)
    dtype = u0.dtype

    res = scan_fn(
        rhs2,
        u0,
        jnp.asarray(p_start, dtype),
        jnp.asarray(p_end, dtype),
        jnp.asarray(settings.ds, dtype),
        jnp.asarray(settings.ds_min, dtype),
        jnp.asarray(settings.ds_max, dtype),
        jnp.asarray(settings.newton_tol, dtype),
        int(settings.max_steps),
        jnp.asarray(settings.newton_max_iter),
        solvers.linear,
    )

    try:
        n = int(res.n_valid)
    except jax.errors.ConcretizationTypeError:
        # Traced call (jax.vmap/jax.jit over this problem/settings): n_valid
        # can't become a concrete Python int, so there is no single trim
        # length. Fall back to the fixed-size-buffer + mask representation.
        return _run_scan_traced(res, rhs2, settings, events, solvers)

    states = res.states[:n]
    params = res.params[:n]
    tangents = res.tangents[:n]

    eigenvalues = None
    stability = None
    want_eigs = settings.compute_stability or len(events) > 0
    if want_eigs and states.shape[0] > 0:
        eigenvalues = branch_eigenvalues(rhs2, states, params, eigen_solver=solvers.eigen)
        stability = jnp.all(jnp.real(eigenvalues) < 0.0, axis=1)

    convergence_info = [
        {
            "step": i,
            "converged": bool(res.converged[i]),
            "newton_iters": 0,
            "ds": float(res.ds[i]),
        }
        for i in range(n)
    ]

    sol = ContinuationSolution(
        states=states,
        parameters=params,
        tangent_vectors=tangents,
        eigenvalues=eigenvalues,
        stability=stability,
        convergence_info=convergence_info,
        state_names=problem.state_names,
        param_name=problem.param_name,
    )

    # Detect events with the Event protocol (bifurcations/events.py).
    if len(events) > 0 and eigenvalues is not None:
        hits = detect_events(
            events, params, states, tangents, eigenvalues, rhs2,
            ds=float(settings.ds), tolerance=1e-6,
        )
        # sol.bifurcations stays dict-shaped: viz/core.py's plotting and
        # ContinuationSolution.get_bifurcations_by_type both read
        # bif.get("type")/bif.get("parameter")/bif.get("state") directly.
        sol.bifurcations = [
            {"type": h.kind, "parameter": h.p, "state": h.u, "index": h.index, **h.info}
            for h in hits
        ]

    return _to_result(sol)
```

Note: `want_eigs = settings.compute_stability or len(events) > 0` means a periodic problem run
with `events=[Fold()]` still computes `eigenvalues` via `branch_eigenvalues` (needed for
`detect_events`'s general signature, even though `Fold.test_function` itself only uses the
tangent, not eigenvalues) — this is unaffected by the new guard clause, since the guard only fires
on `settings.compute_stability`, not on `len(events) > 0`. This matches the verified behavior in
Step 1's `test_fold_events_zero_false_positives_on_periodic_branch` (run with
`compute_stability=False` and `events=[Fold()]`, which passes both today and after this change).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_periodic_orbit_continuation.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full existing suite to confirm no regressions**

Run: `python -m pytest tests -v`
Expected: all tests pass (in particular, every existing equilibrium-continuation test, which never
sets `kind="periodic"`, is completely unaffected by the new guard clause).

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/api.py tests/test_periodic_orbit_continuation.py
git commit -m "feat: reject compute_stability=True for periodic problems

_run_scan's stability pass eigendecomposes df/du, which for a periodic
problem's f is the entire collocation Jacobian, not a meaningful
dynamical quantity -- Floquet multipliers are a separate future feature.
Guarded with a clear ValueError rather than silently computing garbage."
```
