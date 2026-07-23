# LinearSolver / EigenSolver Protocols Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every hardcoded `jnp.linalg.solve`/`jnp.linalg.eigvals` call in
the live continuation engine with a pluggable `LinearSolver`/`EigenSolver`
protocol boundary (ARCHITECTURE.md §4.6), with `Dense`/`DenseEigen` as the only
concrete implementations for now, exposed on `continuation()` via a `Solvers`
bundle.

**Architecture:** A new `src/jaxcont/solvers/protocols.py` module defines the two
`typing.Protocol`s and their `Dense`/`DenseEigen` defaults. `core/scan_continuation.py`'s
private helpers and public `pseudo_arclength_scan`/`natural_scan` entry points
gain a trailing, defaulted `linear_solver`/`eigen_solver` parameter that replaces
the hardcoded calls. `api.py` gains a `Solvers` bundle threaded through
`continuation()`'s `solvers=` parameter down to the engine call.

**Tech Stack:** JAX (`jax.jit` static arguments), Python `dataclasses`/`typing.Protocol`.

## Global Constraints

- `Dense()`/`DenseEigen()` must be numerically identical to today's hardcoded
  `jnp.linalg.solve`/`jnp.linalg.eigvals` calls — this is a pure refactor of the
  solve boundary, not a numerics change.
- All new function parameters are trailing and defaulted; no existing call site
  (library code, tests, or examples) may require modification to keep working.
- `Dense`/`DenseEigen` must be `@dataclass(frozen=True)` with no fields, to get
  value-based `__eq__`/`__hash__` and remain valid as `jax.jit` static arguments.
- `Solvers` is a plain Python-level dataclass (like `ContinuationPar`), never
  itself crossing a `jax.jit` boundary.
- No `NewtonSolver` protocol, no new example script, no changes to
  `src/jaxcont/solvers/implicit.py`, `src/jaxcont/stability/eigenvalue.py`,
  `src/jaxcont/stability/floquet.py`, or `src/jaxcont/bifurcations/period_doubling.py`
  in this plan.

---

### Task 1: `LinearSolver`/`EigenSolver` protocols and `Dense`/`DenseEigen`

**Files:**
- Create: `src/jaxcont/solvers/protocols.py`
- Modify: `src/jaxcont/solvers/__init__.py`
- Test: `tests/test_solver_protocols.py`

**Interfaces:**
- Produces: `LinearSolver` (Protocol, `__call__(self, A: Array, b: Array) -> Array`),
  `EigenSolver` (Protocol, `__call__(self, A: Array) -> Array`), `Dense`
  (frozen dataclass, no fields, implements `LinearSolver`), `DenseEigen`
  (frozen dataclass, no fields, implements `EigenSolver`) — all importable from
  `jaxcont.solvers.protocols` and from `jaxcont.solvers`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_solver_protocols.py`:

```python
"""
Tests for jaxcont.solvers.protocols: LinearSolver/EigenSolver structural
protocols and their Dense/DenseEigen default implementations -- see
docs/superpowers/specs/2026-07-23-linear-eigen-solver-protocols-design.md.
"""

import jax.numpy as jnp

from jaxcont.solvers.protocols import Dense, DenseEigen, EigenSolver, LinearSolver


def test_dense_matches_jnp_linalg_solve():
    A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
    b = jnp.array([3.0, 5.0])
    assert jnp.array_equal(Dense()(A, b), jnp.linalg.solve(A, b))


def test_dense_eigen_matches_jnp_linalg_eigvals():
    A = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    assert jnp.array_equal(DenseEigen()(A), jnp.linalg.eigvals(A))


def test_dense_instances_are_value_equal_and_hashable():
    # Required for Dense() to be safe as a jax.jit static argument: two
    # independently-constructed instances (not the same object) must
    # compare and hash equal, or every call with a fresh default would
    # force a spurious recompile.
    assert Dense() == Dense()
    assert hash(Dense()) == hash(Dense())


def test_dense_eigen_instances_are_value_equal_and_hashable():
    assert DenseEigen() == DenseEigen()
    assert hash(DenseEigen()) == hash(DenseEigen())


def test_dense_satisfies_linearsolver_protocol():
    assert isinstance(Dense(), LinearSolver)


def test_dense_eigen_satisfies_eigensolver_protocol():
    assert isinstance(DenseEigen(), EigenSolver)


def test_runtime_checkable_isinstance_cannot_distinguish_dense_from_eigensolver():
    # Dense.__call__ takes (A, b); EigenSolver's runtime_checkable isinstance
    # check only verifies the __call__ attribute exists (Protocol does not
    # check signatures at runtime) -- so isinstance(Dense(), EigenSolver) is
    # True even though calling it as an EigenSolver (one argument) would
    # TypeError at runtime. This test documents that limitation so a future
    # reader isn't surprised by it when relying on isinstance checks
    # elsewhere -- it is not a requirement being asserted, just a recorded
    # fact about Python's Protocol machinery.
    assert isinstance(Dense(), EigenSolver)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_solver_protocols.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jaxcont.solvers.protocols'`

- [ ] **Step 3: Create `src/jaxcont/solvers/protocols.py`**

```python
"""
Pluggable linear/eigen solver protocols (ARCHITECTURE.md §4.6).

Dense/DenseEigen are the only concrete implementations for now -- direct
LAPACK calls via jnp.linalg. The protocol boundary exists so a future
GMRES()/Arnoldi() (large systems) or ChebyshevDDE() (ARCHITECTURE.md §10.2,
the DDE eigensolver seam) can swap in without touching the continuation
loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array


@runtime_checkable
class LinearSolver(Protocol):
    """Solves ``A @ x = b`` for ``x``."""

    def __call__(self, A: Array, b: Array) -> Array: ...


@runtime_checkable
class EigenSolver(Protocol):
    """Returns the eigenvalues of ``A``."""

    def __call__(self, A: Array) -> Array: ...


@dataclass(frozen=True)
class Dense:
    """Default LinearSolver: ``jnp.linalg.solve``.

    No fields -- the dataclass machinery gives value-based __eq__/__hash__
    for free, which is required for this to be a safe jax.jit static
    argument (two independently-constructed Dense() instances must
    compare/hash equal, or every call with a fresh default would force a
    recompile).
    """

    def __call__(self, A: Array, b: Array) -> Array:
        return jnp.linalg.solve(A, b)


@dataclass(frozen=True)
class DenseEigen:
    """Default EigenSolver: ``jnp.linalg.eigvals``. No fields -- see Dense."""

    def __call__(self, A: Array) -> Array:
        return jnp.linalg.eigvals(A)
```

- [ ] **Step 4: Update `src/jaxcont/solvers/__init__.py`**

Read the current file first (`src/jaxcont/solvers/__init__.py`) -- it currently
reads:

```python
"""Numerical solvers (Newton, corrector methods)."""

from jaxcont.solvers.newton import NewtonSolver
from jaxcont.solvers.corrector import Corrector

__all__ = ["NewtonSolver", "Corrector"]
```

Replace it with:

```python
"""Numerical solvers (Newton, corrector methods)."""

from jaxcont.solvers.newton import NewtonSolver
from jaxcont.solvers.corrector import Corrector
from jaxcont.solvers.protocols import Dense, DenseEigen, EigenSolver, LinearSolver

__all__ = [
    "NewtonSolver",
    "Corrector",
    "LinearSolver",
    "EigenSolver",
    "Dense",
    "DenseEigen",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_solver_protocols.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/solvers/protocols.py src/jaxcont/solvers/__init__.py tests/test_solver_protocols.py
git commit -m "feat: add LinearSolver/EigenSolver protocols with Dense/DenseEigen defaults"
```

---

### Task 2: Wire the protocol into `core/scan_continuation.py`

**Files:**
- Modify: `src/jaxcont/core/scan_continuation.py`
- Test: `tests/test_solver_wiring.py`

**Interfaces:**
- Consumes: `LinearSolver`, `EigenSolver`, `Dense`, `DenseEigen` from
  `jaxcont.solvers.protocols` (Task 1).
- Produces: `_tangent(f, u, p, prev_tangent, linear_solver=Dense())`,
  `_newton_correct(f, u_pred, p_pred, u_prev, p_prev, du0, dp0, ds, tol, max_iter, linear_solver=Dense())`,
  `_natural_correct(f, u_pred, p_fixed, tol, max_iter, linear_solver=Dense())`,
  `branch_eigenvalues(f, states, params, eigen_solver=DenseEigen())`,
  `pseudo_arclength_scan(f, u0, p0, p_end, ds0, ds_min, ds_max, tol, max_steps, max_iter, linear_solver=Dense())`,
  `natural_scan(f, u0, p0, p_end, ds0, ds_min, ds_max, tol, max_steps, max_iter, linear_solver=Dense())`
  — `linear_solver` is the 11th positional parameter (index 10) on both scan
  functions, and both add index 10 to their `jax.jit` `static_argnums`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_solver_wiring.py`:

```python
"""
Tests proving the LinearSolver/EigenSolver protocol seam wired into
core/scan_continuation.py is real (routes actual calls through a supplied
solver), not decorative -- see
docs/superpowers/specs/2026-07-23-linear-eigen-solver-protocols-design.md.
"""

import jax
import jax.numpy as jnp

from jaxcont.core.scan_continuation import (
    branch_eigenvalues,
    natural_scan,
    pseudo_arclength_scan,
)
from jaxcont.solvers.protocols import Dense


def pitchfork(u, p):
    return jnp.array([p * u[0] - u[0] ** 3])


class _CountingLinearSolver:
    """Delegates to jnp.linalg.solve but records how many times it ran.

    Deliberately a plain class, not a frozen dataclass like Dense --
    LinearSolver implementations only need to be valid jax.jit static
    arguments (hashable + equality-comparable), which the default
    identity-based __eq__/__hash__ on a normal class already satisfies for
    a single-use, stateful test double. (A frozen dataclass with a list
    field would generate a __hash__ that crashes -- lists aren't hashable --
    the exact trap Dense's "no fields" design in Task 1 exists to avoid.)
    """

    def __init__(self):
        self.calls = []

    def __call__(self, A, b):
        self.calls.append(1)
        return jnp.linalg.solve(A, b)


class _CountingEigenSolver:
    def __init__(self):
        self.calls = []

    def __call__(self, A):
        self.calls.append(1)
        return jnp.linalg.eigvals(A)


_SCAN_ARGS = (
    pitchfork, jnp.array([0.1]), jnp.array(0.5), jnp.array(1.5),
    jnp.array(0.05), jnp.array(1e-5), jnp.array(0.2),
    jnp.array(1e-6), 60, jnp.array(20),
)


def test_pseudo_arclength_scan_routes_through_custom_linear_solver():
    solver = _CountingLinearSolver()
    res = pseudo_arclength_scan(*_SCAN_ARGS, solver)
    assert int(res.n_valid) > 1
    assert len(solver.calls) > 0


def test_natural_scan_routes_through_custom_linear_solver():
    solver = _CountingLinearSolver()
    res = natural_scan(*_SCAN_ARGS, solver)
    assert int(res.n_valid) > 1
    assert len(solver.calls) > 0


def test_branch_eigenvalues_routes_through_custom_eigen_solver():
    solver = _CountingEigenSolver()
    states = jnp.array([[0.5], [0.6]])
    params = jnp.array([1.0, 1.0])
    branch_eigenvalues(pitchfork, states, params, eigen_solver=solver)
    assert len(solver.calls) > 0


def test_pseudo_arclength_scan_accepts_independently_constructed_dense_instances():
    # Two separately-constructed Dense() instances (not the same object)
    # must be usable interchangeably as a jax.jit static argument -- no
    # recompile-related error, and identical results.
    res_a = pseudo_arclength_scan(*_SCAN_ARGS, Dense())
    res_b = pseudo_arclength_scan(*_SCAN_ARGS, Dense())
    assert jnp.array_equal(res_a.states, res_b.states)
    assert jnp.array_equal(res_a.params, res_b.params)


def test_pseudo_arclength_scan_vmap_with_explicit_linear_solver():
    def run(p0):
        return pseudo_arclength_scan(
            pitchfork, jnp.array([0.1]), p0, p0 + 1.0,
            jnp.array(0.05), jnp.array(1e-5), jnp.array(0.2),
            jnp.array(1e-6), 80, jnp.array(20), Dense(),
        )

    batch = jax.vmap(run)(jnp.linspace(0.5, 3.0, 16))
    assert batch.params.shape == (16, 81)
    assert batch.n_valid.shape == (16,)


def test_pseudo_arclength_scan_matches_pre_protocol_baseline():
    # Regression guard: captured from the unmodified (pre-Task-2) engine on
    # 2026-07-23 by running pseudo_arclength_scan directly on this exact
    # pitchfork problem. Dense() must reproduce these numbers exactly.
    res = pseudo_arclength_scan(*_SCAN_ARGS)
    n = int(res.n_valid)
    assert n == 9
    expected_params = [
        0.5, 0.5298426151275635, 0.6048426032066345, 0.7173426151275635,
        0.8860926032066345, 1.0860925912857056, 1.2860926389694214,
        1.4860926866531372, 1.686092734336853,
    ]
    expected_states0 = [
        0.10000000149011612, 9.813811630010605e-08, 8.424652264693577e-08,
        6.857676737581642e-08, 5.244454825970024e-08, 4.0607286422300604e-08,
        3.3129602172721206e-08, 2.7977623773267624e-08, 2.4212363669562365e-08,
    ]
    assert res.params[:n].tolist() == expected_params
    assert res.states[:n, 0].tolist() == expected_states0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_solver_wiring.py -v`
Expected: FAIL — `pseudo_arclength_scan()`/`natural_scan()`/`branch_eigenvalues()`
raise `TypeError` (too many positional arguments) or the `eigen_solver=`
keyword is rejected, since these parameters don't exist yet.

- [ ] **Step 3: Wire `linear_solver`/`eigen_solver` into `src/jaxcont/core/scan_continuation.py`**

First, add the import. In the existing import block near the top of the file
(currently ending `from jax import Array, jacfwd, lax`), add immediately
after it:

```python
from jaxcont.solvers.protocols import Dense, DenseEigen, EigenSolver, LinearSolver
```

Next, replace the `_tangent` function with:

```python
def _tangent(f, u, p, prev_tangent, linear_solver: LinearSolver = Dense()):
    """
    Keller tangent: the null vector of ``[df/du | df/dp]`` aligned with the
    previous tangent, normalized. Solving the bordered system with ``prev`` as
    the last row (rhs picking the last unit) both selects the null direction and
    orients it continuously — and stays well-posed at folds (where df/du alone
    is singular).
    """
    jac_u = jacfwd(f, argnums=0)(u, p)          # (n, n)
    df_dp = jacfwd(f, argnums=1)(u, p)          # (n,)
    M = _bordered_matrix(jac_u, df_dp, prev_tangent)
    rhs = jnp.zeros(u.shape[0] + 1).at[-1].set(1.0)
    t = linear_solver(M, rhs)
    t = t / jnp.linalg.norm(t)
    # keep continuous orientation
    t = jnp.where(jnp.dot(t, prev_tangent) < 0.0, -t, t)
    return t
```

Replace the `_newton_correct` function with:

```python
def _newton_correct(
    f, u_pred, p_pred, u_prev, p_prev, du0, dp0, ds, tol, max_iter,
    linear_solver: LinearSolver = Dense(),
):
    """
    Bordered Newton corrector for the pseudo-arclength system

        f(u, p) = 0
        g(u, p) = (u - u_prev)·du0 + (p - p_prev)·dp0 - ds = 0

    Solves the full (n+1, n+1) bordered system each iteration (well-posed through
    folds), inside a bounded ``while_loop`` with a finiteness guard. Returns
    ``(u, p, converged, iters)``; ``converged`` requires a finite, small residual.
    """

    def residual(u, p):
        f_val = f(u, p)
        g_val = jnp.dot(u - u_prev, du0) + (p - p_prev) * dp0 - ds
        return f_val, g_val

    def res_norm(f_val, g_val):
        return jnp.sqrt(jnp.sum(f_val ** 2) + g_val ** 2)

    def cond_fun(carry):
        _, _, it, done, _ = carry
        return jnp.logical_and(jnp.logical_not(done), it < max_iter)

    def body(carry):
        u, p, it, _, _ = carry
        f_val, g_val = residual(u, p)
        jac_u = jacfwd(f, argnums=0)(u, p)
        df_dp = jacfwd(f, argnums=1)(u, p)
        M = _bordered_matrix(jac_u, df_dp, jnp.concatenate([du0, dp0.reshape(1)]))
        rhs = -jnp.concatenate([f_val, g_val.reshape(1)])
        delta = linear_solver(M, rhs)
        u_new = u + delta[:-1]
        p_new = p + delta[-1]
        f_new, g_new = residual(u_new, p_new)
        r_new = res_norm(f_new, g_new)
        # stop iterating once converged OR the iterate went non-finite
        converged = r_new < tol
        blew_up = jnp.logical_not(jnp.isfinite(r_new))
        done = jnp.logical_or(converged, blew_up)
        return u_new, p_new, it + 1, done, r_new

    f0, g0 = residual(u_pred, p_pred)
    r0 = res_norm(f0, g0)
    init = (u_pred, p_pred, 0, r0 < tol, r0)
    u_f, p_f, it_f, _, r_f = lax.while_loop(cond_fun, body, init)
    converged = jnp.logical_and(
        r_f < tol,
        jnp.logical_and(jnp.all(jnp.isfinite(u_f)), jnp.isfinite(p_f)),
    )
    return u_f, p_f, converged, it_f
```

`_bordered_matrix` and `_adapt_ds` are unchanged — leave them as-is.

Replace the `pseudo_arclength_scan` function (including its `@partial(jax.jit, ...)`
decorator) with:

```python
@partial(jax.jit, static_argnums=(0, 8, 10))
def pseudo_arclength_scan(
    f: Callable[[Array, Array], Array],
    u0: Array,
    p0: Array,
    p_end: Array,
    ds0: Array,
    ds_min: Array,
    ds_max: Array,
    tol: Array,
    max_steps: int,
    max_iter: Array,
    linear_solver: LinearSolver = Dense(),
) -> ScanResult:
    """
    Continue ``f(u, p) = 0`` in ``p`` from ``(u0, p0)`` toward ``p_end``.

    ``f`` and ``max_steps`` are static (they set the compiled program & buffer
    sizes); everything else may be a traced array, so this whole function is
    ``jit``/``vmap``/``grad``-friendly. ``linear_solver`` is also static (a
    strategy object, not a traced value) -- see ``solvers/protocols.py``.
    """
    u0 = jnp.asarray(u0)
    n = u0.shape[0]
    dtype = u0.dtype
    p0 = jnp.asarray(p0, dtype)
    p_end = jnp.asarray(p_end, dtype)
    direction = jnp.sign(p_end - p0)

    # Initial tangent: seed prev with the parameter axis pointing in `direction`,
    # so the branch is traversed toward p_end.
    seed = jnp.zeros(n + 1, dtype).at[-1].set(direction)
    tan0 = _tangent(f, u0, p0, seed, linear_solver)

    # Fixed-size output buffers; slot 0 is the initial point.
    P = jnp.zeros((max_steps + 1, n), dtype).at[0].set(u0)
    Q = jnp.zeros((max_steps + 1,), dtype).at[0].set(p0)
    T = jnp.zeros((max_steps + 1, n + 1), dtype).at[0].set(tan0)
    C = jnp.zeros((max_steps + 1,), dtype=bool).at[0].set(True)

    ds_mag0 = jnp.asarray(ds0, dtype)
    D = jnp.zeros((max_steps + 1,), dtype).at[0].set(ds_mag0)

    class Carry(NamedTuple):
        u: Array
        p: Array
        tan: Array
        ds: Array         # positive magnitude; direction lives in the tangent
        idx: Array        # int; number of accepted points so far (write pointer)
        stop: Array       # bool
        P: Array
        Q: Array
        T: Array
        C: Array
        D: Array

    def cond_fun(c: Carry):
        return jnp.logical_and(c.idx < max_steps, jnp.logical_not(c.stop))

    def body(c: Carry):
        du0 = c.tan[:-1]
        dp0 = c.tan[-1]

        # Predict along the tangent, then correct.
        u_pred = c.u + c.ds * du0
        p_pred = c.p + c.ds * dp0
        u_new, p_new, converged, iters = _newton_correct(
            f, u_pred, p_pred, c.u, c.p, du0, dp0, c.ds, tol, max_iter, linear_solver
        )

        # New tangent only meaningful if we accept; compute anyway (branch-free).
        tan_new = _tangent(f, u_new, p_new, c.tan, linear_solver)

        write = c.idx + 1  # slot for the next accepted point
        P = c.P.at[write].set(jnp.where(converged, u_new, c.P[write]))
        Q = c.Q.at[write].set(jnp.where(converged, p_new, c.Q[write]))
        T = c.T.at[write].set(jnp.where(converged, tan_new, c.T[write]))
        C = c.C.at[write].set(converged)
        D = c.D.at[write].set(jnp.where(converged, c.ds, c.D[write]))

        # Accept -> advance state; reject -> stay put (and ds already shrinks).
        u = jnp.where(converged, u_new, c.u)
        p = jnp.where(converged, p_new, c.p)
        tan = jnp.where(converged, tan_new, c.tan)
        idx = c.idx + converged.astype(c.idx.dtype)

        ds = _adapt_ds(c.ds, iters, converged, ds_min, ds_max)

        # Stop conditions: reached p_end (after an accept), stalled at ds_min on a
        # failure, or the iterate went non-finite.
        reached = jnp.where(
            direction >= 0, p >= p_end, p <= p_end
        )
        stalled = jnp.logical_and(jnp.logical_not(converged), ds <= ds_min)
        nonfinite = jnp.logical_not(jnp.all(jnp.isfinite(u)))
        stop = jnp.logical_or(reached, jnp.logical_or(stalled, nonfinite))

        return Carry(u, p, tan, ds, idx, stop, P, Q, T, C, D)

    init = Carry(
        u=u0, p=p0, tan=tan0, ds=ds_mag0,
        idx=jnp.array(0, jnp.int32), stop=jnp.array(False),
        P=P, Q=Q, T=T, C=C, D=D,
    )
    final = lax.while_loop(cond_fun, body, init)

    return ScanResult(
        params=final.Q,
        states=final.P,
        tangents=final.T,
        converged=final.C,
        ds=final.D,
        n_valid=final.idx + 1,   # +1 for the initial point in slot 0
    )
```

Replace the `branch_eigenvalues` function with:

```python
def branch_eigenvalues(f, states, params, eigen_solver: EigenSolver = DenseEigen()):
    """
    Vectorized (vmap) eigenvalues of df/du along a stored branch. Kept out of the
    continuation loop so the loop stays simple; this is itself one batched kernel.
    """
    def eig_at(u, p):
        return eigen_solver(jacfwd(f, argnums=0)(u, p))
    return jax.vmap(eig_at)(states, params)
```

Replace the `_natural_correct` function with:

```python
def _natural_correct(f, u_pred, p_fixed, tol, max_iter, linear_solver: LinearSolver = Dense()):
    """
    Plain Newton on ``f(u, p_fixed) = 0`` with ``p_fixed`` held constant --
    no bordered system, no arclength constraint. This is natural
    continuation's corrector: because it has no extra degree of freedom to
    absorb an ill-conditioned ``df/du`` (unlike the bordered solve in
    ``_newton_correct``), it necessarily fails to converge at a fold, where
    ``df/du`` itself is singular -- by design, not a bug.
    """

    def cond_fun(carry):
        _, it, done, _ = carry
        return jnp.logical_and(jnp.logical_not(done), it < max_iter)

    def body(carry):
        u, it, _, _ = carry
        f_val = f(u, p_fixed)
        jac_u = jacfwd(f, argnums=0)(u, p_fixed)
        delta = linear_solver(jac_u, -f_val)
        u_new = u + delta
        f_new = f(u_new, p_fixed)
        r_new = jnp.sqrt(jnp.sum(f_new ** 2))
        converged = r_new < tol
        blew_up = jnp.logical_not(jnp.isfinite(r_new))
        done = jnp.logical_or(converged, blew_up)
        return u_new, it + 1, done, r_new

    f0 = f(u_pred, p_fixed)
    r0 = jnp.sqrt(jnp.sum(f0 ** 2))
    init = (u_pred, 0, r0 < tol, r0)
    u_f, it_f, _, r_f = lax.while_loop(cond_fun, body, init)
    converged = jnp.logical_and(r_f < tol, jnp.all(jnp.isfinite(u_f)))
    return u_f, converged, it_f
```

Replace the `natural_scan` function (including its `@partial(jax.jit, ...)`
decorator) with:

```python
@partial(jax.jit, static_argnums=(0, 8, 10))
def natural_scan(
    f: Callable[[Array, Array], Array],
    u0: Array,
    p0: Array,
    p_end: Array,
    ds0: Array,
    ds_min: Array,
    ds_max: Array,
    tol: Array,
    max_steps: int,
    max_iter: Array,
    linear_solver: LinearSolver = Dense(),
) -> ScanResult:
    """
    Continue ``f(u, p) = 0`` in ``p`` from ``(u0, p0)`` toward ``p_end``
    using natural (fixed-parameter) continuation: predict by incrementing
    ``p``, correct ``u`` via plain Newton with ``p`` held fixed. Cannot pass
    fold points -- a rejected step there shrinks ``ds`` toward ``ds_min`` and
    the loop terminates via the same ``stalled`` condition
    ``pseudo_arclength_scan`` uses, rather than hanging.

    Same fixed-size-buffer / jit / vmap contract as ``pseudo_arclength_scan``:
    ``f`` and ``max_steps`` are static; buffers are ``(max_steps + 1, ...)``.
    Returns the same :class:`ScanResult` shape -- ``tangents`` is zero-filled
    (natural continuation has no tangent concept) so both engines share one
    reassembly path in ``api.py``. ``linear_solver`` is static -- same
    contract as ``pseudo_arclength_scan``.
    """
    u0 = jnp.asarray(u0)
    n = u0.shape[0]
    dtype = u0.dtype
    p0 = jnp.asarray(p0, dtype)
    p_end = jnp.asarray(p_end, dtype)
    direction = jnp.sign(p_end - p0)

    P = jnp.zeros((max_steps + 1, n), dtype).at[0].set(u0)
    Q = jnp.zeros((max_steps + 1,), dtype).at[0].set(p0)
    T = jnp.zeros((max_steps + 1, n + 1), dtype)
    C = jnp.zeros((max_steps + 1,), dtype=bool).at[0].set(True)
    ds_mag0 = jnp.asarray(ds0, dtype)
    D = jnp.zeros((max_steps + 1,), dtype).at[0].set(ds_mag0)

    class Carry(NamedTuple):
        u: Array
        p: Array
        ds: Array
        idx: Array
        stop: Array
        P: Array
        Q: Array
        T: Array
        C: Array
        D: Array

    def cond_fun(c: Carry):
        return jnp.logical_and(c.idx < max_steps, jnp.logical_not(c.stop))

    def body(c: Carry):
        p_pred = c.p + direction * c.ds
        u_new, converged, iters = _natural_correct(f, c.u, p_pred, tol, max_iter, linear_solver)

        write = c.idx + 1
        P = c.P.at[write].set(jnp.where(converged, u_new, c.P[write]))
        Q = c.Q.at[write].set(jnp.where(converged, p_pred, c.Q[write]))
        C = c.C.at[write].set(converged)
        D = c.D.at[write].set(jnp.where(converged, c.ds, c.D[write]))

        u = jnp.where(converged, u_new, c.u)
        p = jnp.where(converged, p_pred, c.p)
        idx = c.idx + converged.astype(c.idx.dtype)

        ds = _adapt_ds(c.ds, iters, converged, ds_min, ds_max)

        reached = jnp.where(direction >= 0, p >= p_end, p <= p_end)
        stalled = jnp.logical_and(jnp.logical_not(converged), ds <= ds_min)
        nonfinite = jnp.logical_not(jnp.all(jnp.isfinite(u)))
        stop = jnp.logical_or(reached, jnp.logical_or(stalled, nonfinite))

        return Carry(u, p, ds, idx, stop, P, Q, c.T, C, D)

    init = Carry(
        u=u0, p=p0, ds=ds_mag0,
        idx=jnp.array(0, jnp.int32), stop=jnp.array(False),
        P=P, Q=Q, T=T, C=C, D=D,
    )
    final = lax.while_loop(cond_fun, body, init)

    return ScanResult(
        params=final.Q,
        states=final.P,
        tangents=final.T,
        converged=final.C,
        ds=final.D,
        n_valid=final.idx + 1,
    )
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `python -m pytest tests/test_solver_wiring.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the existing scan-engine tests to confirm no regressions**

Run: `python -m pytest tests/test_functional_api.py tests/test_bordered_newton.py -v`
Expected: all pass, same counts as before this task (these tests don't pass
`linear_solver` at all, exercising the new trailing-default-argument path).

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/core/scan_continuation.py tests/test_solver_wiring.py
git commit -m "refactor: route scan_continuation's solves through LinearSolver/EigenSolver"
```

---

### Task 3: `Solvers` bundle on `continuation()`, exports, and docs

**Files:**
- Modify: `src/jaxcont/api.py`
- Modify: `src/jaxcont/__init__.py`
- Modify: `notes/ARCHITECTURE.md`
- Test: `tests/test_solver_wiring.py` (append)

**Interfaces:**
- Consumes: `LinearSolver`, `EigenSolver`, `Dense`, `DenseEigen` from
  `jaxcont.solvers.protocols` (Task 1); `pseudo_arclength_scan`/`natural_scan`/
  `branch_eigenvalues`'s new `linear_solver`/`eigen_solver` parameters (Task 2).
- Produces: `Solvers` (frozen dataclass: `linear: LinearSolver = Dense()`,
  `eigen: EigenSolver = DenseEigen()`) importable from `jaxcont.api` and
  `jaxcont`; `continuation(..., solvers: Solvers = Solvers())`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_solver_wiring.py`:

```python
def test_continuation_routes_through_custom_solvers_bundle():
    # The spec-required proof that continuation() itself -- the public
    # entry point, not just the lower-level scan functions -- actually
    # routes through a user-supplied Solvers bundle.
    import jaxcont as jc

    linear_solver = _CountingLinearSolver()
    eigen_solver = _CountingEigenSolver()

    prob = jc.bif_problem(lambda u, p, args: pitchfork(u, p), u0=jnp.array([0.1]), p0=0.5)
    result = jc.continuation(
        prob, p_span=(0.5, 1.5),
        solvers=jc.Solvers(linear=linear_solver, eigen=eigen_solver),
    )

    assert result.branch.n_valid > 1
    assert len(linear_solver.calls) > 0
    assert len(eigen_solver.calls) > 0


def test_continuation_default_solvers_matches_prior_behavior():
    # Exercises the default Solvers() end to end through continuation() --
    # the public-API analogue of Task 2's pseudo_arclength_scan baseline
    # test. Settings are pinned to match that captured baseline exactly
    # (ds=0.05, ds_min=1e-5, ds_max=0.2, newton_tol=1e-6, max_steps=60,
    # newton_max_iter=20) rather than ContinuationPar's defaults, so the
    # same reference numbers apply.
    import jaxcont as jc

    prob = jc.bif_problem(lambda u, p, args: pitchfork(u, p), u0=jnp.array([0.1]), p0=0.5)
    settings = jc.ContinuationPar(
        ds=0.05, ds_min=1e-5, ds_max=0.2, max_steps=60,
        newton_tol=1e-6, newton_max_iter=20,
    )
    result = jc.continuation(prob, p_span=(0.5, 1.5), settings=settings)

    expected_params = [
        0.5, 0.5298426151275635, 0.6048426032066345, 0.7173426151275635,
        0.8860926032066345, 1.0860925912857056, 1.2860926389694214,
        1.4860926866531372, 1.686092734336853,
    ]
    assert result.branch.params.tolist() == expected_params
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_solver_wiring.py -v -k continuation_routes_through_custom_solvers_bundle`
Expected: FAIL — `TypeError: continuation() got an unexpected keyword argument 'solvers'`
(and `jc.Solvers` does not exist yet)

- [ ] **Step 3: Add the `Solvers` bundle and import in `src/jaxcont/api.py`**

In the existing import block at the top of the file, immediately after:

```python
from jaxcont.bifurcations.events import Event, Fold, Hopf, EventHit, detect_events
```

add:

```python
from jaxcont.solvers.protocols import Dense, DenseEigen, EigenSolver, LinearSolver
```

In the `__all__` list, add `"Solvers"` immediately after `"ContinuationPar"`:

```python
__all__ = [
    "BifProblem",
    "bif_problem",
    "continuation",
    "ContinuationPar",
    "Solvers",
    "ContinuationAlgorithm",
    "PseudoArclength",
    "Natural",
    "Event",
    "Fold",
    "Hopf",
    "EventHit",
    "Branch",
    "ContinuationResult",
]
```

Immediately after the `ContinuationPar` class definition (after its
`compute_stability: bool = True` field and closing of the class), add:

```python
@dataclass(frozen=True)
class Solvers:
    """Pluggable linear-algebra bundle for continuation() (ARCHITECTURE.md §4.6).

    Dense()/DenseEigen() are the only implementations today; the boundary
    exists so a future GMRES()/Arnoldi() (large systems) or ChebyshevDDE()
    (the DDE eigensolver seam, ARCHITECTURE.md §10.2) can swap in without
    touching the continuation loop. Never itself crosses a jax.jit boundary
    -- only its .linear/.eigen fields do, as static arguments to
    pseudo_arclength_scan/natural_scan.
    """

    linear: LinearSolver = Dense()
    eigen: EigenSolver = DenseEigen()
```

- [ ] **Step 4: Thread `solvers` through `_run_scan`, `_run_scan_traced`, and `continuation`**

Replace the `_run_scan` function signature and body with:

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

Replace the `_run_scan_traced` function signature and body with:

```python
def _run_scan_traced(
    res,
    rhs2: Callable[[Array, Array], Array],
    settings: ContinuationPar,
    events: Sequence[Event],
    solvers: Solvers,
) -> ContinuationResult:
    """
    ``_run_scan``'s path when ``res.n_valid`` is a tracer (called inside
    ``jax.vmap``/``jax.jit``). No concrete trim length exists, so the fixed-
    size engine buffers are returned as-is with a ``valid`` mask instead of
    the legacy ``ContinuationSolution``/``detect_events`` machinery,
    neither of which is traceable (Python loops, ``float()``, ``list.sort()``).
    """
    if len(events) > 0:
        raise NotImplementedError(
            "events=[...] is not supported when continuation() runs inside "
            "jax.vmap/jax.jit: detect_events uses Python-level control "
            "flow (loops, list.sort(), float()) that isn't traceable. Call "
            "continuation() without events inside the trace -- e.g. inspect "
            "branch.states/branch.params/branch.valid -- or run it eagerly "
            "per point of interest outside the trace to get events."
        )

    from jaxcont.core.scan_continuation import branch_eigenvalues

    states, params, tangents = res.states, res.params, res.tangents
    valid = jnp.arange(states.shape[0]) < res.n_valid

    eigenvalues = None
    stability = None
    if settings.compute_stability:
        eigenvalues = branch_eigenvalues(rhs2, states, params, eigen_solver=solvers.eigen)
        stability = jnp.all(jnp.real(eigenvalues) < 0.0, axis=1)

    branch = Branch(
        params=params,
        states=states,
        tangents=tangents,
        eigenvalues=eigenvalues,
        stable=stability,
        valid=valid,
    )
    return ContinuationResult(
        branch=branch, events=[], stats={"n_valid": res.n_valid}, _solution=None,
    )
```

Replace the `continuation` function with:

```python
def continuation(
    problem: BifProblem,
    alg: ContinuationAlgorithm = PseudoArclength(),
    *,
    p_span: Tuple[float, float],
    settings: ContinuationPar = ContinuationPar(),
    events: Sequence[Event] = (),
    solvers: Solvers = Solvers(),
    verbose: bool = False,
) -> ContinuationResult:
    """
    Continue a solution branch of ``problem`` across ``p_span``.

    Args:
        problem: the :class:`BifProblem` to continue.
        alg: :class:`PseudoArclength` (default) or :class:`Natural`.
        p_span: ``(p_start, p_end)`` range for the continuation parameter.
        settings: numerical settings (:class:`ContinuationPar`).
        events: detectors to run along the branch (e.g. ``[Fold(), Hopf()]``).
            An empty list disables detection.
        solvers: linear-algebra bundle (:class:`Solvers`); defaults to dense
            direct solves (``Dense()``/``DenseEigen()``).
        verbose: print a bifurcation summary.

    Returns:
        :class:`ContinuationResult` with ``.branch`` and ``.events``.
    """
    from jaxcont.core.scan_continuation import natural_scan, pseudo_arclength_scan

    if isinstance(alg, Natural):
        return _run_scan(natural_scan, problem, p_span, settings, events, solvers, verbose)
    elif isinstance(alg, PseudoArclength):
        return _run_scan(pseudo_arclength_scan, problem, p_span, settings, events, solvers, verbose)
    else:
        raise TypeError(f"Unknown continuation algorithm: {alg!r}")
```

- [ ] **Step 5: Update `src/jaxcont/__init__.py` exports**

In the existing `from jaxcont.api import (...)` block, add `Solvers`
immediately after `ContinuationPar`:

```python
from jaxcont.api import (
    BifProblem,
    bif_problem,
    continuation,
    ContinuationPar,
    Solvers,
    ContinuationAlgorithm,
    PseudoArclength,
    Natural,
    Event,
    Fold,
    Hopf,
    EventHit,
    Branch,
    ContinuationResult,
)
```

In the existing `# Solvers` import block, add the protocol imports immediately
after `from jaxcont.solvers.corrector import Corrector`:

```python
# Solvers
from jaxcont.solvers.newton import NewtonSolver
from jaxcont.solvers.corrector import Corrector
from jaxcont.solvers.protocols import Dense, DenseEigen, EigenSolver, LinearSolver
```

In `__all__`, add `"Solvers"` immediately after `"ContinuationPar"`, and add
the four protocol names to the `# Solvers` section:

```python
__all__ = [
    # Functional API (blessed surface)
    "BifProblem",
    "bif_problem",
    "continuation",
    "ContinuationPar",
    "Solvers",
    "ContinuationAlgorithm",
    "PseudoArclength",
    "Natural",
    "Event",
    "Fold",
    "Hopf",
    "EventHit",
    "Branch",
    "ContinuationResult",
    "fold_point",
    "fold_parameter",
    # Core
    "ContinuationProblem",
    "ContinuationSolution",
    # Problems
    "EquilibriumProblem",
    # Solvers
    "NewtonSolver",
    "Corrector",
    "LinearSolver",
    "EigenSolver",
    "Dense",
    "DenseEigen",
    # Stability
    "compute_eigenvalues",
    "analyze_stability",
    # Utilities
    "Config",
    "plot_bifurcation_diagram",
    "plot_continuation",
]
```

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `python -m pytest tests/test_solver_wiring.py -v`
Expected: 8 passed

- [ ] **Step 7: Update `notes/ARCHITECTURE.md` §4.6 to match the actual naming**

Read `notes/ARCHITECTURE.md` around line 287 first. Replace the `### 4.6
Solver protocols` section's code block (currently using a single `Dense()` for
both `linear` and `eigen` defaults):

```python
class LinearSolver(Protocol):
    def __call__(self, A, b) -> Array: ...     # Dense() default; GMRES()/BiCGStab() later
class EigenSolver(Protocol):
    def __call__(self, A) -> Array: ...        # Dense() default; Arnoldi()/shift-invert later
class NewtonSolver(Protocol): ...

class Solvers(eqx.Module):                     # bundle passed to continuation()
    linear: LinearSolver = Dense()
    eigen: EigenSolver = Dense()
    newton: NewtonSolver = BorderedNewton()
```

with:

```python
class LinearSolver(Protocol):
    def __call__(self, A, b) -> Array: ...     # Dense() default; GMRES()/BiCGStab() later
class EigenSolver(Protocol):
    def __call__(self, A) -> Array: ...        # DenseEigen() default; Arnoldi()/shift-invert later

@dataclass(frozen=True)                        # plain dataclass bundle, not eqx.Module --
class Solvers:                                 # never crosses a jax.jit boundary itself
    linear: LinearSolver = Dense()
    eigen: EigenSolver = DenseEigen()
```

(`NewtonSolver`/`BorderedNewton` removed from this sketch — not implemented as
part of the `Solvers` bundle; see `src/jaxcont/solvers/protocols.py` for the
implemented `LinearSolver`/`EigenSolver`/`Dense`/`DenseEigen`.)

- [ ] **Step 8: Run the full existing suite**

Run: `python -m pytest tests -v`
Expected: all tests pass, no failures or errors.

- [ ] **Step 9: Re-run the example scripts headlessly and confirm identical output**

Run:
```bash
MPLBACKEND=Agg python examples/example_01_pitchfork.py
MPLBACKEND=Agg python examples/example_02_lorenz.py
MPLBACKEND=Agg python examples/example_05_neural_mass.py
MPLBACKEND=Agg python examples/example_06_vmap_sweep.py
MPLBACKEND=Agg python examples/example_07_differentiable.py
```
Expected: all five exit 0. `example_02_lorenz.py`'s output must still include:
```
fold       F=1.5466         <-> bp/fold    F=1.546648
hopf       F=1.6197         <-> hopf       F=1.619658
hopf       F=2.4672         <-> hopf       F=2.467222
hopf       F=2.8599         <-> hopf       F=2.859876
```
`example_05_neural_mass.py`'s output must still include:
```
hopf       E0=-1.8501        <-> hopf   E0=-1.850125
fold       E0=-1.4630        <-> fold   E0=-1.463027
```

- [ ] **Step 10: Commit**

```bash
git add src/jaxcont/api.py src/jaxcont/__init__.py notes/ARCHITECTURE.md tests/test_solver_wiring.py
git commit -m "feat: expose Solvers bundle on continuation(), update ARCHITECTURE.md §4.6"
```
