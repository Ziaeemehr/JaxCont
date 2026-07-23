# Differentiable-Root Primitive Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the Newton-in-`lax.while_loop` + `jax.custom_vjp` implicit-differentiation pattern currently bespoke to `src/jaxcont/bifurcations/fold_solve.py` into a reusable `differentiable_root(G, x0, theta) -> x*` primitive in `src/jaxcont/solvers/implicit.py`, so future extended-system events (Hopf, LPC, PD, NS) can reuse it instead of reimplementing `custom_vjp` from scratch.

**Architecture:** One new module, `src/jaxcont/solvers/implicit.py`, holds the generic solver. `fold_solve.py` is refactored to build its fold-specific `G` and initial guess, then delegate the solve to `differentiable_root` — its public API (`fold_point`, `fold_parameter`) and existing tests are unchanged from the outside. A new domain-independent test file validates the primitive on toy root-finding problems, separate from bifurcation theory.

**Tech Stack:** `jax`, `pytest`.

## Global Constraints

- `differentiable_root` is NOT exported from `jaxcont.__init__` or from `src/jaxcont/solvers/__init__.py`'s `__all__` — it's an internal building block, matching how `fold_solve.py`'s own helpers (`_make_fold_solver`, `newton`, `_pack`) are not exported today.
- No Hopf/LPC/PD/NS extended-system solver is built in this plan — only the primitive extraction and the fold refactor.
- No change to the Newton step's linear solve (stays hardcoded `jnp.linalg.solve`) — making it pluggable via a `LinearSolver` protocol is a separate, later task (v0.2 prep item 5).
- `fold_point`/`fold_parameter`'s public signatures, defaults, and docstrings do not change; `tests/test_functional_api.py::TestDifferentiableFold` must pass unmodified after the refactor.
- Reference spec: [docs/superpowers/specs/2026-07-23-differentiable-root-primitive-design.md](../specs/2026-07-23-differentiable-root-primitive-design.md).

---

### Task 1: Build the `differentiable_root` primitive and its tests

**Files:**
- Create: `src/jaxcont/solvers/implicit.py`
- Test: `tests/test_differentiable_root.py`

**Interfaces:**
- Produces: `differentiable_root(G, x0, theta, *, tol=1e-8, max_iter=50) -> Array`, importable as `from jaxcont.solvers.implicit import differentiable_root`. `G: Callable[[Array, PyTree], Array]` (residual function), `x0: Array | Callable[[PyTree], Array]` (fixed-shape Newton seed — plain `Array` if theta-independent, or `theta -> Array` if the seed depends on `theta`), `theta: PyTree` (differentiable parameters).

**Design note (why `x0` can be a callable):** a naive `x0: Array` (always precomputed by the caller before calling `differentiable_root`) breaks when the caller's seed genuinely depends on `theta` — closing a `theta`-derived tracer into the `custom_vjp`-wrapped Newton loop leaks a `LinearizeTracer` across `lax.while_loop`'s internal trace boundary (`jax.errors.UnexpectedTracerError`) during `jax.grad`. `fold_solve.py`'s real seed (`_initial_v`, SVD-based) is exactly this case. The fix: when `x0` is callable, `newton(theta)` evaluates it as its first step, so the `theta`-dependence is computed *inside* the traced primal instead of leaking in from outside — this is verified in Step 4 below and is why `test_callable_seed_theta_dependent` exists.

- [ ] **Step 1: Write the failing test**

Create `tests/test_differentiable_root.py`:

```python
"""
Domain-independent tests for jaxcont.solvers.implicit.differentiable_root,
extracted from bifurcations/fold_solve.py's implicit-diff Newton solver. See
docs/superpowers/specs/2026-07-23-differentiable-root-primitive-design.md.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxcont.solvers.implicit import differentiable_root


def _G_scalar(x, theta):
    # root: x* = sqrt(theta)
    return x**2 - theta


def test_root_matches_analytic():
    x0 = jnp.array([1.0])
    theta = jnp.array(4.0)
    x_star = differentiable_root(_G_scalar, x0, theta)
    assert float(x_star[0]) == pytest.approx(2.0, abs=1e-6)


def test_reverse_mode_grad_matches_analytic():
    x0 = jnp.array([1.0])

    def root(theta):
        return differentiable_root(_G_scalar, x0, theta)[0]

    for theta_val in (4.0, 9.0):
        theta = jnp.array(theta_val)
        g = jax.grad(root)(theta)
        expected = 1.0 / (2.0 * jnp.sqrt(theta))
        assert float(g) == pytest.approx(float(expected), abs=1e-5)


def _G_vector(x, theta):
    # x^2 - a*x + b = 0 ; theta = [a, b]
    a, b = theta[0], theta[1]
    return x**2 - a * x + b


def test_vector_theta_jacobian_matches_analytic():
    x0 = jnp.array([1.8])  # seeds convergence to the larger root
    theta = jnp.array([3.0, 2.0])  # roots at x=1, x=2

    def root(theta):
        return differentiable_root(_G_vector, x0, theta)[0]

    x_star = root(theta)
    assert float(x_star) == pytest.approx(2.0, abs=1e-6)

    J = jax.jacobian(root)(theta)
    # G_x = 2x - a ; G_a = -x ; G_b = 1 ; dx/da = x/(2x-a) ; dx/db = -1/(2x-a)
    Gx = 2.0 * float(x_star) - float(theta[0])
    expected = jnp.array([float(x_star) / Gx, -1.0 / Gx])
    assert jnp.allclose(J, expected, atol=1e-5)


def test_callable_seed_theta_dependent():
    # x0 depends on theta itself (mirrors fold_solve.py's SVD-based seed).
    # A precomputed-array x0 built the same way would leak a tracer under
    # jax.grad; the callable form computes it inside the traced primal.
    def x0_from_theta(theta):
        return jnp.array([jnp.sqrt(theta) * 0.5])

    def root(theta):
        return differentiable_root(_G_scalar, x0_from_theta, theta)[0]

    for theta_val in (4.0, 9.0):
        theta = jnp.array(theta_val)
        x_star = root(theta)
        assert float(x_star) == pytest.approx(float(jnp.sqrt(theta)), abs=1e-6)
        g = jax.grad(root)(theta)
        expected = 1.0 / (2.0 * jnp.sqrt(theta))
        assert float(g) == pytest.approx(float(expected), abs=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu pytest tests/test_differentiable_root.py -v`
Expected: FAIL (ERROR at collection) with `ModuleNotFoundError: No module named 'jaxcont.solvers.implicit'`

- [ ] **Step 3: Implement the primitive**

Create `src/jaxcont/solvers/implicit.py`:

```python
"""
Generic differentiable-root primitive: solve ``G(x, theta) = 0`` for ``x``
via Newton's method, differentiable in ``theta`` via the implicit function
theorem.

Extracted from ``bifurcations/fold_solve.py`` so every future extended-system
event (Hopf, LPC, PD, NS) can reuse this ``lax.while_loop`` Newton +
``jax.custom_vjp`` scaffolding instead of reimplementing it — see
docs/superpowers/specs/2026-07-23-differentiable-root-primitive-design.md.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array, lax

PyTree = Any


def differentiable_root(
    G: Callable[[Array, PyTree], Array],
    x0: Array | Callable[[PyTree], Array],
    theta: PyTree,
    *,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> Array:
    """
    Solve ``G(x, theta) = 0`` for ``x`` via Newton's method. The result is
    differentiable in ``theta`` via the implicit function theorem
    (``dx*/dtheta = -G_x^-1 G_theta``), so reverse-mode
    ``jax.grad``/``jax.jacobian`` does not differentiate through the inner
    ``lax.while_loop``.

    ``x0`` seeds Newton's method and does not itself receive a gradient
    (the root ``x*`` is uniquely determined by ``G(x*, theta) = 0`` near the
    seed, independent of how the seed was chosen). Pass a plain ``Array``
    for a ``theta``-independent seed. Pass a callable ``theta -> Array`` if
    the seed depends on ``theta`` (e.g. built from a ``theta``-dependent
    Jacobian) — computing such a seed *outside* this function and passing
    the resulting array in would leak a tracer across ``lax.while_loop``'s
    trace boundary under ``jax.grad``; the callable form evaluates it inside
    the traced primal instead, where it's safe.
    """

    def newton(theta):
        x_seed = x0(theta) if callable(x0) else x0

        def cond(carry):
            x, it, done = carry
            return jnp.logical_and(jnp.logical_not(done), it < max_iter)

        def body(carry):
            x, it, _ = carry
            r = G(x, theta)
            J = jax.jacobian(G, argnums=0)(x, theta)
            dx = jnp.linalg.solve(J, -r)
            x_new = x + dx
            r_new = G(x_new, theta)
            done = jnp.logical_or(
                jnp.linalg.norm(r_new) < tol,
                jnp.logical_not(jnp.all(jnp.isfinite(r_new))),
            )
            return x_new, it + 1, done

        x_star, _, _ = lax.while_loop(cond, body, (x_seed, 0, jnp.array(False)))
        return x_star

    @jax.custom_vjp
    def solve(theta):
        return newton(theta)

    def solve_fwd(theta):
        x_star = newton(theta)
        return x_star, (x_star, theta)

    def solve_bwd(res, x_bar):
        x_star, theta = res
        # implicit function theorem: dx*/dtheta = -G_x^-1 G_theta
        #   theta_bar = -(G_theta)^T G_x^T^-1 x_bar
        Gx = jax.jacobian(G, argnums=0)(x_star, theta)
        y = jnp.linalg.solve(Gx.T, x_bar)
        _, vjp_theta = jax.vjp(lambda t: G(x_star, t), theta)
        (theta_bar,) = vjp_theta(-y)
        return (theta_bar,)

    solve.defvjp(solve_fwd, solve_bwd)
    return solve(theta)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu pytest tests/test_differentiable_root.py -v`
Expected: PASS (all 4 tests, including `test_callable_seed_theta_dependent` — this is the case that would raise `jax.errors.UnexpectedTracerError` without the `callable(x0)` handling above)

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/solvers/implicit.py tests/test_differentiable_root.py
git commit -m "feat: extract differentiable_root primitive into solvers/implicit.py"
```

---

### Task 2: Refactor `fold_solve.py` onto `differentiable_root`

**Files:**
- Modify: `src/jaxcont/bifurcations/fold_solve.py` (full-file rewrite, currently 152 lines)

**Interfaces:**
- Consumes: `differentiable_root` (Task 1) as `from jaxcont.solvers.implicit import differentiable_root`.
- Produces: nothing new — `fold_point`/`fold_parameter` keep their existing signatures, consumed by `jaxcont/__init__.py` and `tests/test_functional_api.py::TestDifferentiableFold` unchanged.

- [ ] **Step 1: Confirm the baseline passes before refactoring**

Run: `JAX_PLATFORMS=cpu pytest tests/test_functional_api.py -k TestDifferentiableFold -v`
Expected: PASS (3 tests) — this is the regression baseline the refactor must not break.

- [ ] **Step 2: Rewrite `fold_solve.py` to delegate to `differentiable_root`**

Replace the full contents of `src/jaxcont/bifurcations/fold_solve.py` with:

```python
"""
Differentiable fold (saddle-node) solver via the extended system + implicit diff.

A fold of ``f(u, p; args) = 0`` is the solution of the extended system

    G1:  f(u, p)          = 0            (n eqs)   equilibrium
    G2:  f_u(u, p) · v    = 0            (n eqs)   singular Jacobian (null vector v)
    G3:  vᵀv - 1          = 0            (1 eq)    normalization

in the unknowns ``x = (u, p, v)`` (dimension ``2n + 1``). The extended-system
Newton solve and its implicit-function-theorem gradient live in
:func:`jaxcont.solvers.implicit.differentiable_root`, shared with any future
extended-system event (Hopf, LPC, PD, NS); this module only builds the
fold-specific ``G`` and initial guess.

Public entry points:
- :func:`fold_point`     -> (u*, p*, v*), differentiable in ``args``
- :func:`fold_parameter` -> p*,            differentiable in ``args``  (grad-ready)
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax.numpy as jnp
from jax import Array, jacfwd

from jaxcont.solvers.implicit import differentiable_root

PyTree = Any


def _pack(u, p, v):
    return jnp.concatenate([u, jnp.reshape(p, (1,)), v])


def _unpack(x, n):
    return x[:n], x[n], x[n + 1:]


def _extended_residual(x, f, args, n):
    """G(x, args) for the fold extended system."""
    u, p, v = _unpack(x, n)
    f0 = f(u, p, args)                       # (n,)
    jac_u = jacfwd(f, argnums=0)(u, p, args)  # (n, n)
    f1 = jac_u @ v                           # (n,)
    f2 = jnp.dot(v, v) - 1.0                 # scalar
    return jnp.concatenate([f0, f1, jnp.reshape(f2, (1,))])


def _initial_v(f, u, p, args, n):
    """Seed the null vector with the smallest right singular vector of f_u."""
    jac_u = jacfwd(f, argnums=0)(u, p, args)
    # jac_u = U S Vh  ->  smallest singular direction is the last row of Vh
    _, _, vh = jnp.linalg.svd(jac_u)
    v = vh[-1]
    return v / jnp.linalg.norm(v)


def fold_point(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: float | Array,
    args: PyTree = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> Tuple[Array, Array, Array]:
    """
    Locate a fold near ``(u_guess, p_guess)``, differentiable in ``args``.

    Returns ``(u*, p*, v*)`` where ``v*`` is the (unit) null vector of ``f_u``.
    """
    u_guess = jnp.asarray(u_guess)
    n = u_guess.shape[0]
    p_guess = jnp.asarray(p_guess, u_guess.dtype)

    def G(x, theta):
        return _extended_residual(x, f, theta, n)

    def x0(theta):
        v0 = _initial_v(f, u_guess, p_guess, theta, n)
        return _pack(u_guess, p_guess, v0)

    x_star = differentiable_root(G, x0, args, tol=tol, max_iter=max_iter)
    u, p, v = _unpack(x_star, n)
    return u, p, v


def fold_parameter(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: float | Array,
    args: PyTree = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> Array:
    """
    Parameter value ``p*`` at the fold — a scalar, differentiable in ``args``.

    ``jax.grad(lambda a: fold_parameter(f, u0, p0, a))(theta)`` gives the exact
    sensitivity of the fold location to the design parameters.
    """
    _, p, _ = fold_point(f, u_guess, p_guess, args, tol=tol, max_iter=max_iter)
    return p
```

Note what changed from the original: `_make_fold_solver` (the fold-specific `newton`/`custom_vjp` pair) is deleted — that logic now lives generically in `differentiable_root`. `_pack`, `_unpack`, `_extended_residual`, `_initial_v` are unchanged (they're fold-domain-specific: building `G` and the initial guess, not part of the generic solver). `x0` is passed to `differentiable_root` as a **callable** (`def x0(theta): ...`), not a precomputed array — `_initial_v` depends on `args`/`theta` (it's an SVD of the `theta`-dependent Jacobian `f_u`), so it must be evaluated inside `differentiable_root`'s traced primal, exactly as it was inside the original `newton(args)`; see Task 1's design note for why a precomputed array here would break under `jax.grad`. The unused `jax`/`lax` top-level imports are dropped since nothing in this file calls `jax.custom_vjp`/`lax.while_loop` directly anymore.

- [ ] **Step 3: Run the fold regression tests**

Run: `JAX_PLATFORMS=cpu pytest tests/test_functional_api.py -k TestDifferentiableFold -v`
Expected: PASS (3 tests, same as Step 1's baseline) — confirms the refactor didn't change fold behavior or gradients.

- [ ] **Step 4: Run the new primitive tests too**

Run: `JAX_PLATFORMS=cpu pytest tests/test_differentiable_root.py -v`
Expected: PASS (4 tests, unaffected by this file's changes — sanity check only).

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/bifurcations/fold_solve.py
git commit -m "refactor: fold_solve.py delegates its Newton+custom_vjp solve to differentiable_root"
```

---

## Final verification

- [ ] Run the full suite: `JAX_PLATFORMS=cpu pytest tests/ -q` — expect all green, same pass count as before this plan plus the 4 new `test_differentiable_root.py` tests.
- [ ] Confirm `differentiable_root` is NOT re-exported from the top level: `python3 -c "import jaxcont as jc; assert not hasattr(jc, 'differentiable_root')"` — expect no `AssertionError`.
- [ ] Sanity-run the differentiable-fold example end to end: `JAX_PLATFORMS=cpu python3 examples/example_07_differentiable.py` — expect it to run to completion with no traceback (it exercises `jc.fold_point`/`jc.fold_parameter` under `jax.grad`, the exact code path this plan refactored).
