# Codim-2 Direct Point Solvers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five direct codim-2 bifurcation point solvers (cusp, Bogdanov-Takens, generalized
Hopf, zero-Hopf, double Hopf) plus the fold's quadratic normal-form coefficient `a`, all
differentiable in `args` via the existing implicit-function-theorem primitive.

**Architecture:** Each solver builds a square extended system `G(x, θ) = 0` and hands it to
`solvers/implicit.py:differentiable_root`, exactly as `fold_solve.py`/`hopf_normal_form.py` already
do for codim-1. Two new modules: `bifurcations/fold_normal_form.py` (the coefficient `a`) and
`bifurcations/codim2.py` (a shared solve-and-check harness plus the five solvers). No changes to
the continuation engine, `api.py`, or the `Event` protocol.

**Tech Stack:** JAX (`jax.numpy`, `jax.jvp`, `jax.jacfwd`, `jax.grad`), pytest, and Julia +
BifurcationKit.jl v0.5.2 (installed in this dev environment) for the final cross-check.

## Global Constraints

- **Design spec:** `docs/superpowers/specs/2026-08-05-codim2-direct-solvers-design.md` — read it
  first. All code in this plan already incorporates its findings; do not "simplify" away the
  omega normalization, the bordered left-null-vector solve, or the shifted test systems.
- **`p` is shape `(2,)`** in every function here, not a scalar. Existing scalar-`p` code
  (`api.py`, `scan_continuation.py`, `fold_solve.py`, `hopf_normal_form.py`) is untouched.
- **Default `tol=1e-6`, not `1e-8`.** Measured float32 residual floors during planning: BT `0.0`,
  CP `≈4.4e-8`, GH `≈5.96e-08`. A `tol` below that floor makes `converged` report `False` forever
  even when the answer is exact — this repo's recurring issue #12. `tol=1e-4` is too loose (GH
  stops early with parameter error `1.9e-5`). Do not lower these defaults.
- **Never raise on non-convergence.** Every solver returns a trailing `converged` boolean array so
  the surface stays `jit`/`vmap`-safe. Use `jnp` boolean ops, never Python `if` on traced values.
- **Dtype discipline:** follow `fold_solve.py`'s convention — `p_guess = jnp.asarray(p_guess,
  u_guess.dtype)`. Never hardcode `float64`/`complex128`.
- **Every public function needs a docstring citing Kuznetsov** (*Elements of Applied Bifurcation
  Theory*, 3rd ed.) where the design spec does, matching `fold_solve.py`'s existing style.
- **Run `make test` after every task** and confirm no regressions before moving on. Baseline at
  the start of this plan: **211 passed**. (`make test` runs `JAX_PLATFORMS=cpu pytest tests/ -n
  auto`; do not run `-n auto` against the GPU backend — the workers thrash over one device.)

---

### Task 1: `fold_coefficient` — the fold's quadratic normal-form coefficient

**Files:**
- Create: `src/jaxcont/bifurcations/fold_normal_form.py`
- Test: `tests/test_fold_normal_form.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `fold_coefficient(f, u, p, v, args=None) -> Array` (scalar). Task 2's cusp residual
  calls this. `f` has signature `f(u, p, args) -> Array` with `p` shape `(2,)`; `v` is the unit
  right null vector of `f_u`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fold_normal_form.py`:

```python
"""
Tests for jaxcont.bifurcations.fold_normal_form: the fold's quadratic
normal-form coefficient a = 1/2 * <w, B(v,v)>. See
docs/superpowers/specs/2026-08-05-codim2-direct-solvers-design.md.
"""

import jax
import jax.numpy as jnp

from jaxcont.bifurcations.fold_normal_form import fold_coefficient


def _fold_1d(u, p, args):
    # x' = b1 + b2*x + k*x^2, with k defaulting to 1.
    # At u=0, p=(0,0): f=0 and f_u=0, so this is a fold with v=w=1 and
    # B(v,v) = f_uu = 2k, giving the exact coefficient a = k.
    k = 1.0 if args is None else args
    x = u[0]
    b1, b2 = p[0], p[1]
    return jnp.array([b1 + b2 * x + k * x**2])


def test_fold_coefficient_matches_exact_value_of_quadratic_normal_form():
    a = fold_coefficient(_fold_1d, jnp.zeros(1), jnp.zeros(2), jnp.ones(1))
    assert jnp.isclose(float(a), 1.0, atol=1e-5)


def test_fold_coefficient_flips_sign_with_the_quadratic_term():
    a = fold_coefficient(_fold_1d, jnp.zeros(1), jnp.zeros(2), jnp.ones(1), -1.0)
    assert jnp.isclose(float(a), -1.0, atol=1e-5)


def test_fold_coefficient_scales_linearly_with_the_quadratic_term():
    a2 = fold_coefficient(_fold_1d, jnp.zeros(1), jnp.zeros(2), jnp.ones(1), 2.0)
    a5 = fold_coefficient(_fold_1d, jnp.zeros(1), jnp.zeros(2), jnp.ones(1), 5.0)
    assert jnp.isclose(float(a2), 2.0, atol=1e-5)
    assert jnp.isclose(float(a5), 5.0, atol=1e-5)


def _fold_2d(u, p, args):
    # The same fold plus a decoupled stable direction, so the left null
    # vector w is a genuine solve rather than a 1x1 triviality.
    # v = w = (1, 0); a = k as above.
    k = 1.0 if args is None else args
    x, y = u[0], u[1]
    b1, b2 = p[0], p[1]
    return jnp.array([b1 + b2 * x + k * x**2, -y])


def test_fold_coefficient_handles_multidimensional_left_null_vector():
    a = fold_coefficient(
        _fold_2d, jnp.zeros(2), jnp.zeros(2), jnp.array([1.0, 0.0]), 3.0
    )
    assert jnp.isclose(float(a), 3.0, atol=1e-5)


def test_fold_coefficient_grad_matches_finite_difference():
    # a(k) = k exactly, so da/dk == 1; check autodiff agrees with a central
    # difference rather than trusting the closed form alone.
    def a_of_k(k):
        return fold_coefficient(_fold_2d, jnp.zeros(2), jnp.zeros(2),
                                jnp.array([1.0, 0.0]), k)

    g = jax.grad(a_of_k)(2.0)
    h = 1e-3
    fd = (a_of_k(2.0 + h) - a_of_k(2.0 - h)) / (2 * h)
    assert jnp.isfinite(g)
    assert jnp.isclose(float(g), float(fd), atol=1e-4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu pytest tests/test_fold_normal_form.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jaxcont.bifurcations.fold_normal_form'`

- [ ] **Step 3: Create `src/jaxcont/bifurcations/fold_normal_form.py`**

```python
"""
Fold (limit-point) normal-form coefficient.

The quadratic coefficient ``a = 1/2 * <w, B(v,v)>`` of the fold normal form
``y' = a*y^2 + ...`` (Kuznetsov, *Elements of Applied Bifurcation Theory*,
3rd ed., eq. 3.16). ``a != 0`` is the fold's non-degeneracy condition;
``a = 0`` is precisely the cusp condition, which is what
``bifurcations/codim2.py:cusp_point`` appends to the fold extended system.

Sibling to ``hopf_normal_form.py``'s ``lyapunov_coefficient``: pure algebra
(directional derivatives via ``jax.jvp`` plus one linear solve), no Newton
iteration, differentiable wherever its inputs are. Unlike ``l1`` this needs
only real arithmetic and only second derivatives, so it carries no
holomorphy requirement on ``f``.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
from jax import Array, jacfwd, jvp

PyTree = Any


def fold_coefficient(
    f: Callable[[Array, Array, PyTree], Array],
    u: Array,
    p: Array,
    v: Array,
    args: PyTree = None,
) -> Array:
    """
    Quadratic normal-form coefficient ``a`` of the fold at ``(u, p)`` with
    unit right null vector ``v`` (``f_u @ v == 0``).

    ``a = 1/2 * <w, B(v, v)>`` where ``B`` is the second directional
    derivative of ``f`` in ``u`` and ``w`` is the left null vector
    (``w @ f_u == 0``) normalized so ``w @ v == 1``.

    ``p`` has shape ``(2,)`` — these solvers work in two parameters.
    """
    n = u.shape[0]
    jac_u = jacfwd(lambda uu: f(uu, p, args))(u)

    # Left null vector via a BORDERED SOLVE, not an SVD nullspace. This
    # quantity sits inside cusp_point's extended residual and is therefore
    # differentiated by Newton on every iteration, and jnp.linalg.svd's
    # gradient is nan when singular values repeat (found during the Hopf
    # normal-form design -- see that spec). The bordered matrix
    #     [[f_u^T, v], [v^T, 0]]
    # is nonsingular exactly when the zero eigenvalue is simple, which is
    # the fold's own non-degeneracy condition, so this is well-posed
    # wherever `a` is meaningful. Verified equivalent to the SVD route to
    # float32 precision during planning.
    border = jnp.block([
        [jac_u.T, jnp.reshape(v, (n, 1))],
        [jnp.reshape(v, (1, n)), jnp.zeros((1, 1), u.dtype)],
    ])
    rhs = jnp.concatenate([jnp.zeros(n, u.dtype), jnp.ones(1, u.dtype)])
    w = jnp.linalg.solve(border, rhs)[:n]

    def d1(uu, y):
        return jvp(lambda z: f(z, p, args), (uu,), (y,))[1]

    def d2(uu, y, z):
        return jvp(lambda uu_: d1(uu_, y), (uu,), (z,))[1]

    return 0.5 * jnp.dot(w, d2(u, v, v))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu pytest tests/test_fold_normal_form.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the full suite**

Run: `make test`
Expected: 217 passed (211 baseline + 6 new), 0 failed

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/bifurcations/fold_normal_form.py tests/test_fold_normal_form.py
git commit -m "feat: add fold_coefficient, the fold's quadratic normal-form coefficient"
```

---

### Task 2: codim-2 harness + `cusp_point` / `cusp_parameters`

**Files:**
- Create: `src/jaxcont/bifurcations/codim2.py`
- Test: `tests/test_codim2.py`

**Interfaces:**
- Consumes: `fold_coefficient` (Task 1);
  `jaxcont.bifurcations.fold_solve._initial_v(f, u, p, args, n) -> Array` (existing, the SVD-based
  right-null-vector seed); `jaxcont.solvers.implicit.differentiable_root(G, x0, theta, *, tol,
  max_iter) -> Array` (existing).
- Produces:
  - `_solve_and_check(G, x0, args, *, tol, max_iter) -> (x_star, converged)` — used by Tasks 3-6.
  - `_normalize_omega(q2, omega) -> (q2, omega)` — used by Tasks 4-6.
  - `cusp_point(f, u_guess, p_guess, args=None, *, tol=1e-6, max_iter=50) -> (u, p, v, converged)`
  - `cusp_parameters(...) -> p` (shape `(2,)`, no flag)

- [ ] **Step 1: Write the failing test**

Create `tests/test_codim2.py`:

```python
"""
Tests for jaxcont.bifurcations.codim2: direct codim-2 point solvers. See
docs/superpowers/specs/2026-08-05-codim2-direct-solvers-design.md.

Every system here is deliberately SHIFTED so its codim-2 point is at a
non-trivial location. The textbook normal forms all put their codim-2 point
at u=0, p=(0,0), which means a stub returning zeros would pass them -- they
have no discriminating power on their own.
"""

import jax
import jax.numpy as jnp

from jaxcont.bifurcations.codim2 import cusp_point, cusp_parameters


def _cusp_shifted(u, p, args):
    # x' = b1 + b2*xi + k*xi^3   with   xi = x - 2, b1 = p0 - 1, b2 = p1 + 4
    # Cusp of x' = b1 + b2*x + k*x^3 is at x=0, (b1,b2)=(0,0), so this one
    # sits at u*=(2,), p*=(1,-4) -- a non-trivial location.
    k = 1.0 if args is None else args
    xi = u[0] - 2.0
    b1 = p[0] - 1.0
    b2 = p[1] + 4.0
    return jnp.array([b1 + b2 * xi + k * xi**3])


def test_cusp_point_recovers_exact_shifted_cusp():
    u, p, v, ok = cusp_point(
        _cusp_shifted, jnp.array([2.2]), jnp.array([0.8, -3.7]),
    )
    assert bool(ok)
    assert jnp.allclose(u, jnp.array([2.0]), atol=1e-4)
    assert jnp.allclose(p, jnp.array([1.0, -4.0]), atol=1e-4)
    assert jnp.isclose(float(jnp.linalg.norm(v)), 1.0, atol=1e-5)


def test_cusp_point_does_not_merely_return_its_guess():
    # Guards against a trivial implementation that echoes the seed back.
    guess_u = jnp.array([2.2])
    guess_p = jnp.array([0.8, -3.7])
    u, p, _, ok = cusp_point(_cusp_shifted, guess_u, guess_p)
    assert bool(ok)
    assert float(jnp.max(jnp.abs(p - guess_p))) > 1e-2


def test_cusp_parameters_returns_bare_parameter_array():
    p = cusp_parameters(_cusp_shifted, jnp.array([2.2]), jnp.array([0.8, -3.7]))
    assert p.shape == (2,)
    assert jnp.allclose(p, jnp.array([1.0, -4.0]), atol=1e-4)


def test_cusp_reports_not_converged_for_a_hopeless_guess():
    # A system with no cusp anywhere: x' = 1 + x^2 has no equilibrium at all.
    def no_cusp(u, p, args):
        return jnp.array([1.0 + u[0] ** 2 + 0.0 * p[0]])

    _, _, _, ok = cusp_point(no_cusp, jnp.array([5.0]), jnp.array([0.0, 0.0]))
    assert not bool(ok)


def test_cusp_parameters_grad_matches_finite_difference():
    # Move the cusp with an args scalar and check the gradient of its
    # location. This is the headline claim of the whole feature.
    def cusp_moving(u, p, shift):
        xi = u[0] - 2.0
        b1 = p[0] - 1.0 - shift
        b2 = p[1] + 4.0
        return jnp.array([b1 + b2 * xi + xi**3])

    def p0_star(shift):
        return cusp_parameters(
            cusp_moving, jnp.array([2.1]), jnp.array([0.9, -3.8]), shift
        )[0]

    g = jax.grad(p0_star)(0.1)
    h = 1e-3
    fd = (p0_star(0.1 + h) - p0_star(0.1 - h)) / (2 * h)
    assert jnp.isfinite(g)
    assert jnp.isclose(float(g), float(fd), atol=1e-3)


def test_cusp_agrees_with_the_codim1_fold_solver_there():
    # Cross-check against the existing scalar-p fold solver: a cusp IS a
    # fold, so freezing the second parameter at its cusp value and running
    # fold_point must land on the same (u, p0).
    from jaxcont.bifurcations.fold_solve import fold_point

    u_c, p_c, _, ok = cusp_point(
        _cusp_shifted, jnp.array([2.2]), jnp.array([0.8, -3.7]),
    )
    assert bool(ok)

    def f_scalar(u, p, args):
        return _cusp_shifted(u, jnp.array([p, p_c[1]]), args)

    u_f, p_f, _ = fold_point(f_scalar, jnp.array([2.1]), float(p_c[0]) + 0.05)
    assert jnp.allclose(u_f, u_c, atol=1e-3)
    assert jnp.isclose(float(p_f), float(p_c[0]), atol=1e-3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu pytest tests/test_codim2.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jaxcont.bifurcations.codim2'`

- [ ] **Step 3: Create `src/jaxcont/bifurcations/codim2.py`**

```python
"""
Direct codim-2 bifurcation point solvers.

Each solver builds a square extended system ``G(x, theta) = 0`` whose root
is a codim-2 point, and solves it with
``solvers/implicit.py:differentiable_root`` -- the same Newton-in-
``lax.while_loop`` + ``jax.custom_vjp`` machinery ``fold_solve.py`` and
``hopf_normal_form.py`` already use for codim-1. The result is therefore
differentiable in ``args`` via the implicit function theorem, so
``jax.grad`` of a codim-2 *location* with respect to design parameters
works without differentiating through the iteration.

Codim-2 points need TWO free parameters, so throughout this module ``p``
has shape ``(2,)`` rather than being a scalar. The right-hand side keeps
the usual ``f(u, p, args)`` signature; only ``p``'s shape changes.

These are refinement tools: every solver needs a guess already near the
codim-2 point and none of them search. Finding codim-2 points you cannot
already approximate requires two-parameter continuation, which is a
separate (unstarted) roadmap item.

See docs/superpowers/specs/2026-08-05-codim2-direct-solvers-design.md.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax.numpy as jnp
from jax import Array, jacfwd

from jaxcont.bifurcations.fold_normal_form import fold_coefficient
from jaxcont.bifurcations.fold_solve import _initial_v
from jaxcont.solvers.implicit import differentiable_root

PyTree = Any


def _solve_and_check(
    G: Callable[[Array, PyTree], Array],
    x0: Callable[[PyTree], Array],
    args: PyTree,
    *,
    tol: float,
    max_iter: int,
) -> Tuple[Array, Array]:
    """
    Solve ``G(x, args) = 0`` and report whether the result is trustworthy.

    ``differentiable_root`` returns only the root, and its Newton loop exits
    on non-finite iterates as well as on convergence, so the caller cannot
    tell success from failure without re-checking. ``converged`` is a JAX
    boolean (not a Python bool) so callers stay ``jit``/``vmap``-safe.
    """
    x_star = differentiable_root(G, x0, args, tol=tol, max_iter=max_iter)
    residual = jnp.linalg.norm(G(x_star, args))
    converged = (
        jnp.isfinite(residual)
        & (residual < tol)
        & jnp.all(jnp.isfinite(x_star))
    )
    return x_star, converged


def _normalize_omega(q2: Array, omega: Array) -> Tuple[Array, Array]:
    """
    Pin the critical frequency to ``omega >= 0``.

    The Hopf block of every extended system below is EXACTLY invariant under
    ``(omega, q2) -> (-omega, -q2)``: substituting flips the sign of the
    ``J q2 - omega q1`` row and leaves ``J q1 + omega q2`` unchanged, so both
    signs are genuine roots and Newton picks whichever the seed falls toward
    (during planning it chose the negative one from positive seeds on more
    than one system). Flipping afterwards selects the conjugate of the same
    eigenvector, so this is exact rather than a heuristic.

    It matters beyond cosmetics: ``bifurcations/events.py:Hopf.refine()``
    already treats ``omega0 <= 0`` as a failed solve. Applied AFTER the
    solve, never as an extra equation, which would break squareness.
    """
    flip = omega < 0
    return jnp.where(flip, -q2, q2), jnp.where(flip, -omega, omega)


# --------------------------------------------------------------------------
# CP -- cusp
# --------------------------------------------------------------------------

def _cp_unpack(x: Array, n: int) -> Tuple[Array, Array, Array]:
    return x[:n], x[n:n + 2], x[n + 2:]


def _cp_residual(x, f, args, n):
    """
    Cusp extended system, ``2n+2`` equations in ``2n+2`` unknowns
    ``(u, p, v)``: the fold system plus ``a = 0``.
    """
    u, p, v = _cp_unpack(x, n)
    jac_u = jacfwd(f, argnums=0)(u, p, args)
    return jnp.concatenate([
        f(u, p, args),                                        # n
        jac_u @ v,                                            # n
        jnp.reshape(jnp.dot(v, v) - 1.0, (1,)),               # 1
        jnp.reshape(fold_coefficient(f, u, p, v, args), (1,)),  # 1
    ])


def cusp_point(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Tuple[Array, Array, Array, Array]:
    """
    Locate a cusp (``CP``) near ``(u_guess, p_guess)``, differentiable in
    ``args``. Kuznetsov, 3rd ed., Sec. 8.2.

    A cusp is a fold at which the quadratic normal-form coefficient ``a``
    also vanishes. Returns ``(u*, p*, v*, converged)`` where ``v*`` is the
    unit right null vector of ``f_u`` and ``p*`` has shape ``(2,)``.

    ``tol`` defaults to ``1e-6`` rather than ``1e-8`` because the achievable
    float32 residual floor for this system is around ``4e-8``; a tighter
    tolerance reports ``converged=False`` even when the answer is exact.
    """
    u_guess = jnp.asarray(u_guess)
    n = u_guess.shape[0]
    p_guess = jnp.asarray(p_guess, u_guess.dtype)

    def G(x, theta):
        return _cp_residual(x, f, theta, n)

    def x0(theta):
        v0 = _initial_v(f, u_guess, p_guess, theta, n)
        return jnp.concatenate([u_guess, p_guess, v0])

    x_star, converged = _solve_and_check(G, x0, args, tol=tol, max_iter=max_iter)
    u, p, v = _cp_unpack(x_star, n)
    return u, p, v, converged


def cusp_parameters(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Array:
    """
    Parameter pair ``p*`` (shape ``(2,)``) at the cusp -- differentiable in
    ``args``. Returns a bare array with no convergence flag so
    ``jax.grad(...)`` applies directly; use :func:`cusp_point` when you need
    the flag.
    """
    _, p, _, _ = cusp_point(f, u_guess, p_guess, args, tol=tol, max_iter=max_iter)
    return p
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu pytest tests/test_codim2.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the full suite**

Run: `make test`
Expected: 223 passed, 0 failed

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/bifurcations/codim2.py tests/test_codim2.py
git commit -m "feat: add codim-2 solve harness and cusp_point/cusp_parameters"
```

---

### Task 3: `bogdanov_takens_point` / `bogdanov_takens_parameters`

**Files:**
- Modify: `src/jaxcont/bifurcations/codim2.py` (append)
- Test: `tests/test_codim2.py` (append)

**Interfaces:**
- Consumes: `_solve_and_check`, `_initial_v` (Task 2).
- Produces: `bogdanov_takens_point(f, u_guess, p_guess, args=None, *, tol=1e-6, max_iter=50) ->
  (u, p, v0, v1, converged)` and `bogdanov_takens_parameters(...) -> p`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_codim2.py`:

```python
from jaxcont.bifurcations.codim2 import (
    bogdanov_takens_point, bogdanov_takens_parameters,
)


def _bt_shifted(u, p, args):
    # x' = y ; y' = b1 + b2*x + x^2 + x*y  has its BT at u=0, p=(0,0).
    # Shifted by x -> X-5, y -> Y-2, b1 -> B1-3, b2 -> B2+1, so the BT sits
    # at u*=(5,2), p*=(3,-1).
    X, Y = u[0], u[1]
    x, y = X - 5.0, Y - 2.0
    b1 = p[0] - 3.0
    b2 = p[1] + 1.0
    return jnp.array([y, b1 + b2 * x + x**2 + x * y])


def test_bogdanov_takens_point_recovers_exact_shifted_bt():
    u, p, v0, v1, ok = bogdanov_takens_point(
        _bt_shifted, jnp.array([5.3, 1.7]), jnp.array([2.6, -0.8]),
    )
    assert bool(ok)
    assert jnp.allclose(u, jnp.array([5.0, 2.0]), atol=1e-4)
    assert jnp.allclose(p, jnp.array([3.0, -1.0]), atol=1e-4)
    assert jnp.isclose(float(jnp.linalg.norm(v0)), 1.0, atol=1e-5)


def test_bogdanov_takens_jacobian_has_a_genuine_jordan_block():
    # The defining property: f_u has a DOUBLE zero eigenvalue with only one
    # eigenvector, i.e. f_u @ v1 == v0 and v0 . v1 == 0.
    u, p, v0, v1, ok = bogdanov_takens_point(
        _bt_shifted, jnp.array([5.3, 1.7]), jnp.array([2.6, -0.8]),
    )
    assert bool(ok)
    jac = jax.jacfwd(lambda uu: _bt_shifted(uu, p, None))(u)
    assert jnp.allclose(jac @ v0, jnp.zeros(2), atol=1e-4)
    assert jnp.allclose(jac @ v1, v0, atol=1e-4)
    assert jnp.isclose(float(jnp.dot(v0, v1)), 0.0, atol=1e-5)


def test_bogdanov_takens_parameters_grad_matches_finite_difference():
    def bt_moving(u, p, shift):
        X, Y = u[0], u[1]
        x, y = X - 5.0, Y - 2.0
        b1 = p[0] - 3.0 - shift
        b2 = p[1] + 1.0
        return jnp.array([y, b1 + b2 * x + x**2 + x * y])

    def p0_star(shift):
        return bogdanov_takens_parameters(
            bt_moving, jnp.array([5.2, 1.8]), jnp.array([2.7, -0.9]), shift
        )[0]

    g = jax.grad(p0_star)(0.1)
    h = 1e-3
    fd = (p0_star(0.1 + h) - p0_star(0.1 - h)) / (2 * h)
    assert jnp.isfinite(g)
    assert jnp.isclose(float(g), float(fd), atol=1e-3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu pytest tests/test_codim2.py -v -k bogdanov`
Expected: FAIL — `ImportError: cannot import name 'bogdanov_takens_point'`

- [ ] **Step 3: Append to `src/jaxcont/bifurcations/codim2.py`**

```python
# --------------------------------------------------------------------------
# BT -- Bogdanov-Takens
# --------------------------------------------------------------------------

def _bt_unpack(x: Array, n: int) -> Tuple[Array, Array, Array, Array]:
    return x[:n], x[n:n + 2], x[n + 2:2 * n + 2], x[2 * n + 2:]


def _bt_residual(x, f, args, n):
    """
    Bogdanov-Takens extended system, ``3n+2`` equations in ``3n+2`` unknowns
    ``(u, p, v0, v1)``.

    Uses the JORDAN-CHAIN formulation (``v1`` is the generalized
    eigenvector). The textbook left/right-null-vector alternative
    ``f=0, J v=0, J^T w=0, |v|=1, |w|=1, w.v=0`` is OVERDETERMINED -- it has
    ``3n+3`` equations for ``3n+2`` unknowns -- which is why it is not used
    here. Confirmed by direct count during planning.
    """
    u, p, v0, v1 = _bt_unpack(x, n)
    jac_u = jacfwd(f, argnums=0)(u, p, args)
    return jnp.concatenate([
        f(u, p, args),                              # n
        jac_u @ v0,                                 # n
        jac_u @ v1 - v0,                            # n
        jnp.reshape(jnp.dot(v0, v0) - 1.0, (1,)),   # 1
        jnp.reshape(jnp.dot(v0, v1), (1,)),         # 1
    ])


def bogdanov_takens_point(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Locate a Bogdanov-Takens point (``BT``) near ``(u_guess, p_guess)``,
    differentiable in ``args``. Kuznetsov, 3rd ed., Sec. 8.4.

    ``BT`` is where ``f_u`` has a double zero eigenvalue with a single
    eigenvector -- the organizing centre at which fold and Hopf curves meet.
    Returns ``(u*, p*, v0*, v1*, converged)`` with ``f_u v0 = 0`` and
    ``f_u v1 = v0`` (the Jordan chain), ``p*`` of shape ``(2,)``.
    """
    u_guess = jnp.asarray(u_guess)
    n = u_guess.shape[0]
    p_guess = jnp.asarray(p_guess, u_guess.dtype)

    def G(x, theta):
        return _bt_residual(x, f, theta, n)

    def x0(theta):
        v0 = _initial_v(f, u_guess, p_guess, theta, n)
        # A UNIFORM nonzero seed for the generalized eigenvector. Seeding
        # v1 with zeros makes the very first Newton step produce nan --
        # verified during planning, and NOT obvious from the residual being
        # linear in v1. Do not "simplify" this to jnp.zeros(n).
        v1 = jnp.ones(n, u_guess.dtype) / jnp.sqrt(jnp.asarray(n, u_guess.dtype))
        return jnp.concatenate([u_guess, p_guess, v0, v1])

    x_star, converged = _solve_and_check(G, x0, args, tol=tol, max_iter=max_iter)
    u, p, v0, v1 = _bt_unpack(x_star, n)
    return u, p, v0, v1, converged


def bogdanov_takens_parameters(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Array:
    """
    Parameter pair ``p*`` (shape ``(2,)``) at the Bogdanov-Takens point --
    differentiable in ``args``, no convergence flag (see
    :func:`cusp_parameters`).
    """
    _, p, _, _, _ = bogdanov_takens_point(
        f, u_guess, p_guess, args, tol=tol, max_iter=max_iter
    )
    return p
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu pytest tests/test_codim2.py -v`
Expected: 9 passed

- [ ] **Step 5: Run the full suite**

Run: `make test`
Expected: 226 passed, 0 failed

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/bifurcations/codim2.py tests/test_codim2.py
git commit -m "feat: add bogdanov_takens_point via the Jordan-chain extended system"
```

---

### Task 4: `generalized_hopf_point` / `generalized_hopf_parameters`

**Files:**
- Modify: `src/jaxcont/bifurcations/codim2.py` (append)
- Test: `tests/test_codim2.py` (append)

**Interfaces:**
- Consumes: `_solve_and_check`, `_normalize_omega` (Task 2);
  `jaxcont.bifurcations.hopf_normal_form._seed(f, u, p, args, n) -> (q1, q2, omega)` and
  `lyapunov_coefficient(f, u, p, q1, q2, omega0, args=None) -> Array` (both existing).
- Produces: `generalized_hopf_point(...) -> (u, p, q1, q2, omega, converged)` and
  `generalized_hopf_parameters(...) -> p`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_codim2.py`:

```python
from jaxcont.bifurcations.codim2 import (
    generalized_hopf_point, generalized_hopf_parameters,
)
from jaxcont.bifurcations.hopf_normal_form import lyapunov_coefficient


def _gh_shifted(u, p, args):
    # Bautin normal form  r' = r*(b1 + b2*r^2 - r^4),  theta' = 1,  in
    # Cartesian coordinates. Hopf at b1=0; l1 is proportional to b2, so the
    # generalized Hopf is at (b1,b2)=(0,0). Shifted to p*=(2,-3).
    x, y = u[0], u[1]
    b1 = p[0] - 2.0
    b2 = p[1] + 3.0
    r2 = x**2 + y**2
    g = b1 + b2 * r2 - r2**2
    return jnp.array([-y + x * g, x + y * g])


def test_generalized_hopf_point_recovers_exact_shifted_gh():
    u, p, q1, q2, omega, ok = generalized_hopf_point(
        _gh_shifted, jnp.array([0.02, -0.03]), jnp.array([2.05, -2.90]),
    )
    assert bool(ok)
    assert jnp.allclose(u, jnp.zeros(2), atol=1e-4)
    assert jnp.allclose(p, jnp.array([2.0, -3.0]), atol=1e-4)
    assert jnp.isclose(float(omega), 1.0, atol=1e-4)


def test_generalized_hopf_omega_is_normalized_positive_from_either_seed():
    # The Hopf block is invariant under (omega, q2) -> (-omega, -q2), so the
    # raw solve can land on either sign. This is the regression test for
    # that -- without it the bug reappears invisibly.
    for seed_u in (jnp.array([0.02, -0.03]), jnp.array([-0.02, 0.03])):
        _, _, _, _, omega, ok = generalized_hopf_point(
            _gh_shifted, seed_u, jnp.array([2.05, -2.90]),
        )
        assert bool(ok)
        assert float(omega) > 0.0


def test_generalized_hopf_has_vanishing_lyapunov_coefficient():
    # The defining condition: l1 == 0 at the returned point.
    u, p, q1, q2, omega, ok = generalized_hopf_point(
        _gh_shifted, jnp.array([0.02, -0.03]), jnp.array([2.05, -2.90]),
    )
    assert bool(ok)
    l1 = lyapunov_coefficient(_gh_shifted, u, p, q1, q2, omega, None)
    assert abs(float(l1)) < 1e-4


def test_generalized_hopf_parameters_grad_matches_finite_difference():
    def gh_moving(u, p, shift):
        x, y = u[0], u[1]
        b1 = p[0] - 2.0 - shift
        b2 = p[1] + 3.0
        r2 = x**2 + y**2
        g = b1 + b2 * r2 - r2**2
        return jnp.array([-y + x * g, x + y * g])

    def p0_star(shift):
        return generalized_hopf_parameters(
            gh_moving, jnp.array([0.02, -0.03]), jnp.array([2.05, -2.90]), shift
        )[0]

    g = jax.grad(p0_star)(0.05)
    h = 1e-3
    fd = (p0_star(0.05 + h) - p0_star(0.05 - h)) / (2 * h)
    assert jnp.isfinite(g)
    assert jnp.isclose(float(g), float(fd), atol=1e-3)


def test_generalized_hopf_agrees_with_the_codim1_hopf_solver_there():
    # Cross-check against the existing scalar-p Hopf solver: a GH IS a Hopf
    # point, so freezing the second parameter at its GH value and running
    # hopf_point must land on the same p0 and frequency.
    from jaxcont.bifurcations.hopf_normal_form import hopf_point

    _, p_g, _, _, om_g, ok = generalized_hopf_point(
        _gh_shifted, jnp.array([0.02, -0.03]), jnp.array([2.05, -2.90]),
    )
    assert bool(ok)

    def f_scalar(u, p, args):
        return _gh_shifted(u, jnp.array([p, p_g[1]]), args)

    _, p_h, _, _, om_h = hopf_point(
        f_scalar, jnp.array([0.01, -0.01]), float(p_g[0]) + 0.02, tol=1e-6,
    )
    assert jnp.isclose(float(p_h), float(p_g[0]), atol=1e-3)
    assert jnp.isclose(abs(float(om_h)), float(om_g), atol=1e-3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu pytest tests/test_codim2.py -v -k generalized`
Expected: FAIL — `ImportError: cannot import name 'generalized_hopf_point'`

- [ ] **Step 3: Append to `src/jaxcont/bifurcations/codim2.py`**

First extend the import block at the top of the file — change

```python
from jaxcont.bifurcations.fold_solve import _initial_v
```

to

```python
from jaxcont.bifurcations.fold_solve import _initial_v
from jaxcont.bifurcations.hopf_normal_form import _seed as _hopf_seed
from jaxcont.bifurcations.hopf_normal_form import lyapunov_coefficient
```

Then append:

```python
# --------------------------------------------------------------------------
# GH -- generalized Hopf (Bautin)
# --------------------------------------------------------------------------

def _gh_unpack(x, n):
    return (x[:n], x[n:n + 2], x[n + 2:2 * n + 2],
            x[2 * n + 2:3 * n + 2], x[-1])


def _gh_residual(x, f, args, n, u_guess, p_guess):
    """
    Generalized-Hopf extended system, ``3n+3`` equations in ``3n+3``
    unknowns ``(u, p, q1, q2, omega)``: the Hopf system plus ``l1 = 0``.
    """
    u, p, q1, q2, omega = _gh_unpack(x, n)
    # Seed recomputed inside the traced primal on every iteration, exactly
    # as hopf_normal_form.py does -- differentiable_root's contract requires
    # theta-dependent seeds to be resolved here, not hoisted out. Wrapped in
    # stop_gradient because jnp.linalg.eig (inside _hopf_seed) has no
    # gradient rule for non-symmetric eigenvectors, and this seed only feeds
    # the phase condition -- it must not carry a gradient path. Found and
    # fixed by Task 4's implementer: an earlier draft of this code was
    # missing this wrapper and failed the gradient test with
    # NotImplementedError on eig's vjp.
    q1_seed, q2_seed, _ = _hopf_seed(f, u_guess, p_guess, lax.stop_gradient(args), n)
    jac_u = jacfwd(f, argnums=0)(u, p, args)
    l1 = lyapunov_coefficient(f, u, p, q1, q2, omega, args)
    return jnp.concatenate([
        f(u, p, args),                                                   # n
        jac_u @ q1 + omega * q2,                                         # n
        jac_u @ q2 - omega * q1,                                         # n
        jnp.reshape(jnp.dot(q1, q1) + jnp.dot(q2, q2) - 1.0, (1,)),      # 1
        # Seed-based phase condition, matching hopf_normal_form.py's g5.
        # The naive `q1 . q2 == 0` alternative is recorded as broken in the
        # Hopf design spec -- do not substitute it.
        jnp.reshape(jnp.dot(q1_seed, q2) - jnp.dot(q2_seed, q1), (1,)),  # 1
        jnp.reshape(l1, (1,)),                                           # 1
    ])


def generalized_hopf_point(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """
    Locate a generalized Hopf / Bautin point (``GH``) near
    ``(u_guess, p_guess)``, differentiable in ``args``. Kuznetsov, 3rd ed.,
    Sec. 8.3.

    ``GH`` is a Hopf point at which the first Lyapunov coefficient ``l1``
    also vanishes -- the point separating supercritical from subcritical
    Hopf along a Hopf curve. Returns
    ``(u*, p*, q1*, q2*, omega*, converged)``; ``omega* >= 0`` by
    construction (see :func:`_normalize_omega`) and ``p*`` has shape ``(2,)``.

    ``f`` must be complex-analytic (holomorphic) in ``u`` near the point,
    inherited from :func:`~jaxcont.bifurcations.hopf_normal_form.lyapunov_coefficient`.

    ``tol`` defaults to ``1e-6``: the achievable float32 residual floor for
    this system is about ``6e-8`` (measured), so ``1e-8`` would report
    ``converged=False`` even where the answer is exact, while ``1e-4`` stops
    early with visible parameter error.
    """
    u_guess = jnp.asarray(u_guess)
    n = u_guess.shape[0]
    p_guess = jnp.asarray(p_guess, u_guess.dtype)

    def G(x, theta):
        return _gh_residual(x, f, theta, n, u_guess, p_guess)

    def x0(theta):
        q1_0, q2_0, omega_0 = _hopf_seed(f, u_guess, p_guess, theta, n)
        return jnp.concatenate(
            [u_guess, p_guess, q1_0, q2_0, jnp.reshape(omega_0, (1,))]
        )

    x_star, converged = _solve_and_check(G, x0, args, tol=tol, max_iter=max_iter)
    u, p, q1, q2, omega = _gh_unpack(x_star, n)
    q2, omega = _normalize_omega(q2, omega)
    return u, p, q1, q2, omega, converged


def generalized_hopf_parameters(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Array:
    """
    Parameter pair ``p*`` (shape ``(2,)``) at the generalized Hopf point --
    differentiable in ``args``, no convergence flag (see
    :func:`cusp_parameters`).
    """
    _, p, _, _, _, _ = generalized_hopf_point(
        f, u_guess, p_guess, args, tol=tol, max_iter=max_iter
    )
    return p
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu pytest tests/test_codim2.py -v`
Expected: 14 passed

- [ ] **Step 5: Run the full suite**

Run: `make test`
Expected: 231 passed, 0 failed

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/bifurcations/codim2.py tests/test_codim2.py
git commit -m "feat: add generalized_hopf_point with omega sign normalization"
```

---

### Task 5: `zero_hopf_point` / `zero_hopf_parameters`

**Files:**
- Modify: `src/jaxcont/bifurcations/codim2.py` (append)
- Test: `tests/test_codim2.py` (append)

**Interfaces:**
- Consumes: `_solve_and_check`, `_normalize_omega`, `_initial_v`, `_hopf_seed` (Tasks 2, 4).
- Produces: `zero_hopf_point(...) -> (u, p, v, q1, q2, omega, converged)` and
  `zero_hopf_parameters(...) -> p`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_codim2.py`:

```python
from jaxcont.bifurcations.codim2 import zero_hopf_point, zero_hopf_parameters


def _zh_shifted(u, p, args):
    # A fold block decoupled from a Hopf block:
    #   w' = b1 + w^2                    (zero eigenvalue at w=0, b1=0)
    #   x' = b2*x - y ; y' = x + b2*y    (pair b2 +- i, imaginary at b2=0)
    # Zero-Hopf at u=0, p=(0,0); shifted here to u*=(1,0,0), p*=(4,-2).
    w, x, y = u[0] - 1.0, u[1], u[2]
    b1 = p[0] - 4.0
    b2 = p[1] + 2.0
    return jnp.array([b1 + w**2, b2 * x - y, x + b2 * y])


def test_zero_hopf_point_recovers_exact_shifted_zh():
    u, p, v, q1, q2, omega, ok = zero_hopf_point(
        _zh_shifted, jnp.array([1.05, 0.03, -0.02]), jnp.array([4.04, -1.94]),
    )
    assert bool(ok)
    assert jnp.allclose(u, jnp.array([1.0, 0.0, 0.0]), atol=1e-4)
    assert jnp.allclose(p, jnp.array([4.0, -2.0]), atol=1e-4)
    assert jnp.isclose(float(omega), 1.0, atol=1e-4)
    assert float(omega) > 0.0


def test_zero_hopf_has_both_a_zero_eigenvalue_and_an_imaginary_pair():
    u, p, v, q1, q2, omega, ok = zero_hopf_point(
        _zh_shifted, jnp.array([1.05, 0.03, -0.02]), jnp.array([4.04, -1.94]),
    )
    assert bool(ok)
    jac = jax.jacfwd(lambda uu: _zh_shifted(uu, p, None))(u)
    # zero eigenvalue, witnessed by the null vector
    assert jnp.allclose(jac @ v, jnp.zeros(3), atol=1e-4)
    assert jnp.isclose(float(jnp.linalg.norm(v)), 1.0, atol=1e-5)
    # imaginary pair, witnessed by the eigenvector relations
    assert jnp.allclose(jac @ q1 + omega * q2, jnp.zeros(3), atol=1e-4)
    assert jnp.allclose(jac @ q2 - omega * q1, jnp.zeros(3), atol=1e-4)


def test_zero_hopf_parameters_grad_matches_finite_difference():
    def zh_moving(u, p, shift):
        w, x, y = u[0] - 1.0, u[1], u[2]
        b1 = p[0] - 4.0 - shift
        b2 = p[1] + 2.0
        return jnp.array([b1 + w**2, b2 * x - y, x + b2 * y])

    def p0_star(shift):
        return zero_hopf_parameters(
            zh_moving, jnp.array([1.05, 0.03, -0.02]),
            jnp.array([4.04, -1.94]), shift,
        )[0]

    g = jax.grad(p0_star)(0.05)
    h = 1e-3
    fd = (p0_star(0.05 + h) - p0_star(0.05 - h)) / (2 * h)
    assert jnp.isfinite(g)
    assert jnp.isclose(float(g), float(fd), atol=1e-3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu pytest tests/test_codim2.py -v -k zero_hopf`
Expected: FAIL — `ImportError: cannot import name 'zero_hopf_point'`

- [ ] **Step 3: Append to `src/jaxcont/bifurcations/codim2.py`**

```python
# --------------------------------------------------------------------------
# ZH -- zero-Hopf
# --------------------------------------------------------------------------

def _zh_unpack(x, n):
    return (x[:n], x[n:n + 2], x[n + 2:2 * n + 2],
            x[2 * n + 2:3 * n + 2], x[3 * n + 2:4 * n + 2], x[-1])


def _zh_residual(x, f, args, n, u_guess, p_guess):
    """
    Zero-Hopf extended system, ``4n+3`` equations in ``4n+3`` unknowns
    ``(u, p, v, q1, q2, omega)``: one equilibrium condition carrying both a
    fold block and a Hopf block.
    """
    u, p, v, q1, q2, omega = _zh_unpack(x, n)
    # stop_gradient required here -- see the identical comment in
    # _gh_residual above. Task 4 found this the hard way (a missing
    # stop_gradient here fails the gradient test with a NotImplementedError
    # from jnp.linalg.eig's vjp on non-symmetric eigenvectors); `lax` is
    # already imported in codim2.py by Task 4's fix, no new import needed.
    q1_seed, q2_seed, _ = _hopf_seed(f, u_guess, p_guess, lax.stop_gradient(args), n)
    jac_u = jacfwd(f, argnums=0)(u, p, args)
    return jnp.concatenate([
        f(u, p, args),                                                   # n
        jac_u @ v,                                                       # n
        jnp.reshape(jnp.dot(v, v) - 1.0, (1,)),                          # 1
        jac_u @ q1 + omega * q2,                                         # n
        jac_u @ q2 - omega * q1,                                         # n
        jnp.reshape(jnp.dot(q1, q1) + jnp.dot(q2, q2) - 1.0, (1,)),      # 1
        jnp.reshape(jnp.dot(q1_seed, q2) - jnp.dot(q2_seed, q1), (1,)),  # 1
    ])


def zero_hopf_point(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array]:
    """
    Locate a zero-Hopf point (``ZH``) near ``(u_guess, p_guess)``,
    differentiable in ``args``. Kuznetsov, 3rd ed., Sec. 8.5.

    ``ZH`` is where ``f_u`` simultaneously has a zero eigenvalue and a pair
    of purely imaginary eigenvalues -- the intersection of a fold curve and
    a Hopf curve. Requires ``n >= 3``. Returns
    ``(u*, p*, v*, q1*, q2*, omega*, converged)`` with ``omega* >= 0`` and
    ``p*`` of shape ``(2,)``.
    """
    u_guess = jnp.asarray(u_guess)
    n = u_guess.shape[0]
    p_guess = jnp.asarray(p_guess, u_guess.dtype)

    def G(x, theta):
        return _zh_residual(x, f, theta, n, u_guess, p_guess)

    def x0(theta):
        v0 = _initial_v(f, u_guess, p_guess, theta, n)
        q1_0, q2_0, omega_0 = _hopf_seed(f, u_guess, p_guess, theta, n)
        return jnp.concatenate(
            [u_guess, p_guess, v0, q1_0, q2_0, jnp.reshape(omega_0, (1,))]
        )

    x_star, converged = _solve_and_check(G, x0, args, tol=tol, max_iter=max_iter)
    u, p, v, q1, q2, omega = _zh_unpack(x_star, n)
    q2, omega = _normalize_omega(q2, omega)
    return u, p, v, q1, q2, omega, converged


def zero_hopf_parameters(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Array:
    """
    Parameter pair ``p*`` (shape ``(2,)``) at the zero-Hopf point --
    differentiable in ``args``, no convergence flag (see
    :func:`cusp_parameters`).
    """
    _, p, _, _, _, _, _ = zero_hopf_point(
        f, u_guess, p_guess, args, tol=tol, max_iter=max_iter
    )
    return p
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu pytest tests/test_codim2.py -v`
Expected: 17 passed

- [ ] **Step 5: Run the full suite**

Run: `make test`
Expected: 234 passed, 0 failed

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/bifurcations/codim2.py tests/test_codim2.py
git commit -m "feat: add zero_hopf_point combining fold and Hopf blocks"
```

---

### Task 6: `double_hopf_point` / `double_hopf_parameters` + pair-separation guard

**Files:**
- Modify: `src/jaxcont/bifurcations/codim2.py` (append)
- Test: `tests/test_codim2.py` (append)

**Interfaces:**
- Consumes: `_solve_and_check`, `_normalize_omega`, `_hopf_seed` (Tasks 2, 4).
- Produces: `double_hopf_point(f, u_guess, p_guess, args=None, *, seed_b, tol=1e-6, max_iter=50,
  separation_tolerance=1e-3) -> (u, p, q1a, q2a, omega_a, q1b, q2b, omega_b, converged)` and
  `double_hopf_parameters(f, u_guess, p_guess, args=None, *, seed_b, tol=1e-6, max_iter=50,
  separation_tolerance=1e-3) -> p`.

Note `seed_b` is **keyword-only and required** — unlike every other solver here. The two Hopf
blocks need *distinct* seeds; the eigen-decomposition heuristic finds only one pair, so without an
independent `seed_b` both blocks converge onto it and the system goes structurally singular.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_codim2.py`:

```python
from jaxcont.bifurcations.codim2 import double_hopf_point, double_hopf_parameters


def _hh_shifted(u, p, args):
    # Two decoupled linear rotations with distinct frequencies 1 and 2:
    #   pair A = b1 +- 1i,  pair B = b2 +- 2i
    # Double Hopf where both real parts vanish; shifted to p*=(5,-6).
    x1, y1, x2, y2 = u[0], u[1], u[2], u[3]
    b1 = p[0] - 5.0
    b2 = p[1] + 6.0
    return jnp.array([
        b1 * x1 - 1.0 * y1,
        1.0 * x1 + b1 * y1,
        b2 * x2 - 2.0 * y2,
        2.0 * x2 + b2 * y2,
    ])


def test_double_hopf_point_recovers_exact_shifted_hh():
    u, p, q1a, q2a, oa, q1b, q2b, ob, ok = double_hopf_point(
        _hh_shifted,
        jnp.array([0.03, -0.02, 0.04, 0.01]),
        jnp.array([5.05, -5.93]),
        seed_b=jnp.array([0.0, 0.0, 1.0, 0.0]),
    )
    assert bool(ok)
    assert jnp.allclose(p, jnp.array([5.0, -6.0]), atol=1e-4)
    # both frequencies normalized positive, and the two distinct pairs found
    assert float(oa) > 0.0 and float(ob) > 0.0
    found = sorted([float(oa), float(ob)])
    assert jnp.isclose(found[0], 1.0, atol=1e-3)
    assert jnp.isclose(found[1], 2.0, atol=1e-3)


def test_double_hopf_reports_not_converged_when_both_pairs_collapse():
    # Seeding block B onto the SAME physical pair as block A makes the
    # extended system structurally singular. During planning this produced
    # nan rather than a plausible wrong answer; the separation check turns
    # that into an explicit converged=False instead of a bare nan.
    _, _, _, _, oa, _, _, ob, ok = double_hopf_point(
        _hh_shifted,
        jnp.array([0.03, -0.02, 0.04, 0.01]),
        jnp.array([5.05, -5.93]),
        seed_b=jnp.array([1.0, 0.0, 0.0, 0.0]),
    )
    assert not bool(ok)


def test_double_hopf_parameters_grad_matches_finite_difference():
    def hh_moving(u, p, shift):
        x1, y1, x2, y2 = u[0], u[1], u[2], u[3]
        b1 = p[0] - 5.0 - shift
        b2 = p[1] + 6.0
        return jnp.array([
            b1 * x1 - 1.0 * y1, 1.0 * x1 + b1 * y1,
            b2 * x2 - 2.0 * y2, 2.0 * x2 + b2 * y2,
        ])

    def p0_star(shift):
        return double_hopf_parameters(
            hh_moving, jnp.array([0.03, -0.02, 0.04, 0.01]),
            jnp.array([5.05, -5.93]), shift,
            seed_b=jnp.array([0.0, 0.0, 1.0, 0.0]),
        )[0]

    g = jax.grad(p0_star)(0.05)
    h = 1e-3
    fd = (p0_star(0.05 + h) - p0_star(0.05 - h)) / (2 * h)
    assert jnp.isfinite(g)
    assert jnp.isclose(float(g), float(fd), atol=1e-3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu pytest tests/test_codim2.py -v -k double_hopf`
Expected: FAIL — `ImportError: cannot import name 'double_hopf_point'`

- [ ] **Step 3: Append to `src/jaxcont/bifurcations/codim2.py`**

```python
# --------------------------------------------------------------------------
# HH -- double Hopf
# --------------------------------------------------------------------------

def _hh_unpack(x, n):
    o = n + 2
    o2 = o + 2 * n + 1
    return (
        x[:n], x[n:o],
        x[o:o + n], x[o + n:o + 2 * n], x[o + 2 * n],
        x[o2:o2 + n], x[o2 + n:o2 + 2 * n], x[o2 + 2 * n],
    )


def _hopf_block(jac_u, q1, q2, omega, s1, s2):
    """One Hopf block: eigenvector relations, normalization, phase."""
    return jnp.concatenate([
        jac_u @ q1 + omega * q2,                                     # n
        jac_u @ q2 - omega * q1,                                     # n
        jnp.reshape(jnp.dot(q1, q1) + jnp.dot(q2, q2) - 1.0, (1,)),  # 1
        jnp.reshape(jnp.dot(s1, q2) - jnp.dot(s2, q1), (1,)),        # 1
    ])


def _hh_residual(x, f, args, n, seed_a, seed_b):
    """
    Double-Hopf extended system, ``5n+4`` equations in ``5n+4`` unknowns:
    one ``f(u,p) = 0`` plus two independent Hopf blocks, each with its own
    normalization and its own DISTINCT phase seed.
    """
    u, p, q1a, q2a, oa, q1b, q2b, ob = _hh_unpack(x, n)
    jac_u = jacfwd(f, argnums=0)(u, p, args)
    return jnp.concatenate([
        f(u, p, args),
        _hopf_block(jac_u, q1a, q2a, oa, seed_a[0], seed_a[1]),
        _hopf_block(jac_u, q1b, q2b, ob, seed_b[0], seed_b[1]),
    ])


def double_hopf_point(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    seed_b: Array,
    tol: float = 1e-6,
    max_iter: int = 50,
    separation_tolerance: float = 1e-3,
) -> Tuple[Array, ...]:
    """
    Locate a double-Hopf point (``HH``) near ``(u_guess, p_guess)``,
    differentiable in ``args``. Kuznetsov, 3rd ed., Sec. 8.6.

    ``HH`` is where ``f_u`` has TWO distinct pairs of purely imaginary
    eigenvalues. Requires ``n >= 4``. Returns
    ``(u*, p*, q1a, q2a, omega_a, q1b, q2b, omega_b, converged)`` with both
    frequencies normalized non-negative and ``p*`` of shape ``(2,)``.

    ``seed_b`` (required, keyword-only) is a real direction seeding the
    SECOND Hopf block, and it must point at a different physical pair than
    the first. The first block is seeded from the usual eigen-decomposition
    heuristic, which by construction finds only one pair; without an
    independent ``seed_b`` both blocks would converge onto that same pair,
    making the extended system structurally singular. That case is detected
    -- ``converged`` is ``False`` when
    ``abs(|omega_a| - |omega_b|) <= separation_tolerance`` -- rather than
    returned as a plausible-looking "double Hopf" that is one pair counted
    twice.
    """
    u_guess = jnp.asarray(u_guess)
    n = u_guess.shape[0]
    p_guess = jnp.asarray(p_guess, u_guess.dtype)
    seed_b = jnp.asarray(seed_b, u_guess.dtype)

    def G(x, theta):
        # stop_gradient required here -- same reasoning as _gh_residual/
        # _zh_residual (Tasks 4-5): jnp.linalg.eig (inside _hopf_seed) has
        # no gradient rule for non-symmetric eigenvectors, and this seed
        # only feeds the phase condition, recomputed inside the traced
        # primal on every Newton iteration. Task 6's implementer found this
        # a third time (same bug class as Tasks 4-5) -- the gradient test
        # fails with NotImplementedError without it.
        q1_a, q2_a, _ = _hopf_seed(f, u_guess, p_guess, lax.stop_gradient(theta), n)
        # Phase seeds: block A from the eigen-decomposition, block B from
        # the caller-supplied direction and its image under f_u, so the two
        # phase conditions are genuinely independent.
        jac_g = jacfwd(f, argnums=0)(u_guess, p_guess, theta)
        s_b2 = jac_g @ seed_b
        s_b2 = s_b2 / (jnp.linalg.norm(s_b2) + jnp.finfo(u_guess.dtype).eps)
        return _hh_residual(
            x, f, theta, n, (q1_a, q2_a), (seed_b, s_b2)
        )

    def x0(theta):
        q1_a, q2_a, omega_a = _hopf_seed(f, u_guess, p_guess, theta, n)
        jac_g = jacfwd(f, argnums=0)(u_guess, p_guess, theta)
        q2_b = jac_g @ seed_b
        q2_b = q2_b / (jnp.linalg.norm(q2_b) + jnp.finfo(u_guess.dtype).eps)
        # Unit-normalize the (q1, q2) pair as the block's own condition wants.
        scale = jnp.sqrt(jnp.dot(seed_b, seed_b) + jnp.dot(q2_b, q2_b))
        q1_b = seed_b / scale
        q2_b = q2_b / scale
        omega_b = 2.0 * omega_a
        return jnp.concatenate([
            u_guess, p_guess,
            q1_a, q2_a, jnp.reshape(omega_a, (1,)),
            q1_b, q2_b, jnp.reshape(jnp.asarray(omega_b, u_guess.dtype), (1,)),
        ])

    x_star, converged = _solve_and_check(G, x0, args, tol=tol, max_iter=max_iter)
    u, p, q1a, q2a, oa, q1b, q2b, ob = _hh_unpack(x_star, n)
    q2a, oa = _normalize_omega(q2a, oa)
    q2b, ob = _normalize_omega(q2b, ob)
    separated = jnp.abs(oa - ob) > separation_tolerance
    converged = converged & separated
    return u, p, q1a, q2a, oa, q1b, q2b, ob, converged


def double_hopf_parameters(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    seed_b: Array,
    tol: float = 1e-6,
    max_iter: int = 50,
    separation_tolerance: float = 1e-3,
) -> Array:
    """
    Parameter pair ``p*`` (shape ``(2,)``) at the double-Hopf point --
    differentiable in ``args``, no convergence flag (see
    :func:`cusp_parameters`).
    """
    result = double_hopf_point(
        f, u_guess, p_guess, args, seed_b=seed_b, tol=tol, max_iter=max_iter,
        separation_tolerance=separation_tolerance,
    )
    return result[1]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu pytest tests/test_codim2.py -v`
Expected: 20 passed

If `test_double_hopf_point_recovers_exact_shifted_hh` does not converge, the seeding for block B
is the thing to adjust — the residual itself is verified correct (24 = 24 equations, residual
3.1e-14, `cond(J) = 4.0` in planning). Try `omega_b` seeded from the imaginary part of the
eigenvalue whose eigenvector is closest to `seed_b`, instead of the `2.0 * omega_a` placeholder.
Do NOT relax `separation_tolerance` to make a failing case pass; that check is load-bearing.

- [ ] **Step 5: Run the full suite**

Run: `make test`
Expected: 237 passed, 0 failed

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/bifurcations/codim2.py tests/test_codim2.py
git commit -m "feat: add double_hopf_point with pair-separation degeneracy guard"
```

---

### Task 7: Public exports, API docs, and the stale taxonomy docstring

**Files:**
- Modify: `src/jaxcont/__init__.py:45-51` (import block) and its `__all__` list
- Modify: `docs/source/api/index.rst`
- Modify: `src/jaxcont/bifurcations/taxonomy.py:27` (comment only)
- Test: `tests/test_codim2.py` (append)

**Interfaces:**
- Consumes: all eleven public names from Tasks 1-6.
- Produces: `jc.cusp_point`, `jc.cusp_parameters`, `jc.bogdanov_takens_point`,
  `jc.bogdanov_takens_parameters`, `jc.generalized_hopf_point`,
  `jc.generalized_hopf_parameters`, `jc.zero_hopf_point`, `jc.zero_hopf_parameters`,
  `jc.double_hopf_point`, `jc.double_hopf_parameters`, `jc.fold_coefficient`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_codim2.py`:

```python
def test_codim2_functions_are_exported_at_top_level():
    import jaxcont as jc

    from jaxcont.bifurcations.codim2 import (
        bogdanov_takens_parameters, bogdanov_takens_point,
        cusp_parameters, cusp_point,
        double_hopf_parameters, double_hopf_point,
        generalized_hopf_parameters, generalized_hopf_point,
        zero_hopf_parameters, zero_hopf_point,
    )
    from jaxcont.bifurcations.fold_normal_form import fold_coefficient

    assert jc.cusp_point is cusp_point
    assert jc.cusp_parameters is cusp_parameters
    assert jc.bogdanov_takens_point is bogdanov_takens_point
    assert jc.bogdanov_takens_parameters is bogdanov_takens_parameters
    assert jc.generalized_hopf_point is generalized_hopf_point
    assert jc.generalized_hopf_parameters is generalized_hopf_parameters
    assert jc.zero_hopf_point is zero_hopf_point
    assert jc.zero_hopf_parameters is zero_hopf_parameters
    assert jc.double_hopf_point is double_hopf_point
    assert jc.double_hopf_parameters is double_hopf_parameters
    assert jc.fold_coefficient is fold_coefficient


def test_codim2_names_are_listed_in_dunder_all():
    import jaxcont as jc

    for name in (
        "cusp_point", "cusp_parameters",
        "bogdanov_takens_point", "bogdanov_takens_parameters",
        "generalized_hopf_point", "generalized_hopf_parameters",
        "zero_hopf_point", "zero_hopf_parameters",
        "double_hopf_point", "double_hopf_parameters",
        "fold_coefficient",
    ):
        assert name in jc.__all__
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu pytest tests/test_codim2.py -v -k "exported or dunder"`
Expected: FAIL — `AttributeError: module 'jaxcont' has no attribute 'cusp_point'`

- [ ] **Step 3: Update `src/jaxcont/__init__.py`**

Find the existing block (around line 47-51):

```python
# Differentiable Hopf-point solver + first Lyapunov coefficient (Hopf
# criticality) -- see docs/superpowers/specs/2026-08-04-hopf-normal-form-design.md
from jaxcont.bifurcations.hopf_normal_form import (
    hopf_point, hopf_parameter, lyapunov_coefficient,
)
```

and add directly below it:

```python

# Fold normal-form coefficient + direct codim-2 point solvers (cusp,
# Bogdanov-Takens, generalized Hopf, zero-Hopf, double Hopf). These take p
# with shape (2,) -- codim-2 needs two free parameters -- and are
# differentiable in args like their codim-1 siblings above. See
# docs/superpowers/specs/2026-08-05-codim2-direct-solvers-design.md
from jaxcont.bifurcations.fold_normal_form import fold_coefficient
from jaxcont.bifurcations.codim2 import (
    bogdanov_takens_parameters, bogdanov_takens_point,
    cusp_parameters, cusp_point,
    double_hopf_parameters, double_hopf_point,
    generalized_hopf_parameters, generalized_hopf_point,
    zero_hopf_parameters, zero_hopf_point,
)
```

Then in the `__all__` list, find:

```python
    "hopf_point",
    "hopf_parameter",
    "lyapunov_coefficient",
```

and add directly below it:

```python
    "fold_coefficient",
    "cusp_point",
    "cusp_parameters",
    "bogdanov_takens_point",
    "bogdanov_takens_parameters",
    "generalized_hopf_point",
    "generalized_hopf_parameters",
    "zero_hopf_point",
    "zero_hopf_parameters",
    "double_hopf_point",
    "double_hopf_parameters",
```

- [ ] **Step 4: Fix the stale `taxonomy.py` status docstring**

`taxonomy.py`'s `status` field currently reads:

```python
    #: "v0.1" (implemented today), "v0.2"/"v0.3" (planned), or "out of scope".
    status: str
```

This was written before v0.2 shipped and now misdescribes the field — `LC` reads `"v0.2"` and is
implemented. The field holds the version a label *lands in*. Replace that one comment line with:

```python
    #: The JaxCont version this label lands in ("v0.1", "v0.2", "v0.3"), or
    #: "out of scope". Not a boolean: "v0.2" entries are implemented today,
    #: "v0.3" entries are landing in the current development series.
    status: str
```

**Do not change any `BifurcationLabel(...)` data.** `CP`/`BT`/`ZH`/`HH`/`GH` already read `"v0.3"`,
which is correct for work landing in v0.3.

- [ ] **Step 5: Add the API reference section**

In `docs/source/api/index.rst`, find the existing "Differentiable Hopf solver" section (added by
the Hopf normal-form work) and add a new section directly after it, mirroring its structure:

```rst
Codim-2 point solvers
---------------------

Direct solvers for codimension-2 equilibrium bifurcations. These take a
parameter array ``p`` of shape ``(2,)`` (codim-2 needs two free parameters)
and are differentiable in ``args`` via the implicit function theorem, like
their codim-1 counterparts above. Each returns a trailing ``converged``
flag rather than raising.

.. autofunction:: jaxcont.fold_coefficient

.. autofunction:: jaxcont.cusp_point

.. autofunction:: jaxcont.cusp_parameters

.. autofunction:: jaxcont.bogdanov_takens_point

.. autofunction:: jaxcont.bogdanov_takens_parameters

.. autofunction:: jaxcont.generalized_hopf_point

.. autofunction:: jaxcont.generalized_hopf_parameters

.. autofunction:: jaxcont.zero_hopf_point

.. autofunction:: jaxcont.zero_hopf_parameters

.. autofunction:: jaxcont.double_hopf_point

.. autofunction:: jaxcont.double_hopf_parameters
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu pytest tests/test_codim2.py -v`
Expected: 22 passed

- [ ] **Step 7: Run the full suite and build the docs**

Run: `make test`
Expected: 239 passed, 0 failed

Run: `make docs`
Expected: builds with no new warnings about the names added above.

- [ ] **Step 8: Commit**

```bash
git add src/jaxcont/__init__.py src/jaxcont/bifurcations/taxonomy.py \
        docs/source/api/index.rst tests/test_codim2.py
git commit -m "feat: export codim-2 solvers, add API docs, fix stale taxonomy status comment"
```

---

### Task 8: BifurcationKit.jl cross-validation

**Files:**
- Create: `examples/BifurcationKit/05_codim2.jl`
- Test: `tests/test_codim2.py` (append)

**Interfaces:**
- Consumes: `bogdanov_takens_point` and `generalized_hopf_point` (Tasks 3, 4).
- Produces: no new library code — one test asserting agreement with hardcoded reference values.

**Why this task exists:** closed-form normal forms verify the mathematics but cannot catch a
*convention* mismatch. This repo already cross-checks `lyapunov_coefficient` against an
independent Julia run (`04_hopf_normal_form.jl`); codim-2 needs the equivalent.

**Known-unresolved going in:** during planning, a BifurcationKit codim-2 run was attempted on
Bazykin's predator-prey model, but the parameter regime tried exposed only a transcritical point —
no BT or Hopf — so **no reference values exist yet and this task has a genuine research
component.** Budget for iteration. Julia and BifurcationKit v0.5.2 are installed and working
(`julia -e 'using BifurcationKit'` succeeds).

- [ ] **Step 1: Write the Julia reference script**

Create `examples/BifurcationKit/05_codim2.jl`. Start from a system with documented codim-2
structure. Route (a), preferred — a real applied model neither implementation was tuned for:

```julia
# Independent BifurcationKit.jl reference values for JaxCont's codim-2 solvers.
# Run offline:  julia examples/BifurcationKit/05_codim2.jl
# Copy the printed values into tests/test_codim2.py.
using BifurcationKit, LinearAlgebra

# Pick a model with documented BT / GH structure and continue a codim-1
# branch in one parameter, then follow the detected fold/Hopf point into the
# second parameter with codim-2 detection switched on:
#
#   br  = continuation(prob, PALC(), opts)
#   br2 = continuation(br, index, (@optic _.b), opts2;
#                      detect_codim2_bifurcation = 2,
#                      start_with_eigen = true)
#
# then print br2.specialpoint entries whose .type is :bt / :gh / :cusp / :zh.
```

Fill in a concrete model and print `type`, both parameter values, and the state vector. If route
(a) resists after reasonable effort, fall back to route (b): run BifurcationKit on the **shifted
normal forms already used in `tests/test_codim2.py`** (`_bt_shifted`, `_gh_shifted`). That still
cross-checks two independent implementations against each other, and is explicitly sanctioned by
the design spec as the fallback.

- [ ] **Step 2: Run the Julia script and capture reference values**

Run: `julia examples/BifurcationKit/05_codim2.jl`
Record the printed parameter values and state vectors. If nothing is detected, change the model or
the parameter window and iterate — do not fabricate reference numbers, and do not skip to Step 3
with invented values.

- [ ] **Step 3: Write the cross-validation test**

Append to `tests/test_codim2.py`, substituting the ACTUAL values printed in Step 2 for the
`BK_*` constants and the actual model for `_bk_model`:

**If route (a) succeeded**, transcribe the model into JAX and hardcode the printed values:

```python
# Reference values from an independent BifurcationKit.jl v0.5.2 run --
# see examples/BifurcationKit/05_codim2.jl. Regenerate with:
#     julia examples/BifurcationKit/05_codim2.jl
BK_BT_U = (...)   # <-- the state vector printed in Step 2
BK_BT_P = (...)   # <-- the two parameter values printed in Step 2


def _bk_model(u, p, args):
    # The same right-hand side as in 05_codim2.jl, transcribed to JAX.
    # Keep the two files' equations character-for-character comparable so a
    # future reader can verify they are the same system.
    ...


def test_bogdanov_takens_matches_bifurcationkit_jl_independent_run():
    u, p, _, _, ok = bogdanov_takens_point(
        _bk_model,
        jnp.array(BK_BT_U) + 0.05,
        jnp.array(BK_BT_P) + 0.05,
    )
    assert bool(ok)
    assert jnp.allclose(u, jnp.array(BK_BT_U), atol=1e-3)
    assert jnp.allclose(p, jnp.array(BK_BT_P), atol=1e-3)
```

**If route (a) did not pan out**, use this fallback verbatim — it is complete as written, needing
only the two constants from the Julia run. It reuses `_bt_shifted`, already defined in this file
by Task 3, whose BT is at `u*=(5,2)`, `p*=(3,-1)`:

```python
# BifurcationKit.jl v0.5.2 run on the SAME shifted Bogdanov-Takens system
# used above (examples/BifurcationKit/05_codim2.jl). Ground truth is already
# known analytically here, so this checks the two implementations against
# each other rather than establishing the answer -- the fallback route
# sanctioned by the design spec when a real applied model proves intractable.
BK_BT_U = (5.0, 2.0)   # <-- replace with the values Julia actually printed
BK_BT_P = (3.0, -1.0)  # <-- replace with the values Julia actually printed


def test_bogdanov_takens_matches_bifurcationkit_jl_independent_run():
    u, p, _, _, ok = bogdanov_takens_point(
        _bt_shifted, jnp.array([5.3, 1.7]), jnp.array([2.6, -0.8]),
    )
    assert bool(ok)
    assert jnp.allclose(u, jnp.array(BK_BT_U), atol=1e-3)
    assert jnp.allclose(p, jnp.array(BK_BT_P), atol=1e-3)
```

Replace the placeholder constants with what Julia printed even in the fallback — if BifurcationKit
disagrees with the analytic values, that disagreement is the finding and must not be hidden by
writing the analytic numbers in.

- [ ] **Step 4: Run the test**

Run: `JAX_PLATFORMS=cpu pytest tests/test_codim2.py -v -k bifurcationkit`
Expected: PASS. A mismatch here is a real finding — investigate whether it is a convention
difference (sign/normalization) or a genuine bug before adjusting either side.

- [ ] **Step 5: Run the full suite**

Run: `make test`
Expected: 240 passed, 0 failed

- [ ] **Step 6: Commit**

```bash
git add examples/BifurcationKit/05_codim2.jl tests/test_codim2.py
git commit -m "test: cross-validate codim-2 solvers against BifurcationKit.jl"
```

---

### Task 9: Roadmap update

**Files:**
- Modify: `notes/ROADMAP.md`

- [ ] **Step 1: Check off the codim-2 item**

In the `## v0.3.0+ — Advanced (demand-driven)` section, change

```markdown
- [ ] Codim-2 bifurcations (cusp, Bogdanov-Takens, GH/generalized Hopf, ...) — `hopf_normal_form.py`
      above is now the prerequisite building block (per its design spec) but no codim-2 detection
      exists yet.
```

to a checked entry following the style of every other completed roadmap item: what shipped, what
was found/fixed during verification, and what was explicitly left out. Include at minimum:

- The five solvers plus `fold_coefficient`, and that `p` has shape `(2,)`.
- Direct point solves, **not** two-parameter continuation (still its own open item).
- The four planning findings: omega's sign is unconstrained (post-solve normalization required);
  HH degenerates to `nan` when both pairs collapse (hence the separation guard); BT's textbook
  left/right-null-vector system is overdetermined so the Jordan-chain form is used; and the
  textbook normal forms have no discriminating power because their codim-2 point sits at the
  origin, so all tests use shifted systems.
- The measured float32 residual floors (BT `0.0`, CP `≈4.4e-8`, GH `≈5.96e-08`) and the resulting
  `tol=1e-6` default.
- Whichever BifurcationKit cross-validation route Task 8 actually landed, stated honestly.
- Explicitly deferred: codim-2 normal-form coefficients (BT's `(a,b)`, GH's `l2`), `Event`
  integration, branch switching, and the cycle codim-2 labels.

Also update the `**Last updated:**` date at the top of the file.

- [ ] **Step 2: Commit**

```bash
git add notes/ROADMAP.md
git commit -m "docs: mark codim-2 direct point solvers done in ROADMAP"
```

---

## Notes for the implementer

**Things verified during planning — trust these, do not re-derive:**

| Claim | Evidence |
|---|---|
| All five extended systems are square | BT 8=8, GH 9=9, ZH 15=15, HH 24=24 |
| They converge from perturbed guesses | residuals 1e-13 to 1e-26 (float64) |
| Conditioning is benign | `cond(J)` 2.0–9.5; no preconditioning needed |
| BT's textbook w-form is wrong | `3n+3` equations for `3n+2` unknowns |
| omega's sign is unconstrained | GH/ZH/HH converged to negative omega from positive seeds |
| BT's `v1` seed must be nonzero | `zeros` → `nan`; `ones/sqrt(n)` → exact |
| GH works with the repo's real `lyapunov_coefficient` | residual 5.96e-08, exact `p*` |
| GH's float32 floor is ~6e-8 | `tol=1e-8` → `converged=False` despite an exact answer |
| Bordered solve ≡ SVD for the left null vector | agree to float32 precision, both grads finite |
| Origin-centred tests do not discriminate | a `return zeros` stub passes them; shifted ones fail it |

**Things NOT verified — expect to iterate:**

- HH's exact seeding strategy (Task 6 Step 4 has the fallback).
- Whether ZH/HH converge on any *non-decoupled* system. The planning systems were block-diagonal
  by construction. If a realistic coupled system proves hard to seed, that is worth recording in
  the roadmap rather than papering over.
- BifurcationKit codim-2 reference values (Task 8 — genuinely open).
