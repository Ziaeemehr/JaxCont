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
