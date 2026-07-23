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
