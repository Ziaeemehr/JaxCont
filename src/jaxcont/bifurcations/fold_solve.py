"""
Differentiable fold (saddle-node) solver via the extended system + implicit diff.

A fold of ``f(u, p; args) = 0`` is the solution of the extended system

    G1:  f(u, p)          = 0            (n eqs)   equilibrium
    G2:  f_u(u, p) · v    = 0            (n eqs)   singular Jacobian (null vector v)
    G3:  vᵀv - 1          = 0            (1 eq)    normalization

in the unknowns ``x = (u, p, v)`` (dimension ``2n + 1``). We solve it with Newton
and wrap the solve in :func:`jax.custom_vjp` so that the fold — in particular the
fold **parameter** ``p*`` — is a *reverse-mode differentiable* function of the
design parameters ``args``. The gradient comes from the implicit function
theorem (``dx*/dθ = -G_x⁻¹ G_θ``), so it is exact and does **not** differentiate
through Newton's iterations — which is what lets it work despite the inner
``while_loop`` (ARCHITECTURE.md §3.2).

Public entry points:
- :func:`fold_point`     -> (u*, p*, v*), differentiable in ``args``
- :func:`fold_parameter` -> p*,            differentiable in ``args``  (grad-ready)
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array, jacfwd, lax

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


def _make_fold_solver(f, u_guess, p_guess, n, tol, max_iter):
    """Build a ``custom_vjp`` solver ``args -> x*`` for fixed guesses/shapes."""

    def G(x, args):
        return _extended_residual(x, f, args, n)

    def newton(args):
        v0 = _initial_v(f, u_guess, p_guess, args, n)
        x0 = _pack(u_guess, p_guess, v0)

        def cond(carry):
            x, it, done = carry
            return jnp.logical_and(jnp.logical_not(done), it < max_iter)

        def body(carry):
            x, it, _ = carry
            r = G(x, args)
            J = jax.jacobian(G, argnums=0)(x, args)
            dx = jnp.linalg.solve(J, -r)
            x_new = x + dx
            r_new = G(x_new, args)
            done = jnp.logical_or(
                jnp.linalg.norm(r_new) < tol,
                jnp.logical_not(jnp.all(jnp.isfinite(r_new))),
            )
            return x_new, it + 1, done

        x_star, _, _ = lax.while_loop(cond, body, (x0, 0, jnp.array(False)))
        return x_star

    @jax.custom_vjp
    def solve(args):
        return newton(args)

    def solve_fwd(args):
        x_star = newton(args)
        return x_star, (x_star, args)

    def solve_bwd(res, x_bar):
        x_star, args = res
        # implicit function theorem:  dx*/dθ = -G_x⁻¹ G_θ
        #   args_bar = -(G_θ)ᵀ G_xᵀ⁻¹ x_bar
        Gx = jax.jacobian(G, argnums=0)(x_star, args)
        y = jnp.linalg.solve(Gx.T, x_bar)
        _, vjp_args = jax.vjp(lambda a: G(x_star, a), args)
        (args_bar,) = vjp_args(-y)
        return (args_bar,)

    solve.defvjp(solve_fwd, solve_bwd)
    return solve


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
    solve = _make_fold_solver(f, u_guess, p_guess, n, tol, max_iter)
    x_star = solve(args)
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
