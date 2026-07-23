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
