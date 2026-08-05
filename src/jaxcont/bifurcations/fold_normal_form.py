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
