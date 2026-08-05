"""
Infinitesimal phase response curves (iPRC) via the collocation adjoint
method -- see docs/superpowers/specs/2026-08-05-prc-dprc-design.md.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from jaxcont.core.collocation import Collocation, collocation_matrices, interval_propagators
from jaxcont.solvers.protocols import Dense, LinearSolver

PyTree = Any


def prc_curve(
    raw_f: Callable[[Array, Array, PyTree], Array],
    mesh: Collocation,
    U: Array,
    p: Array,
    linear_solver: LinearSolver = Dense(),
) -> Array:
    """Infinitesimal PRC ``Z(t)`` (shape ``(ntst, n)``, one row per mesh
    point) at one converged periodic-orbit branch point, via the adjoint
    method: ``Z(0)`` is the left-eigenvector of the monodromy matrix
    ``Phi(T)`` for eigenvalue 1, found via a bordered linear system rather
    than eigendecomposition (``jnp.linalg.eig`` has no gradient rule for
    general non-symmetric matrices -- see ``dprc_curve`` in this module),
    normalized by ``Z(0) . f(x_0, p) = omega = 2*pi/T``. The rest of the
    period is filled in by adjoint-propagating backward through the same
    per-interval propagator blocks ``interval_propagators`` builds for
    ``monodromy_matrix``: ``Z_i = M_i^T @ Z_{i+1}``. ``U``/``raw_f`` follow
    the same conventions as ``stability.floquet.floquet_multipliers``."""
    ntst, ncol = mesh.ntst, mesh.ncol
    n = (U.shape[-1] - 1) // (ntst * (1 + ncol))
    h = 1.0 / ntst

    D_np, E_np, _, _ = collocation_matrices(ncol)
    D, E = jnp.asarray(D_np), jnp.asarray(E_np)

    mesh_states = U[: ntst * n].reshape(ntst, n)
    coll_states = U[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
    T = U[-1]

    M_all = interval_propagators(raw_f, D, E, h, mesh_states, coll_states, T, p)
    Phi, _ = jax.lax.scan(lambda carry, M: (M @ carry, None), jnp.eye(n), M_all)

    x0 = mesh_states[0]
    f0 = raw_f(x0, p, None)
    omega = 2 * jnp.pi / T

    A_top = jnp.concatenate([Phi.T - jnp.eye(n), f0[:, None]], axis=1)
    A_bot = jnp.concatenate([f0[None, :], jnp.zeros((1, 1))], axis=1)
    A_border = jnp.concatenate([A_top, A_bot], axis=0)
    b_border = jnp.concatenate([jnp.zeros(n), omega[None]])
    Z0 = linear_solver(A_border, b_border)[:n]

    def step(Z_next, M_i):
        Z_i = M_i.T @ Z_next
        return Z_i, Z_i

    _, Z_rest_rev = jax.lax.scan(step, Z0, M_all[::-1])
    return Z_rest_rev[::-1]
