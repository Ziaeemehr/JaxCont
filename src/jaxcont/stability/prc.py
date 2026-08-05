"""
Infinitesimal phase response curves (iPRC) via the collocation adjoint
method -- see docs/superpowers/specs/2026-08-05-prc-dprc-design.md.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from jaxcont.api import BifProblem
from jaxcont.core.collocation import Collocation, collocation_matrices, interval_propagators
from jaxcont.solvers.implicit import differentiable_root
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


def branch_prc(
    raw_f: Callable[[Array, Array, PyTree], Array],
    mesh: Collocation,
    states: Array,
    params: Array,
    linear_solver: LinearSolver = Dense(),
) -> Array:
    """Vectorized (vmap) iPRC curves along a stored periodic branch --
    the PRC analogue of ``stability.floquet.branch_floquet_multipliers``."""
    def at(U, p):
        return prc_curve(raw_f, mesh, U, p, linear_solver)

    return jax.vmap(at)(states, params)


def dprc_curve(
    problem: "BifProblem",
    linear_solver: LinearSolver = Dense(),
    newton_tol: float = 1e-5,
) -> Array:
    """Parameter derivative of the iPRC curve, ``d(prc_curve)/dp``, shape
    ``(ntst, n) + p.shape``. Takes the periodic orbit's ``BifProblem``
    (residual ``problem.f``, seed ``problem.u0``, collocation bookkeeping
    ``problem.args``) rather than ``prc_curve``'s ``(raw_f, mesh, U, p)`` --
    it must differentiate through a re-solve of ``U(p)`` via
    ``differentiable_root`` (the same call
    ``problems.periodic.periodic_orbit_problem`` already makes to build its
    own ``u0``), not through ``prc_curve`` alone at a fixed ``U``: the latter
    is differentiable but not physically meaningful (see
    docs/superpowers/specs/2026-08-05-prc-dprc-design.md, "Design findings
    from prototyping").

    The re-solve's phase-condition anchor (the ``uref_prime_coll`` component
    of ``problem.args``) is recomputed from ``p`` at every Newton step,
    exactly the way ``periodic_orbit_problem`` itself derives it
    (``f(coll_guess, p0, None)``) -- it is *not* held frozen at
    ``problem.args``'s baked-in value from the original construction (which
    is how ``jc.continuation()`` treats ``args`` across an ordinary branch,
    but is not what this function's re-solve needs, since the re-solve is
    meant to reproduce what a *fresh* ``periodic_orbit_problem`` call would
    build at the perturbed ``p``). Confirmed empirically: holding the anchor
    frozen disagrees with a finite difference of independently re-converged
    orbits by up to ~0.2 absolute; recomputing it from ``p`` matches that
    finite difference to ~1e-4."""
    u_ref_coll, _uref_prime_coll0, raw_f, mesh = problem.args

    def anchor_at(p: Array) -> PyTree:
        uref_prime_coll_p = jax.vmap(jax.vmap(lambda u: raw_f(u, p, None)))(u_ref_coll)
        return (u_ref_coll, uref_prime_coll_p, raw_f, mesh)

    def prc_at(p: Array) -> Array:
        U_p = differentiable_root(
            lambda U, pp: problem.f(U, pp, anchor_at(pp)),
            problem.u0,
            p,
            tol=newton_tol,
        )
        return prc_curve(raw_f, mesh, U_p, p, linear_solver)

    # jax.jacrev, not jax.jacfwd: differentiable_root is built on
    # jax.custom_vjp (reverse-mode only -- see solvers/implicit.py), so
    # forward-mode AD (what jax.jacfwd needs) raises "can't apply
    # forward-mode autodiff (jvp) to a custom_vjp function". jacrev gives
    # the identical Jacobian via the implemented reverse-mode path (p0 is
    # scalar here, so there's no efficiency loss from the mode switch).
    return jax.jacrev(prc_at)(problem.p0)
