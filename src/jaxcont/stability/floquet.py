"""
Floquet multipliers for periodic-orbit stability, via the collocation
monodromy matrix -- see
docs/superpowers/specs/2026-07-24-floquet-multipliers-design.md.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from jaxcont.core.collocation import Collocation, collocation_matrices, monodromy_matrix
from jaxcont.solvers.protocols import DenseEigen, EigenSolver

PyTree = Any


def floquet_multipliers(
    raw_f: Callable[[Array, Array, PyTree], Array],
    mesh: Collocation,
    U: Array,
    p: Array,
    eigen_solver: EigenSolver = DenseEigen(),
    *,
    args: PyTree = None,
) -> tuple[Array, Array]:
    """Floquet multipliers (eigenvalues of the monodromy matrix ``Phi(T)``)
    at one converged periodic-orbit branch point. ``U`` is the flat
    collocation unknown vector (same layout as a periodic ``BifProblem``'s
    ``u0``: ``ntst`` mesh-point states, then ``ntst*ncol`` collocation-point
    states, then the period ``T``). ``raw_f`` is the ODE right-hand side
    called as ``raw_f(u, p, args)`` with the caller's ``args`` (the
    ``PeriodicArgs.user_args`` of the problem), not the assembled collocation
    residual.
    ``mesh`` has no ``n`` (state dimension) field, so it is derived
    algebraically from ``U``'s length. Returns ``(multipliers, trivial_index)``;
    the index selects the eigenvector most aligned with ``raw_f(x0, p, args)``."""
    ntst, ncol = mesh.ntst, mesh.ncol
    n = (U.shape[-1] - 1) // (ntst * (1 + ncol))
    h = 1.0 / ntst

    D_np, E_np, _, _ = collocation_matrices(ncol)
    D, E = jnp.asarray(D_np), jnp.asarray(E_np)

    mesh_states = U[: ntst * n].reshape(ntst, n)
    coll_states = U[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
    T = U[-1]

    Phi = monodromy_matrix(raw_f, D, E, h, mesh_states, coll_states, T, p, args)
    # Mode identity is discrete; multiplier derivatives use the supplied solver.
    values, vectors = jnp.linalg.eig(jax.lax.stop_gradient(Phi))
    flow = raw_f(mesh_states[0], p, args)
    # Eigenvector columns are normalized independently; complex phase cancels.
    alignment = jnp.abs(jnp.conj(vectors).T @ flow) / jnp.linalg.norm(vectors, axis=0)
    trivial = jnp.argmax(alignment)
    multipliers = eigen_solver(Phi)
    return multipliers, jnp.argmin(jnp.abs(multipliers - values[trivial]))


def branch_floquet_multipliers(
    raw_f: Callable[[Array, Array, PyTree], Array],
    mesh: Collocation,
    states: Array,
    params: Array,
    eigen_solver: EigenSolver = DenseEigen(),
    *,
    args: PyTree = None,
) -> tuple[Array, Array]:
    """Vectorized (vmap) Floquet multipliers along a stored periodic branch
    returning ``(multipliers, trivial_index)`` with shapes ``(N, n)``, ``(N,)``
    -- the periodic-orbit analogue of
    ``core.scan_continuation.branch_eigenvalues``."""
    def at(U, p):
        return floquet_multipliers(raw_f, mesh, U, p, eigen_solver, args=args)

    return jax.vmap(at)(states, params)


def floquet_stable(multipliers: Array, trivial_index: Array) -> Array:
    """Stability per point, excluding the supplied flow-direction multiplier.

    ``multipliers`` has shape ``(N, n)`` and ``trivial_index`` shape ``(N,)``.
    All remaining multipliers must lie strictly inside the unit circle.
    """
    is_trivial = jnp.arange(multipliers.shape[-1]) == trivial_index[..., None]
    return jnp.all(is_trivial | (jnp.abs(multipliers) < 1.0), axis=-1)
