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
) -> Array:
    """Floquet multipliers (eigenvalues of the monodromy matrix ``Phi(T)``)
    at one converged periodic-orbit branch point. ``U`` is the flat
    collocation unknown vector (same layout as a periodic ``BifProblem``'s
    ``u0``: ``ntst`` mesh-point states, then ``ntst*ncol`` collocation-point
    states, then the period ``T``). ``raw_f`` is the ODE right-hand side
    (``args=None`` internally), not the assembled collocation residual.
    ``mesh`` has no ``n`` (state dimension) field, so it is derived
    algebraically from ``U``'s length."""
    ntst, ncol = mesh.ntst, mesh.ncol
    n = (U.shape[-1] - 1) // (ntst * (1 + ncol))
    h = 1.0 / ntst

    D_np, E_np, _, _ = collocation_matrices(ncol)
    D, E = jnp.asarray(D_np), jnp.asarray(E_np)

    mesh_states = U[: ntst * n].reshape(ntst, n)
    coll_states = U[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
    T = U[-1]

    Phi = monodromy_matrix(raw_f, D, E, h, mesh_states, coll_states, T, p)
    return eigen_solver(Phi)


def branch_floquet_multipliers(
    raw_f: Callable[[Array, Array, PyTree], Array],
    mesh: Collocation,
    states: Array,
    params: Array,
    eigen_solver: EigenSolver = DenseEigen(),
) -> Array:
    """Vectorized (vmap) Floquet multipliers along a stored periodic branch
    -- the periodic-orbit analogue of
    ``core.scan_continuation.branch_eigenvalues``."""
    def at(U, p):
        return floquet_multipliers(raw_f, mesh, U, p, eigen_solver)

    return jax.vmap(at)(states, params)


def floquet_stable(multipliers: Array) -> Array:
    """``(n_valid,)`` stability booleans from ``(n_valid, n)`` Floquet
    multipliers. A periodic orbit always has exactly one trivial multiplier
    equal to ``1`` (tangent to the flow) -- identified per-point as the one
    closest to ``1`` (``argmin(|multiplier - 1|)``) and excluded. Stability
    is a magnitude condition on the remaining multipliers (inside the unit
    circle), unlike equilibria's real-part condition."""
    def stable_at(row: Array) -> Array:
        trivial_idx = jnp.argmin(jnp.abs(row - 1.0))
        is_trivial = jnp.arange(row.shape[0]) == trivial_idx
        return jnp.all(jnp.where(is_trivial, True, jnp.abs(row) < 1.0))

    return jax.vmap(stable_at)(multipliers)
