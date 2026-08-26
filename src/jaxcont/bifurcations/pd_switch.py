"""
Period-doubling branch switching -- see
docs/superpowers/specs/2026-08-27-period-doubling-switching-design.md.

Switches from a detected PeriodDoubling event onto the period-doubled cycle
that bifurcates there. The corrector is the existing
``problems/periodic.py:periodic_orbit_problem`` used unchanged: this module
only builds the seed and verifies the result.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxcont.core.collocation import (
    Collocation,
    collocation_matrices,
    monodromy_matrix,
)

PyTree = Any


def _unpack(U: Array, mesh: Collocation):
    """Split a periodic ``U`` into ``(mesh_states, coll_states, T, n)``.

    Mirrors ``stability/floquet.py:floquet_multipliers``: ``mesh`` carries no
    state dimension, so ``n`` is derived algebraically from ``U``'s length.
    """
    ntst, ncol = mesh.ntst, mesh.ncol
    n = (U.shape[-1] - 1) // (ntst * (1 + ncol))
    mesh_states = U[: ntst * n].reshape(ntst, n)
    coll_states = U[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
    return mesh_states, coll_states, U[-1], n


def _pd_eigenvector(
    raw_f: Callable[[Array, Array, PyTree], Array],
    U: Array,
    p: Array,
    mesh: Collocation,
    *,
    pd_tol: float = 1e-2,
    imag_tol: float = 1e-3,
) -> Array:
    """The real, unit-norm ``-1`` Floquet eigenvector at a period-doubling point.

    Recomputes the monodromy matrix rather than reading a cached eigenvector
    off the event: ``PeriodDoubling.refine`` does not store one, and
    ``branch_switch_pd`` already needs ``raw_f``/``mesh`` for the seed, so
    recomputation costs one small ``n x n`` eigendecomposition and keeps
    ``events.py`` untouched.

    ``pd_tol`` is deliberately loose (1e-2): the event point is a bisection
    midpoint sitting at the float32 collocation residual floor, not a
    Newton-refined point.
    """
    mesh_states, coll_states, T, _n = _unpack(U, mesh)
    D_np, E_np, _, _ = collocation_matrices(mesh.ncol)
    D, E = jnp.asarray(D_np), jnp.asarray(E_np)
    Phi = monodromy_matrix(
        raw_f, D, E, 1.0 / mesh.ntst, mesh_states, coll_states, T, p
    )
    values, vectors = jnp.linalg.eig(Phi)
    idx = int(jnp.argmin(jnp.abs(values + 1.0)))
    lam = values[idx]
    if float(jnp.abs(lam + 1.0)) > pd_tol or float(jnp.abs(jnp.imag(lam))) > imag_tol:
        raise ValueError(
            "not a period-doubling point: the Floquet multiplier nearest -1 is "
            f"{complex(lam):.6g} (needs |lambda + 1| < {pd_tol} and "
            f"|Im lambda| < {imag_tol}). Check that this event came from a "
            "jc.PeriodDoubling() detection on a periodic-orbit branch."
        )
    v = jnp.real(vectors[:, idx])
    norm = float(jnp.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError(
            "the -1 Floquet eigenvector is numerically zero; cannot build a "
            "switching direction."
        )
    return v / norm


def _doubling_measure(U: Array, mesh: Collocation, n: int) -> float:
    """How much the orbit differs from its own half-period shift, relative to
    its scale.

    On a uniform mesh of ``ntst`` points over the doubled period ``2T``, the
    shift ``t -> t + T`` is an index roll of ``ntst // 2`` (``ntst`` is even by
    construction, so the roll is exact). A genuine period-doubled orbit differs
    from that shift; the original orbit traversed twice is identical to it.
    This is the only signal that separates the two -- an under-sized seed
    converges onto the retraced original with a perfectly healthy residual.
    """
    mesh_states = U[: mesh.ntst * n].reshape(mesh.ntst, n)
    shifted = jnp.roll(mesh_states, mesh.ntst // 2, axis=0)
    spread = float(jnp.max(jnp.linalg.norm(mesh_states - shifted, axis=1)))
    scale = float(jnp.max(jnp.linalg.norm(mesh_states, axis=1)))
    if scale < 1e-12:
        return 0.0
    return spread / scale
