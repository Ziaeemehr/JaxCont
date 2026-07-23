"""
Pluggable linear/eigen solver protocols (ARCHITECTURE.md §4.6).

Dense/DenseEigen are the only concrete implementations for now -- direct
LAPACK calls via jnp.linalg. The protocol boundary exists so a future
GMRES()/Arnoldi() (large systems) or ChebyshevDDE() (ARCHITECTURE.md §10.2,
the DDE eigensolver seam) can swap in without touching the continuation
loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array


@runtime_checkable
class LinearSolver(Protocol):
    """Solves ``A @ x = b`` for ``x``."""

    def __call__(self, A: Array, b: Array) -> Array: ...


@runtime_checkable
class EigenSolver(Protocol):
    """Returns the eigenvalues of ``A``."""

    def __call__(self, A: Array) -> Array: ...


@dataclass(frozen=True)
class Dense:
    """Default LinearSolver: ``jnp.linalg.solve``.

    No fields -- the dataclass machinery gives value-based __eq__/__hash__
    for free, which is required for this to be a safe jax.jit static
    argument (two independently-constructed Dense() instances must
    compare/hash equal, or every call with a fresh default would force a
    recompile).
    """

    def __call__(self, A: Array, b: Array) -> Array:
        return jnp.linalg.solve(A, b)


@dataclass(frozen=True)
class DenseEigen:
    """Default EigenSolver: ``jnp.linalg.eigvals``. No fields -- see Dense."""

    def __call__(self, A: Array) -> Array:
        return jnp.linalg.eigvals(A)
