"""
Fixed-mesh Gauss-Legendre orthogonal collocation building blocks for
periodic-orbit continuation -- see
docs/superpowers/specs/2026-07-24-periodic-orbit-collocation-design.md.

Pure numerics -- no BifProblem/API concerns here, mirroring
core/scan_continuation.py's role as the engine's pure-numerics layer.
"""

from __future__ import annotations

import equinox as eqx
import numpy as np


class Collocation(eqx.Module):
    """Fixed collocation mesh config: ntst subintervals x ncol Gauss-Legendre
    points per subinterval. Both static (compile-time constants, since they
    fix the collocation unknown vector's shape for jit). The mesh itself is
    uniform (mesh point i is at tau=i/ntst), so it is derived on the fly
    rather than stored as a field. No adaptive mesh redistribution -- ntst/
    ncol are fixed for the lifetime of a continuation run (see design spec's
    explicit scope cut)."""

    ntst: int = eqx.field(static=True)
    ncol: int = eqx.field(static=True)


def gauss_legendre_01(ncol: int):
    """Gauss-Legendre nodes/weights of degree ``ncol`` on ``[0, 1]`` (mapped
    from the standard ``[-1, 1]`` via an affine transform)."""
    x, w = np.polynomial.legendre.leggauss(ncol)
    nodes = 0.5 * (x + 1.0)
    weights = 0.5 * w
    return nodes, weights


def lagrange_diff_matrix(nodes: np.ndarray) -> np.ndarray:
    """``(m, m)`` Lagrange differentiation matrix for ``nodes`` (any 1D
    array): ``D[j, k] = L_k'(nodes[j])``, where ``L_k`` is the k-th Lagrange
    basis polynomial for these nodes. For nodal values ``v`` of a
    degree-<m polynomial, ``D @ v`` gives its derivative at each node
    exactly."""
    m = len(nodes)
    D = np.zeros((m, m))
    for k in range(m):
        others = [nodes[i] for i in range(m) if i != k]
        denom = np.prod([nodes[k] - o for o in others])
        for j in range(m):
            xj = nodes[j]
            s = 0.0
            for i in range(m):
                if i == k:
                    continue
                term = 1.0
                for l in range(m):
                    if l == k or l == i:
                        continue
                    term *= (xj - nodes[l])
                s += term
            D[j, k] = s / denom
    return D


def lagrange_eval_weights(nodes: np.ndarray, x: float) -> np.ndarray:
    """Weight vector ``w`` such that ``w @ v`` evaluates the Lagrange
    interpolant through ``(nodes, v)`` at ``x`` (used to extrapolate each
    collocation interval's polynomial to its right endpoint, ``x=1``, for
    the continuity/periodicity equations)."""
    m = len(nodes)
    w = np.zeros(m)
    for k in range(m):
        Lk = 1.0
        for i in range(m):
            if i == k:
                continue
            Lk *= (x - nodes[i]) / (nodes[k] - nodes[i])
        w[k] = Lk
    return w


def collocation_matrices(ncol: int):
    """Precompute the local ``(ncol+1, ncol+1)`` differentiation matrix
    ``D``, the ``(ncol+1,)`` right-endpoint extrapolation weights ``E``, the
    ``(ncol,)`` interior Gauss-Legendre nodes ``gauss``, and the ``(ncol,)``
    quadrature weights ``gw`` for a degree-``ncol`` collocation scheme.
    Local node 0 is the left mesh point (x=0); local nodes 1..ncol are the
    interior Gauss-Legendre points. Pure numpy -- ``ncol`` is a Python int
    (static), so this is meant to be called once at problem-construction
    time and its results closed over as jax.jit-time constants, not
    traced."""
    gauss, gw = gauss_legendre_01(ncol)
    local_nodes = np.concatenate([[0.0], gauss])
    D = lagrange_diff_matrix(local_nodes)
    E = lagrange_eval_weights(local_nodes, 1.0)
    return D, E, gauss, gw
