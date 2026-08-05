"""
Fixed-mesh Gauss-Legendre orthogonal collocation building blocks for
periodic-orbit continuation -- see
docs/superpowers/specs/2026-07-24-periodic-orbit-collocation-design.md.

Pure numerics -- no BifProblem/API concerns here, mirroring
core/scan_continuation.py's role as the engine's pure-numerics layer.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
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


def interval_propagators(raw_f, D: "jnp.ndarray", E: "jnp.ndarray", h: float, mesh_states, coll_states, T, p):
    """``(ntst, n, n)`` per-interval propagator blocks ``M_i``, such that
    ``Phi(T) = M_{ntst-1} @ ... @ M_0``. Extracted from ``monodromy_matrix``
    (unchanged math) so ``stability/prc.py`` can adjoint-propagate a vector
    across each interval individually, not just consume the composed
    endpoint map -- see
    docs/superpowers/specs/2026-08-05-prc-dprc-design.md."""
    ntst, n = mesh_states.shape
    ncol = coll_states.shape[1]
    eye_n = jnp.eye(n)

    def interval_map(mesh_state_i, coll_states_i):
        Jm = jax.vmap(jax.jacfwd(lambda u: raw_f(u, p, None)))(coll_states_i)  # (ncol, n, n)

        def build_A_row(m):
            def build_block(k):
                coeff = D[m + 1, k + 1]
                block = coeff * eye_n
                return jnp.where(k == m, block - T * h * Jm[m], block)

            return jax.vmap(build_block)(jnp.arange(ncol))

        A_blocks = jax.vmap(build_A_row)(jnp.arange(ncol))  # (ncol, ncol, n, n)
        A = jnp.transpose(A_blocks, (0, 2, 1, 3)).reshape(ncol * n, ncol * n)
        b0 = (-D[1:, 0][:, None, None] * eye_n[None, :, :]).reshape(ncol * n, n)
        S = jnp.linalg.solve(A, b0).reshape(ncol, n, n)
        return E[0] * eye_n + jnp.sum(E[1:][:, None, None] * S, axis=0)

    return jax.vmap(interval_map)(mesh_states, coll_states)


def monodromy_matrix(raw_f, D: "jnp.ndarray", E: "jnp.ndarray", h: float, mesh_states, coll_states, T, p):
    """``(n, n)`` monodromy matrix ``Phi(T)`` -- see ``interval_propagators``
    for the per-interval blocks this composes. Behavior/numerics unchanged
    from before the ``interval_propagators`` extraction; see
    docs/superpowers/specs/2026-07-24-floquet-multipliers-design.md and
    docs/superpowers/specs/2026-08-05-prc-dprc-design.md."""
    n = mesh_states.shape[1]
    M_all = interval_propagators(raw_f, D, E, h, mesh_states, coll_states, T, p)
    Phi, _ = jax.lax.scan(lambda carry, M: (M @ carry, None), jnp.eye(n), M_all)
    return Phi
