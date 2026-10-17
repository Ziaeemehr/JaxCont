"""
Periodic Schur decomposition of ordered interval factors, see
docs/superpowers/specs/2026-10-17-periodic-schur-floquet-design.md.

For factors ``M_0, ..., M_(m-1)`` (cyclic product ``M_(m-1) @ ... @ M_0``)
every routine works on the factors themselves and never forms their product.
All decompositions satisfy ``M_i Q_i = Q_(i+1) H_i`` with unitary ``Q_i`` and
cyclic index ``Q_m = Q_0``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def _complex_dtype(dtype):
    return jnp.result_type(dtype, jnp.complex64)


def _householder(x: Array, start: Array) -> tuple[Array, Array]:
    """Reflector ``P = I - beta v v^H`` mapping ``x[start:]`` onto ``e_start``.

    ``v`` vanishes outside ``start:``; ``P`` is the identity (``beta = 0``)
    when ``x[start + 1:]`` is already zero. The vector is scaled by its largest
    entry before any norm is formed.
    """
    n = x.shape[0]
    idx = jnp.arange(n)
    masked = jnp.where(idx >= start, x, jnp.zeros_like(x))
    scale = jnp.max(jnp.abs(masked))
    safe_scale = jnp.where(scale > 0, scale, jnp.ones_like(scale))
    y = masked / safe_scale
    alpha = y[start]
    sub_norm = jnp.linalg.norm(jnp.where(idx > start, y, jnp.zeros_like(y)))
    abs_alpha = jnp.abs(alpha)
    norm = jnp.sqrt(abs_alpha**2 + sub_norm**2)
    sign = jnp.where(abs_alpha > 0, alpha / jnp.where(abs_alpha > 0, abs_alpha, 1), 1)
    v = y + jnp.where(idx == start, sign * norm, jnp.zeros_like(y))
    vnorm2 = jnp.sum(jnp.abs(v) ** 2)
    active = sub_norm > 0
    beta = jnp.where(active, 2.0 / jnp.where(active, vnorm2, 1), 0).astype(x.dtype)
    return v, beta


def _reflect_left(A: Array, v: Array, beta: Array) -> Array:
    return A - beta * jnp.outer(v, jnp.conj(v) @ A)


def _reflect_right(A: Array, v: Array, beta: Array) -> Array:
    return A - beta * jnp.outer(A @ v, jnp.conj(v))


def _triangularize_factor(H: Array, Q: Array, i, start) -> tuple[Array, Array]:
    """Zero columns ``start:n-1`` of factor ``i`` below its diagonal.

    Each left reflector of ``H_i`` is a right reflector of ``Q_(i+1)`` and of
    ``H_(i+1)``, so ``M_i Q_i = Q_(i+1) H_i`` is preserved for every ``i``.
    Requires ``i < m - 1`` so that the two factors are distinct.
    """
    m, n = H.shape[0], H.shape[1]
    nxt = (i + 1) % m
    rows = jnp.arange(n)

    def column(j, carry):
        H, Q = carry
        Hi = H[i]
        v, beta = _householder(Hi[:, j], j)
        Hi = _reflect_left(Hi, v, beta)
        Hi = Hi.at[:, j].set(jnp.where(rows > j, jnp.zeros_like(Hi[:, j]), Hi[:, j]))
        H = H.at[i].set(Hi)
        H = H.at[nxt].set(_reflect_right(H[nxt], v, beta))
        Q = Q.at[nxt].set(_reflect_right(Q[nxt], v, beta))
        return H, Q

    return jax.lax.fori_loop(start, n - 1, column, (H, Q))


def periodic_hessenberg(factors: Array) -> tuple[Array, Array]:
    """Periodic Hessenberg-triangular reduction of ``(m, n, n)`` factors.

    Returns complex ``(H, Q)`` of the same shape with
    ``factors[i] @ Q[i] = Q[(i + 1) % m] @ H[i]``, unitary ``Q[i]``,
    ``H[0], ..., H[m-2]`` upper triangular and ``H[m-1]`` upper Hessenberg.
    Entries annihilated by a reflector are stored as exact zeros.
    """
    H = jnp.asarray(factors).astype(_complex_dtype(jnp.asarray(factors).dtype))
    m, n = H.shape[0], H.shape[1]
    Q = jnp.broadcast_to(jnp.eye(n, dtype=H.dtype), H.shape)
    rows = jnp.arange(n)
    last = m - 1

    def triangularize_all(start, carry):
        def factor(i, carry):
            return _triangularize_factor(carry[0], carry[1], i, start)

        return jax.lax.fori_loop(0, last, factor, carry)

    H, Q = triangularize_all(0, (H, Q))

    def hessenberg_column(k, carry):
        H, Q = carry
        v, beta = _householder(H[last][:, k], k + 1)
        Hl = _reflect_left(H[last], v, beta)
        Hl = Hl.at[:, k].set(jnp.where(rows > k + 1, jnp.zeros_like(Hl[:, k]), Hl[:, k]))
        H = H.at[last].set(Hl)
        H = H.at[0].set(_reflect_right(H[0], v, beta))
        Q = Q.at[0].set(_reflect_right(Q[0], v, beta))
        return triangularize_all(k + 1, (H, Q))

    return jax.lax.fori_loop(0, max(n - 2, 0), hessenberg_column, (H, Q))
