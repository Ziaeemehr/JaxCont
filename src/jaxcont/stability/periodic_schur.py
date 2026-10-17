"""
Periodic Schur decomposition of ordered interval factors, see
docs/superpowers/specs/2026-10-17-periodic-schur-floquet-design.md.

For factors ``M_0, ..., M_(m-1)`` (cyclic product ``M_(m-1) @ ... @ M_0``)
every routine works on the factors themselves and never forms their product.
All decompositions satisfy ``M_i Q_i = Q_(i+1) H_i`` with unitary ``Q_i`` and
cyclic index ``Q_m = Q_0``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
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
    # A zero map breaks the cyclic basis constraint: start immediately after
    # it, so sequential QR triangularizes every nonzero map independently.
    zero_factor = jnp.all(H == 0, axis=(1, 2))
    offset = jnp.where(jnp.any(zero_factor), jnp.argmax(zero_factor) + 1, 0)
    order = (jnp.arange(m) + offset) % m
    H = H[order]
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

    H, Q = jax.lax.fori_loop(0, max(n - 2, 0), hessenberg_column, (H, Q))
    inverse = (jnp.arange(m) - offset) % m
    return H[inverse], Q[inverse]


_ORDER_RATIO = 1e-2


def _real_eps(dtype) -> float:
    return float(jnp.finfo(jnp.finfo(dtype).dtype).eps)


def _givens(a: Array, b: Array) -> Array:
    """Unitary ``G = [[c, s], [-conj(s), c]]`` (real ``c``) with
    ``G @ [a, b] = [r, 0]``; the identity when ``a = b = 0``. Scale-safe."""
    scale = jnp.maximum(jnp.abs(a), jnp.abs(b))
    nonzero = scale > 0
    safe = jnp.where(nonzero, scale, jnp.ones_like(scale))
    a, b = a / safe, b / safe
    abs_a = jnp.abs(a)
    norm = jnp.sqrt(abs_a**2 + jnp.abs(b) ** 2)
    safe_norm = jnp.where(nonzero, norm, jnp.ones_like(norm))
    a_phase = jnp.where(abs_a > 0, a / jnp.where(abs_a > 0, abs_a, 1), 1)
    c = jnp.where(nonzero, abs_a / safe_norm, 1).astype(a.dtype)
    s = jnp.where(nonzero, a_phase * jnp.conj(b) / safe_norm, 0)
    return jnp.array([[c, s], [-jnp.conj(s), c]])


def _rotate_rows(A: Array, p, G: Array) -> Array:
    rows = jax.lax.dynamic_slice_in_dim(A, p, 2, axis=0)
    return jax.lax.dynamic_update_slice_in_dim(A, G @ rows, p, axis=0)


def _rotate_cols(A: Array, p, G: Array) -> Array:
    cols = jax.lax.dynamic_slice_in_dim(A, p, 2, axis=1)
    return jax.lax.dynamic_update_slice_in_dim(A, cols @ jnp.conj(G).T, p, axis=1)


def _deflate(H_last: Array, eps: float) -> Array:
    """Zero negligible subdiagonals of the Hessenberg factor.

    ``h[k+1,k]`` is negligible when, after scaling it and its neighbouring
    diagonals ``h[k,k], h[k+1,k+1]`` by their largest magnitude,
    ``|h[k+1,k]| <= 100*eps*(|h[k,k]| + |h[k+1,k+1]|)``: three zeros deflate,
    a nonzero subdiagonal between two zero diagonals does not. There is no
    absolute floor, so tiny physical modes are kept.
    """
    n = H_last.shape[0]
    k = jnp.arange(n - 1)
    sub = jnp.abs(H_last[k + 1, k])
    d0, d1 = jnp.abs(H_last[k, k]), jnp.abs(H_last[k + 1, k + 1])
    scale = jnp.maximum(sub, jnp.maximum(d0, d1))
    safe = jnp.where(scale > 0, scale, jnp.ones_like(scale))
    negligible = sub / safe <= 100 * eps * (d0 / safe + d1 / safe)
    return H_last.at[k + 1, k].set(jnp.where(negligible, 0, H_last[k + 1, k]))


def _active_window(H_last: Array) -> tuple[Array, Array]:
    """Rows ``lo..hi`` of the bottom-most unreduced block (``hi = 0`` if none)."""
    n = H_last.shape[0]
    k = jnp.arange(n - 1)
    zero = H_last[k + 1, k] == 0
    hi = jnp.max(jnp.where(zero, -1, k), initial=-1) + 1
    lo = jnp.max(jnp.where(zero & (k <= hi - 2), k + 1, 0), initial=0)
    return lo, hi


def _scaled_term(z: Array, log_scale: Array, ref: Array) -> Array:
    """``z * exp(log_scale - ref)`` evaluated through ``log|z|``."""
    abs_z = jnp.abs(z)
    phase = jnp.where(abs_z > 0, z / jnp.where(abs_z > 0, abs_z, 1), 0)
    exponent = jnp.log(abs_z) + log_scale - ref
    return phase * jnp.exp(jnp.where(jnp.isnan(exponent), -jnp.inf, exponent))


def _shift(H: Array, hi, stalled) -> tuple[Array, Array]:
    """Shift ``mu * exp(log_scale)`` from the trailing ``2x2`` blocks at rows
    ``hi-1, hi``. Their product is accumulated with renormalization; only
    this scaled small block is formed, never the full product.

    The block eigenvalue nearest its last diagonal entry is the shift
    (Wilkinson), unless the other block eigenvalue is more than
    ``1/_ORDER_RATIO`` times smaller in modulus; then the shift is zero, a
    periodic orthogonal-iteration step converging at that ratio. The
    leading ``k+1`` columns of every ``Q_i`` span the forward propagation of
    those of ``Q_0``, and a block can only deflate in factored form when that
    propagation is stable, i.e. when no much smaller multiplier sits above a
    much larger one.
    After ``10`` sweeps without a deflation the shift is exceptional:
    alternately zero and the last diagonal entry plus ``0.75`` times the
    subdiagonal modulus."""
    blocks = jax.vmap(lambda A: jax.lax.dynamic_slice(A, (hi - 1, hi - 1), (2, 2)))(H)

    def accumulate(carry, block):
        B, log_scale = carry
        B = block @ B
        size = jnp.max(jnp.abs(B))
        positive = size > 0
        B = B / jnp.where(positive, size, 1)
        return (B, log_scale + jnp.log(jnp.where(positive, size, 1))), None

    real = jnp.finfo(H.dtype).dtype
    init = (jnp.eye(2, dtype=H.dtype), jnp.zeros((), dtype=real))
    (B, log_scale), _ = jax.lax.scan(accumulate, init, blocks)
    a, b, c, d = B[0, 0], B[0, 1], B[1, 0], B[1, 1]
    half = 0.5 * (a - d)
    disc = jnp.sqrt(half * half + b * c)
    mean = 0.5 * (a + d)
    plus, minus = mean + disc, mean - disc
    plus_nearer = jnp.abs(plus - d) <= jnp.abs(minus - d)
    nearest = jnp.where(plus_nearer, plus, minus)
    other = jnp.where(plus_nearer, minus, plus)
    reorder = jnp.abs(other) < _ORDER_RATIO * jnp.abs(nearest)
    mu = jnp.where(reorder, 0, nearest)
    exceptional = stalled % 10 == 9
    ad_hoc = (stalled // 10) % 2 == 1
    mu = jnp.where(exceptional, jnp.where(ad_hoc, d + 0.75 * jnp.abs(c), 0), mu)
    return mu, log_scale


def _first_rotation_vector(H: Array, lo, mu: Array, log_mu: Array) -> tuple[Array, Array]:
    """Direction of ``(Pi - sigma I) e_lo`` restricted to rows ``lo, lo+1``,
    for the cyclic product ``Pi`` and shift ``sigma = mu*exp(log_mu)``.

    ``Pi e_lo = t * H_last[:, lo]`` with ``t`` the product of the triangular
    diagonals at ``lo``, kept as phase and log-magnitude."""
    H_last = H[-1]
    h0 = H_last[lo, lo]
    h1 = H_last[lo + 1, lo]
    diag = H[:-1, lo, lo]
    abs_diag = jnp.abs(diag)
    log_t = jnp.sum(jnp.log(abs_diag))
    phase_t = jnp.prod(jnp.where(abs_diag > 0, diag / jnp.where(abs_diag > 0, abs_diag, 1), 0))
    column_log = log_t + jnp.log(jnp.maximum(jnp.abs(h0), jnp.abs(h1)))
    ref = jnp.maximum(column_log, log_mu + jnp.log(jnp.abs(mu)))
    ref = jnp.where(jnp.isfinite(ref), ref, 0)
    x0 = phase_t * _scaled_term(h0, log_t, ref) - _scaled_term(mu, log_mu, ref)
    x1 = phase_t * _scaled_term(h1, log_t, ref)
    return x0, x1


def _periodic_qr_sweep(H: Array, Q: Array, lo, hi, x0: Array, x1: Array) -> tuple[Array, Array]:
    """One implicit single-shift periodic QR step on rows ``lo..hi``.

    A rotation of ``Q_i`` multiplies ``H_i`` from the right and ``H_(i-1)``
    from the left; the bulge is chased through every factor at each plane.
    Rotations outside the window are identities."""
    m, n = H.shape[0], H.shape[1]
    last = m - 1
    eye2 = jnp.eye(2, dtype=H.dtype)

    def plane(p, carry):
        H, Q = carry
        active = (p >= lo) & (p < hi)
        first = p == lo
        prev = jnp.maximum(p - 1, 0)
        H_last = H[last]
        a = jnp.where(first, x0, H_last[p, prev])
        b = jnp.where(first, x1, H_last[p + 1, prev])
        G = jnp.where(active, _givens(a, b), eye2)
        H_last = _rotate_rows(H_last, p, G)
        clear = active & ~first
        H_last = H_last.at[p + 1, prev].set(jnp.where(clear, 0, H_last[p + 1, prev]))
        H = H.at[last].set(H_last)
        H = H.at[0].set(_rotate_cols(H[0], p, G))
        Q = Q.at[0].set(_rotate_cols(Q[0], p, G))

        def chase(i, carry):
            H, Q = carry
            Hi = H[i]
            G = jnp.where(active, _givens(Hi[p, p], Hi[p + 1, p]), eye2)
            Hi = _rotate_rows(Hi, p, G)
            Hi = Hi.at[p + 1, p].set(jnp.where(active, 0, Hi[p + 1, p]))
            H = H.at[i].set(Hi)
            H = H.at[i + 1].set(_rotate_cols(H[i + 1], p, G))
            Q = Q.at[i + 1].set(_rotate_cols(Q[i + 1], p, G))
            return H, Q

        return jax.lax.fori_loop(0, last, chase, (H, Q))

    return jax.lax.fori_loop(0, n - 1, plane, (H, Q))


def _diagonal_spectrum(T: Array) -> tuple[Array, Array]:
    """``phase_j = prod_i T_i[j,j]/|T_i[j,j]|`` (zero for a zero entry) and
    ``logabs_j = sum_i log|T_i[j,j]|`` (``-inf`` for a zero entry)."""
    diag = jnp.diagonal(T, axis1=1, axis2=2)
    abs_diag = jnp.abs(diag)
    logabs = jnp.sum(jnp.log(abs_diag), axis=0)
    unit = jnp.where(abs_diag > 0, diag / jnp.where(abs_diag > 0, abs_diag, 1), 0)
    phase = jnp.prod(unit, axis=0)
    abs_phase = jnp.abs(phase)
    phase = jnp.where(abs_phase > 0, phase / jnp.where(abs_phase > 0, abs_phase, 1), 0)
    return phase, logabs


def _cyclic_backward_ok(factors: Array, T: Array, Q: Array, eps: float) -> Array:
    """Relative cyclic reconstruction and orthogonality errors below ``500*eps``."""
    n = factors.shape[1]
    Q_next = jnp.roll(Q, -1, axis=0)
    residual = jnp.linalg.norm(factors @ Q - Q_next @ T, axis=(1, 2))
    scale = jnp.linalg.norm(factors, axis=(1, 2))
    reconstruct = residual <= 500 * eps * scale
    eye = jnp.eye(n, dtype=Q.dtype)
    orthogonality = jnp.linalg.norm(jnp.conj(jnp.swapaxes(Q, 1, 2)) @ Q - eye, axis=(1, 2))
    return jnp.all(reconstruct & (orthogonality <= 500 * eps))


def periodic_schur(
    factors: Array, *, max_sweeps: int | None = None
) -> tuple[Array, Array, Array, Array, Array]:
    """Complex periodic Schur decomposition of ``(m, n, n)`` factors.

    Returns ``(T, Q, phase, logabs, converged)``: unitary ``Q`` and upper
    triangular ``T`` with ``factors[i] @ Q[i] = Q[(i+1) % m] @ T[i]``; the
    multipliers of the cyclic product are ``phase * exp(logabs)`` read from
    the diagonals of ``T`` (``phase = 0``, ``logabs = -inf`` for an exactly
    zero multiplier). ``max_sweeps`` (static, default ``200*n``) bounds the
    shifted periodic QR steps. ``converged`` is true only when every
    subdiagonal deflated and the cyclic reconstruction and orthogonality
    errors are below ``500*eps``; otherwise ``T`` is not triangular and the
    spectrum is not valid.
    """
    factors = jnp.asarray(factors)
    m, n = factors.shape[0], factors.shape[1]
    budget = 200 * n if max_sweeps is None else int(max_sweeps)
    H, Q = periodic_hessenberg(factors)
    eps = _real_eps(H.dtype)
    last = m - 1
    H = H.at[last].set(_deflate(H[last], eps))

    def unfinished(state):
        H, _, sweeps, _, _ = state
        _, hi = _active_window(H[last])
        return (sweeps < budget) & (hi > 0)

    def step(state):
        H, Q, sweeps, stalled, previous_hi = state
        lo, hi = _active_window(H[last])
        stalled = jnp.where(hi == previous_hi, stalled + 1, 0)
        mu, log_mu = _shift(H, hi, stalled)
        x0, x1 = _first_rotation_vector(H, lo, mu, log_mu)
        H, Q = _periodic_qr_sweep(H, Q, lo, hi, x0, x1)
        H = H.at[last].set(_deflate(H[last], eps))
        return H, Q, sweeps + 1, stalled, hi

    zero = jnp.asarray(0)
    state = (H, Q, zero, zero, jnp.asarray(-1))
    if n > 1:
        state = jax.lax.while_loop(unfinished, step, state)
    T, Q = state[0], state[1]
    _, hi = _active_window(T[last])
    converged = (hi == 0) & _cyclic_backward_ok(factors.astype(T.dtype), T, Q, eps)
    phase, logabs = _diagonal_spectrum(T)
    return T, Q, phase, logabs, converged


def _polar(x: Array) -> tuple[Array, Array]:
    magnitude = jnp.abs(x)
    return x / jnp.where(magnitude > 0, magnitude, 1), jnp.log(magnitude)


def _cyclic_log_solve(a, b, c, gain_phase, gain_log, *, backward=False):
    """Solve a scalar cyclic recurrence without forming local ratios.

    Forward equations are b_i*x_(i+1)=a_i*x_i+c_i; backward equations
    are b_i*x_i=a_i*x_(i+1)+c_i. Each phase's cyclic forcing sum uses
    signed log magnitudes, so cancelling local growth stays representable.
    Returns phase and log magnitude of x, before common coordinate scaling.
    """
    ap, al = _polar(a)
    bp, bl = _polar(b)
    cp, cl = _polar(c)
    gp, gl = ap * jnp.conj(bp), al - bl
    ep, el = cp * jnp.conj(bp), cl - bl
    m = a.shape[0]
    target, offset = jnp.arange(m)[:, None], jnp.arange(m)[None, :]
    indices = ((target + offset) if backward else (target - 1 - offset)) % m
    gphase, glog = gp[indices], gl[indices]
    prefix_phase = jnp.concatenate([jnp.ones((m, 1), a.dtype),
                                    jnp.cumprod(gphase[:, :-1], axis=1)], axis=1)
    prefix_log = jnp.concatenate([jnp.zeros((m, 1), al.dtype),
                                  jnp.cumsum(glog[:, :-1], axis=1)], axis=1)
    terms_log = el[indices] + prefix_log
    scale = jnp.max(terms_log, axis=1)
    scale = jnp.where(jnp.isfinite(scale), scale, 0)
    total = jnp.sum(ep[indices] * prefix_phase * jnp.exp(terms_log-scale[:, None]), axis=1)
    total_phase, total_log = _polar(total)
    # Scale 1-gain before subtraction, including the unused growing direction.
    gain_scale = jnp.maximum(gain_log, 0)
    denominator = jnp.exp(-gain_scale) - gain_phase*jnp.exp(gain_log-gain_scale)
    denom_phase, denom_log = _polar(denominator)
    return total_phase*jnp.conj(denom_phase), total_log+scale-denom_log-gain_scale


def _insert_log_coordinate(coordinates, k, phase, logabs, active):
    # One common scale preserves every existing recurrence and its pivot.
    scale = jnp.where(active, jnp.maximum(jnp.max(logabs), 0), 0)
    scale = jnp.where(jnp.isfinite(scale), scale, 0)
    updated = coordinates*jnp.exp(-scale)
    column = phase*jnp.exp(logabs-scale)
    return updated.at[:, k].set(jnp.where(active, column, updated[:, k]))


def _schur_right_vector(T: Array, phase: Array, logabs: Array, j) -> Array:
    """Schur coordinates ``y`` (``(m, n)``) of mode ``j``: ``T_i y_i = t_i y_(i+1)``
    with ``t_i = T_i[j, j]`` and ``y_i[k] = 0`` for ``k > j``. The pivot
    starts at one; common rescaling across phases keeps coordinates bounded.

    Row ``k < j`` is a cyclic first-order recurrence across the phases; it is
    run in the direction whose cyclic gain has modulus at most one (forward
    when ``|lambda_k| < |lambda_j|``)."""
    m, n = T.shape[0], T.shape[1]
    cols = jnp.arange(n)
    b = T[:, j, j]
    y = jnp.zeros((m, n), dtype=T.dtype).at[:, j].set(1)

    def row(idx, y):
        k = n - 1 - idx
        a = T[:, k, k]
        c = jnp.sum(T[:, k, :] * jnp.where(cols > k, y, 0), axis=-1)
        ratio_phase = phase[k] * jnp.conj(phase[j])
        ratio_log = logabs[k] - logabs[j]
        fp, fl = _cyclic_log_solve(a, b, c, ratio_phase, ratio_log)
        bp, bl = _cyclic_log_solve(b, a, -c, jnp.conj(ratio_phase), -ratio_log,
                                  backward=True)
        forward = logabs[k] < logabs[j]
        return _insert_log_coordinate(y, k, jnp.where(forward, fp, bp),
                                      jnp.where(forward, fl, bl), k < j)

    return jax.lax.fori_loop(0, n, row, y)


def _schur_left_vector(T: Array, phase: Array, logabs: Array, j) -> Array:
    """Schur coordinates ``z`` of the left mode ``j``:
    ``z_(i+1)^H T_i = t_i z_i^H``, ``z_i[k] = 0`` for ``k < j``; the pivot
    starts at one with common rescaling across phases. Rows ``k > j`` are
    solved in the direction whose cyclic gain has modulus at most one."""
    m, n = T.shape[0], T.shape[1]
    rows = jnp.arange(n)
    b = jnp.conj(T[:, j, j])
    z = jnp.zeros((m, n), dtype=T.dtype).at[:, j].set(1)

    def column(k, z):
        a = jnp.conj(T[:, k, k])
        z_next = jnp.roll(z, -1, axis=0)
        c = jnp.sum(jnp.conj(T[:, :, k]) * jnp.where(rows < k, z_next, 0), axis=-1)
        ratio_phase = jnp.conj(phase[k]) * phase[j]
        ratio_log = logabs[k] - logabs[j]
        bp, bl = _cyclic_log_solve(a, b, c, ratio_phase, ratio_log, backward=True)
        fp, fl = _cyclic_log_solve(b, a, -c, jnp.conj(ratio_phase), -ratio_log)
        backward = logabs[k] < logabs[j]
        return _insert_log_coordinate(z, k, jnp.where(backward, bp, fp),
                                      jnp.where(backward, bl, fl), k > j)

    return jax.lax.fori_loop(0, n, column, z)


def periodic_eigenvectors(
    factors: Array, T: Array, Q: Array, phase: Array, logabs: Array
) -> tuple[Array, Array]:
    """Right and left periodic eigenvectors from a converged ``periodic_schur``.

    Returns ``(right, left)``, each ``(m, n, n)`` with ``right[i, :, j]`` the
    mode of diagonal position ``j`` at phase ``i``. Each ``right[i, :, j]`` has
    unit norm and ``left[i, :, j]^H right[i, :, j] = 1``; with local
    multipliers ``d_i = left[i+1]^H M_i right[i]`` (product ``phase_j *
    exp(logabs_j)``), ``M_i r_i = d_i r_(i+1)`` and
    ``M_i^H l_(i+1) = conj(d_i) l_i``. Schur vectors are not eigenvectors:
    they come from periodic triangular back substitution. Modes are defined
    only for simple multipliers; repeated or zero multipliers give
    non-finite or meaningless vectors. ``factors`` are the decomposed
    factors and fix the working dtype.
    """
    T = jnp.asarray(T).astype(_complex_dtype(jnp.result_type(factors, T)))
    Q = jnp.asarray(Q).astype(T.dtype)
    modes = jnp.arange(T.shape[1])
    y = jax.vmap(lambda j: _schur_right_vector(T, phase, logabs, j))(modes)
    z = jax.vmap(lambda j: _schur_left_vector(T, phase, logabs, j))(modes)
    Y = jnp.einsum("iab,jib->iaj", Q, y)
    Z = jnp.einsum("iab,jib->iaj", Q, z)
    # Scale before squaring; keeping the scale separate also avoids forming
    # an overflowing norm when the final unit vector is representable.
    size = jnp.max(jnp.abs(Y), axis=1, keepdims=True)
    Y = Y / size
    alpha = jnp.linalg.norm(Y, axis=1, keepdims=True)
    right = Y / alpha
    # Triangular supports overlap only at the pivot. This pairing avoids
    # cancellation in highly nonnormal modes after the unitary transform.
    pivot_y = jnp.diagonal(y, axis1=0, axis2=2)[:, None, :]
    pivot_z = jnp.diagonal(z, axis1=0, axis2=2)[:, None, :]
    left = (Z / pivot_z) * (size * alpha / jnp.conj(pivot_y))
    return right, left


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _spectrum(max_sweeps, factors):
    T, Q, phase, logabs, converged = periodic_schur(factors, max_sweeps=max_sweeps)
    right, _ = periodic_eigenvectors(factors, T, Q, phase, logabs)
    return phase, logabs, converged, right[0]


@_spectrum.defjvp
def _spectrum_jvp(max_sweeps, primals, tangents):
    (factors,), (dfactors,) = primals, tangents
    T, Q, phase, logabs, converged = periodic_schur(factors, max_sweeps=max_sweeps)
    right, left = periodic_eigenvectors(factors, T, Q, phase, logabs)
    M = factors.astype(T.dtype)
    left_next = jnp.conj(jnp.roll(left, -1, axis=0))
    local = jnp.einsum("ikj,ikl,ilj->ij", left_next, M, right)
    weights = left_next[:, :, None, :] * right[:, None, :, :] / local[:, None, None, :]
    simple = jnp.all(jnp.isfinite(weights), axis=(0, 1, 2)) & converged
    weights = jnp.where(simple, weights, 0)
    dlog = jnp.einsum("iklj,ikl->j", weights, dfactors.astype(T.dtype))
    invalid = jnp.where(converged, jnp.nan, 0).astype(logabs.dtype)
    dlogabs = jnp.where(simple, jnp.real(dlog), invalid)
    dphase = jnp.where(simple, 1j * jnp.imag(dlog) * phase, invalid.astype(phase.dtype))
    dconverged = np.zeros(converged.shape, dtype=jax.dtypes.float0)
    # Phase-zero right modes identify modes; identity is not differentiated.
    return ((phase, logabs, converged, right[0]),
            (dphase, dlogabs, dconverged, jnp.zeros_like(right[0])))


def periodic_spectrum(
    factors: Array, *, max_sweeps: int | None = None
) -> tuple[Array, Array, Array]:
    """Differentiable ``(phase, logabs, converged)`` of the cyclic product of
    ``(m, n, n)`` factors, as returned by ``periodic_schur``.

    For a simple multiplier ``lambda`` with periodic modes ``r_i, l_i``
    (``l_i^H r_i = 1``) the JVP is ``d log(lambda) = sum_i
    l_(i+1)^H dM_i r_i / d_i`` with local multipliers ``d_i``, so
    ``d logabs = Re`` and ``d phase = 1j * Im * phase``; it is linear in the
    factor tangents and transposes for reverse mode. The QR iteration,
    deflation and mode order are not differentiated. Repeated, defective
    or zero multipliers have no derivative (NaN tangent); a non-converged
    decomposition has zero tangents.
    """
    return _spectrum(max_sweeps, jnp.asarray(factors))[:3]


def periodic_spectrum_modes(
    factors: Array, *, max_sweeps: int | None = None
) -> tuple[Array, Array, Array, Array]:
    """``periodic_spectrum`` plus the ``(n, n)`` phase-zero right modes.

    Column ``j`` is the unit right eigenvector of the cyclic product for
    multiplier ``j`` (non-finite for repeated or zero multipliers). The
    modes carry zero tangents: they select mode identity only.
    """
    return _spectrum(max_sweeps, jnp.asarray(factors))
