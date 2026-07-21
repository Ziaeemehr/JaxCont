"""
Fully JIT-compiled pseudo-arclength continuation (whole-loop).

Unlike the class-based predictor-corrector (`pseudo_arclength.py`), which runs a
Python outer loop dispatching many small JAX ops per step, this expresses the
*entire* continuation sweep as a single ``lax.while_loop`` over fixed-size
buffers. Consequences:

- **One dispatched program per run** — no per-step Python/host-sync overhead.
- **`vmap`-able** — a batch of continuations (parameter sweeps, multistart,
  ensembles) compiles to one kernel; ideal for GPU.
- **Bounded and hang-proof** — the loop runs at most ``max_steps`` iterations,
  each with at most ``max_iter`` Newton iterations, with an explicit finiteness
  guard, so a degenerate/saturating branch terminates cleanly instead of
  stalling.

Everything operates on a pure ``f(u, p, args) -> residual``. The continuation
parameter is the explicit scalar ``p``; ``args`` is a static PyTree closed over.

Design note (adaptivity): a jittable fixed-iteration loop cannot "retry without
consuming a step". Here a rejected step shrinks ``ds`` and re-enters the loop
without advancing the write index, so a rejection *does* consume one of the
``max_steps`` iterations. This is a deliberate trade of unbounded retries for
determinism/jittability; make ``max_steps`` generous.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import Array, jacfwd, lax

PyTree = Any


class ScanResult(NamedTuple):
    """Fixed-length buffers from a jitted continuation run."""

    params: Array        # (max_steps + 1,)
    states: Array        # (max_steps + 1, n)
    tangents: Array      # (max_steps + 1, n + 1)
    converged: Array     # (max_steps + 1,) bool  (step accepted)
    ds: Array            # (max_steps + 1,) step size used to reach each point
    n_valid: Array       # scalar int; entries [:n_valid] are real points


# ---------------------------------------------------------------------------
# Building blocks (pure; operate on f(u, p))
# ---------------------------------------------------------------------------

def _bordered_matrix(jac_u: Array, df_dp: Array, last_row: Array) -> Array:
    """Assemble the (n+1, n+1) bordered matrix with ``last_row`` at the bottom."""
    n = jac_u.shape[0]
    top = jnp.concatenate([jac_u, df_dp.reshape(n, 1)], axis=1)      # (n, n+1)
    bottom = last_row.reshape(1, n + 1)                              # (1, n+1)
    return jnp.concatenate([top, bottom], axis=0)


def _tangent(f, u, p, prev_tangent):
    """
    Keller tangent: the null vector of ``[df/du | df/dp]`` aligned with the
    previous tangent, normalized. Solving the bordered system with ``prev`` as
    the last row (rhs picking the last unit) both selects the null direction and
    orients it continuously — and stays well-posed at folds (where df/du alone
    is singular).
    """
    jac_u = jacfwd(f, argnums=0)(u, p)          # (n, n)
    df_dp = jacfwd(f, argnums=1)(u, p)          # (n,)
    M = _bordered_matrix(jac_u, df_dp, prev_tangent)
    rhs = jnp.zeros(u.shape[0] + 1).at[-1].set(1.0)
    t = jnp.linalg.solve(M, rhs)
    t = t / jnp.linalg.norm(t)
    # keep continuous orientation
    t = jnp.where(jnp.dot(t, prev_tangent) < 0.0, -t, t)
    return t


def _newton_correct(f, u_pred, p_pred, u_prev, p_prev, du0, dp0, ds, tol, max_iter):
    """
    Bordered Newton corrector for the pseudo-arclength system

        f(u, p) = 0
        g(u, p) = (u - u_prev)·du0 + (p - p_prev)·dp0 - ds = 0

    Solves the full (n+1, n+1) bordered system each iteration (well-posed through
    folds), inside a bounded ``while_loop`` with a finiteness guard. Returns
    ``(u, p, converged, iters)``; ``converged`` requires a finite, small residual.
    """

    def residual(u, p):
        f_val = f(u, p)
        g_val = jnp.dot(u - u_prev, du0) + (p - p_prev) * dp0 - ds
        return f_val, g_val

    def res_norm(f_val, g_val):
        return jnp.sqrt(jnp.sum(f_val ** 2) + g_val ** 2)

    def cond_fun(carry):
        _, _, it, done, _ = carry
        return jnp.logical_and(jnp.logical_not(done), it < max_iter)

    def body(carry):
        u, p, it, _, _ = carry
        f_val, g_val = residual(u, p)
        jac_u = jacfwd(f, argnums=0)(u, p)
        df_dp = jacfwd(f, argnums=1)(u, p)
        M = _bordered_matrix(jac_u, df_dp, jnp.concatenate([du0, dp0.reshape(1)]))
        rhs = -jnp.concatenate([f_val, g_val.reshape(1)])
        delta = jnp.linalg.solve(M, rhs)
        u_new = u + delta[:-1]
        p_new = p + delta[-1]
        f_new, g_new = residual(u_new, p_new)
        r_new = res_norm(f_new, g_new)
        # stop iterating once converged OR the iterate went non-finite
        converged = r_new < tol
        blew_up = jnp.logical_not(jnp.isfinite(r_new))
        done = jnp.logical_or(converged, blew_up)
        return u_new, p_new, it + 1, done, r_new

    f0, g0 = residual(u_pred, p_pred)
    r0 = res_norm(f0, g0)
    init = (u_pred, p_pred, 0, r0 < tol, r0)
    u_f, p_f, it_f, _, r_f = lax.while_loop(cond_fun, body, init)
    converged = jnp.logical_and(
        r_f < tol,
        jnp.logical_and(jnp.all(jnp.isfinite(u_f)), jnp.isfinite(p_f)),
    )
    return u_f, p_f, converged, it_f


def _adapt_ds(ds_mag, iters, converged, ds_min, ds_max):
    """Grow ds on fast convergence, shrink on slow/failed — branch-free."""
    grow = ds_mag * 1.5
    shrink_slow = ds_mag * 0.8
    shrink_fail = ds_mag * 0.5
    new = jnp.where(
        converged,
        jnp.where(iters < 3, grow, jnp.where(iters > 6, shrink_slow, ds_mag)),
        shrink_fail,
    )
    return jnp.clip(new, ds_min, ds_max)


# ---------------------------------------------------------------------------
# Whole-loop continuation
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(0, 8))
def pseudo_arclength_scan(
    f: Callable[[Array, Array], Array],
    u0: Array,
    p0: Array,
    p_end: Array,
    ds0: Array,
    ds_min: Array,
    ds_max: Array,
    tol: Array,
    max_steps: int,
    max_iter: Array,
) -> ScanResult:
    """
    Continue ``f(u, p) = 0`` in ``p`` from ``(u0, p0)`` toward ``p_end``.

    ``f`` and ``max_steps`` are static (they set the compiled program & buffer
    sizes); everything else may be a traced array, so this whole function is
    ``jit``/``vmap``/``grad``-friendly.
    """
    u0 = jnp.asarray(u0)
    n = u0.shape[0]
    dtype = u0.dtype
    p0 = jnp.asarray(p0, dtype)
    p_end = jnp.asarray(p_end, dtype)
    direction = jnp.sign(p_end - p0)

    # Initial tangent: seed prev with the parameter axis pointing in `direction`,
    # so the branch is traversed toward p_end.
    seed = jnp.zeros(n + 1, dtype).at[-1].set(direction)
    tan0 = _tangent(f, u0, p0, seed)

    # Fixed-size output buffers; slot 0 is the initial point.
    P = jnp.zeros((max_steps + 1, n), dtype).at[0].set(u0)
    Q = jnp.zeros((max_steps + 1,), dtype).at[0].set(p0)
    T = jnp.zeros((max_steps + 1, n + 1), dtype).at[0].set(tan0)
    C = jnp.zeros((max_steps + 1,), dtype=bool).at[0].set(True)

    ds_mag0 = jnp.asarray(ds0, dtype)
    D = jnp.zeros((max_steps + 1,), dtype).at[0].set(ds_mag0)

    class Carry(NamedTuple):
        u: Array
        p: Array
        tan: Array
        ds: Array         # positive magnitude; direction lives in the tangent
        idx: Array        # int; number of accepted points so far (write pointer)
        stop: Array       # bool
        P: Array
        Q: Array
        T: Array
        C: Array
        D: Array

    def cond_fun(c: Carry):
        return jnp.logical_and(c.idx < max_steps, jnp.logical_not(c.stop))

    def body(c: Carry):
        du0 = c.tan[:-1]
        dp0 = c.tan[-1]

        # Predict along the tangent, then correct.
        u_pred = c.u + c.ds * du0
        p_pred = c.p + c.ds * dp0
        u_new, p_new, converged, iters = _newton_correct(
            f, u_pred, p_pred, c.u, c.p, du0, dp0, c.ds, tol, max_iter
        )

        # New tangent only meaningful if we accept; compute anyway (branch-free).
        tan_new = _tangent(f, u_new, p_new, c.tan)

        write = c.idx + 1  # slot for the next accepted point
        P = c.P.at[write].set(jnp.where(converged, u_new, c.P[write]))
        Q = c.Q.at[write].set(jnp.where(converged, p_new, c.Q[write]))
        T = c.T.at[write].set(jnp.where(converged, tan_new, c.T[write]))
        C = c.C.at[write].set(converged)
        D = c.D.at[write].set(jnp.where(converged, c.ds, c.D[write]))

        # Accept -> advance state; reject -> stay put (and ds already shrinks).
        u = jnp.where(converged, u_new, c.u)
        p = jnp.where(converged, p_new, c.p)
        tan = jnp.where(converged, tan_new, c.tan)
        idx = c.idx + converged.astype(c.idx.dtype)

        ds = _adapt_ds(c.ds, iters, converged, ds_min, ds_max)

        # Stop conditions: reached p_end (after an accept), stalled at ds_min on a
        # failure, or the iterate went non-finite.
        reached = jnp.where(
            direction >= 0, p >= p_end, p <= p_end
        )
        stalled = jnp.logical_and(jnp.logical_not(converged), ds <= ds_min)
        nonfinite = jnp.logical_not(jnp.all(jnp.isfinite(u)))
        stop = jnp.logical_or(reached, jnp.logical_or(stalled, nonfinite))

        return Carry(u, p, tan, ds, idx, stop, P, Q, T, C, D)

    init = Carry(
        u=u0, p=p0, tan=tan0, ds=ds_mag0,
        idx=jnp.array(0, jnp.int32), stop=jnp.array(False),
        P=P, Q=Q, T=T, C=C, D=D,
    )
    final = lax.while_loop(cond_fun, body, init)

    return ScanResult(
        params=final.Q,
        states=final.P,
        tangents=final.T,
        converged=final.C,
        ds=final.D,
        n_valid=final.idx + 1,   # +1 for the initial point in slot 0
    )


def branch_eigenvalues(f, states, params):
    """
    Vectorized (vmap) eigenvalues of df/du along a stored branch. Kept out of the
    continuation loop so the loop stays simple; this is itself one batched kernel.
    """
    def eig_at(u, p):
        return jnp.linalg.eigvals(jacfwd(f, argnums=0)(u, p))
    return jax.vmap(eig_at)(states, params)


def _natural_correct(f, u_pred, p_fixed, tol, max_iter):
    """
    Plain Newton on ``f(u, p_fixed) = 0`` with ``p_fixed`` held constant --
    no bordered system, no arclength constraint. This is natural
    continuation's corrector: because it has no extra degree of freedom to
    absorb an ill-conditioned ``df/du`` (unlike the bordered solve in
    ``_newton_correct``), it necessarily fails to converge at a fold, where
    ``df/du`` itself is singular -- by design, not a bug.
    """

    def cond_fun(carry):
        _, it, done, _ = carry
        return jnp.logical_and(jnp.logical_not(done), it < max_iter)

    def body(carry):
        u, it, _, _ = carry
        f_val = f(u, p_fixed)
        jac_u = jacfwd(f, argnums=0)(u, p_fixed)
        delta = jnp.linalg.solve(jac_u, -f_val)
        u_new = u + delta
        f_new = f(u_new, p_fixed)
        r_new = jnp.sqrt(jnp.sum(f_new ** 2))
        converged = r_new < tol
        blew_up = jnp.logical_not(jnp.isfinite(r_new))
        done = jnp.logical_or(converged, blew_up)
        return u_new, it + 1, done, r_new

    f0 = f(u_pred, p_fixed)
    r0 = jnp.sqrt(jnp.sum(f0 ** 2))
    init = (u_pred, 0, r0 < tol, r0)
    u_f, it_f, _, r_f = lax.while_loop(cond_fun, body, init)
    converged = jnp.logical_and(r_f < tol, jnp.all(jnp.isfinite(u_f)))
    return u_f, converged, it_f


@partial(jax.jit, static_argnums=(0, 8))
def natural_scan(
    f: Callable[[Array, Array], Array],
    u0: Array,
    p0: Array,
    p_end: Array,
    ds0: Array,
    ds_min: Array,
    ds_max: Array,
    tol: Array,
    max_steps: int,
    max_iter: Array,
) -> ScanResult:
    """
    Continue ``f(u, p) = 0`` in ``p`` from ``(u0, p0)`` toward ``p_end``
    using natural (fixed-parameter) continuation: predict by incrementing
    ``p``, correct ``u`` via plain Newton with ``p`` held fixed. Cannot pass
    fold points -- a rejected step there shrinks ``ds`` toward ``ds_min`` and
    the loop terminates via the same ``stalled`` condition
    ``pseudo_arclength_scan`` uses, rather than hanging.

    Same fixed-size-buffer / jit / vmap contract as ``pseudo_arclength_scan``:
    ``f`` and ``max_steps`` are static; buffers are ``(max_steps + 1, ...)``.
    Returns the same :class:`ScanResult` shape -- ``tangents`` is zero-filled
    (natural continuation has no tangent concept) so both engines share one
    reassembly path in ``api.py``.
    """
    u0 = jnp.asarray(u0)
    n = u0.shape[0]
    dtype = u0.dtype
    p0 = jnp.asarray(p0, dtype)
    p_end = jnp.asarray(p_end, dtype)
    direction = jnp.sign(p_end - p0)

    P = jnp.zeros((max_steps + 1, n), dtype).at[0].set(u0)
    Q = jnp.zeros((max_steps + 1,), dtype).at[0].set(p0)
    T = jnp.zeros((max_steps + 1, n + 1), dtype)
    C = jnp.zeros((max_steps + 1,), dtype=bool).at[0].set(True)
    ds_mag0 = jnp.asarray(ds0, dtype)
    D = jnp.zeros((max_steps + 1,), dtype).at[0].set(ds_mag0)

    class Carry(NamedTuple):
        u: Array
        p: Array
        ds: Array
        idx: Array
        stop: Array
        P: Array
        Q: Array
        T: Array
        C: Array
        D: Array

    def cond_fun(c: Carry):
        return jnp.logical_and(c.idx < max_steps, jnp.logical_not(c.stop))

    def body(c: Carry):
        p_pred = c.p + direction * c.ds
        u_new, converged, iters = _natural_correct(f, c.u, p_pred, tol, max_iter)

        write = c.idx + 1
        P = c.P.at[write].set(jnp.where(converged, u_new, c.P[write]))
        Q = c.Q.at[write].set(jnp.where(converged, p_pred, c.Q[write]))
        C = c.C.at[write].set(converged)
        D = c.D.at[write].set(jnp.where(converged, c.ds, c.D[write]))

        u = jnp.where(converged, u_new, c.u)
        p = jnp.where(converged, p_pred, c.p)
        idx = c.idx + converged.astype(c.idx.dtype)

        ds = _adapt_ds(c.ds, iters, converged, ds_min, ds_max)

        reached = jnp.where(direction >= 0, p >= p_end, p <= p_end)
        stalled = jnp.logical_and(jnp.logical_not(converged), ds <= ds_min)
        nonfinite = jnp.logical_not(jnp.all(jnp.isfinite(u)))
        stop = jnp.logical_or(reached, jnp.logical_or(stalled, nonfinite))

        return Carry(u, p, ds, idx, stop, P, Q, c.T, C, D)

    init = Carry(
        u=u0, p=p0, ds=ds_mag0,
        idx=jnp.array(0, jnp.int32), stop=jnp.array(False),
        P=P, Q=Q, T=T, C=C, D=D,
    )
    final = lax.while_loop(cond_fun, body, init)

    return ScanResult(
        params=final.Q,
        states=final.P,
        tangents=final.T,
        converged=final.C,
        ds=final.D,
        n_valid=final.idx + 1,
    )
