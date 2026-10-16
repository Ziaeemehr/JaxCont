"""
Fully JIT-compiled pseudo-arclength continuation (whole-loop).

Unlike the class-based predictor-corrector (`pseudo_arclength.py`), which runs a
Python outer loop dispatching many small JAX ops per step, this expresses the
*entire* continuation sweep over fixed-size buffers using ``lax.while_loop``
by default or a masked ``lax.scan`` for reverse-mode differentiation. Consequences:

- **One dispatched program per run** — no per-step Python/host-sync overhead.
- **`vmap`-able** — a batch of continuations (parameter sweeps, multistart,
  ensembles) compiles to one kernel; ideal for GPU.
- **Bounded and hang-proof** — the loop makes at most ``max_attempts`` attempts,
  each with at most ``max_iter`` Newton iterations, with an explicit finiteness
  guard, so a degenerate/saturating branch terminates cleanly instead of
  stalling.

Everything operates on a pure ``f(u, p, args) -> residual``. The continuation
parameter is the explicit scalar ``p``; ``args`` is a *dynamic* PyTree input
(traced, not closed over), so changing ``args`` values reuses the compiled
kernel. Forward derivatives work with either loop; reverse derivatives require
``ContinuationPar(loop="scan")``. The scan executes exactly ``max_attempts``
iterations (default ``4 * max_steps``), masking every inactive iteration,
including in the bounded Newton correctors. Derivatives follow the locally
selected acceptance/stopping decisions; those discrete decisions are not smooth.
``f`` and ``step_update`` are static, as are buffer sizes and solver strategies: a fresh lambda per call still recompiles, so
pass a module-level function (or a reused callable) to benefit from the cache.

``step_update(args, u, p) -> args`` (static, optional) refreshes auxiliary
data in ``args`` -- a phase-condition reference orbit, an eigenvector anchor
-- from each accepted point, so the next step is corrected against data that
follows the branch. It runs once on the corrected seed and once after every
accepted step, and must return a PyTree of the same structure, shapes and
dtypes (it is part of the ``lax.while_loop`` carry).

Design note (adaptivity): a rejected step shrinks ``ds`` and re-enters the loop
without advancing the write index, so rejections do not consume ``max_steps``;
they are bounded separately by ``max_attempts`` (default ``4 * max_steps``)
total corrector attempts. A run stalls only when a step already attempted at
``ds <= ds_min`` fails. The reason a run stopped is reported in
``ScanResult.status`` (see :class:`Status`).
"""

from __future__ import annotations

import enum
from functools import partial
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array, jacfwd, lax
from jax.extend import core as jax_core
from jax.interpreters import ad, batching, mlir

from jaxcont.solvers.protocols import Dense, DenseEigen, EigenSolver, LinearSolver

PyTree = Any


class Status(enum.IntEnum):
    """Why a scan stopped (stored as int32 in ``ScanResult.status``)."""

    REACHED_END = 0
    MAX_STEPS = 1
    STALLED = 2
    NONFINITE = 3
    SEED_FAILED = 4
    MAX_ATTEMPTS = 5


class ScanResult(NamedTuple):
    """Fixed-length buffers from a jitted continuation run."""

    params: Array        # (max_steps + 1,)
    states: Array        # (max_steps + 1, n)
    tangents: Array      # (max_steps + 1, n + 1)
    converged: Array     # (max_steps + 1,) bool  (step accepted)
    ds: Array            # (max_steps + 1,) step size used to reach each point
    iters: Array         # (max_steps + 1,) Newton iterations used to reach each point
    n_valid: Array       # scalar int; entries [:n_valid] are real points (0 if the seed failed)
    status: Array        # scalar int32; a :class:`Status` value
    n_attempts: Array    # scalar int32; corrector attempts, accepted or rejected


# ---------------------------------------------------------------------------
# Building blocks (pure; operate on f(u, p, args))
# ---------------------------------------------------------------------------

def _bounded_loop(cond, body, init, length=None):
    """Inactive scan iterations preserve every carry leaf unchanged."""
    if length is None:
        return lax.while_loop(cond, body, init)
    def step(carry, _):
        return lax.cond(cond(carry), body, lambda c: c, carry), None
    return lax.scan(step, init, xs=None, length=length)[0]


# The identity keeps forward derivatives intact and supplies an actionable
# reverse-mode error before JAX transposes the dynamic while loop.
_while_guard_p = jax_core.Primitive("continuation_while_guard")
_while_guard_p.def_impl(lambda x: x)
_while_guard_p.def_abstract_eval(lambda x: x)
mlir.register_lowering(_while_guard_p, lambda ctx, x: [x])
batching.primitive_batchers[_while_guard_p] = lambda args, dims: (_while_guard_p.bind(args[0]), dims[0])
ad.primitive_jvps[_while_guard_p] = lambda primals, tangents: (
    _while_guard_p.bind(primals[0]), _while_guard_p.bind(tangents[0])
)


def _while_guard_transpose(cotangent, x):
    raise ValueError("Reverse-mode continuation requires ContinuationPar(loop='scan')")


ad.primitive_transposes[_while_guard_p] = _while_guard_transpose
_while_guard = _while_guard_p.bind


def _bordered_matrix(jac_u: Array, df_dp: Array, last_row: Array) -> Array:
    """Assemble the (n+1, n+1) bordered matrix with ``last_row`` at the bottom."""
    n = jac_u.shape[0]
    top = jnp.concatenate([jac_u, df_dp.reshape(n, 1)], axis=1)      # (n, n+1)
    bottom = last_row.reshape(1, n + 1)                              # (1, n+1)
    return jnp.concatenate([top, bottom], axis=0)


def _tangent(f, u, p, args, prev_tangent, linear_solver: LinearSolver = Dense(), theta=1.0):
    """
    Keller tangent: the null vector of ``[df/du | df/dp]`` aligned with the
    previous tangent, normalized in theta * ||du||² + dp². Solving the bordered system with ``prev`` as
    the last row (rhs picking the last unit) both selects the null direction and
    orients it continuously — and stays well-posed at folds (where df/du alone
    is singular).
    """
    jac_u = jacfwd(f, argnums=0)(u, p, args)    # (n, n)
    df_dp = jacfwd(f, argnums=1)(u, p, args)    # (n,)
    M = _bordered_matrix(jac_u, df_dp, prev_tangent.at[:-1].multiply(theta))
    rhs = jnp.zeros(u.shape[0] + 1, dtype=u.dtype).at[-1].set(1.0)
    t = linear_solver(M, rhs)
    t = t / jnp.sqrt(theta * jnp.dot(t[:-1], t[:-1]) + t[-1] ** 2)
    # Keep orientation continuous in the same metric as the bordered solve.
    alignment = theta * jnp.dot(t[:-1], prev_tangent[:-1]) + t[-1] * prev_tangent[-1]
    t = jnp.where(alignment < 0.0, -t, t)
    return t


def _newton_correct(
    f, u_pred, p_pred, args, u_prev, p_prev, du0, dp0, ds, tol, max_iter,
    linear_solver: LinearSolver = Dense(), loop="while", theta=1.0,
):
    """
    Bordered Newton corrector for the pseudo-arclength system

        f(u, p, args) = 0
        g(u, p) = theta * (u - u_prev)·du0 + (p - p_prev)·dp0 - ds = 0

    Solves the full (n+1, n+1) bordered system each iteration (well-posed through
    folds), inside a bounded ``while_loop`` with a finiteness guard. Returns
    ``(u, p, converged, iters)``; ``converged`` requires a finite, small residual.
    """

    def residual(u, p):
        f_val = f(u, p, args)
        g_val = theta * jnp.dot(u - u_prev, du0) + (p - p_prev) * dp0 - ds
        return f_val, g_val

    def res_norm(f_val, g_val):
        return jnp.sqrt(jnp.sum(f_val ** 2) + g_val ** 2)

    def cond_fun(carry):
        _, _, it, done, _ = carry
        return jnp.logical_and(jnp.logical_not(done), it < max_iter)

    def body(carry):
        u, p, it, _, _ = carry
        f_val, g_val = residual(u, p)
        jac_u = jacfwd(f, argnums=0)(u, p, args)
        df_dp = jacfwd(f, argnums=1)(u, p, args)
        M = _bordered_matrix(jac_u, df_dp, jnp.concatenate([theta * du0, dp0.reshape(1)]))
        rhs = -jnp.concatenate([f_val, g_val.reshape(1)])
        delta = linear_solver(M, rhs)
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
    u_f, p_f, it_f, _, r_f = _bounded_loop(cond_fun, body, init, max_iter if loop == "scan" else None)
    converged = jnp.logical_and(
        r_f < tol,
        jnp.logical_and(jnp.all(jnp.isfinite(u_f)), jnp.isfinite(p_f)),
    )
    return u_f, p_f, converged, it_f


def _adapt_ds(ds_mag, iters, converged, ds_min, ds_max, adaptive=True):
    """Grow ds on fast convergence, shrink on slow/failed — branch-free.

    When ``adaptive`` is false, a converged step keeps the current ``ds_mag``
    unchanged rather than growing it, instead of growing/shrinking it as the
    adaptive path does. Note this means the step size can still be smaller
    than the originally requested ``ds`` if an earlier step in this run
    failed and shrank it -- there is no recovery back up to the requested
    size, since that would reintroduce adaptation. A rejected step still
    backs off by the same ``shrink_fail`` factor as the adaptive path either
    way, so a fixed-step run that cannot converge at the requested size
    still terminates via the ``stalled`` condition (a failure at an
    attempted ``ds <= ds_min``) instead of retrying the same failing step
    until the attempt budget runs out.

    The returned value is always clipped to ``[ds_min, ds_max]`` regardless
    of ``adaptive``, so a caller-requested ``ds`` outside that range is
    silently clamped on the very first step even under ``adaptive=False``
    (e.g. ``ContinuationPar(ds=0.5, adaptive=False)`` with the default
    ``ds_max=0.1`` immediately clamps to ``0.1``).
    """
    grow = ds_mag * 1.5
    shrink_slow = ds_mag * 0.8
    shrink_fail = ds_mag * 0.5
    adaptive_choice = jnp.where(
        converged,
        jnp.where(iters < 3, grow, jnp.where(iters > 6, shrink_slow, ds_mag)),
        shrink_fail,
    )
    fixed_choice = jnp.where(converged, ds_mag, shrink_fail)
    new = jnp.where(jnp.asarray(adaptive), adaptive_choice, fixed_choice)
    return jnp.clip(new, ds_min, ds_max)


def _attempt_budget(max_attempts, max_steps):
    """Total corrector attempts allowed; ``None`` means ``4 * max_steps``."""
    if max_attempts is None:
        return jnp.asarray(4 * max_steps, jnp.int32)
    return jnp.asarray(max_attempts, jnp.int32)


def _apply_step_update(step_update, args, u, p, accepted):
    """``step_update(args, u, p)`` where ``accepted``, else ``args`` unchanged."""
    if step_update is None:
        return args
    updated = step_update(args, u, p)
    if jax.tree_util.tree_structure(updated) != jax.tree_util.tree_structure(args):
        raise ValueError(
            "step_update must return args with the same PyTree structure it was given"
        )
    return jax.tree_util.tree_map(
        lambda new, old: jnp.where(accepted, new, old), updated, args,
    )


def _final_status(final, seed_converged, max_steps):
    """Status of a finished loop: seed failure, the recorded stop reason, else the exhausted budget."""
    status = jnp.where(
        final.stop,
        final.reason,
        jnp.where(final.idx >= max_steps, int(Status.MAX_STEPS), int(Status.MAX_ATTEMPTS)),
    )
    return jnp.where(seed_converged, status, int(Status.SEED_FAILED)).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Whole-loop continuation
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(0, 1, 10, 12, 15))
def pseudo_arclength_scan(
    f: Callable[[Array, Array, PyTree], Array],
    step_update: Optional[Callable[[PyTree, Array, Array], PyTree]],
    u0: Array,
    p0: Array,
    p_end: Array,
    args: PyTree,
    ds0: Array,
    ds_min: Array,
    ds_max: Array,
    tol: Array,
    max_steps: int,
    max_iter: Array,
    linear_solver: LinearSolver = Dense(),
    adaptive: Array = jnp.array(True),
    max_attempts: Optional[Array] = None,
    loop: str = "while",
    theta: Optional[Array] = None,
) -> ScanResult:
    """
    Continue ``f(u, p, args) = 0`` in ``p`` from ``(u0, p0)`` toward ``p_end``.

    ``theta`` weights the state inner product; ``None`` uses ``1 / n_state``.
    Numeric theta values are dynamic inputs and share a compiled kernel.

    ``f``, ``step_update`` (``None`` or see the module docstring) and
    ``max_steps`` are static (they set the compiled program & buffer
    sizes); everything else, including the ``args`` PyTree, may be traced, so this whole function is
    ``jit``/``vmap``/``jacfwd``-friendly. Use the
    ``pseudo_arclength_lax_scan`` entry point for ``grad`` (static iteration
    and attempt bounds). ``linear_solver`` is also static (a
    strategy object, not a traced value) -- see ``solvers/protocols.py``.
    """
    u0 = jnp.asarray(u0)
    n = u0.shape[0]
    dtype = u0.dtype
    theta = jnp.asarray(1 / n if theta is None else theta, dtype)
    p0 = jnp.asarray(p0, dtype)
    p_end = jnp.asarray(p_end, dtype)
    direction = jnp.sign(p_end - p0)

    # The scan buffer's slot 0 is the branch's starting point. u0 is caller-
    # supplied and not guaranteed to satisfy f(u0, p0) = 0 (e.g. a stale
    # guess, or api.py's BifProblem.u0 built by hand rather than refined).
    # Correct it the same way natural_scan corrects every other point on
    # the branch -- via _natural_correct, plain Newton with p held fixed at
    # p0 -- instead of writing the raw guess into slot 0 and marking it
    # converged unconditionally.
    u0_corrected, seed_converged, seed_iters = _natural_correct(f, u0, p0, args, tol, max_iter, linear_solver, loop)
    # If correction fails to converge, its raw output can be +-inf (a
    # divergent Newton iterate) -- fall back to the caller's original,
    # known-finite u0 rather than publishing that into the branch. This
    # matches the fallback convention events.py's Fold.refine/Hopf.refine
    # already use elsewhere on this branch: on failure, report a known-
    # finite value, not the solver's possibly-non-finite output.
    # `seed_converged` itself is unchanged -- only the value written differs.
    u0_seed = jnp.where(seed_converged, u0_corrected, u0)
    args = _apply_step_update(step_update, args, u0_seed, p0, seed_converged)

    # Initial tangent: seed prev with the parameter axis pointing in `direction`,
    # so the branch is traversed toward p_end.
    seed = jnp.zeros(n + 1, dtype).at[-1].set(direction)
    tan0 = _tangent(f, u0_seed, p0, args, seed, linear_solver, theta)

    # Fixed-size output buffers; slot 0 is the initial point.
    P = jnp.zeros((max_steps + 1, n), dtype).at[0].set(u0_seed)
    Q = jnp.zeros((max_steps + 1,), dtype).at[0].set(p0)
    T = jnp.zeros((max_steps + 1, n + 1), dtype).at[0].set(tan0)
    C = jnp.zeros((max_steps + 1,), dtype=bool).at[0].set(seed_converged)

    ds_mag0 = jnp.clip(jnp.asarray(ds0, dtype), ds_min, ds_max)
    budget = _attempt_budget(max_attempts, max_steps)
    D = jnp.zeros((max_steps + 1,), dtype).at[0].set(ds_mag0)
    I = jnp.zeros((max_steps + 1,), jnp.int32).at[0].set(seed_iters.astype(jnp.int32))

    class Carry(NamedTuple):
        u: Array
        p: Array
        tan: Array
        ds: Array         # positive magnitude; direction lives in the tangent
        idx: Array        # int; number of accepted points so far (write pointer)
        stop: Array       # bool
        reason: Array     # int32 Status recorded when ``stop`` is set
        attempts: Array   # int32 corrector attempts so far
        args: PyTree      # the args the next step is corrected with
        P: Array
        Q: Array
        T: Array
        C: Array
        D: Array
        I: Array          # Newton iterations used to reach each accepted point

    def cond_fun(c: Carry):
        return jnp.logical_and(
            jnp.logical_and(c.idx < max_steps, c.attempts < budget),
            jnp.logical_and(jnp.logical_not(c.stop), seed_converged),
        )

    def body(c: Carry):
        du0 = c.tan[:-1]
        dp0 = c.tan[-1]

        # Predict along the tangent, then correct.
        u_pred = c.u + c.ds * du0
        p_pred = c.p + c.ds * dp0
        u_new, p_new, converged, iters = _newton_correct(
            f, u_pred, p_pred, c.args, c.u, c.p, du0, dp0, c.ds, tol, max_iter, linear_solver, loop, theta
        )

        # New tangent only meaningful if we accept; compute anyway (branch-free).
        tan_new = _tangent(f, u_new, p_new, c.args, c.tan, linear_solver, theta)
        args = _apply_step_update(step_update, c.args, u_new, p_new, converged)

        write = c.idx + 1  # slot for the next accepted point
        P = c.P.at[write].set(jnp.where(converged, u_new, c.P[write]))
        Q = c.Q.at[write].set(jnp.where(converged, p_new, c.Q[write]))
        T = c.T.at[write].set(jnp.where(converged, tan_new, c.T[write]))
        C = c.C.at[write].set(converged)
        D = c.D.at[write].set(jnp.where(converged, c.ds, c.D[write]))
        I = c.I.at[write].set(jnp.where(converged, iters, c.I[write]).astype(jnp.int32))

        # Accept -> advance state; reject -> stay put (and ds already shrinks).
        u = jnp.where(converged, u_new, c.u)
        p = jnp.where(converged, p_new, c.p)
        tan = jnp.where(converged, tan_new, c.tan)
        idx = c.idx + converged.astype(c.idx.dtype)

        ds = _adapt_ds(c.ds, iters, converged, ds_min, ds_max, adaptive)

        # Stop conditions: reached p_end (after an accept), stalled at ds_min on a
        # failure, or the iterate went non-finite.
        reached = jnp.where(
            direction >= 0, p >= p_end, p <= p_end
        )
        stalled = jnp.logical_and(jnp.logical_not(converged), c.ds <= ds_min)
        nonfinite = ~(jnp.all(jnp.isfinite(u_new)) & jnp.isfinite(p_new)
                      & jnp.all(jnp.isfinite(f(u_new, p_new, c.args))))
        stop = jnp.logical_or(reached, jnp.logical_or(stalled, nonfinite))
        reason = jnp.where(
            reached, int(Status.REACHED_END),
            jnp.where(nonfinite, int(Status.NONFINITE), int(Status.STALLED)),
        ).astype(jnp.int32)

        return Carry(u, p, tan, ds, idx, stop, reason, c.attempts + 1, args, P, Q, T, C, D, I)

    init = Carry(
        u=u0_seed, p=p0, tan=tan0, ds=ds_mag0,
        idx=jnp.array(0, jnp.int32), stop=jnp.array(False),
        reason=jnp.array(int(Status.REACHED_END), jnp.int32),
        attempts=jnp.array(0, jnp.int32), args=args,
        P=P, Q=Q, T=T, C=C, D=D, I=I,
    )
    final = _bounded_loop(cond_fun, body, init,
                          (4 * max_steps if max_attempts is None else max_attempts)
                          if loop == "scan" else None)

    return ScanResult(
        params=_while_guard(final.Q) if loop == "while" else final.Q,
        states=_while_guard(final.P) if loop == "while" else final.P,
        tangents=_while_guard(final.T) if loop == "while" else final.T,
        converged=final.C,
        ds=_while_guard(final.D) if loop == "while" else final.D,
        iters=final.I,
        n_valid=jnp.where(seed_converged, final.idx + 1, 0),   # +1 for the initial point in slot 0
        status=_final_status(final, seed_converged, max_steps),
        n_attempts=final.attempts,
    )


def branch_eigenvalues(f, states, params, args, eigen_solver: EigenSolver = DenseEigen()):
    """
    Vectorized (vmap) eigenvalues of df/du along a stored branch. Kept out of the
    continuation loop so the loop stays simple; this is itself one batched kernel.
    """
    def eig_at(u, p):
        return eigen_solver(jacfwd(f, argnums=0)(u, p, args))
    return jax.vmap(eig_at)(states, params)


def _natural_correct(f, u_pred, p_fixed, args, tol, max_iter, linear_solver: LinearSolver = Dense(), loop="while"):
    """
    Plain Newton on ``f(u, p_fixed, args) = 0`` with ``p_fixed`` held constant --
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
        f_val = f(u, p_fixed, args)
        jac_u = jacfwd(f, argnums=0)(u, p_fixed, args)
        delta = linear_solver(jac_u, -f_val)
        u_new = u + delta
        f_new = f(u_new, p_fixed, args)
        r_new = jnp.sqrt(jnp.sum(f_new ** 2))
        converged = r_new < tol
        blew_up = jnp.logical_not(jnp.isfinite(r_new))
        done = jnp.logical_or(converged, blew_up)
        return u_new, it + 1, done, r_new

    f0 = f(u_pred, p_fixed, args)
    r0 = jnp.sqrt(jnp.sum(f0 ** 2))
    init = (u_pred, 0, r0 < tol, r0)
    u_f, it_f, _, r_f = _bounded_loop(cond_fun, body, init, max_iter if loop == "scan" else None)
    converged = jnp.logical_and(r_f < tol, jnp.all(jnp.isfinite(u_f)))
    return u_f, converged, it_f


@partial(jax.jit, static_argnums=(0, 1, 10, 12, 15))
def natural_scan(
    f: Callable[[Array, Array, PyTree], Array],
    step_update: Optional[Callable[[PyTree, Array, Array], PyTree]],
    u0: Array,
    p0: Array,
    p_end: Array,
    args: PyTree,
    ds0: Array,
    ds_min: Array,
    ds_max: Array,
    tol: Array,
    max_steps: int,
    max_iter: Array,
    linear_solver: LinearSolver = Dense(),
    adaptive: Array = jnp.array(True),
    max_attempts: Optional[Array] = None,
    loop: str = "while",
) -> ScanResult:
    """
    Continue ``f(u, p, args) = 0`` in ``p`` from ``(u0, p0)`` toward ``p_end``
    using natural (fixed-parameter) continuation: predict by incrementing
    ``p``, correct ``u`` via plain Newton with ``p`` held fixed. Cannot pass
    fold points -- a rejected step there shrinks ``ds`` toward ``ds_min`` and
    the loop terminates via the same ``stalled`` condition
    ``pseudo_arclength_scan`` uses, rather than hanging.

    Same fixed-size-buffer / jit / vmap contract as ``pseudo_arclength_scan``:
    ``f``, ``step_update`` and ``max_steps`` are static, ``args`` is dynamic;
    buffers are ``(max_steps + 1, ...)``.
    Returns the same :class:`ScanResult` shape -- ``tangents`` is zero-filled
    (natural continuation has no tangent concept) so both engines share one
    reassembly path in ``api.py``. ``linear_solver`` is static -- same
    contract as ``pseudo_arclength_scan``.
    """
    u0 = jnp.asarray(u0)
    n = u0.shape[0]
    dtype = u0.dtype
    p0 = jnp.asarray(p0, dtype)
    p_end = jnp.asarray(p_end, dtype)
    direction = jnp.sign(p_end - p0)

    u0_corrected, seed_converged, seed_iters = _natural_correct(f, u0, p0, args, tol, max_iter, linear_solver, loop)
    # See pseudo_arclength_scan's identical guard: fall back to the caller's
    # original (finite) u0 rather than publishing a possibly-inf failed
    # correction into the branch. `seed_converged` itself is unchanged.
    u0_seed = jnp.where(seed_converged, u0_corrected, u0)
    args = _apply_step_update(step_update, args, u0_seed, p0, seed_converged)

    P = jnp.zeros((max_steps + 1, n), dtype).at[0].set(u0_seed)
    Q = jnp.zeros((max_steps + 1,), dtype).at[0].set(p0)
    T = jnp.zeros((max_steps + 1, n + 1), dtype)
    C = jnp.zeros((max_steps + 1,), dtype=bool).at[0].set(seed_converged)
    ds_mag0 = jnp.clip(jnp.asarray(ds0, dtype), ds_min, ds_max)
    budget = _attempt_budget(max_attempts, max_steps)
    D = jnp.zeros((max_steps + 1,), dtype).at[0].set(ds_mag0)
    I = jnp.zeros((max_steps + 1,), jnp.int32).at[0].set(seed_iters.astype(jnp.int32))

    class Carry(NamedTuple):
        u: Array
        p: Array
        ds: Array
        idx: Array
        stop: Array
        reason: Array
        attempts: Array
        args: PyTree
        P: Array
        Q: Array
        T: Array
        C: Array
        D: Array
        I: Array

    def cond_fun(c: Carry):
        return jnp.logical_and(
            jnp.logical_and(c.idx < max_steps, c.attempts < budget),
            jnp.logical_and(jnp.logical_not(c.stop), seed_converged),
        )

    def body(c: Carry):
        p_pred = c.p + direction * c.ds
        u_new, converged, iters = _natural_correct(f, c.u, p_pred, c.args, tol, max_iter, linear_solver, loop)
        args = _apply_step_update(step_update, c.args, u_new, p_pred, converged)

        write = c.idx + 1
        P = c.P.at[write].set(jnp.where(converged, u_new, c.P[write]))
        Q = c.Q.at[write].set(jnp.where(converged, p_pred, c.Q[write]))
        C = c.C.at[write].set(converged)
        D = c.D.at[write].set(jnp.where(converged, c.ds, c.D[write]))
        I = c.I.at[write].set(jnp.where(converged, iters, c.I[write]).astype(jnp.int32))

        u = jnp.where(converged, u_new, c.u)
        p = jnp.where(converged, p_pred, c.p)
        idx = c.idx + converged.astype(c.idx.dtype)

        ds = _adapt_ds(c.ds, iters, converged, ds_min, ds_max, adaptive)

        reached = jnp.where(direction >= 0, p >= p_end, p <= p_end)
        stalled = jnp.logical_and(jnp.logical_not(converged), c.ds <= ds_min)
        nonfinite = ~(jnp.all(jnp.isfinite(u_new))
                      & jnp.all(jnp.isfinite(f(u_new, p_pred, c.args))))
        stop = jnp.logical_or(reached, jnp.logical_or(stalled, nonfinite))
        reason = jnp.where(
            reached, int(Status.REACHED_END),
            jnp.where(nonfinite, int(Status.NONFINITE), int(Status.STALLED)),
        ).astype(jnp.int32)

        return Carry(u, p, ds, idx, stop, reason, c.attempts + 1, args, P, Q, c.T, C, D, I)

    init = Carry(
        u=u0_seed, p=p0, ds=ds_mag0,
        idx=jnp.array(0, jnp.int32), stop=jnp.array(False),
        reason=jnp.array(int(Status.REACHED_END), jnp.int32),
        attempts=jnp.array(0, jnp.int32), args=args,
        P=P, Q=Q, T=T, C=C, D=D, I=I,
    )
    final = _bounded_loop(cond_fun, body, init,
                          (4 * max_steps if max_attempts is None else max_attempts)
                          if loop == "scan" else None)

    return ScanResult(
        params=_while_guard(final.Q) if loop == "while" else final.Q,
        states=_while_guard(final.P) if loop == "while" else final.P,
        tangents=_while_guard(final.T) if loop == "while" else final.T,
        converged=final.C,
        ds=_while_guard(final.D) if loop == "while" else final.D,
        iters=final.I,
        n_valid=jnp.where(seed_converged, final.idx + 1, 0),
        status=_final_status(final, seed_converged, max_steps),
        n_attempts=final.attempts,
    )


# Scan lengths are compile-time constants; the while entry points retain dynamic bounds.
pseudo_arclength_lax_scan = jax.jit(
    partial(pseudo_arclength_scan.__wrapped__, loop="scan"),
    static_argnums=(0, 1, 10, 11, 12, 14),
)
natural_lax_scan = jax.jit(
    partial(natural_scan.__wrapped__, loop="scan"),
    static_argnums=(0, 1, 10, 11, 12, 14),
)
