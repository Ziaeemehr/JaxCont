"""
Two-parameter continuation of codim-1 curves (fold curve, Hopf curve).

Minimal fold curves continue ``X=(u,p_fixed)`` using a bordered scalar
constraint. The explicit extended chart retains ``X=(u,p_fixed,v)``.
Hopf curves use their extended eigenvector chart. Both feed the ordinary
continuation engines, with accepted-step hooks maintaining chart data.
"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from jaxcont.api import BifProblem
from jaxcont.bifurcations.curve_layout import CurveLayout
from jaxcont.bifurcations.fold_solve import (
    _extended_residual as _fold_extended_residual,
)
from jaxcont.bifurcations.fold_solve import _pack as _fold_pack
from jaxcont.bifurcations.fold_solve import fold_point
from jaxcont.bifurcations.hopf_normal_form import (
    _anchored_residual as _hopf_anchored_residual,
)
from jaxcont.bifurcations.hopf_normal_form import _pack as _hopf_pack
from jaxcont.bifurcations.hopf_normal_form import hopf_point

PyTree = Any


def _assemble_p(p_fixed: Array, q: Array, free: int) -> Array:
    """Rebuild the shape-(2,) parameter vector from its solved and
    continued components. ``free`` is a Python int, so this is static."""
    parts = [None, None]
    parts[free] = jnp.reshape(q, ())
    parts[1 - free] = jnp.reshape(p_fixed, ())
    return jnp.stack(parts)


def _validate(u_guess: Array, p_guess: Array, free: int) -> None:
    if p_guess.shape != (2,):
        raise ValueError(
            f"p_guess must have shape (2,) for two-parameter continuation, "
            f"got {p_guess.shape}. Codim-2 work needs two free parameters "
            f"(see bifurcations/codim2.py)."
        )
    if free not in (0, 1):
        raise ValueError(f"free must be 0 or 1, got {free!r}")
    if u_guess.ndim != 1:
        raise ValueError(f"u_guess must be 1-D, got shape {u_guess.shape}")


def unpack_fold_curve(X: Array, n: int) -> tuple[Array, Array, Array]:
    """Split an extended-only fold-curve state into ``(u, p_fixed, v)``."""
    return X[:n], X[n], X[n + 1:]


def unpack_minimal_fold_curve(X: Array, n: int) -> tuple[Array, Array]:
    """Split a minimal fold state into ``(u, p_fixed)``."""
    return X[:n], X[n]


class FoldCurveArgs(eqx.Module):
    """Frozen corrector borders and original-system data."""

    b: Array
    c: Array
    user_args: PyTree
    formulation: str = eqx.field(static=True)
    n: int = eqx.field(static=True)
    raw_f: Callable = eqx.field(static=True)
    free: int = eqx.field(static=True)


def _fold_border_matrix(A, b, c):
    return jnp.block([[A, b[:, None]], [c[None, :], jnp.zeros((1, 1), dtype=A.dtype)]])


def fold_curve_step_update(args: FoldCurveArgs, X: Array, q: Array) -> FoldCurveArgs:
    """Refresh normalized right/adjoint borders at an accepted point."""
    if args.formulation == "extended":
        return args
    u, fixed = unpack_minimal_fold_curve(X, args.n)
    p = _assemble_p(fixed, q, args.free)
    A = jax.jacfwd(args.raw_f)(u, p, args.user_args)
    M = _fold_border_matrix(A, args.b, args.c)
    rhs = jnp.zeros(args.n + 1, dtype=X.dtype).at[-1].set(1)
    right = jnp.linalg.solve(M, rhs)[:-1]
    left = jnp.linalg.solve(M.T, rhs)[:-1]
    def aligned(v, old):
        v = v / jnp.linalg.norm(v)
        return lax.stop_gradient(jnp.where(jnp.dot(v, old) < 0, -v, v))
    return eqx.tree_at(lambda a: (a.b, a.c), args,
                       (aligned(left, args.b), aligned(right, args.c)))


def fold_curve_problem(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    *,
    free: int = 1,
    args: PyTree = None,
    tol: float | None = None,
    max_iter: int = 50,
    formulation: str = "minimal",
) -> BifProblem:
    """
    Build a ``BifProblem`` whose solution branch is a curve of folds in the
    ``(p[0], p[1])`` plane.

    ``free`` indexes which component of ``p`` is continued; the other is
    solved for and lives inside the packed state. ``p_guess`` has shape
    ``(2,)``.

    The caller's guess is refined to a genuine fold via ``fold_point`` before
    the problem is returned. This factory-level refinement stays necessary
    even though the scan engines (``pseudo_arclength_scan``/``natural_scan``)
    now also Newton-correct their own seed as a general hardening fix: their
    corrector holds the continuation parameter fixed and solves a plain
    (non-extended) system, which is exactly what cannot work here -- the
    seed for a fold *curve* is itself a fold, a point where ``df/du`` is
    singular by definition, so only the extended-system solve ``fold_point``
    performs (with its extra null-vector unknown absorbing that
    singularity) can actually converge there. Refining here also lets this
    factory use a tolerance matched to the extended-system residual, ahead
    of the scan engine's generic corrector.

    ``tol``/``max_iter`` govern initial refinement only; continuation uses
    ``ContinuationPar.newton_tol``. Minimal states pack ``(u,p_fixed)``;
    select ``formulation="extended"`` to retain the null-vector unknowns.

    Pass ``p_span=(p_guess[free], ...)`` to ``continuation()``: its
    ``p_span[0]`` is the literal starting parameter value, and a mismatch is
    rejected there.

    Note: the ``ValueError`` this factory raises below when the seed's
    refinement fails to converge is an eager-mode-only check -- under a
    traced call (``jax.jit``/``jax.vmap`` wrapping this factory), it
    silently becomes a no-op instead of raising, since ``seed_converged``
    is then a tracer and ``bool()`` on it would need to raise
    ``TracerBoolConversionError``, not resolve to a real Python value. Same
    limitation, and the same reason, as ``api.py``'s own ``p_span[0]``
    equality check in ``continuation()``.

    Measured residual floor at refined seed: 2.4e-7 (float32).
    """
    if formulation not in ("minimal", "extended"):
        raise ValueError("formulation must be minimal or extended")
    u_guess = jnp.asarray(u_guess)
    p_guess = jnp.asarray(p_guess)
    _validate(u_guess, p_guess, free)

    n = u_guess.shape[0]
    q0 = p_guess[free]
    fixed0 = p_guess[1 - free]

    def reduced(u, p_fixed, a, q):
        return f(u, _assemble_p(p_fixed, q, free), a)

    # The root depends on (args, q0); initial guesses only select its basin.
    u_star, p_star, v_star, seed_converged = fold_point(
        lambda u, p_fixed, theta: reduced(u, p_fixed, theta[0], theta[1]),
        lax.stop_gradient(u_guess), lax.stop_gradient(fixed0), (args, q0),
        tol=tol, max_iter=max_iter,
    )
    try:
        # Check the seed's convergence only in eager mode: a traced call
        # (jax.jit/jax.vmap wrapping fold_curve_problem, as
        # test_fold_curve_continuation_under_jit exercises) has
        # seed_converged as a tracer, and bool() on it would raise
        # TracerBoolConversionError -- same deferral pattern api.py's
        # continuation() already uses for its p_span[0]==p0 guard.
        if not bool(seed_converged):
            raise ValueError(
                f"fold_curve_problem: the initial fold-point refinement did not "
                f"converge near u_guess={u_guess}, p_guess={p_guess}. Pass a "
                f"guess closer to an actual fold, or increase tol/max_iter."
            )
    except jax.errors.ConcretizationTypeError:
        pass

    def F(X, q, a):
        if formulation == "extended":
            return _fold_extended_residual(
                X, lambda u, p_fixed, aa: reduced(u, p_fixed, aa, q), a.user_args, n
            )
        u, fixed = unpack_minimal_fold_curve(X, n)
        p = _assemble_p(fixed, q, free)
        A = jax.jacfwd(f)(u, p, a.user_args)
        M = _fold_border_matrix(A, lax.stop_gradient(a.b), lax.stop_gradient(a.c))
        rhs = jnp.zeros(n + 1, dtype=X.dtype).at[-1].set(1)
        g = jnp.linalg.solve(M, rhs)[-1]
        return jnp.concatenate([f(u, p, a.user_args), g[None]])

    A = jax.jacfwd(f)(u_star, _assemble_p(p_star, q0, free), args)
    left, _, _ = jnp.linalg.svd(A, full_matrices=False)
    b = lax.stop_gradient(left[:, -1])
    c = lax.stop_gradient(v_star / jnp.linalg.norm(v_star))
    X0 = (_fold_pack(u_star, p_star, v_star) if formulation == "extended"
          else jnp.concatenate([u_star, p_star[None]]))
    return BifProblem(
        f=F, u0=X0, p0=jnp.asarray(q0, X0.dtype),
        args=FoldCurveArgs(b, c, args, formulation, n, f, free),
        kind="fold_curve", curve_layout=CurveLayout("fold", formulation, n, free),
        param_name=f"p[{free}]", step_update=fold_curve_step_update,
    )


class HopfCurveArgs(eqx.Module):
    """The ``args`` of a Hopf-curve :class:`~jaxcont.api.BifProblem`.

    ``(q1_anchor, q2_anchor)`` is the eigenvector the phase condition
    ``q1_anchor . q2 - q2_anchor . q1 = 0`` is anchored to;
    :func:`hopf_curve_step_update` moves it to each accepted point.
    ``user_args`` is the pytree handed to the user's right-hand side.
    """

    q1_anchor: Array
    q2_anchor: Array
    user_args: PyTree


def hopf_curve_step_update(args: HopfCurveArgs, X: Array, q: Array) -> HopfCurveArgs:
    """Anchor the Hopf phase condition on the accepted point's ``(q1, q2)``."""
    n = args.q1_anchor.shape[0]
    _u, _p_fixed, q1, q2, _omega = unpack_hopf_curve(X, n)
    return HopfCurveArgs(
        q1.astype(args.q1_anchor.dtype), q2.astype(args.q2_anchor.dtype), args.user_args,
    )


def unpack_hopf_curve(
    X: Array, n: int
) -> tuple[Array, Array, Array, Array, Array]:
    """Split an extended-only Hopf-curve state into ``(u, p_fixed, q1, q2, omega)``."""
    return (
        X[:n], X[n], X[n + 1:2 * n + 1], X[2 * n + 1:3 * n + 1], X[3 * n + 1]
    )


def hopf_curve_problem(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    *,
    free: int = 1,
    args: PyTree = None,
    tol: float | None = None,
    max_iter: int = 50,
) -> BifProblem:
    """
    Build a ``BifProblem`` whose solution branch is a curve of Hopf points
    in the ``(p[0], p[1])`` plane. See :func:`fold_curve_problem` for the
    shared ``free``/``p_guess``/``tol`` conventions, why this factory's own
    ``hopf_point`` seed refinement stays necessary even though the scan
    engines now also Newton-correct their seed, and why the non-converged-
    seed ``ValueError`` below is an eager-mode-only check (a no-op under
    ``jax.jit``/``jax.vmap`` tracing).

    The returned problem's ``args`` is a :class:`HopfCurveArgs` holding the
    caller's ``args`` and the eigenvector the phase condition is anchored
    to. Its ``step_update`` (:func:`hopf_curve_step_update`) re-anchors that
    condition on every accepted point, so the curve follows the critical
    eigenvector however far it rotates.

    Measured residual floor at refined seed: 5.96e-8 (float32).
    GPU: No precision fix (TF32) needed; CPU and GPU numerics match at newton_tol=1e-5.
    """
    u_guess = jnp.asarray(u_guess)
    p_guess = jnp.asarray(p_guess)
    _validate(u_guess, p_guess, free)

    n = u_guess.shape[0]
    q0 = p_guess[free]
    fixed0 = p_guess[1 - free]

    def reduced(u, p_fixed, a, q):
        return f(u, _assemble_p(p_fixed, q, free), a)

    # The root depends on (args, q0); the moving phase anchor is stored below.
    u_star, p_star, q1_star, q2_star, omega_star, seed_converged = hopf_point(
        lambda u, p_fixed, theta: reduced(u, p_fixed, theta[0], theta[1]),
        lax.stop_gradient(u_guess), lax.stop_gradient(fixed0), (args, q0),
        tol=tol, max_iter=max_iter,
    )
    try:
        # See fold_curve_problem's identical guard above for why this is
        # deferred under a traced call.
        if not bool(seed_converged):
            raise ValueError(
                f"hopf_curve_problem: the initial Hopf-point refinement did not "
                f"converge near u_guess={u_guess}, p_guess={p_guess}. Pass a "
                f"guess closer to an actual Hopf point, or increase tol/max_iter."
            )
    except jax.errors.ConcretizationTypeError:
        pass

    def F(X, q, a):
        return _hopf_anchored_residual(
            X, lambda u, p_fixed, aa: reduced(u, p_fixed, aa, q), a.user_args, n,
            lax.stop_gradient(a.q1_anchor), lax.stop_gradient(a.q2_anchor),
        )

    X0 = _hopf_pack(u_star, p_star, q1_star, q2_star, omega_star)
    return BifProblem(
        f=F,
        u0=X0,
        p0=jnp.asarray(q0, X0.dtype),
        args=HopfCurveArgs(
            lax.stop_gradient(q1_star), lax.stop_gradient(q2_star), args,
        ),
        kind="hopf_curve",
        curve_layout=CurveLayout("hopf", "extended", n, free),
        param_name=f"p[{free}]",
        step_update=hopf_curve_step_update,
    )
