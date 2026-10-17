"""
Two-parameter continuation of codim-1 curves (fold curve, Hopf curve).

Minimal fold curves continue ``X=(u,p_fixed)`` using a bordered scalar
constraint. The explicit extended chart retains ``X=(u,p_fixed,v)``.
Minimal Hopf curves continue ``X=(u,p_fixed,omega)`` at positive frequency.
Both feed the ordinary continuation engines, with accepted-step hooks
maintaining chart data.
"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from jaxcont.api import BifProblem
from jaxcont.bifurcations.curve_layout import CurveLayout, minimal_hopf_floor
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
from jaxcont.utils.precision import default_tol

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
    ``b,c`` are the frozen complex left/right borders for the minimal chart.
    ``v_anchor`` fixes the real BT gauge at the seed and is never refreshed.
    ``omega_floor`` is the minimal chart's lowest frequency,
    ``sqrt(default_tol) * ||A(u_seed, p_seed)||_2``, fixed at the seed.
    ``user_args`` is the pytree handed to the user's right-hand side.
    """

    q1_anchor: Array
    q2_anchor: Array
    user_args: PyTree
    b: Array
    c: Array
    formulation: str = eqx.field(static=True)
    n: int = eqx.field(static=True)
    raw_f: Callable = eqx.field(static=True)
    free: int = eqx.field(static=True)
    v_anchor: Array | None = None
    omega_floor: Array | None = None


def _hopf_border_matrix(A, omega, b, c):
    C = A - 1j * omega * jnp.eye(A.shape[0], dtype=A.dtype)
    return jnp.block([[C, b[:, None]], [c.conj()[None, :], jnp.zeros((1, 1), dtype=C.dtype)]])


def unpack_bt_hopf_curve(X: Array, n: int) -> tuple[Array, Array, Array, Array, Array]:
    """Split a real BT chart into ``(u,p_fixed,v,w,kappa)``."""
    return X[:n], X[n], X[n + 1:2 * n + 1], X[2 * n + 1:3 * n + 1], X[-1]


def hopf_curve_frequency(kappa: Array) -> tuple[Array, Array]:
    """Report frequency and oscillatory eligibility; never used in residuals."""
    return jnp.sqrt(jnp.maximum(kappa, 0)), kappa > 0


def _bt_hopf_residual(X, p, f, args, n):
    u, _, v, w, kappa = unpack_bt_hopf_curve(X, n)
    A = jax.jacfwd(f)(u, p, args.user_args)
    return jnp.concatenate([
        f(u, p, args.user_args), A @ v + kappa * w, A @ w - v,
        jnp.stack([jnp.dot(v, v) - 1, jnp.dot(lax.stop_gradient(args.v_anchor), w)]),
    ])


def unpack_minimal_hopf_curve(X: Array, n: int) -> tuple[Array, Array, Array]:
    """Split a minimal positive-frequency Hopf state into (u,p_fixed,omega)."""
    return X[:n], X[n], X[n + 1]


def hopf_curve_step_update(args: HopfCurveArgs, X: Array, q: Array) -> HopfCurveArgs:
    """Refresh ordinary charts; the BT gauge remains fixed at its seed."""
    if args.formulation == "bt":
        return args
    if args.formulation == "extended":
        _, _, q1, q2, _ = unpack_hopf_curve(X, args.n)
        return eqx.tree_at(lambda a: (a.q1_anchor, a.q2_anchor), args,
                           (lax.stop_gradient(q1), lax.stop_gradient(q2)))
    u, fixed, omega = unpack_minimal_hopf_curve(X, args.n)
    A = jax.jacfwd(args.raw_f)(u, _assemble_p(fixed, q, args.free), args.user_args)
    M = _hopf_border_matrix(A, omega, args.b, args.c)
    rhs = jnp.zeros(args.n + 1, dtype=M.dtype).at[-1].set(1)
    right = jnp.linalg.solve(M, rhs)[:-1]
    left = jnp.linalg.solve(M.conj().T, rhs)[:-1]
    def aligned(v, old):
        v = v / jnp.linalg.norm(v)
        overlap = jnp.vdot(old, v)
        phase = jnp.conj(overlap) / jnp.where(jnp.abs(overlap) > 0, jnp.abs(overlap), 1)
        return lax.stop_gradient(v * phase)
    return eqx.tree_at(lambda a: (a.b, a.c), args,
                       (aligned(left, args.b), aligned(right, args.c)))


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
    formulation: str = "minimal",
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
    condition on every accepted point in the extended chart. The default
    minimal chart packs ``(u,p_fixed,omega)`` and refreshes complex borders
    instead. It requires ``omega > sqrt(default_tol(dtype)) * ||A||_2``, with
    ``A`` the Jacobian at the refined seed (``HopfCurveArgs.omega_floor``, see
    :func:`~jaxcont.bifurcations.curve_layout.minimal_hopf_floor`): a step
    whose corrector leaves the chart is rejected, so a curve running into a
    Bogdanov-Takens point stalls just above the floor and
    ``BogdanovTakens(curve="hopf")`` records it as ``"curve_terminated"``.
    Select ``formulation="bt"`` to locate BT and continue through it, or
    ``formulation="extended"`` for the eigenvector chart.

    ``formulation="bt"`` packs ``(u,p_fixed,v,w,kappa)`` with a seed-fixed
    real Jordan-chain gauge, permitting continuation through zero frequency.

    Measured residual floor at refined seed: 5.96e-8 (float32).
    GPU: No precision fix (TF32) needed; CPU and GPU numerics match at newton_tol=1e-5.
    """
    if formulation not in ("minimal", "extended", "bt"):
        raise ValueError("formulation must be minimal, extended or bt")
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
        if formulation == "bt":
            return _bt_hopf_residual(X, _assemble_p(X[n], q, free), f, a, n)
        if formulation == "extended":
            return _hopf_anchored_residual(
                X, lambda u, p_fixed, aa: reduced(u, p_fixed, aa, q), a.user_args, n,
                lax.stop_gradient(a.q1_anchor), lax.stop_gradient(a.q2_anchor),
            )
        u, fixed, omega = unpack_minimal_hopf_curve(X, n)
        p = _assemble_p(fixed, q, free)
        # The complex chart is valid only above args.omega_floor. Below it
        # the border solve uses the floor and the last equation is replaced
        # by 2*floor - omega >= floor: finite, never a root, and Newton steps
        # it back above the floor, so leaving the chart is a corrector
        # failure (the step is rejected and ds shrinks).
        # The border equations are divided by the seed's ||A||_2, so their
        # tolerance (and the floor) follow the system's time scale.
        floor = lax.stop_gradient(a.omega_floor)
        scale = floor / minimal_hopf_floor(X.dtype)
        in_chart = omega > floor
        omega_chart = jnp.where(in_chart, omega, floor)
        A = jax.jacfwd(f)(u, p, a.user_args)
        M = _hopf_border_matrix(A, omega_chart, lax.stop_gradient(a.b), lax.stop_gradient(a.c))
        rhs = jnp.zeros(n + 1, dtype=M.dtype).at[-1].set(1)
        g = jnp.linalg.solve(M, rhs)[-1]
        last = jnp.where(in_chart, g.imag, 2 * floor - omega)
        return jnp.concatenate([f(u, p, a.user_args), jnp.stack([g.real, last]) / scale])

    A = jax.jacfwd(f)(u_star, _assemble_p(p_star, q0, free), args)
    C = A - 1j * omega_star * jnp.eye(n, dtype=A.dtype)
    left, _, _ = jnp.linalg.svd(C, full_matrices=False)
    b = lax.stop_gradient(left[:, -1])
    z = q1_star + 1j * q2_star
    c = lax.stop_gradient(z / jnp.linalg.norm(z))
    omega_floor = lax.stop_gradient(
        minimal_hopf_floor(u_star.dtype, jnp.linalg.norm(A, 2))).astype(u_star.dtype)
    if formulation in ("minimal", "bt"):
        try:
            if formulation == "minimal" and not bool(omega_star > omega_floor):
                raise ValueError(
                    f"Hopf seed frequency {float(omega_star):.3g} is at or below the minimal "
                    f"chart's floor sqrt(default_tol)*||A||_2 = {float(omega_floor):.3g}; "
                    f"use formulation='extended' or formulation='bt'")
            if formulation == "bt" and not bool(omega_star > 10 * default_tol(u_star.dtype)):
                raise ValueError("Hopf seed requires positive frequency above 10*default_tol")
        except jax.errors.ConcretizationTypeError:
            pass
    X0 = (_hopf_pack(u_star, p_star, q1_star, q2_star, omega_star)
          if formulation == "extended" else jnp.concatenate([u_star, p_star[None], omega_star[None]]))
    v_anchor = lax.stop_gradient(q1_star)
    if formulation == "bt":
        # A phase rotation diagonalizes the real/imaginary Gram matrix.
        gram_cross = lax.stop_gradient(2 * jnp.dot(q1_star, q2_star))
        gram_difference = lax.stop_gradient(
            jnp.dot(q1_star, q1_star) - jnp.dot(q2_star, q2_star))
        anisotropic = jnp.hypot(gram_cross, gram_difference) > default_tol(u_star.dtype)
        # Phase is a discrete chart choice; an isotropic Gram matrix uses zero.
        angle = .5 * jnp.arctan2(jnp.where(anisotropic, gram_cross, 0),
                                  jnp.where(anisotropic, gram_difference, 1))
        real = jnp.cos(angle) * q1_star + jnp.sin(angle) * q2_star
        imag = -jnp.sin(angle) * q1_star + jnp.cos(angle) * q2_star
        scale = jnp.linalg.norm(real)
        v, w = real / scale, imag / (scale * omega_star)
        v_anchor = lax.stop_gradient(v)
        X0 = jnp.concatenate([u_star, p_star[None], v, w, (omega_star**2)[None]])
    return BifProblem(
        f=F, u0=X0, p0=jnp.asarray(q0, X0.dtype),
        args=HopfCurveArgs(lax.stop_gradient(q1_star), lax.stop_gradient(q2_star), args,
                           b, c, formulation, n, f, free, v_anchor=v_anchor,
                           omega_floor=omega_floor),
        kind="hopf_curve", curve_layout=CurveLayout("hopf", formulation, n, free),
        param_name=f"p[{free}]", step_update=hopf_curve_step_update,
    )
