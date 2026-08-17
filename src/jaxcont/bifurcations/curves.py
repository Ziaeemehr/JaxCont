"""
Two-parameter continuation of codim-1 curves (fold curve, Hopf curve).

A fold of ``f(u, p) = 0`` is the root of the extended system in
``fold_solve.py``. A fold *curve* is that same system continued in a second
parameter. With ``X = (u, p_fixed, v)`` and continuation parameter
``q = p[free]``, ``F(X, q) = 0`` is an ordinary residual in a scalar
parameter -- exactly what ``core/scan_continuation.py`` already solves.

The reduction that makes this free: define
``f_reduced(u, p_fixed, args) = f(u, assemble(p_fixed, q), args)``. From
``f_reduced``'s point of view ``p_fixed`` is an ordinary scalar parameter,
so ``fold_solve._extended_residual`` applies UNCHANGED. Same trick that let
periodic-orbit collocation reuse the equilibrium engine.

GPU numerics: Both fold-curve and Hopf-curve residuals are computed via
``jacfwd`` plus matvecs, with no large einsum contractions, so TF32 precision
is not a concern. CPU and GPU numerics match at newton_tol=1e-5 (verified).

See docs/superpowers/specs/2026-08-17-two-parameter-continuation-design.md.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
from jax import Array, lax

from jaxcont.api import BifProblem
from jaxcont.bifurcations.fold_solve import (
    _extended_residual as _fold_extended_residual,
)
from jaxcont.bifurcations.fold_solve import _pack as _fold_pack
from jaxcont.bifurcations.fold_solve import fold_point
from jaxcont.bifurcations.hopf_normal_form import (
    _extended_residual as _hopf_extended_residual,
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
    """Split a packed fold-curve state into ``(u, p_fixed, v)``."""
    return X[:n], X[n], X[n + 1:]


def fold_curve_problem(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    *,
    free: int = 1,
    args: PyTree = None,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> BifProblem:
    """
    Build a ``BifProblem`` whose solution branch is a curve of folds in the
    ``(p[0], p[1])`` plane.

    ``free`` indexes which component of ``p`` is continued; the other is
    solved for and lives inside the packed state. ``p_guess`` has shape
    ``(2,)``.

    The caller's guess is refined to a genuine fold via ``fold_point``
    before the problem is returned -- the scan engines do not Newton-correct
    their starting point, so an unrefined guess would silently be marked
    ``converged=True`` (the same reason ``periodic_orbit_problem`` refines).

    ``tol``/``max_iter`` govern that initial refinement only; the
    continuation-time tolerance is ``ContinuationPar.newton_tol`` (use
    ``1e-5``, see the module notes and this feature's design spec).

    Pass ``p_span=(p_guess[free], ...)`` to ``continuation()``: its
    ``p_span[0]`` is the literal starting parameter value, and a mismatch is
    rejected there.

    Measured residual floor at refined seed: 2.4e-7 (float32).
    """
    u_guess = jnp.asarray(u_guess)
    p_guess = jnp.asarray(p_guess)
    _validate(u_guess, p_guess, free)

    n = u_guess.shape[0]
    q0 = p_guess[free]
    fixed0 = p_guess[1 - free]

    def reduced(u, p_fixed, a, q):
        return f(u, _assemble_p(p_fixed, q, free), a)

    # Refine the seed onto the curve at q = q0.
    u_star, p_star, v_star = fold_point(
        lambda u, p_fixed, a: reduced(u, p_fixed, a, q0),
        u_guess, fixed0, args, tol=tol, max_iter=max_iter,
    )

    def F(X, q, a):
        return _fold_extended_residual(
            X, lambda u, p_fixed, aa: reduced(u, p_fixed, aa, q), a, n
        )

    X0 = _fold_pack(u_star, p_star, v_star)
    return BifProblem(
        f=F,
        u0=X0,
        p0=jnp.asarray(q0, X0.dtype),
        args=args,
        kind="fold_curve",
        param_name=f"p[{free}]",
    )


def unpack_hopf_curve(
    X: Array, n: int
) -> tuple[Array, Array, Array, Array, Array]:
    """Split a packed Hopf-curve state into ``(u, p_fixed, q1, q2, omega)``."""
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
    tol: float = 1e-6,
    max_iter: int = 50,
) -> BifProblem:
    """
    Build a ``BifProblem`` whose solution branch is a curve of Hopf points
    in the ``(p[0], p[1])`` plane. See :func:`fold_curve_problem` for the
    shared ``free``/``p_guess``/``tol`` conventions.

    Known limitation: the phase condition anchors to a seed eigenvector
    recomputed from the refined starting point at each ``q``, so it tracks
    gently along the curve. If the eigenvector rotates far enough to become
    orthogonal to that seed, the phase row degenerates and the curve stalls.
    Fixed-shape buffers (which the jit/vmap story depends on) rule out
    MatCont-style adaptive re-anchoring; the remedy is restarting from a
    later point.

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

    u_star, p_star, q1_star, q2_star, omega_star = hopf_point(
        lambda u, p_fixed, a: reduced(u, p_fixed, a, q0),
        u_guess, fixed0, args, tol=tol, max_iter=max_iter,
    )

    def F(X, q, a):
        return _hopf_extended_residual(
            X, lambda u, p_fixed, aa: reduced(u, p_fixed, aa, q), a, n,
            lax.stop_gradient(u_star), lax.stop_gradient(p_star),
        )

    X0 = _hopf_pack(u_star, p_star, q1_star, q2_star, omega_star)
    return BifProblem(
        f=F,
        u0=X0,
        p0=jnp.asarray(q0, X0.dtype),
        args=args,
        kind="hopf_curve",
        param_name=f"p[{free}]",
    )
