"""
Branch switching: from a detected equilibrium branch point (BP), compute
the crossing branch's tangent and return a seeded BifProblem on it.

See docs/superpowers/specs/2026-08-26-branch-point-switching-design.md.

At a confirmed BP (jaxcont.bifurcations.events.BP), the augmented Jacobian
[f_u | f_p] (n x (n+1)) has a 2-D null space: one direction is the incoming
branch's own known tangent (carried in EventHit.info["tangent"] -- EventHit
itself has no tangent field), the other is the crossing branch's direction.
The crossing direction solves a bordered null-space system orthogonal to
the incoming tangent, pinned by the best-conditioned coordinate border.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
from jax import Array, jacfwd, lax

from jaxcont.api import BifProblem, bif_problem
from jaxcont.bifurcations.events import EventHit
from jaxcont.solvers.implicit import differentiable_root

PyTree = Any


def _switching_tangent(f, u_star, p_star, incoming_tangent, args, n):
    """t_new, shape (n+1,) in (u, p)-order: the crossing branch's unit
    tangent at the confirmed BP. Recomputed from scratch every call (never
    precomputed by the caller) so it stays theta(=args)-dependent when
    called from inside branch_switch's differentiable_root closures -- see
    the module docstring and solvers/implicit.py's "seed must be callable
    when theta-dependent" rule.
    """
    f_u = jacfwd(lambda u: f(u, p_star, args))(u_star)
    f_p = jacfwd(lambda p: f(u_star, p, args))(p_star)
    augmented = jnp.concatenate([f_u, f_p.reshape(n, 1)], axis=1)
    incoming_dir = incoming_tangent / jnp.linalg.norm(incoming_tangent)
    null_system = jnp.vstack([augmented, incoming_dir])
    borders = jnp.eye(n + 1, dtype=augmented.dtype)
    candidates = jnp.concatenate([
        jnp.broadcast_to(null_system, (n + 1,) + null_system.shape),
        borders[:, None, :],
    ], axis=1)
    # The coordinate choice is discrete; only the selected solve is differentiated.
    singular_values = jnp.linalg.svd(lax.stop_gradient(candidates), compute_uv=False)
    k = jnp.argmax(singular_values[:, -1] / singular_values[:, 0])
    bordered = jnp.vstack([null_system, borders[k]])
    rhs = jnp.zeros(n + 2, dtype=augmented.dtype).at[-1].set(1)
    # At a BP these equations are consistent and the coordinate border gives
    # full column rank; reduced QR avoids squaring the condition number.
    q, r = jnp.linalg.qr(bordered, mode="reduced")
    t_new_raw = jnp.linalg.solve(r, q.T @ rhs)
    return t_new_raw / jnp.linalg.norm(t_new_raw)


def branch_switch(
    f: Callable[[Array, Array, PyTree], Array],
    event: EventHit,
    *,
    args: PyTree = None,
    ds: float = 0.01,
    tol: float | None = None,
    max_iter: int = 50,
) -> BifProblem:
    """
    From a detected BP (``jaxcont.bifurcations.events.BP``), compute the
    crossing branch's tangent, take one predictor step, and Newton-correct
    it back onto ``f(u, p, args) = 0`` with a pseudo-arclength corrector
    (jointly solving for ``(u, p)``, NOT holding ``p`` fixed -- see below),
    returning a ready-to-continue :class:`BifProblem`.

    ``event`` must be a ``kind="branch_point"`` hit (from
    ``continuation(..., events=[BP()])``); anything else raises
    ``ValueError`` -- eagerly, except under a traced call
    (``jax.jit``/``jax.vmap``), where the same deferral
    ``fold_curve_problem``/``hopf_curve_problem`` already use applies (the
    check becomes a no-op rather than raising, since ``bool()`` on a
    tracer would itself raise ``TracerBoolConversionError``).

    Why a pseudo-arclength corrector and not a simpler fixed-``p``
    corrector: for a pitchfork, the crossing branch's tangent is exactly
    ``(1, 0, ..., 0)`` in ``(u, p)``-order -- it is tangent to the ``p``
    axis at the BP itself (``dp/ds = 0`` there). A fixed-``p`` corrector
    would try to solve ``f(u, p*, args) = 0`` at the BP's own parameter
    value, which for a supercritical pitchfork has no root other than the
    trivial one -- it silently snaps back onto the ORIGINAL branch instead
    of finding the new one. The pseudo-arclength corrector below (jointly
    solving ``f(u,p,args)=0`` plus an arc-length pinning equation) is the
    same style the engine's own predictor-corrector already uses
    (``core/scan_continuation.py``), just applied once here instead of
    iterated.

    The switching direction's sign is arbitrary (a coordinate-border convention,
    not a physical choice) -- callers that care which of the two crossing
    directions they land on should inspect ``new_problem.u0``/``.p0`` and
    negate ``ds`` and retry if it's the wrong one.
    """
    if event.kind != "branch_point":
        raise ValueError(
            f"branch_switch: expected a BP hit (kind='branch_point'), "
            f"got kind={event.kind!r}. Call continuation(..., events=[BP()]) "
            f"and pass one of the resulting kind='branch_point' events."
        )

    u_star = jnp.asarray(event.u)
    p_star = jnp.asarray(event.p)
    n = u_star.shape[0]
    incoming_tangent = jnp.asarray(event.info["tangent"])
    x_star = jnp.concatenate([u_star, jnp.reshape(p_star, (1,))])

    def G(x, theta):
        t_new = _switching_tangent(f, u_star, p_star, incoming_tangent, theta, n)
        x_pred = x_star + ds * t_new
        u, p = x[:n], x[n]
        residual = f(u, p, theta)
        arclength = jnp.dot(t_new, x - x_pred)
        return jnp.concatenate([residual, jnp.reshape(arclength, (1,))])

    def x0(theta):
        t_new = _switching_tangent(f, u_star, p_star, incoming_tangent, theta, n)
        return x_star + ds * t_new

    x_corrected = differentiable_root(G, x0, args, tol=tol, max_iter=max_iter)
    u_corrected, p_corrected = x_corrected[:n], x_corrected[n]

    return bif_problem(f, u0=u_corrected, p0=p_corrected, args=args)
