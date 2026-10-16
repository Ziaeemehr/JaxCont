"""
Period-doubling branch switching -- see
docs/superpowers/specs/2026-08-27-period-doubling-switching-design.md.

Switches from a detected PeriodDoubling event onto the period-doubled cycle
that bifurcates there. The corrector is the existing
``problems/periodic.py:periodic_orbit_problem`` used unchanged: this module
only builds the seed and verifies the result.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxcont.api import BifProblem
from jaxcont.bifurcations.events import EventHit
from jaxcont.core.collocation import (
    Collocation,
    collocation_matrices,
    monodromy_matrix,
)
from jaxcont.problems.periodic import periodic_orbit_problem

PyTree = Any


def _unpack(U: Array, mesh: Collocation):
    """Split a periodic ``U`` into ``(mesh_states, coll_states, T, n)``.

    Mirrors ``stability/floquet.py:floquet_multipliers``: ``mesh`` carries no
    state dimension, so ``n`` is derived algebraically from ``U``'s length.
    """
    ntst, ncol = mesh.ntst, mesh.ncol
    n = (U.shape[-1] - 1) // (ntst * (1 + ncol))
    mesh_states = U[: ntst * n].reshape(ntst, n)
    coll_states = U[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
    return mesh_states, coll_states, U[-1], n


def _pd_eigenvector(
    raw_f: Callable[[Array, Array, PyTree], Array],
    U: Array,
    p: Array,
    mesh: Collocation,
    *,
    args: PyTree = None,
    pd_tol: float = 1e-2,
    imag_tol: float = 1e-3,
) -> Array:
    """The real, unit-norm ``-1`` Floquet eigenvector at a period-doubling point.

    Recomputes the monodromy matrix rather than reading a cached eigenvector
    off the event: ``PeriodDoubling.refine`` does not store one, and
    ``branch_switch_pd`` already needs ``raw_f``/``mesh`` for the seed, so
    recomputation costs one small ``n x n`` eigendecomposition and keeps
    ``events.py`` untouched.

    ``pd_tol`` is deliberately loose (1e-2): the event point is a bisection
    midpoint sitting at the float32 collocation residual floor, not a
    Newton-refined point.
    """
    mesh_states, coll_states, T, _n = _unpack(U, mesh)
    D_np, E_np, _, _ = collocation_matrices(mesh.ncol)
    D, E = jnp.asarray(D_np), jnp.asarray(E_np)
    Phi = monodromy_matrix(
        raw_f, D, E, 1.0 / mesh.ntst, mesh_states, coll_states, T, p, args
    )
    values, vectors = jnp.linalg.eig(Phi)
    idx = int(jnp.argmin(jnp.abs(values + 1.0)))
    lam = values[idx]
    if float(jnp.abs(lam + 1.0)) > pd_tol or float(jnp.abs(jnp.imag(lam))) > imag_tol:
        raise ValueError(
            "not a period-doubling point: the Floquet multiplier nearest -1 is "
            f"{complex(lam):.6g} (needs |lambda + 1| < {pd_tol} and "
            f"|Im lambda| < {imag_tol}). Check that this event came from a "
            "jc.PeriodDoubling() detection on a periodic-orbit branch."
        )
    v = jnp.real(vectors[:, idx])
    norm = float(jnp.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError(
            "the -1 Floquet eigenvector is numerically zero; cannot build a "
            "switching direction."
        )
    return v / norm


def _doubling_measure(U: Array, mesh: Collocation, n: int) -> float:
    """How much the orbit differs from its own half-period shift, relative to
    its scale.

    On a uniform mesh of ``ntst`` points over the doubled period ``2T``, the
    shift ``t -> t + T`` is an index roll of ``ntst // 2`` (``ntst`` is even by
    construction, so the roll is exact). A genuine period-doubled orbit differs
    from that shift; the original orbit traversed twice is identical to it.
    This is the only signal that separates the two -- an under-sized seed
    converges onto the retraced original with a perfectly healthy residual.
    """
    mesh_states = U[: mesh.ntst * n].reshape(mesh.ntst, n)
    shifted = jnp.roll(mesh_states, mesh.ntst // 2, axis=0)
    spread = float(jnp.max(jnp.linalg.norm(mesh_states - shifted, axis=1)))
    scale = float(jnp.max(jnp.linalg.norm(mesh_states, axis=1)))
    if scale < 1e-12:
        return 0.0
    return spread / scale


def _periodic_base_samples(mesh_states: Array, T: float, t_samples: np.ndarray) -> np.ndarray:
    """Sample the base orbit at ``t mod T`` by periodic linear interpolation.

    The base orbit's mesh points sit at ``tau = i/ntst`` over ``[0, T)``. The
    wrap-around point is appended so a time just below ``T`` interpolates back
    to ``mesh_states[0]`` rather than clamping.
    """
    base = np.asarray(mesh_states)
    ntst = base.shape[0]
    t_knots = np.concatenate([np.arange(ntst) * (T / ntst), np.array([T])])
    u_knots = np.concatenate([base, base[:1]], axis=0)
    wrapped = np.mod(t_samples, T)
    return np.stack(
        [np.interp(wrapped, t_knots, u_knots[:, c]) for c in range(base.shape[1])],
        axis=1,
    )


def branch_switch_pd(
    raw_f: Callable[[Array, Array, PyTree], Array],
    event: EventHit,
    mesh: Collocation,
    *,
    args: PyTree = None,
    dp: float = 0.05,
    dp_sign: Optional[int] = None,
    eps_ladder: Sequence[float] = (0.05, 0.1, 0.2, 0.4, 0.8),
    doubling_tol: float = 1e-3,
    residual_tol: float = 1e-4,
    refine_tol: Optional[float] = None,
    n_samples: Optional[int] = None,
    pd_tol: float = 1e-2,
) -> BifProblem:
    """Seed the period-doubled cycle bifurcating at a ``PeriodDoubling`` event.

    ``raw_f`` is the ODE right-hand side (``f(u, p, args)``), not the assembled
    collocation residual; ``args`` is the user pytree it is called with (the
    original problem's ``args.user_args``) and is carried into the returned
    problem. ``mesh`` is the ``Collocation`` the original branch was
    continued on; the returned problem lives on ``Collocation(ntst=2*ntst,
    ncol=ncol)``, which preserves per-unit-time resolution across the twice-as-long
    period.

    The parameter is stepped to ``event.p + sign*dp`` before correcting: at the
    bifurcation parameter itself the doubled branch has zero amplitude, so a
    correction there collapses to the trivial branch no matter how it is seeded.
    Once stepped off, the existing ``periodic_orbit_problem`` fixed-``p`` solve
    reaches the doubled branch directly.

    Neither the correct seed amplitude nor the side the doubled branch lies on is
    knowable in advance, and both failures look identical from outside -- a clean
    convergence onto the original orbit traversed twice. So this tries
    ``(sign, eps)`` combinations, verifies each against
    :func:`_doubling_measure`, and returns the first that passes. If none does it
    raises, rather than returning a plausible but wrong branch.

    ``refine_tol`` is the Newton residual tolerance for each candidate's
    correction, passed to ``periodic_orbit_problem`` as its ``tol``. The
    default ``None`` uses that function's default, ``default_tol`` of the
    dtype. Systems with a continuous symmetry, such as an S^1-equivariant
    transverse block, leave a family of doubled orbits that the phase
    condition does not pin down; Newton then stalls above ``default_tol`` and
    such systems need a looser explicit ``refine_tol``. ``residual_tol``
    remains the acceptance test on the result.

    This factory is eager-only because its ``(sign, eps)`` search loop converts
    residuals to concrete Python values, branches on the doubling measure, and
    raises if no candidate passes; those operations cannot be traced. The
    returned ``BifProblem`` is itself fine to pass to ``jc.continuation``.

    Raises:
        ValueError: if ``event`` is not a period-doubling event, if the point is
            not actually a period-doubling point, or if no ``(sign, eps)``
            candidate produced a verified period-doubled orbit.
    """
    if event.kind != "period_doubling":
        raise ValueError(
            f"branch_switch_pd needs a period_doubling event, got {event.kind!r}. "
            "Detect one with jc.continuation(..., events=[jc.PeriodDoubling(...)])."
        )

    U_star = jnp.asarray(event.u)
    p_star = float(event.p)
    mesh_states, _coll_states, T_arr, n = _unpack(U_star, mesh)
    T = float(T_arr)

    v0 = np.asarray(
        _pd_eigenvector(raw_f, U_star, jnp.asarray(p_star), mesh, args=args, pd_tol=pd_tol)
    )
    orbit_scale = float(jnp.max(jnp.linalg.norm(mesh_states, axis=1)))

    if n_samples is None:
        n_samples = max(240, 12 * mesh.ntst)
    t_samples = np.linspace(0.0, 2 * T, n_samples, endpoint=False)
    base_samples = _periodic_base_samples(mesh_states, T, t_samples)
    # cos(pi*t/T) is anti-periodic over T -- it flips sign between the two
    # copies of the base orbit, which is what makes the guess a period-DOUBLED
    # object rather than a period-T one.
    modulation = np.cos(np.pi * t_samples / T)[:, None] * v0[None, :]

    doubled_mesh = Collocation(ntst=2 * mesh.ntst, ncol=mesh.ncol)
    signs = (1, -1) if dp_sign is None else (int(np.sign(dp_sign)),)

    attempts: list[str] = []
    for sign in signs:
        for eps_rel in eps_ladder:
            eps = eps_rel * orbit_scale
            u_seed = base_samples + eps * modulation
            label = f"sign={sign:+d} eps={eps:.4g}"
            try:
                candidate = periodic_orbit_problem(
                    raw_f,
                    jnp.asarray(u_seed),
                    jnp.asarray(t_samples),
                    2 * T,
                    p_star + sign * dp,
                    doubled_mesh,
                    args=args,
                    tol=refine_tol,
                )
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                attempts.append(f"{label}: correction raised {type(exc).__name__}")
                continue
            residual = float(
                jnp.linalg.norm(
                    candidate.f(candidate.u0, candidate.p0, candidate.args)
                )
            )
            if not np.isfinite(residual) or residual > residual_tol:
                attempts.append(f"{label}: did not converge (residual {residual:.3g})")
                continue
            measure = _doubling_measure(candidate.u0, doubled_mesh, n)
            if measure > doubling_tol:
                return candidate
            attempts.append(
                f"{label}: collapsed onto the original orbit "
                f"(doubling measure {measure:.3g} <= {doubling_tol})"
            )

    raise ValueError(
        "branch_switch_pd could not find a period-doubled orbit at "
        f"p = {p_star} +/- {dp}. Every candidate either failed to converge or "
        "collapsed back onto the original orbit traversed twice. Try a larger "
        "dp, a wider eps_ladder, or check that this really is a period-doubling "
        "point.\nAttempts:\n  " + "\n  ".join(attempts)
    )
