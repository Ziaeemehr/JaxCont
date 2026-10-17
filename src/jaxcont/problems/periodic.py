"""
Periodic-orbit continuation via fixed-mesh Gauss-Legendre orthogonal
collocation -- see
docs/superpowers/specs/2026-07-24-periodic-orbit-collocation-design.md.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from jaxcont.api import BifProblem
from jaxcont.core.collocation import Collocation, collocation_matrices
from jaxcont.problems.periodic_args import PeriodicArgs
from jaxcont.solvers.implicit import differentiable_root_checked
from jaxcont.utils.precision import default_tol

PyTree = Any


def periodic_residual(U: Array, p: Array, args: PeriodicArgs) -> Array:
    """Collocation residual of a periodic orbit (defects, continuity, phase).

    Everything but the unknowns comes from ``args``: the right-hand side and
    mesh are static fields of :class:`PeriodicArgs`, so one compiled engine
    kernel serves every problem built from the same ``raw_f`` and mesh.
    """
    raw_f, mesh = args.raw_f, args.mesh
    ntst, ncol = mesh.ntst, mesh.ncol
    n = (U.shape[-1] - 1) // (ntst * (1 + ncol))
    h = 1.0 / ntst
    D_np, E_np, gauss_np, gw_np = collocation_matrices(ncol)
    D, E, gw = jnp.asarray(D_np), jnp.asarray(E_np), jnp.asarray(gw_np)

    mesh_states = U[: ntst * n].reshape(ntst, n)
    coll_states = U[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
    T = U[-1]
    v = jnp.concatenate([mesh_states[:, None, :], coll_states], axis=1)  # (ntst, ncol+1, n)
    # D annihilates constants and E preserves them. Contract local differences
    # so float32 rounding of the coefficients cannot turn the orbit offset
    # into an O(eps * ||D|| * ||U||) defect on every mesh interval.
    local = v - mesh_states[:, None, :]
    # Invariant: full-precision contractions; reduced GPU matmul precision corrupts D.
    with jax.default_matmul_precision("float32"):
        Dv = jnp.einsum("jk,ikc->ijc", D, local)
        extrap = mesh_states + jnp.einsum("k,ikc->ic", E, local)  # (ntst, n)
    f_at_v = jax.vmap(jax.vmap(lambda u: raw_f(u, p, args.user_args)))(v[:, 1:, :])
    defect = Dv[:, 1:, :] - T * h * f_at_v  # (ntst, ncol, n)
    u_next = jnp.roll(mesh_states, -1, axis=0)
    continuity = u_next - extrap  # (ntst, n)
    phase = jnp.sum(
        gw[None, :, None] * h * (coll_states - args.u_ref_coll) * args.uref_prime_coll
    )
    return jnp.concatenate([defect.flatten(), continuity.flatten(), jnp.array([phase])])


def periodic_step_update(args: PeriodicArgs, U: Array, p: Array) -> PeriodicArgs:
    """Re-anchor the phase condition on the accepted orbit ``U`` at ``p``.

    The reference becomes ``U``'s collocation states and their time
    derivative ``raw_f(u, p, user_args)``, so the integral phase condition
    always compares a step against the previous orbit, not the seed. Shapes
    and dtypes of ``args`` are preserved.
    """
    ntst, ncol = args.mesh.ntst, args.mesh.ncol
    n = args.u_ref_coll.shape[-1]
    coll = U[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
    prime = jax.vmap(jax.vmap(lambda u: args.raw_f(u, p, args.user_args)))(coll)
    return PeriodicArgs(
        coll.astype(args.u_ref_coll.dtype),
        prime.astype(args.uref_prime_coll.dtype),
        args.user_args, args.raw_f, args.mesh,
    )


def periodic_orbit_problem(
    f: Callable[[Array, Array, PyTree], Array],
    u_trajectory: Array,
    t_trajectory: Array,
    period0: float,
    p0: float,
    mesh: Collocation,
    *,
    args: PyTree = None,
    tol: float | None = None,
    max_iter: int = 50,
) -> BifProblem:
    """
    Build a periodic-orbit :class:`~jaxcont.api.BifProblem` via fixed-mesh
    orthogonal collocation.

    ``tol`` is the seed-refinement residual tolerance (default: ``default_tol`` of
    the trajectory dtype). ``max_iter`` bounds Newton refinement. Eager
    construction raises ``ValueError`` for a nonpositive ``period0``,
    insufficient trajectory coverage, failed refinement, a nonpositive refined
    period, or a collapsed orbit; validation is skipped for traced values
    under JAX transformations.

    ``f(u, p, args)`` is the same right-hand-side convention used for
    equilibrium problems; ``args`` (default ``None``) is passed to ``f``
    everywhere it is evaluated, and is differentiable (``jax.grad`` of the
    converged orbit with respect to it works). The returned problem's own
    ``args`` is a :class:`~jaxcont.problems.periodic_args.PeriodicArgs` that
    wraps it together with the phase-condition reference data. The problem's
    ``step_update`` is :func:`periodic_step_update`, so during continuation
    the phase condition is re-anchored on each accepted orbit.

    ``u_trajectory``/``t_trajectory`` is a caller-supplied coarse trajectory
    guess (from the caller's own simulation -- JaxCont does not integrate
    ODEs itself); ``period0`` is the corresponding period guess. Times are
    rebased to the first sample; only ``[0, period0)`` of the trajectory is
    read, and a gap between its last sample and ``period0`` is closed linearly
    back to the first sample. Both are resampled onto ``mesh`` and refined to
    convergence (via
    :func:`~jaxcont.solvers.implicit.differentiable_root_checked`) before being
    returned as the ``BifProblem``'s ``u0``. This factory-level refinement
    is kept even though ``pseudo_arclength_scan``/``natural_scan`` now also
    Newton-correct their own seed (a later hardening fix, so a hand-built
    ``BifProblem`` no longer risks a silently-unconverged seed at the
    continuation level either): refining here uses a tolerance matched to
    this problem's own collocation residual scale, ahead of the scan
    engine's generic corrector, and produces a better-conditioned starting
    tangent for the branch that follows.

    This factory's *construction* (this function call itself) is safe under
    outer ``jax.grad``, ``jax.vmap``, and ``jax.jit`` transformations,
    including when the caller's own trajectory and period guesses depend on
    the differentiated parameter.

    Note: do not pass ``events=[jc.Hopf()]`` for the returned problem --
    ``Hopf`` eigendecomposes the full collocation Jacobian, which is not a
    meaningful dynamical quantity for a periodic orbit (the periodic-orbit
    analogues are ``jc.PeriodDoubling()`` and ``jc.NeimarkSacker()``).
    ``events=[jc.Fold()]`` detects folds of cycles with a tangent-based scan
    and extended-system refinement.
    """
    ntst, ncol = mesh.ntst, mesh.ncol
    n = u_trajectory.shape[-1]
    period_ok = period0 > 0
    if not isinstance(period_ok, jax.core.Tracer) and not bool(period_ok):
        raise ValueError("period0 must be positive")
    t_trajectory = t_trajectory - t_trajectory[0]
    covered = t_trajectory[-1] >= 0.95 * period0
    if not isinstance(covered, jax.core.Tracer) and not bool(covered):
        raise ValueError("trajectory must cover at least 0.95 * period0")

    _D, _E, gauss_np, _gw = collocation_matrices(ncol)
    gauss = jnp.asarray(gauss_np)

    t_last = t_trajectory[-1]
    gap = period0 - t_last

    def resample(tau: Array) -> Array:
        # Samples past period0 are never read. A gap between the last sample
        # and period0 is closed linearly from u[-1] back to u[0].
        t = tau * period0
        inside = jnp.stack([jnp.interp(t, t_trajectory, u_trajectory[:, c]) for c in range(n)])
        s = jnp.where(gap > 0, (t - t_last) / jnp.where(gap > 0, gap, 1.0), 0.0)
        closing = u_trajectory[-1] + s * (u_trajectory[0] - u_trajectory[-1])
        return jnp.where(t > t_last, closing, inside)

    mesh_tau = jnp.arange(ntst) / ntst
    coll_tau = (jnp.arange(ntst)[:, None] + gauss[None, :]) / ntst

    mesh_guess = jax.vmap(resample)(mesh_tau)  # (ntst, n)
    coll_guess = jax.vmap(jax.vmap(resample))(coll_tau)  # (ntst, ncol, n)

    def pack(mesh_states: Array, coll_states: Array, T: Array) -> Array:
        return jnp.concatenate(
            [mesh_states.flatten(), coll_states.flatten(), jnp.reshape(T, (1,))]
        )

    dtype = mesh_guess.dtype
    p0_arr = jnp.asarray(p0, dtype=dtype)
    period0_arr = jnp.asarray(period0, dtype=dtype)
    uref_prime_coll = jax.vmap(jax.vmap(lambda u: f(u, p0_arr, args)))(coll_guess)
    pargs = PeriodicArgs(coll_guess, uref_prime_coll, args, f, mesh)

    # Every p0-dependent value travels through `theta` instead of being
    # captured by the closure below. Capturing `uref_prime_coll` (which is
    # built from `p0`) used to leak a tracer across `lax.while_loop`'s trace
    # boundary, so `jax.grad` through this factory raised
    # `UnexpectedTracerError`. These values are still computed once, here,
    # outside the Newton loop -- passing them as `theta` entries costs no
    # extra work per iteration, it just makes the dependency explicit to JAX.
    # `differentiable_root`'s callable-seed form does the same for the seed.
    theta = (p0_arr, mesh_guess, coll_guess, uref_prime_coll, period0_arr, args)

    def _seed(th):
        _p, seed_mesh, seed_coll, _uref_prime, seed_period, _user = th
        return pack(seed_mesh, seed_coll, seed_period)

    def _extended(U, th):
        p, _seed_mesh, ref_coll, ref_prime, _seed_period, user = th
        return periodic_residual(U, p, PeriodicArgs(ref_coll, ref_prime, user, f, mesh))

    tol = default_tol(mesh_guess.dtype) if tol is None else tol
    U0, converged = differentiable_root_checked(
        _extended, _seed, theta, tol=tol, max_iter=max_iter
    )
    if not isinstance(converged, jax.core.Tracer):
        if not bool(converged):
            raise ValueError("periodic seed refinement did not converge")
        if not bool(U0[-1] > 0):
            raise ValueError("refined period must be positive")
        states = U0[:-1].reshape(-1, n)
        mean = jnp.mean(states, axis=0)
        amplitude = jnp.max(jnp.linalg.norm(states - mean, axis=1))
        if not bool(amplitude > 10 * tol * (1 + jnp.linalg.norm(mean))):
            raise ValueError("refined orbit amplitude is too small")

    return BifProblem(
        f=periodic_residual, u0=U0, p0=p0_arr, args=pargs, kind="periodic",
        step_update=periodic_step_update,
    )
