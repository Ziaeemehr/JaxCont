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
from jaxcont.solvers.implicit import differentiable_root

PyTree = Any


def periodic_orbit_problem(
    f: Callable[[Array, Array, PyTree], Array],
    u_trajectory: Array,
    t_trajectory: Array,
    period0: float,
    p0: float,
    mesh: Collocation,
) -> BifProblem:
    """
    Build a periodic-orbit :class:`~jaxcont.api.BifProblem` via fixed-mesh
    orthogonal collocation.

    ``f(u, p, args)`` is the same right-hand-side convention used for
    equilibrium problems (``args`` is passed through as ``None`` when this
    factory calls ``f`` internally -- the periodic problem's own ``args``,
    on the returned ``BifProblem``, carries the phase-condition reference
    data instead, not anything ``f`` itself sees).

    ``u_trajectory``/``t_trajectory`` is a caller-supplied coarse trajectory
    guess (from the caller's own simulation -- JaxCont does not integrate
    ODEs itself); ``period0`` is the corresponding period guess. Both are
    resampled onto ``mesh`` and refined to convergence (via
    :func:`~jaxcont.solvers.implicit.differentiable_root`) before being
    returned as the ``BifProblem``'s ``u0`` --
    ``pseudo_arclength_scan``/``natural_scan`` do not Newton-correct their
    starting point, so an unrefined guess would silently be marked
    ``converged=True``.

    Note: this factory's *construction* (this function call itself) is not
    guaranteed safe under an outer ``jax.grad``/``jax.vmap`` -- the
    phase-condition reference derivative is computed from ``p0`` before the
    internal refinement's ``lax.while_loop``, so wrapping this whole call in
    ``jax.grad`` can leak a tracer across that boundary. The *returned*
    ``BifProblem``, once built, is fully ``jit``/``vmap``/``grad``-safe when
    passed to ``jc.continuation()`` (that's an ordinary call into the
    existing scan engine). Making construction itself differentiable is a
    possible future enhancement, not required by the current design spec.

    Note: pass ``settings=jc.ContinuationPar(newton_tol=1e-5, ...)`` (not
    the default ``1e-6``) to ``jc.continuation()`` for the returned
    problem. The collocation residual's achievable precision floor on this
    project's default float32 (~3e-6, verified during design) is tighter
    than the default corrector tolerance can reliably satisfy every step,
    which stalls continuation (every step rejected, step size shrinks to
    ``ds_min``, branch terminates after the initial point).

    Note: do not pass ``events=[jc.Hopf()]`` for the returned problem --
    ``Hopf`` eigendecomposes the full collocation Jacobian, which is not a
    meaningful dynamical quantity for a periodic orbit (the periodic-orbit
    analogues, period-doubling and Neimark-Sacker detection, are future
    features, not yet implemented). ``events=[jc.Fold()]`` is fine and
    meaningful (fold-of-cycles) -- ``Fold``'s tangent-based test is
    dimension-agnostic and needs no special-casing here.
    """
    ntst, ncol = mesh.ntst, mesh.ncol
    n = u_trajectory.shape[-1]
    h = 1.0 / ntst

    D_np, E_np, gauss_np, gw_np = collocation_matrices(ncol)
    D = jnp.asarray(D_np)
    E = jnp.asarray(E_np)
    gauss = jnp.asarray(gauss_np)
    gw = jnp.asarray(gw_np)

    def resample(tau: Array) -> Array:
        t = tau * period0
        return jnp.stack(
            [jnp.interp(t, t_trajectory, u_trajectory[:, c]) for c in range(n)]
        )

    mesh_tau = jnp.arange(ntst) / ntst
    coll_tau = (jnp.arange(ntst)[:, None] + gauss[None, :]) / ntst

    mesh_guess = jax.vmap(resample)(mesh_tau)  # (ntst, n)
    coll_guess = jax.vmap(jax.vmap(resample))(coll_tau)  # (ntst, ncol, n)

    def pack(mesh_states: Array, coll_states: Array, T: Array) -> Array:
        return jnp.concatenate(
            [mesh_states.flatten(), coll_states.flatten(), jnp.array([T])]
        )

    def unpack(U: Array):
        mesh_states = U[: ntst * n].reshape(ntst, n)
        coll_states = U[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
        T = U[-1]
        return mesh_states, coll_states, T

    def residual(U: Array, p: Array, args: PyTree) -> Array:
        u_ref_coll, uref_prime_coll = args
        mesh_states, coll_states, T = unpack(U)
        v = jnp.concatenate([mesh_states[:, None, :], coll_states], axis=1)  # (ntst, ncol+1, n)
        # On GPU, jnp.einsum defaults to reduced (TensorFloat32-like)
        # matmul precision (~1e-3 relative), which corrupts this
        # collocation Jacobian's entries (D's rows can be O(10)) badly
        # enough to stall Newton convergence entirely -- force genuine
        # float32 precision for these two contractions. Verified during
        # design: without this, residual plateaus at ~0.02 regardless of
        # Newton tolerance/iteration count; with it, ~3.4e-6 (the real
        # float32 floor for this system).
        with jax.default_matmul_precision("float32"):
            Dv = jnp.einsum("jk,ikc->ijc", D, v)
            extrap = jnp.einsum("k,ikc->ic", E, v)  # (ntst, n)
        f_at_v = jax.vmap(jax.vmap(lambda u: f(u, p, None)))(v[:, 1:, :])
        defect = Dv[:, 1:, :] - T * h * f_at_v  # (ntst, ncol, n)
        u_next = jnp.roll(mesh_states, -1, axis=0)
        continuity = u_next - extrap  # (ntst, n)
        phase = jnp.sum(
            gw[None, :, None] * h * (coll_states - u_ref_coll) * uref_prime_coll
        )
        return jnp.concatenate(
            [defect.flatten(), continuity.flatten(), jnp.array([phase])]
        )

    U_guess = pack(mesh_guess, coll_guess, jnp.asarray(period0, dtype=mesh_guess.dtype))
    uref_prime_coll = jax.vmap(jax.vmap(lambda u: f(u, p0, None)))(coll_guess)
    args: PyTree = (coll_guess, uref_prime_coll)

    p0_arr = jnp.asarray(p0, dtype=mesh_guess.dtype)
    # tol=1e-5, not differentiable_root's default 1e-8: this project runs
    # float32 by default (no jax_enable_x64 anywhere -- see
    # tests/test_functional_api.py's "tol=1e-6 (float32-reachable)" note),
    # and even with the matmul-precision fix inside residual() above, this
    # ~100-dimensional collocation Newton solve's achievable float32
    # residual floor is ~3e-6 -- below default tol=1e-8's threshold, which
    # would just burn all max_iter iterations without ever satisfying
    # "done". Verified during design: with tol=1e-5 (comfortably above
    # that floor), residual converges cleanly to ~3.4e-6.
    U0 = differentiable_root(lambda U, p: residual(U, p, args), U_guess, p0_arr, tol=1e-5)

    return BifProblem(f=residual, u0=U0, p0=p0_arr, args=args, kind="periodic")
