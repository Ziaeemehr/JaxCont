"""
Floquet multipliers for periodic-orbit stability from the collocation
interval propagators -- see
docs/superpowers/specs/2026-07-24-floquet-multipliers-design.md and
docs/superpowers/specs/2026-10-17-periodic-schur-floquet-design.md.

Two strategies compute the spectrum. ``PeriodicSchur`` decomposes the ordered
interval factors without forming their product, so small physical multipliers
keep their relative accuracy. ``ProductFloquet`` eigendecomposes the formed
monodromy matrix with a matrix ``EigenSolver`` and keeps that solver's order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxcont.core.collocation import (
    Collocation,
    collocation_matrices,
    interval_propagators,
    monodromy_matrix,
)
from jaxcont.solvers.protocols import DenseEigen, EigenSolver
from jaxcont.stability.periodic_schur import periodic_spectrum_modes
from jaxcont.utils.precision import default_tol

PyTree = Any


@dataclass(frozen=True)
class ProductFloquet:
    """Eigenvalues of the formed monodromy matrix ``Phi(T)``.

    ``eigen_solver`` (a matrix ``EigenSolver``) fixes the multiplier order and
    the derivative path. Forming ``Phi`` loses multipliers much smaller than
    the largest one to roundoff. Status is false for non-finite values and when
    no eigenvector resolves the flow direction (the trivial multiplier cannot
    be identified); eager public calls then raise ``RuntimeError``.
    """

    eigen_solver: EigenSolver = DenseEigen()


@dataclass(frozen=True)
class PeriodicSchur:
    """Complex periodic Schur decomposition of the interval propagators.

    ``max_sweeps`` (static, default ``200*n``) bounds the shifted periodic QR
    iteration; an exhausted budget or failed cyclic backward check reports
    non-convergence. Simple multipliers are differentiable; repeated,
    defective or zero multipliers have no derivative.
    """

    max_sweeps: Optional[int] = None


class FloquetSpectrum(NamedTuple):
    """Floquet spectrum at one point, or ``(N, ...)`` along a branch.

    ``values = phase * exp(logabs)`` may under/overflow to zero/infinity
    while ``logabs`` stays usable (``-inf`` only for an exactly zero
    multiplier). ``trivial_index`` is the flow-direction multiplier in this
    strategy's order. When ``converged`` is false the decomposition failed
    or flow identity is unresolved: values, phase and logabs are NaN and the index is
    ``-1``.
    """

    values: Array
    phase: Array
    logabs: Array
    trivial_index: Array
    converged: Array


def _orbit(mesh: Collocation, U: Array):
    ntst, ncol = mesh.ntst, mesh.ncol
    n = (U.shape[-1] - 1) // (ntst * (1 + ncol))
    D_np, E_np, _, _ = collocation_matrices(ncol)
    mesh_states = U[: ntst * n].reshape(ntst, n)
    coll_states = U[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
    return jnp.asarray(D_np), jnp.asarray(E_np), 1.0 / ntst, mesh_states, coll_states, U[-1]


def _aligned(vectors: Array, flow: Array, log_multipliers: Array | None = None) -> tuple[Array, Array]:
    """Select a reconstructed column only if it represents the flow direction.

    The normalized projection residual permits discretization error while
    rejecting physical modes that merely overlap a missing flow mode.
    Complex phase and column/flow scaling do not change the test.

    With ``log_multipliers`` (complex ``log`` of the multipliers, column
    order), a flow along no single column is still resolved when it lies in
    the plane of two columns whose multipliers are both 1 to the same
    tolerance: next to a Hopf point the monodromy is nearly the identity on
    that plane, so its computed eigenvectors are arbitrary there (torBPC at
    ``ntst=25``: ``0.999995 +/- 1.7e-5j``). The trivial index is then the pair
    member nearest 1.
    """
    norms = jnp.linalg.norm(vectors, axis=0)
    flow_norm = jnp.linalg.norm(flow)
    unit_flow = flow.astype(vectors.dtype) / jnp.where(flow_norm > 0, flow_norm, 1)
    unit_vectors = vectors / jnp.where(norms > 0, norms, 1)
    overlap = jnp.conj(unit_vectors).T @ unit_flow
    residual = jnp.linalg.norm(unit_flow[:, None] - unit_vectors * overlap, axis=0)
    usable = (norms > 0) & jnp.isfinite(norms) & jnp.isfinite(residual)
    residual = jnp.where(usable, residual, jnp.inf)
    column = jnp.argmin(residual)
    # Coarse collocation can rotate the flow mode beyond the solve tolerance.
    limit = 10 * jnp.sqrt(default_tol(flow.real.dtype))
    flow_ok = (flow_norm > 0) & jnp.isfinite(flow_norm)
    ok = (residual[column] <= limit) & flow_ok
    if log_multipliers is None or vectors.shape[1] < 2:
        return column, ok
    # Projection of the unit flow onto span(u_i, u_j) for every pair, from
    # the Gram matrix and the overlaps alone (O(n^2) memory, review R1-04):
    # |P f|^2 = (|f_i|^2 + |f_j|^2 - 2 Re(conj(f_i) g_ij f_j)) / (1 - |g_ij|^2).
    m = vectors.shape[1]
    safe_vectors = jnp.where(usable[None, :], unit_vectors, 0)
    gram = jnp.conj(safe_vectors).T @ safe_vectors
    gram_det = 1 - jnp.abs(gram) ** 2
    fi, fj = overlap[:, None], overlap[None, :]
    projected = ((jnp.abs(fi) ** 2 + jnp.abs(fj) ** 2 - 2 * jnp.real(jnp.conj(fi) * gram * fj))
                 / jnp.where(gram_det > limit, gram_det, 1))
    plane_residual = jnp.sqrt(jnp.maximum(1 - projected, 0))
    at_one = jnp.abs(log_multipliers) <= limit
    upper = jnp.asarray(np.triu(np.ones((m, m), bool), 1))
    plane_ok = (upper & usable[:, None] & usable[None, :] & (gram_det > limit)
                & at_one[:, None] & at_one[None, :] & jnp.isfinite(plane_residual))
    plane_residual = jnp.where(plane_ok, plane_residual, jnp.inf)
    pair = jnp.argmin(plane_residual)
    i, j = pair // m, pair % m
    nearer = jnp.where(jnp.abs(log_multipliers[i]) <= jnp.abs(log_multipliers[j]), i, j)
    use_pair = ~ok & (plane_residual.reshape(-1)[pair] <= limit) & flow_ok
    return jnp.where(use_pair, nearer, column), ok | use_pair


def _product_spectrum(strategy, raw_f, mesh, U, p, args):
    D, E, h, mesh_states, coll_states, T = _orbit(mesh, U)
    Phi = monodromy_matrix(raw_f, D, E, h, mesh_states, coll_states, T, p, args)
    # Mode identity is discrete; multiplier derivatives use the supplied solver.
    reference, vectors = jnp.linalg.eig(jax.lax.stop_gradient(Phi))
    column, ok = _aligned(vectors, jax.lax.stop_gradient(raw_f(mesh_states[0], p, args)),
                          jnp.log(reference.astype(jnp.result_type(reference, jnp.complex64))))
    values = strategy.eigen_solver(Phi)
    values = values.astype(jnp.result_type(values, jnp.complex64))
    trivial = jnp.argmin(jnp.abs(values - reference[column]))
    magnitude = jnp.abs(values)
    nonzero = magnitude > 0
    phase = jnp.where(nonzero, values / jnp.where(nonzero, magnitude, 1), 0)
    logabs = jnp.log(magnitude)
    converged = ok & jnp.all(jnp.isfinite(values))
    return values, phase, logabs, trivial, converged


def _schur_spectrum(strategy, raw_f, mesh, U, p, args):
    D, E, h, mesh_states, coll_states, T = _orbit(mesh, U)
    factors = interval_propagators(raw_f, D, E, h, mesh_states, coll_states, T, p, args)
    phase, logabs, converged, right = periodic_spectrum_modes(
        factors, max_sweeps=strategy.max_sweeps)
    # Reconstructed phase-zero right eigenvectors, not Schur vectors, carry
    # the flow identity.
    log_multipliers = jax.lax.stop_gradient(
        logabs + 1j * jnp.angle(phase)).astype(jnp.result_type(phase, jnp.complex64))
    trivial, ok = _aligned(jax.lax.stop_gradient(right),
                           jax.lax.stop_gradient(raw_f(mesh_states[0], p, args)), log_multipliers)
    converged = converged & ok
    # Failed lanes use finite surrogates so masked gradients stay zero.
    safe_logabs = jnp.where(converged, logabs, 0)
    values = jnp.where(converged, phase, 0) * jnp.exp(safe_logabs).astype(phase.dtype)
    return values, phase, logabs, trivial, converged


def _point_spectrum(strategy, raw_f, mesh, U, p, args) -> FloquetSpectrum:
    """Status-reporting spectrum at one point; never raises on failure."""
    if isinstance(strategy, PeriodicSchur):
        values, phase, logabs, trivial, converged = _schur_spectrum(
            strategy, raw_f, mesh, U, p, args)
    elif isinstance(strategy, ProductFloquet):
        values, phase, logabs, trivial, converged = _product_spectrum(
            strategy, raw_f, mesh, U, p, args)
    else:
        raise TypeError(f"unknown Floquet strategy {strategy!r}; use PeriodicSchur or ProductFloquet")
    nan = jnp.asarray(jnp.nan, logabs.dtype)
    return FloquetSpectrum(
        values=jnp.where(converged, values, nan.astype(values.dtype)),
        phase=jnp.where(converged, phase, nan.astype(phase.dtype)),
        logabs=jnp.where(converged, logabs, nan),
        trivial_index=jnp.where(converged, trivial, -1),
        converged=converged,
    )


_compiled_point_spectrum = jax.jit(_point_spectrum, static_argnums=(0, 1, 2))


def _eager_point_spectrum(strategy, raw_f, mesh, U, p, args) -> FloquetSpectrum:
    """Compiled ``_point_spectrum`` for repeated eager calls (event refinement)."""
    return _compiled_point_spectrum(strategy, raw_f, mesh, U, p, args)


def _branch_spectrum(strategy, raw_f, mesh, states, params, args) -> FloquetSpectrum:
    """Status-reporting spectra along ``(N, len(U))`` states (vmapped)."""
    return jax.vmap(lambda U, p: _point_spectrum(strategy, raw_f, mesh, U, p, args))(
        states, params)


def _require_converged(spectrum: FloquetSpectrum) -> FloquetSpectrum:
    if not isinstance(spectrum.converged, jax.core.Tracer) and not bool(
            jnp.all(spectrum.converged)):
        raise RuntimeError(
            "Floquet spectrum did not converge (periodic Schur iteration "
            "budget exhausted, failed backward check, non-finite values or "
            "unresolved flow direction); increase PeriodicSchur(max_sweeps=...) "
            "or check the orbit")
    return spectrum


DEFAULT_FLOQUET = PeriodicSchur()


def _default_strategy():
    return DEFAULT_FLOQUET


def floquet_spectrum(
    raw_f: Callable[[Array, Array, PyTree], Array],
    mesh: Collocation,
    U: Array,
    p: Array,
    floquet_solver=None,
    *,
    args: PyTree = None,
) -> FloquetSpectrum:
    """Floquet spectrum at one converged periodic-orbit point.

    ``U`` is the flat collocation unknown vector (``ntst`` mesh-point states,
    then ``ntst*ncol`` collocation-point states, then the period ``T``), and
    ``raw_f(u, p, args)`` is the ODE right-hand side with the caller's
    ``args`` (a periodic problem's ``PeriodicArgs.user_args``).
    ``floquet_solver`` is ``PeriodicSchur()`` or
    ``ProductFloquet(eigen_solver)``; ``None`` selects ``DEFAULT_FLOQUET``,
    the same default as ``Solvers.floquet``. The trivial index selects the right
    eigenvector most aligned with ``raw_f(x0, p, args)``, never the value
    nearest ``+1`` alone (a pair of multipliers at 1 whose plane holds the
    flow resolves to its member nearest 1). Eager evaluation raises
    ``RuntimeError`` when the spectrum did not converge; traced evaluation
    returns the failed status instead.
    """
    strategy = _default_strategy() if floquet_solver is None else floquet_solver
    return _require_converged(_point_spectrum(strategy, raw_f, mesh, U, p, args))


def floquet_multipliers(
    raw_f: Callable[[Array, Array, PyTree], Array],
    mesh: Collocation,
    U: Array,
    p: Array,
    floquet_solver=None,
    *,
    args: PyTree = None,
) -> tuple[Array, Array]:
    """``(multipliers, trivial_index)`` at one point; see ``floquet_spectrum``."""
    spectrum = floquet_spectrum(raw_f, mesh, U, p, floquet_solver, args=args)
    return spectrum.values, spectrum.trivial_index


def branch_floquet_multipliers(
    raw_f: Callable[[Array, Array, PyTree], Array],
    mesh: Collocation,
    states: Array,
    params: Array,
    floquet_solver=None,
    *,
    args: PyTree = None,
) -> tuple[Array, Array]:
    """Vectorized (vmap) Floquet multipliers along a stored periodic branch
    returning ``(multipliers, trivial_index)`` with shapes ``(N, n)``, ``(N,)``
    -- the periodic-orbit analogue of
    ``core.scan_continuation.branch_eigenvalues``. Eager evaluation raises
    ``RuntimeError`` if any point did not converge."""
    strategy = _default_strategy() if floquet_solver is None else floquet_solver
    spectrum = _require_converged(_branch_spectrum(strategy, raw_f, mesh, states, params, args))
    return spectrum.values, spectrum.trivial_index


def floquet_stable(multipliers: Array, trivial_index: Array, *, logabs: Array | None = None) -> Array:
    """Stability per point, excluding the supplied flow-direction multiplier.

    ``multipliers`` has shape ``(N, n)`` and ``trivial_index`` shape ``(N,)``.
    All remaining multipliers must lie strictly inside the unit circle,
    tested on ``logabs < 0`` when supplied (robust to underflowed values).
    A point without a trivial index (``-1``) or with NaN values is unstable.
    """
    trivial_index = jnp.asarray(trivial_index)
    is_trivial = jnp.arange(multipliers.shape[-1]) == trivial_index[..., None]
    inside = jnp.abs(multipliers) < 1.0 if logabs is None else logabs < 0
    return jnp.all(is_trivial | inside, axis=-1) & (trivial_index >= 0)
