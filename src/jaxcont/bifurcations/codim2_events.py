"""
Codim-2 events along two-parameter curves (see bifurcations/curves.py).

Kept separate from ``events.py`` deliberately: that module covers codim-1
events along ordinary branches, these are only meaningful along curves.

Every event here carries its own ``raw_f`` and ``free`` index.
``detect_events``'s generic ``rhs`` parameter is the EXTENDED-system
residual for a curve problem, not the original ODE, so reusing it would
reproduce -- in reverse -- the equilibrium-only footgun ``Hopf`` originally
had. This is the same reason ``PeriodDoubling``/``NeimarkSacker`` carry
``raw_f``/``mesh``.

See docs/superpowers/specs/2026-08-17-two-parameter-continuation-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax.numpy as jnp
import numpy as np
from jax import Array, jacfwd

from jaxcont.bifurcations.codim2 import (
    bogdanov_takens_point,
    cusp_point,
    double_hopf_point,
    generalized_hopf_point,
    zero_hopf_point,
)
from jaxcont.bifurcations.curve_layout import decode_curve_state, CurveModes, orient_curve_modes
from jaxcont.bifurcations.events import (
    BranchPoint,
    Event,
    EventHit,
    _classify_tol,
    failed_hit,
)
from jaxcont.bifurcations.fold_normal_form import fold_coefficient
from jaxcont.bifurcations.hopf_normal_form import lyapunov_coefficient
from jaxcont.bifurcations.modes import crossings, mode_brackets, track_modes
from jaxcont.utils.precision import default_tol

PyTree = Any


class _CurveEvent:
    """Shared decoding for events that live on a two-parameter curve.

    Deliberately NOT a dataclass: it declares no fields and exists only to
    share methods. Making it one would put an empty dataclass in the MRO
    ahead of each event's own field list for no benefit.

    Subclasses set ``curve`` to ``"fold"`` or ``"hopf"``. The curve type is
    checked against the explicit layout carried by each BranchPoint.
    """

    def _decode(self, point: BranchPoint):
        """Return ``(u, p, extra)`` in the ORIGINAL system's coordinates.

        ``extra`` is the null vector ``v`` for a fold curve, or
        ``(q1, q2, omega)`` for a Hopf curve.
        """
        layout = point.curve_layout
        if layout is None:
            raise ValueError("curve layout is required for curve decoding")
        if layout.kind != self.curve or layout.free != self.free:
            raise ValueError("Event and curve layout disagree")
        u, p, modes = decode_curve_state(point.u, point.p, layout, self.raw_f, self.args)
        extra = modes.v if self.curve == "fold" else (modes.q1, modes.q2, modes.omega)
        return u, p, extra

    def _eigenvalues(self, u: Array, p: Array) -> Array:
        """Spectrum of the ORIGINAL system's Jacobian -- never the extended
        system's, which is what the branch state actually holds."""
        jac = jacfwd(lambda uu: self.raw_f(uu, p, self.args))(u)
        return jnp.linalg.eigvals(jac)

    def _tracked_spectra(self, points):
        """``(spectra, extras)``: the original system's spectra at ``points``
        in tracked-mode order (``(N, n)``), and each point's decoded
        ``extra`` (see ``_decode``)."""
        rows, extras = [], []
        for point in points:
            u, p, extra = self._decode(point)
            rows.append(np.asarray(self._eigenvalues(u, p)))
            extras.append(extra)
        return track_modes(np.stack(rows)), extras

    def _pinned_mask(self, spectra, extras):
        """``(N, n)`` mask excluding the eigenvalues pinned by the curve's
        own defining condition: the zero on a fold curve, the pair
        ``+-i*omega`` on a Hopf curve."""
        if self.curve == "fold":
            rows = [_drop_nearest(jnp.asarray(r), 0.0 + 0.0j) for r in spectra]
        else:
            rows = [
                _drop_nearest_pinned_pair(jnp.asarray(r), extra[2])
                for r, extra in zip(spectra, extras)
            ]
        return np.asarray(jnp.stack(rows))

    def _scan_modes(self, points):
        """Brackets from ``self.mode_values`` (a tracked-mode ``scan``)."""
        if len(points) < 2:
            return []
        return mode_brackets(*self.mode_values(points))


def _drop_nearest(values: Array, target: complex) -> Array:
    """Mask out the single entry closest to ``target``.

    On a fold curve one eigenvalue is pinned at 0 at every point; on a Hopf
    curve the pair +-i*omega is pinned to the axis. Those are the curve's
    OWN defining conditions, so they carry no information and must be
    excluded before looking for a codim-2 crossing -- the same rule
    stability.floquet.floquet_stable uses to exclude the trivial Floquet
    multiplier 1.
    """
    idx = jnp.argmin(jnp.abs(values - target))
    return jnp.arange(values.shape[0]) != idx


def _drop_nearest_pinned_pair(values: Array, omega, near_zero: float = 1e-6) -> Array:
    """Mask out the pinned Hopf-curve pair +-i*omega, one entry per half.

    Two independent ``_drop_nearest`` calls -- one at ``+i*omega``, one at
    ``-i*omega`` -- work everywhere EXCEPT right near a Bogdanov-Takens
    point, where ``omega`` genuinely approaches 0: there ``+i*omega`` and
    ``-i*omega`` coincide, so ``argmin`` latches onto the SAME single
    entry for both calls, and only one eigenvalue gets excluded instead of
    two. The second (still-pinned) entry then leaks back into the
    candidate mask as if it were a genuine codim-2 candidate.

    Below ``near_zero`` this instead drops the two entries nearest 0 --
    both halves of the collapsing pinned pair -- which is what "exclude
    the pinned pair" means once it has degenerated to a near-double zero.
    """
    if abs(float(omega)) < near_zero:
        first = _drop_nearest(values, 0.0 + 0.0j)
        masked = jnp.where(first, jnp.abs(values), jnp.inf)
        idx2 = jnp.argmin(masked)
        second = jnp.arange(values.shape[0]) != idx2
        return first & second
    return _drop_nearest(values, 1j * omega) & _drop_nearest(values, -1j * omega)


def _refine_tol(tolerance, dtype):
    """Solver tolerance for a refine step: never tighter than ``dtype``'s default."""
    floor = default_tol(dtype)
    return floor if tolerance is None else max(tolerance, floor)


@dataclass(frozen=True)
class Cusp(_CurveEvent, Event):
    """A cusp (CP) point on a fold curve.

    Test function: the fold's quadratic normal-form coefficient ``a``
    (``fold_normal_form.fold_coefficient``). ``a != 0`` is the fold's
    non-degeneracy condition and ``a == 0`` is precisely the cusp
    condition, so this needs no eigenvalues at all.

    Abbreviation **CP**, see ``bifurcations.taxonomy.describe("CP")``.
    """

    raw_f: Callable[[Array, Array, PyTree], Array]
    free: int = 1
    curve: str = "fold"
    args: PyTree = None
    kind: str = "cusp"

    def test_function(self, point: BranchPoint) -> float:
        u, p, v = self._decode(point)
        return float(fold_coefficient(self.raw_f, u, p, v, self.args))

    def scan(self, points):
        if len(points) < 2:
            return []
        decoded = [self._decode(pt) for pt in points]
        raw = jnp.stack([extra for _, _, extra in decoded])
        valid = jnp.array([jnp.all(jnp.isfinite(pt.u)) for pt in points])
        modes, eligible = orient_curve_modes(CurveModes(v=raw), valid)
        values = np.array([float(fold_coefficient(self.raw_f, u, p, v, self.args))
                           for (u, p, _), v in zip(decoded, modes.v)])
        candidates = crossings(values, np.asarray(valid), eligible=np.asarray(eligible))
        return [(i, i+1, None) for i in candidates]

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations, mode=None) -> EventHit:
        u_l, p_l, _ = self._decode(left)
        u_r, p_r, _ = self._decode(right)
        u_star, p_star, _v, converged = cusp_point(
            self.raw_f, (u_l + u_r) / 2, (p_l + p_r) / 2, self.args,
            tol=_refine_tol(tolerance, u_l.dtype), max_iter=max_iterations,
        )
        # codim2 solvers return `converged` as a JAX bool and can return
        # non-finite values on a failed solve. Both `nan < 0` and
        # `abs(nan) < tol` are False, so an unguarded fall-through would
        # emit a confident-looking hit for a point that isn't a cusp --
        # the exact bug Hopf.refine() grew its own guard for.
        ok = bool(converged) and bool(jnp.all(jnp.isfinite(p_star)))
        if not ok:
            return failed_hit(
                self.kind, index, "not_converged", method="extended_system",
            )
        return EventHit(
            kind=self.kind, p=float(p_star[self.free]), u=u_star, index=index,
            info={
                "p": p_star,
                "p_fixed": p_star[1 - self.free],
                "converged": True,
                "method": "extended_system",
            },
        )


@dataclass(frozen=True)
class BogdanovTakens(_CurveEvent, Event):
    """A Bogdanov-Takens (BT) point, where fold and Hopf curves meet.

    On a FOLD curve: a tracked real mode other than the pinned zero
    eigenvalue changes sign.
    On a HOPF curve: the critical frequency ``omega`` changes sign, i.e.
    the imaginary pair collides into a double zero. A Hopf curve that ends
    with ``|omega| < 10 * default_tol`` and no sign change has stalled at
    the BT point instead of passing it; that is recorded as a failed hit
    with reason ``"curve_terminated"`` (in ``rejected_events``), not as a
    located BT.

    Abbreviation **BT**, see ``bifurcations.taxonomy.describe("BT")``.
    """

    raw_f: Callable[[Array, Array, PyTree], Array]
    free: int = 1
    curve: str = "fold"
    args: PyTree = None
    kind: str = "bogdanov_takens"
    # |Im| at or below this counts as real; None -> events._classify_tol.
    tolerance: Optional[float] = None

    def mode_values(self, points):
        """``(spectra, values, where)`` on a fold curve: tracked spectra,
        their real parts, and the mask of real, non-pinned modes."""
        spectra, extras = self._tracked_spectra(points)
        tol = _classify_tol(self.tolerance, spectra)
        where = self._pinned_mask(spectra, extras) & (np.abs(spectra.imag) <= tol)
        return spectra, spectra.real, where

    def scan(self, points):
        if self.curve != "hopf":
            return self._scan_modes(points)
        omega = np.array([float(self._decode(pt)[2][2]) for pt in points])
        brackets = [(i, i + 1, None) for i in crossings(omega, np.isfinite(omega))]
        if not brackets and len(points) >= 2:
            tol = default_tol(jnp.asarray(points[-1].u).dtype)
            if abs(omega[-1]) < 10 * tol:
                n = len(points)
                brackets = [(n - 2, n - 1, "curve_terminated")]
        return brackets

    def refine(
        self, left, right, index, rhs, *, tolerance, max_iterations, mode=None,
    ) -> EventHit:
        if mode == "curve_terminated":
            _u, _p, (_q1, _q2, omega) = self._decode(right)
            return failed_hit(self.kind, index, "curve_terminated", omega=float(omega))
        u_l, p_l, _ = self._decode(left)
        u_r, p_r, _ = self._decode(right)
        u_star, p_star, _v0, _v1, converged = bogdanov_takens_point(
            self.raw_f, (u_l + u_r) / 2, (p_l + p_r) / 2, self.args,
            tol=_refine_tol(tolerance, u_l.dtype), max_iter=max_iterations,
        )
        ok = bool(converged) and bool(jnp.all(jnp.isfinite(p_star)))
        if not ok:
            return failed_hit(
                self.kind, index, "not_converged", method="extended_system",
            )
        return EventHit(
            kind=self.kind, p=float(p_star[self.free]), u=u_star, index=index,
            info={
                "p": p_star,
                "p_fixed": p_star[1 - self.free],
                "converged": True,
                "method": "extended_system",
            },
        )


@dataclass(frozen=True)
class ZeroHopf(_CurveEvent, Event):
    """A zero-Hopf (ZH) point: a zero eigenvalue coincides with an
    imaginary pair. Requires ``n >= 3``.

    On a HOPF curve: a tracked REAL mode changes sign (the imaginary pair
    pinned by the curve's defining condition is excluded).
    On a FOLD curve: a tracked complex mode's real part changes sign (the
    pinned zero eigenvalue is excluded; the upper member of each conjugate
    pair is used, so a pair opens one bracket).

    Abbreviation **ZH**, see ``bifurcations.taxonomy.describe("ZH")``.
    """

    raw_f: Callable[[Array, Array, PyTree], Array]
    free: int = 1
    curve: str = "hopf"
    args: PyTree = None
    kind: str = "zero_hopf"
    # |Im| threshold between real and complex; None -> events._classify_tol.
    tolerance: Optional[float] = None

    def mode_values(self, points):
        """``(spectra, values, where)``: tracked spectra, their real parts,
        and the mask of non-pinned candidate modes (real on a Hopf curve,
        upper-half-plane complex on a fold curve)."""
        spectra, extras = self._tracked_spectra(points)
        tol = _classify_tol(self.tolerance, spectra)
        if self.curve == "hopf":
            kind_ok = np.abs(spectra.imag) <= tol
        else:
            kind_ok = spectra.imag > tol
        return spectra, spectra.real, self._pinned_mask(spectra, extras) & kind_ok

    def scan(self, points):
        return self._scan_modes(points)

    def refine(
        self, left, right, index, rhs, *, tolerance, max_iterations, mode=None,
    ) -> EventHit:
        u_l, p_l, _ = self._decode(left)
        u_r, p_r, _ = self._decode(right)
        result = zero_hopf_point(
            self.raw_f, (u_l + u_r) / 2, (p_l + p_r) / 2, self.args,
            tol=_refine_tol(tolerance, u_l.dtype), max_iter=max_iterations,
        )
        u_star, p_star, _v, _q1, _q2, omega_star, converged = result
        ok = bool(converged) and bool(jnp.all(jnp.isfinite(p_star)))
        if not ok:
            return failed_hit(
                self.kind, index, "not_converged", method="extended_system",
            )
        return EventHit(
            kind=self.kind, p=float(p_star[self.free]), u=u_star, index=index,
            info={
                "p": p_star,
                "p_fixed": p_star[1 - self.free],
                "omega": float(omega_star),
                "converged": True,
                "method": "extended_system",
            },
        )


@dataclass(frozen=True)
class GeneralizedHopf(_CurveEvent, Event):
    """A generalized-Hopf / Bautin (GH) point on a Hopf curve: the first
    Lyapunov coefficient ``l1`` crosses zero, so the Hopf's criticality
    flips between supercritical and subcritical.

    Needs no eigenvalues -- ``l1`` is already a scalar that changes sign.

    Abbreviation **GH**, see ``bifurcations.taxonomy.describe("GH")``.
    """

    raw_f: Callable[[Array, Array, PyTree], Array]
    free: int = 1
    args: PyTree = None
    kind: str = "generalized_hopf"
    curve: str = "hopf"

    def test_function(self, point: BranchPoint) -> float:
        u, p, (q1, q2, omega) = self._decode(point)
        return float(lyapunov_coefficient(self.raw_f, u, p, q1, q2, omega, self.args))

    def scan(self, points):
        if len(points) < 2:
            return []
        decoded = [self._decode(pt) for pt in points]
        q1 = jnp.stack([extra[0] for _, _, extra in decoded])
        q2 = jnp.stack([extra[1] for _, _, extra in decoded])
        valid = jnp.array([jnp.all(jnp.isfinite(pt.u)) for pt in points])
        modes, eligible = orient_curve_modes(CurveModes(q1=q1, q2=q2), valid)
        values = np.array([float(lyapunov_coefficient(self.raw_f, u, p, a, b, extra[2], self.args))
                           for (u, p, extra), a, b in zip(decoded, modes.q1, modes.q2)])
        candidates = crossings(values, np.asarray(valid), eligible=np.asarray(eligible))
        return [(i, i+1, None) for i in candidates]

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations, mode=None) -> EventHit:
        u_l, p_l, _ = self._decode(left)
        u_r, p_r, _ = self._decode(right)
        result = generalized_hopf_point(
            self.raw_f, (u_l + u_r) / 2, (p_l + p_r) / 2, self.args,
            tol=_refine_tol(tolerance, u_l.dtype), max_iter=max_iterations,
        )
        u_star, p_star, _q1, _q2, omega_star, converged = result
        ok = bool(converged) and bool(jnp.all(jnp.isfinite(p_star)))
        if not ok:
            return failed_hit(
                self.kind, index, "not_converged", method="extended_system",
            )
        return EventHit(
            kind=self.kind, p=float(p_star[self.free]), u=u_star, index=index,
            info={
                "p": p_star,
                "p_fixed": p_star[1 - self.free],
                "omega": float(omega_star),
                "converged": True,
                "method": "extended_system",
            },
        )


@dataclass(frozen=True)
class DoubleHopf(_CurveEvent, Event):
    """A double-Hopf (HH) point on a Hopf curve: a SECOND complex pair's
    real part changes sign while the first pair is pinned to the axis.

    Detection (``scan``): on tracked modes, excluding the pinned pair, a
    complex mode (``Im > tolerance``, the upper member of each conjugate
    pair) has its real part change sign.

    ``double_hopf_point`` requires a caller-supplied ``seed_b`` (no
    default) because it cannot guess the second pair and degenerates to
    ``nan`` if both blocks land on the same physical pair. Detection along
    the curve produces that second pair naturally (the tracked mode), so
    this event supplies ``seed_b`` itself.

    Abbreviation **HH**, see ``bifurcations.taxonomy.describe("HH")``.
    """

    raw_f: Callable[[Array, Array, PyTree], Array]
    free: int = 1
    args: PyTree = None
    kind: str = "double_hopf"
    curve: str = "hopf"
    # |Im| above this counts as complex; None -> events._classify_tol.
    tolerance: Optional[float] = None
    separation_tolerance: float = 1e-3

    def mode_values(self, points):
        """``(spectra, values, where)``: tracked spectra, their real parts,
        and the mask of non-pinned upper-half-plane complex modes."""
        spectra, extras = self._tracked_spectra(points)
        tol = _classify_tol(self.tolerance, spectra)
        where = self._pinned_mask(spectra, extras) & (spectra.imag > tol)
        return spectra, spectra.real, where

    def scan(self, points):
        return self._scan_modes(points)

    def refine(
        self, left, right, index, rhs, *, tolerance, max_iterations, mode=None,
    ) -> Optional[EventHit]:
        if mode is None:
            brackets = self.scan([left, right])
            if not brackets:
                return None
            mode = brackets[0][2]
        u_l, p_l, _ = self._decode(left)
        u_r, p_r, _ = self._decode(right)
        # seed_b: a real direction pointing at the second Hopf pair, built
        # the same way hopf_normal_form._seed builds a block's q1 -- the
        # real part of the eigenvector of the tracked mode at the bracket's
        # right endpoint. This is exactly what double_hopf_point cannot
        # guess for itself.
        jac = jacfwd(lambda uu: self.raw_f(uu, p_r, self.args))(u_r)
        eigs, evecs = jnp.linalg.eig(jac)
        idx = jnp.argmin(jnp.abs(eigs - mode.right[mode.column]))
        seed_b = jnp.real(evecs[:, idx])
        seed_b = seed_b / jnp.linalg.norm(seed_b)

        result = double_hopf_point(
            self.raw_f, (u_l + u_r) / 2, (p_l + p_r) / 2, self.args,
            seed_b=seed_b,
            tol=_refine_tol(tolerance, u_l.dtype), max_iter=max_iterations,
            separation_tolerance=self.separation_tolerance,
        )
        # double_hopf_point returns a 9-tuple:
        # (u*, p*, q1a, q2a, omega_a, q1b, q2b, omega_b, converged).
        (u_star, p_star, _q1a, _q2a, omega_a,
         _q1b, _q2b, omega_b, converged) = result
        ok = bool(converged) and bool(jnp.all(jnp.isfinite(p_star)))
        if not ok:
            return failed_hit(
                self.kind, index, "not_converged", method="extended_system",
            )
        return EventHit(
            kind=self.kind, p=float(p_star[self.free]), u=u_star, index=index,
            info={
                "p": p_star,
                "p_fixed": p_star[1 - self.free],
                "omega_a": float(omega_a),
                "omega_b": float(omega_b),
                "converged": True,
                "method": "extended_system",
            },
        )
