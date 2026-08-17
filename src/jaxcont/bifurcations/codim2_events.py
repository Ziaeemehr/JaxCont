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
from typing import Any, Callable

import jax.numpy as jnp
from jax import Array, jacfwd

from jaxcont.bifurcations.codim2 import (
    bogdanov_takens_point,
    cusp_point,
    double_hopf_point,
    generalized_hopf_point,
    zero_hopf_point,
)
from jaxcont.bifurcations.curves import (
    _assemble_p,
    unpack_fold_curve,
    unpack_hopf_curve,
)
from jaxcont.bifurcations.events import BranchPoint, Event, EventHit
from jaxcont.bifurcations.fold_normal_form import fold_coefficient
from jaxcont.bifurcations.hopf_normal_form import lyapunov_coefficient

PyTree = Any


class _CurveEvent:
    """Shared decoding for events that live on a two-parameter curve.

    Deliberately NOT a dataclass: it declares no fields and exists only to
    share methods. Making it one would put an empty dataclass in the MRO
    ahead of each event's own field list for no benefit.

    Subclasses set ``curve`` to ``"fold"`` or ``"hopf"``. The curve type is
    NOT inferred: an Event only ever receives a BranchPoint, which carries
    no problem metadata, so it must be supplied at construction.
    """

    def _decode(self, point: BranchPoint):
        """Return ``(u, p, extra)`` in the ORIGINAL system's coordinates.

        ``extra`` is the null vector ``v`` for a fold curve, or
        ``(q1, q2, omega)`` for a Hopf curve.
        """
        if self.curve == "fold":
            n = (point.u.shape[0] - 1) // 2
            u, p_fixed, v = unpack_fold_curve(point.u, n)
            extra = v
        elif self.curve == "hopf":
            n = (point.u.shape[0] - 2) // 3
            u, p_fixed, q1, q2, omega = unpack_hopf_curve(point.u, n)
            extra = (q1, q2, omega)
        else:
            raise ValueError(
                f"curve must be 'fold' or 'hopf', got {self.curve!r}"
            )
        p = _assemble_p(p_fixed, jnp.asarray(point.p), self.free)
        return u, p, extra

    def _eigenvalues(self, u: Array, p: Array) -> Array:
        """Spectrum of the ORIGINAL system's Jacobian -- never the extended
        system's, which is what the branch state actually holds."""
        jac = jacfwd(lambda uu: self.raw_f(uu, p, self.args))(u)
        return jnp.linalg.eigvals(jac)


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

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        u_l, p_l, _ = self._decode(left)
        u_r, p_r, _ = self._decode(right)
        u_star, p_star, _v, converged = cusp_point(
            self.raw_f, (u_l + u_r) / 2, (p_l + p_r) / 2, self.args,
            tol=max(tolerance, 1e-6), max_iter=max_iterations,
        )
        # codim2 solvers return `converged` as a JAX bool and can return
        # non-finite values on a failed solve. Both `nan < 0` and
        # `abs(nan) < tol` are False, so an unguarded fall-through would
        # emit a confident-looking hit for a point that isn't a cusp --
        # the exact bug Hopf.refine() grew its own guard for.
        ok = bool(converged) and bool(jnp.all(jnp.isfinite(p_star)))
        if not ok:
            return EventHit(
                kind=self.kind, p=float(right.p), u=right.u, index=index,
                info={"converged": False, "method": "extended_system"},
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

    On a FOLD curve: the second eigenvalue reaches zero (the first is
    pinned at zero by the curve's own defining condition).
    On a HOPF curve: the critical frequency ``omega`` reaches zero, i.e.
    the imaginary pair collides into a double zero.

    Abbreviation **BT**, see ``bifurcations.taxonomy.describe("BT")``.
    """

    raw_f: Callable[[Array, Array, PyTree], Array]
    free: int = 1
    curve: str = "fold"
    args: PyTree = None
    kind: str = "bogdanov_takens"
    tolerance: float = 1e-6
    # How far a candidate may sit from the critical set before it is
    # treated as "never relevant". Serves the same purpose as
    # PeriodDoubling.near_unit_circle: without a pre-filter, argmin can
    # silently latch onto an unrelated, always-far eigenvalue and fire a
    # false positive; with too tight a window, the genuine candidate is
    # dropped and the detection is silently lost.
    #
    # This is a PLAIN absolute bound on |eigenvalue|, NOT the log-magnitude
    # window PeriodDoubling needs. The difference is real: Floquet
    # multipliers cluster multiplicatively around 1, so a linear window
    # there is capped below 1.0 by construction (a ~0-magnitude multiplier
    # sits at exactly distance 1), which is what caused the v0.3.1 false
    # negative. Eigenvalues cluster additively around 0, so distance from
    # zero is already the natural measure and has no such ceiling.
    near_critical: float = 2.0

    def test_function(self, point: BranchPoint) -> float:
        u, p, extra = self._decode(point)
        if self.curve == "hopf":
            _q1, _q2, omega = extra
            return float(omega)
        eigs = self._eigenvalues(u, p)
        keep = _drop_nearest(eigs, 0.0 + 0.0j)
        near = jnp.abs(eigs) < self.near_critical
        mask = keep & near & (jnp.abs(jnp.imag(eigs)) < self.tolerance)
        if not jnp.any(mask):
            return float("nan")
        candidates = jnp.where(mask, eigs, jnp.nan)
        idx = jnp.nanargmin(jnp.abs(jnp.real(candidates)))
        return float(jnp.real(eigs[idx]))

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        u_l, p_l, _ = self._decode(left)
        u_r, p_r, _ = self._decode(right)
        u_star, p_star, _v0, _v1, converged = bogdanov_takens_point(
            self.raw_f, (u_l + u_r) / 2, (p_l + p_r) / 2, self.args,
            tol=max(tolerance, 1e-6), max_iter=max_iterations,
        )
        ok = bool(converged) and bool(jnp.all(jnp.isfinite(p_star)))
        if not ok:
            return EventHit(
                kind=self.kind, p=float(right.p), u=right.u, index=index,
                info={"converged": False, "method": "extended_system"},
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

    On a HOPF curve: a REAL eigenvalue crosses zero (the imaginary pair is
    pinned to the axis by the curve's defining condition).
    On a FOLD curve: a complex pair's real part crosses zero (the zero
    eigenvalue is pinned by the curve's defining condition).

    Abbreviation **ZH**, see ``bifurcations.taxonomy.describe("ZH")``.
    """

    raw_f: Callable[[Array, Array, PyTree], Array]
    free: int = 1
    curve: str = "hopf"
    args: PyTree = None
    kind: str = "zero_hopf"
    tolerance: float = 1e-6
    near_critical: float = 2.0

    def test_function(self, point: BranchPoint) -> float:
        u, p, extra = self._decode(point)
        eigs = self._eigenvalues(u, p)
        if self.curve == "hopf":
            # Exclude the pinned imaginary pair; watch a real eigenvalue.
            # The candidate IS real here, so its full magnitude equals the
            # quantity being watched -- |eigs| is the correct pre-filter.
            near = jnp.abs(eigs) < self.near_critical
            _q1, _q2, omega = extra
            keep = _drop_nearest_pinned_pair(eigs, omega)
            mask = keep & near & (jnp.abs(jnp.imag(eigs)) < self.tolerance)
        else:
            # Exclude the pinned zero; watch a complex pair's REAL PART
            # crossing zero. The pre-filter must gate on that real part,
            # not the full complex magnitude |eigs| -- a genuine candidate
            # can have Re(eigs) ~ 0 while sitting at a large |eigs| (e.g.
            # omega=3), which the magnitude filter would wrongly reject.
            # Matches DoubleHopf._second_pair's identical situation below.
            near = jnp.abs(jnp.real(eigs)) < self.near_critical
            keep = _drop_nearest(eigs, 0.0 + 0.0j)
            mask = keep & near & (jnp.abs(jnp.imag(eigs)) > self.tolerance)
        if not jnp.any(mask):
            return float("nan")
        candidates = jnp.where(mask, eigs, jnp.nan)
        idx = jnp.nanargmin(jnp.abs(jnp.real(candidates)))
        return float(jnp.real(eigs[idx]))

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        u_l, p_l, _ = self._decode(left)
        u_r, p_r, _ = self._decode(right)
        result = zero_hopf_point(
            self.raw_f, (u_l + u_r) / 2, (p_l + p_r) / 2, self.args,
            tol=max(tolerance, 1e-6), max_iter=max_iterations,
        )
        u_star, p_star, _v, _q1, _q2, omega_star, converged = result
        ok = bool(converged) and bool(jnp.all(jnp.isfinite(p_star)))
        if not ok:
            return EventHit(
                kind=self.kind, p=float(right.p), u=right.u, index=index,
                info={"converged": False, "method": "extended_system"},
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

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        u_l, p_l, _ = self._decode(left)
        u_r, p_r, _ = self._decode(right)
        result = generalized_hopf_point(
            self.raw_f, (u_l + u_r) / 2, (p_l + p_r) / 2, self.args,
            tol=max(tolerance, 1e-6), max_iter=max_iterations,
        )
        u_star, p_star, _q1, _q2, omega_star, converged = result
        ok = bool(converged) and bool(jnp.all(jnp.isfinite(p_star)))
        if not ok:
            return EventHit(
                kind=self.kind, p=float(right.p), u=right.u, index=index,
                info={"converged": False, "method": "extended_system"},
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
    real part crosses zero while the first pair is pinned to the axis.

    ``double_hopf_point`` requires a caller-supplied ``seed_b`` (no
    default) because it cannot guess the second pair and degenerates to
    ``nan`` if both blocks land on the same physical pair. Detection along
    the curve produces that second pair naturally, so this event supplies
    ``seed_b`` itself.

    Abbreviation **HH**, see ``bifurcations.taxonomy.describe("HH")``.
    """

    raw_f: Callable[[Array, Array, PyTree], Array]
    free: int = 1
    args: PyTree = None
    kind: str = "double_hopf"
    curve: str = "hopf"
    tolerance: float = 1e-6
    near_critical: float = 2.0
    separation_tolerance: float = 1e-3

    def _second_pair(self, point: BranchPoint):
        """Full eigendecomposition of the ORIGINAL system's Jacobian at
        ``point``, masked to the SECOND complex pair (the pinned pair
        excluded via ``_drop_nearest_pinned_pair``). Returns
        ``(eigs, evecs, mask)``.

        Unlike ``_CurveEvent._eigenvalues`` (which uses ``eigvals`` only),
        this needs the eigenVECTORS too: ``refine()`` must hand
        ``double_hopf_point`` a real direction (``seed_b``) pointing at
        this pair, not just its eigenvalue.
        """
        u, p, (_q1, _q2, omega) = self._decode(point)
        jac = jacfwd(lambda uu: self.raw_f(uu, p, self.args))(u)
        eigs, evecs = jnp.linalg.eig(jac)
        keep = _drop_nearest_pinned_pair(eigs, omega)
        near = jnp.abs(jnp.real(eigs)) < self.near_critical
        mask = keep & near & (jnp.abs(jnp.imag(eigs)) > self.tolerance)
        return eigs, evecs, mask

    def test_function(self, point: BranchPoint) -> float:
        eigs, _evecs, mask = self._second_pair(point)
        if not jnp.any(mask):
            return float("nan")
        candidates = jnp.where(mask, eigs, jnp.nan)
        idx = jnp.nanargmin(jnp.abs(jnp.real(candidates)))
        return float(jnp.real(eigs[idx]))

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        u_l, p_l, _ = self._decode(left)
        u_r, p_r, _ = self._decode(right)
        # seed_b: a real direction pointing at the second Hopf pair, built
        # the same way hopf_normal_form._seed builds a block's q1 -- the
        # real part of the eigenvector at the nearest-to-critical
        # surviving complex eigenvalue, taken from the bracket's right
        # endpoint. This is exactly what double_hopf_point cannot guess
        # for itself; it is the whole reason this event exists.
        eigs, evecs, mask = self._second_pair(right)
        candidates = jnp.where(mask, eigs, jnp.nan)
        idx = jnp.nanargmin(jnp.abs(jnp.real(candidates)))
        seed_b = jnp.real(evecs[:, idx])
        seed_b = seed_b / jnp.linalg.norm(seed_b)

        result = double_hopf_point(
            self.raw_f, (u_l + u_r) / 2, (p_l + p_r) / 2, self.args,
            seed_b=seed_b,
            tol=max(tolerance, 1e-6), max_iter=max_iterations,
            separation_tolerance=self.separation_tolerance,
        )
        # double_hopf_point returns a 9-tuple:
        # (u*, p*, q1a, q2a, omega_a, q1b, q2b, omega_b, converged) --
        # confirmed against the live docstring (Task 7 Step 1). Unpacked
        # by full position, not the brief's `result[-3:]` slice, which
        # would have misaligned (that lands on q2b, omega_b, converged).
        (u_star, p_star, _q1a, _q2a, omega_a,
         _q1b, _q2b, omega_b, converged) = result
        ok = bool(converged) and bool(jnp.all(jnp.isfinite(p_star)))
        if not ok:
            return EventHit(
                kind=self.kind, p=float(right.p), u=right.u, index=index,
                info={"converged": False, "method": "extended_system"},
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
