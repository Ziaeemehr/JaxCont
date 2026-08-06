"""
Event protocol for bifurcation detection along a continuation branch.

Replaces the monolithic BifurcationDetector/FoldBifurcation/HopfBifurcation
with small, independently-testable Event implementations (Fold, Hopf), per
ARCHITECTURE.md §4.7. Also fixes issue #7 (duplicate/spurious fold-vs-Hopf
flags): Fold's test function no longer touches eigenvalues at all (it uses
the pseudo-arclength tangent's parameter-component sign change instead), so
a Hopf pair's crossing can no longer masquerade as a fold. See
docs/superpowers/specs/2026-07-23-event-protocol-rewrite-design.md.

Eager-only: this module uses plain Python loops (sign-change scanning,
bisection) and is not jax.jit/jax.vmap-traceable -- matches api.py's
existing NotImplementedError for events=[...] under jax.vmap/jax.jit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import jax.numpy as jnp
from jax import Array

from jaxcont.bifurcations.fold_solve import fold_point
from jaxcont.bifurcations.hopf_normal_form import hopf_point, lyapunov_coefficient
from jaxcont.stability.floquet import floquet_multipliers

PyTree = Any


@dataclass(frozen=True)
class BranchPoint:
    """One point along a continuation branch, as seen by an Event."""

    p: float
    u: Array
    tangent: Optional[Array] = None       # (n+1,); last entry is the dp/ds component
    eigenvalues: Optional[Array] = None   # (n,) complex, or None


@runtime_checkable
class Event(Protocol):
    kind: str

    def test_function(self, point: BranchPoint) -> float:
        """Scalar; a sign change between consecutive points signals an event."""
        ...

    def refine(
        self,
        left: BranchPoint,
        right: BranchPoint,
        index: Tuple[int, int],
        rhs: Callable[[Array, float], Array],
        *,
        tolerance: float,
        max_iterations: int,
    ) -> "EventHit":
        """Precisely locate the event between `left` and `right`."""
        ...


@dataclass(frozen=True)
class EventHit:
    """A detected event along the branch."""

    kind: str
    p: float
    u: Array
    index: Optional[Tuple[int, int]] = None
    info: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Fold(Event):
    """A limit point / fold bifurcation of equilibria.

    Test function: the pseudo-arclength tangent's parameter-component
    (``point.tangent[-1]``). A fold is where the branch turns around in the
    parameter direction, so this component changes sign there -- the
    standard AUTO/MatCont fold indicator. Unlike an eigenvalue-based test,
    this never touches eigenvalues, so a Hopf point's complex pair cannot
    masquerade as a fold (issue #7's root cause).

    Naming follows the standard abbreviations used throughout the
    bifurcation-theory literature (see
    ``jaxcont.bifurcations.taxonomy.BIFURCATION_TYPES``) -- a fold is
    abbreviation **LP**, see ``jaxcont.bifurcations.taxonomy.describe("LP")``.
    """

    kind: str = "fold"

    def test_function(self, point: BranchPoint) -> float:
        return float(point.tangent[-1])

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        u_guess = (left.u + right.u) / 2
        p_guess = (left.p + right.p) / 2
        # fold_point expects f(u, p, args) (3-arg, per fold_solve.py); `rhs`
        # here is the 2-arg (u, p) -> Array callable used throughout this
        # module (matches api.py's rhs2), so adapt with an ignored 3rd arg.
        u_bif, p_bif, null_vector = fold_point(
            lambda u, p, _args: rhs(u, p),
            u_guess, p_guess, tol=tolerance, max_iter=max_iterations,
        )
        return EventHit(
            kind="fold", p=float(p_bif), u=u_bif, index=index,
            info={"null_vector": null_vector, "method": "extended_system"},
        )


@dataclass(frozen=True)
class Hopf(Event):
    """A Hopf bifurcation of equilibria.

    Test function: real part of the complex-conjugate eigenvalue pair with
    smallest ``|Re|`` (``nan`` if no eigenvalue is genuinely complex --
    NOT ``inf``: ``inf`` produces a false sign-change whenever the branch's
    eigenvalue structure transitions from all-real to complex, regardless
    of whether the resulting pair is anywhere near the imaginary axis;
    ``nan`` avoids this for free since ``nan < 0`` is always ``False``).

    Abbreviation **H**, see ``jaxcont.bifurcations.taxonomy.describe("H")``.
    """

    kind: str = "hopf"
    tolerance: float = 1e-6
    # Absolute threshold on a scale-dependent, float32-computed quantity
    # (the first Lyapunov coefficient): l1's magnitude depends on the
    # system's own units/normalization, so a genuinely-degenerate Hopf point
    # in one system may have |l1| well above 1e-6 while a healthy, clearly
    # non-degenerate point in another (differently scaled) system may sit
    # well below it. The default is a reasonable starting point, not a
    # universal constant -- tune per-system if "degenerate" is triggering
    # too eagerly or too rarely.
    l1_tolerance: float = 1e-6

    def test_function(self, point: BranchPoint) -> float:
        eigs = point.eigenvalues
        complex_mask = jnp.abs(jnp.imag(eigs)) > self.tolerance
        if not jnp.any(complex_mask):
            return float("nan")
        complex_eigs = eigs[complex_mask]
        idx = jnp.argmin(jnp.abs(jnp.real(complex_eigs)))
        return float(jnp.real(complex_eigs[idx]))

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        u_guess = (left.u + right.u) / 2
        p_guess = (left.p + right.p) / 2
        u, p, q1, q2, omega0 = hopf_point(
            lambda u, p, _args: rhs(u, p), u_guess, p_guess,
            tol=tolerance, max_iter=max_iterations,
        )
        l1 = lyapunov_coefficient(lambda u, p, _args: rhs(u, p), u, p, q1, q2, omega0)
        # hopf_point's Newton solve (via differentiable_root) has no
        # convergence guarantee: if the bracket's sign change wasn't a real
        # Hopf point (a known occurrence -- see the "no close match --
        # spurious" branches in examples/example_05_neural_mass.py), p/l1/
        # omega0 can come back non-finite (e.g. p=-inf, l1=nan). Both
        # `abs(nan) < tol` and `nan < 0` are False, so without this guard a
        # non-convergent solve would silently fall through to the
        # "subcritical" else-branch below -- a confident-looking label for
        # a result that isn't a Hopf point at all. Check finiteness (and
        # omega0 > 0, since a genuine Hopf point always has a nonzero
        # critical frequency) before trusting the sign of l1.
        finite = jnp.isfinite(p) & jnp.isfinite(l1) & jnp.isfinite(omega0)
        if not bool(finite) or not (float(omega0) > 0.0):
            criticality = "unknown"
        elif abs(l1) < self.l1_tolerance:
            criticality = "degenerate"
        else:
            criticality = "supercritical" if l1 < 0 else "subcritical"
        return EventHit(
            kind="hopf", p=float(p), u=u, index=index,
            info={"omega0": float(omega0), "l1": float(l1),
                  "criticality": criticality, "method": "extended_system"},
        )


def detect_events(
    events: Sequence[Event],
    params: Array,
    states: Array,
    tangents: Optional[Array],
    eigenvalues: Optional[Array],
    rhs: Callable[[Array, float], Array],
    *,
    ds: float,
    tolerance: float = 1e-6,
    max_iterations: int = 20,
) -> List[EventHit]:
    """Detect and refine all requested events along a branch, deduped.

    `params`/`states`/`tangents`/`eigenvalues` are the branch's per-step
    arrays (already trimmed to real points, eager-only). `rhs(u, p)` is the
    system's right-hand side. `ds` sizes the dedup merge window
    (`2 * abs(ds)`): two hits of the SAME kind within that many parameter
    units of each other are treated as the same physical point, keeping the
    earlier one. Hits of different kinds are never merged with each other,
    even if close in parameter -- see Global Constraints for why a
    kind-agnostic merge is wrong (it drops real, distinct, independently-
    verified bifurcations that happen to sit close together).
    """
    points = [
        BranchPoint(
            p=float(params[i]), u=states[i],
            tangent=tangents[i] if tangents is not None else None,
            eigenvalues=eigenvalues[i] if eigenvalues is not None else None,
        )
        for i in range(params.shape[0])
    ]

    hits: List[EventHit] = []
    for event in events:
        test_vals = [event.test_function(pt) for pt in points]
        prev_idx: Optional[int] = None
        prev_val: Optional[float] = None
        for i, val in enumerate(test_vals):
            if not math.isfinite(val):
                continue
            if prev_idx is not None and prev_val is not None and prev_val * val < 0:
                hits.append(event.refine(
                    points[prev_idx], points[i], (prev_idx, i), rhs,
                    tolerance=tolerance, max_iterations=max_iterations,
                ))
            prev_idx, prev_val = i, val

    hits.sort(key=lambda h: h.p)
    merge_window = 2.0 * abs(ds)
    deduped: List[EventHit] = []
    last_p_by_kind: dict = {}
    for hit in hits:
        prev_p = last_p_by_kind.get(hit.kind)
        if prev_p is not None and abs(hit.p - prev_p) < merge_window:
            continue
        last_p_by_kind[hit.kind] = hit.p
        deduped.append(hit)
    return deduped


@dataclass(frozen=True)
class PeriodDoubling(Event):
    """A period-doubling (flip) bifurcation of a periodic orbit.

    Test function: a real Floquet multiplier crosses ``-1`` -- the
    periodic-orbit analogue of ``Hopf``'s imaginary-axis crossing, but a
    magnitude/sign condition on multipliers, not a real-part condition (see
    ``docs/superpowers/specs/2026-07-24-floquet-multipliers-design.md``).
    Only meaningful for ``kind="periodic"`` branches (``point.eigenvalues``
    must be Floquet multipliers, not equilibrium eigenvalues) -- using this
    on an equilibrium branch is the mirror image of the existing
    ``Hopf()``-on-periodic footgun: not enforced by a raise, just
    meaningless.

    ``raw_f``/``mesh`` are required (no meaningful default) because
    ``refine`` must recompute Floquet multipliers at bisection midpoints via
    ``stability.floquet.floquet_multipliers``, whose signature
    (``raw_f, mesh, U, p``) is incompatible with ``detect_events``'s generic
    ``rhs`` parameter (the assembled collocation *residual*, not the raw
    ODE) -- so this event carries its own copy of what
    ``periodic_orbit_problem`` was built with.
    """

    raw_f: Callable[[Array, Array, PyTree], Array]
    mesh: Any
    kind: str = "period_doubling"
    tolerance: float = 1e-6
    # Log-magnitude window (|ln|mult|| < near_unit_circle), not a linear
    # distance from 1: a linear "|mag - 1| < threshold" window is capped
    # below 1.0 by construction, since a magnitude-~0 decaying multiplier
    # (the always-present trivial-like candidate this filter exists to
    # reject -- see test_period_doubling_near_unit_circle_filter_excludes_
    # far_multipliers) sits at *exactly* distance 1.0 from 1, leaving almost
    # no room to widen the window for the true transverse candidate's travel.
    # That ceiling is what caused a recurring false negative: with
    # ds_max=0.1 (default), the worst-case single accepted step can move a
    # multiplier by a factor of up to exp(ds_max * T) (T~2*pi here) =~ 1.87x
    # -- the old linear threshold of 0.9 (magnitude ceiling 1.9) left only
    # ~1.4% margin over that bound, thin enough that ordinary hardware-
    # dependent float32 rounding (different Newton iteration counts feed the
    # adaptive step controller) pushed the post-crossing sample outside the
    # window on some CI runners but not others -- 0 events instead of 1,
    # reproduced on CI, not locally. Log space removes the ceiling: a
    # magnitude-~0 candidate has ln|mult| ~ -12.6 (jnp.log(3.4e-6)), so any
    # reasonable threshold rejects it, while the true candidate gets far more
    # room -- e^2.0 =~ 7.4x its magnitude at the crossing -- before falling
    # out of the window. See docs/superpowers/specs/2026-07-24-period-
    # doubling-neimark-sacker-design.md for the filter's original rationale.
    near_unit_circle: float = 2.0

    def test_function(self, point: BranchPoint) -> float:
        mult = point.eigenvalues
        trivial_idx = jnp.argmin(jnp.abs(mult - 1.0))
        keep = jnp.arange(mult.shape[0]) != trivial_idx
        near_unit = jnp.abs(jnp.log(jnp.abs(mult) + 1e-30)) < self.near_unit_circle
        candidates_mask = keep & near_unit & (jnp.abs(jnp.imag(mult)) < self.tolerance)
        if not jnp.any(candidates_mask):
            return float("nan")
        candidates = jnp.where(candidates_mask, mult, jnp.nan)
        idx = jnp.nanargmin(jnp.abs(jnp.real(candidates) + 1.0))
        return float(jnp.real(mult[idx]) + 1.0)

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        p_left, p_right = left.p, right.p
        u_left, u_right = left.u, right.u
        t_left = self.test_function(left)
        t_right = self.test_function(right)
        for _ in range(max_iterations):
            if abs(p_right - p_left) < tolerance:
                break
            p_mid = (p_left + p_right) / 2
            alpha = (p_mid - p_left) / (p_right - p_left)
            u_mid = u_left + alpha * (u_right - u_left)
            mult_mid = floquet_multipliers(self.raw_f, self.mesh, u_mid, p_mid)
            mid_point = BranchPoint(p=p_mid, u=u_mid, eigenvalues=mult_mid)
            t_mid = self.test_function(mid_point)
            # Three-way branch, not "left-half or else" -- see this file's
            # existing Global Constraints (Hopf has the same shape, for the
            # same reason: a two-way version degenerates whenever t_mid
            # lands on an exact zero).
            if t_left * t_mid < 0:
                p_right, u_right, t_right = p_mid, u_mid, t_mid
            elif t_mid * t_right < 0:
                p_left, u_left, t_left = p_mid, u_mid, t_mid
            else:
                break
        p_bif, u_bif = (p_left + p_right) / 2, (u_left + u_right) / 2
        return EventHit(
            kind="period_doubling", p=float(p_bif), u=u_bif, index=index,
            info={"method": "bisection"},
        )


@dataclass(frozen=True)
class NeimarkSacker(Event):
    """A Neimark-Sacker (torus) bifurcation of a periodic orbit.

    Test function: a complex-conjugate pair of Floquet multipliers crosses
    the unit circle away from the real axis. See ``PeriodDoubling``'s
    docstring for the shared rationale (equilibrium-branch footgun,
    ``raw_f``/``mesh`` fields, why ``detect_events``'s generic ``rhs`` isn't
    used).
    """

    raw_f: Callable[[Array, Array, PyTree], Array]
    mesh: Any
    kind: str = "neimark_sacker"
    tolerance: float = 1e-6
    # See PeriodDoubling.near_unit_circle for why this is a log-magnitude
    # window (not a linear distance from 1, and not 0.9): a linear window is
    # capped below 1.0 by construction and left too thin a margin against
    # ordinary hardware-dependent float32 noise, causing a recurring false
    # negative. Log space rejects a ~0-magnitude decaying candidate just as
    # reliably (ln|mult| ~ -12.6) while giving the true complex-pair
    # candidate far more margin before it's mistaken for "lost" as it moves
    # away from the unit circle.
    near_unit_circle: float = 2.0

    def test_function(self, point: BranchPoint) -> float:
        mult = point.eigenvalues
        trivial_idx = jnp.argmin(jnp.abs(mult - 1.0))
        keep = jnp.arange(mult.shape[0]) != trivial_idx
        near_unit = jnp.abs(jnp.log(jnp.abs(mult) + 1e-30)) < self.near_unit_circle
        candidates_mask = keep & near_unit & (jnp.abs(jnp.imag(mult)) > self.tolerance)
        if not jnp.any(candidates_mask):
            return float("nan")
        candidates = jnp.where(candidates_mask, mult, jnp.nan)
        idx = jnp.nanargmin(jnp.abs(jnp.abs(candidates) - 1.0))
        return float(jnp.abs(mult[idx]) - 1.0)

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        p_left, p_right = left.p, right.p
        u_left, u_right = left.u, right.u
        t_left = self.test_function(left)
        t_right = self.test_function(right)
        for _ in range(max_iterations):
            if abs(p_right - p_left) < tolerance:
                break
            p_mid = (p_left + p_right) / 2
            alpha = (p_mid - p_left) / (p_right - p_left)
            u_mid = u_left + alpha * (u_right - u_left)
            mult_mid = floquet_multipliers(self.raw_f, self.mesh, u_mid, p_mid)
            mid_point = BranchPoint(p=p_mid, u=u_mid, eigenvalues=mult_mid)
            t_mid = self.test_function(mid_point)
            if t_left * t_mid < 0:
                p_right, u_right, t_right = p_mid, u_mid, t_mid
            elif t_mid * t_right < 0:
                p_left, u_left, t_left = p_mid, u_mid, t_mid
            else:
                break
        p_bif, u_bif = (p_left + p_right) / 2, (u_left + u_right) / 2
        return EventHit(
            kind="neimark_sacker", p=float(p_bif), u=u_bif, index=index,
            info={"method": "bisection"},
        )
