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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
    runtime_checkable,
)

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
        index: tuple[int, int],
        rhs: Callable[[Array, float], Array],
        *,
        tolerance: float,
        max_iterations: int,
    ) -> EventHit:
        """Precisely locate the event between `left` and `right`."""
        ...


@dataclass(frozen=True)
class EventHit:
    """A detected event along the branch."""

    kind: str
    p: float
    u: Array
    index: Optional[tuple[int, int]] = None
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
        u_bif, p_bif, null_vector, converged = fold_point(
            lambda u, p, _args: rhs(u, p),
            u_guess, p_guess, tol=tolerance, max_iter=max_iterations,
        )
        # A bracket sign-change is not a convergence guarantee: the same
        # "silent unchecked bifurcation location" risk Hopf.refine already
        # guards against applies here (Codim-2's Cusp.refine established
        # this fallback shape -- see bifurcations/codim2_events.py).
        if not bool(converged):
            return EventHit(
                kind="fold", p=float(right.p), u=right.u, index=index,
                info={"converged": False, "method": "extended_system"},
            )
        return EventHit(
            kind="fold", p=float(p_bif), u=u_bif, index=index,
            info={"null_vector": null_vector, "converged": True, "method": "extended_system"},
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
        u, p, q1, q2, omega0, converged = hopf_point(
            lambda u, p, _args: rhs(u, p), u_guess, p_guess,
            tol=tolerance, max_iter=max_iterations,
        )
        l1 = lyapunov_coefficient(lambda u, p, _args: rhs(u, p), u, p, q1, q2, omega0)
        # hopf_point's Newton solve (via differentiable_root) has no
        # convergence guarantee: if the bracket's sign change wasn't a real
        # Hopf point (a known occurrence -- see the "no close match --
        # spurious" branches in examples/example_05_neural_mass.py), p/l1/
        # omega0 can come back non-finite (e.g. p=-inf, l1=nan), or --the
        # gap `converged` closes -- come back finite but never actually
        # satisfy the extended-system residual within tol. Both `abs(nan) <
        # tol` and `nan < 0` are False, so without these guards a
        # non-convergent solve would silently fall through to the
        # "subcritical" else-branch below -- a confident-looking label for
        # a result that isn't a Hopf point at all. omega0 > 0 is checked too
        # since a genuine Hopf point always has a nonzero critical frequency.
        finite = jnp.isfinite(p) & jnp.isfinite(l1) & jnp.isfinite(omega0)
        ok = bool(converged) and bool(finite) and (float(omega0) > 0.0)
        if not ok:
            return EventHit(
                kind="hopf", p=float(right.p), u=right.u, index=index,
                info={"omega0": float(omega0), "l1": float(l1),
                      "criticality": "unknown", "converged": False,
                      "method": "extended_system"},
            )
        criticality = (
            "degenerate" if abs(l1) < self.l1_tolerance
            else "supercritical" if l1 < 0 else "subcritical"
        )
        return EventHit(
            kind="hopf", p=float(p), u=u, index=index,
            info={"omega0": float(omega0), "l1": float(l1),
                  "criticality": criticality, "converged": True,
                  "method": "extended_system"},
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
) -> list[EventHit]:
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

    hits: list[EventHit] = []
    for event in events:
        # PeriodDoubling/NeimarkSacker opt into continuity-based candidate
        # tracking via `select_candidate` (see their docstrings for why: a
        # per-point absolute-window filter recurringly false-negatived under
        # hardware-dependent float32 adaptive-step noise). Events without it
        # (Fold, Hopf) fall back to the plain per-point test_function scan.
        select = getattr(event, "select_candidate", None)
        if select is None:
            test_vals = [event.test_function(pt) for pt in points]
            selected: list[Any] = [None] * len(points)
        else:
            test_vals = []
            selected = []
            prev_value = None
            for pt in points:
                value = select(pt, prev_value)
                selected.append(value)
                test_vals.append(float("nan") if value is None else event.test_value(value))
                if value is not None:
                    prev_value = value
        for i in range(len(points) - 1):
            if test_vals[i] * test_vals[i + 1] < 0:
                kwargs = {} if select is None else {"prev_value": selected[i]}
                hits.append(event.refine(
                    points[i], points[i + 1], (i, i + 1), rhs,
                    tolerance=tolerance, max_iterations=max_iterations, **kwargs,
                ))

    hits.sort(key=lambda h: h.p)
    merge_window = 2.0 * abs(ds)
    deduped: list[EventHit] = []
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
    # Only used to pick the FIRST point's candidate (no prior selection to
    # track continuity from yet) -- see select_candidate. Every later point
    # tracks continuity from the previous step's selected multiplier instead
    # of re-filtering by this absolute window, which is what let a recurring
    # false negative resurface three times (v0.3.0, v0.3.1, and again on CI
    # after the log-magnitude fix): whatever window is chosen, an adaptive
    # step can grow the true candidate's magnitude by hardware-dependent
    # amounts (different Newton iteration counts feed the step controller
    # under this project's float32-by-default policy) and land the very next
    # sample just outside it, turning that bracket point's test_function to
    # nan and silently dropping the crossing (0 events instead of 1,
    # reproduced on CI, not locally -- nan comparisons are never `< 0`, so
    # detect_events's sign-change scan sees nothing to report). Nearest-
    # neighbor continuity tracking has no such window to outrun: the
    # candidate is "whichever multiplier is closest to where we last saw it,"
    # which holds regardless of step size, as long as no other multiplier
    # crosses closer to the previous value first (a genericity assumption
    # standard continuation software like AUTO/MatCont also relies on). See
    # docs/superpowers/specs/2026-07-24-period-doubling-neimark-sacker-design.md
    # for the filter's original rationale and events.py's git history for the
    # three false-negative incidents this replaces.
    near_unit_circle: float = 2.0

    def select_candidate(self, point: BranchPoint, prev_value: Optional[Array]) -> Optional[Array]:
        """Pick the real Floquet multiplier tracked as the PD candidate.

        With `prev_value=None` (first point on the branch, or refine's own
        bootstrap), picks the real multiplier nearest -1 within the
        `near_unit_circle` log-magnitude window, excluding the ~1
        trivial-like multiplier. With `prev_value` set, ignores both the
        window AND the real/imaginary classification, picking whichever
        multiplier (still excluding trivial) is nearest `prev_value` in the
        complex plane -- continuity, not a threshold. Dropping the
        real/imaginary gate here matters: a multiplicity-2 real multiplier
        (this module's own test system deliberately has one -- see
        tests/test_period_doubling_neimark_sacker.py's module docstring) is
        a degenerate eigenvalue, and eigensolvers routinely return degenerate
        real roots as a complex-conjugate pair with a tiny (float32-noise
        scale, ~1e-6) nonzero imaginary part. That's comfortably past
        `tolerance` (1e-6) on some steps, which would otherwise make the
        genuinely-tracked candidate fail its OWN classification check and
        fall back to whatever else remains in the pool -- including the
        always-present near-zero decaying multiplier this filter exists to
        reject in the first place. The classification only needs to hold at
        bootstrap, to pick the right *kind* of candidate to start tracking;
        once tracking is anchored to a specific multiplier's trajectory,
        nearest-in-value is a strictly more reliable identity check than
        re-classifying its noisy imaginary part every step.
        """
        mult = point.eigenvalues
        trivial_idx = jnp.argmin(jnp.abs(mult - 1.0))
        keep = jnp.arange(mult.shape[0]) != trivial_idx
        if prev_value is None:
            keep = keep & (jnp.abs(jnp.imag(mult)) < self.tolerance)
            keep = keep & (jnp.abs(jnp.log(jnp.abs(mult) + 1e-30)) < self.near_unit_circle)
            metric = jnp.abs(jnp.real(mult) + 1.0)
        else:
            metric = jnp.abs(mult - prev_value)
        if not bool(jnp.any(keep)):
            return None
        idx = jnp.nanargmin(jnp.where(keep, metric, jnp.nan))
        return mult[idx]

    def test_value(self, value: Array) -> float:
        return float(jnp.real(value) + 1.0)

    def test_function(self, point: BranchPoint) -> float:
        value = self.select_candidate(point, None)
        return float("nan") if value is None else self.test_value(value)

    def refine(
        self, left, right, index, rhs, *, tolerance, max_iterations, prev_value=None,
    ) -> EventHit:
        p_left, p_right = left.p, right.p
        u_left, u_right = left.u, right.u
        v_left = prev_value if prev_value is not None else self.select_candidate(left, None)
        v_right = self.select_candidate(right, v_left)
        t_left = self.test_value(v_left)
        t_right = self.test_value(v_right)
        for _ in range(max_iterations):
            if abs(p_right - p_left) < tolerance:
                break
            p_mid = (p_left + p_right) / 2
            alpha = (p_mid - p_left) / (p_right - p_left)
            u_mid = u_left + alpha * (u_right - u_left)
            mult_mid = floquet_multipliers(self.raw_f, self.mesh, u_mid, p_mid)
            mid_point = BranchPoint(p=p_mid, u=u_mid, eigenvalues=mult_mid)
            v_mid = self.select_candidate(mid_point, v_left)
            t_mid = self.test_value(v_mid)
            # Three-way branch, not "left-half or else" -- see this file's
            # existing Global Constraints (Hopf has the same shape, for the
            # same reason: a two-way version degenerates whenever t_mid
            # lands on an exact zero).
            if t_left * t_mid < 0:
                p_right, u_right, t_right, v_right = p_mid, u_mid, t_mid, v_mid
            elif t_mid * t_right < 0:
                p_left, u_left, t_left, v_left = p_mid, u_mid, t_mid, v_mid
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
    # See PeriodDoubling.near_unit_circle/select_candidate for why this only
    # bounds the FIRST point's candidate: every later point tracks continuity
    # from the previous step's selected complex pair instead of re-filtering
    # by this window, which is what let the same false negative resurface
    # repeatedly under hardware-dependent float32 adaptive-step noise.
    near_unit_circle: float = 2.0

    def select_candidate(self, point: BranchPoint, prev_value: Optional[Array]) -> Optional[Array]:
        """Pick the complex Floquet multiplier tracked as the NS candidate.

        See PeriodDoubling.select_candidate for the bootstrap-vs-continuity
        shape and why the real/imaginary classification only applies at
        bootstrap (dropped once `prev_value` anchors continuity tracking);
        here the bootstrap pool is genuinely-complex multipliers (imaginary
        part above `tolerance`) instead of real ones, and the bootstrap
        target is the unit circle (|mult|=1) instead of -1.
        """
        mult = point.eigenvalues
        trivial_idx = jnp.argmin(jnp.abs(mult - 1.0))
        keep = jnp.arange(mult.shape[0]) != trivial_idx
        if prev_value is None:
            keep = keep & (jnp.abs(jnp.imag(mult)) > self.tolerance)
            keep = keep & (jnp.abs(jnp.log(jnp.abs(mult) + 1e-30)) < self.near_unit_circle)
            metric = jnp.abs(jnp.abs(mult) - 1.0)
        else:
            metric = jnp.abs(mult - prev_value)
        if not bool(jnp.any(keep)):
            return None
        idx = jnp.nanargmin(jnp.where(keep, metric, jnp.nan))
        return mult[idx]

    def test_value(self, value: Array) -> float:
        return float(jnp.abs(value) - 1.0)

    def test_function(self, point: BranchPoint) -> float:
        value = self.select_candidate(point, None)
        return float("nan") if value is None else self.test_value(value)

    def refine(
        self, left, right, index, rhs, *, tolerance, max_iterations, prev_value=None,
    ) -> EventHit:
        p_left, p_right = left.p, right.p
        u_left, u_right = left.u, right.u
        v_left = prev_value if prev_value is not None else self.select_candidate(left, None)
        v_right = self.select_candidate(right, v_left)
        t_left = self.test_value(v_left)
        t_right = self.test_value(v_right)
        for _ in range(max_iterations):
            if abs(p_right - p_left) < tolerance:
                break
            p_mid = (p_left + p_right) / 2
            alpha = (p_mid - p_left) / (p_right - p_left)
            u_mid = u_left + alpha * (u_right - u_left)
            mult_mid = floquet_multipliers(self.raw_f, self.mesh, u_mid, p_mid)
            mid_point = BranchPoint(p=p_mid, u=u_mid, eigenvalues=mult_mid)
            v_mid = self.select_candidate(mid_point, v_left)
            t_mid = self.test_value(v_mid)
            if t_left * t_mid < 0:
                p_right, u_right, t_right, v_right = p_mid, u_mid, t_mid, v_mid
            elif t_mid * t_right < 0:
                p_left, u_left, t_left, v_left = p_mid, u_mid, t_mid, v_mid
            else:
                break
        p_bif, u_bif = (p_left + p_right) / 2, (u_left + u_right) / 2
        return EventHit(
            kind="neimark_sacker", p=float(p_bif), u=u_bif, index=index,
            info={"method": "bisection"},
        )
