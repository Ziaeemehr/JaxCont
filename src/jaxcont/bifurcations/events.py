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

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import jax.numpy as jnp
from jax import Array, jacfwd

from jaxcont.bifurcations.fold_solve import fold_point
from jaxcont.stability.eigenvalue import compute_eigenvalues

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


def _eigenvalues_at(rhs: Callable[[Array, float], Array], u: Array, p: float) -> Array:
    """Eigenvalues of df/du at (u, p)."""
    jac = jacfwd(lambda u_eval: rhs(u_eval, p))(u)
    return compute_eigenvalues(jac)


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

    def test_function(self, point: BranchPoint) -> float:
        eigs = point.eigenvalues
        complex_mask = jnp.abs(jnp.imag(eigs)) > self.tolerance
        if not jnp.any(complex_mask):
            return float("nan")
        complex_eigs = eigs[complex_mask]
        idx = jnp.argmin(jnp.abs(jnp.real(complex_eigs)))
        return float(jnp.real(complex_eigs[idx]))

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
            mid_point = BranchPoint(
                p=p_mid, u=u_mid, eigenvalues=_eigenvalues_at(rhs, u_mid, p_mid),
            )
            t_mid = self.test_function(mid_point)
            # Three-way branch, not "left-half or else": a two-way version
            # degenerates (marches toward the wrong endpoint) whenever
            # t_mid lands on an exact zero -- see Global Constraints.
            if t_left * t_mid < 0:
                p_right, u_right, t_right = p_mid, u_mid, t_mid
            elif t_mid * t_right < 0:
                p_left, u_left, t_left = p_mid, u_mid, t_mid
            else:
                break
        p_bif, u_bif = (p_left + p_right) / 2, (u_left + u_right) / 2
        return EventHit(
            kind="hopf", p=float(p_bif), u=u_bif, index=index,
            info={"method": "bisection"},
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
        for i in range(len(points) - 1):
            if test_vals[i] * test_vals[i + 1] < 0:
                hits.append(event.refine(
                    points[i], points[i + 1], (i, i + 1), rhs,
                    tolerance=tolerance, max_iterations=max_iterations,
                ))

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
