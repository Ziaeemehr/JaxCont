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
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jacfwd

from jaxcont.bifurcations.fold_solve import fold_point
from jaxcont.bifurcations.hopf_normal_form import hopf_point, lyapunov_coefficient
from jaxcont.bifurcations.modes import crossings, mode_brackets, track_modes
from jaxcont.solvers.implicit import differentiable_root_checked
from jaxcont.stability.floquet import floquet_multipliers
from jaxcont.utils.precision import default_tol

PyTree = Any


@dataclass(frozen=True)
class BranchPoint:
    """One point along a continuation branch, as seen by an Event."""

    p: float
    u: Array
    tangent: Optional[Array] = None       # (n+1,); last entry is the dp/ds component
    eigenvalues: Optional[Array] = None   # (n,) complex, or None
    trivial_index: Optional[Array] = None  # index in eigenvalues, periodic only


@runtime_checkable
class Event(Protocol):
    """A detectable event along a branch.

    Detection is one of two hooks. ``test_function(point) -> float`` is a
    scalar whose sign change between consecutive points opens a bracket; it
    suits events whose test is continuous along the branch by construction
    (Fold, Cusp, GeneralizedHopf). ``scan(points) -> list[(i, j, mode)]``
    returns the brackets itself, plus a mode identity that ``detect_events``
    passes to ``refine(..., mode=mode)``; spectral events use it to follow
    one tracked mode (``bifurcations.modes``) instead of re-selecting a
    candidate at every point. ``detect_events`` uses ``scan`` when present.
    """

    kind: str

    def refine(
        self,
        left: BranchPoint,
        right: BranchPoint,
        index: tuple[int, int],
        rhs: Callable[[Array, float], Array],
        *,
        tolerance: float,
        max_iterations: int,
    ) -> Optional[EventHit]:
        """Precisely locate the event between `left` and `right`, or
        return None if what looked like a candidate isn't a real hit
        (e.g. BP's classification step rejecting an ordinary fold)."""
        ...


@dataclass(frozen=True)
class EventHit:
    """A detected event along the branch."""

    kind: str
    p: float
    u: Array
    index: Optional[tuple[int, int]] = None
    info: dict = field(default_factory=dict)
    # "validated": refined, converged and passed every check. "failed": the
    # refinement did not produce a trustworthy event; ``info["reason"]`` is one
    # of "not_converged", "outside_bracket", "spectral_check_failed" and ``p``
    # is NaN (no location is claimed).
    status: Literal["validated", "failed"] = "validated"


def failed_hit(kind: str, index, reason: str, **info) -> EventHit:
    """An EventHit recording a refinement that must not be reported as an event."""
    return EventHit(
        kind=kind, p=float("nan"), u=None, index=index,
        info={"reason": reason, **info}, status="failed",
    )


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

    def test_function(self, point: BranchPoint):
        return point.tangent[-1]

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
        # A bracket sign-change is not a convergence guarantee: a failed
        # solve is reported as a failed hit, never as a located fold.
        if not bool(converged):
            return failed_hit("fold", index, "not_converged", method="extended_system")
        return EventHit(
            kind="fold", p=float(p_bif), u=u_bif, index=index,
            info={"null_vector": null_vector, "converged": True, "method": "extended_system"},
        )


def _classify_tol(tolerance: Optional[float], values) -> float:
    """Threshold on ``|Im|`` separating real from complex eigenvalues.

    ``None`` resolves to ``sqrt(default_tol)`` of the eigenvalues' real
    dtype. The square root is deliberate: a perturbed double eigenvalue
    splits by the square root of the perturbation, and Floquet multipliers
    also carry the collocation discretization error, so a real double
    multiplier can come back as a complex pair with an imaginary part far
    above roundoff.
    """
    if tolerance is not None:
        return tolerance
    return default_tol(np.finfo(np.asarray(values).real.dtype).dtype) ** 0.5


def _branch_spectra(points: Sequence[BranchPoint]) -> Optional[np.ndarray]:
    """The points' spectra as one ``(N, n)`` array in tracked-mode order."""
    if len(points) == 0 or any(pt.eigenvalues is None for pt in points):
        return None
    return track_modes(np.stack([np.asarray(pt.eigenvalues) for pt in points]))


def _jacobian_eigenvalues(rhs, u, p) -> np.ndarray:
    return np.asarray(jnp.linalg.eigvals(jacfwd(lambda uu: rhs(uu, p))(u)))


@dataclass(frozen=True)
class Hopf(Event):
    """A Hopf bifurcation of equilibria.

    Detection (``scan``): on tracked modes, ``Re lambda_k`` changes sign
    while ``Im lambda_k > tolerance`` at both bracket ends (the upper member
    of each conjugate pair, so a pair opens one bracket). A refined point is
    accepted only if the eigenvalue nearest ``i*omega0`` is complex.

    Abbreviation **H**, see ``jaxcont.bifurcations.taxonomy.describe("H")``.
    """

    kind: str = "hopf"
    # |Im| above this counts as complex; None -> see _classify_tol.
    tolerance: Optional[float] = None
    # Absolute threshold on a scale-dependent, float32-computed quantity
    # (the first Lyapunov coefficient): l1's magnitude depends on the
    # system's own units/normalization, so a genuinely-degenerate Hopf point
    # in one system may have |l1| well above 1e-6 while a healthy, clearly
    # non-degenerate point in another (differently scaled) system may sit
    # well below it. The default is a reasonable starting point, not a
    # universal constant -- tune per-system if "degenerate" is triggering
    # too eagerly or too rarely.
    l1_tolerance: float = 1e-6

    def scan(self, points: Sequence[BranchPoint]) -> list[tuple[int, int, Any]]:
        spectra = _branch_spectra(points)
        if spectra is None:
            return []
        tol = _classify_tol(self.tolerance, spectra)
        return mode_brackets(spectra, spectra.real, spectra.imag > tol)

    def refine(
        self, left, right, index, rhs, *, tolerance, max_iterations, mode=None,
    ) -> EventHit:
        u_guess = (left.u + right.u) / 2
        p_guess = (left.p + right.p) / 2
        u, p, q1, q2, omega0, converged = hopf_point(
            lambda u, p, _args: rhs(u, p), u_guess, p_guess,
            tol=tolerance, max_iter=max_iterations,
            eigenvalue=None if mode is None else (mode.left[mode.column] + mode.right[mode.column]) / 2,
        )
        l1 = lyapunov_coefficient(lambda u, p, _args: rhs(u, p), u, p, q1, q2, omega0)
        # A converged extended-system solve can still be no Hopf point: a
        # non-finite p/l1/omega0 or omega0 <= 0 (a genuine Hopf point has a
        # nonzero critical frequency) is rejected, never labelled.
        finite = jnp.isfinite(p) & jnp.isfinite(l1) & jnp.isfinite(omega0)
        if not bool(converged):
            return failed_hit("hopf", index, "not_converged", method="extended_system")
        if not (bool(finite) and float(omega0) > 0.0):
            return failed_hit(
                "hopf", index, "spectral_check_failed",
                omega0=float(omega0), l1=float(l1), method="extended_system",
            )
        eigs = _jacobian_eigenvalues(rhs, u, p)
        nearest = eigs[np.argmin(np.abs(eigs - 1j * float(omega0)))]
        if not abs(nearest.imag) > _classify_tol(self.tolerance, eigs):
            return failed_hit(
                "hopf", index, "spectral_check_failed",
                omega0=float(omega0), eigenvalue=complex(nearest), method="extended_system",
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


@dataclass(frozen=True)
class BP(Event):
    """A branch point: two equilibrium branches crossing (pitchfork,
    transcritical). See docs/superpowers/specs/2026-08-26-branch-point-
    switching-design.md for the full derivation.

    Detection (``scan``): a tracked real mode (``|Im lambda_k| <=
    tolerance`` at both bracket ends) changes sign -- the SAME signal an
    ordinary fold gives (df/du singular). Distinguishing a branch point from
    a plain fold requires a strictly stronger condition (the augmented
    [df/du | df/dp] matrix ALSO dropping rank, i.e. df/dp lying in the range
    of df/du) that only refine() can afford to check; refine() rejects
    plain folds by returning None. A refined point is accepted only if the
    eigenvalue nearest 0 is real.

    Abbreviation **BP**, see jaxcont.bifurcations.taxonomy.describe("BP").
    """

    kind: str = "branch_point"
    # |Im| at or below this counts as real; None -> see _classify_tol.
    tolerance: Optional[float] = None
    # Threshold on sigma_min([f_u|f_p]) at the refined point over the largest
    # sigma_max across the bracket and that point; None -> 1e3 * default_tol(dtype).
    rank_tolerance: Optional[float] = None

    def scan(self, points: Sequence[BranchPoint]) -> list[tuple[int, int, Any]]:
        spectra = _branch_spectra(points)
        if spectra is None:
            return []
        tol = _classify_tol(self.tolerance, spectra)
        return mode_brackets(spectra, spectra.real, np.abs(spectra.imag) <= tol)

    def refine(
        self, left, right, index, rhs, *, tolerance, max_iterations, mode=None,
    ) -> Optional[EventHit]:
        u_guess = (left.u + right.u) / 2
        p_guess = (left.p + right.p) / 2
        u_bif, p_bif, _null_vector, converged = fold_point(
            lambda u, p, _args: rhs(u, p),
            u_guess, p_guess, tol=tolerance, max_iter=max_iterations,
        )
        if not bool(converged):
            return failed_hit("branch_point", index, "not_converged", method="extended_system")

        eigs = _jacobian_eigenvalues(rhs, u_bif, p_bif)
        nearest = eigs[np.argmin(np.abs(eigs))]
        if not abs(nearest.imag) <= _classify_tol(self.tolerance, eigs):
            return failed_hit(
                "branch_point", index, "spectral_check_failed",
                eigenvalue=complex(nearest), method="extended_system",
            )

        n = u_bif.shape[0]

        def augmented_svals(u, p):
            f_u = jacfwd(lambda x: rhs(x, p))(u)
            f_p = jacfwd(lambda q: rhs(u, q))(p)
            augmented = jnp.concatenate([f_u, f_p.reshape(n, 1)], axis=1)
            return jnp.linalg.svd(augmented, compute_uv=False)

        singular_values = augmented_svals(u_bif, p_bif)
        # Reference scale: the largest sigma_max of [f_u | f_p] over the bracket
        # endpoints and the refined point. Scaling the RHS cancels in the ratio,
        # and for n = 1 (one singular value, so sigma_min / sigma_max == 1 at
        # any single point) the neighbouring points supply a non-degenerate scale.
        scale = singular_values[0]
        for end in (left, right):
            scale = jnp.maximum(scale, augmented_svals(end.u, end.p)[0])
        # A zero scale means the zero matrix, which is rank deficient.
        rank_ratio = jnp.where(scale == 0, 0, singular_values[-1] / jnp.where(scale == 0, 1, scale))
        rank_tolerance = self.rank_tolerance
        if rank_tolerance is None:
            rank_tolerance = 1e3 * default_tol(u_bif.dtype)
        if not bool(rank_ratio < rank_tolerance):
            return None  # ordinary fold, not a branch point

        return EventHit(
            kind="branch_point", p=float(p_bif), u=u_bif, index=index,
            info={"tangent": right.tangent, "converged": True, "method": "extended_system"},
        )


def _within_bracket(hit: EventHit, left: BranchPoint, right: BranchPoint) -> bool:
    """Whether ``hit`` lies within the bracket widened by one width per side.

    The bracket is the segment from ``left`` to ``right``; its width is that
    segment's length. Distances are taken in the joint ``(u, p)`` space, since
    a fold's parameter does not lie between the bracket's parameters (the
    branch turns back there). When ``hit.u`` has a different shape than the
    branch state (codim-2 events on extended-system branches), only the
    parameter interval is compared.
    """
    if hit.u is not None and jnp.shape(hit.u) == jnp.shape(left.u):
        def vec(u, p):
            u = jnp.ravel(jnp.asarray(u))
            return jnp.concatenate([u, jnp.asarray([p], dtype=u.dtype)])
        a, b, h = vec(left.u, left.p), vec(right.u, right.p), vec(hit.u, hit.p)
        # Floor the width so a degenerate (zero-length) bracket still accepts
        # a hit that sits on it.
        width = jnp.maximum(jnp.linalg.norm(b - a), default_tol(h.dtype))
        dist = jnp.linalg.norm(h - (a + b) / 2)
        return bool(dist <= 1.5 * width)
    width = max(abs(right.p - left.p), default_tol(jnp.asarray(left.u).dtype))
    lo, hi = min(left.p, right.p) - width, max(left.p, right.p) + width
    return bool(lo <= hit.p <= hi)


def _test_values(event, points: Sequence[BranchPoint]) -> np.ndarray:
    """``event.test_function`` at every point, in one ``vmap`` when it allows.

    The batched call is used only when it returns one finite-shaped value per
    point; a test function that is not array-pure (it raises under ``vmap``,
    or the batched shape is wrong) is evaluated point by point instead.
    """
    def optional(name):
        vals = [getattr(pt, name) for pt in points]
        return None if vals[0] is None else jnp.stack([jnp.asarray(v) for v in vals])

    try:
        batched = jax.vmap(
            lambda p, u, t, e: event.test_function(BranchPoint(p, u, t, e))
        )(
            jnp.asarray([pt.p for pt in points]),
            jnp.stack([jnp.asarray(pt.u) for pt in points]),
            optional("tangent"), optional("eigenvalues"),
        )
        vals = np.asarray(batched)
        if vals.shape == (len(points),):
            return vals
    except Exception:  # not array-pure: fall back to the scalar loop
        pass
    return np.array([float(event.test_function(pt)) for pt in points])


def _refined_points_match(hit, kept) -> bool:
    """Same-kind roots agree within dtype-derived refinement accuracy."""
    if hit.kind != kept.kind:
        return False
    if hit.u is None or kept.u is None or np.shape(hit.u) != np.shape(kept.u):
        return False
    u = np.asarray(hit.u)
    v = np.asarray(kept.u)
    x = np.concatenate([u.ravel(), np.asarray([hit.p], dtype=u.dtype)])
    y = np.concatenate([v.ravel(), np.asarray([kept.p], dtype=v.dtype)])
    window = np.sqrt(default_tol(u.dtype)) * (1 + max(np.linalg.norm(x), np.linalg.norm(y)))
    return bool(np.linalg.norm(x - y) <= window)


def _same_point(hit, kept, starts) -> bool:
    """Only adjacent brackets can own duplicate refined roots."""
    return (abs(starts[id(hit)] - starts[id(kept)]) <= 1
            and _refined_points_match(hit, kept))


def detect_events(
    events: Sequence[Event],
    params: Array,
    states: Array,
    tangents: Optional[Array],
    eigenvalues: Optional[Array],
    rhs: Callable[[Array, float], Array],
    *,
    ds: float,
    ds_values: Optional[Array] = None,
    trivial_index: Optional[Array] = None,
    tolerance: float | None = None,
    max_iterations: int = 50,
    args: PyTree = None,
    step_update: Optional[Callable[[PyTree, Array, Array], PyTree]] = None,
    f: Optional[Callable[[Array, Array, PyTree], Array]] = None,
) -> list[EventHit]:
    """Detect and refine all requested events along a branch, deduped.

    `params`/`states`/`tangents`/`eigenvalues` are the branch's per-step
    arrays (already trimmed to real points, eager-only). `rhs(u, p)` is the
    system's right-hand side. `ds` and `ds_values` are accepted for call compatibility; deduplication
    depends on refinement accuracy rather than step size. Two validated hits of the SAME kind are
    one physical point when their brackets are adjacent (start indices
    differ by at most one) and their joint (u, p) distance is at most
    `sqrt(default_tol(dtype)) * (1 + max(norm(root_i), norm(root_j)))`;
    the earlier one is kept. Hits on different sheets of a folded branch are far apart in u
    and never merge; hits of different kinds are never merged. A test value
    that is exactly zero at a sample is reported once (see
    `modes.crossings`).

    ``trivial_index`` supplies each periodic point's flow-direction index in
    ``eigenvalues``; PD/NS scans require it.

    `args` is the problem's ``args``; it is handed to events that declare
    ``needs_args`` (the Floquet events read the periodic problem's right-hand
    side, mesh and user ``args`` from it).

    With the problem's `step_update` (see :class:`~jaxcont.api.BifProblem`)
    and its `f(u, p, args)`, each bracket ``(left, right)`` is refined with
    ``step_update(args, left.u, left.p)`` -- the args its right end was
    corrected with -- both as the ``args`` of ``needs_args`` events and in
    the ``rhs(u, p) = f(u, p, bracket_args)`` handed to ``refine``. Test
    functions read only the branch points, so they need no args.

    The result holds the validated hits (sorted, deduped) followed by the
    failed ones (``status == "failed"``), which are never deduped.
    """
    params = np.asarray(params)
    states = np.asarray(states)
    tangents = None if tangents is None else np.asarray(tangents)
    eigenvalues = None if eigenvalues is None else np.asarray(eigenvalues)
    trivial_index = None if trivial_index is None else np.asarray(trivial_index)
    points = [
        BranchPoint(
            p=params[i], u=states[i],
            tangent=tangents[i] if tangents is not None else None,
            eigenvalues=eigenvalues[i] if eigenvalues is not None else None,
            trivial_index=trivial_index[i] if trivial_index is not None else None,
        )
        for i in range(params.shape[0])
    ]

    if step_update is not None and f is None:
        raise ValueError("detect_events: step_update requires the problem's f(u, p, args)")

    def bracket_inputs(left):
        if step_update is None:
            return rhs, args
        u = jnp.asarray(left.u)
        b_args = step_update(args, u, jnp.asarray(left.p, u.dtype))
        return (lambda uu, pp: f(uu, pp, b_args)), b_args

    hits: list[EventHit] = []
    starts: dict[int, int] = {}
    for event in events:
        scan = getattr(event, "scan", None)
        if scan is not None:
            brackets = scan(points)
        else:
            test_vals = _test_values(event, points)
            brackets = [
                (i, i + 1, None)
                for i in crossings(test_vals, np.ones(len(points), dtype=bool))
            ]
        for i, j, mode in brackets:
            kwargs = {} if scan is None else {"mode": mode}
            b_rhs, b_args = bracket_inputs(points[i])
            if getattr(event, "needs_args", False):
                kwargs["args"] = b_args
            hit = event.refine(
                points[i], points[j], (i, j), b_rhs,
                tolerance=tolerance, max_iterations=max_iterations, **kwargs,
            )
            if hit is None:
                continue
            if hit.status == "validated":
                # Locality: a refined root must lie within the bracket
                # widened by one bracket width on each side; a solve that
                # wandered to a distant root is not this bracket's event.
                if not _within_bracket(hit, points[i], points[j]):
                    hit = failed_hit(
                        hit.kind, hit.index, "outside_bracket", refined_p=hit.p,
                    )
            starts[id(hit)] = i
            hits.append(hit)

    rejected = [h for h in hits if h.status == "failed"]
    hits = [h for h in hits if h.status == "validated"]
    hits.sort(key=lambda h: h.p)
    deduped: list[EventHit] = []
    for hit in hits:
        if not any(_same_point(hit, kept, starts) for kept in deduped):
            deduped.append(hit)
    return deduped + rejected


def _nontrivial_mask(spectra: np.ndarray, points) -> np.ndarray:
    """Exclude each flow-direction multiplier after mode tracking reorders it."""
    if any(pt.trivial_index is None for pt in points):
        raise ValueError("Floquet events require flow-direction trivial_index at every point")
    trivial = np.array([np.asarray(pt.eigenvalues)[int(pt.trivial_index)] for pt in points])
    indices = np.argmin(np.abs(spectra - trivial[:, None]), axis=1)
    return np.arange(spectra.shape[1])[None, :] != indices[:, None]


class _FloquetEvent:
    """Shared detection and refinement for events on Floquet multipliers.

    Not a dataclass: it declares no fields and only shares methods.
    Subclasses define the per-mode test value (``_test_value``), the mode
    condition used while scanning (``_mode_mask``), the critical value the
    refined spectrum is checked against (``_critical``) and the condition the
    multiplier nearest it must satisfy (``_spectral_ok``).
    """

    def scan(self, points: Sequence[BranchPoint]) -> list[tuple[int, int, Any]]:
        spectra = _branch_spectra(points)
        if spectra is None:
            return []
        tol = _classify_tol(self.tolerance, spectra)
        where = self._mode_mask(spectra, tol) & _nontrivial_mask(spectra, points)
        return mode_brackets(spectra, self._test_value(spectra), where)

    needs_args = True

    @staticmethod
    def _multipliers(u, p, args) -> tuple[np.ndarray, int]:
        values, trivial_index = floquet_multipliers(
            args.raw_f, args.mesh, u, p, args=args.user_args,
        )
        return np.asarray(values), int(trivial_index)

    def refine(
        self, left, right, index, rhs, *, tolerance, max_iterations, mode=None,
        args=None,
    ) -> Optional[EventHit]:
        from jaxcont.problems.periodic_args import PeriodicArgs

        if not isinstance(args, PeriodicArgs):
            raise ValueError("Floquet refinement requires PeriodicArgs")
        if mode is None:
            brackets = self.scan([left, right])
            if not brackets:
                return None
            mode = brackets[0][2]
        k = mode.column
        p_left, p_right = left.p, right.p
        u_left, u_right = left.u, right.u
        row_left = mode.left
        t_left = float(self._test_value(mode.left[k]))
        t_right = float(self._test_value(mode.right[k]))
        dtype = jnp.asarray(u_left).dtype
        root_tol = default_tol(dtype) if self.corrector_tol is None else self.corrector_tol
        if tolerance is None:
            tolerance = default_tol(dtype)

        def failed(reason, **info):
            return failed_hit(self.kind, index, reason, method="bisection", **info)

        if not (np.isfinite(t_left) and np.isfinite(t_right)):
            return failed("not_converged")
        if t_left != 0 and t_right != 0 and np.sign(t_left) == np.sign(t_right):
            return failed("not_converged")

        # An exact zero at a bracket end is the event itself.
        if t_left == 0.0:
            p_right, u_right = p_left, u_left
        elif t_right == 0.0:
            p_left, u_left, row_left = p_right, u_right, mode.right
        for _ in range(max_iterations):
            if abs(p_right - p_left) < tolerance:
                break
            p_mid = (p_left + p_right) / 2
            alpha = (p_mid - p_left) / (p_right - p_left)
            u_interp = u_left + alpha * (u_right - u_left)
            # An interpolated collocation state is not on the residual
            # manifold of a curved branch; its multipliers are trusted only
            # after correction at the interpolated parameter.
            u_mid, corrected = differentiable_root_checked(rhs, u_interp, p_mid, tol=root_tol)
            if not bool(corrected):
                return failed("not_converged")
            row_mid = track_modes(np.stack([row_left, self._multipliers(u_mid, p_mid, args)[0]]))[1]
            t_mid = float(self._test_value(row_mid[k]))
            if not np.isfinite(t_mid):
                return failed("not_converged")
            if np.sign(t_left) != np.sign(t_mid) and t_mid != 0:
                p_right, u_right, t_right = p_mid, u_mid, t_mid
            elif np.sign(t_mid) != np.sign(t_right) and t_mid != 0:
                p_left, u_left, t_left, row_left = p_mid, u_mid, t_mid, row_mid
            else:
                if t_mid == 0.0:
                    p_left = p_right = p_mid
                    u_left = u_right = u_mid
                    row_left = row_mid
                break
        if abs(p_right - p_left) >= tolerance:
            return failed("not_converged")
        p_bif = (p_left + p_right) / 2
        u_bif, corrected = differentiable_root_checked(
            rhs, (u_left + u_right) / 2, p_bif, tol=root_tol,
        )
        if not bool(corrected):
            return failed("not_converged")

        multipliers, trivial_index = self._multipliers(u_bif, p_bif, args)
        tracked = track_modes(np.stack([row_left, multipliers]))[1][k]
        keep = np.arange(multipliers.shape[0]) != trivial_index
        distance = np.where(keep, np.abs(multipliers - self._critical(tracked)), np.inf)
        nearest = multipliers[np.argmin(distance)]
        if not self._spectral_ok(nearest, _classify_tol(self.tolerance, multipliers)):
            return failed("spectral_check_failed", multiplier=complex(nearest))
        return EventHit(
            kind=self.kind, p=float(p_bif), u=u_bif, index=index,
            info={"method": "bisection", "corrected": True, "multiplier": complex(nearest)},
        )


@dataclass(frozen=True)
class PeriodDoubling(_FloquetEvent, Event):
    """A period-doubling (flip) bifurcation of a periodic orbit.

    Detection (``scan``): on tracked Floquet multipliers (trivial multiplier
    excluded), a real ``mu_k`` (``|Im mu_k| <= tolerance`` at both bracket
    ends) has ``Re mu_k + 1`` change sign. Refinement bisects on the same
    tracked mode, correcting each midpoint onto the collocation residual, and
    accepts the result only if the non-trivial multiplier nearest ``-1`` is
    real. See ``docs/superpowers/specs/2026-07-24-floquet-multipliers-design.md``.

    Only meaningful for ``kind="periodic"`` branches (``point.eigenvalues``
    must be Floquet multipliers, not equilibrium eigenvalues) -- using this
    on an equilibrium branch is the mirror image of the existing
    ``Hopf()``-on-periodic footgun: not enforced by a raise, just
    meaningless.

    ``refine`` recomputes Floquet multipliers at bisection midpoints, which
    needs the raw ODE right-hand side and the mesh rather than
    ``detect_events``'s generic ``rhs`` (the assembled collocation
    *residual*); it reads them, and the user ``args``, from the problem's
    :class:`~jaxcont.problems.periodic_args.PeriodicArgs` that
    ``detect_events`` passes as ``args``.
    """

    kind: str = "period_doubling"
    corrector_tol: Optional[float] = None  # collocation re-correction; None -> default_tol(dtype)
    # |Im| at or below this counts as real; None -> see _classify_tol.
    tolerance: Optional[float] = None

    @staticmethod
    def _test_value(multipliers):
        return np.real(multipliers) + 1.0

    @staticmethod
    def _mode_mask(spectra, tol):
        return np.abs(spectra.imag) <= tol

    @staticmethod
    def _critical(tracked):
        return -1.0

    @staticmethod
    def _spectral_ok(multiplier, tol):
        return abs(multiplier.imag) <= tol


@dataclass(frozen=True)
class NeimarkSacker(_FloquetEvent, Event):
    """A Neimark-Sacker (torus) bifurcation of a periodic orbit.

    Detection (``scan``): on tracked Floquet multipliers (trivial multiplier
    excluded), a complex ``mu_k`` (``Im mu_k > tolerance`` at both bracket
    ends, the upper member of each conjugate pair) has ``|mu_k| - 1`` change
    sign. The refined point is accepted only if the multiplier nearest the
    tracked mode's projection onto the unit circle is complex. See
    ``PeriodDoubling``'s docstring for the shared rationale
    (equilibrium-branch footgun, why ``detect_events``'s generic ``rhs``
    isn't used).
    """

    kind: str = "neimark_sacker"
    corrector_tol: Optional[float] = None  # collocation re-correction; None -> default_tol(dtype)
    # |Im| above this counts as complex; None -> see _classify_tol.
    tolerance: Optional[float] = None

    @staticmethod
    def _test_value(multipliers):
        return np.abs(multipliers) - 1.0

    @staticmethod
    def _mode_mask(spectra, tol):
        return spectra.imag > tol

    @staticmethod
    def _critical(tracked):
        return tracked / abs(tracked)

    @staticmethod
    def _spectral_ok(multiplier, tol):
        return abs(multiplier.imag) > tol
