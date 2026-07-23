# Event Protocol Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `BifurcationDetector`/`FoldBifurcation`/`HopfBifurcation` with small, independently-testable `Event` implementations (`Fold`, `Hopf`) in a new `bifurcations/events.py`, fixing issue #7 (duplicate/spurious fold-vs-Hopf flags) at the root by making `Fold`'s test function tangent-based instead of eigenvalue-based.

**Architecture:** One new module, `src/jaxcont/bifurcations/events.py`, holds `BranchPoint`, the `Event` protocol, `Fold`, `Hopf`, `EventHit`, and a `detect_events(...)` orchestrator (scan for sign changes → refine each crossing → dedup near-coincident hits across kinds). `api.py` re-exports `Event`/`Fold`/`Hopf`/`EventHit` from there instead of defining its own thin marker dataclasses, and `_run_scan`'s detection call site switches from `BifurcationDetector` to `detect_events`. `detector.py`, and the detection classes in `fold.py`/`hopf.py`, are deleted once nothing references them.

**Tech Stack:** `jax`, `pytest`.

## Global Constraints

- Stays **eager-only**: `events=[...]` under `jax.vmap`/`jax.jit` must keep raising the same `NotImplementedError` it does today (`api.py`'s `_run_scan_traced`, unmodified by this plan). Making detection trace-safe is explicitly out of scope — a separate future roadmap item.
- `Fold`'s test function is the pseudo-arclength tangent's parameter-component (`point.tangent[-1]`) — no eigenvalues anywhere in `Fold`. This is the issue #7 root-cause fix; do not revert to an eigenvalue-based fold test.
- `Fold.refine` always uses the extended-system solve (`fold_solve.fold_point`) — no bisection fallback for folds (the old `fold_extended_system` toggle is gone; there is only one fold-refinement path now).
- `Hopf`'s test function and bisection-refinement behavior are carried over from today's `HopfBifurcation`/`BifurcationDetector.locate_bifurcation`, with two fixes found and verified during design (do not revert either):
  - The **three-way bisection branch** (`left-half / right-half / break-if-neither`) — a simplified two-way branch degenerates to marching toward the wrong endpoint whenever a midpoint's test value lands on an exact zero.
  - The "no complex eigenvalues" sentinel is **`nan`, not `inf`**. `inf` produces a false sign-change (`inf * finite_negative < 0`) at every point where the eigenvalue structure transitions from all-real to complex, regardless of whether the resulting pair's real part is anywhere near zero — verified as the exact cause of a real spurious detection in `example_05_neural_mass.py` (see Task 2). `nan` avoids this for free (`nan < 0` is always `False`), no other code changes needed.
- Dedup is **same-kind only**: after refining all crossings, group by kind, and within a kind drop any hit within `2 * abs(ds)` of the previous kept hit of that same kind. A kind-agnostic merge was tried during design and found to be wrong — it silently dropped a real Hopf point in `example_05_neural_mass.py` that sits only 0.015 in parameter from a real fold, both independently confirmed by BifurcationKit.jl and well inside a naive `2*ds=0.04` window. Do not merge across kinds.
- `sol.bifurcations` (on the legacy `ContinuationSolution`) must stay dict-shaped with `"type"`/`"parameter"`/`"state"` keys — `viz/core.py`'s plotting and `ContinuationSolution.get_bifurcations_by_type` both read it directly and are not touched by this plan.
- `BifurcationDetector`, `FoldBifurcation`, `HopfBifurcation` (including their non-functional `compute_normal_form`/`compute_first_lyapunov_coefficient` placeholder stubs) are deleted outright, not deprecated — matches this project's established pre-1.0 convention.
- `differentiable_root`/`fold_solve.fold_point` (already built) are consumed as-is; no changes to `src/jaxcont/solvers/implicit.py` or `src/jaxcont/bifurcations/fold_solve.py`.
- The `Event`/`Fold`/`Hopf` attribute renames from `_kind` to public `kind`. `tests/test_taxonomy.py:49-50` reads `Fold()._kind`/`Hopf()._kind` directly and must be updated to `.kind` in Task 2 (found during design by running the full suite against the wired-up implementation — not caught by the earlier `events=` usage sweep since this test doesn't use `events=`).
- Reference spec: [docs/superpowers/specs/2026-07-23-event-protocol-rewrite-design.md](../specs/2026-07-23-event-protocol-rewrite-design.md).

---

### Task 1: Build `bifurcations/events.py` and its tests

**Files:**
- Create: `src/jaxcont/bifurcations/events.py`
- Modify (full-file rewrite): `tests/test_bifurcations.py` (currently 45 lines, unit-tests the classes this plan deletes)

**Interfaces:**
- Produces: `BranchPoint(p, u, tangent=None, eigenvalues=None)`; `Event` (a `typing.Protocol`, `@runtime_checkable`, with `kind: str`, `test_function(point) -> float`, `refine(left, right, index, rhs, *, tolerance, max_iterations) -> EventHit`); `Fold()` and `Hopf(tolerance=1e-6)` (both implement `Event`); `EventHit(kind, p, u, index=None, info={})`; `detect_events(events, params, states, tangents, eigenvalues, rhs, *, ds, tolerance=1e-6, max_iterations=20) -> List[EventHit]`. All importable from `jaxcont.bifurcations.events`.
- Consumes: `jaxcont.bifurcations.fold_solve.fold_point` (existing), `jaxcont.stability.eigenvalue.compute_eigenvalues` (existing).

- [ ] **Step 1: Write the failing test**

Replace the full contents of `tests/test_bifurcations.py` with:

```python
"""
Tests for the Event protocol (jaxcont.bifurcations.events): Fold/Hopf test
functions, refinement, and the detect_events orchestrator's dedup logic.
Replaces the old direct tests of FoldBifurcation/HopfBifurcation (deleted
along with BifurcationDetector -- see
docs/superpowers/specs/2026-07-23-event-protocol-rewrite-design.md).
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import jacfwd
import pytest

from jaxcont.bifurcations.events import BranchPoint, EventHit, Fold, Hopf, detect_events


def test_fold_test_function_is_tangent_dp_component():
    # tangent = (du_1, ..., du_n, dp); last entry is the dp component.
    fold = Fold()
    point = BranchPoint(p=0.0, u=jnp.zeros(1), tangent=jnp.array([0.5, 0.02]))
    assert jnp.isclose(fold.test_function(point), 0.02, atol=1e-6)


def test_hopf_test_function():
    hopf = Hopf()
    point = BranchPoint(
        p=0.0, u=jnp.zeros(1),
        eigenvalues=jnp.array([0.01 + 2.0j, 0.01 - 2.0j, -1.0]),
    )
    assert jnp.isclose(hopf.test_function(point), 0.01, atol=1e-6)


def test_hopf_test_function_no_complex_returns_nan():
    # nan, not inf: inf * (a finite real part far from zero) is negative,
    # which would falsely read as a sign-change crossing in detect_events's
    # scan whenever the eigenvalue structure flips from all-real to complex
    # -- regardless of whether that complex pair is anywhere near the
    # imaginary axis. nan avoids this for free (nan < 0 is always False).
    hopf = Hopf()
    point = BranchPoint(
        p=0.0, u=jnp.zeros(1),
        eigenvalues=jnp.array([-0.5, -1.0, -2.0]),
    )
    assert jnp.isnan(hopf.test_function(point))


def test_detect_events_ignores_real_to_complex_transition_far_from_axis():
    # Regression for the inf-sentinel bug found during design (reproduced
    # from example_05_neural_mass.py's actual data): eigenvalue structure
    # changes from all-real to a genuine complex pair between two branch
    # points, but that pair's real part (-9.0) is nowhere near zero. With
    # the old inf sentinel this false-positived as a hopf crossing; with
    # nan it must not.
    def rhs(u, p):
        return jnp.zeros_like(u)

    p_grid = jnp.array([0.0, 0.02])
    states = jnp.zeros((2, 3))
    tangents = jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0]), (2, 1))
    eigenvalues = jnp.stack([
        jnp.array([-0.3 + 0j, -13.0 + 0j, -9.7 + 0j]),
        jnp.array([-9.0 + 4.5j, -9.0 - 4.5j, -0.2 + 0j]),
    ])

    hits = detect_events([Hopf()], p_grid, states, tangents, eigenvalues, rhs, ds=0.01)
    assert hits == []


def test_hopf_refine_converges_to_exact_zero_via_three_way_bisection():
    # Regression for the two-way-bisection bug found during design: a
    # midpoint landing on an exact zero must not make refine() converge to
    # the wrong bracket endpoint.
    def rhs(u, p):
        x, y = u[0], u[1]
        return jnp.array([p * x - 0.1 * y, 0.1 * x + p * y])

    def eigs_at(u, p):
        jac = jacfwd(lambda u_: rhs(u_, p))(u)
        return jnp.linalg.eigvals(jac)

    hopf = Hopf()
    left = BranchPoint(p=-0.05, u=jnp.zeros(2), eigenvalues=eigs_at(jnp.zeros(2), -0.05))
    right = BranchPoint(p=0.05, u=jnp.zeros(2), eigenvalues=eigs_at(jnp.zeros(2), 0.05))
    hit = hopf.refine(left, right, (3, 4), rhs, tolerance=1e-8, max_iterations=50)
    assert hit.kind == "hopf"
    assert abs(hit.p) < 1e-6
    assert hit.info["method"] == "bisection"


def test_detect_events_finds_hopf_not_fold_synthetic_repro():
    # The exact issue #7 mechanism: eigenvalues p +/- 0.1i (a Hopf pair
    # crossing Re=0 at p=0) plus a third eigenvalue fixed at -5. Near p=0
    # the Hopf pair's magnitude (~0.1) is far smaller than 5, so the *old*
    # eigenvalue-closest-to-zero fold test would have fired at p=0 too.
    def rhs(u, p):
        x, y, z = u[0], u[1], u[2]
        return jnp.array([p * x - 0.1 * y, 0.1 * x + p * y, -5.0 * z])

    def eigs_at(u, p):
        jac = jacfwd(lambda u_: rhs(u_, p))(u)
        return jnp.linalg.eigvals(jac)

    p_grid = jnp.array([-0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3])
    states = jnp.zeros((8, 3))
    tangents = jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0]), (8, 1))
    eigenvalues = jnp.stack([eigs_at(jnp.zeros(3), float(p)) for p in p_grid])

    hits = detect_events(
        [Fold(), Hopf()], p_grid, states, tangents, eigenvalues, rhs, ds=0.1,
    )

    assert len(hits) == 1
    assert hits[0].kind == "hopf"
    assert abs(hits[0].p) < 1e-6


def test_detect_events_old_fold_test_would_have_false_positived():
    # Proves the repro actually exercises the old bug: an eigenvalue-based
    # "closest to zero" fold test (today's deleted FoldBifurcation logic)
    # DOES cross zero at the same p=0 as the Hopf pair above.
    def rhs(u, p):
        x, y, z = u[0], u[1], u[2]
        return jnp.array([p * x - 0.1 * y, 0.1 * x + p * y, -5.0 * z])

    def eigs_at(u, p):
        jac = jacfwd(lambda u_: rhs(u_, p))(u)
        return jnp.linalg.eigvals(jac)

    def old_fold_test(eigs):
        idx = jnp.argmin(jnp.abs(eigs))
        return float(jnp.real(eigs[idx]))

    p_grid = [-0.05, 0.05]
    vals = [old_fold_test(eigs_at(jnp.zeros(3), p)) for p in p_grid]
    assert vals[0] * vals[1] < 0  # old test WOULD have flagged a fold here


@dataclass(frozen=True)
class _ThresholdEvent:
    """Minimal test-only Event: fires once, at a fixed parameter value.

    Doesn't subclass Fold/Hopf -- demonstrates Event is a structural
    protocol (ARCHITECTURE.md §4.7: "users subclass Event for custom
    detections") and lets this test drive detect_events's real scan/refine/
    dedup pipeline without depending on fold_point's Newton convergence or
    real eigenvalue math.
    """

    kind: str
    threshold: float
    hit_p: float

    def test_function(self, point: BranchPoint) -> float:
        return point.p - self.threshold

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        return EventHit(kind=self.kind, p=self.hit_p, u=left.u, index=index)


def test_dedup_merges_same_kind_but_not_cross_kind():
    def rhs(u, p):
        return jnp.array([p * u[0]])

    p_grid = jnp.array([0.0, 0.15, 0.3, 0.6])
    states = jnp.zeros((4, 1))

    # near_fold_a and near_fold_b are the SAME kind, both crossing between
    # the same two branch points (p=0.0, p=0.15), 0.01 apart -- within the
    # ds=0.05 merge window (2*abs(ds)=0.1) -- these must merge, keeping the
    # earlier one. near_hopf is a DIFFERENT kind crossing at nearly the same
    # location and must survive dedup (same-kind-only merging is the fix
    # verified during design -- see Global Constraints). far_hopf crosses
    # far away and is unaffected either way.
    near_fold_a = _ThresholdEvent(kind="fold", threshold=0.10, hit_p=0.10)
    near_fold_b = _ThresholdEvent(kind="fold", threshold=0.11, hit_p=0.11)
    near_hopf = _ThresholdEvent(kind="hopf", threshold=0.12, hit_p=0.12)
    far_hopf = _ThresholdEvent(kind="hopf", threshold=0.50, hit_p=0.50)

    hits = detect_events(
        [near_fold_a, near_fold_b, near_hopf, far_hopf],
        p_grid, states, None, None, rhs, ds=0.05,
    )

    assert [(h.kind, h.p) for h in hits] == [("fold", 0.10), ("hopf", 0.12), ("hopf", 0.50)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_bifurcations.py -v`
Expected: FAIL (ERROR at collection) with `ModuleNotFoundError: No module named 'jaxcont.bifurcations.events'`

- [ ] **Step 3: Implement `bifurcations/events.py`**

Create `src/jaxcont/bifurcations/events.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_bifurcations.py -v`
Expected: PASS (all 8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/bifurcations/events.py tests/test_bifurcations.py
git commit -m "feat: add Event protocol (Fold/Hopf/detect_events) fixing issue #7's fold-vs-Hopf false positives"
```

---

### Task 2: Wire `detect_events` into `api.py`

**Files:**
- Modify: `src/jaxcont/api.py:211-258` (the `Event`/`Fold`/`Hopf`/`EventHit` section)
- Modify: `src/jaxcont/api.py:24` (top imports)
- Modify: `src/jaxcont/api.py:438-460` (`_run_scan`'s detection call site)
- Modify: `tests/test_taxonomy.py:49-50` (`Fold()._kind`/`Hopf()._kind` → `.kind`)
- Modify: `examples/example_05_neural_mass.py` (stale issue-#7-open comment → resolved)
- Modify: `examples/example_02_lorenz.py` (stale issue-#7-open comment → resolved)

**Interfaces:**
- Consumes: `Event`, `Fold`, `Hopf`, `EventHit`, `detect_events` from `jaxcont.bifurcations.events` (Task 1).
- Produces: nothing new — `jc.continuation(..., events=[jc.Fold(), jc.Hopf()])`'s public behavior is unchanged in shape (still returns `ContinuationResult` with `.events: List[EventHit]`), only the underlying detection logic and the fold false-positive behavior change.

- [ ] **Step 1: Confirm the baseline passes before wiring**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_functional_api.py tests/test_bifurcation_workflow.py -v`
Expected: PASS (all tests) — this is the regression baseline this task must not break.

- [ ] **Step 2: Capture the pre-fix example output (evidence issue #7 is currently present)**

Run: `MPLBACKEND=Agg JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python examples/example_05_neural_mass.py 2>&1 | grep -c "<->"`
Expected: `5` (today's actual bug shape, verified during design: exact-duplicate
detections, not the mislabeled/no-counterpart pattern the 2026-07-18 issue
#7 write-up originally described — that description has drifted from
current behavior, likely because later commits like the engine
consolidation changed how the detector receives branch data. The 5 raw
rows are 2× `fold E0=-1.8652` (identical), 1× `hopf E0=-1.8500`, and 2×
`fold E0=-1.4630` (identical) — 3 real distinct locations, each doubled
except the Hopf.)

- [ ] **Step 3: Add the `bifurcations.events` import**

In `src/jaxcont/api.py`, change (currently line 24):

```python
from jaxcont.core.continuation import ContinuationProblem, ContinuationSolution
```

to:

```python
from jaxcont.bifurcations.events import Event, Fold, Hopf, EventHit, detect_events
from jaxcont.core.continuation import ContinuationProblem, ContinuationSolution
```

- [ ] **Step 4: Delete `api.py`'s own `Event`/`Fold`/`Hopf`/`EventHit` definitions**

In `src/jaxcont/api.py`, delete (currently lines 211-258):

```python
# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class Event:
    """Marker base for a bifurcation/event detector.

    Naming follows the standard abbreviations used throughout the
    bifurcation-theory literature (see
    ``jaxcont.bifurcations.taxonomy.BIFURCATION_TYPES``) rather than
    inventing new names -- e.g. a fold is ``jc.Fold`` (abbreviation **LP**)
    and a Hopf point is ``jc.Hopf`` (abbreviation **H**).
    """

    #: legacy detector key this event maps onto ("fold" | "hopf")
    _kind: str = ""


@dataclass(frozen=True)
class Fold(Event):
    """A limit point / fold bifurcation of equilibria.

    Abbreviation: **LP** -- ``jaxcont.bifurcations.taxonomy.describe("LP")``.
    """

    _kind: str = "fold"


@dataclass(frozen=True)
class Hopf(Event):
    """A Hopf bifurcation of equilibria.

    Abbreviation: **H** -- ``jaxcont.bifurcations.taxonomy.describe("H")``.
    """

    _kind: str = "hopf"


@dataclass(frozen=True)
class EventHit:
    """A detected event along the branch."""

    kind: str
    p: float
    u: Array
    index: Optional[Tuple[int, int]] = None
    info: dict = field(default_factory=dict)
```

Replace it with a single comment noting these are imported from
`bifurcations.events` (Step 3 already added the import at the top of the
file):

```python
# ---------------------------------------------------------------------------
# Events -- Event, Fold, Hopf, EventHit are imported from
# jaxcont.bifurcations.events (see top of file) and re-exported here.
# ---------------------------------------------------------------------------
```

- [ ] **Step 5: Swap `_run_scan`'s detection call site**

In `src/jaxcont/api.py`, change (currently lines 438-460):

```python
    # Reuse the existing detector on the reassembled solution.
    if len(events) > 0 and eigenvalues is not None:
        from jaxcont.bifurcations.detector import BifurcationDetector

        requested = {e._kind for e in events if getattr(e, "_kind", "")}
        detector = BifurcationDetector(
            detect_fold="fold" in requested,
            detect_hopf="hopf" in requested,
            tolerance=1e-6,
        )
        sol.bifurcations = detector.detect_along_branch(
            sol,
            eigenvalues,
            refine_location=True,
            problem=_to_legacy_problem(problem),
            fold_extended_system=True,
        )

    result = _to_result(sol)
    requested = {e._kind for e in events if getattr(e, "_kind", "")}
    if requested:
        result.events = [h for h in result.events if h.kind in requested]
    return result
```

to:

```python
    # Detect events with the Event protocol (bifurcations/events.py).
    if len(events) > 0 and eigenvalues is not None:
        hits = detect_events(
            events, params, states, tangents, eigenvalues, rhs2,
            ds=float(settings.ds), tolerance=1e-6,
        )
        # sol.bifurcations stays dict-shaped: viz/core.py's plotting and
        # ContinuationSolution.get_bifurcations_by_type both read
        # bif.get("type")/bif.get("parameter")/bif.get("state") directly.
        sol.bifurcations = [
            {"type": h.kind, "parameter": h.p, "state": h.u, "index": h.index, **h.info}
            for h in hits
        ]

    return _to_result(sol)
```

(`_to_result` is unchanged — it already converts `sol.bifurcations` dicts
back into `EventHit`s for `ContinuationResult.events`, so no further
kind-filtering step is needed: `detect_events` only ever processes the
exact `events` the caller passed in, so there's nothing left over to
filter.)

- [ ] **Step 6: Update `tests/test_taxonomy.py`'s `_kind` references**

`Event`/`Fold`/`Hopf` now come from `bifurcations/events.py`, where the
attribute is public `kind` (not `_kind`). `tests/test_taxonomy.py` is the
one place outside `api.py` itself that reads this attribute directly (found
by running the full suite against this task's change during design — it
doesn't use `events=`, so it wasn't caught by the earlier repo-wide
`events=` usage sweep). In `tests/test_taxonomy.py`, change (currently
lines 49-50):

```python
    assert Fold()._kind == "fold"
    assert Hopf()._kind == "hopf"
```

to:

```python
    assert Fold().kind == "fold"
    assert Hopf().kind == "hopf"
```

- [ ] **Step 7: Run the regression baseline again**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_functional_api.py tests/test_bifurcation_workflow.py tests/test_taxonomy.py -v`
Expected: PASS (all tests, same as Step 1's baseline plus `test_taxonomy.py`
green again) — in particular `TestFolds::test_scan_passes_and_detects_fold`
(asserts `folds[0].info["method"] == "extended_system"`),
`TestHopf::test_scan_detects_and_refines_hopf` (asserts
`hopf[0].info["method"] == "bisection"`), and
`TestVmapSafety::test_vmap_with_events_raises_clearly` (the `NotImplementedError`
under `vmap` must still fire) all continue passing unmodified.

- [ ] **Step 8: Confirm the fix end-to-end on the real cross-validated examples**

Run: `MPLBACKEND=Agg JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python examples/example_05_neural_mass.py 2>&1 | grep -c "<->"`
Expected: `3` (down from Step 2's `5` — one row per real, distinct
bifurcation: `fold E0=-1.8652`, `hopf E0=-1.8500`, `fold E0=-1.4630`).

Run: `MPLBACKEND=Agg JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python examples/example_05_neural_mass.py 2>&1 | grep -c "no close match"`
Expected: `0` (no unmatched/spurious rows).

Run: `MPLBACKEND=Agg JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python examples/example_02_lorenz.py 2>&1 | grep -c "<->"`
Expected: `4` (down from `6` today, verified during design — today's F=1.5466
fold alone is reported 3 times; after the fix, one row per distinct
bifurcation: `fold F=1.5466`, and the three `hopf` rows).

Run: `MPLBACKEND=Agg JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python examples/example_02_lorenz.py 2>&1 | grep -c "no close match"`
Expected: `0`.

- [ ] **Step 9: Update the now-stale bug-description comments in both examples**

Both examples carry inline comments (written 2026-07-18) describing the
duplicate/spurious flags as an open, unresolved limitation. That's no
longer true after this task's fix, and left as-is it would actively
mislead a future reader. In `examples/example_05_neural_mass.py`, change
the comment block (currently around the `bk_reference` list):

```python
# Reference values from running BifurcationKit.jl v0.5.2 (``PALC()``,
# ``bothside=true``) independently, offline, on the identical equations and
# parameters. The two solid fold matches below agree to within 0.0015 in E0.
# The Hopf near E0=-1.85 is also found, but -- as with the Lorenz-84 example
# -- the detector additionally flags a nearby spurious "fold" at the same
# location, and one more spurious fold appears near E0=-1.55 with no
# BifurcationKit.jl counterpart at all. This is an honest snapshot of the
# current detector's precision, not a hidden failure: the *locations* it does
# match are accurate; duplicate/spurious flags near closely-spaced or
# lower-quality crossings are a known limitation to improve on.
```

to:

```python
# Reference values from running BifurcationKit.jl v0.5.2 (``PALC()``,
# ``bothside=true``) independently, offline, on the identical equations and
# parameters. All three reachable bifurcations match to within 0.0015 in E0
# (issue #7's duplicate/spurious fold-vs-Hopf flags -- fixed 2026-07-23, see
# docs/superpowers/plans/2026-07-23-event-protocol-rewrite.md).
```

In `examples/example_02_lorenz.py`, change (currently just above the
`bk_reference` list):

```python
# The reference values below come from running BifurcationKit.jl v0.5.2
# (``PALC()``, ``bothside=true``) on the *identical* right-hand side and
# parameters, independently, offline. JaxCont's detected fold/Hopf parameter
# values agree with BifurcationKit.jl's to within about one continuation step
# (:math:`\Delta F \lesssim 0.005`) -- close bifurcations occasionally produce
# an extra/duplicate flag (visible below as an unmatched "fold" near the first
# Hopf), which is a known precision limitation of the current detector, not a
# location error.
```

to:

```python
# The reference values below come from running BifurcationKit.jl v0.5.2
# (``PALC()``, ``bothside=true``) on the *identical* right-hand side and
# parameters, independently, offline. JaxCont's detected fold/Hopf parameter
# values agree with BifurcationKit.jl's to within about one continuation step
# (:math:`\Delta F \lesssim 0.005`) -- issue #7's duplicate/spurious flags are
# fixed as of 2026-07-23 (see
# docs/superpowers/plans/2026-07-23-event-protocol-rewrite.md).
```

- [ ] **Step 10: Run the wider test suite**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -q`
Expected: same pass count as the pre-Task-1 baseline plus Task 1's 8 new tests, 0 failures.

- [ ] **Step 11: Commit**

```bash
git add src/jaxcont/api.py tests/test_taxonomy.py examples/example_02_lorenz.py examples/example_05_neural_mass.py
git commit -m "feat: wire detect_events into continuation(), fixing issue #7's duplicate/spurious flags end-to-end"
```

---

### Task 3: Delete the old detector/classes and update exports

**Files:**
- Delete: `src/jaxcont/bifurcations/detector.py`
- Delete: `src/jaxcont/bifurcations/fold.py`
- Delete: `src/jaxcont/bifurcations/hopf.py`
- Modify: `src/jaxcont/bifurcations/__init__.py`
- Modify: `src/jaxcont/__init__.py`

**Interfaces:**
- Consumes: nothing (this task only removes now-unreferenced code and updates package exports).
- Produces: nothing consumed by later tasks (this is the last task in the plan).

- [ ] **Step 1: Confirm nothing outside the deleted files still imports them**

Run: `rtk proxy grep -rn "BifurcationDetector\|FoldBifurcation\|HopfBifurcation" src/jaxcont/ tests/ examples/`
Expected: matches only inside `src/jaxcont/bifurcations/detector.py`,
`src/jaxcont/bifurcations/fold.py`, `src/jaxcont/bifurcations/hopf.py`
themselves, and their import/export lines in `src/jaxcont/bifurcations/__init__.py`
and `src/jaxcont/__init__.py` (both about to be edited in this task) — no
matches anywhere else. If any other match turns up, stop and report it
rather than deleting (something still depends on the old classes).

- [ ] **Step 2: Delete the three files**

```bash
git rm src/jaxcont/bifurcations/detector.py src/jaxcont/bifurcations/fold.py src/jaxcont/bifurcations/hopf.py
```

- [ ] **Step 3: Update `bifurcations/__init__.py`**

Replace the full contents of `src/jaxcont/bifurcations/__init__.py` with:

```python
"""Bifurcation detection and analysis."""

from jaxcont.bifurcations.events import BranchPoint, Event, Fold, Hopf, EventHit, detect_events
from jaxcont.bifurcations.period_doubling import PeriodDoublingBifurcation
from jaxcont.bifurcations.taxonomy import LABELS, BIFURCATION_TYPES, BifurcationLabel, describe

__all__ = [
    "BranchPoint",
    "Event",
    "Fold",
    "Hopf",
    "EventHit",
    "detect_events",
    "PeriodDoublingBifurcation",
    "LABELS",
    "BIFURCATION_TYPES",
    "BifurcationLabel",
    "describe",
]
```

- [ ] **Step 4: Update `jaxcont/__init__.py`**

In `src/jaxcont/__init__.py`, remove the now-dead import block (currently):

```python
# Bifurcation detection
from jaxcont.bifurcations.detector import BifurcationDetector
from jaxcont.bifurcations.fold import FoldBifurcation
from jaxcont.bifurcations.hopf import HopfBifurcation
```

(delete these three lines and the `# Bifurcation detection` comment entirely
— `Event`/`Fold`/`Hopf`/`EventHit` are already imported from `jaxcont.api`
at the top of this file, unchanged by this task).

Then remove the corresponding `__all__` entries (currently):

```python
    # Bifurcations
    "BifurcationDetector",
    "FoldBifurcation",
    "HopfBifurcation",
```

(delete these three lines and the `# Bifurcations` comment).

- [ ] **Step 5: Verify imports still resolve**

Run: `/home/ziaee/envs/jaxcont/bin/python -c "import jaxcont as jc; print(jc.Fold(), jc.Hopf())"`
Expected: prints `Fold(kind='fold') Hopf(kind='hopf', tolerance=1e-06)`, no
`ImportError`/`AttributeError`.

Run: `/home/ziaee/envs/jaxcont/bin/python -c "import jaxcont as jc; assert not hasattr(jc, 'BifurcationDetector')"`
Expected: no `AssertionError`.

- [ ] **Step 6: Final full-suite regression**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -q`
Expected: same pass count as Task 2 Step 10, 0 failures — deleting the old
files changes no runtime behavior (nothing references them anymore, per
Step 1's check).

- [ ] **Step 7: Commit**

```bash
git add -u src/jaxcont/bifurcations/__init__.py src/jaxcont/__init__.py
git commit -m "refactor: delete BifurcationDetector/FoldBifurcation/HopfBifurcation, superseded by the Event protocol"
```

---

## Final verification

- [ ] Run the full suite one more time: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -q` — expect all green.
- [ ] Re-run both cross-validated examples one more time and eyeball the printed comparison tables in full (not just the "no close match" count):
  `MPLBACKEND=Agg JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python examples/example_02_lorenz.py` and
  `MPLBACKEND=Agg JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python examples/example_05_neural_mass.py` —
  confirm every printed fold/Hopf location is still within ~0.01 of its `bk_reference` match (no location regressions, not just fewer spurious rows).
- [ ] Confirm `jax.vmap`/`jax.jit` + `events=[...]` still raises the documented `NotImplementedError` (unchanged): `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_functional_api.py -k test_vmap_with_events_raises_clearly -v`.
