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
