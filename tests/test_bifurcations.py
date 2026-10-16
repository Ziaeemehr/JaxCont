"""
Tests for the Event protocol (jaxcont.bifurcations.events): Fold's test
function, the tracked-mode scans of Hopf/BP/PD/NS, refinement and its
spectral re-checks, and the detect_events orchestrator's dedup logic.
Replaces the old direct tests of FoldBifurcation/HopfBifurcation (deleted
along with BifurcationDetector -- see
docs/superpowers/specs/2026-07-23-event-protocol-rewrite-design.md).
"""

from dataclasses import dataclass, field, replace

import jax.numpy as jnp
import numpy as np
import pytest
from jax import jacfwd

import jaxcont as jc
from jaxcont.bifurcations.events import BP, BranchPoint as EventPoint
from jaxcont.bifurcations.events import (
    BranchPoint,
    EventHit,
    Fold,
    Hopf,
    NeimarkSacker,
    PeriodDoubling,
    detect_events,
    failed_hit,
)


def _pitchfork(u, p, args):
    return jnp.array([p * u[0] - u[0] ** 3])


def _saddle_node(u, p, args):
    # f = u^2 + p: an ordinary fold at (u, p) = (0, 0), NOT a branch point.
    return jnp.array([u[0] ** 2 + p])


def test_bp_fires_on_pitchfork_but_not_on_a_plain_fold():
    prob = jc.bif_problem(_pitchfork, u0=jnp.array([0.0]), p0=-0.5)
    sol = jc.continuation(
        prob, jc.PseudoArclength(), p_span=(-0.5, 0.5),
        settings=jc.ContinuationPar(ds=0.02, max_steps=60, newton_tol=1e-6),
        events=[BP()],
    )
    bp_hits = [e for e in sol.events if e.kind == "branch_point"]
    assert len(bp_hits) == 1
    assert bp_hits[0].p == pytest.approx(0.0, abs=1e-3)
    assert bp_hits[0].u[0] == pytest.approx(0.0, abs=1e-3)
    assert "tangent" in bp_hits[0].info

    # Negative control: a plain fold must NOT be classified as a branch
    # point, even though it gives the same real-eigenvalue-crossing signal.
    fold_prob = jc.bif_problem(_saddle_node, u0=jnp.array([1.0]), p0=-1.0)
    fold_sol = jc.continuation(
        fold_prob, jc.PseudoArclength(), p_span=(-1.0, 1.0),
        settings=jc.ContinuationPar(ds=0.05, max_steps=60, newton_tol=1e-6),
        events=[BP()],
    )
    assert [e for e in fold_sol.events if e.kind == "branch_point"] == []


def test_refine_returning_none_is_skipped_by_detect_events():
    # Direct unit test of the widened protocol, independent of continuation().
    left = EventPoint(p=-0.1, u=jnp.array([0.1]), tangent=jnp.array([1.0, 0.0]))
    right = EventPoint(p=0.1, u=jnp.array([-0.1]), tangent=jnp.array([1.0, 0.0]))
    hit = BP().refine(
        left, right, (0, 1), lambda u, p: _saddle_node(u, p, None),
        tolerance=1e-6, max_iterations=50,
    )
    assert hit is None


def test_fold_test_function_is_tangent_dp_component():
    # tangent = (du_1, ..., du_n, dp); last entry is the dp component.
    fold = Fold()
    point = BranchPoint(p=0.0, u=jnp.zeros(1), tangent=jnp.array([0.5, 0.02]))
    assert jnp.isclose(fold.test_function(point), 0.02, atol=1e-6)


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


def test_hopf_refine_converges_via_extended_system():
    # This rhs is purely linear in u, so its 2nd/3rd derivatives are zero
    # everywhere -- l1 must come out exactly 0 (degenerate), giving this
    # test double duty as the "degenerate" edge case from the design spec.
    def rhs(u, p):
        x, y = u[0], u[1]
        return jnp.array([p * x - 0.1 * y, 0.1 * x + p * y])

    def eigs_at(u, p):
        jac = jacfwd(lambda u_: rhs(u_, p))(u)
        return jnp.linalg.eigvals(jac)

    hopf = Hopf()
    left = BranchPoint(p=-0.05, u=jnp.zeros(2), eigenvalues=eigs_at(jnp.zeros(2), -0.05))
    right = BranchPoint(p=0.05, u=jnp.zeros(2), eigenvalues=eigs_at(jnp.zeros(2), 0.05))
    # tolerance=1e-6, not 1e-8: the Hopf extended system's achievable float32
    # residual floor is ~6e-8 (eigenvector-normalization roundoff), the same
    # phenomenon codim2.py's cusp_point/generalized_hopf_point already
    # document with their own tol=1e-6 defaults. Since Task 2 wired
    # hopf_point's own Newton-convergence check into Hopf.refine's pass/fail
    # gate, 1e-8 here would make even an exact solve report converged=False.
    hit = hopf.refine(left, right, (3, 4), rhs, tolerance=1e-6, max_iterations=50)
    assert hit.kind == "hopf"
    assert abs(hit.p) < 1e-6
    assert hit.info["method"] == "extended_system"
    assert jnp.isclose(hit.info["omega0"], 0.1, atol=1e-4)
    assert jnp.isclose(hit.info["l1"], 0.0, atol=1e-4)
    assert hit.info["criticality"] == "degenerate"


def test_hopf_refine_marks_non_finite_result_as_unknown(monkeypatch):
    # Regression for the "silent subcritical mislabel" finding: hopf_point's
    # Newton solve (via differentiable_root) has no convergence guarantee --
    # if a bracket's test-function sign change wasn't actually a real Hopf
    # point (a known occurrence, see the "no close match -- spurious"
    # branches printed by examples/example_05_neural_mass.py), p/l1/omega0
    # can come back non-finite (e.g. p=-inf, l1=nan). Both `abs(nan) < tol`
    # and `nan < 0` are False in Python/NumPy/JAX, so without a finiteness
    # guard, refine()'s sign-based classifier would silently fall through to
    # the "subcritical" else-branch for a result that isn't a Hopf point at
    # all. Directly control hopf_point/lyapunov_coefficient's return values
    # (rather than trying to engineer a genuinely non-convergent bracket, an
    # awkward and less deterministic route) to exercise the real refine()
    # code path with a controlled non-finite result.
    import jaxcont.bifurcations.events as events_mod

    def fake_hopf_point(f, u_guess, p_guess, args=None, **kwargs):
        return (
            jnp.zeros(2), jnp.array(-jnp.inf),
            jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]), jnp.array(1.0),
            jnp.array(False),
        )

    def fake_lyapunov_coefficient(f, u, p, q1, q2, omega0, args=None):
        return jnp.array(jnp.nan)

    monkeypatch.setattr(events_mod, "hopf_point", fake_hopf_point)
    monkeypatch.setattr(events_mod, "lyapunov_coefficient", fake_lyapunov_coefficient)

    def rhs(u, p):
        return u

    hopf = Hopf()
    left = BranchPoint(p=-0.05, u=jnp.zeros(2))
    right = BranchPoint(p=0.05, u=jnp.zeros(2))
    hit = hopf.refine(left, right, (0, 1), rhs, tolerance=1e-8, max_iterations=50)
    assert hit.status == "failed"
    assert hit.info["reason"] == "not_converged"


def test_hopf_refine_zero_omega_marks_unknown(monkeypatch):
    # omega0 <= 0 is also not a genuine Hopf point (a real Hopf point always
    # has a nonzero critical frequency); it must not be trusted even if it
    # happens to be finite.
    import jaxcont.bifurcations.events as events_mod

    def fake_hopf_point(f, u_guess, p_guess, args=None, **kwargs):
        return (
            jnp.zeros(2), jnp.array(0.02),
            jnp.array([1.0, 0.0]), jnp.array([0.0, 0.0]), jnp.array(0.0),
            jnp.array(True),
        )

    def fake_lyapunov_coefficient(f, u, p, q1, q2, omega0, args=None):
        return jnp.array(0.3)

    monkeypatch.setattr(events_mod, "hopf_point", fake_hopf_point)
    monkeypatch.setattr(events_mod, "lyapunov_coefficient", fake_lyapunov_coefficient)

    def rhs(u, p):
        return u

    hopf = Hopf()
    left = BranchPoint(p=-0.05, u=jnp.zeros(2))
    right = BranchPoint(p=0.05, u=jnp.zeros(2))
    hit = hopf.refine(left, right, (0, 1), rhs, tolerance=1e-8, max_iterations=50)
    assert hit.status == "failed"
    assert hit.info["reason"] == "spectral_check_failed"


def test_hopf_refine_supercritical_label_end_to_end():
    # The feature's headline output, end-to-end: the textbook example
    # (l1=-2 exactly, see test_hopf_normal_form.py's
    # test_lyapunov_coefficient_matches_exact_textbook_value) must produce
    # criticality == "supercritical" through Hopf().refine(), not just a
    # correctly-signed l1 float that nothing asserts the label of.
    def rhs(u, p):
        x, y = u[0], u[1]
        r2 = x**2 + y**2
        return jnp.array([-y + x * (p - r2), x + y * (p - r2)])

    def eigs_at(u, p):
        jac = jacfwd(lambda u_: rhs(u_, p))(u)
        return jnp.linalg.eigvals(jac)

    hopf = Hopf()
    left = BranchPoint(p=-0.05, u=jnp.zeros(2), eigenvalues=eigs_at(jnp.zeros(2), -0.05))
    right = BranchPoint(p=0.05, u=jnp.zeros(2), eigenvalues=eigs_at(jnp.zeros(2), 0.05))
    # tolerance=1e-6, not 1e-10 -- see test_hopf_refine_converges_via_extended_
    # system's comment: the achievable float32 residual floor for this
    # extended system is ~6e-8, and Task 2 made Hopf.refine gate on actually
    # reaching `tolerance`, not just finiteness.
    hit = hopf.refine(left, right, (0, 1), rhs, tolerance=1e-6, max_iterations=50)
    assert jnp.isclose(hit.info["l1"], -2.0, atol=1e-4)
    assert hit.info["criticality"] == "supercritical"


def test_hopf_refine_subcritical_label_end_to_end():
    # Same family with the cubic term negated (k=-1 in the existing scaling
    # test's parametrization in test_hopf_normal_form.py) has l1=+2 exactly,
    # so refine() must label it "subcritical".
    def rhs(u, p):
        x, y = u[0], u[1]
        r2 = x**2 + y**2
        return jnp.array([-y + x * (p + r2), x + y * (p + r2)])

    def eigs_at(u, p):
        jac = jacfwd(lambda u_: rhs(u_, p))(u)
        return jnp.linalg.eigvals(jac)

    hopf = Hopf()
    left = BranchPoint(p=-0.05, u=jnp.zeros(2), eigenvalues=eigs_at(jnp.zeros(2), -0.05))
    right = BranchPoint(p=0.05, u=jnp.zeros(2), eigenvalues=eigs_at(jnp.zeros(2), 0.05))
    # tolerance=1e-6 -- see test_hopf_refine_converges_via_extended_system's
    # comment (float32 residual floor).
    hit = hopf.refine(left, right, (0, 1), rhs, tolerance=1e-6, max_iterations=50)
    assert jnp.isclose(hit.info["l1"], 2.0, atol=1e-4)
    assert hit.info["criticality"] == "subcritical"


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

    # Two independent detections refine to the same fold; a different kind
    # nearby must survive even when the states also agree.
    near_fold_a = _ThresholdEvent(kind="fold", threshold=0.10, hit_p=0.10)
    near_fold_b = _ThresholdEvent(kind="fold", threshold=0.11, hit_p=0.10)
    near_hopf = _ThresholdEvent(kind="hopf", threshold=0.12, hit_p=0.12)
    far_hopf = _ThresholdEvent(kind="hopf", threshold=0.50, hit_p=0.50)

    hits = detect_events(
        [near_fold_a, near_fold_b, near_hopf, far_hopf],
        p_grid, states, None, None, rhs, ds=0.05,
    )

    assert [(h.kind, h.p) for h in hits] == [("fold", 0.10), ("hopf", 0.12), ("hopf", 0.50)]


def test_fold_refine_reports_not_converged_when_no_fold_exists(monkeypatch):
    # Regression for finding #2: a bracket sign-change is not a convergence
    # guarantee. Drive refine() with a fake fold_point that reports
    # converged=False and confirm the EventHit says so instead of silently
    # publishing an unchecked (u_bif, p_bif).
    import jaxcont.bifurcations.events as events_mod

    def fake_fold_point(f, u_guess, p_guess, args=None, **kwargs):
        return jnp.zeros(1), jnp.array(jnp.inf), jnp.zeros(1), jnp.array(False)

    monkeypatch.setattr(events_mod, "fold_point", fake_fold_point)

    def rhs(u, p):
        return u + p

    fold = Fold()
    left = BranchPoint(p=-0.05, u=jnp.zeros(1))
    right = BranchPoint(p=0.05, u=jnp.zeros(1))
    hit = fold.refine(left, right, (0, 1), rhs, tolerance=1e-8, max_iterations=50)
    assert hit.status == "failed"
    assert hit.info["reason"] == "not_converged"
    assert jnp.isnan(hit.p)


def test_hopf_refine_reports_converged_true_on_a_real_hopf_point():
    # Companion to the two "unknown" monkeypatched tests: confirm a genuine
    # convergent solve now also carries converged=True in info.
    def rhs(u, p):
        x, y = u[0], u[1]
        return jnp.array([p * x - 0.1 * y, 0.1 * x + p * y])

    def eigs_at(u, p):
        jac = jacfwd(lambda u_: rhs(u_, p))(u)
        return jnp.linalg.eigvals(jac)

    hopf = Hopf()
    left = BranchPoint(p=-0.05, u=jnp.zeros(2), eigenvalues=eigs_at(jnp.zeros(2), -0.05))
    right = BranchPoint(p=0.05, u=jnp.zeros(2), eigenvalues=eigs_at(jnp.zeros(2), 0.05))
    # tolerance=1e-6 -- see test_hopf_refine_converges_via_extended_system's
    # comment (float32 residual floor); at 1e-8 even this exact solve would
    # report converged=False, defeating the point of this test.
    hit = hopf.refine(left, right, (3, 4), rhs, tolerance=1e-6, max_iterations=50)
    assert hit.info["converged"] is True


def test_failed_refine_not_in_events(monkeypatch):
    import jaxcont.bifurcations.events as events_mod

    def fake_fold_point(f, u, p, **kwargs):
        return u, p, None, jnp.array(False)

    monkeypatch.setattr(events_mod, "fold_point", fake_fold_point)
    prob = jc.bif_problem(_saddle_node, u0=jnp.array([1.0]), p0=-1.0)
    result = jc.continuation(
        prob, jc.PseudoArclength(), p_span=(-1.0, 1.0),
        settings=jc.ContinuationPar(ds=0.05, max_steps=60, newton_tol=1e-6),
        events=[Fold()],
    )
    assert result.events == []
    assert len(result.rejected_events) >= 1
    rejected = result.rejected_events[0]
    assert rejected.status == "failed"
    assert rejected.info["reason"] == "not_converged"


def _fold_scan_inputs(left_u, left_p, right_u, right_p):
    params = jnp.array([left_p, right_p])
    states = jnp.array([[left_u], [right_u]])
    tangents = jnp.array([[0.0, 1.0], [0.0, -1.0]])  # dp component flips sign
    return params, states, tangents


def test_refine_outside_bracket_rejected(monkeypatch):
    # A converged refinement far from the bracket must be rejected by the
    # locality rule, not merely fail to converge.
    import jaxcont.bifurcations.events as events_mod

    def far_fold_point(f, u, p, **kwargs):
        return jnp.array([5.0]), 5.0, jnp.ones(1), jnp.array(True)

    monkeypatch.setattr(events_mod, "fold_point", far_fold_point)
    params, states, tangents = _fold_scan_inputs(0.1, -0.1, -0.1, 0.1)
    hits = detect_events(
        [Fold()], params, states, tangents, None,
        lambda u, p: _saddle_node(u, p, None), ds=0.2,
    )
    assert [h for h in hits if h.status == "validated"] == []
    rejected = [h for h in hits if h.status == "failed"]
    assert len(rejected) == 1
    assert rejected[0].info["reason"] == "outside_bracket"
    assert rejected[0].info["refined_p"] == pytest.approx(5.0)


def test_fold_outside_bracket_p_range_still_validates():
    # f = u**2 + p turns at p = 0, which lies beyond both bracket parameters
    # (both are < 0); the joint (u, p) rule must still accept it.
    params, states, tangents = _fold_scan_inputs(0.01, -1e-4, -0.01, -1e-4)
    hits = detect_events(
        [Fold()], params, states, tangents, None,
        lambda u, p: _saddle_node(u, p, None), ds=0.2,
    )
    validated = [h for h in hits if h.status == "validated"]
    assert len(validated) == 1
    assert validated[0].p == pytest.approx(0.0, abs=1e-3)


def test_within_bracket_p_interval_fallback_and_zero_width():
    from jaxcont.bifurcations.events import _within_bracket

    # Extended-system branch state (dim 3) vs hit state (dim 1): p-interval rule.
    left = BranchPoint(p=0.0, u=jnp.zeros(3))
    right = BranchPoint(p=1.0, u=jnp.zeros(3))
    hit = lambda p: EventHit(kind="x", p=p, u=jnp.zeros(1))
    assert _within_bracket(hit(0.5), left, right)
    assert _within_bracket(hit(1.9), left, right)
    assert not _within_bracket(hit(2.5), left, right)
    # Zero-width bracket accepts a hit sitting on it.
    pt = BranchPoint(p=0.3, u=jnp.zeros(1))
    assert _within_bracket(EventHit(kind="x", p=0.3, u=jnp.zeros(1)), pt, pt)
    assert not _within_bracket(EventHit(kind="x", p=0.9, u=jnp.zeros(1)), pt, pt)


def _two_pair_problem(re_a, re_b, p0):
    """Linear 4-D system with two complex pairs, ``re_a(p) +- 1i`` and
    ``re_b(p) +- 2i``; its equilibrium is ``u = 0`` for every ``p``."""
    def rhs(u, p, args):
        a, b = re_a(p), re_b(p)
        return jnp.array([
            a * u[0] - u[1], u[0] + a * u[1],
            b * u[2] - 2.0 * u[3], 2.0 * u[2] + b * u[3],
        ])
    return jc.bif_problem(rhs, u0=jnp.zeros(4), p0=p0)


def _hopf_sweep(prob, p_span):
    return jc.continuation(
        prob, jc.PseudoArclength(), p_span=p_span,
        settings=jc.ContinuationPar(ds=0.02, ds_max=0.05, max_steps=200),
        events=[Hopf()],
    )


def test_hopf_no_event_when_pairs_swap_order():
    # Neither pair crosses the axis (Re in [0.1, 0.3] and [-0.3, -0.1]), but
    # the pair nearest the axis changes at p = 0, where a nearest-to-axis
    # selection jumps from -0.2 to +0.2.
    prob = _two_pair_problem(lambda p: 0.2 + 0.1 * p, lambda p: -0.2 + 0.1 * p, -1.0)
    result = _hopf_sweep(prob, (-1.0, 1.0))
    assert result.events == []
    assert result.rejected_events == []


def test_hopf_spurious_case_from_review():
    # Review H1: a fixed pair at Re = -0.1 and a moving pair at Re = p. The
    # only Hopf point is p = 0; the selection swap at p = 0.1 must not
    # produce a second (spurious) bracket.
    prob = _two_pair_problem(lambda p: p, lambda p: -0.1 + 0.0 * p, -0.5)
    result = _hopf_sweep(prob, (-0.5, 0.5))
    hopf = [e for e in result.events if e.kind == "hopf"]
    assert len(hopf) == 1
    assert abs(hopf[0].p) < 1e-3
    assert result.rejected_events == []


@dataclass(frozen=True)
class _StubPeriodDoubling(PeriodDoubling):
    """PeriodDoubling whose refine reports the bracket midpoint unchecked, so
    a test can see which brackets the detection stage hands to refine."""

    def refine(self, left, right, index, rhs, **kwargs):
        return EventHit(kind=self.kind, p=(left.p + right.p) / 2, u=left.u, index=index)


def test_pd_not_reported_for_complex_pair_crossing_re_minus_one():
    # Two real multipliers (-0.9, -0.7) collide at -0.8 and leave the real
    # axis as a complex pair whose real part then passes -1. That is not a
    # period doubling: no real multiplier reaches -1.
    t = np.linspace(0.0, 1.0, 21)
    rows = []
    for ti in t:
        if ti <= 0.3:
            d = 0.1 * (1.0 - ti / 0.3)
            pair = [-0.8 - d, -0.8 + d]
        else:
            re, im = -0.8 - 0.6 * (ti - 0.3), 0.5 * (ti - 0.3)
            pair = [re + 1j * im, re - 1j * im]
        rows.append([1.0, 1e-3, *pair])
    multipliers = jnp.asarray(np.array(rows, dtype=complex))
    hits = detect_events(
        [_StubPeriodDoubling()],
        jnp.asarray(t), jnp.zeros((t.size, 1)), None, multipliers,
        lambda u, p: u, ds=0.05, trivial_index=jnp.zeros(t.size, dtype=int),
    )
    assert hits == []


def _points(spectra):
    return [
        BranchPoint(p=float(i), u=jnp.zeros(1), eigenvalues=jnp.asarray(row))
        for i, row in enumerate(np.asarray(spectra, dtype=complex))
    ]


def test_hopf_scan_brackets_one_pair_crossing():
    re = np.linspace(-0.1, 0.1, 6)
    spectra = np.stack([re + 2j, re - 2j, -1.0 + 0 * re], axis=1)
    brackets = Hopf().scan(_points(spectra))
    assert [(i, j) for i, j, _ in brackets] == [(2, 3)]
    mode = brackets[0][2]
    assert mode.left[mode.column].imag > 0


def test_hopf_scan_ignores_real_eigenvalue_crossing():
    re = np.linspace(-0.1, 0.1, 6)
    spectra = np.stack([-0.5 + 2j + 0 * re, -0.5 - 2j + 0 * re, re], axis=1)
    assert Hopf().scan(_points(spectra)) == []
    assert len(BP().scan(_points(spectra))) == 1


def test_hopf_scan_without_eigenvalues_is_empty():
    assert Hopf().scan([BranchPoint(p=0.0, u=jnp.zeros(1))]) == []


def test_period_doubling_scan_follows_the_crossing_multiplier_far_past_minus_one():
    # The moving real multiplier passes -1 once and keeps going to -20; the
    # decaying multiplier and the trivial one never open a bracket.
    moving = -np.geomspace(0.5, 20.0, 12)
    spectra = np.stack([1.0 + 0 * moving, 3.4e-6 + 0 * moving, moving], axis=1)
    pd = PeriodDoubling()
    brackets = pd.scan([replace(pt, trivial_index=0) for pt in _points(spectra)])
    assert len(brackets) == 1
    i, j, mode = brackets[0]
    assert moving[i] > -1.0 > moving[j]


def test_period_doubling_scan_excludes_trivial_multiplier():
    # Only the trivial multiplier moves (and sits nearest 1 throughout).
    spectra = np.stack([np.linspace(0.99, 1.01, 5), -0.5 + 0 * np.ones(5)], axis=1)
    pd = PeriodDoubling()
    assert pd.scan([replace(pt, trivial_index=0) for pt in _points(spectra)]) == []


def test_neimark_sacker_scan_brackets_complex_pair_leaving_unit_circle():
    radius = np.linspace(0.7, 1.3, 6)
    pair = radius * np.exp(0.6j)
    spectra = np.stack([1.0 + 0 * radius, pair, np.conj(pair), -0.5 + 0 * radius], axis=1)
    ns = NeimarkSacker()
    brackets = ns.scan([replace(pt, trivial_index=0) for pt in _points(spectra)])
    assert [(i, j) for i, j, _ in brackets] == [(2, 3)]


def test_neimark_sacker_scan_ignores_real_multiplier_crossing():
    moving = np.linspace(-0.8, -1.2, 5)
    spectra = np.stack([1.0 + 0 * moving, moving, 0.5 + 0.5j + 0 * moving,
                        0.5 - 0.5j + 0 * moving], axis=1)
    ns = NeimarkSacker()
    assert ns.scan([replace(pt, trivial_index=0) for pt in _points(spectra)]) == []


def test_hopf_no_event_on_a_center_family():
    # Re(lambda) is exactly 0 along the whole branch (a center, +-i*sqrt(1+p^2)):
    # no mode crosses the axis, so neither Hopf nor BP may open a bracket.
    def rhs(u, p, args):
        return jnp.array([-(1.0 + p**2) * u[1], u[0], -u[2] + p])

    prob = jc.bif_problem(rhs, u0=jnp.array([0.0, 0.0, -0.5]), p0=-0.5)
    result = jc.continuation(
        prob, jc.PseudoArclength(), p_span=(-0.5, 0.5),
        settings=jc.ContinuationPar(ds=0.02, ds_max=0.05, max_steps=100),
        events=[Hopf(), BP()],
    )
    assert result.events == []
    assert result.rejected_events == []


def test_hopf_refine_spectral_check_rejects_real_spectrum(monkeypatch):
    # A converged extended-system solve with omega0 > 0, but the spectrum at
    # the refined point is real: the eigenvalue nearest i*omega0 is real.
    import jaxcont.bifurcations.events as events_mod

    def fake_hopf_point(f, u_guess, p_guess, args=None, **kwargs):
        return (
            jnp.zeros(2), jnp.array(0.0),
            jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]), jnp.array(1.0),
            jnp.array(True),
        )

    monkeypatch.setattr(events_mod, "hopf_point", fake_hopf_point)
    monkeypatch.setattr(
        events_mod, "lyapunov_coefficient", lambda *a, **k: jnp.array(-1.0)
    )

    def rhs(u, p):
        return jnp.array([0.5 * u[0], -0.5 * u[1]])

    left = BranchPoint(p=-0.05, u=jnp.zeros(2))
    right = BranchPoint(p=0.05, u=jnp.zeros(2))
    hit = Hopf().refine(left, right, (0, 1), rhs, tolerance=None, max_iterations=20)
    assert hit.status == "failed"
    assert hit.info["reason"] == "spectral_check_failed"
    assert "eigenvalue" in hit.info


def test_bp_refine_spectral_check_rejects_complex_spectrum(monkeypatch):
    # A converged fold solve whose spectrum nearest 0 is a complex pair.
    import jaxcont.bifurcations.events as events_mod

    def fake_fold_point(f, u, p, **kwargs):
        return jnp.zeros(2), jnp.array(0.0), jnp.ones(2), jnp.array(True)

    monkeypatch.setattr(events_mod, "fold_point", fake_fold_point)

    def rhs(u, p):
        return jnp.array([-u[1], u[0]])

    left = BranchPoint(p=-0.05, u=jnp.zeros(2))
    right = BranchPoint(p=0.05, u=jnp.zeros(2))
    hit = BP().refine(left, right, (0, 1), rhs, tolerance=None, max_iterations=20)
    assert hit.status == "failed"
    assert hit.info["reason"] == "spectral_check_failed"


@pytest.mark.parametrize(
    "event_cls, row, final",
    [
        # PD: the multiplier nearest -1 at the refined point is complex.
        (PeriodDoubling, [1.0, -1.0, 0.3], [1.0, -1.0 + 0.5j, -1.0 - 0.5j]),
        # NS: the multiplier nearest the tracked mode's circle point is real.
        (NeimarkSacker, [1.0, np.exp(0.6j), np.exp(-0.6j)], [1.0, 0.95, 0.3]),
    ],
)
def test_floquet_refine_spectral_check_failed(monkeypatch, event_cls, row, final):
    import jaxcont.bifurcations.events as events_mod
    from jaxcont.bifurcations.modes import TrackedMode

    monkeypatch.setattr(
        events_mod, "differentiable_root_checked",
        lambda rhs, u, p, tol=None: (u, jnp.array(True)),
    )
    monkeypatch.setattr(
        events_mod, "floquet_multipliers",
        lambda raw_f, mesh, u, p, args=None: (jnp.asarray(np.array(final, dtype=complex)), jnp.array(0)),
    )
    row = np.array(row, dtype=complex)
    mode = TrackedMode(1, row, row)
    # A zero-width bracket skips bisection and goes straight to the re-check.
    point = BranchPoint(p=0.0, u=jnp.zeros(1))
    event = event_cls()
    from jaxcont.problems.periodic_args import PeriodicArgs
    stub_args = PeriodicArgs(jnp.zeros((1, 1, 1)), jnp.zeros((1, 1, 1)), None, None, None)
    hit = event.refine(
        point, point, (0, 1), lambda u, p: u,
        tolerance=None, max_iterations=20, mode=mode, args=stub_args,
    )
    assert hit.status == "failed"
    assert hit.info["reason"] == "spectral_check_failed"
    assert "multiplier" in hit.info


# --- exact zeros, sheet-aware dedup, batched test values -------------------


@dataclass
class _CountingEvent:
    """Test-only Event with a test function ``p`` that records refine calls."""

    kind: str = "marker"
    calls: list = field(default_factory=list)
    fail: bool = False

    def test_function(self, point: BranchPoint):
        return point.p

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations):
        self.calls.append(index)
        if self.fail:
            return failed_hit(self.kind, index, "not_converged")
        return EventHit(kind=self.kind, p=0.0, u=left.u, index=index)


def _grid_branch(ps):
    ps = jnp.asarray(ps)
    return ps, jnp.zeros((ps.shape[0], 1))


def test_exact_zero_reported_once():
    ps, states = _grid_branch([-0.2, -0.1, 0.0, 0.1, 0.2])
    event = _CountingEvent()
    hits = detect_events([event], ps, states, None, None, lambda u, p: u, ds=0.01)
    assert [h.kind for h in hits] == ["marker"]
    assert len(event.calls) == 1


def test_exact_zero_failed_refine_reported_once():
    ps, states = _grid_branch([-0.2, -0.1, 0.0, 0.1, 0.2])
    event = _CountingEvent(fail=True)
    hits = detect_events([event], ps, states, None, None, lambda u, p: u, ds=0.01)
    assert [h.status for h in hits] == ["failed"]
    assert len(event.calls) == 1


def test_exact_zero_at_branch_ends_reported_once():
    for ps in ([0.0, 0.1, 0.2], [-0.2, -0.1, 0.0]):
        event = _CountingEvent()
        hits = detect_events(
            [event], *_grid_branch(ps), None, None, lambda u, p: u, ds=0.01,
        )
        assert len(hits) == 1 and len(event.calls) == 1


def test_exact_zero_in_spectral_scan_reported_once():
    # A real pair whose real part is exactly zero at sample 2.
    re = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    spectra = np.stack([re + 1j, re - 1j], axis=1)
    events = Hopf().scan(
        [BranchPoint(p=float(i), u=jnp.zeros(1), eigenvalues=jnp.asarray(s))
         for i, s in enumerate(spectra)]
    )
    assert len(events) == 1


def _s_curve_branch(u1, u2, p0):
    """S-curve ``p = u^3/3 - u`` with an oscillatory block whose damping
    ``(u - u1)(u - u2)`` vanishes on the lower (u1) and upper (u2) sheets."""
    def damping(u):
        return (u - u1) * (u - u2)

    def rhs(z, p):
        u, x, y = z
        d = damping(u)
        return jnp.array([p + u - u**3 / 3, d * x - y, x + d * y])

    us = np.linspace(-2.2, 2.2, 177)
    ps = us**3 / 3 - us
    states = jnp.stack([jnp.asarray(us), jnp.zeros_like(jnp.asarray(us)),
                        jnp.zeros_like(jnp.asarray(us))], axis=1)
    eigs = jnp.stack([
        jnp.linalg.eigvals(jacfwd(rhs)(states[i], ps[i])) for i in range(len(us))
    ])
    return jnp.asarray(ps), states, eigs, rhs


def test_dedup_keeps_hits_on_different_sheets():
    p0 = 0.3
    roots = np.sort(np.roots([1 / 3, 0.0, -1.0, -p0]).real)
    u_lower, u_upper = roots[0], roots[-1]
    ps, states, eigs, rhs = _s_curve_branch(u_lower, u_upper, p0)
    hits = detect_events(
        [Hopf()], ps, states, None, eigs, rhs, ds=0.05,
    )
    hopfs = [h for h in hits if h.kind == "hopf" and h.status == "validated"]
    assert len(hopfs) == 2
    assert abs(hopfs[0].p - hopfs[1].p) < 1e-3
    assert abs(float(hopfs[0].u[0]) - float(hopfs[1].u[0])) > 1.0


def test_dedup_preserves_distinct_roots_independent_of_step_size():
    ps, states = _grid_branch([0.0, 0.15, 0.3, 0.6])
    a = _ThresholdEvent(kind="fold", threshold=0.10, hit_p=0.10)
    b = _ThresholdEvent(kind="fold", threshold=0.11, hit_p=0.11)
    ds_values = np.array([0.5, 0.5, 0.5, 0.5])
    hits = detect_events(
        [a, b], ps, states, None, None, lambda u, p: u,
        ds=0.001, ds_values=ds_values,
    )
    assert [h.p for h in hits] == [0.10, 0.11]
    # Step sizes do not define refinement accuracy.
    hits = detect_events([a, b], ps, states, None, None, lambda u, p: u, ds=0.001)
    assert [h.p for h in hits] == [0.10, 0.11]


def test_batched_test_values_match_per_point_and_fallback_works():
    ps, states = _grid_branch([-0.3, -0.1, 0.1, 0.3])

    class HostSide:
        kind = "host"

        def test_function(self, point):
            return float(np.asarray(point.p))  # raises under vmap

        def refine(self, left, right, index, rhs, *, tolerance, max_iterations):
            return EventHit(kind=self.kind, p=0.0, u=left.u, index=index)

    batched = _CountingEvent()
    for ev in (batched, HostSide()):
        hits = detect_events([ev], ps, states, None, None, lambda u, p: u, ds=0.01)
        assert [h.index for h in hits] == [(1, 2)]


def test_detect_events_refines_each_bracket_with_step_updated_args():
    # Bracket (i, i+1) is refined with step_update(args, u_i, p_i): the args
    # its right end was corrected with. Both needs_args events and the rhs
    # handed to refine see them.
    seen = []

    class Probe:
        kind = "probe"
        needs_args = True

        def test_function(self, point):
            return np.cos(3.0 * point.p)

        def refine(self, left, right, index, rhs, *, tolerance, max_iterations, args=None):
            seen.append((index, args, float(rhs(left.u, left.p)[0])))
            return None

    def f(u, p, args):
        return u - args["ref"]

    def step_update(args, u, p):
        return {"ref": u + p}

    params = jnp.linspace(0.0, 3.0, 7)
    states = params[:, None] * 2.0
    hits = detect_events(
        [Probe()], params, states, None, None, lambda u, p: f(u, p, {"ref": 0.0}),
        ds=0.5, args={"ref": jnp.array([0.0])}, step_update=step_update, f=f,
    )
    assert hits == []
    assert [index for index, _, _ in seen] == [(1, 2), (3, 4), (5, 6)]
    for (i, j), b_args, r in seen:
        np.testing.assert_allclose(b_args["ref"], states[i] + params[i])
        np.testing.assert_allclose(r, -float(params[i]))


def test_detect_events_step_update_requires_f():
    with pytest.raises(ValueError, match="requires the problem's f"):
        detect_events(
            [Fold()], jnp.zeros(2), jnp.zeros((2, 1)), None, None, lambda u, p: u,
            ds=0.1, step_update=lambda a, u, p: a,
        )


@pytest.mark.parametrize('event, spectra', [
    (PeriodDoubling(), np.array([[-0.9, 1.00001], [1.00001, -1.1]], complex)),
    (NeimarkSacker(tolerance=0.0), np.array([
        [0.9999 * np.exp(0.01j), 1.00001],
        [1.00001, 1.0001 * np.exp(0.01j)],
    ])),
])
def test_floquet_scan_excludes_flow_mode_after_reordering(event, spectra):
    points = [BranchPoint(p=float(i), u=jnp.zeros(2), eigenvalues=row,
                          trivial_index=i) for i, row in enumerate(spectra)]
    assert event.scan(points) == []


def test_floquet_scan_requires_flow_identity():
    points = [BranchPoint(p=0., u=jnp.zeros(2), eigenvalues=jnp.array([1., -.9]))]
    with pytest.raises(ValueError, match='trivial_index'):
        PeriodDoubling().scan(points)
