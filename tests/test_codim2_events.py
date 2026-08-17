"""Codim-2 events detected along two-parameter curves."""

import jax.numpy as jnp

import jaxcont as jc
from jaxcont.bifurcations.codim2_events import Cusp
from jaxcont.bifurcations.curves import fold_curve_problem


def _cusp_shifted(u, p, args):
    """Same shifted cusp normal form as tests/test_curves.py. Its cusp point
    is exactly u=0.7, p=(0.3, 1.2) (where the discriminant's two fold
    branches meet)."""
    x = u[0] - 0.7
    a = p[0] - 0.3
    b = p[1] - 1.2
    return jnp.array([a + b * x - x**3])


def test_cusp_detected_on_the_fold_curve():
    # Trace the fold curve DOWN toward the cusp at p1 = 1.2. Start at
    # b = p1-1.2 = 3 (x = 1, a = -2), continue p1 from 4.2 down past 1.2.
    prob = fold_curve_problem(
        _cusp_shifted, jnp.array([1.7]), jnp.array([-1.7, 4.2]), free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(4.2, 0.9),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        events=[Cusp(raw_f=_cusp_shifted, free=1)],
    )
    hits = [h for h in sol.events if h.kind == "cusp"]
    assert len(hits) == 1
    assert abs(hits[0].p - 1.2) < 5e-2
    assert abs(float(hits[0].info["p_fixed"]) - 0.3) < 5e-2


from jaxcont.bifurcations.codim2_events import BogdanovTakens


def _bt_shifted(u, p, args):
    """Same system as tests/test_codim2.py: BT at u*=(5,2), p*=(3,-1).
    Exact fold curve p0 = 3 + (p1+1)**2/4; the non-trivial eigenvalue along
    it is exactly -(p1+1)/2, crossing zero at p1 = -1."""
    X, Y = u[0], u[1]
    x, y = X - 5.0, Y - 2.0
    b1 = p[0] - 3.0
    b2 = p[1] + 1.0
    return jnp.array([y, b1 + b2 * x + x**2 + x * y])


def _bt_fold_curve_seed():
    # p1 = -2 -> b2 = -1 -> x = -b2/2 = 0.5 -> X = 5.5, Y = 2
    # b1 = b2**2/4 = 0.25 -> p0 = 3.25
    return jnp.array([5.5, 2.0]), jnp.array([3.25, -2.0])


def _bt_fold_curve_solution(events):
    u_guess, p_guess = _bt_fold_curve_seed()
    prob = fold_curve_problem(_bt_shifted, u_guess, p_guess, free=1)
    return jc.continuation(
        prob,
        p_span=(-2.0, 0.0),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        events=events,
    )


def test_fold_curve_matches_the_exact_bt_parabola():
    sol = _bt_fold_curve_solution([])
    from jaxcont.bifurcations.curves import unpack_fold_curve
    for i in range(sol.branch.params.shape[0]):
        p1 = float(sol.branch.params[i])
        _, p0, _ = unpack_fold_curve(sol.branch.states[i], n=2)
        assert abs(float(p0) - (3.0 + (p1 + 1.0) ** 2 / 4.0)) < 1e-3


def test_bogdanov_takens_detected_on_the_fold_curve():
    sol = _bt_fold_curve_solution(
        [BogdanovTakens(raw_f=_bt_shifted, free=1, curve="fold")]
    )
    hits = [h for h in sol.events if h.kind == "bogdanov_takens"]
    assert len(hits) == 1
    assert hits[0].info["converged"] is True
    assert abs(hits[0].p - (-1.0)) < 5e-2
    assert jnp.allclose(hits[0].u, jnp.array([5.0, 2.0]), atol=5e-2)


def test_bt_test_function_ignores_the_pinned_zero_eigenvalue():
    """DISCRIMINATING POWER: one eigenvalue is identically zero along the
    whole fold curve. If the exclusion in _nontrivial_eigenvalues were
    removed, the test function would be identically ~0, never change sign,
    and detect nothing. This asserts it instead tracks -(p1+1)/2."""
    from jaxcont.bifurcations.events import BranchPoint
    sol = _bt_fold_curve_solution([])
    ev = BogdanovTakens(raw_f=_bt_shifted, free=1, curve="fold")
    saw_positive = saw_negative = False
    for i in range(sol.branch.params.shape[0]):
        p1 = float(sol.branch.params[i])
        val = ev.test_function(
            BranchPoint(p=p1, u=sol.branch.states[i])
        )
        assert abs(val - (-(p1 + 1.0) / 2.0)) < 5e-2
        saw_positive |= val > 0.05
        saw_negative |= val < -0.05
    assert saw_positive and saw_negative, "no sign change -> nothing to detect"


def test_bt_near_critical_filter_is_not_too_aggressive():
    """DISCRIMINATING POWER (the other direction): a pre-filter tight enough
    to reject the genuine candidate would silently drop the detection. The
    candidate reaches |-(p1+1)/2| = 0.5 at the ends of this span, so a
    filter narrower than that would zero out the detection."""
    sol = _bt_fold_curve_solution(
        [BogdanovTakens(raw_f=_bt_shifted, free=1, curve="fold",
                        near_critical=2.0)]
    )
    assert len([h for h in sol.events if h.kind == "bogdanov_takens"]) == 1
