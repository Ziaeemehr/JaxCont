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
    whole fold curve. If the exclusion in _drop_nearest were
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


def test_bogdanov_takens_detected_on_the_hopf_curve():
    """Same BT point as the fold-curve tests above (u*=(5,2), p*=(3,-1)),
    approached from the HOPF-curve side instead: the Hopf curve's own
    critical frequency omega -- the BogdanovTakens test function for
    curve="hopf" -- crosses zero exactly at p1 = -1.

    The Hopf curve for this system sits at b1 = p0-3 = 0 (so p0 = 3 for
    every point) with b2 = p1+1 < 0, omega = sqrt(-b2). Seeding at p1=-2
    (omega=1) and continuing toward p1=-0.5, pseudo-arclength continuation
    passes through the p1=-1 singularity (arc length does not need p1 to
    move monotonically) and omega changes sign, giving the crossing.
    """
    prob = hopf_curve_problem(
        _bt_shifted, jnp.array([5.0, 2.0]), jnp.array([3.0, -2.0]), free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(-2.0, -0.5),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        events=[BogdanovTakens(raw_f=_bt_shifted, free=1, curve="hopf")],
    )
    hits = [h for h in sol.events if h.kind == "bogdanov_takens"]
    assert len(hits) == 1
    assert hits[0].info["converged"] is True
    assert abs(hits[0].p - (-1.0)) < 5e-2
    assert jnp.allclose(hits[0].u, jnp.array([5.0, 2.0]), atol=5e-2)


from jaxcont.bifurcations.codim2_events import ZeroHopf
from jaxcont.bifurcations.curves import hopf_curve_problem


def _zh_system(u, p, args):
    """2-D Hopf block (+- i at criticality) decoupled from a 2-D UPPER
    TRIANGULAR z block (z1' = lam*z1 + z2, z2' = -K*z2, K > 0 fixed).

    Equilibrium is pinned at (1, -2, 3, -1) for every p. The Hopf block's
    mu = p0 + p1**2 - 1 gives the exact Hopf curve p0 = 1 - p1**2; along
    it the z block's eigenvalues are exactly {lam, -K} with
    lam = p1 - 0.5, so the zero-Hopf point sits exactly at p1 = 0.5,
    p0 = 0.75, and n = 4 satisfies zero_hopf_point's n >= 3 requirement.

    The z block is NOT the simpler single-state ``lam*z1``: with a scalar
    z, its own equilibrium row is the ONLY equation in the whole extended
    system that touches z, and that row's z-sensitivity IS the eigenvalue
    itself -- identically zero exactly at the zero-Hopf point. That makes
    zero_hopf_point's Newton system exactly rank-deficient there (confirmed
    via SVD), and refine()'s bracket-midpoint guess lands EXACTLY on the
    pinned equilibrium (since it never moves), so the very first Newton
    step lands almost exactly on the singular point and the next explodes
    to nan. The constant coupling ``+z2`` (independent of z1, lam, or K)
    gives the null-vector/eigenvector rows touching z1 a genuine, always
    -nonzero z2-sensitivity, which is enough rank to keep Newton
    well-conditioned through convergence -- confirmed by direct SVD
    (18/19, vs. 14/15 for the scalar block) and by running refine()'s own
    tolerance/iteration budget (tol=1e-6, max_iter=20) against the actual
    bracket points this system produces.
    """
    x = u[0] - 1.0
    y = u[1] + 2.0
    z1 = u[2] - 3.0
    z2 = u[3] + 1.0
    mu = p[0] + p[1] ** 2 - 1.0
    lam = p[1] - 0.5
    K = 2.0
    r2 = x * x + y * y
    return jnp.array([
        mu * x - y - x * r2,
        x + mu * y - y * r2,
        lam * z1 + z2,
        -K * z2,
    ])


def test_zero_hopf_detected_on_the_hopf_curve():
    # Seed at p1 = 0 -> p0 = 1, equilibrium (1,-2,3,-1).
    prob = hopf_curve_problem(
        _zh_system, jnp.array([1.0, -2.0, 3.0, -1.0]), jnp.array([1.0, 0.0]), free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(0.0, 1.0),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        events=[ZeroHopf(raw_f=_zh_system, free=1, curve="hopf")],
    )
    hits = [h for h in sol.events if h.kind == "zero_hopf"]
    assert len(hits) == 1
    assert hits[0].info["converged"] is True
    assert abs(hits[0].p - 0.5) < 5e-2
    assert abs(float(hits[0].info["p_fixed"]) - 0.75) < 5e-2


def _zh_fold_system(u, p, args):
    """Same fold-generating (X, Y) block as ``_bt_shifted`` (pinned zero
    eigenvalue + a real eigenvalue -(p1+1)/2), decoupled from a 2-D
    rotation block (z1, z2) whose real part ``a = p1 + 0.4`` crosses zero
    at p1 = -0.4 with a FIXED imaginary part omega = 3.0.

    REGRESSION for the near_critical pre-filter bug (review finding 1):
    with near_critical=2.0 and omega=3.0, ``|a +- 3j| ~= 3`` for every
    p1 in this span, so a pre-filter gated on the full complex magnitude
    (the bug) rejects the candidate everywhere and detects nothing. Gating
    on ``|Re(eigenvalue)| = |a|`` (the fix) sees ``|a| <= 1.6`` throughout,
    well inside the window, and the crossing at p1 = -0.4 is detected.
    """
    X, Y, z1, z2 = u[0], u[1], u[2], u[3]
    x, y = X - 5.0, Y - 2.0
    b1 = p[0] - 3.0
    b2 = p[1] + 1.0
    a = p[1] + 0.4
    omega = 3.0
    return jnp.array([
        y,
        b1 + b2 * x + x**2 + x * y,
        a * z1 - omega * z2,
        omega * z1 + a * z2,
    ])


def test_zero_hopf_detected_on_the_fold_curve_with_large_omega():
    # Seed on the fold curve at p1 = -2 (b2 = -1 -> x = 0.5 -> X = 5.5).
    prob = fold_curve_problem(
        _zh_fold_system, jnp.array([5.5, 2.0, 0.0, 0.0]), jnp.array([3.25, -2.0]),
        free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(-2.0, 0.5),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        events=[ZeroHopf(raw_f=_zh_fold_system, free=1, curve="fold")],
    )
    hits = [h for h in sol.events if h.kind == "zero_hopf"]
    assert len(hits) == 1
    assert hits[0].info["converged"] is True
    assert abs(hits[0].p - (-0.4)) < 5e-2
    assert abs(hits[0].info["omega"] - 3.0) < 5e-2


from jaxcont.bifurcations.codim2_events import GeneralizedHopf


def _bautin(u, p, args):
    """Bautin (generalized-Hopf) normal form, equilibrium shifted to
    (0.4, -0.6). mu = p0 so the Hopf curve is exactly p0 = 0; the cubic
    coefficient b = p1 - 0.4 sets sign(l1), so GH sits exactly at p1 = 0.4.
    """
    x = u[0] - 0.4
    y = u[1] + 0.6
    mu = p[0]
    b = p[1] - 0.4
    r2 = x * x + y * y
    return jnp.array([
        mu * x - y + b * x * r2 - x * r2 * r2,
        x + mu * y + b * y * r2 - y * r2 * r2,
    ])


def test_generalized_hopf_detected_on_the_hopf_curve():
    prob = hopf_curve_problem(
        _bautin, jnp.array([0.4, -0.6]), jnp.array([0.0, 0.0]), free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(0.0, 0.9),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        events=[GeneralizedHopf(raw_f=_bautin, free=1)],
    )
    hits = [h for h in sol.events if h.kind == "generalized_hopf"]
    assert len(hits) == 1
    assert hits[0].info["converged"] is True
    assert abs(hits[0].p - 0.4) < 5e-2


from jaxcont.bifurcations.codim2_events import DoubleHopf


def _hh_system(u, p, args):
    """Two decoupled 2-D Hopf blocks with distinct frequencies (1 and 2, so
    they clear double_hopf_point's separation check). Equilibrium pinned at
    (0.2, -0.1, 0.3, -0.4). Block A's mu_a = p0 gives the Hopf curve
    p0 = 0; along it block B's mu_b = p1 - 0.3 crosses zero exactly at
    p1 = 0.3, which is the double-Hopf point."""
    xa, ya = u[0] - 0.2, u[1] + 0.1
    xb, yb = u[2] - 0.3, u[3] + 0.4
    mu_a = p[0]
    mu_b = p[1] - 0.3
    ra2 = xa * xa + ya * ya
    rb2 = xb * xb + yb * yb
    return jnp.array([
        mu_a * xa - 1.0 * ya - xa * ra2,
        1.0 * xa + mu_a * ya - ya * ra2,
        mu_b * xb - 2.0 * yb - xb * rb2,
        2.0 * xb + mu_b * yb - yb * rb2,
    ])


def test_double_hopf_detected_with_automatic_seed_b():
    prob = hopf_curve_problem(
        _hh_system,
        jnp.array([0.2, -0.1, 0.3, -0.4]),
        jnp.array([0.0, 0.0]),
        free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(0.0, 0.7),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        events=[DoubleHopf(raw_f=_hh_system, free=1)],
    )
    hits = [h for h in sol.events if h.kind == "double_hopf"]
    assert len(hits) == 1
    assert hits[0].info["converged"] is True
    assert abs(hits[0].p - 0.3) < 5e-2
    # The two frequencies must be genuinely distinct -- a collapsed pair is
    # the degenerate case double_hopf_point returns nan for.
    assert abs(hits[0].info["omega_a"] - hits[0].info["omega_b"]) > 0.5


# --------------------------------------------------------------------------
# Review finding 4: near omega=0 (approaching a Bogdanov-Takens point), two
# independent _drop_nearest(+-i*omega) calls can collide on the SAME index,
# under-excluding the pinned pair.
# --------------------------------------------------------------------------


def test_drop_nearest_pinned_pair_excludes_both_halves_near_bt():
    from jaxcont.bifurcations.codim2_events import (
        _drop_nearest, _drop_nearest_pinned_pair,
    )

    # Two eigenvalues sit near zero (the collapsing pinned pair as omega ->
    # 0 near a BT point) alongside two clearly unrelated real eigenvalues.
    eigs = jnp.array([0.1 + 0j, -0.05 + 0.001j, 2.0 + 0j, -3.0 + 0j])
    omega = 1e-8

    # DISCRIMINATING POWER: the naive paired-_drop_nearest pattern this
    # replaces collapses +i*omega and -i*omega onto the SAME nearest
    # index at this omega, dropping only one of the two pinned entries.
    naive = _drop_nearest(eigs, 1j * omega) & _drop_nearest(eigs, -1j * omega)
    assert int(jnp.sum(naive)) == 3, "test setup: naive pattern should under-exclude"

    fixed = _drop_nearest_pinned_pair(eigs, omega)
    assert jnp.array_equal(fixed, jnp.array([False, False, True, True]))

    # Far from BT (omega well above the threshold), behaviour must match
    # the original paired-_drop_nearest pattern exactly.
    eigs2 = jnp.array([0.5 + 3j, 0.5 - 3j, 2.0 + 0j, -3.0 + 0j])
    omega2 = 3.0
    expected = _drop_nearest(eigs2, 1j * omega2) & _drop_nearest(eigs2, -1j * omega2)
    assert jnp.array_equal(_drop_nearest_pinned_pair(eigs2, omega2), expected)
