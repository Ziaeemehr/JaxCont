"""Tests for period-doubling branch switching (jc.branch_switch_pd).

Fixture: a cubic-saturated variant of the repo's 4D "circle (+) transverse"
system. The repo's EXISTING PD fixture in
tests/test_period_doubling_neimark_sacker.py has a LINEAR transverse block, so
w=0 solves it for every parameter and NO period-doubled branch exists to switch
onto -- correct for detection, useless here. Adding the cubic saturation gives a
genuine doubled branch with closed-form ground truth:

    x' = (1 - r^2) x - y            r^2 = x^2 + y^2
    y' = (1 - r^2) y + x            -> circle, period T = 2*pi
    w' = (alpha - k + i*beta) w + k (x + i*y) conj(w) - c |w|^2 w
                                    w = w1 + i*w2, beta = pi/T, c = 1, k = 0.1

The rotation part turns w by exactly pi per period, so w(T) = -w(0):
anti-periodic. The coupling k (x + i*y) conj(w) breaks the S^1 symmetry
w -> exp(i*phi) w, which would otherwise make the doubled orbit a circle of
solutions and its collocation Jacobian singular. In the frame rotating with w,
the transverse linearization is diag(alpha, alpha - 2k), so the -1 multiplier
is simple: a generic period-doubling at alpha = 0 (the second transverse
multiplier reaches -1 at alpha = 2k; the local switching tests stay near
the first crossing). For alpha > 0 the exact doubled orbit is w = sqrt(alpha/c) exp(i*beta*t)
with period 2T = 4*pi.
"""

from jax import enable_x64
import jax.numpy as jnp
import numpy as np
import pytest

import jaxcont as jc
from jaxcont.bifurcations.events import EventHit
from jaxcont.bifurcations.pd_switch import _doubling_measure, _pd_eigenvector
from jaxcont.bifurcations.pd_switch import branch_switch_pd
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

T_EXACT = 2 * np.pi
BETA = np.pi / T_EXACT
C_SAT = 1.0
KAPPA = 0.1
N_STATE = 4


def rhs(u, p, args):
    x, y, w1, w2 = u[0], u[1], u[2], u[3]
    r2 = x * x + y * y
    m2 = w1 * w1 + w2 * w2
    return jnp.array([
        (1.0 - r2) * x - y,
        (1.0 - r2) * y + x,
        (p - KAPPA) * w1 - BETA * w2 - C_SAT * w1 * m2 + KAPPA * (x * w1 + y * w2),
        BETA * w1 + (p - KAPPA) * w2 - C_SAT * w2 * m2 + KAPPA * (y * w1 - x * w2),
    ])


def base_problem(alpha, ntst=10, ncol=4):
    """The w=0 circle orbit at parameter `alpha`, period T."""
    t = np.linspace(0.0, T_EXACT, 120, endpoint=False)
    u = np.stack([np.cos(t), np.sin(t), np.zeros_like(t), np.zeros_like(t)], axis=1)
    mesh = Collocation(ntst=ntst, ncol=ncol)
    prob = periodic_orbit_problem(
        rhs, jnp.asarray(u), jnp.asarray(t), T_EXACT, alpha, mesh
    )
    return prob, mesh


def test_pd_eigenvector_is_supported_on_the_transverse_block():
    # At alpha=0 the transverse multiplier is -exp(0)= -1, and its eigenvector
    # lies entirely in the (w1, w2) plane -- components 2 and 3.
    prob, mesh = base_problem(0.0)
    v = _pd_eigenvector(rhs, prob.u0, prob.p0, mesh)
    v = np.asarray(v)
    assert v.shape == (N_STATE,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)
    assert abs(v[0]) < 1e-3
    assert abs(v[1]) < 1e-3
    assert np.linalg.norm(v[2:]) > 0.99


def test_pd_eigenvector_rejects_a_point_that_is_not_period_doubling():
    # At alpha=0.5 the transverse multiplier is -exp(0.5*T) ~ -23, nowhere
    # near -1, so this is not a PD point and must be rejected loudly.
    prob, mesh = base_problem(0.5)
    with pytest.raises(ValueError, match="period-doubling"):
        _pd_eigenvector(rhs, prob.u0, prob.p0, mesh)


def _pack_mesh_only(mesh_states, mesh, n):
    """Build a U vector whose mesh-state block is `mesh_states`.

    _doubling_measure only reads the mesh-state block, so the collocation
    block and period can be zeros for this unit test.
    """
    ntst, ncol = mesh.ntst, mesh.ncol
    return jnp.concatenate([
        jnp.asarray(mesh_states).flatten(),
        jnp.zeros(ntst * ncol * n),
        jnp.zeros(1),
    ])


def test_doubling_measure_separates_a_doubled_orbit_from_a_retraced_one():
    mesh = Collocation(ntst=20, ncol=4)
    t = np.linspace(0.0, 2 * T_EXACT, 20, endpoint=False)

    # The ORIGINAL orbit traversed twice: identical under a t -> t+T shift.
    retraced = np.stack(
        [np.cos(t), np.sin(t), np.zeros_like(t), np.zeros_like(t)], axis=1
    )
    d_retraced = _doubling_measure(
        _pack_mesh_only(retraced, mesh, N_STATE), mesh, N_STATE
    )
    assert d_retraced < 1e-3

    # A genuine doubled orbit: w rotates by pi per period, so w(t+T) = -w(t).
    amp = np.sqrt(0.05 / C_SAT)
    doubled = np.stack([
        np.cos(t), np.sin(t),
        amp * np.cos(BETA * t), amp * np.sin(BETA * t),
    ], axis=1)
    d_doubled = _doubling_measure(
        _pack_mesh_only(doubled, mesh, N_STATE), mesh, N_STATE
    )
    # Closed form: d = 2*amp, scale = sqrt(1 + amp^2)  ->  ~0.436
    assert d_doubled == pytest.approx(2 * amp / np.sqrt(1 + amp**2), abs=1e-3)
    assert d_doubled > 0.4


def pd_event_at_origin():
    """An EventHit at the exact PD point (alpha = 0), built directly.

    Building the event directly rather than running a continuation keeps these
    tests fast and deterministic; Task 4's example covers the detect-then-switch
    path end to end.
    """
    prob, mesh = base_problem(0.0)
    event = EventHit(kind="period_doubling", p=float(prob.p0), u=prob.u0, index=None)
    return event, mesh, prob


def transverse_amplitude(prob, mesh):
    """max |w| over the mesh points of a doubled-period problem."""
    ntst = mesh.ntst
    ms = np.asarray(prob.u0[: ntst * N_STATE].reshape(ntst, N_STATE))
    return float(np.max(np.sqrt(ms[:, 2] ** 2 + ms[:, 3] ** 2)))


def test_switch_lands_on_the_period_doubled_branch():
    event, mesh, _ = pd_event_at_origin()
    dp = 0.05
    switched = branch_switch_pd(rhs, event, mesh, dp=dp)

    doubled_mesh = Collocation(ntst=2 * mesh.ntst, ncol=mesh.ncol)
    # Closed-form ground truth: |w| = sqrt(alpha/c), period = 2T.
    assert transverse_amplitude(switched, doubled_mesh) == pytest.approx(
        np.sqrt(dp / C_SAT), abs=1e-3
    )
    assert float(switched.u0[-1]) == pytest.approx(2 * T_EXACT, abs=1e-3)
    assert float(switched.p0) == pytest.approx(dp, abs=1e-6)
    assert switched.kind == "periodic"


def test_switch_doubles_the_mesh():
    event, mesh, _ = pd_event_at_origin()
    switched = branch_switch_pd(rhs, event, mesh, dp=0.05)
    expected_len = 2 * mesh.ntst * (1 + mesh.ncol) * N_STATE + 1
    assert switched.u0.shape[-1] == expected_len


def test_switch_finds_the_branch_without_being_told_which_side_it_is_on():
    # The doubled branch exists only for alpha > 0 here. The default search
    # spans both signs, so it must succeed without a dp_sign hint.
    event, mesh, _ = pd_event_at_origin()
    switched = branch_switch_pd(rhs, event, mesh, dp=0.05, dp_sign=None)
    assert float(switched.p0) > 0


def test_switch_raises_on_the_wrong_side_when_the_sign_is_pinned():
    # Pinned to the side with no doubled branch: every eps collapses, so this
    # must raise rather than hand back the retraced original orbit.
    event, mesh, _ = pd_event_at_origin()
    with pytest.raises(ValueError, match="could not find"):
        branch_switch_pd(rhs, event, mesh, dp=0.05, dp_sign=-1)


def test_switch_raises_rather_than_returning_a_collapsed_branch():
    # THE key regression test for this feature's main hazard: an under-sized
    # seed converges cleanly (residual ~7e-6) onto the ORIGINAL orbit traversed
    # twice. Without the detector this returns a plausible, wrong BifProblem.
    event, mesh, _ = pd_event_at_origin()
    with pytest.raises(ValueError, match="collapsed"):
        branch_switch_pd(rhs, event, mesh, dp=0.05, eps_ladder=(0.01, 0.02))


def test_switch_rejects_a_non_period_doubling_event():
    _event, mesh, prob = pd_event_at_origin()
    fold_event = EventHit(kind="fold", p=float(prob.p0), u=prob.u0, index=None)
    with pytest.raises(ValueError, match="period_doubling"):
        branch_switch_pd(rhs, fold_event, mesh, dp=0.05)


def test_detect_then_switch_end_to_end():
    """Continue the base branch, detect the PD event with jc.PeriodDoubling, and
    switch on the detected event.

    Continuing the switched branch is covered by
    examples/example_22_period_doubling_switch.py rather than here, to keep this
    test's runtime down -- the switch itself is what this file is testing."""
    prob, mesh = base_problem(-0.1)
    result = jc.continuation(
        prob,
        p_span=(-0.1, 0.1),
        settings=jc.ContinuationPar(
            ds=0.02, max_steps=40, newton_tol=1e-5, compute_stability=True
        ),
        events=[jc.PeriodDoubling()],
    )
    pd_events = [e for e in result.events if e.kind == "period_doubling"]
    assert pd_events, "no period-doubling event detected on the base branch"

    event = pd_events[0]
    assert event.p == pytest.approx(0.0, abs=2e-2)

    dp = 0.05
    switched = jc.branch_switch_pd(rhs, event, mesh, dp=dp)
    doubled_mesh = Collocation(ntst=2 * mesh.ntst, ncol=mesh.ncol)
    amp = transverse_amplitude(switched, doubled_mesh)
    # Ground truth is sqrt((p_switched - p_bif)/c); the detected p_bif carries
    # the event's own bisection error, so this is looser than the exact-point test.
    expected = np.sqrt(max(float(switched.p0) - event.p, 0.0) / C_SAT)
    assert amp == pytest.approx(expected, abs=2e-2)
    assert float(switched.u0[-1]) == pytest.approx(2 * T_EXACT, abs=1e-2)


@enable_x64()
def test_pd_switch_invariant_under_translation():
    event, mesh, _ = pd_event_at_origin()
    offset = jnp.full((N_STATE,), 1e4)

    def translated_rhs(u, p, args):
        return rhs(u - offset, p, args)

    shifted_u = jnp.concatenate([
        (event.u[:-1].reshape(-1, N_STATE) + offset).flatten(), event.u[-1:]
    ])
    shifted_event = EventHit(kind=event.kind, p=event.p, u=shifted_u, index=None)
    # Float32 quantization at 1e4 exceeds the default Newton tolerance.
    switched = branch_switch_pd(translated_rhs, shifted_event, mesh, dp=0.05)
    doubled_mesh = Collocation(ntst=2 * mesh.ntst, ncol=mesh.ncol)
    states = np.asarray(switched.u0[:-1]).reshape(-1, N_STATE) - np.asarray(offset)
    amplitude = np.max(np.linalg.norm(states[:doubled_mesh.ntst, 2:], axis=1))
    assert amplitude == pytest.approx(np.sqrt(0.05 / C_SAT), abs=2e-2)
    assert float(switched.u0[-1]) == pytest.approx(2 * T_EXACT, abs=2e-2)
    assert _doubling_measure(switched.u0, doubled_mesh, N_STATE) > 0.4


@pytest.mark.parametrize("offset", [0.0, 1e4])
def test_doubling_measure_uses_orbit_amplitude(offset):
    mesh = Collocation(ntst=20, ncol=4)
    t = np.linspace(0, 2 * T_EXACT, mesh.ntst, endpoint=False)
    amp = np.sqrt(0.05)
    states = np.stack([
        np.cos(t), np.sin(t),
        amp * np.cos(BETA * t), amp * np.sin(BETA * t),
    ], axis=1)
    measure = _doubling_measure(
        _pack_mesh_only(states + offset, mesh, N_STATE), mesh, N_STATE
    )
    assert measure == pytest.approx(2 * amp / np.sqrt(1 + amp**2), abs=2e-3)
