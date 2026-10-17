"""Analytic regressions for the loop-6 audit."""
import pathlib
import re

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxcont as jc
from jaxcont.stability.periodic_schur import periodic_spectrum

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _dtype():
    return jnp.float64 if jax.config.x64_enabled else jnp.float32


# ---------------------------------------------------------------------------
# I1: repeated multipliers have no reverse-mode derivative either
# ---------------------------------------------------------------------------

def _repeated_factors(t):
    """Cyclic product ``diag(2 e^t, 2, 0.5)``: a repeated pair at ``t = 0``."""
    dt = _dtype()
    scale = jnp.array([2.0, 2.0, 0.5], dt) * jnp.stack([jnp.exp(t), jnp.ones_like(t), jnp.ones_like(t)])
    return jnp.stack([jnp.diag(scale), jnp.eye(3, dtype=dt) + 0.0 * t])


@pytest.mark.parametrize('compiled', [False, True])
def test_repeated_multiplier_reverse_mode_is_nan(compiled):
    t = jnp.asarray(0.0, _dtype())
    logabs = lambda t: periodic_spectrum(_repeated_factors(t))[1]
    transform = jax.jit if compiled else (lambda g: g)
    grad_sum = transform(jax.grad(lambda t: jnp.sum(logabs(t))))(t)
    assert bool(jnp.isnan(grad_sum))
    # The simple mode of the same spectrum keeps its (zero) derivative.
    simple = int(jnp.argmin(logabs(t)))
    grad_simple = transform(jax.grad(lambda t: logabs(t)[simple]))(t)
    assert float(grad_simple) == 0.0
    # Away from the collision the derivative of log|det| is one.
    t1 = jnp.asarray(1e-3, _dtype())
    np.testing.assert_allclose(transform(jax.grad(lambda t: jnp.sum(logabs(t))))(t1), 1.0,
                               rtol=1e-3)


def test_repeated_multiplier_phase_reverse_mode_is_nan():
    t = jnp.asarray(0.0, _dtype())
    phase = lambda t: periodic_spectrum(_repeated_factors(t))[0]
    grad = jax.grad(lambda t: jnp.sum(jnp.imag(phase(t))))(t)
    assert bool(jnp.isnan(grad))


def test_branch_repeated_multiplier_reverse_mode_is_nan():
    from jaxcont.core.collocation import Collocation
    from jaxcont.problems.periodic import periodic_orbit_problem

    dt = _dtype()

    def rhs(u, p, c):
        x, y, z, w = u
        r2 = x * x + y * y
        return jnp.array([(p - r2) * x - y, (p - r2) * y + x, -c * z, -c * w])

    t = jnp.linspace(0, 2 * np.pi, 40, endpoint=False)
    U = jnp.stack([jnp.sqrt(.5) * jnp.cos(t), jnp.sqrt(.5) * jnp.sin(t), 0 * t, 0 * t], 1).astype(dt)
    problem = periodic_orbit_problem(rhs, U, t.astype(dt), 2 * np.pi, .5, Collocation(8, 4),
                                     args=jnp.asarray(.3, dt))
    settings = jc.ContinuationPar(ds=.05, max_steps=2, adaptive=False, loop='scan')

    def loss(c):
        args = eqx.tree_at(lambda q: q.user_args, problem.args, c)
        r = jc.continuation(problem.at(args=args), jc.Natural(), p_span=(.5, 1.), settings=settings)
        logabs = r.branch.floquet_logabs
        # The repeated exp(-c T) pair, selected by value.
        select = jax.lax.stop_gradient(r.branch.valid[:, None]
                                       & (jnp.abs(logabs + .3 * 2 * np.pi) < .05))
        return jnp.sum(jnp.where(select, logabs, 0.))

    assert bool(jnp.isnan(jax.jit(jax.grad(loss))(jnp.asarray(.3, dt))))


# ---------------------------------------------------------------------------
# I2: a final nonfinite trial does not poison reverse-mode derivatives
# ---------------------------------------------------------------------------

def _sqrt_branch_loss(alg):
    def f(u, p, a):
        # Defined for p < 0.95 only; the run toward p = 1 ends on a NaN trial.
        return jnp.array([u[0] - a * jnp.sqrt(0.95 - p)])

    settings = jc.ContinuationPar(ds=0.1, ds_max=0.1, max_steps=20, adaptive=False,
                                  loop='scan', compute_stability=False)

    def loss(a):
        problem = jc.bif_problem(f, jnp.array([a * jnp.sqrt(0.95)]), 0.0, args=a)
        r = jc.continuation(problem, alg, p_span=(0.0, 1.0), settings=settings)
        return jnp.sum(r.branch.states[:, 0] * r.branch.valid), r.stats['status']

    return loss


@pytest.mark.parametrize('alg', [jc.Natural(), jc.PseudoArclength()], ids=['natural', 'palc'])
def test_final_nonfinite_trial_keeps_reverse_gradient(alg):
    from jaxcont.core.scan_continuation import Status

    loss = _sqrt_branch_loss(alg)
    a = jnp.asarray(1.0, _dtype())
    _, status = jax.jit(loss)(a)
    assert int(status) == int(Status.NONFINITE)
    reverse = jax.jit(jax.grad(lambda a: loss(a)[0]))(a)
    forward = jax.jit(jax.jacfwd(lambda a: loss(a)[0]))(a)
    assert bool(jnp.isfinite(reverse))
    np.testing.assert_allclose(reverse, forward, rtol=1e-4)
    if isinstance(alg, jc.Natural):
        # Fixed parameter grid 0, 0.1, ..., 0.9: d/da = sum sqrt(0.95 - p).
        expected = np.sum(np.sqrt(0.95 - 0.1 * np.arange(10)))
        np.testing.assert_allclose(reverse, expected, rtol=1e-4)


def test_corrector_reverse_mode_ignores_rejected_trial():
    from jaxcont.core.scan_continuation import _natural_correct

    def f(u, p, a):
        return jnp.array([u[0] - a * jnp.sqrt(0.5 - p)])

    def rejected(a):
        u, converged, _ = _natural_correct(f, jnp.array([1.0]), jnp.array(0.9), a,
                                           jnp.array(1e-4), 4, loop='scan',
                                           p_safe=jnp.array(0.0))
        return jnp.where(converged, u[0], 0.0)

    a = jnp.asarray(1.0, _dtype())
    assert float(jax.grad(rejected)(a)) == 0.0


# ---------------------------------------------------------------------------
# I3: minimal Hopf curves report a terminated approach to BT
# ---------------------------------------------------------------------------

def _shifted_bt(u, p, args):
    x, y = u - jnp.array([.7, -.4])
    return jnp.array([y, (p[0] - .3) + (p[1] - 1.2) * y + x * x + x * y])


def _bt_problem(**kwargs):
    from jaxcont.bifurcations.curves import hopf_curve_problem

    dt = _dtype()
    return hopf_curve_problem(_shifted_bt, jnp.array([.6, -.4], dt), jnp.array([.29, 1.3], dt),
                              **kwargs)


_BT_SETTINGS = jc.ContinuationPar(ds=.01, ds_max=.02, max_steps=60, compute_stability=False)


@pytest.mark.parametrize('alg', [jc.Natural(), jc.PseudoArclength()], ids=['natural', 'palc'])
def test_minimal_hopf_curve_reports_terminated_bt(alg):
    from jaxcont.bifurcations.codim2_events import BogdanovTakens

    problem = _bt_problem()
    assert problem.curve_layout.formulation == 'minimal'
    events = [BogdanovTakens(_shifted_bt, curve='hopf')]
    r = jc.continuation(problem, alg, p_span=(1.3, 1.0), settings=_BT_SETTINGS, events=events)
    # Leaving the chart is a rejected step, not a nonfinite trial.
    assert r.status == 'stalled'
    assert r.events == []
    reasons = [(h.kind, h.info['reason']) for h in r.rejected_events]
    assert reasons == [('bogdanov_takens', 'curve_terminated')]
    # The run approaches the BT point (0.3, 1.2) closely.
    assert abs(float(r.branch.params[-1]) - 1.2) < 1e-2
    assert float(r.branch.states[-1, -1]) < 0.1

    traced = jax.jit(lambda pr: jc.continuation(pr, alg, p_span=(1.3, 1.0), settings=_BT_SETTINGS,
                                                events=events))(problem)
    refined = jc.refine_event_candidates(traced, problem, events)
    assert [(h.kind, h.info['reason']) for h in refined.rejected_events] == reasons


def test_minimal_hopf_curve_that_reaches_its_end_is_not_terminated():
    from jaxcont.bifurcations.codim2_events import BogdanovTakens

    r = jc.continuation(_bt_problem(), jc.PseudoArclength(), p_span=(1.3, 1.25),
                        settings=_BT_SETTINGS,
                        events=[BogdanovTakens(_shifted_bt, curve='hopf')])
    assert r.status == 'reached_end'
    assert r.events == [] and r.rejected_events == []


def test_minimal_hopf_chart_rejects_out_of_chart_trial():
    """Below the frequency floor the residual stays finite and nonzero."""
    problem = _bt_problem()
    X = problem.u0.at[-1].set(-0.5)
    residual = problem.f(X, problem.p0, problem.args)
    assert bool(jnp.all(jnp.isfinite(residual)))
    assert float(jnp.linalg.norm(residual)) > 0.1


# ---------------------------------------------------------------------------
# M1: theta validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('theta', [0.0, -1.0, float('nan'), float('inf')])
def test_invalid_theta_is_rejected(theta):
    with pytest.raises(ValueError, match='theta'):
        jc.ContinuationPar(theta=theta)


def test_traced_theta_is_accepted():
    def g(u, p, a):
        return jnp.array([p - u[0] ** 2])

    def end(theta):
        problem = jc.bif_problem(g, jnp.array([1.0], _dtype()), 1.0)
        settings = jc.ContinuationPar(ds=.05, max_steps=4, theta=theta, compute_stability=False)
        return jc.continuation(problem, p_span=(1.0, 0.0), settings=settings).branch.params

    assert bool(jnp.all(jnp.isfinite(jax.jit(end)(jnp.asarray(0.5, _dtype())))))


# ---------------------------------------------------------------------------
# M2: persistence of the 0.5.0 branch fields
# ---------------------------------------------------------------------------

def test_solution_round_trip_keeps_floquet_fields(tmp_path):
    sol = jc.ContinuationSolution(
        states=jnp.zeros((2, 3)), parameters=jnp.array([0.0, 0.1]),
        eigenvalues=jnp.array([[1.0 + 0j, 0.0, 0.5j], [1.0, 0.0, 0.5j]]),
        trivial_index=jnp.array([0, 0]),
        floquet_logabs=jnp.array([[0.0, -800.0, np.log(.5)], [0.0, -900.0, np.log(.5)]]),
        floquet_converged=jnp.array([True, False]),
        status='stalled',
        rejected_events=[{'type': 'fold', 'parameter': float('nan'), 'state': None,
                          'index': [0, 1], 'reason': 'not_converged'}],
    )
    path = tmp_path / 'periodic.npz'
    sol.save(str(path))
    loaded = jc.ContinuationSolution.load(str(path))
    np.testing.assert_array_equal(loaded.trivial_index, [0, 0])
    np.testing.assert_array_equal(loaded.floquet_logabs, sol.floquet_logabs)
    np.testing.assert_array_equal(loaded.floquet_converged, [True, False])
    assert loaded.status == 'stalled'
    assert loaded.rejected_events[0]['reason'] == 'not_converged'
    assert loaded.curve_layout is None


def test_curve_result_round_trip_is_plottable(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    from jaxcont.bifurcations.codim2_events import BogdanovTakens
    from jaxcont.viz import plot_two_parameter_diagram

    problem = _bt_problem(formulation='extended')
    result = jc.continuation(problem, jc.Natural(), p_span=(1.3, 1.0), settings=_BT_SETTINGS,
                             events=[BogdanovTakens(_shifted_bt, curve='hopf')])
    path = tmp_path / 'curve.npz'
    result.save(str(path))
    loaded = jc.ContinuationResult.load(str(path))
    assert loaded.branch.curve_layout == result.branch.curve_layout
    assert loaded.status == result.status
    assert [h.kind for h in loaded.events] == [h.kind for h in result.events]
    np.testing.assert_array_equal(loaded.branch.states, result.branch.states)
    plot_two_parameter_diagram([(loaded, 'hopf')])


# ---------------------------------------------------------------------------
# M3-M5: documentation
# ---------------------------------------------------------------------------

def test_api_reference_lists_the_0_5_0_surface():
    text = (ROOT / 'docs/source/api/index.rst').read_text()
    for name in ['PeriodicSchur', 'ProductFloquet', 'FloquetSpectrum', 'floquet_spectrum',
                 'refine_event_candidates', 'EventCandidates', 'EventScanInput',
                 'CurveLayout', 'decode_curve_state', 'unpack_minimal_fold_curve',
                 'unpack_minimal_hopf_curve', 'pseudo_arclength_lax_scan',
                 'natural_lax_scan']:
        assert re.search(rf'\b{name}\b', text), name


def test_vmap_example_does_not_claim_events_are_unsupported():
    text = (ROOT / 'examples/example_06_vmap_sweep.py').read_text()
    assert 'not yet supported inside jax.vmap' not in text


def test_changelog_0_5_0_has_one_section_per_kind():
    text = (ROOT / 'CHANGELOG.md').read_text()
    assert text.count('## [Unreleased]') + text.count('## [0.5.0] - Unreleased') == 1
    start = text.index('## [0.5.0] - Unreleased')
    end = text.index('\n## [', start + 1)
    section = text[start:end]
    headings = re.findall(r'^### (.+)$', section, flags=re.M)
    assert headings == ['Breaking', 'Added', 'Fixed', 'Known limitations']
    assert 'matching the existing `events=[...]` restriction' not in text


# ---------------------------------------------------------------------------
# M6: Natural lands on p_end
# ---------------------------------------------------------------------------

def test_natural_final_step_is_clipped_to_p_end():
    def f(u, p, a):
        return jnp.array([u[0] - jnp.sqrt(1.02 - p)])

    problem = jc.bif_problem(f, jnp.array([np.sqrt(1.02)], _dtype()), 0.0)
    for loop in ('while', 'scan'):
        settings = jc.ContinuationPar(ds=0.1, ds_max=0.1, max_steps=40, adaptive=False,
                                      compute_stability=False, loop=loop)
        r = jc.continuation(problem, jc.Natural(), p_span=(0.0, 1.0), settings=settings)
        assert r.status == 'reached_end'
        assert float(r.branch.params[-1]) == 1.0
        assert len(r.branch.params) == 11
    backward = jc.continuation(jc.bif_problem(f, jnp.array([np.sqrt(1.02 - 0.75)], _dtype()), 0.75),
                               jc.Natural(), p_span=(0.75, 0.0),
                               settings=jc.ContinuationPar(ds=0.1, ds_max=0.1, max_steps=40,
                                                           adaptive=False, compute_stability=False))
    assert backward.status == 'reached_end'
    assert float(backward.branch.params[-1]) == 0.0


# ---------------------------------------------------------------------------
# Fix round 1: the minimal Hopf floor follows the system's time scale
# ---------------------------------------------------------------------------

def _scaled_bt(s):
    def f(u, p, args):
        return s * _shifted_bt(u, p, args)
    return f


def test_low_frequency_minimal_seed_is_accepted():
    from jaxcont.bifurcations.curves import hopf_curve_problem
    from jaxcont.utils.precision import default_tol

    dt = _dtype()
    s = 1e-5 if jax.config.x64_enabled else 1e-3
    problem = hopf_curve_problem(_scaled_bt(s), jnp.array([.6, -.4], dt), jnp.array([.29, 1.3], dt))
    # The floor follows the seed frequency, omega_seed = s * sqrt(0.2).
    np.testing.assert_allclose(problem.args.omega_floor, default_tol(dt) ** .5 * s * np.sqrt(.2),
                               rtol=1e-3)
    r = jc.continuation(problem, jc.PseudoArclength(), p_span=(1.3, 1.25), settings=_BT_SETTINGS)
    assert r.status == 'reached_end'


@pytest.mark.parametrize('alg', [jc.Natural(), jc.PseudoArclength()], ids=['natural', 'palc'])
def test_low_frequency_minimal_curve_terminates_near_bt(alg):
    from jaxcont.bifurcations.codim2_events import BogdanovTakens
    from jaxcont.bifurcations.curves import hopf_curve_problem

    dt = _dtype()
    s = 1e-4 if jax.config.x64_enabled else 1e-2
    f = _scaled_bt(s)
    problem = hopf_curve_problem(f, jnp.array([.6, -.4], dt), jnp.array([.29, 1.3], dt))
    r = jc.continuation(problem, alg, p_span=(1.3, 1.0), settings=_BT_SETTINGS,
                        events=[BogdanovTakens(f, curve='hopf')])
    assert r.status == 'stalled'
    assert [(h.kind, h.info['reason']) for h in r.rejected_events] == [
        ('bogdanov_takens', 'curve_terminated')]
    # Same tolerance as the unscaled fixture.
    assert abs(float(r.branch.params[-1]) - 1.2) < 1e-2


def test_minimal_seed_below_floor_names_other_charts():
    from jaxcont.bifurcations.curves import hopf_curve_problem

    dt = _dtype()
    # A seed exactly at the BT point has zero frequency.
    with pytest.raises(ValueError, match="formulation='extended' or formulation='bt'"):
        hopf_curve_problem(_shifted_bt, jnp.array([.7, -.4], dt), jnp.array([.3, 1.2], dt))


def test_eager_bt_scan_is_gated_on_status():
    from jaxcont.bifurcations.codim2_events import BogdanovTakens
    from jaxcont.bifurcations.events import detect_events

    problem = _bt_problem()
    r = jc.continuation(problem, jc.PseudoArclength(), p_span=(1.3, 1.0), settings=_BT_SETTINGS)
    assert r.status == 'stalled'
    b = r.branch

    def reasons(status):
        hits = detect_events([BogdanovTakens(_shifted_bt, curve='hopf')], b.params, b.states,
                             b.tangents, None, lambda u, p: problem.f(u, p, problem.args), ds=0.,
                             curve_layout=b.curve_layout, args=problem.args, status=status)
        return [h.info.get('reason') for h in hits]

    assert reasons('stalled') == ['curve_terminated']
    assert reasons('reached_end') == []


# ---------------------------------------------------------------------------
# Fix round 2: the scale is the critical pair's seed frequency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('K', [1.0, 1e3, 1e6])
def test_stiff_mode_does_not_change_minimal_hopf_scale(K):
    from jaxcont.bifurcations.codim2_events import BogdanovTakens
    from jaxcont.bifurcations.curves import hopf_curve_problem

    dt = _dtype()

    def f(u, p, a):
        return jnp.concatenate([_shifted_bt(u[:2], p, a), -K * u[2:]])

    problem = hopf_curve_problem(f, jnp.array([.6, -.4, 0.], dt), jnp.array([.29, 1.3], dt))
    r = jc.continuation(problem, jc.PseudoArclength(), p_span=(1.3, 1.0), settings=_BT_SETTINGS,
                        events=[BogdanovTakens(f, curve='hopf')])
    assert r.status == 'stalled'
    assert [(h.kind, h.info['reason']) for h in r.rejected_events] == [
        ('bogdanov_takens', 'curve_terminated')]
    X, q = r.branch.states, r.branch.params
    assert abs(float(q[-1]) - 1.2) < 1e-3
    # Exact Hopf curve: trace q - 1.2 + x = 0 and omega**2 = -2 x.
    x = X[:, 0] - .7
    tol = 1e-5 if jax.config.x64_enabled else 1e-4
    assert float(jnp.max(jnp.abs(q - 1.2 + x))) < tol
    assert float(jnp.max(jnp.abs(X[:, 4] ** 2 + 2 * x))) < tol


def test_fast_brusselator_hopf_curve_is_unchanged():
    """Seed frequency 2 > 1: the border equations stay unscaled (as at 7eb53f1)."""
    from jaxcont.bifurcations.curves import hopf_curve_problem

    dt = _dtype()

    def bru(u, p, a):
        x, y = u
        return jnp.array([p[1] - (p[0] + 1) * x + x * x * y, p[0] * x - x * x * y])

    problem = hopf_curve_problem(bru, jnp.array([2.0, 2.5], dt), jnp.array([5.0, 2.0], dt))
    settings = jc.ContinuationPar(ds=.01, ds_max=.05, max_steps=60, compute_stability=False)
    r = jc.continuation(problem, jc.Natural(), p_span=(2.0, 3.0), settings=settings)
    assert r.status == 'reached_end'
    assert len(r.branch.params) == (47 if jax.config.x64_enabled else 24)
    # Hopf curve b = 1 + a**2 at x = a, y = b/a, omega = a.
    np.testing.assert_allclose(r.branch.states[-1], [3.0, 10 / 3, 10.0, 3.0],
                               rtol=1e-6 if jax.config.x64_enabled else 1e-5)
