"""Explicit curve layouts and diagnostic mode conventions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxcont as jc
from jaxcont.utils.precision import default_tol


def test_curve_layout_roundtrip_extended():
    from jaxcont.bifurcations.curve_layout import decode_curve_state
    from jaxcont.bifurcations.curves import hopf_curve_problem

    def f(u, p, args):
        x, y = u - jnp.array([0.5, -0.3])
        mu = p[0] + p[1] ** 2 - 2
        return jnp.array([mu * x - y, x + mu * y])

    problem = hopf_curve_problem(f, jnp.array([0.5, -0.3]), jnp.array([1.75, 0.5]), formulation="extended")
    layout = problem.curve_layout
    u, p, modes = jax.jit(lambda X: decode_curve_state(X, 0.5, layout, f, None))(problem.u0)
    tol = default_tol(u.dtype)
    assert jnp.max(jnp.abs(u - jnp.array([0.5, -0.3]))) < 100 * tol
    assert jnp.max(jnp.abs(p - jnp.array([1.75, 0.5]))) < 100 * tol
    assert abs(modes.omega - 1) < 100 * tol
    assert modes.w is None
    assert jax.tree.unflatten(*reversed(jax.tree.flatten(problem))).curve_layout == layout
    settings = jc.ContinuationPar(compute_stability=False, max_steps=4, bothside=True)
    result = jc.continuation(problem, p_span=(0.4, 0.6), settings=settings)
    assert result.branch.curve_layout == layout
    rebuilt = jax.tree.unflatten(*reversed(jax.tree.flatten(result)))
    assert rebuilt.branch.curve_layout == layout
    traced = jax.jit(
        lambda: jc.continuation(
            problem,
            p_span=(0.5, 0.6),
            settings=jc.ContinuationPar(compute_stability=False, max_steps=2),
        )
    )()
    assert traced.branch.curve_layout == layout


def test_curve_layout_required_for_curve_decoding():
    from jaxcont.bifurcations.curve_layout import decode_curve_state

    with pytest.raises(ValueError, match="layout"):
        decode_curve_state(jnp.zeros(8), 0.5, None, None, None)


def test_layout_fold_diagnostic_orientation_no_false_cp():
    from jaxcont.bifurcations.curve_layout import orient_curve_modes, CurveModes
    from jaxcont.bifurcations.modes import crossings

    raw = jnp.array([[1.0, 0.0], [-2.0, 0.0], [0.0, 0.0], [3.0, 0.0], [-1.0, 0.0]])
    valid = jnp.array([True, True, False, True, False])

    def run(v):
        modes, eligible = orient_curve_modes(CurveModes(v=v), valid)
        return modes.v, eligible

    vectors, eligible = jax.jit(jax.vmap(run))(jnp.stack([raw, -raw]))
    np.testing.assert_allclose(vectors[:, [0, 1, 3], 0], 1)
    np.testing.assert_array_equal(eligible, [[True, False, False, False]] * 2)
    assert crossings(np.asarray(vectors[0, :, 0]), np.asarray(valid)) == []
    assert len(crossings(np.array([-1.0, 1.0, 0.0, 1.0, 0.0]), np.asarray(valid))) == 1
    ambiguous = CurveModes(v=jnp.eye(2))
    assert not bool(orient_curve_modes(ambiguous, jnp.ones(2, bool))[1][0])


def test_layout_hopf_diagnostic_unit_norm():
    from jaxcont.bifurcations.curve_layout import CurveLayout, decode_curve_state
    from jaxcont.bifurcations.hopf_normal_form import lyapunov_coefficient

    def f(u, p, args):
        x, y = u
        r = x * x + y * y
        return jnp.array([-y - x * r, x - y * r])

    values = []
    for scale, phase in [(1.0, 0.0), (3.0, 0.7), (0.2, 2.0)]:
        q = scale * jnp.exp(1j * phase) * jnp.array([1.0, -1j])
        X = jnp.concatenate([jnp.zeros(2), jnp.zeros(1), q.real, q.imag, jnp.ones(1)])
        u, p, m = decode_curve_state(X, 0.5, CurveLayout("hopf", "extended", 2, 1), f, None)
        assert abs(jnp.linalg.norm(m.q1 + 1j * m.q2) - 1) < 100 * default_tol(u.dtype)
        values.append(lyapunov_coefficient(f, u, p, m.q1, m.q2, m.omega))
    np.testing.assert_allclose(values, values[0], atol=100 * default_tol(u.dtype))


def test_layout_cusp_eager_scan_signs_and_padding():
    from jaxcont.bifurcations.curve_layout import CurveLayout
    from jaxcont.bifurcations.codim2_events import Cusp
    from jaxcont.bifurcations.events import BranchPoint

    def f(u, p, args):
        return jnp.array([p[0] + p[1] * u[0] + u[0] ** 3])

    event = Cusp(f)
    layout = CurveLayout("fold", "extended", 1, 1)

    def point(x, sign):
        return BranchPoint(p=-3 * x * x, u=jnp.array([x, 2 * x**3, sign]), curve_layout=layout)

    assert event.scan([point(1.0, 1.0), point(2.0, -1.0), point(3.0, 1.0)]) == []
    assert event.scan([point(-1.0, -1.0), point(1.0, 1.0)]) == [(0, 1, None)]
    padded = BranchPoint(p=0.0, u=jnp.full(3, jnp.nan), curve_layout=layout)
    assert event.scan([point(-1.0, -1.0), padded, point(1.0, 1.0), padded]) == []
    with pytest.raises(ValueError, match="layout"):
        event.test_function(BranchPoint(p=0.0, u=jnp.zeros(3)))


def test_layout_hopf_phase_alignment_preserves_unit_modes_after_orthogonal_step():
    from jaxcont.bifurcations.curve_layout import CurveModes, orient_curve_modes

    z = jnp.array([[1., 0.], [0., 1j], [0., -2.]])
    modes, eligible = jax.jit(orient_curve_modes)(
        CurveModes(q1=z.real, q2=z.imag), jnp.ones(3, dtype=bool)
    )
    aligned = modes.q1 + 1j*modes.q2
    np.testing.assert_allclose(jnp.linalg.norm(aligned, axis=-1), jnp.ones(3))
    np.testing.assert_array_equal(eligible, [False, True])


@pytest.mark.parametrize("kind", ["fold", "hopf"])
def test_layout_exact_zero_owned_by_eligible_outgoing_bracket(kind):
    from jaxcont.bifurcations.codim2_events import Cusp, GeneralizedHopf
    from jaxcont.bifurcations.curve_layout import CurveLayout
    from jaxcont.bifurcations.events import BranchPoint

    # The critical subspace changes orthogonally into the zero-coefficient
    # sample, then stays fixed: only the outgoing bracket is reliable.
    if kind == "fold":
        def f(u, p, args):
            x, y = u
            t = p[1]
            return jnp.array([(t + 1) * x - x*x, t*(t - 1)*y + t*y*y])

        event = Cusp(f)
        vectors = jnp.array([[1., 0.], [0., 1.], [0., 1.]])
        states = [jnp.concatenate([jnp.zeros(3), v]) for v in vectors]
        n = 2
    else:
        def f(u, p, args):
            x, y, z, w = u
            t = p[1]
            r1, r2 = x*x + y*y, z*z + w*w
            return jnp.array([(t + 1)*x - y - x*r1,
                              x + (t + 1)*y - y*r1,
                              t*(t - 1)*z - w + t*z*r2,
                              z + t*(t - 1)*w + t*w*r2])

        event = GeneralizedHopf(f)
        vectors = jnp.array([[1., -1j, 0., 0.],
                             [0., 0., 1., -1j], [0., 0., 1., -1j]])
        states = [jnp.concatenate([jnp.zeros(5), v.real, v.imag, jnp.ones(1)])
                  for v in vectors]
        n = 4
    layout = CurveLayout(kind, "extended", n, 1)
    points = [BranchPoint(p=t, u=X, curve_layout=layout)
              for t, X in zip([-1., 0., 1.], states)]
    values = [event.test_function(pt) for pt in points]
    assert values[0] < 0
    assert values[1] == 0
    assert values[2] > 0
    assert event.scan(points) == [(1, 2, None)]
    # A zero at the start of a branch still opens its first bracket.
    assert event.scan(points[1:]) == [(0, 1, None)]


def _shifted_cusp(u, p, args):
    x = u[0] - 0.7
    return jnp.array([(p[0] - .3) + (p[1] - 1.2)*x - x**3])


def test_minimal_fold_shifted_cusp():
    from jaxcont.bifurcations.curves import fold_curve_problem
    problem = fold_curve_problem(_shifted_cusp, jnp.array([1.7]), jnp.array([-1.7, 4.2]))
    assert problem.u0.shape == (2,)
    result = jc.continuation(problem, p_span=(4.2, 5.2), alg=jc.Natural(), settings=jc.ContinuationPar(
        ds=.1, adaptive=False, max_steps=10, compute_stability=False))
    X = result.branch.states
    q = result.branch.params
    assert X.shape == (11, 2)
    np.testing.assert_allclose(q, jnp.linspace(4.2, 5.2, 11), atol=100*default_tol(X.dtype))
    error = jnp.abs(27*(X[:, 1]-.3)**2 - 4*(q-1.2)**3) / (1+4*(q-1.2)**3)
    assert jnp.max(error) < 100*default_tol(X.dtype)


def test_border_refresh_preserves_root_and_reject_keeps_args():
    import equinox as eqx
    from jaxcont.bifurcations.curves import fold_curve_problem, fold_curve_step_update
    from jaxcont.core.scan_continuation import _apply_step_update
    problem = fold_curve_problem(_shifted_cusp, jnp.array([1.7]), jnp.array([-1.7, 4.2]), formulation="minimal")
    a = eqx.tree_at(lambda a: (a.b, a.c), problem.args, (2*problem.args.b, -3*problem.args.c))
    refreshed = fold_curve_step_update(a, problem.u0, problem.p0)
    np.testing.assert_allclose(jnp.linalg.norm(refreshed.b), 1)
    np.testing.assert_allclose(jnp.linalg.norm(refreshed.c), 1)
    for updated in [a, refreshed]:
        assert jnp.max(jnp.abs(problem.f(problem.u0, problem.p0, updated))) < 100*default_tol(problem.u0.dtype)
    rejected = jax.jit(lambda a: _apply_step_update(fold_curve_step_update, a, problem.u0, problem.p0, False))(a)
    for before, after in zip(jax.tree.leaves(a), jax.tree.leaves(rejected)):
        np.testing.assert_array_equal(before, after)


def test_fold_diagnostic_orientation_no_false_cp():
    test_layout_fold_diagnostic_orientation_no_false_cp()
    from jaxcont.bifurcations.curve_layout import CurveLayout, decode_curve_state
    from jaxcont.bifurcations.codim2_events import Cusp
    from jaxcont.bifurcations.events import BranchPoint
    layout = CurveLayout("fold", "minimal", 1, 1)
    def point(x):
        return BranchPoint(p=1.2+3*x*x, u=jnp.array([.7+x, .3-2*x**3]), curve_layout=layout)
    event = Cusp(_shifted_cusp)
    assert event.scan([point(1.), point(2.), point(3.)]) == []
    assert event.scan([point(-1.), point(1.)]) == [(0, 1, None)]
    _, _, modes = jax.jit(lambda X: decode_curve_state(X, 4.2, layout, _shifted_cusp, None))(point(1.).u)
    np.testing.assert_allclose(modes.v, [1.])


def test_minimal_fold_array_cp_padding_and_batches():
    from jaxcont.bifurcations.curve_layout import CurveLayout
    from jaxcont.bifurcations.codim2_events import Cusp
    from jaxcont.bifurcations.event_scan import EventScanInput

    event = Cusp(_shifted_cusp)
    layout = CurveLayout("fold", "minimal", 1, 1)
    valid = jnp.array([True, True, False, True, True, False])
    def scan(x):
        states = jnp.stack([.7+x, .3-2*x**3], axis=-1)
        data = EventScanInput(1.2+3*x*x, states, None, None, None, valid, layout)
        return event.scan_arrays(data, None).bracket_mask
    segments = jnp.array([[1., 2., 0., 3., 4., 0.], [-1., 1., 0., -1., 1., 0.]])
    masks = jax.jit(jax.vmap(scan))(segments)
    np.testing.assert_array_equal(masks[..., 0], [[False]*5, [True, False, False, True, False]])


def test_minimal_fold_default_plotting():
    import matplotlib.pyplot as plt
    from jaxcont.bifurcations.curves import fold_curve_problem
    from jaxcont.viz.two_parameter import plot_two_parameter_diagram
    problem = fold_curve_problem(_shifted_cusp, jnp.array([1.7]), jnp.array([-1.7, 4.2]))
    result = jc.continuation(problem, p_span=(4.2, 4.4), alg=jc.Natural(),
        settings=jc.ContinuationPar(ds=.1, adaptive=False, max_steps=2, compute_stability=False))
    ax = plot_two_parameter_diagram([(result, "fold")])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), result.branch.states[:, 1])
    plt.close(ax.figure)


def _shifted_hopf(u, p, args):
    x, y = u - args
    mu = p[0] + p[1]**2 - 2
    r = x*x + y*y
    return jnp.array([mu*x-y-x*r, x+mu*y-y*r])


def _hopf_problem(q=0., formulation="minimal"):
    from jaxcont.bifurcations.curves import hopf_curve_problem
    center = jnp.array([.5, -.3])
    return hopf_curve_problem(_shifted_hopf, center, jnp.array([2-q*q, q]),
                              args=center, formulation=formulation)


def _hopf_branch(formulation="minimal"):
    return jc.continuation(_hopf_problem(formulation=formulation), p_span=(0., 1.),
        alg=jc.Natural(), settings=jc.ContinuationPar(ds=.25, ds_max=.25, adaptive=False,
            max_steps=4, compute_stability=False, loop="scan"))


def test_minimal_hopf_shifted_parabola():
    result = _hopf_branch()
    X = result.branch.states
    tol = default_tol(X.dtype)
    assert X.shape == (5, 4)
    assert result.branch.valid is None or jnp.all(result.branch.valid)
    assert jnp.max(jnp.abs(X[:, :2] - jnp.array([.5, -.3]))) < 100*tol
    assert jnp.max(jnp.abs(X[:, 2] - jnp.array([2, 1.9375, 1.75, 1.4375, 1]))) < 100*tol
    assert jnp.max(jnp.abs(X[:, 3] - 1)) < 100*tol


def test_minimal_and_extended_agree_at_fixed_parameters():
    minimal, extended = _hopf_branch(), _hopf_branch("extended")
    assert jnp.max(jnp.abs(minimal.branch.states[:, :3] - extended.branch.states[:, :3])) < 100*default_tol(minimal.branch.states.dtype)
    np.testing.assert_allclose(minimal.branch.params, [0, .25, .5, .75, 1])


def test_minimal_curve_decoding_events_and_plotting():
    import matplotlib.pyplot as plt
    from jaxcont.bifurcations.curve_layout import decode_curve_state
    from jaxcont.bifurcations.codim2_events import GeneralizedHopf
    from jaxcont.bifurcations.events import BranchPoint
    from jaxcont.viz.two_parameter import plot_two_parameter_diagram
    result = _hopf_branch()
    layout = result.branch.curve_layout
    event = GeneralizedHopf(_shifted_hopf, args=jnp.array([.5, -.3]))
    points = []
    for X, q in zip(result.branch.states, result.branch.params):
        u, p, modes = decode_curve_state(X, q, layout, _shifted_hopf, event.args)
        assert u.shape == (2,)
        assert abs(jnp.linalg.norm(modes.q1+1j*modes.q2)-1) < 100*default_tol(X.dtype)
        points.append(BranchPoint(p=q, u=X, curve_layout=layout))
    assert event.scan(points) == []
    ax = plot_two_parameter_diagram([(result, "hopf")])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [2, 1.9375, 1.75, 1.4375, 1], atol=100*default_tol(X.dtype))
    plt.close(ax.figure)


def test_minimal_curve_seed_q_sensitivity():
    seed = lambda q: _hopf_problem(q).u0[2]
    derivative = jax.jit(jax.grad(seed))(.5)
    assert abs(derivative + 1) < 100*default_tol(derivative.dtype)
    np.testing.assert_allclose(jax.jit(jax.vmap(seed))(jnp.array([0., .5, 1.])), [2, 1.75, 1], atol=100*default_tol(derivative.dtype))
    def branch_sum(q):
        problem = _hopf_problem(q)
        result = jc.continuation(problem, p_span=(q, q+.5), alg=jc.Natural(),
            settings=jc.ContinuationPar(ds=.25, ds_max=.25, adaptive=False, max_steps=2,
                compute_stability=False, loop="scan"))
        return result.branch.states.sum()
    assert jnp.isfinite(jax.jit(jax.grad(branch_sum))(.25))


def test_hopf_diagnostic_unit_norm():
    test_layout_hopf_diagnostic_unit_norm()
    from jaxcont.bifurcations.curve_layout import decode_curve_state
    from jaxcont.bifurcations.hopf_normal_form import lyapunov_coefficient
    problem = _hopf_problem(.5)
    u, p, m = decode_curve_state(problem.u0, .5, problem.curve_layout, _shifted_hopf, problem.args.user_args)
    assert abs(jnp.linalg.norm(m.q1+1j*m.q2)-1) < 100*default_tol(u.dtype)
    minimal = lyapunov_coefficient(_shifted_hopf, u, p, m.q1, m.q2, m.omega, problem.args.user_args)
    extended = _hopf_problem(.5, "extended")
    _, _, e = decode_curve_state(extended.u0, .5, extended.curve_layout, _shifted_hopf, extended.args.user_args)
    expected = lyapunov_coefficient(_shifted_hopf, u, p, e.q1, e.q2, e.omega, problem.args.user_args)
    assert abs(minimal-expected) < 100*default_tol(u.dtype)


def test_minimal_hopf_border_refresh_and_frequency_limit():
    import equinox as eqx
    from jaxcont.bifurcations.curves import hopf_curve_step_update, unpack_minimal_hopf_curve
    from jaxcont.core.scan_continuation import _apply_step_update
    problem = _hopf_problem(.5)
    a = eqx.tree_at(lambda a: (a.b, a.c), problem.args,
        (2*jnp.exp(.7j)*problem.args.b, -3*jnp.exp(.2j)*problem.args.c))
    refreshed = jax.jit(hopf_curve_step_update)(a, problem.u0, problem.p0)
    tol = default_tol(problem.u0.dtype)
    for border in [refreshed.b, refreshed.c]:
        assert abs(jnp.linalg.norm(border)-1) < 100*tol
    for updated in [a, refreshed]:
        assert jnp.max(jnp.abs(problem.f(problem.u0, problem.p0, updated))) < 100*tol
    rejected = jax.jit(lambda a: _apply_step_update(hopf_curve_step_update, a, problem.u0, problem.p0, False))(a)
    for before, after in zip(jax.tree.leaves(a), jax.tree.leaves(rejected)):
        np.testing.assert_array_equal(before, after)
    u, fixed, omega = unpack_minimal_hopf_curve(problem.u0, 2)
    assert u.shape == (2,) and abs(fixed-1.75) < 100*tol and abs(omega-1) < 100*tol
    # Below the chart's floor the residual is finite and bounded away from a root.
    from jaxcont.bifurcations.curve_layout import minimal_hopf_floor
    floor = minimal_hopf_floor(problem.u0.dtype)
    for frequency in [0., -1., 10*tol, floor]:
        residual = problem.f(problem.u0.at[-1].set(frequency), problem.p0, a)
        assert jnp.all(jnp.isfinite(residual))
        assert jnp.max(jnp.abs(residual)) >= 0.5 * floor


def test_minimal_hopf_array_gh_crossing():
    from jaxcont.bifurcations.codim2_events import GeneralizedHopf
    from jaxcont.bifurcations.curve_layout import CurveLayout
    from jaxcont.bifurcations.event_scan import EventScanInput
    def f(u, p, args):
        x, y = u
        r = x*x + y*y
        return jnp.array([p[0]*x-y+p[1]*x*r, x+p[0]*y+p[1]*y*r])
    event = GeneralizedHopf(f)
    valid = jnp.array([True, True, True, False])
    data = EventScanInput(jnp.array([-.5, 0., .5, .5]),
        jnp.tile(jnp.array([0., 0., 0., 1.]), (4, 1)),
        None, None, None, valid, CurveLayout("hopf", "minimal", 2, 1))
    candidates = jax.jit(event.scan_arrays)(data, None)
    np.testing.assert_array_equal(candidates.bracket_mask[:, 0], [True, False, False])
    assert jnp.all(jnp.isfinite(candidates.values))


def test_minimal_hopf_diagnostic_derivative_with_repeated_stable_modes():
    from jaxcont.bifurcations.curve_layout import CurveLayout, decode_curve_state

    def f(u, p, angle):
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        rotation = jnp.eye(4).at[0, 0].set(cosine).at[0, 2].set(-sine)
        rotation = rotation.at[2, 0].set(sine).at[2, 2].set(cosine)
        matrix = jnp.array([[0., -1., 0., 0.], [1., 0., 0., 0.],
                            [0., 0., -2., 0.], [0., 0., 0., -2.]])
        return rotation @ matrix @ rotation.T @ u

    layout = CurveLayout("hopf", "minimal", 4, 1)
    state = jnp.array([0., 0., 0., 0., 0., 1.])

    def objective(angle):
        _, _, modes = decode_curve_state(state, 0., layout, f, angle)
        # A projector entry is independent of the SVD mode's arbitrary phase.
        return modes.q1[0] ** 2 + modes.q2[0] ** 2

    angles = jnp.array([0., .2, .4])
    values, derivatives = jax.jit(jax.vmap(jax.value_and_grad(objective)))(angles)
    tol = 100 * default_tol(state.dtype)
    assert jnp.all(jnp.isfinite(derivatives))
    np.testing.assert_allclose(values, .5 * jnp.cos(angles) ** 2, atol=tol)
    np.testing.assert_allclose(derivatives, -jnp.sin(angles) * jnp.cos(angles), atol=tol)
