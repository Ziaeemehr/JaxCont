"""Real Jordan-chain Hopf chart through a shifted BT point."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import jaxcont as jc
from jaxcont.bifurcations.curves import hopf_curve_problem
from jaxcont.utils.precision import default_tol


def shifted(u, p, args):
    x, y = u - jnp.array([.7, -.4])
    return jnp.array([y, (p[0]-.3)+(p[1]-1.2)*y+x*x+x*y])


def seed(f=shifted, u=None):
    return hopf_curve_problem(f, jnp.array([.6, -.4]) if u is None else u,
                              jnp.array([.29, 1.3]), formulation="bt")


def test_bt_chart_residual_and_jacobian_at_zero():
    problem = seed()
    a = eqx.tree_at(lambda a: a.v_anchor, problem.args, jnp.array([1., 0.]))
    X = jnp.array([.7, -.4, .3, 1., 0., 0., 1., 0.])
    assert jnp.max(jnp.abs(problem.f(X, 1.2, a))) < 100*default_tol(X.dtype)
    assert jnp.linalg.svd(jax.jacfwd(problem.f)(X, 1.2, a), compute_uv=False)[-1] > .05


def run(problem):
    return jc.continuation(problem, p_span=(1.3, 1.1), alg=jc.Natural(),
        settings=jc.ContinuationPar(ds=.01, adaptive=False, max_steps=20,
                                   compute_stability=False)).branch


def check_locus(branch):
    X, q = branch.states, branch.params
    tol = 100*default_tol(X.dtype)
    assert len(q) == 21
    assert branch.valid is None or jnp.all(branch.valid)
    np.testing.assert_allclose(q, jnp.linspace(1.3, 1.1, 21), atol=tol)
    x = 1.2-q
    assert jnp.max(jnp.abs(X[:, 2]-(.3-x*x))) < tol
    assert jnp.max(jnp.abs(X[:, -1]+2*x)) < tol
    assert X[0, -1] > 0 and X[-1, -1] < 0
    return x


def test_hopf_bt_kappa_crosses_shifted_bt():
    branch = run(seed())
    x = check_locus(branch)
    assert jnp.max(jnp.abs(branch.states[:, :2]-jnp.stack([.7+x, jnp.full_like(x, -.4)], axis=-1))) < 100*default_tol(x.dtype)


def transform(q):
    return jnp.array([[1., .2+(q-1.2)], [.3*(q-1.2), 1.]])


def nonnormal(u, p, args):
    S = transform(p[1])
    x, y = jnp.linalg.solve(S, u-jnp.array([.7, -.4]))
    return S @ jnp.array([y, (p[0]-.3)+(p[1]-1.2)*y+x*x+x*y])


def test_bt_fixed_anchor_nonnormal_similarity():
    from jaxcont.bifurcations.curve_layout import decode_curve_state
    from jaxcont.core.scan_continuation import _apply_step_update
    problem = seed(nonnormal, jnp.array([.7, -.4])+transform(1.3)@jnp.array([-.1, 0.]))
    branch = run(problem)
    x = check_locus(branch)
    expected = jax.vmap(lambda q, x: jnp.array([.7, -.4])+transform(q)@jnp.array([x, 0.]))(branch.params, x)
    tol = 100*default_tol(x.dtype)
    assert jnp.max(jnp.abs(branch.states[:, :2]-expected)) < tol
    for X, q in zip(branch.states, branch.params):
        for accepted in [False, True]:
            a = _apply_step_update(problem.step_update, problem.args, X, q, accepted)
            for before, after in zip(jax.tree.leaves(problem.args), jax.tree.leaves(a)):
                np.testing.assert_array_equal(before, after)
            assert jnp.max(jnp.abs(problem.f(X, q, a))) < tol
        assert jnp.linalg.svd(jax.jacfwd(problem.f)(X, q, a), compute_uv=False)[-1] > .01
        _, _, m = decode_curve_state(X, q, problem.curve_layout, nonnormal, None)
        np.testing.assert_array_equal(m.v, X[3:5])
        np.testing.assert_array_equal(m.w, X[5:7])
        assert m.q1.shape == m.q2.shape == (2,)
    A = jnp.array([[.2, 1.], [-.04, -.2]])
    def linear(u, p, args):
        return A @ u
    w = jnp.array([0., 1.])/jnp.sqrt(1.04)
    v = A@w
    X = jnp.concatenate([jnp.zeros(3), v, w, jnp.zeros(1)])
    a = eqx.tree_at(lambda a: a.v_anchor, problem.args, jnp.array([1., 0.]))
    from jaxcont.bifurcations.curves import _bt_hopf_residual
    assert jnp.max(jnp.abs(_bt_hopf_residual(X, jnp.zeros(2), linear, a, 2))) < tol
    bad = eqx.tree_at(lambda a: a.v_anchor, a, v)
    assert abs(_bt_hopf_residual(X, jnp.zeros(2), linear, bad, 2)[-1]+.1923076923) < tol


def test_bt_decoder_zero_and_neutral_modes_are_real_and_trace_safe():
    from jaxcont.bifurcations.curve_layout import CurveLayout, decode_curve_state
    from jaxcont.bifurcations.curves import hopf_curve_frequency, unpack_bt_hopf_curve
    layout = CurveLayout("hopf", "bt", 2, 1)
    def decode(kappa):
        X = jnp.array([.7, -.4, .3, 1., 0., 0., 1., 0.]).at[-1].set(kappa)
        u, fixed, v, w, k = unpack_bt_hopf_curve(X, 2)
        decoded_u, p, modes = decode_curve_state(X, 1.2, layout, None, None)
        return modes, decoded_u-u, p[0]-fixed, modes.v-v, modes.w-w, modes.kappa-k
    modes, *errors = jax.jit(jax.vmap(decode))(jnp.array([.2, 0., -.2]))
    assert all(jnp.all(e == 0) for e in errors)
    np.testing.assert_array_equal(modes.oscillatory, [True, False, False])
    np.testing.assert_array_equal(modes.q2[1:], jnp.zeros((2, 2)))
    omega, oscillatory = jax.jit(hopf_curve_frequency)(jnp.array([.2, 0., -.2]))
    np.testing.assert_allclose(omega, modes.omega)
    np.testing.assert_array_equal(oscillatory, modes.oscillatory)


def test_bt_singular_gauge_reports_failure_without_reanchoring():
    problem = seed()
    # A zero gauge row makes the local corrector Jacobian singular.
    args = eqx.tree_at(lambda a: a.v_anchor, problem.args, jnp.zeros(2))
    problem = eqx.tree_at(lambda p: p.args, problem, args)
    result = jc.continuation(problem, p_span=(1.3, 1.1), alg=jc.Natural(),
        settings=jc.ContinuationPar(ds=.01, adaptive=False, max_steps=2,
                                   compute_stability=False))
    assert result.status in ("nonfinite", "stalled", "seed_failed")
    assert result.branch.states.shape[0] <= 1
    np.testing.assert_array_equal(problem.args.v_anchor, jnp.zeros(2))


def scan_data(kappas):
    from jaxcont.bifurcations.event_scan import EventScanInput
    from jaxcont.bifurcations.curve_layout import CurveLayout
    k = jnp.asarray(kappas)
    x = -k/2
    states = jnp.stack([.7+x, jnp.full_like(k, -.4), .3-x*x,
                       jnp.ones_like(k), jnp.zeros_like(k), jnp.zeros_like(k),
                       jnp.ones_like(k), k], axis=1)
    return EventScanInput(1.2-x, states, None, None, None,
                          jnp.ones(k.shape, bool), CurveLayout('hopf', 'bt', 2, 1))


def test_bt_kappa_event_once_and_validated():
    from jaxcont.bifurcations.codim2_events import BogdanovTakens
    event = BogdanovTakens(shifted, curve='hopf')
    from jaxcont.bifurcations.events import BranchPoint
    data = scan_data([.2, 0., -.2])
    points = [BranchPoint(q, X, None, None, None, data.curve_layout)
              for q, X in zip(data.params, data.states)]
    assert event.scan(points) == [(0, 1, None)]
    # Kappa owns BT brackets: a tiny positive kappa without a crossing is
    # neither a candidate nor an omega-style curve_terminated stall.
    stall = scan_data([.2, .1, 1e-12])
    stall_points = [BranchPoint(q, X, None, None, None, stall.curve_layout)
                    for q, X in zip(stall.params, stall.states)]
    assert event.scan(stall_points) == []
    c = event.scan_arrays(stall, None)
    assert not jnp.any(c.bracket_mask) and not jnp.any(c.terminal_mask)
    result = jc.continuation(seed(), p_span=(1.3, 1.1), alg=jc.Natural(),
        settings=jc.ContinuationPar(ds=.01, adaptive=False, max_steps=20,
                                   compute_stability=False), events=[event])
    assert len(result.events) == 1
    hit = result.events[0]
    assert hit.info['converged']
    assert jnp.max(jnp.abs(hit.info['p']-jnp.array([.3, 1.2]))) < 100*default_tol(hit.u.dtype)
    assert not any(h.info.get('reason') == 'curve_terminated' for h in result.rejected_events)


def test_bt_chart_masks_hopf_only_events():
    from jaxcont.bifurcations.codim2_events import GeneralizedHopf, ZeroHopf, DoubleHopf
    from jaxcont.bifurcations.curve_layout import CurveLayout
    data = scan_data([.2, 0., -.2])
    def extra_modes(u, p, args):
        beta = p[1]-1.15
        return jnp.concatenate([shifted(u[:2], p, args),
            jnp.array([beta*u[2], beta*u[3]-2*u[4], 2*u[3]+beta*u[4]])])
    n = 5
    states = jnp.zeros((3, 3*n+2), dtype=data.states.dtype)
    states = states.at[:, :2].set(data.states[:, :2]).at[:, n].set(data.states[:, 2])
    states = states.at[:, n+1].set(1).at[:, 2*n+2].set(1).at[:, -1].set(data.states[:, -1])
    data = replace(data, states=states, curve_layout=CurveLayout('hopf', 'bt', n, 1))
    for cls in (GeneralizedHopf, ZeroHopf, DoubleHopf):
        c = jax.jit(lambda d: cls(extra_modes).scan_arrays(d, None))(data)
        assert jnp.all(jnp.isfinite(c.values))
        assert not jnp.any(c.eligible[1:])
        assert not jnp.any(c.bracket_mask)


def test_bt_chart_jit_vmap_scan_gradient():
    from jaxcont.bifurcations.codim2_events import BogdanovTakens, GeneralizedHopf
    def solve(a):
        def f(u, p, args):
            return shifted(u, p, None).at[1].add(.3-args)
        problem = hopf_curve_problem(f, jnp.array([.6, -.4]),
            jnp.array([a-.01, 1.3]), args=a, formulation='bt')
        result = jc.continuation(problem, p_span=(1.3, 1.1), alg=jc.Natural(),
            settings=jc.ContinuationPar(ds=.01, adaptive=False, max_steps=20,
                compute_stability=False, loop='scan'),
            events=[BogdanovTakens(f, curve='hopf', args=a), GeneralizedHopf(f, args=a)])
        return jnp.mean(result.branch.states[:, 2])
    values = jax.jit(jax.vmap(solve))(jnp.array([.3, .4]))
    assert jnp.all(jnp.isfinite(values))
    grad = jax.jit(jax.grad(solve))(jnp.asarray(.3))
    assert abs(grad-1) < 100*default_tol(grad.dtype)


def test_bt_mixed_kappa_coefficient_grad():
    from jaxcont.bifurcations.codim2_events import GeneralizedHopf
    from jaxcont.bifurcations.hopf_normal_form import lyapunov_coefficient
    def coefficient(a, k):
        def f(z, p, args):
            hopf = jnp.sqrt(.2)*jnp.array([-z[1], z[0]])-args*jnp.dot(z,z)*z
            return jnp.where(k > 0, hopf, shifted(z, p, None))
        d = scan_data(jnp.reshape(k, (1,)))
        d = replace(d, states=d.states.at[0, :2].set(jnp.where(k > 0, jnp.zeros(2), d.states[0,:2])).at[0, 6].set(jnp.where(k > 0, -1/jnp.sqrt(.2), 1.)))
        return GeneralizedHopf(f, args=a).scan_arrays(d, None).values[0,0]
    k = jnp.array([.2, 0., -.2])
    a = jnp.full((3,), .7)
    values = jax.jit(jax.vmap(coefficient))(a, k)
    grads = jax.jit(jax.grad(lambda aa: jnp.sum(jax.vmap(coefficient)(aa, k))))(a)
    def reference(a):
        def f(z,p,args):
            return jnp.sqrt(.2)*jnp.array([-z[1],z[0]])-args*jnp.dot(z,z)*z
        q = jnp.array([1., -1j])
        q = q/jnp.linalg.norm(q)
        return lyapunov_coefficient(f,jnp.zeros(2),jnp.zeros(2),q.real,q.imag,jnp.sqrt(.2),a)
    tol = 100*default_tol(a.dtype)
    assert abs(values[0]-reference(a[0])) < tol
    assert abs(grads[0]-jax.grad(reference)(a[0])) < tol
    assert jnp.all(jnp.isfinite(grads))
    np.testing.assert_array_equal(values[1:], 0)
    np.testing.assert_array_equal(grads[1:], 0)
    eligible = jax.vmap(lambda kk: GeneralizedHopf(shifted).scan_arrays(
        scan_data(jnp.reshape(kk, (1,))), None).eligible[0, 0])(k)
    np.testing.assert_array_equal(eligible, [True, False, False])


def test_bt_plot_neutral_segment_dashed():
    import matplotlib.pyplot as plt
    from types import SimpleNamespace
    from jaxcont.viz.two_parameter import plot_two_parameter_diagram
    d = scan_data([.2, 0., -.2])
    branch = SimpleNamespace(states=d.states, params=d.params, valid=d.valid, curve_layout=d.curve_layout)
    ax = plot_two_parameter_diagram([(SimpleNamespace(branch=branch, events=[]), 'hopf')])
    assert any('oscillatory' in line.get_label() for line in ax.lines)
    assert any('neutral' in line.get_label() and line.get_linestyle() == '--' for line in ax.lines)
    plt.close(ax.figure)
    # Without a sampled zero the dashed segment still joins the solid one.
    d = scan_data([.2, .1, -.1, -.2])
    branch = SimpleNamespace(states=d.states, params=d.params, valid=d.valid, curve_layout=d.curve_layout)
    ax = plot_two_parameter_diagram([(SimpleNamespace(branch=branch, events=[]), 'hopf')])
    solid, dashed = (np.isfinite(line.get_xdata()) for line in ax.lines[:2])
    np.testing.assert_array_equal(solid, [True, True, False, False])
    np.testing.assert_array_equal(dashed, [False, True, True, True])
    plt.close(ax.figure)


def test_bt_domain_restricted_coefficient_grad():
    from jaxcont.bifurcations.codim2_events import GeneralizedHopf
    from jaxcont.bifurcations.hopf_normal_form import lyapunov_coefficient

    def rhs(z, p, a):
        center = .7+(1.2-p[1])
        cubic = a*(jnp.log(z[0])-jnp.log(center))**3
        return shifted(z, p, None).at[1].add(cubic)

    k = jnp.array([.2, 0., -.2])
    a = jnp.full((3,), .7)

    def coefficient(strength, kappa):
        data = scan_data(jnp.reshape(kappa, (1,)))
        return GeneralizedHopf(rhs, args=strength).scan_arrays(data, None).values[0, 0]

    values = jax.jit(jax.vmap(coefficient))(a, k)
    gradients = jax.jit(jax.grad(
        lambda strengths: jnp.sum(jax.vmap(coefficient)(strengths, k))))(a)

    def reference(strength):
        data = scan_data(k[:1])
        q = jnp.array([1., 1j*jnp.sqrt(k[0])])
        q = q/jnp.linalg.norm(q)
        return lyapunov_coefficient(rhs, data.states[0, :2],
            jnp.array([data.states[0, 2], data.params[0]]),
            q.real, q.imag, jnp.sqrt(k[0]), strength)

    tol = 100*default_tol(a.dtype)
    assert abs(values[0]-reference(a[0])) < tol
    assert abs(gradients[0]-jax.grad(reference)(a[0])) < tol
    assert jnp.all(jnp.isfinite(gradients))
    np.testing.assert_array_equal(values[1:], 0)
    np.testing.assert_array_equal(gradients[1:], 0)
