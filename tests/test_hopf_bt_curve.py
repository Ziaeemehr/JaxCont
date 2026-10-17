"""Real Jordan-chain Hopf chart through a shifted BT point."""

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
