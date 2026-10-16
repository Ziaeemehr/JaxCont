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

    problem = hopf_curve_problem(f, jnp.array([0.5, -0.3]), jnp.array([1.75, 0.5]))
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
