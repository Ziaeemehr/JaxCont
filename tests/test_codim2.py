"""
Tests for jaxcont.bifurcations.codim2: direct codim-2 point solvers. See
docs/superpowers/specs/2026-08-05-codim2-direct-solvers-design.md.

Every system here is deliberately SHIFTED so its codim-2 point is at a
non-trivial location. The textbook normal forms all put their codim-2 point
at u=0, p=(0,0), which means a stub returning zeros would pass them -- they
have no discriminating power on their own.
"""

import jax
import jax.numpy as jnp

from jaxcont.bifurcations.codim2 import (
    cusp_point, cusp_parameters,
    bogdanov_takens_point, bogdanov_takens_parameters,
    generalized_hopf_point, generalized_hopf_parameters,
)
from jaxcont.bifurcations.hopf_normal_form import lyapunov_coefficient


def _cusp_shifted(u, p, args):
    # x' = b1 + b2*xi + k*xi^3   with   xi = x - 2, b1 = p0 - 1, b2 = p1 + 4
    # Cusp of x' = b1 + b2*x + k*x^3 is at x=0, (b1,b2)=(0,0), so this one
    # sits at u*=(2,), p*=(1,-4) -- a non-trivial location.
    k = 1.0 if args is None else args
    xi = u[0] - 2.0
    b1 = p[0] - 1.0
    b2 = p[1] + 4.0
    return jnp.array([b1 + b2 * xi + k * xi**3])


def test_cusp_point_recovers_exact_shifted_cusp():
    u, p, v, ok = cusp_point(
        _cusp_shifted, jnp.array([2.2]), jnp.array([0.8, -3.7]),
    )
    assert bool(ok)
    assert jnp.allclose(u, jnp.array([2.0]), atol=1e-4)
    assert jnp.allclose(p, jnp.array([1.0, -4.0]), atol=1e-4)
    assert jnp.isclose(float(jnp.linalg.norm(v)), 1.0, atol=1e-5)


def test_cusp_point_does_not_merely_return_its_guess():
    # Guards against a trivial implementation that echoes the seed back.
    guess_u = jnp.array([2.2])
    guess_p = jnp.array([0.8, -3.7])
    u, p, _, ok = cusp_point(_cusp_shifted, guess_u, guess_p)
    assert bool(ok)
    assert float(jnp.max(jnp.abs(p - guess_p))) > 1e-2


def test_cusp_parameters_returns_bare_parameter_array():
    p = cusp_parameters(_cusp_shifted, jnp.array([2.2]), jnp.array([0.8, -3.7]))
    assert p.shape == (2,)
    assert jnp.allclose(p, jnp.array([1.0, -4.0]), atol=1e-4)


def test_cusp_reports_not_converged_for_a_hopeless_guess():
    # A system with no cusp anywhere: x' = 1 + x^2 has no equilibrium at all.
    def no_cusp(u, p, args):
        return jnp.array([1.0 + u[0] ** 2 + 0.0 * p[0]])

    _, _, _, ok = cusp_point(no_cusp, jnp.array([5.0]), jnp.array([0.0, 0.0]))
    assert not bool(ok)


def test_cusp_parameters_grad_matches_finite_difference():
    # Move the cusp with an args scalar and check the gradient of its
    # location. This is the headline claim of the whole feature.
    def cusp_moving(u, p, shift):
        xi = u[0] - 2.0
        b1 = p[0] - 1.0 - shift
        b2 = p[1] + 4.0
        return jnp.array([b1 + b2 * xi + xi**3])

    def p0_star(shift):
        return cusp_parameters(
            cusp_moving, jnp.array([2.1]), jnp.array([0.9, -3.8]), shift
        )[0]

    g = jax.grad(p0_star)(0.1)
    h = 1e-3
    fd = (p0_star(0.1 + h) - p0_star(0.1 - h)) / (2 * h)
    assert jnp.isfinite(g)
    assert jnp.isclose(float(g), float(fd), atol=1e-3)


def test_cusp_agrees_with_the_codim1_fold_solver_there():
    # Cross-check against the existing scalar-p fold solver: a cusp IS a
    # fold, so freezing the second parameter at its cusp value and running
    # fold_point must land on the same (u, p0).
    from jaxcont.bifurcations.fold_solve import fold_point

    u_c, p_c, _, ok = cusp_point(
        _cusp_shifted, jnp.array([2.2]), jnp.array([0.8, -3.7]),
    )
    assert bool(ok)

    def f_scalar(u, p, args):
        return _cusp_shifted(u, jnp.array([p, p_c[1]]), args)

    u_f, p_f, _ = fold_point(f_scalar, jnp.array([2.1]), float(p_c[0]) + 0.05)
    assert jnp.allclose(u_f, u_c, atol=1e-3)
    assert jnp.isclose(float(p_f), float(p_c[0]), atol=1e-3)


def _bt_shifted(u, p, args):
    # x' = y ; y' = b1 + b2*x + x^2 + x*y  has its BT at u=0, p=(0,0).
    # Shifted by x -> X-5, y -> Y-2, b1 -> B1-3, b2 -> B2+1, so the BT sits
    # at u*=(5,2), p*=(3,-1).
    X, Y = u[0], u[1]
    x, y = X - 5.0, Y - 2.0
    b1 = p[0] - 3.0
    b2 = p[1] + 1.0
    return jnp.array([y, b1 + b2 * x + x**2 + x * y])


def test_bogdanov_takens_point_recovers_exact_shifted_bt():
    u, p, v0, v1, ok = bogdanov_takens_point(
        _bt_shifted, jnp.array([5.3, 1.7]), jnp.array([2.6, -0.8]),
    )
    assert bool(ok)
    assert jnp.allclose(u, jnp.array([5.0, 2.0]), atol=1e-4)
    assert jnp.allclose(p, jnp.array([3.0, -1.0]), atol=1e-4)
    assert jnp.isclose(float(jnp.linalg.norm(v0)), 1.0, atol=1e-5)


def test_bogdanov_takens_jacobian_has_a_genuine_jordan_block():
    # The defining property: f_u has a DOUBLE zero eigenvalue with only one
    # eigenvector, i.e. f_u @ v1 == v0 and v0 . v1 == 0.
    u, p, v0, v1, ok = bogdanov_takens_point(
        _bt_shifted, jnp.array([5.3, 1.7]), jnp.array([2.6, -0.8]),
    )
    assert bool(ok)
    jac = jax.jacfwd(lambda uu: _bt_shifted(uu, p, None))(u)
    assert jnp.allclose(jac @ v0, jnp.zeros(2), atol=1e-4)
    assert jnp.allclose(jac @ v1, v0, atol=1e-4)
    assert jnp.isclose(float(jnp.dot(v0, v1)), 0.0, atol=1e-5)


def test_bogdanov_takens_parameters_grad_matches_finite_difference():
    def bt_moving(u, p, shift):
        X, Y = u[0], u[1]
        x, y = X - 5.0, Y - 2.0
        b1 = p[0] - 3.0 - shift
        b2 = p[1] + 1.0
        return jnp.array([y, b1 + b2 * x + x**2 + x * y])

    def p0_star(shift):
        return bogdanov_takens_parameters(
            bt_moving, jnp.array([5.2, 1.8]), jnp.array([2.7, -0.9]), shift
        )[0]

    g = jax.grad(p0_star)(0.1)
    h = 1e-3
    fd = (p0_star(0.1 + h) - p0_star(0.1 - h)) / (2 * h)
    assert jnp.isfinite(g)
    assert jnp.isclose(float(g), float(fd), atol=1e-3)


def _gh_shifted(u, p, args):
    # Bautin normal form  r' = r*(b1 + b2*r^2 - r^4),  theta' = 1,  in
    # Cartesian coordinates. Hopf at b1=0; l1 is proportional to b2, so the
    # generalized Hopf is at (b1,b2)=(0,0). Shifted to p*=(2,-3).
    x, y = u[0], u[1]
    b1 = p[0] - 2.0
    b2 = p[1] + 3.0
    r2 = x**2 + y**2
    g = b1 + b2 * r2 - r2**2
    return jnp.array([-y + x * g, x + y * g])


def test_generalized_hopf_point_recovers_exact_shifted_gh():
    u, p, q1, q2, omega, ok = generalized_hopf_point(
        _gh_shifted, jnp.array([0.02, -0.03]), jnp.array([2.05, -2.90]),
    )
    assert bool(ok)
    assert jnp.allclose(u, jnp.zeros(2), atol=1e-4)
    assert jnp.allclose(p, jnp.array([2.0, -3.0]), atol=1e-4)
    assert jnp.isclose(float(omega), 1.0, atol=1e-4)


def test_generalized_hopf_omega_is_normalized_positive_from_either_seed():
    # The Hopf block is invariant under (omega, q2) -> (-omega, -q2), so the
    # raw solve can land on either sign. This is the regression test for
    # that -- without it the bug reappears invisibly.
    for seed_u in (jnp.array([0.02, -0.03]), jnp.array([-0.02, 0.03])):
        _, _, _, _, omega, ok = generalized_hopf_point(
            _gh_shifted, seed_u, jnp.array([2.05, -2.90]),
        )
        assert bool(ok)
        assert float(omega) > 0.0


def test_generalized_hopf_has_vanishing_lyapunov_coefficient():
    # The defining condition: l1 == 0 at the returned point.
    u, p, q1, q2, omega, ok = generalized_hopf_point(
        _gh_shifted, jnp.array([0.02, -0.03]), jnp.array([2.05, -2.90]),
    )
    assert bool(ok)
    l1 = lyapunov_coefficient(_gh_shifted, u, p, q1, q2, omega, None)
    assert abs(float(l1)) < 1e-4


def test_generalized_hopf_parameters_grad_matches_finite_difference():
    def gh_moving(u, p, shift):
        x, y = u[0], u[1]
        b1 = p[0] - 2.0 - shift
        b2 = p[1] + 3.0
        r2 = x**2 + y**2
        g = b1 + b2 * r2 - r2**2
        return jnp.array([-y + x * g, x + y * g])

    def p0_star(shift):
        return generalized_hopf_parameters(
            gh_moving, jnp.array([0.02, -0.03]), jnp.array([2.05, -2.90]), shift
        )[0]

    g = jax.grad(p0_star)(0.05)
    h = 1e-3
    fd = (p0_star(0.05 + h) - p0_star(0.05 - h)) / (2 * h)
    assert jnp.isfinite(g)
    assert jnp.isclose(float(g), float(fd), atol=1e-3)


def test_generalized_hopf_agrees_with_the_codim1_hopf_solver_there():
    # Cross-check against the existing scalar-p Hopf solver: a GH IS a Hopf
    # point, so freezing the second parameter at its GH value and running
    # hopf_point must land on the same p0 and frequency.
    from jaxcont.bifurcations.hopf_normal_form import hopf_point

    _, p_g, _, _, om_g, ok = generalized_hopf_point(
        _gh_shifted, jnp.array([0.02, -0.03]), jnp.array([2.05, -2.90]),
    )
    assert bool(ok)

    def f_scalar(u, p, args):
        return _gh_shifted(u, jnp.array([p, p_g[1]]), args)

    _, p_h, _, _, om_h = hopf_point(
        f_scalar, jnp.array([0.01, -0.01]), float(p_g[0]) + 0.02, tol=1e-6,
    )
    assert jnp.isclose(float(p_h), float(p_g[0]), atol=1e-3)
    assert jnp.isclose(abs(float(om_h)), float(om_g), atol=1e-3)
