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

from jaxcont.bifurcations.codim2 import cusp_point, cusp_parameters


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
