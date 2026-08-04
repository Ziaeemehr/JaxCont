"""
Tests for jaxcont.bifurcations.hopf_normal_form: the differentiable Hopf-
point solver (hopf_point/hopf_parameter) and the first Lyapunov coefficient
(lyapunov_coefficient). See docs/superpowers/specs/2026-08-04-hopf-normal-
form-design.md.
"""

import jax
import jax.numpy as jnp

from jaxcont.bifurcations.hopf_normal_form import hopf_point, hopf_parameter, lyapunov_coefficient


def _textbook_hopf(u, p, args):
    # Standard supercritical-Hopf textbook example (Kuznetsov Sec 3.2/3.4):
    # exact Hopf at u=(0,0), p=0, omega0=1, l1=-1.
    x, y = u[0], u[1]
    r2 = x**2 + y**2
    return jnp.array([-y + x * (p - r2), x + y * (p - r2)])


def test_hopf_point_recovers_exact_hopf_of_textbook_example():
    u, p, q1, q2, omega0 = hopf_point(
        _textbook_hopf, jnp.zeros(2), 0.05, tol=1e-10, max_iter=50,
    )
    assert jnp.allclose(u, jnp.zeros(2), atol=1e-5)
    assert jnp.isclose(float(p), 0.0, atol=1e-5)
    assert jnp.isclose(float(omega0), 1.0, atol=1e-5)
    # q1+i*q2 is a unit vector (G4's normalization).
    assert jnp.isclose(jnp.dot(q1, q1) + jnp.dot(q2, q2), 1.0, atol=1e-5)


def _hopf_with_param_shift(u, p, shift):
    x, y = u[0], u[1]
    r2 = x**2 + y**2
    p_eff = p - shift
    return jnp.array([-y + x * (p_eff - r2), x + y * (p_eff - r2)])


def test_hopf_parameter_grad_matches_finite_difference():
    # p*(shift) = shift exactly for this family, so d(hopf_parameter)/d(shift) = 1
    # exactly -- a strong, non-trivial gradient check (not a coincidental 0).
    def p_star(shift):
        return hopf_parameter(
            _hopf_with_param_shift, jnp.zeros(2), 0.05, shift, tol=1e-10, max_iter=50,
        )

    grad = jax.grad(p_star)(0.1)
    eps = 1e-4
    fd = (p_star(0.1 + eps) - p_star(0.1 - eps)) / (2 * eps)
    assert jnp.isclose(grad, fd, atol=1e-3)
    assert jnp.isclose(grad, 1.0, atol=1e-3)


def test_lyapunov_coefficient_matches_exact_textbook_value():
    u, p, q1, q2, omega0 = hopf_point(
        _textbook_hopf, jnp.zeros(2), 0.05, tol=1e-10, max_iter=50,
    )
    l1 = lyapunov_coefficient(_textbook_hopf, u, p, q1, q2, omega0)
    assert jnp.isclose(float(l1), -1.0, atol=1e-4)


def test_lyapunov_coefficient_scales_linearly_with_cubic_coefficient():
    # dr/dt = mu*r + k*r^3 in polar form has l1 = k exactly by definition of
    # normal form -- scaling the cubic term by k must scale l1 by exactly k.
    def hopf_scaled(u, p, k):
        x, y = u[0], u[1]
        r2 = x**2 + y**2
        return jnp.array([-y + x * (p - k * r2), x + y * (p - k * r2)])

    for k in (1.0, 2.0, 0.5, -1.0):
        u, p, q1, q2, omega0 = hopf_point(
            hopf_scaled, jnp.zeros(2), 0.05, k, tol=1e-10, max_iter=50,
        )
        l1 = lyapunov_coefficient(hopf_scaled, u, p, q1, q2, omega0, k)
        assert jnp.isclose(float(l1), -k, atol=1e-4)


def test_lyapunov_coefficient_grad_matches_finite_difference():
    # l1(scale) = -scale exactly (see the scaling test above), so
    # d(l1)/d(scale) = -1 exactly -- a strong, non-trivial gradient check.
    def hopf_scaled(u, p, scale):
        x, y = u[0], u[1]
        r2 = x**2 + y**2
        return jnp.array([-y + x * (p - scale * r2), x + y * (p - scale * r2)])

    def l1_of_scale(scale):
        u, p, q1, q2, omega0 = hopf_point(
            hopf_scaled, jnp.zeros(2), 0.05, scale, tol=1e-10, max_iter=50,
        )
        return lyapunov_coefficient(hopf_scaled, u, p, q1, q2, omega0, scale)

    grad = jax.grad(l1_of_scale)(1.0)
    eps = 1e-4
    fd = (l1_of_scale(1.0 + eps) - l1_of_scale(1.0 - eps)) / (2 * eps)
    assert jnp.isclose(grad, fd, atol=1e-2)
    assert jnp.isclose(grad, -1.0, atol=1e-2)
