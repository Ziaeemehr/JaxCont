"""
Domain-independent tests for jaxcont.solvers.implicit.differentiable_root,
extracted from bifurcations/fold_solve.py's implicit-diff Newton solver. See
docs/superpowers/specs/2026-07-23-differentiable-root-primitive-design.md.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxcont.solvers.implicit import differentiable_root


def _G_scalar(x, theta):
    # root: x* = sqrt(theta)
    return x**2 - theta


def test_root_matches_analytic():
    x0 = jnp.array([1.0])
    theta = jnp.array(4.0)
    x_star = differentiable_root(_G_scalar, x0, theta)
    assert float(x_star[0]) == pytest.approx(2.0, abs=1e-6)


def test_reverse_mode_grad_matches_analytic():
    x0 = jnp.array([1.0])

    def root(theta):
        return differentiable_root(_G_scalar, x0, theta)[0]

    for theta_val in (4.0, 9.0):
        theta = jnp.array(theta_val)
        g = jax.grad(root)(theta)
        expected = 1.0 / (2.0 * jnp.sqrt(theta))
        assert float(g) == pytest.approx(float(expected), abs=1e-5)


def _G_vector(x, theta):
    # x^2 - a*x + b = 0 ; theta = [a, b]
    a, b = theta[0], theta[1]
    return x**2 - a * x + b


def test_vector_theta_jacobian_matches_analytic():
    x0 = jnp.array([1.8])  # seeds convergence to the larger root
    theta = jnp.array([3.0, 2.0])  # roots at x=1, x=2

    def root(theta):
        return differentiable_root(_G_vector, x0, theta)[0]

    x_star = root(theta)
    assert float(x_star) == pytest.approx(2.0, abs=1e-6)

    J = jax.jacobian(root)(theta)
    # G_x = 2x - a ; G_a = -x ; G_b = 1 ; dx/da = x/(2x-a) ; dx/db = -1/(2x-a)
    Gx = 2.0 * float(x_star) - float(theta[0])
    expected = jnp.array([float(x_star) / Gx, -1.0 / Gx])
    assert jnp.allclose(J, expected, atol=1e-5)


def test_callable_seed_theta_dependent():
    # x0 depends on theta itself (mirrors fold_solve.py's SVD-based seed).
    # A precomputed-array x0 built the same way would leak a tracer under
    # jax.grad; the callable form computes it inside the traced primal.
    def x0_from_theta(theta):
        return jnp.array([jnp.sqrt(theta) * 0.5])

    def root(theta):
        return differentiable_root(_G_scalar, x0_from_theta, theta)[0]

    for theta_val in (4.0, 9.0):
        theta = jnp.array(theta_val)
        x_star = root(theta)
        assert float(x_star) == pytest.approx(float(jnp.sqrt(theta)), abs=1e-6)
        g = jax.grad(root)(theta)
        expected = 1.0 / (2.0 * jnp.sqrt(theta))
        assert float(g) == pytest.approx(float(expected), abs=1e-5)
