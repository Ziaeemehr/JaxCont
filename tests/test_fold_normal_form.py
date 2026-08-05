"""
Tests for jaxcont.bifurcations.fold_normal_form: the fold's quadratic
normal-form coefficient a = 1/2 * <w, B(v,v)>. See
docs/superpowers/specs/2026-08-05-codim2-direct-solvers-design.md.
"""

import jax
import jax.numpy as jnp

from jaxcont.bifurcations.fold_normal_form import fold_coefficient


def _fold_1d(u, p, args):
    # x' = b1 + b2*x + k*x^2, with k defaulting to 1.
    # At u=0, p=(0,0): f=0 and f_u=0, so this is a fold with v=w=1 and
    # B(v,v) = f_uu = 2k, giving the exact coefficient a = k.
    k = 1.0 if args is None else args
    x = u[0]
    b1, b2 = p[0], p[1]
    return jnp.array([b1 + b2 * x + k * x**2])


def test_fold_coefficient_matches_exact_value_of_quadratic_normal_form():
    a = fold_coefficient(_fold_1d, jnp.zeros(1), jnp.zeros(2), jnp.ones(1))
    assert jnp.isclose(float(a), 1.0, atol=1e-5)


def test_fold_coefficient_flips_sign_with_the_quadratic_term():
    a = fold_coefficient(_fold_1d, jnp.zeros(1), jnp.zeros(2), jnp.ones(1), -1.0)
    assert jnp.isclose(float(a), -1.0, atol=1e-5)


def test_fold_coefficient_scales_linearly_with_the_quadratic_term():
    a2 = fold_coefficient(_fold_1d, jnp.zeros(1), jnp.zeros(2), jnp.ones(1), 2.0)
    a5 = fold_coefficient(_fold_1d, jnp.zeros(1), jnp.zeros(2), jnp.ones(1), 5.0)
    assert jnp.isclose(float(a2), 2.0, atol=1e-5)
    assert jnp.isclose(float(a5), 5.0, atol=1e-5)


def _fold_2d(u, p, args):
    # The same fold plus a decoupled stable direction, so the left null
    # vector w is a genuine solve rather than a 1x1 triviality.
    # v = w = (1, 0); a = k as above.
    k = 1.0 if args is None else args
    x, y = u[0], u[1]
    b1, b2 = p[0], p[1]
    return jnp.array([b1 + b2 * x + k * x**2, -y])


def test_fold_coefficient_handles_multidimensional_left_null_vector():
    a = fold_coefficient(
        _fold_2d, jnp.zeros(2), jnp.zeros(2), jnp.array([1.0, 0.0]), 3.0
    )
    assert jnp.isclose(float(a), 3.0, atol=1e-5)


def test_fold_coefficient_grad_matches_finite_difference():
    # a(k) = k exactly, so da/dk == 1; check autodiff agrees with a central
    # difference rather than trusting the closed form alone.
    def a_of_k(k):
        return fold_coefficient(_fold_2d, jnp.zeros(2), jnp.zeros(2),
                                jnp.array([1.0, 0.0]), k)

    g = jax.grad(a_of_k)(2.0)
    h = 1e-3
    fd = (a_of_k(2.0 + h) - a_of_k(2.0 - h)) / (2 * h)
    assert jnp.isfinite(g)
    assert jnp.isclose(float(g), float(fd), atol=1e-4)
