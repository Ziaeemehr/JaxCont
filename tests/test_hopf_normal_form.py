"""
Tests for jaxcont.bifurcations.hopf_normal_form: the differentiable Hopf-
point solver (hopf_point/hopf_parameter) and the first Lyapunov coefficient
(lyapunov_coefficient). See docs/superpowers/specs/2026-08-04-hopf-normal-
form-design.md.
"""

import jax
import jax.numpy as jnp

from jaxcont.bifurcations.hopf_normal_form import (
    _seed,
    hopf_parameter,
    hopf_point,
    lyapunov_coefficient,
)


def _textbook_hopf(u, p, args):
    # Standard supercritical-Hopf textbook example (Kuznetsov Sec 3.2/3.4):
    # exact Hopf at u=(0,0), p=0, omega0=1, l1=-1.
    x, y = u[0], u[1]
    r2 = x**2 + y**2
    return jnp.array([-y + x * (p - r2), x + y * (p - r2)])


def test_hopf_point_recovers_exact_hopf_of_textbook_example():
    u, p, q1, q2, omega0, _converged = hopf_point(
        _textbook_hopf, jnp.zeros(2), 0.05, tol=1e-10, max_iter=50,
    )
    assert jnp.allclose(u, jnp.zeros(2), atol=1e-5)
    assert jnp.isclose(float(p), 0.0, atol=1e-5)
    assert jnp.isclose(float(omega0), 1.0, atol=1e-5)
    # q1+i*q2 is a unit vector (G4's normalization).
    assert jnp.isclose(jnp.dot(q1, q1) + jnp.dot(q2, q2), 1.0, atol=1e-5)


def _hopf_plus_slow_real_mode(u, p, args):
    # 3-D system: a Hopf pair near 0.05 +/- 1i in (x, y) (same linear part as
    # _textbook_hopf) plus a decoupled slow real mode z with eigenvalue
    # -0.001 -- much closer to the imaginary axis (in |Re|) than the Hopf
    # pair's own real part (0.05 at this p). Regression for Important #1: a
    # bare argmin over |Re(eigenvalue)| across ALL eigenvalues (the old
    # _seed behavior) would select the real z-mode (|Re|=0.001) instead of
    # the genuinely complex Hopf pair (|Re|=0.05), handing
    # differentiable_root a seed with omega=0 and a purely real eigenvector
    # -- garbage for a Hopf extended-system Newton solve.
    x, y, z = u[0], u[1], u[2]
    r2 = x**2 + y**2
    return jnp.array([-y + x * (p - r2), x + y * (p - r2), -0.001 * z])


def test_seed_masks_out_near_zero_real_eigenvalue():
    # Direct unit test of the _seed heuristic itself: at u=0, the jacobian's
    # eigenvalues are exactly p +/- 1i and -0.001 (independent of the
    # nonlinear terms, which vanish at the origin). The seed must pick the
    # complex pair (omega ~ 1, Im > 0 per the seed's orientation
    # convention), never the near-zero real mode (which would give omega=0).
    q1, q2, omega = _seed(_hopf_plus_slow_real_mode, jnp.zeros(3), 0.05, None, 3)
    assert jnp.isclose(omega, 1.0, atol=1e-6)
    # The real z-component must not be excited by a seed drawn from the
    # complex (x, y) pair's eigenvector.
    assert jnp.isclose(q1[2], 0.0, atol=1e-8)
    assert jnp.isclose(q2[2], 0.0, atol=1e-8)


def test_hopf_point_end_to_end_ignores_slow_real_mode():
    # End-to-end version of the same regression: hopf_point on the 3-D
    # system must converge to the same Hopf point as the pure 2-D textbook
    # example (the z-mode is decoupled and irrelevant to the Hopf point),
    # not diverge/NaN from a bad real-eigenvalue seed.
    u, p, _q1, _q2, omega0, _converged = hopf_point(
        _hopf_plus_slow_real_mode, jnp.array([0.0, 0.0, 0.0]), 0.05,
        tol=1e-10, max_iter=50,
    )
    assert jnp.all(jnp.isfinite(u))
    assert jnp.isfinite(p) and jnp.isfinite(omega0)
    assert jnp.allclose(u, jnp.zeros(3), atol=1e-5)
    assert jnp.isclose(float(p), 0.0, atol=1e-5)
    assert jnp.isclose(float(omega0), 1.0, atol=1e-5)


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
    u, p, q1, q2, omega0, _converged = hopf_point(
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
        u, p, q1, q2, omega0, _converged = hopf_point(
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
        u, p, q1, q2, omega0, _converged = hopf_point(
            hopf_scaled, jnp.zeros(2), 0.05, scale, tol=1e-10, max_iter=50,
        )
        return lyapunov_coefficient(hopf_scaled, u, p, q1, q2, omega0, scale)

    grad = jax.grad(l1_of_scale)(1.0)
    eps = 1e-4
    fd = (l1_of_scale(1.0 + eps) - l1_of_scale(1.0 - eps)) / (2 * eps)
    assert jnp.isclose(grad, fd, atol=1e-2)
    assert jnp.isclose(grad, -1.0, atol=1e-2)


def test_lyapunov_coefficient_matches_bifurcationkit_jl_independent_run():
    # Independent cross-check against BifurcationKit.jl v0.5.2's own hopf
    # normal form (examples/BifurcationKit/04_hopf_normal_form.jl, run
    # 2026-08-04): Hopf at p=5.392241290723402e-7, omega0=1.0, BK's own
    # normal-form b=-1.7500010784482587, giving l1=b/2=-0.8750005392241293.
    def f(u, p, args):
        x, y = u[0], u[1]
        return jnp.array([
            p * x - y + x**2 - x**3 - x * y**2,
            x + p * y + x * y - y**3,
        ])

    u, p, q1, q2, omega0, _converged = hopf_point(f, jnp.zeros(2), 0.05, tol=1e-10, max_iter=50)
    l1 = lyapunov_coefficient(f, u, p, q1, q2, omega0)

    # This literal is the value actually observed running this repo's own
    # JaxCont side (not BifurcationKit.jl's) when the cross-check was
    # performed 2026-08-04; the design spec and plan docs record
    # -0.8750005392241294 (differs only in the last digit, from floating-
    # point summation order between the two independent implementations --
    # numerically irrelevant given the atol=1e-4 comparison below). Kept as
    # the value actually observed here, not edited to match the docs.
    bk_l1_reference = -0.8750005392241293
    assert jnp.isclose(float(omega0), 1.0, atol=1e-6)
    assert jnp.isclose(float(l1), bk_l1_reference, atol=1e-4)


def test_hopf_normal_form_functions_are_exported_at_top_level():
    import jaxcont as jc

    assert jc.hopf_point is hopf_point
    assert jc.hopf_parameter is hopf_parameter
    assert jc.lyapunov_coefficient is lyapunov_coefficient
