"""
Domain-independent tests for jaxcont.solvers.implicit.differentiable_root,
extracted from bifurcations/fold_solve.py's implicit-diff Newton solver. See
docs/superpowers/specs/2026-07-23-differentiable-root-primitive-design.md.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxcont.solvers.implicit import differentiable_root


def _eps_J():
    return float(jnp.finfo(jnp.asarray(0.).dtype).eps)



def _G_scalar(x, theta):
    # root: x* = sqrt(theta)
    return x**2 - theta


def test_root_matches_analytic():
    x0 = jnp.array([1.0])
    theta = jnp.array(4.0)
    x_star = differentiable_root(_G_scalar, x0, theta)
    # Measured |error| x32/x64: 0/2.22e-15; Newton root, unit conditioning.
    assert float(x_star[0]) == pytest.approx(2.0, abs=1e-06 if not jax.config.x64_enabled else 1e-09, rel=0)


def test_reverse_mode_grad_matches_analytic():
    x0 = jnp.array([1.0])

    def root(theta):
        return differentiable_root(_G_scalar, x0, theta)[0]

    for theta_val in (4.0, 9.0):
        theta = jnp.array(theta_val)
        g = jax.grad(root)(theta)
        expected = 1.0 / (2.0 * jnp.sqrt(theta))
        # Measured |error| x32/x64: 0/2.78e-16; analytic implicit derivative, well-conditioned roots.
        assert float(g) == pytest.approx(float(expected), abs=0, rel=64 * _eps_J())


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
    # Measured |error| x32/x64: 0/0; implicit derivative evaluated at the same root.
    assert jnp.allclose(J, expected, atol=0, rtol=16 * _eps_J())


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
        # Measured |error| x32/x64: 1.49e-08/2.78e-16; analytic implicit derivative, well-conditioned roots.
        assert float(g) == pytest.approx(float(expected), abs=0, rel=64 * _eps_J())


def test_differentiable_root_keeps_exact_seed():
    # The seed already solves G = x**2 = 0 exactly, where the Jacobian is
    # singular: it must be returned untouched, not replaced by a NaN step.
    from jaxcont.solvers.implicit import differentiable_root_checked

    x_star, converged = differentiable_root_checked(
        lambda x, theta: x**2, jnp.array([0.0]), jnp.array(0.0)
    )
    assert float(x_star[0]) == 0.0
    assert bool(converged)


def test_scan_natural_corrector_reverse_mode_matches_forward():
    import jaxcont as jc

    def total(a, loop):
        problem = jc.bif_problem(lambda u, p, a: u**2 - a * p,
                                 jnp.array([0.9]), 1.0, args=a)
        branch = jc.continuation(
            problem, jc.Natural(), p_span=(1.0, 1.2),
            settings=jc.ContinuationPar(loop=loop, max_steps=3,
                                       newton_max_iter=8, compute_stability=False),
        ).branch
        return jnp.sum(branch.states * branch.valid[:, None])

    forward = jax.jit(lambda a: total(a, 'while'))
    reverse = jax.jit(lambda a: total(a, 'scan'))
    a = jnp.array(1.4)
    np_expected = jax.jacfwd(forward)(a)
    # Measured |error| x32/x64: 0/2.22e-16; forward/reverse implicit derivative consistency.
    assert jnp.allclose(jax.grad(reverse)(a), np_expected, rtol=16 * _eps_J(), atol=0)


def _scaled_cube_root():
    """G(x) = theta * (x^3 - 2): the root 2**(1/3) is not representable, so its
    best float has residual ~ theta * 3 x^2 * eps * x, far above default_tol
    in float32 for theta = 1e4."""
    return (lambda x, theta: theta * (x ** 3 - 2.0)), 2.0 ** (1.0 / 3.0)


def test_checked_root_accepts_a_root_at_the_rounding_floor():
    """With the default tolerance, a root as accurate as the dtype allows is converged."""
    from jaxcont.solvers.implicit import differentiable_root_checked
    G, root = _scaled_cube_root()
    x, converged = differentiable_root_checked(G, jnp.array([1.5]), jnp.asarray(1e4))
    assert bool(converged)
    assert abs(float(x[0]) - root) <= 4 * float(jnp.finfo(x.dtype).eps) * root


def test_checked_root_still_rejects_a_non_root():
    from jaxcont.solvers.implicit import differentiable_root_checked
    G = lambda x, theta: x ** 2 + theta  # no real root for theta > 0
    _, converged = differentiable_root_checked(G, jnp.array([0.5]), jnp.asarray(1.0))
    assert not bool(converged)


def test_checked_root_floor_is_per_equation():
    """Review A-01: a huge derivative in one satisfied equation must not relax
    another. x[1]**2 + 1 = 0 has no real root, whatever theta * x[0] does."""
    import jax
    from jaxcont.solvers.implicit import differentiable_root_checked
    theta = 1e16 if jnp.asarray(0.).dtype == jnp.float64 else 1e8
    G = lambda x, t: jnp.array([t * x[0], x[1] ** 2 + 1])
    for solve in (differentiable_root_checked, jax.jit(differentiable_root_checked, static_argnums=0)):
        _, converged = solve(G, jnp.array([0., .5]), jnp.asarray(theta))
        assert not bool(converged)


def test_checked_root_floor_still_accepts_a_badly_scaled_equation_beside_a_regular_one():
    """The per-equation floor keeps the L2-F2 behaviour for the scaled equation."""
    from jaxcont.solvers.implicit import differentiable_root_checked
    G1, root = _scaled_cube_root()
    G = lambda x, t: jnp.array([G1(x[:1], t)[0], x[1] - 0.5])
    x, converged = differentiable_root_checked(G, jnp.array([1.5, 0.]), jnp.asarray(1e4))
    assert bool(converged)
    assert abs(float(x[0]) - root) <= 4 * float(jnp.finfo(x.dtype).eps) * root


def test_checked_root_keeps_an_explicit_tolerance_strict():
    """A caller-chosen tol below the floor is not silently relaxed."""
    from jaxcont.solvers.implicit import differentiable_root_checked
    G, _ = _scaled_cube_root()
    x, converged = differentiable_root_checked(G, jnp.array([1.5]), jnp.asarray(1e4), tol=1e-30)
    if float(jnp.abs(G(x, 1e4))[0]) > 0:  # an exactly zero residual would legitimately meet any tol
        assert not bool(converged)
