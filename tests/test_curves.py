"""Two-parameter curve factories (bifurcations/curves.py)."""

import jax.numpy as jnp
import pytest

import jaxcont as jc
from jaxcont.bifurcations.curves import fold_curve_problem, unpack_fold_curve


def _cusp_shifted(u, p, args):
    """u' = (p0-0.3) + (p1-1.2)*(u-0.7) - (u-0.7)**3.

    The cusp normal form, affinely shifted off the origin (origin-centred
    normal forms have no discriminating power -- a stub returning zeros
    passes them all). Its fold set is the exact discriminant
        27*(p0-0.3)**2 == 4*(p1-1.2)**3
    derived by eliminating u from f=0 and df/du=0.
    """
    x = u[0] - 0.7
    a = p[0] - 0.3
    b = p[1] - 1.2
    return jnp.array([a + b * x - x**3])


def _cusp_discriminant(p0, p1):
    return 27.0 * (p0 - 0.3) ** 2 - 4.0 * (p1 - 1.2) ** 3


def test_fold_curve_traces_the_exact_cusp_discriminant():
    # Seed: b = p1-1.2 = 3  ->  x = sqrt(b/3) = 1 -> u = 1.7,
    # a = p0-0.3 = -2*x*b/3 = -2 -> p0 = -1.7.
    prob = fold_curve_problem(
        _cusp_shifted,
        jnp.array([1.7]),
        jnp.array([-1.7, 4.2]),
        free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(4.2, 6.2),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
    )
    assert sol.branch.params.shape[0] > 5
    for i in range(sol.branch.params.shape[0]):
        p1 = float(sol.branch.params[i])
        _, p0, _ = unpack_fold_curve(sol.branch.states[i], n=1)
        assert abs(_cusp_discriminant(float(p0), p1)) < 1e-2


def test_fold_curve_problem_rejects_wrong_p_shape():
    with pytest.raises(ValueError, match="shape"):
        fold_curve_problem(_cusp_shifted, jnp.array([1.7]), jnp.array([-1.7]))


from jaxcont.bifurcations.curves import hopf_curve_problem, unpack_hopf_curve


def _hopf_parabola(u, p, args):
    """Hopf normal form with equilibrium shifted to (0.5, -0.3) and
    mu = p0 + p1**2 - 2, so the exact Hopf curve is the parabola
    p0 = 2 - p1**2 and the critical frequency is exactly omega = 1."""
    x = u[0] - 0.5
    y = u[1] + 0.3
    mu = p[0] + p[1] ** 2 - 2.0
    r2 = x * x + y * y
    return jnp.array([mu * x - y - x * r2, x + mu * y - y * r2])


def test_hopf_curve_traces_the_exact_parabola():
    prob = hopf_curve_problem(
        _hopf_parabola,
        jnp.array([0.5, -0.3]),
        jnp.array([2.0, 0.0]),
        free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(0.0, 1.2),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
    )
    assert sol.branch.params.shape[0] > 5
    for i in range(sol.branch.params.shape[0]):
        p1 = float(sol.branch.params[i])
        _, p0, _, _, omega = unpack_hopf_curve(sol.branch.states[i], n=2)
        assert abs(float(p0) + p1**2 - 2.0) < 1e-3
        assert abs(float(omega) - 1.0) < 1e-3
