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
