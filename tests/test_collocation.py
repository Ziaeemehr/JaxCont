"""
Tests for jaxcont.core.collocation: Gauss-Legendre nodes/weights, the local
Lagrange differentiation matrix, and the Collocation config type. Pure
numerics -- see docs/superpowers/specs/2026-07-24-periodic-orbit-collocation-design.md.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcont.core.collocation import (
    Collocation,
    collocation_matrices,
    gauss_legendre_01,
    lagrange_diff_matrix,
    lagrange_eval_weights,
    monodromy_matrix,
)
from jaxcont.problems.periodic import periodic_orbit_problem


def test_gauss_legendre_01_matches_numpy_reference():
    for ncol in (2, 3, 4, 5):
        nodes, weights = gauss_legendre_01(ncol)
        x_ref, w_ref = np.polynomial.legendre.leggauss(ncol)
        expected_nodes = 0.5 * (x_ref + 1.0)
        expected_weights = 0.5 * w_ref
        assert np.allclose(nodes, expected_nodes)
        assert np.allclose(weights, expected_weights)


def test_gauss_legendre_nodes_are_interior_to_01():
    nodes, _ = gauss_legendre_01(4)
    assert np.all(nodes > 0.0)
    assert np.all(nodes < 1.0)


def test_lagrange_diff_matrix_is_exact_on_degree_ncol_polynomial():
    # Regression for the exact scheme verified during design: p(x) =
    # x^4 - 2x^3 + x - 1 (degree 4), p'(x) = 4x^3 - 6x^2 + 1. Local nodes
    # are [0, four interior Gauss-Legendre points] -- 5 nodes, degree-4
    # exact fit. Verified during design at max abs error 5.6e-15.
    ncol = 4
    gauss, _ = gauss_legendre_01(ncol)
    local_nodes = np.concatenate([[0.0], gauss])
    D = lagrange_diff_matrix(local_nodes)

    def p(x):
        return x**4 - 2 * x**3 + x - 1

    def pprime(x):
        return 4 * x**3 - 6 * x**2 + 1

    v = p(local_nodes)
    Dv = D @ v
    assert np.max(np.abs(Dv - pprime(local_nodes))) < 1e-12


def test_lagrange_eval_weights_extrapolate_exactly():
    # A degree-ncol polynomial evaluated at x=1 via the weight vector must
    # match direct evaluation, since the interpolant is exact for it.
    ncol = 4
    gauss, _ = gauss_legendre_01(ncol)
    local_nodes = np.concatenate([[0.0], gauss])
    E = lagrange_eval_weights(local_nodes, 1.0)

    def p(x):
        return x**4 - 2 * x**3 + x - 1

    v = p(local_nodes)
    assert abs(float(E @ v) - p(1.0)) < 1e-12


def test_collocation_matrices_shapes():
    ncol = 4
    D, E, gauss, gw = collocation_matrices(ncol)
    assert D.shape == (ncol + 1, ncol + 1)
    assert E.shape == (ncol + 1,)
    assert gauss.shape == (ncol,)
    assert gw.shape == (ncol,)


def test_collocation_ntst_ncol_are_static_python_ints():
    m = Collocation(ntst=10, ncol=4)
    assert isinstance(m.ntst, int)
    assert isinstance(m.ncol, int)

    # Changing a static field changes the pytree's structure (jit cache key).
    import jax

    m2 = Collocation(ntst=15, ncol=4)
    _, treedef1 = jax.tree_util.tree_flatten(m)
    _, treedef2 = jax.tree_util.tree_flatten(m2)
    assert treedef1 != treedef2


def _circle_rhs(u, p, args):
    x, y = u[0], u[1]
    r2 = x * x + y * y
    rho = p
    return jnp.array([(rho - r2) * x - y, (rho - r2) * y + x])


def _circle_periodic_problem():
    import numpy as np

    rng = np.random.default_rng(0)
    t_traj = np.sort(rng.uniform(0, 5.5, size=40))
    t_traj[0] = 0.0
    theta = lambda t: 2 * np.pi * t / 5.5 + 0.3
    u_traj = np.stack(
        [0.8 * np.cos(theta(t_traj)), 0.8 * np.sin(theta(t_traj))], axis=1
    )
    mesh = Collocation(ntst=10, ncol=4)
    return periodic_orbit_problem(
        _circle_rhs, jnp.asarray(u_traj), jnp.asarray(t_traj), 5.5, 1.0, mesh
    ), mesh


def test_monodromy_matrix_matches_closed_form_circle_multipliers():
    # r' = r*(rho - r^2), theta' = 1 at rho=1: exact circle, T=2*pi.
    # Closed-form Floquet multipliers {1, exp(-4*pi)} -- see
    # docs/superpowers/specs/2026-07-24-floquet-multipliers-design.md.
    # Verified during design: NumPy prototype gave [3.4873e-06, 0.999998],
    # JAX/jit gave [3.4570694e-06, 1.0000001] -- both float32-level
    # agreement with the exact exp(-4*pi) = 3.4873423562089973e-06.
    prob, mesh = _circle_periodic_problem()
    ntst, ncol, n = mesh.ntst, mesh.ncol, 2
    h = 1.0 / ntst

    D_np, E_np, _, _ = collocation_matrices(ncol)
    D, E = jnp.asarray(D_np), jnp.asarray(E_np)

    mesh_states = prob.u0[: ntst * n].reshape(ntst, n)
    coll_states = prob.u0[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
    T = prob.u0[-1]
    p = prob.p0

    Phi = monodromy_matrix(_circle_rhs, D, E, h, mesh_states, coll_states, T, p)
    assert Phi.shape == (n, n)

    multipliers = jnp.sort(jnp.abs(jnp.linalg.eigvals(Phi)))
    expected = jnp.sort(jnp.array([np.exp(-4 * np.pi), 1.0]))
    assert float(jnp.max(jnp.abs(multipliers - expected))) < 1e-5


def test_monodromy_matrix_is_jit_compatible():
    prob, mesh = _circle_periodic_problem()
    ntst, ncol, n = mesh.ntst, mesh.ncol, 2
    h = 1.0 / ntst
    D_np, E_np, _, _ = collocation_matrices(ncol)
    D, E = jnp.asarray(D_np), jnp.asarray(E_np)
    mesh_states = prob.u0[: ntst * n].reshape(ntst, n)
    coll_states = prob.u0[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
    T, p = prob.u0[-1], prob.p0

    jitted = jax.jit(monodromy_matrix, static_argnums=(0,))
    Phi = jitted(_circle_rhs, D, E, h, mesh_states, coll_states, T, p)
    assert Phi.shape == (n, n)
