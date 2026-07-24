"""
Tests for jaxcont.core.collocation: Gauss-Legendre nodes/weights, the local
Lagrange differentiation matrix, and the Collocation config type. Pure
numerics -- see docs/superpowers/specs/2026-07-24-periodic-orbit-collocation-design.md.
"""

import numpy as np
import pytest

from jaxcont.core.collocation import (
    Collocation,
    collocation_matrices,
    gauss_legendre_01,
    lagrange_diff_matrix,
    lagrange_eval_weights,
)


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
