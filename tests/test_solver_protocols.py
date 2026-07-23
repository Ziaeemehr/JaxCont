"""
Tests for jaxcont.solvers.protocols: LinearSolver/EigenSolver structural
protocols and their Dense/DenseEigen default implementations -- see
docs/superpowers/specs/2026-07-23-linear-eigen-solver-protocols-design.md.
"""

import jax.numpy as jnp

from jaxcont.solvers.protocols import Dense, DenseEigen, EigenSolver, LinearSolver


def test_dense_matches_jnp_linalg_solve():
    A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
    b = jnp.array([3.0, 5.0])
    assert jnp.array_equal(Dense()(A, b), jnp.linalg.solve(A, b))


def test_dense_eigen_matches_jnp_linalg_eigvals():
    A = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    assert jnp.array_equal(DenseEigen()(A), jnp.linalg.eigvals(A))


def test_dense_instances_are_value_equal_and_hashable():
    # Required for Dense() to be safe as a jax.jit static argument: two
    # independently-constructed instances (not the same object) must
    # compare and hash equal, or every call with a fresh default would
    # force a spurious recompile.
    assert Dense() == Dense()
    assert hash(Dense()) == hash(Dense())


def test_dense_eigen_instances_are_value_equal_and_hashable():
    assert DenseEigen() == DenseEigen()
    assert hash(DenseEigen()) == hash(DenseEigen())


def test_dense_satisfies_linearsolver_protocol():
    assert isinstance(Dense(), LinearSolver)


def test_dense_eigen_satisfies_eigensolver_protocol():
    assert isinstance(DenseEigen(), EigenSolver)


def test_runtime_checkable_isinstance_cannot_distinguish_dense_from_eigensolver():
    # Dense.__call__ takes (A, b); EigenSolver's runtime_checkable isinstance
    # check only verifies the __call__ attribute exists (Protocol does not
    # check signatures at runtime) -- so isinstance(Dense(), EigenSolver) is
    # True even though calling it as an EigenSolver (one argument) would
    # TypeError at runtime. This test documents that limitation so a future
    # reader isn't surprised by it when relying on isinstance checks
    # elsewhere -- it is not a requirement being asserted, just a recorded
    # fact about Python's Protocol machinery.
    assert isinstance(Dense(), EigenSolver)
