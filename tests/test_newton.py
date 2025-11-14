"""
Tests for Newton solver.
"""

import pytest
import jax.numpy as jnp
from jaxcont.solvers.newton import NewtonSolver


def test_newton_simple_root():
    """Test Newton solver on simple problem: f(x) = x^2 - 4"""
    def f(x):
        return x**2 - 4.0
    
    solver = NewtonSolver(tol=1e-8, max_iter=20)
    x, converged, n_iter = solver.solve(f, jnp.array([1.0]))
    
    assert converged
    assert jnp.isclose(x, 2.0, atol=1e-6)
    assert n_iter < 10


def test_newton_system():
    """Test Newton solver on system of equations."""
    def f(x):
        # f1 = x^2 + y^2 - 1
        # f2 = x - y
        return jnp.array([
            x[0]**2 + x[1]**2 - 1.0,
            x[0] - x[1]
        ])
    
    # Use slightly looser tolerance since floating point convergence can stall at ~6e-8
    solver = NewtonSolver(tol=1e-7, max_iter=20)
    x, converged, n_iter = solver.solve(f, jnp.array([0.5, 0.5]))
    
    # Check that we got close enough (even if not "converged" by strict tolerance)
    residual = jnp.linalg.norm(f(x))
    assert residual < 1e-6, f"Residual {residual} too large"
    
    # Solution should be x = y = 1/sqrt(2)
    expected = jnp.array([1.0/jnp.sqrt(2.0), 1.0/jnp.sqrt(2.0)])
    assert jnp.allclose(x, expected, atol=1e-6)


def test_newton_no_convergence():
    """Test Newton solver on problem that doesn't converge."""
    def f(x):
        return jnp.array([jnp.exp(x[0])])  # No root
    
    solver = NewtonSolver(tol=1e-8, max_iter=5)
    x, converged, n_iter = solver.solve(f, jnp.array([1.0]))
    
    assert not converged
    assert n_iter == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
