"""
Tests for core continuation functionality.
"""

import jax.numpy as jnp
import pytest

from jaxcont.core.continuation import ContinuationProblem, ContinuationSolution


def _eps_J():
    return float(jnp.finfo(jnp.float64).eps)



def simple_rhs(state, params):
    """Simple test RHS: dx/dt = r*x - x^2"""
    x = state[0]
    r = params['r']
    return jnp.array([r * x - x**2])


def test_continuation_problem_creation():
    """Test creating a continuation problem."""
    problem = ContinuationProblem(
        rhs=simple_rhs,
        u0=jnp.array([0.1]),
        params={'r': 1.0},
        continuation_param='r'
    )
    
    assert problem.state_dim == 1
    assert problem.param_value == 1.0
    assert problem.continuation_param == 'r'


def test_continuation_problem_invalid_param():
    """Test that invalid continuation parameter raises error."""
    with pytest.raises(ValueError):
        ContinuationProblem(
            rhs=simple_rhs,
            u0=jnp.array([0.1]),
            params={'r': 1.0},
            continuation_param='invalid_param'
        )


def test_continuation_solution_creation():
    """Test creating a continuation solution."""
    states = jnp.array([[0.0], [0.5], [1.0]])
    parameters = jnp.array([0.0, 0.5, 1.0])
    
    solution = ContinuationSolution(
        states=states,
        parameters=parameters
    )
    
    assert solution.n_points == 3
    assert solution.state_dim == 1


def test_continuation_solution_get_point():
    """Test getting a specific point from solution."""
    states = jnp.array([[0.0], [0.5], [1.0]])
    parameters = jnp.array([0.0, 0.5, 1.0])
    
    solution = ContinuationSolution(
        states=states,
        parameters=parameters
    )
    
    state, param = solution.get_point(1)
    # Measured |error|: 0; linear scalar root half, unit conditioning.
    assert jnp.allclose(state, jnp.array([0.5]), atol=0, rtol=16 * _eps_J())
    # Measured |error|: 0; linear scalar root half, unit conditioning.
    assert jnp.isclose(param, 0.5, atol=0, rtol=16 * _eps_J())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
