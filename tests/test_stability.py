"""
Tests for stability analysis.
"""

import pytest
import jax.numpy as jnp
from jaxcont.stability.eigenvalue import (
    compute_eigenvalues,
    analyze_stability,
    compute_eigenvalues_along_branch,
    compute_stability_along_branch,
)
from jaxcont.core.continuation import ContinuationProblem, ContinuationSolution


def test_compute_eigenvalues():
    """Test eigenvalue computation."""
    # 2x2 matrix with known eigenvalues
    A = jnp.array([[1.0, 0.0], [0.0, -2.0]])
    eigs = compute_eigenvalues(A)
    
    # Should be sorted by real part (descending)
    assert jnp.allclose(jnp.sort(eigs), jnp.array([-2.0, 1.0]))


def test_analyze_stability_stable():
    """Test stability analysis for stable equilibrium."""
    # All eigenvalues in left half-plane
    eigs = jnp.array([-1.0, -2.0, -0.5])
    analysis = analyze_stability(eigs)
    
    assert analysis['is_stable']
    assert not analysis['is_unstable']
    assert analysis['n_unstable'] == 0
    assert analysis['n_stable'] == 3


def test_analyze_stability_unstable():
    """Test stability analysis for unstable equilibrium."""
    # Some eigenvalues in right half-plane
    eigs = jnp.array([1.0, -2.0, -0.5])
    analysis = analyze_stability(eigs)
    
    assert not analysis['is_stable']
    assert analysis['is_unstable']
    assert analysis['n_unstable'] == 1
    assert analysis['type'] == 'saddle'


def test_analyze_stability_focus():
    """Test stability analysis for focus (complex eigenvalues)."""
    # Complex conjugate pair in left half-plane
    eigs = jnp.array([-0.5 + 2.0j, -0.5 - 2.0j, -1.0])
    analysis = analyze_stability(eigs)

    assert analysis['is_stable']
    assert 'focus' in analysis['type']


def test_analyze_stability_unstable_node():
    """Test stability analysis for an unstable node (all real, all positive)."""
    # No eigenvalues in the left half-plane at all -> "unstable node", not "saddle"
    eigs = jnp.array([1.0, 2.0, 0.5])
    analysis = analyze_stability(eigs)

    assert not analysis['is_stable']
    assert analysis['is_unstable']
    assert analysis['n_stable'] == 0
    assert analysis['type'] == 'unstable node'


def test_analyze_stability_unstable_focus():
    """Test stability analysis for an unstable focus (complex, right half-plane only)."""
    eigs = jnp.array([1.0 + 2.0j, 1.0 - 2.0j])
    analysis = analyze_stability(eigs)

    assert analysis['is_unstable']
    assert analysis['n_stable'] == 0
    assert analysis['type'] == 'unstable focus'


def test_analyze_stability_center():
    """Test stability analysis for a center (purely imaginary eigenvalues)."""
    eigs = jnp.array([2.0j, -2.0j])
    analysis = analyze_stability(eigs)

    assert not analysis['is_stable']
    assert not analysis['is_unstable']
    assert analysis['n_center'] == 2
    assert analysis['type'] == 'center'


def test_analyze_stability_dominant_eigenvalue():
    """dominant_eigenvalue is the rightmost eigenvalue (array is pre-sorted descending)."""
    eigs = jnp.array([1.0, -2.0, -0.5])
    analysis = analyze_stability(eigs)

    assert jnp.allclose(jnp.real(analysis['dominant_eigenvalue']), 1.0)


def _linear_problem() -> ContinuationProblem:
    """A trivial 1-D problem f(u, p) = p * u, whose Jacobian is just p."""
    def rhs(u, params):
        return params["p"] * u

    return ContinuationProblem(
        rhs=rhs,
        u0=jnp.array([1.0]),
        params={"p": 1.0},
        continuation_param="p",
    )


def test_compute_eigenvalues_along_branch():
    """Eigenvalues along a branch should track the (trivial, scalar) Jacobian at each point."""
    problem = _linear_problem()
    solution = ContinuationSolution(
        states=jnp.array([[1.0], [1.0], [1.0]]),
        parameters=jnp.array([-2.0, -0.5, 3.0]),
    )

    eigs = compute_eigenvalues_along_branch(problem, solution)

    assert eigs.shape == (3, 1)
    assert jnp.allclose(jnp.real(eigs[:, 0]), jnp.array([-2.0, -0.5, 3.0]))


def test_compute_stability_along_branch():
    """Stability flags along a branch should flip sign with the parameter for f = p*u."""
    problem = _linear_problem()
    solution = ContinuationSolution(
        states=jnp.array([[1.0], [1.0]]),
        parameters=jnp.array([-1.0, 1.0]),
    )

    stability = compute_stability_along_branch(problem, solution)

    assert stability.shape == (2,)
    assert bool(stability[0]) is True   # p = -1 -> stable
    assert bool(stability[1]) is False  # p = +1 -> unstable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
