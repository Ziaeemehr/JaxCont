"""
Tests for stability analysis.
"""

import pytest
import jax.numpy as jnp
from jaxcont.stability.eigenvalue import compute_eigenvalues, analyze_stability


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
