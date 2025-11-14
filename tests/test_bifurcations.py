"""
Tests for bifurcation detection.
"""

import pytest
import jax.numpy as jnp
from jaxcont.bifurcations.fold import FoldBifurcation
from jaxcont.bifurcations.hopf import HopfBifurcation


def test_fold_test_function():
    """Test fold bifurcation test function."""
    fold = FoldBifurcation()
    
    # Eigenvalues with one near zero
    eigs = jnp.array([0.01, -1.0, -2.0])
    test_val = fold.test_function(eigs)
    
    assert jnp.isclose(test_val, 0.01, atol=1e-6)


def test_hopf_test_function():
    """Test Hopf bifurcation test function."""
    hopf = HopfBifurcation()
    
    # Complex conjugate pair crossing imaginary axis
    eigs = jnp.array([0.01 + 2.0j, 0.01 - 2.0j, -1.0])
    test_val = hopf.test_function(eigs)
    
    assert jnp.isclose(test_val, 0.01, atol=1e-6)


def test_hopf_test_function_no_complex():
    """Test Hopf test function with no complex eigenvalues."""
    hopf = HopfBifurcation()
    
    # All real eigenvalues
    eigs = jnp.array([-0.5, -1.0, -2.0])
    test_val = hopf.test_function(eigs)
    
    assert jnp.isinf(test_val)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
