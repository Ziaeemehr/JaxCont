"""
Hopf bifurcation detection.
"""

from typing import List, Dict, Any, Optional
import jax.numpy as jnp
from jax import Array

from jaxcont.core.continuation import ContinuationSolution


class HopfBifurcation:
    """
    Hopf bifurcation detection and analysis.
    
    A Hopf bifurcation occurs when a complex conjugate pair of eigenvalues
    crosses the imaginary axis.
    
    Test function: Real part of complex eigenvalue pair = 0
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize Hopf bifurcation detector.
        
        Args:
            tolerance: Detection tolerance
        """
        self.tolerance = tolerance
    
    def test_function(self, eigenvalues: Array) -> float:
        """
        Compute test function for Hopf bifurcation.
        
        Looks for complex eigenvalue pair with smallest |Re(lambda)|.
        
        Args:
            eigenvalues: Eigenvalues at a point
        
        Returns:
            Test function value (real part of critical eigenvalue)
        """
        # Find complex eigenvalues
        complex_mask = jnp.abs(jnp.imag(eigenvalues)) > self.tolerance
        
        if not jnp.any(complex_mask):
            return float('inf')  # No complex eigenvalues
        
        complex_eigs = eigenvalues[complex_mask]
        
        # Find pair with smallest |Re(lambda)|
        idx = jnp.argmin(jnp.abs(jnp.real(complex_eigs)))
        critical_eig = complex_eigs[idx]
        
        return float(jnp.real(critical_eig))
    
    def detect(
        self,
        solution: ContinuationSolution,
        eigenvalues: Optional[Array] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect Hopf bifurcations along continuation branch.
        
        Args:
            solution: Continuation solution
            eigenvalues: Eigenvalues at each point
        
        Returns:
            List of detected Hopf bifurcations
        """
        if eigenvalues is None:
            return []
        
        hopf_points = []
        n_points = solution.n_points
        
        # Compute test function along branch
        test_vals = []
        for i in range(n_points):
            test_val = self.test_function(eigenvalues[i])
            test_vals.append(test_val)
        
        test_vals = jnp.array(test_vals)
        
        # Detect sign changes
        for i in range(n_points - 1):
            if jnp.isfinite(test_vals[i]) and jnp.isfinite(test_vals[i + 1]):
                if test_vals[i] * test_vals[i + 1] < 0:
                    # Sign change - Hopf bifurcation
                    hopf_info = {
                        "type": "hopf",
                        "index": (i, i + 1),
                        "parameter": float((solution.parameters[i] + solution.parameters[i + 1]) / 2),
                        "state": (solution.states[i] + solution.states[i + 1]) / 2,
                        "test_function": (float(test_vals[i]), float(test_vals[i + 1])),
                        "frequency": self._estimate_frequency(eigenvalues[i], eigenvalues[i + 1])
                    }
                    hopf_points.append(hopf_info)
        
        return hopf_points
    
    def _estimate_frequency(self, eigs1: Array, eigs2: Array) -> float:
        """
        Estimate the frequency at Hopf bifurcation.
        
        Args:
            eigs1: Eigenvalues before bifurcation
            eigs2: Eigenvalues after bifurcation
        
        Returns:
            Estimated frequency
        """
        # Find complex eigenvalue with smallest real part
        def get_freq(eigs):
            complex_mask = jnp.abs(jnp.imag(eigs)) > self.tolerance
            if not jnp.any(complex_mask):
                return 0.0
            complex_eigs = eigs[complex_mask]
            idx = jnp.argmin(jnp.abs(jnp.real(complex_eigs)))
            return float(jnp.abs(jnp.imag(complex_eigs[idx])))
        
        freq1 = get_freq(eigs1)
        freq2 = get_freq(eigs2)
        
        return (freq1 + freq2) / 2
    
    def compute_first_lyapunov_coefficient(
        self,
        jacobian: Array,
        hessians: Array
    ) -> float:
        """
        Compute first Lyapunov coefficient to determine criticality.
        
        If l1 < 0: supercritical Hopf (stable limit cycle)
        If l1 > 0: subcritical Hopf (unstable limit cycle)
        
        Args:
            jacobian: Jacobian at bifurcation point
            hessians: Second derivatives
        
        Returns:
            First Lyapunov coefficient
        """
        # This requires complex center manifold reduction
        # Placeholder for full implementation
        return 0.0
