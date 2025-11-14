"""
Fold (saddle-node) bifurcation detection.
"""

from typing import List, Dict, Any, Optional
import jax.numpy as jnp
from jax import Array

from jaxcont.core.continuation import ContinuationSolution


class FoldBifurcation:
    """
    Fold (saddle-node) bifurcation detection and analysis.
    
    A fold bifurcation occurs when an eigenvalue crosses zero through the real axis.
    
    Test function: det(Jacobian) = 0 or smallest |eigenvalue| = 0
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize fold bifurcation detector.
        
        Args:
            tolerance: Detection tolerance
        """
        self.tolerance = tolerance
    
    def test_function(self, eigenvalues: Array) -> float:
        """
        Compute test function for fold bifurcation.
        
        The test function is the minimum absolute eigenvalue.
        It changes sign at a fold bifurcation.
        
        Args:
            eigenvalues: Eigenvalues at a point
        
        Returns:
            Test function value
        """
        # Find eigenvalue closest to zero
        min_eig = eigenvalues[jnp.argmin(jnp.abs(eigenvalues))]
        return float(jnp.real(min_eig))
    
    def detect(
        self,
        solution: ContinuationSolution,
        eigenvalues: Optional[Array] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect fold bifurcations along continuation branch.
        
        Args:
            solution: Continuation solution
            eigenvalues: Eigenvalues at each point
        
        Returns:
            List of detected fold bifurcations
        """
        if eigenvalues is None:
            # Would need to compute eigenvalues
            return []
        
        fold_points = []
        n_points = solution.n_points
        
        # Compute test function along branch
        test_vals = jnp.array([
            self.test_function(eigenvalues[i])
            for i in range(n_points)
        ])
        
        # Detect sign changes
        for i in range(n_points - 1):
            if test_vals[i] * test_vals[i + 1] < 0:
                # Sign change detected - fold bifurcation
                fold_info = {
                    "type": "fold",
                    "index": (i, i + 1),
                    "parameter": float((solution.parameters[i] + solution.parameters[i + 1]) / 2),
                    "state": (solution.states[i] + solution.states[i + 1]) / 2,
                    "test_function": (test_vals[i], test_vals[i + 1])
                }
                fold_points.append(fold_info)
        
        return fold_points
    
    def compute_normal_form(
        self,
        jacobian: Array,
        parameter_derivative: Array
    ) -> Dict[str, float]:
        """
        Compute normal form coefficients for fold bifurcation.
        
        Args:
            jacobian: Jacobian at bifurcation point
            parameter_derivative: df/dp at bifurcation point
        
        Returns:
            Dictionary with normal form coefficients
        """
        # Compute null vector and adjoint null vector
        # This requires computing the kernel of J and J^T
        # Placeholder for full implementation
        return {"a": 0.0, "b": 0.0}
