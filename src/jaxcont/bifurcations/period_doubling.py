"""
Period-doubling (flip) bifurcation detection for periodic orbits.
"""

from typing import List, Dict, Any, Optional
import jax.numpy as jnp
from jax import Array

from jaxcont.core.continuation import ContinuationSolution


class PeriodDoublingBifurcation:
    """
    Period-doubling (flip) bifurcation detection.
    
    A period-doubling bifurcation occurs when a Floquet multiplier crosses -1
    through the real axis.
    
    Test function: (largest_multiplier + 1) with |multiplier| ≈ 1
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize period-doubling detector.
        
        Args:
            tolerance: Detection tolerance
        """
        self.tolerance = tolerance
    
    def test_function(self, floquet_multipliers: Array) -> float:
        """
        Compute test function for period-doubling bifurcation.
        
        Args:
            floquet_multipliers: Floquet multipliers
        
        Returns:
            Test function value
        """
        # Find real multipliers near the unit circle
        real_mask = jnp.abs(jnp.imag(floquet_multipliers)) < self.tolerance
        near_unit = jnp.abs(jnp.abs(floquet_multipliers) - 1.0) < 0.1
        
        candidates = floquet_multipliers[real_mask & near_unit]
        
        if len(candidates) == 0:
            return float('inf')
        
        # Find the one closest to -1
        idx = jnp.argmin(jnp.abs(candidates + 1.0))
        critical_mult = candidates[idx]
        
        return float(jnp.real(critical_mult) + 1.0)
    
    def detect(
        self,
        solution: ContinuationSolution,
        floquet_multipliers: Optional[Array] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect period-doubling bifurcations along periodic orbit branch.
        
        Args:
            solution: Continuation solution for periodic orbits
            floquet_multipliers: Floquet multipliers at each point
        
        Returns:
            List of detected period-doubling bifurcations
        """
        if floquet_multipliers is None:
            return []
        
        pd_points = []
        n_points = solution.n_points
        
        # Compute test function along branch
        test_vals = []
        for i in range(n_points):
            test_val = self.test_function(floquet_multipliers[i])
            test_vals.append(test_val)
        
        test_vals = jnp.array(test_vals)
        
        # Detect sign changes
        for i in range(n_points - 1):
            if jnp.isfinite(test_vals[i]) and jnp.isfinite(test_vals[i + 1]):
                if test_vals[i] * test_vals[i + 1] < 0:
                    # Sign change - period-doubling bifurcation
                    pd_info = {
                        "type": "period-doubling",
                        "index": (i, i + 1),
                        "parameter": float((solution.parameters[i] + solution.parameters[i + 1]) / 2),
                        "state": (solution.states[i] + solution.states[i + 1]) / 2,
                        "test_function": (float(test_vals[i]), float(test_vals[i + 1]))
                    }
                    pd_points.append(pd_info)
        
        return pd_points
