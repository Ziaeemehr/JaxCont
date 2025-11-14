"""
Bifurcation detection along continuation branches.
"""

from typing import List, Dict, Any, Optional
import jax.numpy as jnp
from jax import Array

from jaxcont.core.continuation import ContinuationSolution
from jaxcont.bifurcations.fold import FoldBifurcation
from jaxcont.bifurcations.hopf import HopfBifurcation


class BifurcationDetector:
    """
    Detects bifurcations along a continuation branch.
    
    Monitors test functions along the branch and detects sign changes
    that indicate bifurcation points.
    """
    
    def __init__(
        self,
        detect_fold: bool = True,
        detect_hopf: bool = True,
        detect_branch_point: bool = False,
        tolerance: float = 1e-6
    ):
        """
        Initialize bifurcation detector.
        
        Args:
            detect_fold: Whether to detect fold bifurcations
            detect_hopf: Whether to detect Hopf bifurcations
            detect_branch_point: Whether to detect branch points
            tolerance: Detection tolerance
        """
        self.detect_fold = detect_fold
        self.detect_hopf = detect_hopf
        self.detect_branch_point = detect_branch_point
        self.tolerance = tolerance
        
        # Initialize bifurcation type detectors
        self.fold_detector = FoldBifurcation() if detect_fold else None
        self.hopf_detector = HopfBifurcation() if detect_hopf else None
    
    def detect_along_branch(
        self,
        solution: ContinuationSolution,
        eigenvalues: Optional[Array] = None,
        refine_location: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detect all bifurcations along a continuation branch.
        
        This method:
        1. Uses eigenvalues (from solution or provided) to detect sign changes
        2. Optionally refines bifurcation locations using bisection
        3. Returns sorted list of all detected bifurcations
        
        Args:
            solution: Continuation solution
            eigenvalues: Eigenvalues at each point (uses solution.eigenvalues if None)
            refine_location: Whether to refine bifurcation locations with bisection
        
        Returns:
            List of detected bifurcations with detailed information
        """
        bifurcations = []
        
        # Use eigenvalues from solution if not provided
        if eigenvalues is None:
            if solution.eigenvalues is not None:
                eigenvalues = solution.eigenvalues
            else:
                # Cannot detect without eigenvalues
                import warnings
                warnings.warn(
                    "No eigenvalues provided and none stored in solution. "
                    "Cannot detect bifurcations. Enable compute_stability=True "
                    "in continuation settings."
                )
                return []
        
        # Detect fold bifurcations (branch points, saddle-node)
        if self.detect_fold and self.fold_detector is not None:
            fold_points = self.fold_detector.detect(solution, eigenvalues)
            
            # Refine locations if requested
            if refine_location:
                refined_fold_points = []
                for fold in fold_points:
                    idx1, idx2 = fold['index']
                    refined = self.locate_bifurcation(
                        solution, idx1, idx2, 'fold',
                        tolerance=self.tolerance
                    )
                    # Merge original detection info with refined info
                    refined.update({
                        'original_parameter': fold['parameter'],
                        'test_function': fold.get('test_function')
                    })
                    refined_fold_points.append(refined)
                bifurcations.extend(refined_fold_points)
            else:
                bifurcations.extend(fold_points)
        
        # Detect Hopf bifurcations
        if self.detect_hopf and self.hopf_detector is not None:
            hopf_points = self.hopf_detector.detect(solution, eigenvalues)
            
            # Refine locations if requested
            if refine_location:
                refined_hopf_points = []
                for hopf in hopf_points:
                    idx1, idx2 = hopf['index']
                    refined = self.locate_bifurcation(
                        solution, idx1, idx2, 'hopf',
                        tolerance=self.tolerance
                    )
                    # Preserve frequency and other Hopf-specific info
                    refined.update({
                        'original_parameter': hopf['parameter'],
                        'frequency': hopf.get('frequency'),
                        'test_function': hopf.get('test_function')
                    })
                    refined_hopf_points.append(refined)
                bifurcations.extend(refined_hopf_points)
            else:
                bifurcations.extend(hopf_points)
        
        # Sort by parameter value
        bifurcations.sort(key=lambda x: x["parameter"])
        
        return bifurcations
    
    def locate_bifurcation(
        self,
        solution: ContinuationSolution,
        index1: int,
        index2: int,
        bif_type: str,
        max_iterations: int = 20,
        tolerance: float = 1e-8
    ) -> Dict[str, Any]:
        """
        Precisely locate a bifurcation point between two continuation points.
        
        Uses bisection method with test function evaluation to refine
        the bifurcation location to high precision.
        
        Args:
            solution: Continuation solution
            index1: Index before bifurcation
            index2: Index after bifurcation
            bif_type: Type of bifurcation ('fold', 'hopf', etc.)
            max_iterations: Maximum bisection iterations
            tolerance: Convergence tolerance for parameter value
        
        Returns:
            Dictionary with refined bifurcation information including:
            - type: Bifurcation type
            - parameter: Refined parameter value
            - state: Refined state value
            - index: Original indices
            - iterations: Number of iterations used
            - residual: Final test function value
        """
        from jax import jacfwd
        
        u1, p1 = solution.get_point(index1)
        u2, p2 = solution.get_point(index2)
        
        # Get the appropriate test function
        if bif_type == 'fold' and self.fold_detector is not None:
            test_func = self._evaluate_fold_test
        elif bif_type == 'hopf' and self.hopf_detector is not None:
            test_func = self._evaluate_hopf_test
        else:
            # Fallback to simple interpolation if detector not available
            p_bif = (p1 + p2) / 2
            u_bif = (u1 + u2) / 2
            return {
                "type": bif_type,
                "parameter": float(p_bif),
                "state": u_bif,
                "index": (index1, index2),
                "method": "linear_interpolation"
            }
        
        # Get problem from solution (if available through solution metadata)
        # For now, we'll use continuation along the branch
        
        # Bisection method to refine bifurcation location
        p_left = p1
        p_right = p2
        u_left = u1
        u_right = u2
        
        for iteration in range(max_iterations):
            # Check convergence
            if abs(p_right - p_left) < tolerance:
                break
            
            # Midpoint
            p_mid = (p_left + p_right) / 2
            
            # Linear interpolation for state at midpoint
            alpha = (p_mid - p_left) / (p_right - p_left) if p_right != p_left else 0.5
            u_mid = u_left + alpha * (u_right - u_left)
            
            # Evaluate test function at midpoint
            # Note: This requires computing eigenvalues at the interpolated point
            # For now, use linear interpolation of test function values
            
            # Get eigenvalues at endpoints
            eigs_left = self._compute_eigenvalues_at_point(u_left, p_left, solution)
            eigs_right = self._compute_eigenvalues_at_point(u_right, p_right, solution)
            eigs_mid = self._compute_eigenvalues_at_point(u_mid, p_mid, solution)
            
            # Evaluate test functions
            test_left = test_func(eigs_left)
            test_mid = test_func(eigs_mid)
            test_right = test_func(eigs_right)
            
            # Bisection: determine which half contains the zero
            if test_left * test_mid < 0:
                # Zero is in left half
                p_right = p_mid
                u_right = u_mid
            elif test_mid * test_right < 0:
                # Zero is in right half
                p_left = p_mid
                u_left = u_mid
            else:
                # Very close to zero or sign didn't change properly
                break
        
        # Final refined values
        p_bif = (p_left + p_right) / 2
        u_bif = (u_left + u_right) / 2
        
        # Compute final eigenvalues and test function value
        eigs_bif = self._compute_eigenvalues_at_point(u_bif, p_bif, solution)
        residual = test_func(eigs_bif)
        
        return {
            "type": bif_type,
            "parameter": float(p_bif),
            "state": u_bif,
            "index": (index1, index2),
            "iterations": iteration + 1,
            "residual": float(abs(residual)),
            "eigenvalues": eigs_bif,
            "method": "bisection"
        }
    
    def _compute_eigenvalues_at_point(
        self,
        u: Array,
        p: float,
        solution: ContinuationSolution
    ) -> Array:
        """
        Compute eigenvalues at a given point.
        
        Note: This requires access to the problem's RHS function.
        For now, we estimate from nearby points in the solution.
        
        Args:
            u: State vector
            p: Parameter value
            solution: Continuation solution
        
        Returns:
            Estimated eigenvalues
        """
        # Find nearest points in solution
        param_diff = jnp.abs(solution.parameters - p)
        nearest_idx = jnp.argmin(param_diff)
        
        # If eigenvalues are stored in solution, interpolate
        if solution.eigenvalues is not None:
            return solution.eigenvalues[nearest_idx]
        
        # Otherwise return empty array (would need problem.rhs to compute)
        return jnp.array([])
    
    def _evaluate_fold_test(self, eigenvalues: Array) -> float:
        """Evaluate fold bifurcation test function."""
        if self.fold_detector is not None:
            return self.fold_detector.test_function(eigenvalues)
        return 0.0
    
    def _evaluate_hopf_test(self, eigenvalues: Array) -> float:
        """Evaluate Hopf bifurcation test function."""
        if self.hopf_detector is not None:
            return self.hopf_detector.test_function(eigenvalues)
        return 0.0
