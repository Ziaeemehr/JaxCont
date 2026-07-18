"""
Base predictor-corrector framework for continuation methods.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict, Any
import jax.numpy as jnp
from jax import Array

from jaxcont.core.continuation import ContinuationProblem, ContinuationSolution


class PredictorCorrector(ABC):
    """
    Abstract base class for predictor-corrector continuation methods.
    
    The predictor-corrector approach consists of:
    1. Predictor: Estimate next solution point along the branch
    2. Corrector: Refine the prediction using Newton iteration
    """
    
    def __init__(
        self,
        ds: float = 0.01,
        ds_min: float = 1e-5,
        ds_max: float = 0.1,
        max_steps: int = 1000,
        adaptive_stepsize: bool = True,
        newton_tol: float = 1e-6,
        newton_max_iter: int = 20,
        detect_bifurcations: bool = True,
        bifurcation_tolerance: float = 1e-4,
        compute_stability: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize predictor-corrector method.
        
        Args:
            ds: Initial step size
            ds_min: Minimum step size
            ds_max: Maximum step size
            max_steps: Maximum number of continuation steps
            adaptive_stepsize: Whether to adapt step size based on convergence
            newton_tol: Newton solver tolerance
            newton_max_iter: Maximum Newton iterations
            detect_bifurcations: Whether to detect bifurcations during continuation
            bifurcation_tolerance: Tolerance for bifurcation detection
            compute_stability: Whether to compute stability along the branch
            verbose: Whether to print bifurcation information
        """
        self.ds = ds
        self.ds_min = ds_min
        self.ds_max = ds_max
        self.max_steps = max_steps
        self.adaptive_stepsize = adaptive_stepsize
        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self.detect_bifurcations = detect_bifurcations
        self.bifurcation_tolerance = bifurcation_tolerance
        self.compute_stability = compute_stability
        self.verbose = verbose
        
        # Initialize bifurcation detector
        if self.detect_bifurcations:
            from jaxcont.bifurcations.detector import BifurcationDetector
            self.bifurcation_detector = BifurcationDetector(
                detect_fold=True,
                detect_hopf=True,
                tolerance=bifurcation_tolerance
            )
    
    @abstractmethod
    def predict(
        self,
        u: Array,
        param: float,
        tangent: Array,
        ds: float
    ) -> Tuple[Array, float]:
        """
        Predict next point on the continuation branch.
        
        Args:
            u: Current state
            param: Current parameter value
            tangent: Tangent vector at current point
            ds: Step size
        
        Returns:
            (predicted_state, predicted_param) tuple
        """
        pass
    
    @abstractmethod
    def correct(
        self,
        problem: ContinuationProblem,
        u_pred: Array,
        param_pred: float,
        u_prev: Array,
        param_prev: float,
        tangent: Array,
        ds: float
    ) -> Tuple[Array, float, bool, int]:
        """
        Correct predicted point using Newton iteration.
        
        Args:
            problem: Continuation problem
            u_pred: Predicted state
            param_pred: Predicted parameter
            u_prev: Previous state
            param_prev: Previous parameter
            tangent: Tangent vector
            ds: Step size
        
        Returns:
            (corrected_state, corrected_param, converged, iterations) tuple
        """
        pass
    
    @abstractmethod
    def compute_tangent(
        self,
        problem: ContinuationProblem,
        u: Array,
        param: float,
        prev_tangent: Optional[Array] = None
    ) -> Array:
        """
        Compute tangent vector to the continuation branch.
        
        Args:
            problem: Continuation problem
            u: Current state
            param: Current parameter value
            prev_tangent: Previous tangent vector (for orientation)
        
        Returns:
            Tangent vector (normalized)
        """
        pass
    
    def adapt_stepsize(
        self,
        ds: float,
        newton_iters: int,
        converged: bool
    ) -> float:
        """
        Adapt step size based on convergence behavior.
        
        Args:
            ds: Current step size
            newton_iters: Number of Newton iterations used
            converged: Whether Newton converged
        
        Returns:
            New step size
        """
        if not self.adaptive_stepsize:
            return ds
        
        if not converged:
            # Reduce step size if Newton failed
            return max(ds * 0.5, self.ds_min)
        
        # Adjust based on Newton iterations
        if newton_iters < 3:
            # Fast convergence - increase step size
            return min(ds * 1.5, self.ds_max)
        elif newton_iters > 6:
            # Slow convergence - decrease step size
            return max(ds * 0.8, self.ds_min)
        
        return ds
    
    def run(
        self,
        problem: ContinuationProblem,
        param_range: Tuple[float, float]
    ) -> ContinuationSolution:
        """
        Run the continuation algorithm.
        
        Args:
            problem: Continuation problem
            param_range: (min, max) parameter range
        
        Returns:
            ContinuationSolution object
        """
        from jax import jacfwd
        
        # Initialize storage
        states = [problem.u0]
        parameters = [problem.param_value]
        tangents = []
        convergence_info = []
        eigenvalues_list = [] if (self.detect_bifurcations or self.compute_stability) else None
        stability_list = [] if self.compute_stability else None
        
        # Initial tangent
        u_current = problem.u0
        param_current = problem.param_value
        tangent = self.compute_tangent(problem, u_current, param_current)
        tangents.append(tangent)
        
        # Compute initial eigenvalues if needed
        if self.detect_bifurcations or self.compute_stability:
            eigs = self._compute_eigenvalues(problem, u_current, param_current)
            eigenvalues_list.append(eigs)
            if self.compute_stability:
                is_stable = self._check_stability(eigs, problem.problem_type)
                stability_list.append(is_stable)
        
        # Determine direction
        param_min, param_max = param_range
        direction = 1.0 if param_max > param_current else -1.0
        
        ds = self.ds * direction
        step = 0
        
        while step < self.max_steps:
            # Check if we've reached the parameter range
            if direction > 0 and param_current >= param_max:
                break
            if direction < 0 and param_current <= param_min:
                break
            
            # Predict
            u_pred, param_pred = self.predict(u_current, param_current, tangent, ds)
            
            # Correct
            u_new, param_new, converged, newton_iters = self.correct(
                problem, u_pred, param_pred, u_current, param_current, tangent, ds
            )
            
            # Store convergence info
            convergence_info.append({
                "step": step,
                "converged": converged,
                "newton_iters": newton_iters,
                "ds": abs(ds),
            })
            
            if converged:
                # Accept step
                states.append(u_new)
                parameters.append(param_new)
                
                # Compute eigenvalues if needed
                if self.detect_bifurcations or self.compute_stability:
                    eigs = self._compute_eigenvalues(problem, u_new, param_new)
                    eigenvalues_list.append(eigs)
                    if self.compute_stability:
                        is_stable = self._check_stability(eigs, problem.problem_type)
                        stability_list.append(is_stable)
                
                # Compute new tangent
                tangent = self.compute_tangent(problem, u_new, param_new, tangent)
                tangents.append(tangent)
                
                # Update current point
                u_current = u_new
                param_current = param_new
                
                step += 1
            
            # Adapt step size
            ds_new = self.adapt_stepsize(abs(ds), newton_iters, converged)
            ds = ds_new * direction
            
            # Check minimum step size. Note the "<=": adapt_stepsize clamps a
            # shrinking ds to exactly self.ds_min (via max(ds*0.5, ds_min)), so
            # a strict "<" here would never fire once ds saturates at the
            # floor -- if the corrector keeps failing at ds_min, ds stays
            # pinned there and the loop would spin forever.
            if abs(ds) <= self.ds_min and not converged:
                print(f"Warning: Step size below minimum ({self.ds_min}), stopping.")
                break
        
        # Convert to arrays
        states_array = jnp.array(states)
        parameters_array = jnp.array(parameters)
        tangents_array = jnp.array(tangents) if tangents else None
        eigenvalues_array = jnp.array(eigenvalues_list) if eigenvalues_list else None
        stability_array = jnp.array(stability_list) if stability_list else None
        
        # Create solution object
        solution = ContinuationSolution(
            states=states_array,
            parameters=parameters_array,
            tangent_vectors=tangents_array,
            eigenvalues=eigenvalues_array,
            stability=stability_array,
            convergence_info=convergence_info,
        )
        
        # Detect bifurcations
        if self.detect_bifurcations and eigenvalues_array is not None:
            bifurcations = self.bifurcation_detector.detect_along_branch(
                solution, eigenvalues_array
            )
            solution.bifurcations = bifurcations
            
            # Print bifurcation information if verbose
            if self.verbose and bifurcations:
                self._print_bifurcation_summary(bifurcations)
        
        return solution
    
    def _compute_eigenvalues(self, problem: ContinuationProblem, u: Array, param: float) -> Array:
        """
        Compute eigenvalues at a point for equilibrium problems.
        
        Args:
            problem: Continuation problem
            u: State vector
            param: Parameter value
        
        Returns:
            Array of eigenvalues
        """
        from jax import jacfwd
        
        if problem.problem_type == "equilibrium":
            # Compute Jacobian of RHS
            def rhs_func(state):
                return problem.evaluate_rhs(state, param)
            
            jac = jacfwd(rhs_func)(u)
            
            # Handle scalar case
            if jac.ndim == 0:
                return jnp.array([jac])
            
            eigs = jnp.linalg.eigvals(jac)
            return eigs
        else:
            # For periodic orbits, would compute Floquet multipliers
            return jnp.array([])
    
    def _check_stability(self, eigenvalues: Array, problem_type: str) -> bool:
        """
        Check stability based on eigenvalues.
        
        Args:
            eigenvalues: Array of eigenvalues or multipliers
            problem_type: Type of problem ('equilibrium', 'periodic', etc.)
        
        Returns:
            True if stable, False otherwise
        """
        if len(eigenvalues) == 0:
            return True
        
        if problem_type == "equilibrium":
            # Stable if all eigenvalues have negative real part
            return bool(jnp.all(jnp.real(eigenvalues) < 0))
        elif problem_type == "periodic":
            # Stable if all Floquet multipliers have magnitude < 1
            return bool(jnp.all(jnp.abs(eigenvalues) < 1))
        else:
            return True
    
    def _print_bifurcation_summary(self, bifurcations: List[Dict[str, Any]]):
        """
        Print summary of detected bifurcations.
        
        Args:
            bifurcations: List of bifurcation dictionaries
        """
        print(f"\n{'═'*70}")
        print(f"{'BIFURCATION ANALYSIS':^70}")
        print(f"{'═'*70}")
        print(f"Detected {len(bifurcations)} bifurcation point(s) during continuation:\n")
        
        for i, bif in enumerate(bifurcations, 1):
            bif_type = bif.get('type', 'unknown')
            param = bif.get('parameter', 0.0)
            state = bif.get('state', jnp.array([0.0]))
            
            # Format bifurcation type name
            type_names = {
                'fold': 'Fold (Branch Point)',
                'hopf': 'Hopf',
                'period-doubling': 'Period Doubling',
                'branch-point': 'Branch Point'
            }
            type_name = type_names.get(bif_type, bif_type.capitalize())
            
            print(f"Bifurcation #{i}:")
            print(f"  Type:      {type_name}")
            print(f"  Parameter: {param:.8f}")
            
            if isinstance(state, (list, jnp.ndarray)) and len(state) > 0:
                if len(state) == 1:
                    print(f"  State:     x = {state[0]:.8f}")
                else:
                    print(f"  State:     {state}")
            
            # Additional info for specific bifurcation types
            if bif_type == 'hopf' and 'frequency' in bif:
                print(f"  Frequency: ω = {bif['frequency']:.6f}")
            
            if 'index' in bif:
                idx = bif['index']
                print(f"  Location:  between steps {idx[0]} and {idx[1]}")
            
            print()
        
        print(f"{'═'*70}\n")
