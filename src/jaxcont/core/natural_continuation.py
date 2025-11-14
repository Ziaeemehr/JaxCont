"""
Natural parameter continuation (simplest continuation method).
"""

from typing import Tuple, Optional
import jax.numpy as jnp
from jax import Array, jacfwd

from jaxcont.core.predictor_corrector import PredictorCorrector
from jaxcont.core.continuation import ContinuationProblem
from jaxcont.solvers.newton import NewtonSolver


class NaturalContinuation(PredictorCorrector):
    """
    Natural parameter continuation.
    
    This is the simplest continuation method where the parameter is 
    incremented by a fixed amount and Newton's method is used to find
    the corresponding equilibrium.
    
    Limitation: Cannot pass turning points (fold bifurcations).
    """
    
    def predict(
        self,
        u: Array,
        param: float,
        tangent: Array,
        ds: float
    ) -> Tuple[Array, float]:
        """
        Predict next point by incrementing parameter and keeping state constant.
        
        Args:
            u: Current state
            param: Current parameter value
            tangent: Tangent vector (unused in natural continuation)
            ds: Step size in parameter
        
        Returns:
            (u, param + ds) tuple
        """
        return u, param + ds
    
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
        Correct prediction by solving f(u, param_pred) = 0 using Newton.
        
        Args:
            problem: Continuation problem
            u_pred: Predicted state (initial guess)
            param_pred: Predicted parameter (fixed)
            u_prev: Previous state (unused)
            param_prev: Previous parameter (unused)
            tangent: Tangent vector (unused)
            ds: Step size (unused)
        
        Returns:
            (corrected_state, param_pred, converged, iterations) tuple
        """
        newton = NewtonSolver(
            tol=self.newton_tol,
            max_iter=self.newton_max_iter
        )
        
        # Define residual function at fixed parameter
        def residual(u):
            return problem.evaluate_rhs(u, param_pred)
        
        # Solve using Newton
        u_corrected, converged, n_iter = newton.solve(residual, u_pred)
        
        return u_corrected, param_pred, converged, n_iter
    
    def compute_tangent(
        self,
        problem: ContinuationProblem,
        u: Array,
        param: float,
        prev_tangent: Optional[Array] = None
    ) -> Array:
        """
        Compute tangent vector for natural continuation.
        
        The tangent is simply [du/dparam, 1] where du/dparam is computed
        from the implicit function theorem.
        
        Args:
            problem: Continuation problem
            u: Current state
            param: Current parameter value
            prev_tangent: Previous tangent (for orientation)
        
        Returns:
            Tangent vector [du/dparam, 1] (normalized)
        """
        # Compute Jacobian df/du
        def f_u(u_eval):
            return problem.evaluate_rhs(u_eval, param)
        
        jac_u = jacfwd(f_u)(u)
        
        # Compute df/dparam by finite difference
        eps = 1e-6
        f_p_plus = problem.evaluate_rhs(u, param + eps)
        f_p_minus = problem.evaluate_rhs(u, param - eps)
        df_dparam = (f_p_plus - f_p_minus) / (2 * eps)
        
        # Solve: df/du * du/dparam + df/dparam = 0
        # => du/dparam = -(df/du)^{-1} * df/dparam
        try:
            du_dparam = -jnp.linalg.solve(jac_u, df_dparam)
        except:
            # If singular, use small random perturbation
            du_dparam = jnp.zeros_like(u)
        
        # Tangent vector is [du/dparam, 1]
        tangent = jnp.concatenate([du_dparam, jnp.array([1.0])])
        
        # Normalize
        tangent = tangent / jnp.linalg.norm(tangent)
        
        # Orient with previous tangent if available
        if prev_tangent is not None:
            if jnp.dot(tangent, prev_tangent) < 0:
                tangent = -tangent
        
        return tangent
