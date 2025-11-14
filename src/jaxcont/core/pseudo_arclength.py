"""
Pseudo-arclength continuation (most robust method).
"""

from typing import Tuple, Optional
import jax.numpy as jnp
from jax import Array, jacfwd, jacobian

from jaxcont.core.predictor_corrector import PredictorCorrector
from jaxcont.core.continuation import ContinuationProblem
from jaxcont.solvers.newton import NewtonSolver


class PseudoArclengthContinuation(PredictorCorrector):
    """
    Pseudo-arclength continuation method.
    
    This is the most robust continuation method that can pass turning points
    (fold bifurcations) by parametrizing the solution branch by arclength
    rather than by a single parameter.
    
    The method solves:
        f(u, p) = 0
        g(u, p) = (u - u0)^T * du0 + (p - p0) * dp0 - ds = 0
    
    where (du0, dp0) is the tangent vector and ds is the arclength step.
    """
    
    def predict(
        self,
        u: Array,
        param: float,
        tangent: Array,
        ds: float
    ) -> Tuple[Array, float]:
        """
        Predict next point by stepping along the tangent.
        
        Args:
            u: Current state
            param: Current parameter value
            tangent: Tangent vector [du, dp]
            ds: Step size in arclength
        
        Returns:
            (u + ds*du, param + ds*dp) tuple
        """
        # Split tangent into state and parameter components
        du = tangent[:-1]
        dp = tangent[-1]
        
        u_pred = u + ds * du
        param_pred = param + ds * dp
        
        return u_pred, param_pred
    
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
        Correct prediction using bordered Newton system.
        
        The bordered system is:
            [ df/du    df/dp ] [ Delta_u ]   [ -f(u, p)              ]
            [ du0^T    dp0    ] [ Delta_p ] = [ -g(u, p) = ds - (...) ]
        
        Args:
            problem: Continuation problem
            u_pred: Predicted state
            param_pred: Predicted parameter
            u_prev: Previous state
            param_prev: Previous parameter
            tangent: Tangent vector [du0, dp0]
            ds: Step size
        
        Returns:
            (corrected_state, corrected_param, converged, iterations) tuple
        """
        # Extract tangent components
        du0 = tangent[:-1]
        dp0 = tangent[-1]
        
        # Current guess
        u_current = u_pred
        p_current = param_pred
        
        converged = False
        n_iter = 0
        
        for iteration in range(self.newton_max_iter):
            # Evaluate residual f(u, p)
            f_val = problem.evaluate_rhs(u_current, p_current)
            
            # Evaluate arclength constraint g(u, p)
            g_val = jnp.dot(u_current - u_prev, du0) + (p_current - param_prev) * dp0 - ds
            
            # Check convergence
            residual_norm = jnp.sqrt(jnp.sum(f_val**2) + g_val**2)
            if residual_norm < self.newton_tol:
                converged = True
                n_iter = iteration
                break
            
            # Compute Jacobian df/du
            def f_u(u_eval):
                return problem.evaluate_rhs(u_eval, p_current)
            
            jac_u = jacfwd(f_u)(u_current)
            
            # Compute df/dp by finite difference
            eps = 1e-7
            f_p_plus = problem.evaluate_rhs(u_current, p_current + eps)
            f_p_minus = problem.evaluate_rhs(u_current, p_current - eps)
            df_dp = (f_p_plus - f_p_minus) / (2 * eps)
            
            # Construct bordered system
            # [ jac_u   df_dp ] [ Delta_u ]   [ -f_val ]
            # [ du0^T   dp0   ] [ Delta_p ] = [ -g_val ]
            
            # Solve using block elimination
            # First solve: jac_u * w = -f_val
            try:
                w = jnp.linalg.solve(jac_u, -f_val)
            except:
                # Singular Jacobian, cannot continue
                break
            
            # Then solve: jac_u * v = df_dp
            try:
                v = jnp.linalg.solve(jac_u, df_dp)
            except:
                break
            
            # Now: Delta_p = (-g_val - du0^T * w) / (dp0 - du0^T * v)
            denominator = dp0 - jnp.dot(du0, v)
            if abs(denominator) < 1e-12:
                break
            
            delta_p = (-g_val - jnp.dot(du0, w)) / denominator
            
            # And: Delta_u = w - v * Delta_p
            delta_u = w - v * delta_p
            
            # Update
            u_current = u_current + delta_u
            p_current = p_current + delta_p
            
            n_iter = iteration + 1
        
        return u_current, p_current, converged, n_iter
    
    def compute_tangent(
        self,
        problem: ContinuationProblem,
        u: Array,
        param: float,
        prev_tangent: Optional[Array] = None
    ) -> Array:
        """
        Compute tangent vector to the solution branch.
        
        The tangent satisfies:
            df/du * du + df/dp * dp = 0
            ||[du, dp]|| = 1
        
        Args:
            problem: Continuation problem
            u: Current state
            param: Current parameter value
            prev_tangent: Previous tangent (for orientation)
        
        Returns:
            Normalized tangent vector [du, dp]
        """
        # Compute Jacobian df/du
        def f_u(u_eval):
            return problem.evaluate_rhs(u_eval, param)
        
        jac_u = jacfwd(f_u)(u)
        
        # Compute df/dp by finite difference
        eps = 1e-7
        f_p_plus = problem.evaluate_rhs(u, param + eps)
        f_p_minus = problem.evaluate_rhs(u, param - eps)
        df_dp = (f_p_plus - f_p_minus) / (2 * eps)
        
        # We need to find [du, dp] such that:
        # jac_u * du + df_dp * dp = 0 and ||[du, dp]|| = 1
        
        # From the first equation: du = -jac_u^{-1} * df_dp * dp
        # Let's set dp = 1 initially and normalize
        try:
            du_unnorm = -jnp.linalg.solve(jac_u, df_dp)
            dp_unnorm = 1.0
        except:
            # If singular, try dp = 0 and du along null space
            # For simplicity, use random direction
            du_unnorm = jnp.ones_like(u)
            dp_unnorm = 0.0
        
        # Combine and normalize
        tangent = jnp.concatenate([du_unnorm, jnp.array([dp_unnorm])])
        tangent = tangent / jnp.linalg.norm(tangent)
        
        # Orient with previous tangent if available
        if prev_tangent is not None:
            if jnp.dot(tangent, prev_tangent) < 0:
                tangent = -tangent
        
        return tangent
