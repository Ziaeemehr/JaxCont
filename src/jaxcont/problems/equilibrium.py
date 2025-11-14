"""
Equilibrium problem definition.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional
import jax.numpy as jnp
from jax import Array, jacfwd


@dataclass
class EquilibriumProblem:
    """
    Defines an equilibrium continuation problem.
    
    Find solutions to: f(u, p) = 0
    
    Attributes:
        rhs: Right-hand side function f(u, params)
        params: System parameters
        initial_state: Initial equilibrium point
        continuation_param: Parameter to continue in
    """
    rhs: Callable[[Array, Dict[str, float]], Array]
    params: Dict[str, float]
    initial_state: Array
    continuation_param: str
    
    def residual(self, u: Array, param_value: float) -> Array:
        """
        Evaluate residual f(u, p) at given state and parameter.
        
        Args:
            u: State vector
            param_value: Parameter value
        
        Returns:
            Residual vector
        """
        params = self.params.copy()
        params[self.continuation_param] = param_value
        return self.rhs(u, params)
    
    def jacobian(self, u: Array, param_value: float) -> Array:
        """
        Compute Jacobian df/du at given state and parameter.
        
        Args:
            u: State vector
            param_value: Parameter value
        
        Returns:
            Jacobian matrix
        """
        def f(u_eval):
            return self.residual(u_eval, param_value)
        
        return jacfwd(f)(u)
    
    def parameter_derivative(self, u: Array, param_value: float, eps: float = 1e-7) -> Array:
        """
        Compute df/dp using finite differences.
        
        Args:
            u: State vector
            param_value: Parameter value
            eps: Finite difference step
        
        Returns:
            Parameter derivative vector
        """
        f_plus = self.residual(u, param_value + eps)
        f_minus = self.residual(u, param_value - eps)
        return (f_plus - f_minus) / (2 * eps)
    
    def to_continuation_problem(self):
        """Convert to ContinuationProblem format."""
        from jaxcont.core.continuation import ContinuationProblem
        
        return ContinuationProblem(
            rhs=self.rhs,
            u0=self.initial_state,
            params=self.params,
            continuation_param=self.continuation_param,
            problem_type="equilibrium"
        )
