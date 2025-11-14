"""
Periodic orbit problem using shooting or collocation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Literal
import jax.numpy as jnp
from jax import Array
from scipy.integrate import solve_ivp


@dataclass
class PeriodicOrbitProblem:
    """
    Defines a periodic orbit continuation problem.
    
    Find periodic solutions u(t) such that u(T) = u(0), where T is the period.
    
    Methods:
        - shooting: Single shooting method
        - multiple_shooting: Multiple shooting for better stability
        - collocation: Orthogonal collocation
    
    Attributes:
        rhs: Right-hand side function f(u, params)
        params: System parameters
        initial_state: Initial point on periodic orbit
        initial_period: Initial period guess
        continuation_param: Parameter to continue in
        method: Solution method ('shooting', 'multiple_shooting', 'collocation')
        n_mesh: Number of mesh points (for multiple shooting/collocation)
    """
    rhs: Callable[[Array, Dict[str, float]], Array]
    params: Dict[str, float]
    initial_state: Array
    initial_period: float
    continuation_param: str
    method: Literal["shooting", "multiple_shooting", "collocation"] = "shooting"
    n_mesh: int = 20
    
    def shooting_residual(self, u0: Array, period: float, param_value: float) -> Array:
        """
        Shooting method residual: F(u0, T) = Phi(T, u0) - u0
        
        Args:
            u0: Initial state
            period: Period
            param_value: Parameter value
        
        Returns:
            Residual vector
        """
        params = self.params.copy()
        params[self.continuation_param] = param_value
        
        # Integrate from 0 to T
        def ode_rhs(t, u):
            return self.rhs(jnp.array(u), params)
        
        sol = solve_ivp(
            ode_rhs,
            (0, period),
            u0,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        u_T = jnp.array(sol.y[:, -1])
        
        # Residual: u(T) - u(0) = 0
        return u_T - u0
    
    def phase_condition(self, u0: Array, u0_ref: Array) -> float:
        """
        Phase condition to fix the phase of the periodic orbit.
        
        Typically: <u0 - u0_ref, du0_ref/dt> = 0
        
        Args:
            u0: Current initial state
            u0_ref: Reference initial state
        
        Returns:
            Phase condition value
        """
        # Simple phase condition: fix first component
        return u0[0] - u0_ref[0]
    
    def to_continuation_problem(self):
        """Convert to ContinuationProblem format."""
        from jaxcont.core.continuation import ContinuationProblem
        
        # For periodic orbits, the state includes both u0 and T
        # We'll encode this as [u0..., T]
        u0_extended = jnp.concatenate([self.initial_state, jnp.array([self.initial_period])])
        
        def extended_rhs(state_extended, params):
            """Extended RHS for periodic orbit problem."""
            u0 = state_extended[:-1]
            period = state_extended[-1]
            
            param_val = params[self.continuation_param]
            
            # Shooting residual
            residual = self.shooting_residual(u0, period, param_val)
            
            # Phase condition (fix first component for now)
            phase_cond = u0[0] - self.initial_state[0]
            
            # Combined residual
            return jnp.concatenate([residual, jnp.array([phase_cond])])
        
        return ContinuationProblem(
            rhs=extended_rhs,
            u0=u0_extended,
            params=self.params,
            continuation_param=self.continuation_param,
            problem_type="periodic"
        )
