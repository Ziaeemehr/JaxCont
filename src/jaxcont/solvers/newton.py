"""
Newton's method for solving nonlinear systems.
"""

from typing import Callable, Tuple
import jax.numpy as jnp
from jax import Array, jacfwd, jit, lax


class NewtonSolver:
    """
    Newton's method for solving f(x) = 0.
    
    Uses JAX's automatic differentiation for computing Jacobians.
    JIT-compiled for high performance.
    """
    
    def __init__(
        self,
        tol: float = 1e-6,
        max_iter: int = 20,
    ):
        """
        Initialize Newton solver.
        
        Args:
            tol: Convergence tolerance
            max_iter: Maximum number of iterations
        """
        self.tol = tol
        self.max_iter = max_iter
    
    def solve(
        self,
        f: Callable[[Array], Array],
        x0: Array
    ) -> Tuple[Array, bool, int]:
        """
        Solve f(x) = 0 using Newton's method (JIT-compiled).
        
        Args:
            f: Function to find root of
            x0: Initial guess
        
        Returns:
            (solution, converged, n_iterations) tuple
        """
        return self._solve_jit(f, x0)
    
    def _solve_jit(
        self,
        f: Callable[[Array], Array],
        x0: Array
    ) -> Tuple[Array, bool, int]:
        """JIT-compiled implementation using JAX control flow."""
        
        def newton_step(carry):
            """Single Newton iteration."""
            x, iteration, converged, residual = carry
            
            # Evaluate function
            f_val = f(x)
            
            # Compute Jacobian using automatic differentiation
            jac = jacfwd(f)(x)
            
            # Solve linear system: J * dx = -f
            dx = jnp.linalg.solve(jac, -f_val)
            
            # Update solution
            x_new = x + dx
            
            # Compute new residual with updated x
            f_val_new = f(x_new)
            residual_new = jnp.linalg.norm(f_val_new)
            
            # Check convergence
            converged_new = residual_new < self.tol
            
            return x_new, iteration + 1, converged_new, residual_new
        
        def cond_fun(carry):
            """Continue if not converged and within max iterations."""
            x, iteration, converged, residual = carry
            return jnp.logical_and(
                jnp.logical_not(converged),
                iteration < self.max_iter
            )
        
        # Initial state
        initial_residual = jnp.linalg.norm(f(x0))
        initial_converged = initial_residual < self.tol
        initial_carry = (x0, 0, initial_converged, initial_residual)
        
        # Run Newton iterations using JAX while_loop for JIT compatibility
        final_x, final_iter, final_converged, final_residual = lax.while_loop(
            cond_fun, newton_step, initial_carry
        )
        
        return final_x, final_converged, final_iter
    
    def solve_with_jacobian(
        self,
        f: Callable[[Array], Array],
        jac: Callable[[Array], Array],
        x0: Array
    ) -> Tuple[Array, bool, int]:
        """
        Solve f(x) = 0 with user-provided Jacobian (JIT-compiled).
        
        Args:
            f: Function to find root of
            jac: Jacobian function
            x0: Initial guess
        
        Returns:
            (solution, converged, n_iterations) tuple
        """
        return self._solve_with_jacobian_jit(f, jac, x0)
    
    def _solve_with_jacobian_jit(
        self,
        f: Callable[[Array], Array],
        jac: Callable[[Array], Array],
        x0: Array
    ) -> Tuple[Array, bool, int]:
        """JIT-compiled implementation with user-provided Jacobian."""
        
        def newton_step(carry):
            """Single Newton iteration."""
            x, iteration, converged, residual = carry
            
            # Evaluate function
            f_val = f(x)
            
            # Compute Jacobian
            jac_val = jac(x)
            
            # Solve linear system: J * dx = -f
            dx = jnp.linalg.solve(jac_val, -f_val)
            
            # Update solution
            x_new = x + dx
            
            # Compute new residual with updated x
            f_val_new = f(x_new)
            residual_new = jnp.linalg.norm(f_val_new)
            
            # Check convergence
            converged_new = residual_new < self.tol
            
            return x_new, iteration + 1, converged_new, residual_new
        
        def cond_fun(carry):
            """Continue if not converged and within max iterations."""
            x, iteration, converged, residual = carry
            return jnp.logical_and(
                jnp.logical_not(converged),
                iteration < self.max_iter
            )
        
        # Initial state
        initial_residual = jnp.linalg.norm(f(x0))
        initial_converged = initial_residual < self.tol
        initial_carry = (x0, 0, initial_converged, initial_residual)
        
        # Run Newton iterations using JAX while_loop for JIT compatibility
        final_x, final_iter, final_converged, final_residual = lax.while_loop(
            cond_fun, newton_step, initial_carry
        )
        
        return final_x, final_converged, final_iter
