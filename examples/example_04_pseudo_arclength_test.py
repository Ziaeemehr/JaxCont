"""
Simple test to debug pseudo-arclength continuation.
"""

import jax.numpy as jnp
from jaxcont import ContinuationProblem, PseudoArclengthContinuation


def test_simple_linear():
    """Test pseudo-arclength on simplest system."""
    print("\n" + "="*70)
    print("Testing Pseudo-Arclength on Linear System: dx/dt = r - x")
    print("="*70)
    
    def rhs(state, params):
        x = state[0]
        r = params["r"]
        return jnp.array([r - x])
    
    problem = ContinuationProblem(
        rhs=rhs,
        u0=jnp.array([0.0]),
        params={"r": 0.0},
        continuation_param="r",
        problem_type="equilibrium",
    )
    
    cont = PseudoArclengthContinuation(newton_tol=1e-8, newton_max_iter=100)
    
    u = problem.u0
    param = problem.params[problem.continuation_param]
    
    print(f"\nStarting point: r={param:.6f}, x={u[0]:.6f}")
    
    # Compute tangent
    tangent = cont.compute_tangent(problem, u, param)
    print(f"Initial tangent: {tangent}")
    print(f"Tangent norm: {jnp.linalg.norm(tangent):.6f}")
    
    # Take a few steps
    ds = 0.1
    for step in range(10):
        print(f"\n--- Step {step+1} ---")
        print(f"Current: r={param:.6f}, x={u[0]:.6f}")
        
        # Predict
        u_pred, param_pred = cont.predict(u, param, tangent, ds)
        print(f"Predicted: r={param_pred:.6f}, x={u_pred[0]:.6f}")
        
        # Correct
        u_new, param_new, converged, n_iter = cont.correct(
            problem, u_pred, param_pred, u, param, tangent, ds
        )
        
        print(f"Corrected: r={param_new:.6f}, x={u_new[0]:.6f}")
        print(f"Converged: {converged}, Iterations: {n_iter}")
        
        if not converged:
            # Check residual
            residual = jnp.linalg.norm(problem.evaluate_rhs(u_new, param_new))
            print(f"WARNING: Not converged! Residual: {residual:.2e}")
            
            # Check arclength constraint
            g_val = jnp.dot(u_new - u, tangent[:-1]) + (param_new - param) * tangent[-1] - ds
            print(f"Arclength constraint error: {abs(g_val):.2e}")
            break
        
        # Verify solution
        error = abs(u_new[0] - param_new)
        print(f"Error (x should equal r): {error:.2e}")
        
        # Update
        u = u_new
        param = param_new
        tangent = cont.compute_tangent(problem, u, param, tangent)
        
        if param > 1.0:
            print(f"\nReached target r=1.0")
            break
    
    print("\n" + "="*70)


def test_quadratic():
    """Test pseudo-arclength on quadratic system."""
    print("\n" + "="*70)
    print("Testing Pseudo-Arclength on Quadratic System: dx/dt = r - x^2")
    print("="*70)
    
    def rhs(state, params):
        x = state[0]
        r = params["r"]
        return jnp.array([r - x**2])
    
    r0 = 0.5
    x0 = jnp.sqrt(r0)
    
    problem = ContinuationProblem(
        rhs=rhs,
        u0=jnp.array([x0]),
        params={"r": r0},
        continuation_param="r",
        problem_type="equilibrium",
    )
    
    cont = PseudoArclengthContinuation(newton_tol=1e-8, newton_max_iter=100)
    
    u = problem.u0
    param = problem.params[problem.continuation_param]
    
    print(f"\nStarting point: r={param:.6f}, x={u[0]:.6f}")
    print(f"Expected: x = sqrt(r) = {jnp.sqrt(param):.6f}")
    
    # Compute tangent
    tangent = cont.compute_tangent(problem, u, param)
    print(f"Initial tangent: {tangent}")
    
    # Take a few steps
    ds = 0.05
    for step in range(5):
        print(f"\n--- Step {step+1} ---")
        print(f"Current: r={param:.6f}, x={u[0]:.6f}")
        
        # Predict
        u_pred, param_pred = cont.predict(u, param, tangent, ds)
        print(f"Predicted: r={param_pred:.6f}, x={u_pred[0]:.6f}")
        
        # Correct
        u_new, param_new, converged, n_iter = cont.correct(
            problem, u_pred, param_pred, u, param, tangent, ds
        )
        
        print(f"Corrected: r={param_new:.6f}, x={u_new[0]:.6f}")
        print(f"Converged: {converged}, Iterations: {n_iter}")
        
        if not converged:
            residual = jnp.linalg.norm(problem.evaluate_rhs(u_new, param_new))
            print(f"WARNING: Not converged! Residual: {residual:.2e}")
            
            g_val = jnp.dot(u_new - u, tangent[:-1]) + (param_new - param) * tangent[-1] - ds
            print(f"Arclength constraint error: {abs(g_val):.2e}")
            break
        
        # Verify solution
        expected = jnp.sqrt(param_new) if param_new > 0 else 0
        error = abs(u_new[0] - expected)
        print(f"Error (x should equal sqrt(r)): {error:.2e}")
        
        # Update
        u = u_new
        param = param_new
        tangent = cont.compute_tangent(problem, u, param, tangent)
    
    print("\n" + "="*70)


if __name__ == "__main__":
    test_simple_linear()
    test_quadratic()
