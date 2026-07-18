"""
Natural-parameter continuation
==============================

This script tests the natural parameter continuation method on several
simple ODE systems to validate the implementation.
"""

import jax.numpy as jnp
from jaxcont import ContinuationProblem, NaturalContinuation, NewtonSolver
import matplotlib.pyplot as plt


def test_1_simple_linear():
    """
    Test 1: Simple linear system
    dx/dt = r - x
    
    Exact solution: x = r
    This should work perfectly with natural continuation.
    """
    print("\n" + "="*70)
    print("TEST 1: Linear System - dx/dt = r - x")
    print("="*70)
    
    def rhs(state, params):
        x = state[0]
        r = params["r"]
        return jnp.array([r - x])
    
    # Define problem
    problem = ContinuationProblem(
        rhs=rhs,
        u0=jnp.array([0.0]),
        params={"r": 0.0},
        continuation_param="r",
        problem_type="equilibrium",
    )
    
    # Create natural continuation solver
    cont = NaturalContinuation(newton_tol=1e-10, newton_max_iter=100)
    
    # Manual continuation loop
    u = problem.u0
    param = problem.params[problem.continuation_param]
    param_values = []
    state_values = []
    
    print(f"Starting at r={param:.6f}, x={u[0]:.6f}")
    print(f"Expected: x = r (exact solution)")
    print("\nContinuing from r=0 to r=1 with ds=0.1...")
    
    ds = 0.1
    param_end = 1.0
    step = 0
    max_steps = 15
    
    while param < param_end and step < max_steps:
        # Store current point
        param_values.append(param)
        state_values.append(u[0])
        
        # Predict
        tangent = cont.compute_tangent(problem, u, param)
        u_pred, param_pred = cont.predict(u, param, tangent, ds)
        
        # Correct
        u_new, param_new, converged, n_iter = cont.correct(
            problem, u_pred, param_pred, u, param, tangent, ds
        )
        
        if not converged:
            print(f"WARNING: Step {step} did not converge!")
            break
        
        # Update
        u = u_new
        param = param_new
        step += 1
        
        error = abs(u[0] - param)
        print(f"Step {step:2d}: r={param:.6f}, x={u[0]:.6f}, error={error:.2e}, iters={n_iter}")
    
    # Store final point
    param_values.append(param)
    state_values.append(u[0])
    
    # Check accuracy
    max_error = max([abs(x - r) for x, r in zip(state_values, param_values)])
    print(f"\nMaximum error: {max_error:.2e}")
    print("✓ PASS" if max_error < 1e-8 else "✗ FAIL")
    
    return param_values, state_values


def test_2_quadratic():
    """
    Test 2: Quadratic system
    dx/dt = r - x^2
    
    Exact solution: x = sqrt(r) for r > 0
    Fold bifurcation at r = 0
    """
    print("\n" + "="*70)
    print("TEST 2: Quadratic System - dx/dt = r - x^2")
    print("="*70)
    
    def rhs(state, params):
        x = state[0]
        r = params["r"]
        return jnp.array([r - x**2])
    
    # Start from r=0.1 (away from bifurcation)
    r0 = 0.1
    x0 = jnp.sqrt(r0)
    
    problem = ContinuationProblem(
        rhs=rhs,
        u0=jnp.array([x0]),
        params={"r": r0},
        continuation_param="r",
        problem_type="equilibrium",
    )
    
    cont = NaturalContinuation(newton_tol=1e-10, newton_max_iter=50)
    
    u = problem.u0
    param = problem.params[problem.continuation_param]
    param_values = []
    state_values = []
    errors = []
    
    print(f"Starting at r={param:.6f}, x={u[0]:.6f}")
    print(f"Expected: x = sqrt(r)")
    print("\nContinuing from r=0.1 to r=1.0 with ds=0.05...")
    
    ds = 0.05
    param_end = 1.0
    step = 0
    max_steps = 30
    
    while param < param_end and step < max_steps:
        param_values.append(param)
        state_values.append(u[0])
        
        # Predict
        tangent = cont.compute_tangent(problem, u, param)
        u_pred, param_pred = cont.predict(u, param, tangent, ds)
        
        # Correct
        u_new, param_new, converged, n_iter = cont.correct(
            problem, u_pred, param_pred, u, param, tangent, ds
        )
        
        if not converged:
            residual = jnp.linalg.norm(problem.evaluate_rhs(u_new, param_new))
            print(f"  WARNING: Newton did not converge! Residual={residual:.2e}")
            if residual > 1e-6:
                print(f"  ERROR: Residual too large, stopping!")
                break
            else:
                print(f"  But residual is acceptable, continuing...")
        
        u = u_new
        param = param_new
        step += 1
        
        # Compute error
        expected = jnp.sqrt(param)
        error = abs(u[0] - expected)
        errors.append(error)
        
        print(f"Step {step:2d}: r={param:.6f}, x={u[0]:.6f}, expected={expected:.6f}, error={error:.2e}, iters={n_iter}")
    
    param_values.append(param)
    state_values.append(u[0])
    
    max_error = max(errors)
    print(f"\nMaximum error: {max_error:.2e}")
    print("✓ PASS" if max_error < 1e-6 else "✗ FAIL")
    
    return param_values, state_values


def test_3_cubic():
    """
    Test 3: Cubic system (pitchfork-like)
    dx/dt = r*x - x^3
    
    For r > 0: three equilibria at x = 0, ±sqrt(r)
    For r < 0: one equilibrium at x = 0
    Bifurcation at r = 0
    """
    print("\n" + "="*70)
    print("TEST 3: Cubic System - dx/dt = r*x - x^3")
    print("="*70)
    
    def rhs(state, params):
        x = state[0]
        r = params["r"]
        return jnp.array([r * x - x**3])
    
    # Start on upper branch
    r0 = 0.5
    x0 = jnp.sqrt(r0)
    
    problem = ContinuationProblem(
        rhs=rhs,
        u0=jnp.array([x0]),
        params={"r": r0},
        continuation_param="r",
        problem_type="equilibrium",
    )
    
    cont = NaturalContinuation(newton_tol=1e-8, newton_max_iter=100)
    
    u = problem.u0
    param = problem.params[problem.continuation_param]
    param_values = []
    state_values = []
    errors = []
    
    print(f"Starting at r={param:.6f}, x={u[0]:.6f} (upper branch)")
    print(f"Expected: x = sqrt(r) for r > 0")
    print("\nContinuing from r=0.5 to r=2.0 with ds=0.1...")
    
    ds = 0.1
    param_end = 2.0
    step = 0
    max_steps = 30
    
    while param < param_end and step < max_steps:
        param_values.append(param)
        state_values.append(u[0])
        
        # Predict
        tangent = cont.compute_tangent(problem, u, param)
        u_pred, param_pred = cont.predict(u, param, tangent, ds)
        
        # Correct
        u_new, param_new, converged, n_iter = cont.correct(
            problem, u_pred, param_pred, u, param, tangent, ds
        )
        
        if not converged:
            residual = jnp.linalg.norm(problem.evaluate_rhs(u_new, param_new))
            print(f"  WARNING: Newton did not converge! Residual={residual:.2e}")
            if residual > 1e-6:
                print(f"  ERROR: Residual too large, stopping!")
                break
            else:
                print(f"  But residual is acceptable, continuing...")
        
        u = u_new
        param = param_new
        step += 1
        
        # Compute error (should stay on upper branch)
        expected = jnp.sqrt(param)
        error = abs(u[0] - expected)
        errors.append(error)
        
        print(f"Step {step:2d}: r={param:.6f}, x={u[0]:.6f}, expected={expected:.6f}, error={error:.2e}, iters={n_iter}")
    
    param_values.append(param)
    state_values.append(u[0])
    
    max_error = max(errors) if errors else 0.0
    print(f"\nMaximum error: {max_error:.2e}")
    print("✓ PASS" if max_error < 1e-6 else "✗ FAIL")
    
    return param_values, state_values


def plot_results(results):
    """Plot all test results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    titles = [
        "Test 1: Linear (x = r)",
        "Test 2: Quadratic (x = √r)",
        "Test 3: Cubic (x = √r)"
    ]
    
    for i, (ax, (params, states), title) in enumerate(zip(axes, results, titles)):
        ax.plot(params, states, 'bo-', markersize=6, linewidth=2, label='Computed')
        
        # Plot exact solution
        if i == 0:  # Linear
            ax.plot(params, params, 'r--', linewidth=2, label='Exact: x=r')
        elif i == 1:  # Quadratic
            exact = [jnp.sqrt(r) for r in params]
            ax.plot(params, exact, 'r--', linewidth=2, label='Exact: x=√r')
        elif i == 2:  # Cubic
            exact = [jnp.sqrt(r) for r in params]
            ax.plot(params, exact, 'r--', linewidth=2, label='Exact: x=√r')
        
        ax.set_xlabel('Parameter r', fontsize=11)
        ax.set_ylabel('State x', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('images/natural_continuation_tests.png', dpi=150, bbox_inches='tight')
    print("\n" + "="*70)
    print("Plot saved to: images/natural_continuation_tests.png")
    print("="*70)
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("NATURAL CONTINUATION - SIMPLE TESTS")
    print("="*70)
    
    # Run tests
    results = []
    results.append(test_1_simple_linear())
    results.append(test_2_quadratic())
    results.append(test_3_cubic())
    
    # Plot results
    plot_results(results)
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)
