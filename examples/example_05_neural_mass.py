"""
Neural-mass equilibrium continuation
=====================================

Translated from BifurcationKit.jl example

This example demonstrates:
- 3D neural mass model from computational neuroscience
- Equilibrium continuation in parameter E0
- Automatic bifurcation detection (Hopf bifurcations expected)
- Comparison with BifurcationKit.jl results

This example tracks how steady neural activity changes with external input.
"""

import os
import jax.numpy as jnp
from jaxcont import ContinuationProblem, equilibrium_continuation
import matplotlib.pyplot as plt

path = "images"
os.makedirs(path, exist_ok=True)

LABELSIZE = 13
plt.rc("axes", labelsize=LABELSIZE)
plt.rc("axes", titlesize=LABELSIZE)
plt.rc("figure", titlesize=LABELSIZE)
plt.rc("legend", fontsize=LABELSIZE)
plt.rc("xtick", labelsize=LABELSIZE)
plt.rc("ytick", labelsize=LABELSIZE)
plt.rc("legend", fontsize=10)


def TMvf(state, params):
    """
    Tamir's neural mass model vector field.
    
    Variables:
    - E: neural activity
    - x: recovery variable  
    - u: adaptation variable
    
    Parameters:
    - J: synaptic coupling strength
    - α: gain parameter
    - E0: external input (bifurcation parameter)
    - τ: membrane time constant
    - τD: depression time constant
    - τF: facilitation time constant
    - U0: facilitation strength
    """
    # Extract state variables
    E, x, u = state[0], state[1], state[2]
    
    # Extract parameters
    J = params['J']
    α = params['α'] 
    E0 = params['E0']
    τ = params['τ']
    τD = params['τD']
    τF = params['τF']
    U0 = params['U0']
    
    # Compute synaptic input
    SS0 = J * u * x * E + E0
    SS1 = α * jnp.log(1 + jnp.exp(SS0 / α))
    
    # System equations
    dE_dt = (-E + SS1) / τ
    dx_dt = (1.0 - x) / τD - u * x * E  
    du_dt = (U0 - u) / τF + U0 * (1.0 - u) * E
    
    return jnp.array([dE_dt, dx_dt, du_dt])


def find_equilibrium(params, initial_guess):
    """Find equilibrium point for given parameters using Newton's method."""
    from jaxcont.solvers.newton import NewtonSolver
    
    def equilibrium_residual(state):
        return TMvf(state, params)
    
    solver = NewtonSolver(tol=1e-10, max_iter=50)
    equilibrium, converged, n_iter = solver.solve(equilibrium_residual, initial_guess)
    
    if converged:
        residual_norm = jnp.linalg.norm(TMvf(equilibrium, params))
        print(f"Found equilibrium: E={equilibrium[0]:.6f}, x={equilibrium[1]:.6f}, u={equilibrium[2]:.6f}")
        print(f"Residual norm: {residual_norm:.2e} (converged in {n_iter} iterations)")
        return equilibrium
    else:
        print(f"Failed to find equilibrium (after {n_iter} iterations)")
        return initial_guess


def run_neural_mass_example():
    """Run neural mass model continuation example."""
    print("=" * 70)
    print("Example 5: Neural Mass Model")
    print("=" * 70)
    print("\nSystem: 3D neural mass model")
    print("Variables: E (activity), x (recovery), u (adaptation)")
    print("Continuing EQUILIBRIA in parameter E0 (external input)")
    print("\nWhy equilibrium continuation?")
    print("- We seek steady states where dE/dt = dx/dt = du/dt = 0")
    print("- These represent stable neural activity patterns")
    print("- We track how these equilibria change as external input E0 varies")
    print("\nExpected: Hopf bifurcations for E0 ∈ [-4.0, -0.9]")

    # Parameter values from Julia code
    params = {
        'α': 1.5,
        'τ': 0.013, 
        'J': 3.07,
        'E0': -2.0,  # This will be the continuation parameter
        'τD': 0.200,
        'U0': 0.3,
        'τF': 1.5,
        'τS': 0.007
    }
    
    # Initial condition from Julia code (but need to find actual equilibrium)
    z0_guess = jnp.array([0.238616, 0.982747, 0.367876])
    
    print(f"\nInitial guess: E = {z0_guess[0]:.6f}, x = {z0_guess[1]:.6f}, u = {z0_guess[2]:.6f}")
    print(f"Initial E0 = {params['E0']}")
    
    # Check if initial guess is equilibrium
    rhs_initial = TMvf(z0_guess, params)
    residual_norm = jnp.linalg.norm(rhs_initial)
    print(f"RHS at initial point: [{rhs_initial[0]:.2e}, {rhs_initial[1]:.2e}, {rhs_initial[2]:.2e}]")
    print(f"Residual norm: {residual_norm:.2e} (should be ≈ 0 for equilibrium)")
    
    # Find actual equilibrium
    if residual_norm > 1e-6:
        print("\nFinding equilibrium...")
        z0 = find_equilibrium(params, z0_guess)
    else:
        z0 = z0_guess
        print("Initial guess is already at equilibrium!")

    # Define problem 
    # Note: problem_type='equilibrium' tells JaxCont we want to find steady states
    # where f(u, p) = 0, not track solution curves of du/dt = f(u, p)
    problem = ContinuationProblem(
        rhs=TMvf,
        u0=z0,
        params=params,
        continuation_param='E0'
        # problem_type is automatically inferred from using equilibrium_continuation()
    )

    print("\nRunning continuation from E0=-4.0 to E0=-0.9...")
    print("(Bifurcation detection enabled)\n")

    # Run continuation with both directions like in Julia (bothside=true)
    solution = equilibrium_continuation(
        problem,
        param_range=(-4.0, -0.9),
        ds=0.02,  # Reasonable step size
        max_steps=400,
        detect_bifurcations=True,
        compute_stability=True,
        verbose=True,
        bifurcation_tolerance=1e-4,
        newton_tol=1e-8
    )

    print(f"{'─'*70}")
    print(f"Continuation completed: {solution.n_points} points computed")
    
    if solution.bifurcations:
        print(f"Found {len(solution.bifurcations)} bifurcation(s):")
        for i, bif in enumerate(solution.bifurcations, 1):
            param = bif['parameter']
            state = bif['state']
            print(f"  Bifurcation #{i}: E0 = {param:.6f}")
            print(f"    State: E = {state[0]:.6f}, x = {state[1]:.6f}, u = {state[2]:.6f}")
    else:
        print("No bifurcations detected")
    print(f"{'─'*70}")

    # Plot results
    print("Plotting bifurcation diagram...")
    fig = plot_neural_mass_diagram(solution)
    plt.savefig(f"{path}/neural_mass_bifurcation.png", dpi=150, bbox_inches='tight')
    print(f"Saved to: {path}/neural_mass_bifurcation.png")
    print(f"{'═'*70}\n")

    return solution


def plot_neural_mass_diagram(solution):
    """Plot bifurcation diagram for neural mass model."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Extract data
    params = solution.parameters
    states = solution.states
    
    # Variable names and labels
    var_names = ['E', 'x', 'u']
    var_labels = ['Neural Activity E', 'Recovery Variable x', 'Adaptation Variable u']
    colors = ['blue', 'green', 'red']
    
    for i, (ax, name, label, color) in enumerate(zip(axes, var_names, var_labels, colors)):
        # Plot the branch
        ax.plot(params, states[:, i], color=color, linewidth=2, 
                label=f'{name} equilibrium', alpha=0.7)
        ax.plot(params, states[:, i], '.', color=color, markersize=3, alpha=0.5)
        
        # Mark bifurcation points
        if solution.bifurcations:
            for bif in solution.bifurcations:
                param = bif['parameter']
                state = bif['state'][i]
                ax.plot(param, state, 'rs', markersize=10, 
                       markeredgewidth=2, markerfacecolor='red',
                       markeredgecolor='darkred', zorder=10)
                
                # Add annotation for first subplot only
                if i == 0:
                    ax.annotate(f'Bifurcation\nE0={param:.3f}',
                              xy=(param, state), xytext=(15, 15),
                              textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                              arrowprops=dict(arrowstyle='->', color='red'),
                              fontsize=9)
        
        # Styling
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')
        
        # Only add x-label to bottom plot
        if i == 2:
            ax.set_xlabel('External Input E0')
    
    plt.suptitle('Neural Mass Model - Bifurcation Diagram\n' + 
                 'Variables vs External Input E0', fontsize=14)
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    solution = run_neural_mass_example()
    plt.show()
