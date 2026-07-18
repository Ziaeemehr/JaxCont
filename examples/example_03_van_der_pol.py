"""
Van der Pol equilibrium branch
==============================

Continue the equilibrium at the origin as the nonlinear damping parameter
changes. Periodic-orbit continuation is outside JaxCont's supported v0.1 API.
"""

import jax.numpy as jnp
from jaxcont import ContinuationProblem, equilibrium_continuation
import matplotlib.pyplot as plt


def van_der_pol_rhs(state, params):
    """
    Van der Pol oscillator.
    
    dx/dt = y
    dy/dt = mu * (1 - x^2) * y - x
    
    For mu > 0: stable limit cycle
    For mu = 0: simple harmonic oscillator
    """
    x, y = state
    mu = params['mu']
    
    dx = y
    dy = mu * (1.0 - x**2) * y - x
    
    return jnp.array([dx, dy])


def run_van_der_pol_example():
    """Run Van der Pol oscillator example."""
    print("=" * 60)
    print("Example 3: Van der Pol Oscillator")
    print("=" * 60)
    print("\nSystem: x'' - mu*(1-x^2)*x' + x = 0")
    
    # Start from origin (equilibrium)
    problem = ContinuationProblem(
        rhs=van_der_pol_rhs,
        u0=jnp.array([0.0, 0.0]),
        params={'mu': 0.0},
        continuation_param='mu',
        problem_type='equilibrium'
    )
    
    # Run continuation
    print("\nRunning equilibrium continuation from mu=0 to mu=5...")
    solution = equilibrium_continuation(
        problem,
        param_range=(0.0, 5.0),
        ds=0.05,
        max_steps=200
    )
    
    print(f"Continuation completed: {solution.n_points} points computed")
    print("Note: Equilibrium becomes unstable via Hopf bifurcation")
    print("      Limit cycle emerges (would need periodic continuation)")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bifurcation diagram
    solution.plot(state_index=0, ax=ax1)
    ax1.set_ylabel('x', fontsize=12)
    ax1.set_title('Bifurcation Diagram', fontsize=14)
    
    # Phase portrait at selected parameters
    from jaxcont.utils.plotting import plot_phase_portrait
    plot_phase_portrait(solution, state_indices=(0, 1), ax=ax2)
    
    plt.tight_layout()
    plt.savefig('van_der_pol.png', dpi=150, bbox_inches='tight')
    print("\nSaved to: van_der_pol.png")
    
    return solution


if __name__ == "__main__":
    solution = run_van_der_pol_example()
    plt.show()
