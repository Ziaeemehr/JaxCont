"""
Equilibrium continuation of a cubic normal form
================================================

Modified system: dx/dt = r + x - x^3/3

This example demonstrates:
- Automatic bifurcation detection during continuation
- Branch points (fold bifurcations) at r ≈ ±2/3
- Plotting the bifurcation diagram and marking detected bifurcations
"""

import os
import jax.numpy as jnp
from jaxcont import ContinuationProblem, equilibrium_continuation
from jaxcont.utils.plotting import plot_continuation
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


def pitchfork_rhs(state, params):
    """
    Modified ODE system.

    dx/dt = r + x - x^3/3

    This is a modified form with additive parameter r.
    """
    x = state[0] if len(state.shape) == 1 and state.shape[0] > 1 else state
    r = params["r"]

    dxdt = r + x - x**3 / 3
    return jnp.array([dxdt]) if isinstance(dxdt, (int, float)) else dxdt


def run_pitchfork_example():
    """Run pitchfork bifurcation example with automatic bifurcation detection."""
    print("=" * 70)
    print("Example 1: Modified ODE System")
    print("=" * 70)
    print("\nSystem: dx/dt = r + x - x^3/3")
    print("Continuing equilibria in parameter r")
    print("\nTheoretical branch points at r = ±2/3 ≈ ±0.6667")
    print("(where dx/dr at equilibrium has a singularity)")

    # Define problem
    problem = ContinuationProblem(
        rhs=pitchfork_rhs,
        u0=jnp.array([-2.0]),
        params={"r": -1.0},
        continuation_param="r",
        problem_type="equilibrium",
    )

    # Run continuation with automatic bifurcation detection
    print("\nRunning continuation from r=-1 to r=1...")

    solution = equilibrium_continuation(
        problem,
        param_range=(-1.0, 1.0),
        ds=0.01,  # Smaller step size for better accuracy
        max_steps=300,
        detect_bifurcations=True,  # Enable automatic detection
        compute_stability=True,  # Compute stability along branch
        verbose=True,  # Print bifurcation info
        bifurcation_tolerance=1e-4,
    )

    print(f"{'─'*70}")
    print(f"Continuation completed: {solution.n_points} points computed")
    print(f"{'─'*70}")
    
    plot_continuation(solution)

if __name__ == "__main__":
    solution = run_pitchfork_example()
    plt.show()
