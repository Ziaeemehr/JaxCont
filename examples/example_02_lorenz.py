"""
Example 2: Lorenz84 system - equilibrium continuation
Extended Lorenz model with 4 variables and rich bifurcation structure

This example demonstrates:
- Multi-dimensional system continuation
- Automatic bifurcation detection
- Both-sided continuation (forward and backward)
- BifurcationKit-style output and plotting
"""

import os
import jax.numpy as jnp
from jaxcont import ContinuationProblem, equilibrium_continuation
import matplotlib.pyplot as plt
from jax import jit

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


@jit
def lorenz84_rhs(state, params):
    """
    Lorenz84 system (extended Lorenz model).

    This is a low-order model of atmospheric circulation with 4 variables.

    Equations:
        dX/dt = -Y^2 - Z^2 - α*X + α*F - γ*U^2
        dY/dt = X*Y - β*X*Z - Y + G
        dZ/dt = β*X*Y + X*Z - Z
        dU/dt = -δ*U + γ*U*X + T

    Parameters:
        α (alpha): Dissipation coefficient
        β (beta): Rotation parameter
        γ (gamma): Coupling coefficient
        δ (delta): Damping coefficient
        G: External forcing (latitude)
        F: External forcing (longitude) - continuation parameter
        T: Temperature forcing

    Reference: Lorenz, E. N. (1984). Irregularity: a fundamental property of the atmosphere.
    """
    X, Y, Z, U = state

    # Extract parameters
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    delta = params["delta"]
    G = params["G"]
    F = params["F"]
    T = params["T"]

    # System equations
    dX = -(Y**2) - Z**2 - alpha * X + alpha * F - gamma * U**2
    dY = X * Y - beta * X * Z - Y + G
    dZ = beta * X * Y + X * Z - Z
    dU = -delta * U + gamma * U * X + T

    return jnp.array([dX, dY, dZ, dU])


def run_lorenz84_example():
    """
    Run Lorenz84 system equilibrium continuation.
    System: Extended Lorenz model with 4 variables (X, Y, Z, U)
    Continuing equilibria in parameter F (external forcing)
    """
    # Parameters
    # parlor = (α = 1/4, β = 1., G = .25, δ = 1.04, γ = 0.987, F = 1.7620532879639, T = .0001265)
    params = {
        "alpha": 0.25,          # α = 1/4
        "beta": 1.0,            # β = 1.0
        "gamma": 0.987,         # γ = 0.987
        "delta": 1.04,          # δ = 1.04
        "G": 0.25,              # External forcing (latitude)
        "F": 1.7620532879639,   # External forcing (longitude) - continuation parameter
        # "T": 0.0001265,  # Temperature forcing
        "T": 0.04, 
    }

    # Initial state 
    u0 = jnp.array(
        [2.9787004394953343, 
         -0.03868302503393752, 
         0.058232737694740085, 
         -0.02105288273117459]
    )

    # Define problem
    problem = ContinuationProblem(
        rhs=lorenz84_rhs, u0=u0, params=params, continuation_param="F", problem_type="equilibrium"
    )

    # Run continuation with automatic bifurcation detection
    # p_min = -1.5, p_max = 3.0, ds = 0.002, dsmax = 0.05, bothside = true
    print("\nRunning continuation from F=-1.5 to F=3.0 (both directions)...")
    print("(Bifurcation detection enabled automatically)\n")

    solution = equilibrium_continuation(
        problem,
        param_range=(-1.5, 3.0),
        ds=0.01,  # Initial step size
        ds_max=0.05,  # Maximum step size
        max_steps=200,
        detect_bifurcations=True,  # Enable automatic detection
        compute_stability=True,  # Compute stability along branch
        verbose=True,  # Print bifurcation info
        # bifurcation_tolerance=1e-4,
        # newton_tol=1e-12,  # High precision
    )

    print(f"\n{'─'*70}")
    print(f"Continuation completed: {solution.n_points} points computed")
    print(f"{'─'*70}")

    # Print bifurcation summary
    if solution.bifurcations:
        print(f"\nDetected {len(solution.bifurcations)} bifurcation(s):")
        print(f"{'─'*70}")
        for i, bif in enumerate(solution.bifurcations, 1):
            param = bif["parameter"]
            bif_type = bif.get("type", "unknown")
            print(f"  #{i}: {bif_type.capitalize()} at F = {param:.6f}")
        print()

    # Plot bifurcation diagram
    print(f"{'─'*70}")
    print("Plotting bifurcation diagram...")
    fig = plot_lorenz84_diagram(solution)
    plt.savefig(f"{path}/lorenz84_bifurcation.png", dpi=150, bbox_inches="tight")
    print(f"Saved to: {path}/lorenz84_bifurcation.png")
    print(f"{'═'*70}\n")

    return solution


def plot_lorenz84_diagram(solution):
    """Plot bifurcation diagram for Lorenz84 system - X variable only (BifurcationKit style)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data for X variable (index 0)
    params = solution.parameters
    X_states = solution.states[:, 0]

    # Plot the X branch
    ax.plot(params, X_states, 'b-', linewidth=2, alpha=0.7, label='X(F)')
    ax.plot(params, X_states, 'b.', markersize=4, alpha=0.5)

    # Mark bifurcations
    if solution.bifurcations:
        for bif in solution.bifurcations:
            param = bif["parameter"]
            state_X = bif["state"][0]  # X component
            bif_type = bif.get("type", "unknown")

            # Choose marker based on type
            if bif_type == "fold":
                marker, mcolor, label = "s", "red", "Fold"
            elif bif_type == "hopf":
                marker, mcolor, label = "^", "magenta", "Hopf"
            else:
                marker, mcolor, label = "o", "orange", bif_type

            ax.plot(
                param,
                state_X,
                marker,
                color=mcolor,
                markersize=12,
                markeredgewidth=2,
                markerfacecolor=mcolor,
                markeredgecolor="darkred",
                label=label,
                zorder=10,
            )

            # Add annotation
            ax.annotate(
                f"{label}\nF={param:.3f}\nX={state_X:.3f}",
                xy=(param, state_X),
                xytext=(15, 15),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="red", lw=1.5),
                fontsize=9,
                ha="left",
            )

    # Styling
    ax.set_xlabel("Parameter F (External Forcing)", fontsize=14, fontweight="bold")
    ax.set_ylabel("X", fontsize=14, fontweight="bold")
    ax.set_title("Lorenz84 System: Bifurcation Diagram (X variable)", fontsize=15, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
    
    # Legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best", framealpha=0.9, fontsize=11)
    
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    solution = run_lorenz84_example()
    plt.show()
