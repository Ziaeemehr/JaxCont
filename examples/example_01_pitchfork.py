"""
Example 1: Equilibrium continuation of a simple ODE
Modified system: dx/dt = r + x - x^3/3

This example demonstrates:
- Automatic bifurcation detection during continuation
- Branch points (fold bifurcations) at r ≈ ±2/3
- Plotting the bifurcation diagram and marking detected bifurcations
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
    print("(Bifurcation detection enabled automatically)\n")

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

    # The bifurcations are now automatically detected and stored in solution.bifurcations
    # Print comparison with theory
    if solution.bifurcations:
        print("\nComparison with Theoretical Predictions:")
        print(f"{'─'*70}")
        for i, bif in enumerate(solution.bifurcations, 1):
            param = bif["parameter"]
            state = bif["state"][0]
            theory_r = jnp.sign(param) * 2 / 3
            theory_x = jnp.sign(param) * 1.0  # x = ±1 at branch points

            print(f"Branch Point #{i}:")
            print(f"  Computed:    r = {param:8.6f},  x = {state:8.6f}")
            print(f"  Theory:      r = {theory_r:8.6f},  x = {theory_x:8.6f}")
            print(
                f"  Error:       Δr = {abs(param - theory_r):8.6f},  Δx = {abs(state - theory_x):8.6f}"
            )
            print()

    # Plot results with enhanced style
    print(f"{'─'*70}")
    print("Plotting bifurcation diagram...")
    fig = plot_diagram(solution, solution.bifurcations)
    plt.savefig(f"{path}/pitchfork_bifurcation.png", dpi=150, bbox_inches="tight")
    print(f"Saved to: {path}/pitchfork_bifurcation.png")
    print(f"{'═'*70}\n")

    return solution


def plot_diagram(solution, bifurcations):
    """Plot bifurcation diagram in BifurcationKit style."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract data
    params = solution.parameters
    states = solution.states if solution.state_dim == 1 else solution.states[:, 0]

    # Plot the branch
    ax.plot(params, states, "b-", linewidth=2, label="Equilibrium branch", alpha=0.5)
    ax.plot(params, states, "b.", markersize=4, alpha=0.5)

    # Mark bifurcation points
    if bifurcations:
        for bif in bifurcations:
            param = bif["parameter"]
            state = bif["state"][0]
            ax.plot(
                param,
                state,
                "rs",
                markersize=12,
                markeredgewidth=2,
                markerfacecolor="red",
                markeredgecolor="darkred",
                label="Branch Point",
                zorder=10,
            )

            # Add annotation
            ax.annotate(
                f"BP\nr={param:.4f}\nx={state:.4f}",
                xy=(param, state),
                xytext=(15, 15),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="red", lw=1.5),
                fontsize=9,
                ha="left",
            )

    # Styling to match BifurcationKit
    ax.set_xlabel("Parameter r")
    ax.set_ylabel("State x")
    ax.set_title(
        "Bifurcation Diagram: dx/dt = r + x - x³/3",
    )
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)

    # Add theoretical branch points as vertical lines
    ax.axvline(
        x=2 / 3, color="gray", linestyle=":", linewidth=1.5, alpha=0.5, label="Theory: r=±2/3"
    )
    ax.axvline(x=-2 / 3, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)

    # Legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best", framealpha=0.9)

    ax.tick_params(labelsize=11)
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    solution = run_pitchfork_example()
    plt.show()
