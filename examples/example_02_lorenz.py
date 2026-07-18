"""
Lorenz-84 equilibrium continuation
===================================

A 4-dimensional example: the extended Lorenz-84 model of atmospheric
circulation. This demonstrates continuation of a higher-dimensional system
with a richer bifurcation structure than the 1-D examples, and reproduces a
case from BifurcationKit.jl.

.. math::

    \\dot{X} &= -Y^2 - Z^2 - \\alpha X + \\alpha F - \\gamma U^2 \\\\
    \\dot{Y} &= XY - \\beta XZ - Y + G \\\\
    \\dot{Z} &= \\beta XY + XZ - Z \\\\
    \\dot{U} &= -\\delta U + \\gamma UX + T

Reference: Lorenz, E. N. (1984). *Irregularity: a fundamental property of
the atmosphere.*
"""

# %%
# Setup

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit

from jaxcont import ContinuationProblem, equilibrium_continuation

os.makedirs("images", exist_ok=True)

# %%
# Define the system
# ------------------
# :math:`F` (external forcing / longitude) is the continuation parameter;
# the rest are fixed physical constants.


@jit
def lorenz84_rhs(state, params):
    X, Y, Z, U = state

    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    delta = params["delta"]
    G = params["G"]
    F = params["F"]
    T = params["T"]

    dX = -(Y**2) - Z**2 - alpha * X + alpha * F - gamma * U**2
    dY = X * Y - beta * X * Z - Y + G
    dZ = beta * X * Y + X * Z - Z
    dU = -delta * U + gamma * U * X + T

    return jnp.array([dX, dY, dZ, dU])


# %%
# Set up the problem
# --------------------
# Parameters and the starting equilibrium below match the BifurcationKit.jl
# reference case so the two can be compared directly.

params = {
    "alpha": 0.25,
    "beta": 1.0,
    "gamma": 0.987,
    "delta": 1.04,
    "G": 0.25,
    "F": 1.7620532879639,
    "T": 0.04,
}

u0 = jnp.array(
    [2.9787004394953343, -0.03868302503393752, 0.058232737694740085, -0.02105288273117459]
)

problem = ContinuationProblem(
    rhs=lorenz84_rhs, u0=u0, params=params, continuation_param="F", problem_type="equilibrium"
)

# %%
# Run the continuation
# -----------------------
# Bifurcation detection and stability are both enabled; the branch is swept
# from :math:`F=-1.5` to :math:`F=3.0`.

solution = equilibrium_continuation(
    problem,
    param_range=(-1.5, 3.0),
    ds=0.01,
    ds_max=0.05,
    max_steps=200,
    detect_bifurcations=True,
    compute_stability=True,
    verbose=True,
)

print(f"Continuation completed: {solution.n_points} points computed")

# %%
# Inspect the detected bifurcations

for i, bif in enumerate(solution.bifurcations, 1):
    print(f"  #{i}: {bif.get('type', 'unknown').capitalize()} at F = {bif['parameter']:.6f}")

# %%
# Plot the bifurcation diagram (X variable)
# --------------------------------------------
# With a 4-D state we pick one variable (X) to plot against the parameter;
# detected bifurcations are annotated on the branch.


def plot_lorenz84_diagram(solution):
    fig, ax = plt.subplots(figsize=(10, 6))

    params_arr = solution.parameters
    X_states = solution.states[:, 0]

    ax.plot(params_arr, X_states, "b-", linewidth=2, alpha=0.7, label="X(F)")
    ax.plot(params_arr, X_states, "b.", markersize=4, alpha=0.5)

    for bif in solution.bifurcations:
        param = bif["parameter"]
        state_X = bif["state"][0]
        bif_type = bif.get("type", "unknown")
        marker, mcolor, label = {
            "fold": ("s", "red", "Fold"),
            "hopf": ("^", "magenta", "Hopf"),
        }.get(bif_type, ("o", "orange", bif_type))

        ax.plot(
            param, state_X, marker, color=mcolor, markersize=12, markeredgewidth=2,
            markerfacecolor=mcolor, markeredgecolor="darkred", label=label, zorder=10,
        )
        ax.annotate(
            f"{label}\nF={param:.3f}\nX={state_X:.3f}",
            xy=(param, state_X), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5), fontsize=9,
        )

    ax.set_xlabel("Parameter F (External Forcing)", fontweight="bold")
    ax.set_ylabel("X", fontweight="bold")
    ax.set_title("Lorenz-84 System: Bifurcation Diagram (X variable)", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")
    plt.tight_layout()
    return fig


fig = plot_lorenz84_diagram(solution)
plt.savefig("images/lorenz84_bifurcation.png", dpi=150, bbox_inches="tight")
plt.show()
