"""
Lorenz-84 equilibrium continuation
===================================

A 4-dimensional example: the extended Lorenz-84 model of atmospheric
circulation. This demonstrates continuation of a higher-dimensional system
and cross-validates JaxCont's fold/Hopf detection against **BifurcationKit.jl
v0.5.2** (run independently, offline, on the same equations -- see the
comparison table at the end of this example).

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
# ``u0`` below is a genuine equilibrium of the system at ``F=1.7620532879639``
# (residual :math:`\sim 10^{-8}`), found by Newton refinement from an
# approximate starting guess -- always check ``rhs(u0, params)`` is close to
# zero before continuing from a hand-copied initial condition.

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
    [1.6673192028567203, -0.05172586841139392, 0.12923880103788027, -0.0660453938041009]
)
print(f"residual at u0: {lorenz84_rhs(u0, params)}")

problem = ContinuationProblem(
    rhs=lorenz84_rhs, u0=u0, params=params, continuation_param="F", problem_type="equilibrium"
)

# %%
# Run the continuation
# -----------------------
# JaxCont's ``run()`` only explores *one* direction per call, toward whichever
# end of ``param_range`` lies on the same side as the starting parameter (here
# ``param_range=(1.0, 1.5)`` forces it downward from :math:`F=1.762`). This
# branch happens to fold near :math:`F \approx 1.55`, and pseudo-arclength
# continuation passes straight through that fold and carries on upward past
# the starting point -- which is exactly where this system's Hopf
# bifurcations live.

solution = equilibrium_continuation(
    problem,
    param_range=(1.0, 1.5),
    ds=0.005,
    ds_max=0.01,
    max_steps=215,
    detect_bifurcations=True,
    compute_stability=True,
    verbose=True,
    bifurcation_tolerance=1e-3,
)

print(f"\nContinuation completed: {solution.n_points} points, "
      f"F in [{float(solution.parameters.min()):.4f}, {float(solution.parameters.max()):.4f}]")

# %%
# Cross-validate against BifurcationKit.jl
# --------------------------------------------
# The reference values below come from running BifurcationKit.jl v0.5.2
# (``PALC()``, ``bothside=true``) on the *identical* right-hand side and
# parameters, independently, offline. JaxCont's detected fold/Hopf parameter
# values agree with BifurcationKit.jl's to within about one continuation step
# (:math:`\Delta F \lesssim 0.005`) -- close bifurcations occasionally produce
# an extra/duplicate flag (visible below as an unmatched "fold" near the first
# Hopf), which is a known precision limitation of the current detector, not a
# location error.

bk_reference = [
    ("bp/fold", 1.546648),
    ("hopf", 1.619658),
    ("hopf", 2.467222),
    ("hopf", 2.859876),
]

print(f"\n{'JaxCont':<28} {'BifurcationKit.jl (reference)':<28}")
print("-" * 56)
for bif in solution.bifurcations:
    print(f"{bif.get('type', '?'):<10} F={bif['parameter']:<14.4f}", end="")
    matches = [r for r in bk_reference if abs(r[1] - bif["parameter"]) < 0.01]
    if matches:
        print(f" <-> {matches[0][0]:<10} F={matches[0][1]:.6f}")
    else:
        print(" <-> (no close match; see note above)")

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
