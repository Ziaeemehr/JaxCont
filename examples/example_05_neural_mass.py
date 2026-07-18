"""
Neural-mass equilibrium continuation
=====================================

A 3-D neural-mass model from computational neuroscience, translated from a
BifurcationKit.jl example. We continue the model's *equilibria* (steady
neural activity levels) as the external input :math:`E_0` varies, and expect
JaxCont to find the Hopf bifurcations that bound the region of oscillatory
activity.

.. math::

    \\dot{E} &= \\frac{-E + S(J u x E + E_0)}{\\tau} \\\\
    \\dot{x} &= \\frac{1 - x}{\\tau_D} - u x E \\\\
    \\dot{u} &= \\frac{U_0 - u}{\\tau_F} + U_0 (1 - u) E

where :math:`S(z) = \\alpha \\log(1 + e^{z/\\alpha})` is a soft threshold. The
three state variables are :math:`E` (neural activity), :math:`x`
(synaptic-resource recovery) and :math:`u` (facilitation/adaptation).
"""

# %%
# Setup

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcont import ContinuationProblem, equilibrium_continuation
from jaxcont.solvers.newton import NewtonSolver

os.makedirs("images", exist_ok=True)

# %%
# Define the system

def TMvf(state, params):
    E, x, u = state
    J, alpha, E0 = params["J"], params["α"], params["E0"]
    tau, tauD, tauF, U0 = params["τ"], params["τD"], params["τF"], params["U0"]

    SS0 = J * u * x * E + E0
    SS1 = alpha * jnp.log(1 + jnp.exp(SS0 / alpha))

    dE = (-E + SS1) / tau
    dx = (1.0 - x) / tauD - u * x * E
    du = (U0 - u) / tauF + U0 * (1.0 - u) * E
    return jnp.array([dE, dx, du])


# %%
# Find a starting equilibrium
# -------------------------------
# Continuation needs a point *on* the branch to start from. Rather than
# guessing one, we refine an approximate initial guess with the plain Newton
# solver until the residual is (near) zero.

params = {
    "α": 1.5, "τ": 0.013, "J": 3.07, "E0": -2.0,
    "τD": 0.200, "U0": 0.3, "τF": 1.5, "τS": 0.007,
}
z0_guess = jnp.array([0.238616, 0.982747, 0.367876])

residual_norm = jnp.linalg.norm(TMvf(z0_guess, params))
print(f"Residual at initial guess: {residual_norm:.2e}")

if residual_norm > 1e-6:
    # tol=1e-6: below float32 machine epsilon (~1.2e-7) the residual can
    # oscillate at the precision floor without ever reporting converged=True.
    solver = NewtonSolver(tol=1e-6, max_iter=100)
    z0, converged, n_iter = solver.solve(lambda s: TMvf(s, params), z0_guess)
    print(f"Refined equilibrium in {n_iter} Newton iterations "
          f"(converged={converged}): E={z0[0]:.6f}, x={z0[1]:.6f}, u={z0[2]:.6f}")
else:
    z0 = z0_guess

# %%
# Run the continuation
# -----------------------
# We continue in :math:`E_0` from -4.0 to -0.9. BifurcationKit.jl's reference
# solution reports Hopf bifurcations somewhere in this range.

problem = ContinuationProblem(rhs=TMvf, u0=z0, params=params, continuation_param="E0")

solution = equilibrium_continuation(
    problem,
    param_range=(-4.0, -0.9),
    ds=0.02,
    max_steps=400,
    detect_bifurcations=True,
    compute_stability=True,
    verbose=True,
    bifurcation_tolerance=1e-4,
    newton_tol=1e-6,  # float32-reachable; 1e-8 sits below machine epsilon
)

print(f"Continuation completed: {solution.n_points} points computed")

# %%
# Inspect the detected bifurcations

for i, bif in enumerate(solution.bifurcations, 1):
    state = bif["state"]
    print(f"  #{i}: E0 = {bif['parameter']:.6f}  "
          f"(E={state[0]:.4f}, x={state[1]:.4f}, u={state[2]:.4f})")

# %%
# Plot all three state variables against E0
# ---------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(10, 12))
var_names = ["E", "x", "u"]
var_labels = ["Neural Activity E", "Recovery Variable x", "Adaptation Variable u"]
colors = ["blue", "green", "red"]

for i, (ax, name, label, color) in enumerate(zip(axes, var_names, var_labels, colors)):
    ax.plot(solution.parameters, solution.states[:, i], color=color, linewidth=2,
            label=f"{name} equilibrium", alpha=0.7)
    for bif in solution.bifurcations:
        ax.plot(bif["parameter"], bif["state"][i], "rs", markersize=10,
                markeredgewidth=2, markerfacecolor="red", markeredgecolor="darkred", zorder=10)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best")

axes[-1].set_xlabel("External Input E0")
plt.suptitle("Neural Mass Model - Bifurcation Diagram")
plt.tight_layout()
plt.savefig("images/neural_mass_bifurcation.png", dpi=150, bbox_inches="tight")
plt.show()
