"""
Stability and bifurcation-label plotting
==========================================

A closer look at ``plot_continuation``'s stability coloring and bifurcation
markers, using the same pitchfork normal form as
:doc:`example_01_pitchfork`. Stable segments are drawn solid, unstable
segments dashed, and fold points are labeled without duplicate legend
entries.
"""

# %%
# Setup

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcont import ContinuationProblem, equilibrium_continuation
from jaxcont.utils import plot_continuation

os.makedirs("images", exist_ok=True)


def pitchfork_rhs(state, params):
    return jnp.array([params["r"] + state[0] - state[0] ** 3 / 3])


# %%
# Run the continuation

problem = ContinuationProblem(
    rhs=pitchfork_rhs, u0=jnp.array([-2.0]), params={"r": -1.0},
    continuation_param="r", problem_type="equilibrium",
)
solution = equilibrium_continuation(
    problem, param_range=(-1.0, 1.0), ds=0.01, max_steps=300,
    detect_bifurcations=True, compute_stability=True, bifurcation_tolerance=1e-4,
)
print(f"Continuation completed: {solution.n_points} points, "
      f"{len(solution.bifurcations)} bifurcation(s) detected")

# %%
# Inspect stability transitions
# ----------------------------------
# The state is stable/unstable at each computed point; we can locate exactly
# where along the branch the sign flips.

n_stable = int(jnp.sum(solution.stability))
print(f"Stable points: {n_stable} / {solution.n_points}")

stability_int = jnp.array(solution.stability, dtype=int)
changes = jnp.where(jnp.diff(stability_int) != 0)[0]
for idx in changes:
    print(f"  transition at index {idx}->{idx+1}: "
          f"r={solution.parameters[idx]:.4f} -> r={solution.parameters[idx+1]:.4f}")

# %%
# Plot with stability coloring and bifurcation labels
# ---------------------------------------------------------

fig = plot_continuation(solution, state_index=0, show_bifurcations=True)
plt.savefig("images/pitchfork_stability.png", dpi=150, bbox_inches="tight")
plt.show()
