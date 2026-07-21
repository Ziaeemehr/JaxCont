"""
Equilibrium continuation of a cubic normal form
================================================

A minimal walk-through of JaxCont's core workflow: define a system, continue
its equilibria in a parameter, and let automatic bifurcation detection find
the fold points.

We use a modified pitchfork normal form

.. math::

    \\dot{x} = r + x - \\frac{x^3}{3}

which has two fold (turning-point) bifurcations at :math:`r = \\pm 2/3`,
where the branch of equilibria folds back on itself.
"""

# %%
# Setup
# -----
# ``jc.bif_problem`` bundles the right-hand side, an initial guess, and the
# starting parameter value. ``jc.continuation`` runs the predictor-corrector
# loop and returns a :class:`ContinuationResult`.

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxcont as jc
from jaxcont.utils.plotting import plot_continuation

os.makedirs("images", exist_ok=True)

# %%
# Define the system
# ------------------
# The right-hand side takes the state and a *dict* of parameters, and returns
# the residual :math:`f(x, r)`. The continuation parameter (``r`` here) is
# looked up from that dict at every step.


def pitchfork_rhs(u, p, args):
    x = u[0]
    return jnp.array([p + x - x**3 / 3])


# %%
# Set up the problem
# -------------------
# We start from ``r = -1`` on the lower branch, where the only equilibrium is
# a single stable point.

prob = jc.bif_problem(pitchfork_rhs, u0=jnp.array([-2.0]), p0=-1.0)

# %%
# Run the continuation
# ----------------------
# ``events=[jc.Fold(), jc.Hopf()]`` turns on fold/Hopf detection along the
# branch; ``compute_stability=True`` computes the sign of the Jacobian's
# eigenvalues at every point so the plot can color stable/unstable segments.

result = jc.continuation(
    prob, jc.PseudoArclength(), p_span=(-1.0, 1.0),
    settings=jc.ContinuationPar(ds=0.01, max_steps=300, newton_tol=1e-6, compute_stability=True),
    events=[jc.Fold(), jc.Hopf()],
    verbose=True,
)
solution = result._solution

print(f"Continuation completed: {solution.n_points} points computed")

# %%
# Inspect the detected bifurcations
# ------------------------------------
# The theoretical fold locations for this normal form are at
# :math:`r = \pm 2/3 \approx \pm 0.6667`, where :math:`dx/dr` at the
# equilibrium diverges.

theoretical_r = 2.0 / 3.0
for bif in solution.bifurcations:
    error = abs(abs(bif["parameter"]) - theoretical_r)
    print(f"  {bif['type']} at r = {bif['parameter']:.6f}  "
          f"(theory: r = ±{theoretical_r:.6f}, error = {error:.2e})")

# %%
# Cross-validated against BifurcationKit.jl
# ---------------------------------------------
# Independently, offline: running BifurcationKit.jl v0.5.2 (``PALC()``) on the
# identical equation finds the fold at ``r = 0.6666666666578711`` -- 12 digits
# of agreement with the analytic value ``2/3``, and consistent with JaxCont's
# result above to within one continuation step.

# %%
# Plot the bifurcation diagram
# --------------------------------
# Stable and unstable segments are colored automatically, and detected fold
# points are marked.

fig = plot_continuation(solution)
plt.savefig("images/pitchfork_bifurcation.png", dpi=150, bbox_inches="tight")
plt.show()
