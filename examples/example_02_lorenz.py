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

import jaxcont as jc
from jaxcont.viz import plot_continuation

os.makedirs("images", exist_ok=True)

# %%
# Define the system
# ------------------
# :math:`F` (external forcing / longitude) is the continuation parameter;
# the rest are fixed physical constants.


def lorenz84_rhs(state, F, args):
    X, Y, Z, U = state
    alpha, beta, gamma, delta, G, T = args

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
# approximate starting guess -- always check ``rhs(u0, F0, args)`` is close to
# zero before continuing from a hand-copied initial condition.

F0 = 1.7620532879639
args = (0.25, 1.0, 0.987, 1.04, 0.25, 0.04)  # alpha, beta, gamma, delta, G, T

u0 = jnp.array(
    [1.6673192028567203, -0.05172586841139392, 0.12923880103788027, -0.0660453938041009]
)
print(f"residual at u0: {lorenz84_rhs(u0, F0, args)}")

prob = jc.bif_problem(
    lorenz84_rhs, u0=u0, p0=F0, args=args,
    state_names=["X", "Y", "Z", "U"], param_name="F",
)

# %%
# Run the continuation
# -----------------------
# Unlike the fixed-``params`` dict of the pre-migration API, ``jc.continuation``
# actually starts the scan at ``p_span[0]`` -- so, since ``u0`` is only a
# genuine equilibrium at :math:`F=1.7620532879639`, ``p_span`` must start
# there (not at the old ``param_range`` lower bound). ``p_span=(F0, 1.0)``
# reproduces the original run: pseudo-arclength continuation heads downward
# from :math:`F=1.762` first, folds near :math:`F \approx 1.55`, passes
# straight through, and carries on upward past the starting point -- which is
# exactly where this system's Hopf bifurcations live. ``1.0`` is never
# actually reached (the branch turns upward at the fold), so it only sets the
# initial direction, matching the pre-migration behavior.

result = jc.continuation(
    prob, jc.PseudoArclength(), p_span=(F0, 1.0),
    settings=jc.ContinuationPar(
        ds=0.005, ds_max=0.01, max_steps=215, newton_tol=1e-6, compute_stability=True,
    ),
    events=[jc.Fold(), jc.Hopf()],
    verbose=True,
)
solution = result._solution

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

fig = plot_continuation(solution, annotate=True)
fig.axes[0].set_title("Lorenz-84 System: Bifurcation Diagram (X variable)", fontweight="bold")
plt.savefig("images/lorenz84_bifurcation.png", dpi=150, bbox_inches="tight")
plt.show()
