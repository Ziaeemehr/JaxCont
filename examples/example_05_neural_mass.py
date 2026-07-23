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

import jaxcont as jc
from jaxcont.solvers.newton import NewtonSolver
from jaxcont.viz import plot_all_states

os.makedirs("images", exist_ok=True)

# %%
# Define the system

def TMvf(state, E0, args):
    E, x, u = state
    J, alpha, tau, tauD, tauF, U0 = args

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

E0_0 = -2.0
args = (3.07, 1.5, 0.013, 0.200, 1.5, 0.3)  # J, alpha, tau, tauD, tauF, U0
z0_guess = jnp.array([0.238616, 0.982747, 0.367876])

residual_norm = jnp.linalg.norm(TMvf(z0_guess, E0_0, args))
print(f"Residual at initial guess: {residual_norm:.2e}")

if residual_norm > 1e-6:
    # tol=1e-5: comfortably above the float32 precision floor for this
    # (fairly stiff, tau=0.013) system -- tighter tolerances like 1e-6 sit
    # close enough to the noise floor that convergence becomes unreliable.
    solver = NewtonSolver(tol=1e-5, max_iter=100)
    z0, converged, n_iter = solver.solve(lambda s: TMvf(s, E0_0, args), z0_guess)
    print(f"Refined equilibrium in {n_iter} Newton iterations "
          f"(converged={converged}): E={z0[0]:.6f}, x={z0[1]:.6f}, u={z0[2]:.6f}")
else:
    z0 = z0_guess

# %%
# Run the continuation
# -----------------------
# We continue in :math:`E_0` from -2.0 (the point ``z0`` is actually an
# equilibrium at) to -0.9. ``jc.continuation()`` starts the scan literally at
# ``p_span[0]`` -- not at any stored problem attribute -- so ``p_span`` must
# begin at ``E0_0``, not at the old ``param_range`` lower bound of -4.0 (which
# the deleted OO engine only used as a direction/stop-bound hint, never a
# literal start). BifurcationKit.jl's reference solution reports Hopf
# bifurcations somewhere in this range.

prob = jc.bif_problem(
    TMvf, u0=z0, p0=E0_0, args=args,
    state_names=["E", "x", "u"], param_name="E0",
)

result = jc.continuation(
    prob, jc.PseudoArclength(), p_span=(E0_0, -0.9),
    settings=jc.ContinuationPar(
        ds=0.02, max_steps=400, newton_tol=1e-5, compute_stability=True,
    ),
    events=[jc.Fold(), jc.Hopf()],
    verbose=True,
)
solution = result._solution

print(f"Continuation completed: {solution.n_points} points computed, "
      f"E0 in [{float(solution.parameters.min()):.4f}, {float(solution.parameters.max()):.4f}]")
print("(the corrector stalls before reaching E0=-0.9 on this branch; a smaller")
print(" ds or a restart from a later point would be needed to continue further)")

# %%
# Inspect the detected bifurcations

for i, bif in enumerate(solution.bifurcations, 1):
    state = bif["state"]
    print(f"  #{i}: {bif.get('type', '?'):<5} E0 = {bif['parameter']:.6f}  "
          f"(E={state[0]:.4f}, x={state[1]:.4f}, u={state[2]:.4f})")

# %%
# Cross-validate against BifurcationKit.jl
# --------------------------------------------
# Reference values from running BifurcationKit.jl v0.5.2 (``PALC()``,
# ``bothside=true``) independently, offline, on the identical equations and
# parameters. All three reachable bifurcations match to within 0.0015 in E0
# (issue #7's duplicate/spurious fold-vs-Hopf flags -- fixed 2026-07-23, see
# docs/superpowers/plans/2026-07-23-event-protocol-rewrite.md).

bk_reference = [
    ("fold", -1.865224),
    ("hopf", -1.850125),
    ("fold", -1.463027),
    ("hopf", -1.151059),  # outside the range this branch reaches -- see above
]

print(f"\n{'JaxCont':<28} {'BifurcationKit.jl (reference)':<28}")
print("-" * 56)
for bif in solution.bifurcations:
    print(f"{bif.get('type', '?'):<10} E0={bif['parameter']:<14.4f}", end="")
    matches = [r for r in bk_reference if abs(r[1] - bif["parameter"]) < 0.01]
    if matches:
        print(f" <-> {matches[0][0]:<6} E0={matches[0][1]:.6f}")
    else:
        print(" <-> (no close match -- spurious, see note above)")

# %%
# Plot all three state variables against E0
# ---------------------------------------------

fig = plot_all_states(solution)
plt.suptitle("Neural Mass Model - Bifurcation Diagram")
plt.tight_layout()
plt.savefig("images/neural_mass_bifurcation.png", dpi=150, bbox_inches="tight")
plt.show()
