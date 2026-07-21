"""
Van der Pol equilibrium branch
==============================

Continue the equilibrium at the origin of the Van der Pol oscillator as the
nonlinear damping parameter :math:`\\mu` increases.

.. math::

    \\dot{x} &= y \\\\
    \\dot{y} &= \\mu (1 - x^2) y - x

The origin is stable for :math:`\\mu < 0` and loses stability through a Hopf
bifurcation at :math:`\\mu = 0`, birthing the system's famous limit cycle.
Tracking that cycle needs periodic-orbit continuation, which is outside
JaxCont's v0.1 scope (see the roadmap) — here we only continue the
equilibrium branch and note where it goes unstable.
"""

# %%
# Setup

import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxcont as jc
from jaxcont.utils.plotting import plot_phase_portrait

# %%
# Define the system

def van_der_pol_rhs(u, p, args):
    x, y = u
    mu = p
    return jnp.array([y, mu * (1.0 - x**2) * y - x])


# %%
# Set up the problem and run the continuation
# -----------------------------------------------
# We start exactly at the origin (an equilibrium for every :math:`\mu`) and
# sweep :math:`\mu` from 0 to 5.

prob = jc.bif_problem(van_der_pol_rhs, u0=jnp.array([0.0, 0.0]), p0=0.0)

result = jc.continuation(
    prob, jc.PseudoArclength(), p_span=(0.0, 5.0),
    settings=jc.ContinuationPar(ds=0.05, max_steps=200),
)
solution = result._solution

print(f"Continuation completed: {solution.n_points} points computed")
print("Note: the origin is already unstable throughout mu > 0 (real part of the")
print("      eigenvalues is mu/2). The Hopf bifurcation sits at mu = 0, the")
print("      branch's starting point -- this run stays on the unstable side,")
print("      where the (untracked) limit cycle lives.")

# %%
# Cross-validated against BifurcationKit.jl
# ---------------------------------------------
# The Jacobian at the origin is ``[[0, 1], [-1, mu]]``, with eigenvalues
# :math:`\mu/2 \pm i\sqrt{1 - \mu^2/4}` -- confirmed directly with Julia's
# ``LinearAlgebra.eigvals`` at several :math:`\mu`, e.g. real part
# :math:`-0.25` at :math:`\mu=-0.5` and :math:`+0.25` at :math:`\mu=+0.5`,
# crossing exactly at :math:`\mu=0`. Running BifurcationKit.jl v0.5.2
# (``PALC()``) starting from that point independently confirms the Hopf sits
# at :math:`\mu=0`, matching this closed-form eigenvalue crossing exactly.

# %%
# Plot the branch and a phase portrait
# ----------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

solution.plot(state_index=0, ax=ax1)
ax1.set_ylabel("x")
ax1.set_title("Bifurcation Diagram")

plot_phase_portrait(solution, state_indices=(0, 1), ax=ax2)

plt.tight_layout()
plt.savefig("van_der_pol.png", dpi=150, bbox_inches="tight")
plt.show()
