r"""
Neimark--Sacker bifurcation of a periodic orbit
================================================

Detect a Neimark--Sacker (torus) bifurcation by following a complex-conjugate
pair of Floquet multipliers through the unit circle.  We use the same
four-dimensional circle-plus-transverse system as the period-doubling
example,

.. math::

    \dot{x} &= (1-x^2-y^2)x-y, &
    \dot{y} &= (1-x^2-y^2)y+x, \\
    \dot{w}_1 &= \alpha w_1-\beta w_2, &
    \dot{w}_2 &= \beta w_1+\alpha w_2 .

The exact periodic orbit
:math:`(x,y,w_1,w_2)=(\cos t,\sin t,0,0)` has period :math:`T=2\pi`.
With :math:`\beta=0.3`, its transverse multipliers are
:math:`\exp((\alpha\pm i\beta)T)`: a genuine complex pair whose magnitude
:math:`\exp(\alpha T)` crosses one exactly at :math:`\alpha=0`.
"""

# %%
# Setup

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxcont as jc
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

T_EXACT = 2.0 * np.pi
BETA = 0.3


# %%
# Define the system


def neimark_sacker_rhs(u, alpha, args):
    x, y, w1, w2 = u
    radius_squared = x**2 + y**2
    return jnp.array(
        [
            (1.0 - radius_squared) * x - y,
            (1.0 - radius_squared) * y + x,
            alpha * w1 - BETA * w2,
            BETA * w1 + alpha * w2,
        ]
    )


# %%
# Build the periodic-orbit problem

alpha_start = -0.05
t_guess = np.linspace(0.0, T_EXACT, 60, endpoint=False)
trajectory_guess = np.column_stack(
    [
        np.cos(t_guess),
        np.sin(t_guess),
        np.zeros_like(t_guess),
        np.zeros_like(t_guess),
    ]
)

mesh = Collocation(ntst=10, ncol=4)
problem = periodic_orbit_problem(
    neimark_sacker_rhs,
    jnp.asarray(trajectory_guess),
    jnp.asarray(t_guess),
    T_EXACT,
    alpha_start,
    mesh,
)

# %%
# Continue the orbit and detect the crossing
# ------------------------------------------
# Floquet multipliers are requested with ``compute_stability=True``.
# ``NeimarkSacker`` also needs the raw ODE and mesh so its bisection
# refinement can recompute multipliers between continuation points.

result = jc.continuation(
    problem,
    jc.PseudoArclength(),
    p_span=(alpha_start, 0.05),
    settings=jc.ContinuationPar(
        ds=0.02,
        max_steps=50,
        newton_tol=1e-5,
        compute_stability=True,
    ),
    events=[jc.NeimarkSacker(raw_f=neimark_sacker_rhs, mesh=mesh)],
)

events = [event for event in result.events if event.kind == "neimark_sacker"]
if len(events) != 1:
    raise RuntimeError(
        f"Expected one Neimark--Sacker event, found {len(events)}"
    )

event = events[0]
print(
    f"Neimark--Sacker detected at alpha={event.p:+.8f} "
    f"(analytic alpha=0, error={abs(event.p):.2e})"
)

# %%
# Check the complex Floquet pair
# ------------------------------
# Select the multiplier with positive imaginary part.  Its exact magnitude is
# :math:`\exp(\alpha T)` and its argument stays at :math:`\beta T`.

alphas = np.asarray(result.branch.params)
all_multipliers = np.asarray(result.branch.eigenvalues)
transverse = []
for multipliers in all_multipliers:
    complex_candidates = multipliers[multipliers.imag > 1e-5]
    nearest_unit = np.argmin(np.abs(np.abs(complex_candidates) - 1.0))
    transverse.append(complex_candidates[nearest_unit])

transverse = np.asarray(transverse)
exact_magnitude = np.exp(alphas * T_EXACT)
max_error = np.max(np.abs(np.abs(transverse) - exact_magnitude))
print(f"Maximum multiplier-magnitude error along branch: {max_error:.2e}")

# %%
# Plot the unit-circle crossing
# -----------------------------
# The left panel shows the complex pair crossing the unit circle.  The right
# panel displays the same event as the scalar test condition
# :math:`|\mu|-1=0`.

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

theta = np.linspace(0.0, 2.0 * np.pi, 300)
axes[0].plot(
    np.cos(theta),
    np.sin(theta),
    "--",
    color="0.5",
    label="unit circle",
)
axes[0].plot(transverse.real, transverse.imag, "o-", color="C0")
axes[0].plot(transverse.real, -transverse.imag, "o-", color="C0")
axes[0].set(
    xlabel=r"$\mathrm{Re}(\mu)$",
    ylabel=r"$\mathrm{Im}(\mu)$",
    title="Complex Floquet pair",
    aspect="equal",
)
axes[0].grid(alpha=0.25)
axes[0].legend()

axes[1].plot(alphas, np.abs(transverse), "o-", label="JaxCont")
axes[1].plot(
    alphas,
    exact_magnitude,
    "--",
    color="black",
    label=r"$\exp(\alpha T)$",
)
axes[1].axhline(1.0, color="0.5", linewidth=1)
axes[1].axvline(event.p, color="C3", linestyle=":", label="detected NS")
axes[1].set(
    xlabel=r"$\alpha$",
    ylabel=r"$|\mu|$",
    title="Crossing through the unit circle",
)
axes[1].grid(alpha=0.25)
axes[1].legend()

fig.tight_layout()
plt.savefig("images/example_09_neimark_sacker.png")
plt.show()
