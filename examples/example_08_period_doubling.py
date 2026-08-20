r"""
Period-doubling of a periodic orbit
===================================

Detect a period-doubling (flip) bifurcation by following the Floquet
multipliers of a periodic orbit.  We use a four-dimensional system with an
exact circular orbit,

.. math::

    \dot{x} &= (1-x^2-y^2)x-y, &
    \dot{y} &= (1-x^2-y^2)y+x, \\
    \dot{w}_1 &= \alpha w_1-\beta w_2, &
    \dot{w}_2 &= \beta w_1+\alpha w_2 .

The orbit :math:`(x,y,w_1,w_2)=(\cos t,\sin t,0,0)` has period
:math:`T=2\pi` for every continuation parameter :math:`\alpha`.  Choosing
:math:`\beta=\pi/T` gives the transverse Floquet multipliers
:math:`-\exp(\alpha T)`.  They therefore cross :math:`-1` exactly at
:math:`\alpha=0`, providing a closed-form check of
:class:`jaxcont.PeriodDoubling`.
"""

# %%
# Setup
# -----
# Periodic-orbit continuation currently lives in its dedicated problem
# module.  The bifurcation event itself is part of the top-level functional
# API.

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxcont as jc
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

T_EXACT = 2.0 * np.pi
BETA = np.pi / T_EXACT


# %%
# Define the system
# -----------------


def period_doubling_rhs(u, alpha, args):
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
# Build the collocation problem
# -----------------------------
# ``periodic_orbit_problem`` resamples a caller-supplied trajectory guess
# onto a fixed Gauss--Legendre collocation mesh and refines it before
# continuation starts.  Here the exact orbit supplies a particularly simple
# initial guess.

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
    period_doubling_rhs,
    jnp.asarray(trajectory_guess),
    jnp.asarray(t_guess),
    T_EXACT,
    alpha_start,
    mesh,
)

# %%
# Continue the orbit and detect the crossing
# ------------------------------------------
# ``compute_stability=True`` stores Floquet multipliers on the branch.
# Period-doubling refinement must recompute them at intermediate points, so
# the event receives the raw ODE and collocation mesh explicitly.

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
    events=[jc.PeriodDoubling(raw_f=period_doubling_rhs, mesh=mesh)],
)

events = [event for event in result.events if event.kind == "period_doubling"]
if len(events) != 1:
    raise RuntimeError(
        f"Expected one period-doubling event, found {len(events)}"
    )

event = events[0]
print(
    f"Period-doubling detected at alpha={event.p:+.8f} "
    f"(analytic alpha=0, error={abs(event.p):.2e})"
)

# %%
# Check the Floquet multipliers
# -----------------------------
# At every branch point we exclude the trivial multiplier nearest ``+1`` and
# select the negative real pair.  The sign identifies the transverse pair
# even after it has moved well past ``-1``; selecting whichever multiplier is
# merely *closest* to ``-1`` would eventually switch to the small positive
# radial multiplier.  The numerical values can then be compared directly
# with :math:`-\exp(\alpha T)`.

alphas = np.asarray(result.branch.params)
all_multipliers = np.asarray(result.branch.eigenvalues)
transverse = []
for multipliers in all_multipliers:
    trivial = np.argmin(np.abs(multipliers - 1.0))
    candidates = np.delete(multipliers, trivial)
    real_candidates = candidates[np.abs(candidates.imag) < 1e-5]
    negative_candidates = real_candidates[real_candidates.real < 0.0]
    transverse.append(negative_candidates[0])

transverse = np.asarray(transverse)
exact = -np.exp(alphas * T_EXACT)
max_error = np.max(np.abs(transverse.real - exact))
print(f"Maximum multiplier error along branch: {max_error:.2e}")

# %%
# Plot the orbit and multiplier crossing
# --------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(trajectory_guess[:, 0], trajectory_guess[:, 1], color="C0")
axes[0].set(
    xlabel=r"$x$",
    ylabel=r"$y$",
    title="Periodic orbit",
    aspect="equal",
)
axes[0].grid(alpha=0.25)

axes[1].plot(alphas, transverse.real, "o-", label="JaxCont")
axes[1].plot(alphas, exact, "--", color="black", label=r"$-\exp(\alpha T)$")
axes[1].axhline(-1.0, color="0.5", linewidth=1)
axes[1].axvline(event.p, color="C3", linestyle=":", label="detected PD")
axes[1].set(
    xlabel=r"$\alpha$",
    ylabel="real Floquet multiplier",
    title="Crossing through $-1$",
)
axes[1].grid(alpha=0.25)
axes[1].legend()

fig.tight_layout()
plt.savefig("images/example_08_period_doubling.png")
plt.show()
