r"""
FitzHugh-Nagumo phase plane
===========================

Continue the equilibrium of the FitzHugh-Nagumo neuron model through its Hopf
bifurcation, then read the same transition off the phase plane.

.. math::

    \dot{v} &= v - v^3/3 - w + I \\
    \dot{w} &= 0.08 (v + 0.7 - 0.8 w)

The cubic :math:`v`-nullcline and the linear :math:`w`-nullcline intersect at
the single equilibrium. As the input current :math:`I` grows the intersection
slides up the cubic's middle branch; where it crosses the local maximum the
equilibrium loses stability at a Hopf bifurcation and a limit cycle appears.
The bifurcation diagram reports the crossing; the phase plane shows the
geometry that produces it.
"""

# %%
# Setup

import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxcont as jc
from jaxcont.viz import plot_phase_plane

# %%
# Define the system


def fitzhugh_nagumo_rhs(u, p, args):
    v, w = u
    current = p
    a, b, tau = args
    return jnp.array([
        v - v**3 / 3.0 - w + current,
        tau * (v + a - b * w),
    ])


args = (0.7, 0.8, 0.08)

problem = jc.bif_problem(
    fitzhugh_nagumo_rhs,
    u0=jnp.array([-1.2, -0.6]),
    p0=0.0,
    args=args,
    state_names=["v", "w"],
    param_name="I",
)

# %%
# Continue the equilibrium and detect the Hopf bifurcation

result = jc.continuation(
    problem,
    p_span=(0.0, 1.0),
    settings=jc.ContinuationPar(ds=0.01, max_steps=400),
    events=[jc.Hopf()],
)

for event in result.events:
    print(f"{event.kind} at I = {float(event.p):.4f}")

# %%
# Bifurcation diagram beside the phase plane
# ------------------------------------------
#
# The right panel freezes the system at I = 0.5, past the Hopf point: the
# equilibrium is unstable (hollow marker) and the trajectory spirals outward
# onto the limit cycle.

fig, (ax_diagram, ax_plane) = plt.subplots(1, 2, figsize=(13.0, 5.5))

result.plot(state_index=0, ax=ax_diagram, annotate=True)

plot_phase_plane(
    problem,
    p=0.5,
    xlim=(-2.5, 2.5),
    ylim=(-1.0, 2.0),
    result=result,
    vector_field=True,
    trajectories=[(jnp.array([-1.0, -0.5]), (0.0, 200.0))],
    ax=ax_plane,
)

plt.tight_layout()
plt.show()

# %%
# Below the Hopf point the same picture shows a stable equilibrium
# ----------------------------------------------------------------
#
# At I = 0.0 the intersection sits on the cubic's left branch, the marker is
# filled, and the trajectory spirals inward.

fig = plot_phase_plane(
    problem,
    p=0.0,
    xlim=(-2.5, 2.5),
    ylim=(-1.0, 2.0),
    result=result,
    trajectories=[(jnp.array([1.0, 0.0]), (0.0, 200.0))],
)

plt.tight_layout()
plt.show()
