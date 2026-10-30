r"""
FitzHugh-Nagumo phase plane
===========================

Continue the equilibrium of the FitzHugh-Nagumo neuron model through its Hopf
bifurcation, then read the same transition off the phase plane.

.. math::

    \dot{v} &= v - v^3/3 - w + I \\
    \dot{w} &= 0.08 (v + 0.7 - 0.8 w)

The Hopf occurs before the cubic nullcline turns: the Jacobian trace
vanishes when 1-v squared equals 0.08 times 0.8. The two phase planes show
outward and inward spirals on opposite sides of that crossing.

"""

# %%
# Set up the output directory and imports. Figures stay open for the
# Sphinx-Gallery matplotlib scraper. JaxCont runs in float64 (import jaxcont enables it).

import argparse
import tempfile
from pathlib import Path
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxcont as jc

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "jaxcont" / "generated"
)
options = parser.parse_args() if __name__ == "__main__" and "__file__" in globals() else parser.parse_args([])
options.output_dir.mkdir(parents=True, exist_ok=True)

# %%
from jaxcont.viz import plot_phase_plane


def fitzhugh_nagumo_rhs(u, p, args):
    v, w = u
    current = p
    a, b, tau = args
    return jnp.array([v - v**3 / 3.0 - w + current, tau * (v + a - b * w)])


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
# Continue from the visible seed with the settings below.

result = jc.continuation(
    problem,
    p_span=(0.0, 1.0),
    settings=jc.ContinuationPar(ds=0.01, max_steps=400),
    events=[jc.Hopf()],
)
for event in result.events:
    print(f"{event.kind} at I = {float(event.p):.4f}")
# %%
# Compare the branch with trajectories above the Hopf crossing.

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
# %%
# Below Hopf the same system has a stable equilibrium.

fig.savefig(options.output_dir / "phase_plane_after.png")
fig = plot_phase_plane(
    problem,
    p=0.0,
    xlim=(-2.5, 2.5),
    ylim=(-1.0, 2.0),
    result=result,
    trajectories=[(jnp.array([1.0, 0.0]), (0.0, 200.0))],
)
plt.tight_layout()
# %%
# Save this lesson to the explicit output directory.

fig.savefig(options.output_dir / "plot_11_phase_plane.png", dpi=150, bbox_inches="tight")
