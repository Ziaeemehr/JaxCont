r"""
Batched and differentiable periodic-orbit construction
======================================================

Build a batch of Hopf-normal-form limit cycles with ``jax.vmap`` and
differentiate their radii through ``periodic_orbit_problem`` construction.

The periodic lesson uses the 90-line allowance to keep trajectory seeding,
collocation settings, and the plotted observables visible.
"""

# %%
# Set up the output directory and imports. Figures stay open for the
# Sphinx-Gallery matplotlib scraper. JaxCont runs in float64 (import jaxcont enables it).

import argparse
import tempfile
from pathlib import Path
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import jaxcont as jc

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "jaxcont" / "generated"
)
options = parser.parse_args() if __name__ == "__main__" and "__file__" in globals() else parser.parse_args([])
options.output_dir.mkdir(parents=True, exist_ok=True)

# %%
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem
import jaxcont as jc


def rhs(u, p, args):
    x, y = (u[0], u[1])
    r2 = x * x + y * y
    return jnp.array([(p - r2) * x - y, (p - r2) * y + x])


# %%
# Refine the trajectory on a collocation mesh.

mesh = Collocation(ntst=10, ncol=4)
t = jnp.linspace(0.0, 2 * jnp.pi, 120, endpoint=False)


def cycle_radius(p):
    u_guess = jnp.sqrt(p) * jnp.stack([jnp.cos(t), jnp.sin(t)], axis=1)
    problem = periodic_orbit_problem(rhs, u_guess, t, 2 * jnp.pi, p, mesh)
    mesh_states = problem.u0[: 2 * mesh.ntst].reshape(mesh.ntst, 2)
    return jnp.mean(jnp.linalg.norm(mesh_states, axis=1))


parameters = jnp.linspace(0.5, 1.5, 9)
radii = jax.vmap(cycle_radius)(parameters)
exact_radii = jnp.sqrt(parameters)
radius_error = float(jnp.max(jnp.abs(radii - exact_radii)))
radius_derivatives = jax.vmap(jax.grad(cycle_radius))(parameters)
exact_derivatives = 1.0 / (2.0 * jnp.sqrt(parameters))
derivative_error = float(jnp.max(jnp.abs(radius_derivatives - exact_derivatives)))
# %%
# The sensitivity above belongs to construction. An individual constructed
# seed can also start a continuation; this sweep is outside the gradient.

trajectory = jnp.sqrt(parameters[0]) * jnp.stack([jnp.cos(t), jnp.sin(t)], axis=1)
seed = periodic_orbit_problem(rhs, trajectory, t, 2*jnp.pi, parameters[0], mesh)
settings = jc.ContinuationPar(ds=0.02, max_steps=3, compute_stability=False)
result = jc.continuation(seed, p_span=(0.5, 0.55), settings=settings)
p_plot = np.asarray(parameters)
# %%
# Plot the computed observables.

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(p_plot, np.asarray(radii), "o", label="JaxCont vmap")
axes[0].plot(p_plot, np.asarray(exact_radii), "--", label="exact $\\sqrt{p}$")
axes[0].set_xlabel("p")
axes[0].set_ylabel("cycle radius")
axes[0].set_title("Batched construction")
axes[0].legend()
axes[1].plot(p_plot, np.asarray(radius_derivatives), "o", label="JaxCont grad")
axes[1].plot(p_plot, np.asarray(exact_derivatives), "--", label="exact $1/(2\\sqrt{p})$")
axes[1].set_xlabel("p")
axes[1].set_ylabel("d(radius) / dp")
axes[1].set_title("Differentiable construction")
axes[1].legend()
plt.tight_layout()
# %%
# Save this lesson to the explicit output directory.

fig.savefig(options.output_dir / "plot_16_periodic_transforms.png", dpi=150, bbox_inches="tight")
