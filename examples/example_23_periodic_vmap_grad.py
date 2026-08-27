r"""
Batched and differentiable periodic-orbit construction
=========================================================

Build a batch of Hopf-normal-form limit cycles with ``jax.vmap`` and
differentiate their radii through ``periodic_orbit_problem`` construction.
"""

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem


# %%
# Define the system
# --------------------
#
# :math:`x'=(p-r^2)x-y,\ y'=(p-r^2)y+x` (the Hopf normal form) has an exact
# limit cycle of radius :math:`\sqrt{p}` for every :math:`p>0`, with period
# :math:`2\pi` independent of :math:`p` -- a closed form to check both
# capabilities below against.


def rhs(u, p, args):
    x, y = u[0], u[1]
    r2 = x * x + y * y
    return jnp.array([(p - r2) * x - y, (p - r2) * y + x])


mesh = Collocation(ntst=10, ncol=4)
t = jnp.linspace(0.0, 2 * jnp.pi, 120, endpoint=False)


def cycle_radius(p):
    u_guess = jnp.sqrt(p) * jnp.stack([jnp.cos(t), jnp.sin(t)], axis=1)
    problem = periodic_orbit_problem(rhs, u_guess, t, 2 * jnp.pi, p, mesh)
    mesh_states = problem.u0[: 2 * mesh.ntst].reshape(mesh.ntst, 2)
    return jnp.mean(jnp.linalg.norm(mesh_states, axis=1))


# %%
# Batch many parameter values with vmap
# -----------------------------------------
#
# ``periodic_orbit_problem`` builds a whole collocation problem -- resampling
# a trajectory guess, then Newton-refining it -- inside one function call.
# ``jax.vmap`` turns nine independent calls of that into a single compiled
# kernel rather than a Python loop over nine separate continuations.

parameters = jnp.linspace(0.5, 1.5, 9)
radii = jax.vmap(cycle_radius)(parameters)

exact_radii = jnp.sqrt(parameters)
radius_error = float(jnp.max(jnp.abs(radii - exact_radii)))
print(f"vmap radius max error: {radius_error:.6e}")

# %%
# Differentiate through construction with grad
# -------------------------------------------------
#
# Until this capability shipped, ``jax.grad`` through this same
# construction raised ``UnexpectedTracerError``: the phase-condition
# reference is built from ``p`` and was captured by a closure that crossed
# the Newton corrector's ``lax.while_loop`` trace boundary. Construction now
# threads every ``p``-dependent value through the implicit solve explicitly
# instead, so ``d(radius)/dp`` is available via the same implicit-function-
# theorem path the rest of the library already uses for fold/Hopf locations.

radius_derivatives = jax.vmap(jax.grad(cycle_radius))(parameters)

exact_derivatives = 1.0 / (2.0 * jnp.sqrt(parameters))
derivative_error = float(
    jnp.max(jnp.abs(radius_derivatives - exact_derivatives))
)
print(f"grad radius-derivative max error: {derivative_error:.6e}")

# %%
# Plot both against the closed form
# -------------------------------------

p_plot = np.asarray(parameters)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(p_plot, np.asarray(radii), "o", label="JaxCont vmap")
axes[0].plot(p_plot, np.asarray(exact_radii), "--", label=r"exact $\sqrt{p}$")
axes[0].set_xlabel("p")
axes[0].set_ylabel("cycle radius")
axes[0].set_title("Batched construction")
axes[0].legend()

axes[1].plot(p_plot, np.asarray(radius_derivatives), "o", label="JaxCont grad")
axes[1].plot(
    p_plot, np.asarray(exact_derivatives), "--", label=r"exact $1/(2\sqrt{p})$"
)
axes[1].set_xlabel("p")
axes[1].set_ylabel("d(radius) / dp")
axes[1].set_title("Differentiable construction")
axes[1].legend()

plt.tight_layout()
plt.savefig(
    "images/example_23_periodic_vmap_grad.png", dpi=150, bbox_inches="tight"
)
plt.show()
