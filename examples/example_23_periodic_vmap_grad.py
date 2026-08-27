"""
Batched and differentiable periodic-orbit construction
=======================================================

Build a batch of Hopf-normal-form limit cycles with ``jax.vmap`` and
differentiate their radii through ``periodic_orbit_problem`` construction.
The exact cycle radius is ``sqrt(p)`` and its derivative is
``1 / (2 * sqrt(p))``.
"""

import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem


def rhs(u, p, args):
    x, y = u[0], u[1]
    r2 = x * x + y * y
    return jnp.array([(p - r2) * x - y, (p - r2) * y + x])


def main():
    os.makedirs("images", exist_ok=True)

    mesh = Collocation(ntst=10, ncol=4)
    t = jnp.linspace(0.0, 2 * jnp.pi, 120, endpoint=False)

    def cycle_radius(p):
        u_guess = jnp.sqrt(p) * jnp.stack([jnp.cos(t), jnp.sin(t)], axis=1)
        problem = periodic_orbit_problem(
            rhs, u_guess, t, 2 * jnp.pi, p, mesh
        )
        mesh_states = problem.u0[: 2 * mesh.ntst].reshape(mesh.ntst, 2)
        return jnp.mean(jnp.linalg.norm(mesh_states, axis=1))

    parameters = jnp.linspace(0.5, 1.5, 9)
    radii = jax.vmap(cycle_radius)(parameters)
    radius_derivatives = jax.vmap(jax.grad(cycle_radius))(parameters)

    exact_radii = jnp.sqrt(parameters)
    exact_derivatives = 1.0 / (2.0 * jnp.sqrt(parameters))
    radius_error = float(jnp.max(jnp.abs(radii - exact_radii)))
    derivative_error = float(
        jnp.max(jnp.abs(radius_derivatives - exact_derivatives))
    )
    print(f"vmap radius max error: {radius_error:.6e}")
    print(f"grad radius-derivative max error: {derivative_error:.6e}")

    p_plot = np.asarray(parameters)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(p_plot, np.asarray(radii), "o", label="JaxCont vmap")
    axes[0].plot(p_plot, np.asarray(exact_radii), "--", label=r"exact $\sqrt{p}$")
    axes[0].set_xlabel("p")
    axes[0].set_ylabel("cycle radius")
    axes[0].set_title("Batched construction")
    axes[0].legend()

    axes[1].plot(
        p_plot, np.asarray(radius_derivatives), "o", label="JaxCont grad"
    )
    axes[1].plot(
        p_plot,
        np.asarray(exact_derivatives),
        "--",
        label=r"exact $1/(2\sqrt{p})$",
    )
    axes[1].set_xlabel("p")
    axes[1].set_ylabel("d(radius) / dp")
    axes[1].set_title("Differentiable construction")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        "images/example_23_periodic_vmap_grad.png", dpi=150, bbox_inches="tight"
    )
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
