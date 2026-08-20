"""
Phase response curve of a circular limit cycle
================================================

Compute the infinitesimal phase response curve (iPRC) and its parameter
derivative (dPRC) for ``r' = r*(rho - r^2), theta' = 1`` at ``rho=1`` -- the
same closed-form circle system ``tests/test_floquet.py``/``tests/test_prc.py``
use, whose exact limit cycle is ``x=cos(t), y=sin(t)``, ``T=2*pi``.

See docs/superpowers/specs/2026-08-05-prc-dprc-design.md.
"""

# %%
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem
from jaxcont.stability.prc import dprc_curve, prc_curve
from jaxcont.viz import plot_prc


def rhs(u, p, args):
    x, y = u[0], u[1]
    r2 = x * x + y * y
    rho = p
    return jnp.array([(rho - r2) * x - y, (rho - r2) * y + x])


# %%
# Build a converged periodic orbit
# ---------------------------------
#
# ``periodic_orbit_problem`` Newton-converges a coarse initial guess into a
# collocation representation of the true limit cycle.

rng = np.random.default_rng(0)
t_traj = np.sort(rng.uniform(0, 5.5, size=40))
t_traj[0] = 0.0
theta_guess = 2 * np.pi * t_traj / 5.5 + 0.3
u_traj = jnp.asarray(np.stack([0.8 * np.cos(theta_guess), 0.8 * np.sin(theta_guess)], axis=1))

mesh = Collocation(ntst=40, ncol=4)
problem = periodic_orbit_problem(rhs, u_traj, jnp.asarray(t_traj), 5.5, 1.0, mesh)

# %%
# Compute the iPRC and dPRC
# --------------------------

with jax.default_matmul_precision("float32"):
    Z = prc_curve(rhs, mesh, problem.u0, problem.p0)
    dZ = dprc_curve(problem)

# %%
# Plot both curves against mesh-point index
# -------------------------------------------

fig, (ax_prc, ax_dprc) = plt.subplots(1, 2, figsize=(12, 5))
plot_prc(Z, ax=ax_prc, labels=["x", "y"], title="iPRC")
plot_prc(dZ, ax=ax_dprc, labels=["x", "y"], title="dPRC (d/dρ)")
plt.tight_layout()
plt.savefig("images/example_13_phase_response_curve.png", dpi=180, bbox_inches="tight")
plt.show()
