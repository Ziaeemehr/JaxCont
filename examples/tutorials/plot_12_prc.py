r"""
Phase response curve of a circular limit cycle
==============================================

Compute the infinitesimal phase response curve (iPRC) and its parameter
derivative (dPRC) for ``r' = r*(rho - r^2), theta' = 1`` at ``rho=1`` -- the
same closed-form circle system ``tests/test_floquet.py``/``tests/test_prc.py``
use, whose exact limit cycle is ``x=cos(t), y=sin(t)``, ``T=2*pi``.

See docs/superpowers/specs/2026-08-05-prc-dprc-design.md.

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
from jaxcont.stability.prc import dprc_curve, prc_curve
from jaxcont.viz import plot_prc
import jaxcont as jc


def rhs(u, p, args):
    x, y = (u[0], u[1])
    r2 = x * x + y * y
    rho = p
    return jnp.array([(rho - r2) * x - y, (rho - r2) * y + x])


rng = np.random.default_rng(0)
t_traj = np.sort(rng.uniform(0, 5.5, size=40))
t_traj[0] = 0.0
theta_guess = 2 * np.pi * t_traj / 5.5 + 0.3
u_traj = jnp.asarray(np.stack([0.8 * np.cos(theta_guess), 0.8 * np.sin(theta_guess)], axis=1))
# %%
# Refine the trajectory on a collocation mesh.

mesh = Collocation(ntst=40, ncol=4)
problem = periodic_orbit_problem(rhs, u_traj, jnp.asarray(t_traj), 5.5, 1.0, mesh)
settings = jc.ContinuationPar(ds=0.02, max_steps=3, compute_stability=False)
# %%
# Continue from the visible seed with the settings below.

result = jc.continuation(problem, p_span=(1.0, 1.05), settings=settings)
with jax.default_matmul_precision("float32"):
    Z = prc_curve(rhs, mesh, problem.u0, problem.p0)
    dZ = dprc_curve(problem)
fig, (ax_prc, ax_dprc) = plt.subplots(1, 2, figsize=(12, 5))
plot_prc(Z, ax=ax_prc, labels=["x", "y"], title="iPRC")
plot_prc(dZ, ax=ax_dprc, labels=["x", "y"], title="dPRC (d/dρ)")
plt.tight_layout()
# %%
# Save this lesson to the explicit output directory.

fig.savefig(options.output_dir / "plot_12_prc.png", dpi=150, bbox_inches="tight")
