r"""
Neural mass equilibria
======================

Continue steady neural activity in external input E0:

.. math::

   \dot E&=(-E+S(JuxE+E_0))/\tau,\\
   \dot x&=(1-x)/\tau_D-uxE,\quad
   \dot u=(U_0-u)/\tau_F+U_0(1-u)E,

where :math:`S(z)=\alpha\log(1+e^{z/\alpha})`.
The corrector stalls before E0 reaches the requested endpoint. The figure
shows the computed domain, including the reachable fold and Hopf events.
"""

# %%
# Set up the output directory and imports. Figures stay open for the
# Sphinx-Gallery matplotlib scraper. Select precision with JAX_ENABLE_X64.

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
options = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])
options.output_dir.mkdir(parents=True, exist_ok=True)


# %%
def TMvf(state, E0, args):
    E, x, u = state
    J, alpha, tau, tauD, tauF, U0 = args
    SS0 = J * u * x * E + E0
    SS1 = alpha * jnp.log(1 + jnp.exp(SS0 / alpha))
    dE = (-E + SS1) / tau
    dx = (1.0 - x) / tauD - u * x * E
    du = (U0 - u) / tauF + U0 * (1.0 - u) * E
    return jnp.array([dE, dx, du])


E0_0 = -2.0
args = (3.07, 1.5, 0.013, 0.2, 1.5, 0.3)
z0_guess = jnp.array([0.238616, 0.982747, 0.367876])
residual_norm = jnp.linalg.norm(TMvf(z0_guess, E0_0, args))
z0 = z0_guess
prob = jc.bif_problem(TMvf, u0=z0, p0=E0_0, args=args, state_names=["E", "x", "u"], param_name="E0")
# %%
# Continue from the visible seed with the settings below.

result = jc.continuation(
    prob,
    jc.PseudoArclength(),
    p_span=(E0_0, -0.9),
    settings=jc.ContinuationPar(ds=0.02, max_steps=400, newton_tol=1e-05, compute_stability=True),
    events=[jc.Fold(), jc.Hopf()],
    verbose=False,
)
# %%
# Plot only the computed equilibria.
# %%
# Plot the computed observables.

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, ax in enumerate(axes):
    result.plot(state_index=i, ax=ax)

# %%
# Save this lesson to the explicit output directory.

fig.savefig(options.output_dir / "plot_05_neural_mass.png", dpi=150, bbox_inches="tight")
