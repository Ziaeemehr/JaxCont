r"""
Lorenz-84 equilibrium events
============================

Follow the atmospheric model of Lorenz (1984), extended by U:

.. math::

   \dot X&=-Y^2-Z^2-\alpha X+\alpha F-\gamma U^2,\\
   \dot Y&=XY-\beta XZ-Y+G,\quad \dot Z=\beta XY+XZ-Z,\\
   \dot U&=-\delta U+\gamma UX+T.

The F seed below is on the branch. The lower endpoint sets the initial
direction; the fold turns the branch upward before it reaches that endpoint.
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
options = parser.parse_args() if __name__ == "__main__" and "__file__" in globals() else parser.parse_args([])
options.output_dir.mkdir(parents=True, exist_ok=True)


# %%
def lorenz84_rhs(state, F, args):
    X, Y, Z, U = state
    alpha, beta, gamma, delta, G, T = args
    dX = -(Y**2) - Z**2 - alpha * X + alpha * F - gamma * U**2
    dY = X * Y - beta * X * Z - Y + G
    dZ = beta * X * Y + X * Z - Z
    dU = -delta * U + gamma * U * X + T
    return jnp.array([dX, dY, dZ, dU])


F0 = 1.7620532879639
args = (0.25, 1.0, 0.987, 1.04, 0.25, 0.04)
u0 = jnp.array([1.6673192028567203, -0.05172586841139392, 0.12923880103788027, -0.0660453938041009])
prob = jc.bif_problem(
    lorenz84_rhs, u0=u0, p0=F0, args=args, state_names=["X", "Y", "Z", "U"], param_name="F"
)
# %%
# Continue from the visible seed with the settings below.

result = jc.continuation(
    prob,
    jc.PseudoArclength(),
    p_span=(F0, 1.0),
    settings=jc.ContinuationPar(
        ds=0.005, ds_max=0.01, max_steps=215, newton_tol=1e-06, compute_stability=True
    ),
    events=[jc.Fold(), jc.Hopf()],
    verbose=False,
)
# %%
# Plot only the computed equilibria.
fig, ax = plt.subplots()
result.plot(ax=ax, annotate=True)

# %%
# Save this lesson to the explicit output directory.

fig.savefig(options.output_dir / "plot_03_lorenz84.png", dpi=150, bbox_inches="tight")
