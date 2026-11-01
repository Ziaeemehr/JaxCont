r"""
Van der Pol equilibrium stability
=================================

For :math:`\dot x=y,\ \dot y=\mu(1-x^2)y-x`, the origin changes
stability at zero. This is a degenerate Hopf/center: the first Lyapunov
coefficient is zero and cycles do not emerge with vanishing amplitude.
The eigenvalue plot teaches the crossing; lesson 09 continues the actual cycle.
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
from jaxcont.viz import plot_eigenvalues


def rhs(u, mu, args):
    x, y = u
    return jnp.array([y, mu * (1 - x * x) * y - x])


problem = jc.bif_problem(rhs, u0=jnp.zeros(2), p0=-2.0)
settings = jc.ContinuationPar(
    ds=0.02, ds_max=0.05, max_steps=160, newton_tol=1e-7, compute_stability=True
)
# %%
# Continue from the visible seed with the settings below.

result = jc.continuation(problem, p_span=(-2.0, 2.0), settings=settings, events=[jc.Hopf()])
# %%
# Real parts cross zero while the critical imaginary parts remain nonzero.
fig = plot_eigenvalues(result, shade_stability=True, param_name="mu")

# %%
# Save this lesson to the explicit output directory.

fig.savefig(options.output_dir / "plot_04_vanderpol_equilibrium.png", dpi=150, bbox_inches="tight")
