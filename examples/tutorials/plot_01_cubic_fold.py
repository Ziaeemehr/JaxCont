r"""
Follow a cubic through two folds
================================

The scalar system :math:`\dot x=p+x-x^3/3` folds at :math:`x=\pm1`.
Natural continuation stalls near a fold; pseudo-arclength continues through
both turns. Solid and dashed segments indicate stable and unstable equilibria.
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
def rhs(u, p, args):
    return jnp.array([p + u[0] - u[0] ** 3 / 3])


problem = jc.bif_problem(rhs, u0=jnp.array([-2.1038034027355366]), p0=-1.0)
settings = jc.ContinuationPar(
    ds=0.01, ds_min=1e-5, ds_max=0.05, max_steps=300, newton_tol=1e-6, compute_stability=True
)
# %%
# Continue from the visible seed with the settings below.

result = jc.continuation(
    problem, jc.PseudoArclength(), p_span=(-1.0, 1.0), settings=settings, events=[jc.Fold()]
)
natural = jc.continuation(problem, jc.Natural(), p_span=(-1.0, 1.0), settings=settings)
# %%
# Compare the two methods on the same seed and settings.
fig, ax = plt.subplots()
result.plot(ax=ax, annotate=True)
ax.plot(natural.branch.params, natural.branch.states[:, 0], ":", label="natural")
ax.set(xlabel="p", ylabel="x", title="Cubic: natural and pseudo-arclength")
ax.legend()

# %%
# Save this lesson to the explicit output directory.

fig.savefig(options.output_dir / "plot_01_cubic_fold.png", dpi=150, bbox_inches="tight")
