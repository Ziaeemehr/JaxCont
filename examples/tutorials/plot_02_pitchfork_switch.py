r"""
Switch both arms of a pitchfork
===============================

For :math:`\dot u=pu-u^3`, the trivial equilibrium meets
:math:`u=\pm\sqrt p` at a branch point. Switch with both signs of the
predictor step to find both arms.
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
def rhs(u, p, args):
    return jnp.array([p * u[0] - u[0] ** 3])


problem = jc.bif_problem(rhs, u0=jnp.array([0.0]), p0=-0.5)
settings = jc.ContinuationPar(ds=0.02, max_steps=60, newton_tol=1e-6, compute_stability=True)
# %%
# Continue from the visible seed with the settings below.

result = jc.continuation(problem, p_span=(-0.5, 0.5), settings=settings, events=[jc.BP()])
event = next(e for e in result.events if e.kind == "branch_point")
arms = []
for sign in (-1, 1):
    seed = jc.branch_switch(rhs, event, ds=sign * 0.05)
    arms.append(jc.continuation(seed, p_span=(float(seed.p0), 0.5), settings=settings))
# %%
# The nonzero arms satisfy u squared = p.
fig, ax = plt.subplots()
result.plot(ax=ax)
for arm in arms:
    ax.plot(arm.branch.params, arm.branch.states[:, 0])
ax.set(xlabel="p", ylabel="u", title="Pitchfork branch switching")

# %%
# Save this lesson to the explicit output directory.

fig.savefig(options.output_dir / "plot_02_pitchfork_switch.png", dpi=150, bbox_inches="tight")
