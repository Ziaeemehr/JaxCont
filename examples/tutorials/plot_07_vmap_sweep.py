r"""
Batch equilibrium branches with vmap
====================================

For :math:`f(u,p;b)=pu-u^3+b`, sweep the imperfection b.
Use Branch.valid to discard fixed-buffer padding in each lane. Events are
supported via result.event_candidates and jc.refine_event_candidates;
this lesson plots only the equilibrium branches.
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
def rhs(u, p, b):
    return jnp.array([p * u[0] - u[0] ** 3 + b])


settings = jc.ContinuationPar(
    ds=0.05, ds_min=1e-5, ds_max=0.2, max_steps=120, newton_tol=1e-6, newton_max_iter=20
)


def run_one(b):
    problem = jc.bif_problem(rhs, u0=jnp.array([0.05]), p0=-1.0, args=b)
    return jc.continuation(problem, p_span=(-1.0, 2.0), settings=settings)


bs = jnp.linspace(-0.4, 0.4, 9)
result = jax.vmap(run_one)(bs)
# %%
# Mask each lane separately; branch lengths can differ.
fig, ax = plt.subplots()
for i, b in enumerate(bs):
    mask = result.branch.valid[i]
    ax.plot(result.branch.params[i][mask], result.branch.states[i, :, 0][mask], label=f"b={b:.2f}")
ax.set(xlabel="p", ylabel="u")
ax.legend(ncol=3)

# %%
# Save this lesson to the explicit output directory.

fig.savefig(options.output_dir / "plot_07_vmap_sweep.png", dpi=150, bbox_inches="tight")
