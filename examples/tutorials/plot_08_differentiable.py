r"""
Differentiate an isolated fold
==============================

For :math:`f(u,p;\theta)=u^2-\theta u+p`, the fold is
:math:`p^*=\theta^2/4`. The implicit fold solve supports reverse-mode grad.
The continuation while loop supports forward-mode, not reverse-mode; a
scan supports reverse-mode with a fixed attempt budget and masked inactive
iterations. Its derivative follows the locally selected branch and acceptance
decisions; it does not differentiate changes of branch or event identity.
This lesson differentiates the isolated solve, not the full adaptive sweep.
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
def rhs(u, p, theta):
    return jnp.array([u[0] ** 2 - theta * u[0] + p])


def fold_p(theta):
    return jc.fold_parameter(rhs, jnp.array([0.4]), jnp.array(0.2), theta)


problem = jc.bif_problem(rhs, u0=jnp.array([0.0]), p0=0.0, args=1.0)
settings = jc.ContinuationPar(ds=0.01, max_steps=100, newton_tol=1e-6)
# %%
# Continue from the visible seed with the settings below.

result = jc.continuation(problem, p_span=(0.0, 0.5), settings=settings, events=[jc.Fold()])
thetas = jnp.linspace(0.5, 1.5, 9)
folds = jax.vmap(fold_p)(thetas)
gradients = jax.vmap(jax.grad(fold_p))(thetas)
# %%
# The figure compares the implicit sensitivity with its analytic value.
fig, ax = plt.subplots()
ax.plot(thetas, gradients, "o", label="implicit gradient")
ax.plot(thetas, thetas / 2, "--", label="theta/2")
ax.set(xlabel="theta", ylabel="d(fold parameter)/d(theta)")
ax.legend()

# %%
# Save this lesson to the explicit output directory.

fig.savefig(options.output_dir / "plot_08_differentiable.png", dpi=150, bbox_inches="tight")
