r"""
Unpack a two-parameter fold curve
=================================

Use a shifted Bogdanov–Takens normal form:
:math:`\dot x=y,\ \dot y=b_1+b_2x+x^2+xy`, with
:math:`x=X-5,y=Y-2,b_1=p_0-3,b_2=p_1+1`.
Its fold curve is :math:`p_0=3+(p_1+1)^2/4`. CurveLayout describes
which augmented entries are physical state and solved parameter.
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
from jaxcont.bifurcations.curves import CurveLayout, unpack_minimal_fold_curve


def rhs(u, p, args):
    x, y = u - jnp.array([5.0, 2.0])
    return jnp.array([y, p[0] - 3 + (p[1] + 1) * x + x * x + x * y])


problem = jc.fold_curve_problem(rhs, jnp.array([5.5, 2.0]), jnp.array([3.25, -2.0]), free=1)
settings = jc.ContinuationPar(compute_stability=False, newton_tol=1e-5)
# %%
# Continue from the visible seed with the settings below.

result = jc.continuation(
    problem,
    p_span=(-2.0, 0.0),
    settings=settings,
    events=[jc.BogdanovTakens(raw_f=rhs, free=1, curve="fold")],
)
layout: CurveLayout = problem.curve_layout
unpacked = [unpack_minimal_fold_curve(row, n=layout.n) for row in result.branch.states]
solved = np.array([float(row[1]) for row in unpacked])
# %%
# Plot both parameters rather than interpreting the augmented state as an orbit.
fig, ax = plt.subplots()
ax.plot(result.branch.params, solved, label="fold curve")
ax.set(xlabel="p1 (free)", ylabel="p0 (solved)")
ax.legend()

# %%
# Save this lesson to the explicit output directory.

fig.savefig(options.output_dir / "plot_06_two_parameter.png", dpi=150, bbox_inches="tight")
