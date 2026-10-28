r"""
Radial cycle: an exact Floquet control
======================================

The system
----------
The radial Hopf normal form has state (x,y), free rho and unit angular
frequency:

.. math::

   \dot x=(\rho-x^2-y^2)x-y,\quad
   \dot y=(\rho-x^2-y^2)y+x.

The committed MatCont series uses ``run_radial_cycle``. The cycle has
radius sqrt(rho), period two pi and nontrivial multiplier exp(-4 pi rho).
"""

# %%
# What to look for
# ----------------
# The envelope grows while the period stays constant. The nontrivial
# multiplier stays inside the unit circle, marking a stable cycle; the
# trivial multiplier at one represents time translation. The error panel
# magnifies period differences hidden by the overlapping curves.

import argparse
import tempfile
from pathlib import Path
import sys
if "__file__" in globals():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jaxcont as jc
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem
from examples.validation.reporting import make_report, save_checked_figure
from examples.validation.visualize import render_comparison, render_report_table

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "jaxcont" / "generated"
)
options = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])
import examples.validation as validation_package
reference_dir = Path(validation_package.__file__).resolve().parent / "reference"

# %%
# The JaxCont computation
# -----------------------
# The following equations, seed, settings, and solve produce the fresh series.


def _radial_rhs(u, rho, args):
    del args
    x, y = u
    radius_squared = x * x + y * y
    return jnp.array([(rho - radius_squared) * x - y, (rho - radius_squared) * y + x])


mesh = Collocation(ntst=10, ncol=4)
rho = 0.5
period = 2 * np.pi
time = jnp.linspace(0.0, period, 80, endpoint=False)
trajectory = jnp.sqrt(rho) * jnp.stack([jnp.cos(time), jnp.sin(time)], axis=1)
problem = periodic_orbit_problem(_radial_rhs, trajectory, time, period, rho, mesh)
result = jc.continuation(
    problem,
    jc.PseudoArclength(),
    p_span=(0.5, 1.5),
    settings=jc.ContinuationPar(
        compute_stability=True,
        ds=0.05,
        ds_max=0.08,
        max_steps=35,
    ),
)
from examples.validation.cases.python_cases.periodic import run_radial_cycle

case_result = run_radial_cycle(computed=(result, mesh, problem))
# %%
# Comparison figures
# ------------------
# JaxCont is computed fresh. MatCont 7.6 is a committed reference series
# from the bundled MatCont test systems; reference hashes are in the sidecars.
# The error panel reveals discrepancies hidden by coincident curves.

report = make_report(case_result, reference_dir)
figures = render_comparison(case_result, report, reference_dir)
for index, figure in enumerate(figures):
    save_checked_figure(figure, report, options.output_dir / f"plot_radial_{index}.png")

# %%
# Quantitative summary
# --------------------
# This table is generated from the same report as the figures and sidecars.

summary = render_report_table(report)
save_checked_figure(summary, report, options.output_dir / "plot_radial_table.png")

# %%
# Takeaway
# --------
# The report compares the fresh cycle with the committed series and the
# exact solution. This is an analytic Floquet control, so it does not test
# nonlinear torus or doubled-cycle switching.
