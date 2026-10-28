r"""
Van der Pol: a degenerate equilibrium crossing
==============================================

The system
----------
The Van der Pol oscillator has state (x,y) and free parameter mu:

.. math::

   \dot x=y,\qquad \dot y=\mu(1-x^2)y-x.

The equilibrium seed is the origin; there are no other fixed parameters.
The committed series comes from MatCont's ``run_vanderpol_hopf`` driver.
"""

# %%
# What to look for
# ----------------
# The equilibrium stays at the origin, so the spectral panel is the useful
# bifurcation diagram: the largest real part crosses zero at H. Colour marks
# stable and unstable equilibria. This crossing is degenerate: at zero the
# system is a linear center and its first Lyapunov coefficient vanishes.

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


def _van_der_pol_rhs(u, parameter, args):
    del args
    x, y = u
    return jnp.array([y, parameter * (1.0 - x**2) * y - x])


problem = jc.bif_problem(_van_der_pol_rhs, u0=jnp.array([0.0, 0.0]), p0=-2.0)
result = jc.continuation(
    problem,
    jc.PseudoArclength(),
    p_span=(-2.0, 2.0),
    settings=jc.ContinuationPar(
        ds=0.02,
        ds_max=0.05,
        max_steps=160,
        newton_tol=1e-7,
        compute_stability=True,
    ),
    events=[jc.Hopf()],
)
from examples.validation.cases.python_cases.equilibrium import run_vanderpol_hopf

case_result = run_vanderpol_hopf(computed=result)
# %%
# Comparison figures
# ------------------
# JaxCont is computed fresh. MatCont 7.6 is a committed reference series
# from the bundled MatCont test systems; reference hashes are in the sidecars.
# The error panel reveals discrepancies hidden by coincident curves.

report = make_report(case_result, reference_dir)
figures = render_comparison(case_result, report, reference_dir)
for index, figure in enumerate(figures):
    save_checked_figure(figure, report, options.output_dir / f"plot_vanderpol_{index}.png")

# %%
# Quantitative summary
# --------------------
# This table is generated from the same report as the figures and sidecars.

summary = render_report_table(report)
save_checked_figure(summary, report, options.output_dir / "plot_vanderpol_table.png")

# %%
# Takeaway
# --------
# The report checks the equilibrium, critical spectrum and degenerate
# normal form against MatCont. This agreement does not imply a generic
# small-amplitude Hopf cycle; the finite-amplitude cycle needs its own seed.
