r"""
Cubic folds: JaxCont and MatCont
================================

The system
----------
The calibration system is :math:`\dot x=p+x-x^3/3`, with scalar state x
and continuation parameter p. It is the cubic normal form used in the
committed MatCont ``run_cubic_fold`` driver, with no additional parameters.
"""

# %%
# What to look for
# ----------------
# Expect an S-shaped branch with two LP (fold) points. The outer branches
# are stable and the middle branch unstable; colour identifies stability.
# Natural continuation reaches the first turn, whereas pseudo-arclength
# passes both. The error panel uses x as the monotone coordinate.

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


def _cubic_rhs(u, parameter, args):
    del args
    x = u[0]
    return jnp.array([parameter + x - x**3 / 3.0])


problem = jc.bif_problem(_cubic_rhs, u0=jnp.array([-2.1038034027355366]), p0=-1.0)
settings = jc.ContinuationPar(
    ds=0.01,
    ds_min=1e-5,
    ds_max=0.05,
    max_steps=300,
    newton_tol=1e-6,
    compute_stability=True,
)
result = jc.continuation(
    problem,
    jc.PseudoArclength(),
    p_span=(-1.0, 1.0),
    settings=settings,
    events=[jc.Fold()],
)
natural_result = jc.continuation(
    problem,
    jc.Natural(),
    p_span=(-1.0, 1.0),
    settings=settings,
)
from examples.validation.cases.python_cases.equilibrium import run_cubic_fold

case_result = run_cubic_fold(computed=(result, natural_result))
# %%
# Comparison figures
# ------------------
# JaxCont is computed fresh. MatCont 7.6 is a committed reference series
# from the bundled MatCont test systems; reference hashes are in the sidecars.
# The error panel reveals discrepancies hidden by coincident curves.

report = make_report(case_result, reference_dir)
figures = render_comparison(case_result, report, reference_dir)
for index, figure in enumerate(figures):
    save_checked_figure(figure, report, options.output_dir / f"plot_cubic_{index}.png")

# %%
# Quantitative summary
# --------------------
# This table is generated from the same report as the figures and sidecars.

summary = render_report_table(report)
save_checked_figure(summary, report, options.output_dir / "plot_cubic_table.png")

# %%
# Takeaway
# --------
# The report measures branch, event and spectrum agreement against MatCont.
# This is a calibration control; agreement here does not establish performance
# on applied periodic bifurcations.
