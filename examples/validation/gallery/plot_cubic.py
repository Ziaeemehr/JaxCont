r"""
Cubic folds: JaxCont and MatCont
================================

The system
----------
The scalar equation

.. math:: \dot x = p + x - \frac{x^3}{3}

has one state :math:`x` and one parameter :math:`p`. Its equilibria satisfy
:math:`p = x^3/3 - x`, an S-shaped curve. It is the calibration control of the
validation suite: the answer is known in closed form, so it checks the
exporters and the comparison machinery before the harder systems.
"""

# %%
# What to look for
# ----------------
# Plotted as :math:`x` against :math:`p`, the equilibria form an S. The curve
# turns back at two limit points (LP, folds) where
# :math:`f_x = 1 - x^2 = 0`, i.e. :math:`x = \pm 1` and
# :math:`p = \mp 2/3`. The outer two sheets (:math:`|x| > 1`) are stable and
# the middle sheet between the folds is unstable. A continuation method has to
# follow the curve around each fold; pseudo-arclength continuation does that.
# Agreement means the curves coincide and both tools place the LP points at
# the same :math:`(p, x)`.

import argparse
import sys
import tempfile
from pathlib import Path

if "__file__" in globals():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "jaxcont" / "generated")
options = parser.parse_args() if __name__ == "__main__" and "__file__" in globals() else parser.parse_args([])

import jax.numpy as jnp

import jaxcont as jc
from examples.validation.cases.python_cases.equilibrium import run_cubic_fold
from examples.validation.reporting import make_report, save_checked_figure
from examples.validation.visualize import ReportTable, render_comparison, summarize_report
import examples.validation as validation_package

reference_dir = Path(validation_package.__file__).resolve().parent / "reference"

# %%
# The JaxCont computation
# -----------------------
# The equation, the seed on the lower sheet at :math:`p = -1`, the settings
# and the solve that produce the fresh JaxCont series.


def cubic(u, p, args):
    x = u[0]
    return jnp.array([p + x - x**3 / 3.0])


problem = jc.bif_problem(cubic, u0=jnp.array([-2.1038034027355366]), p0=-1.0)
settings = jc.ContinuationPar(ds=0.01, ds_min=1e-5, ds_max=0.05, max_steps=300,
                              newton_tol=1e-6, compute_stability=True)
result = jc.continuation(problem, jc.PseudoArclength(), p_span=(-1.0, 1.0),
                         settings=settings, events=[jc.Fold()])
natural = jc.continuation(problem, jc.Natural(), p_span=(-1.0, 1.0), settings=settings)
case = run_cubic_fold(computed=(result, natural))

# %%
# Comparison figures
# ------------------
# JaxCont is computed fresh; MatCont 7.6 is a committed reference bundle. The
# top figure overlays the two branches and their LP points. The curves
# coincide, so the second figure shows the pointwise difference in :math:`p`
# on a log scale, at the samples of the sparser series along the monotone
# coordinate :math:`x` (the denser one is evaluated there by a cubic spline,
# so the panel shows disagreement, not interpolation error).

report = make_report(case, reference_dir)
overlay, error = render_comparison(case, report, reference_dir)
save_checked_figure(overlay, report, options.output_dir / "plot_cubic_0.png")
save_checked_figure(error, report, options.output_dir / "plot_cubic_1.png")

# %%
# Quantitative summary
# --------------------
# Generated from the same report as the figures and their ``.checks.json``
# sidecars: each metric's measured value, its limit, and the verdict.

ReportTable(report)

# %%
# Takeaway
# --------

print(summarize_report(report))

# %%
# This is a calibration control: agreement here shows the exporters, the
# event matching and the comparison work, not that harder periodic problems
# agree. Those are on the other pages of this gallery.
