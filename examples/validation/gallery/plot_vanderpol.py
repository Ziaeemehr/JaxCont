r"""
Van der Pol: a degenerate equilibrium crossing
==============================================

The system
----------
The Van der Pol oscillator,

.. math:: \dot x = y, \qquad \dot y = \mu (1 - x^2)\, y - x,

has state :math:`(x, y)` and one parameter :math:`\mu`, with no other fixed
parameters (van der Pol, 1926). The origin is an equilibrium for every
:math:`\mu`, so the branch itself is trivial; what changes is its stability.
"""

# %%
# What to look for
# ----------------
# The Jacobian at the origin has eigenvalues
# :math:`\lambda = \mu/2 \pm \sqrt{\mu^2/4 - 1}`. For :math:`|\mu| < 2` they
# are a complex pair with real part :math:`\mu/2`, which crosses zero at
# :math:`\mu = 0`: a Hopf point (H), stable before and unstable after. At
# :math:`\mu = 2` the pair collides on the real axis, so the largest real
# part has a kink there. The crossing at :math:`\mu = 0` is degenerate: the
# system is a linear centre at that point and its first Lyapunov coefficient
# vanishes, so no small-amplitude cycle is born there. The state panel is
# flat at zero; the spectral panel is the informative bifurcation diagram.

# sphinx_gallery_start_ignore
import argparse
import sys
import tempfile
from pathlib import Path

if "__file__" in globals():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "jaxcont" / "generated")
options = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])
# sphinx_gallery_end_ignore

import jax.numpy as jnp

import jaxcont as jc
from examples.validation.cases.python_cases.equilibrium import run_vanderpol_hopf
from examples.validation.reporting import make_report, save_checked_figure
from examples.validation.visualize import ReportTable, render_comparison, summarize_report
import examples.validation as validation_package

reference_dir = Path(validation_package.__file__).resolve().parent / "reference"

# %%
# The JaxCont computation
# -----------------------
# Continue the origin from :math:`\mu = -2` to :math:`\mu = 2` with
# stability and Hopf detection.


def van_der_pol(u, mu, args):
    x, y = u
    return jnp.array([y, mu * (1.0 - x**2) * y - x])


problem = jc.bif_problem(van_der_pol, u0=jnp.array([0.0, 0.0]), p0=-2.0)
settings = jc.ContinuationPar(ds=0.02, ds_max=0.05, max_steps=160, newton_tol=1e-7, compute_stability=True)
result = jc.continuation(problem, jc.PseudoArclength(), p_span=(-2.0, 2.0), settings=settings,
                         events=[jc.Hopf()])
case = run_vanderpol_hopf(computed=result)

# %%
# Comparison figures
# ------------------
# JaxCont is computed fresh; MatCont 7.6 is a committed reference bundle. The
# first figure shows the branch and its largest eigenvalue real part, with the
# H point from each tool. The second shows the pointwise difference of that
# real part on a log scale. Points at round-off level are drawn on the floor as
# hollow markers. The rise in the last interval before :math:`\mu = 2` is the
# eigenvalue collision (a kink that no smooth interpolant follows), not a
# disagreement between the tools.

report = make_report(case, reference_dir)
overlay, error = render_comparison(case, report, reference_dir)
save_checked_figure(overlay, report, options.output_dir / "plot_vanderpol_0.png")
save_checked_figure(error, report, options.output_dir / "plot_vanderpol_1.png")

# %%
# Quantitative summary
# --------------------
# Generated from the same report as the figures and their ``.checks.json``
# sidecars.

ReportTable(report)

# %%
# Takeaway
# --------

print(summarize_report(report))

# %%
# Both tools place the H point at the analytic crossing and agree on the
# spectrum. Because the crossing is degenerate, this agreement says nothing
# about a small-amplitude Hopf cycle; the finite-amplitude Van der Pol cycle
# is covered by the tutorials with its own seed.
