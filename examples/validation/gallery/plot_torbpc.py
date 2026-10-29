r"""
torBPC: cycles with LPC, PD and NS
===================================

The system
----------
MatCont's ``testtorBPC1`` uses state (x,y,z), free nu, and fixed
beta=0.5, gamma=-0.6, r=0.6, a3=0.32858, b3=0.93358, epsilon=0.001:

.. math::

   r\dot x&=-(\beta+\nu)x+\beta y-a_3x^3+b_3(y-x)^3,\\
   \dot y&=\beta x-(\beta+\gamma)y-z-b_3(y-x)^3,\\
   \dot z&=y+\epsilon.

The committed series is MatCont's full LPC/PD/NS branch, without trimming.
"""

# %%
# What to look for
# ----------------
# LPC is a turning point of cycles, PD a multiplier crossing minus one,
# and NS a complex pair crossing the unit circle. The envelopes and periods
# show the branch; the event-centred spectrum compares the multipliers at
# each located event. JaxCont uses the external tools' mesh (ntst=25) and
# compares its refined events, not the nearest branch samples.

import argparse
import sys
import tempfile
from pathlib import Path

if "__file__" in globals():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "jaxcont" / "generated")
options = parser.parse_args() if __name__ == "__main__" and "__file__" in globals() else parser.parse_args([])
# The Newton tolerance 1e-8 requires double precision in either gallery mode.
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import jaxcont as jc
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem
from examples.validation.reporting import make_report, save_checked_figure
from examples.validation.visualize import ReportTable, render_comparison, summarize_report
import examples.validation as validation_package
reference_dir = Path(validation_package.__file__).resolve().parent / "reference"

# %%
# The JaxCont computation
# -----------------------
# The following equations, seed, settings, and solve produce the fresh series.

from examples.validation.cases.python_cases.periodic import (
    _TORBPC_FIXED,
    _torbpc_hopf_seed,
    run_torbpc_cycle,
)


def _torbpc_rhs(u, nu, args):
    del args
    beta = _TORBPC_FIXED["beta"]
    gamma = _TORBPC_FIXED["gamma"]
    scale = _TORBPC_FIXED["r"]
    a3 = _TORBPC_FIXED["a3"]
    b3 = _TORBPC_FIXED["b3"]
    epsilon = _TORBPC_FIXED["epsilon"]
    x, y, z = u
    difference = y - x
    return jnp.array(
        [
            (-(beta + nu) * x + beta * y - a3 * x**3 + b3 * difference**3) / scale,
            beta * x - (beta + gamma) * y - z - b3 * difference**3,
            y + epsilon,
        ]
    )


start_parameter = -0.589286291695516
trajectory, period = _torbpc_hopf_seed(start_parameter)
time = np.linspace(0.0, period, trajectory.shape[0], endpoint=False)
mesh = Collocation(ntst=25, ncol=4)
problem = periodic_orbit_problem(
    _torbpc_rhs,
    jnp.asarray(trajectory),
    jnp.asarray(time),
    period,
    start_parameter,
    mesh,
    tol=1e-8,
)
result = jc.continuation(
    problem,
    jc.PseudoArclength(),
    p_span=(start_parameter, -0.75),
    settings=jc.ContinuationPar(
        # Preserve dense Euclidean sampling for nearest-sample comparisons.
        theta=1.0,
        compute_stability=True,
        ds=0.005,
        ds_min=1e-5,
        ds_max=0.02,
        max_steps=400,
        # A looser tolerance leaves the orbit too inaccurate to identify
        # the flow mode of the Floquet spectrum.
        newton_tol=1e-8,
    ),
    events=[
        jc.Fold(),
        jc.NeimarkSacker(corrector_tol=1e-8),
        jc.PeriodDoubling(corrector_tol=1e-8),
    ],
)
case_result = run_torbpc_cycle(reference_dir, computed=(result, mesh))
# %%
# Comparison figures
# ------------------
# JaxCont is computed fresh. MatCont 7.6 is a committed reference series
# from the bundled MatCont test systems; reference hashes are in the sidecars.
# The error panel reveals discrepancies hidden by coincident curves.

report = make_report(case_result, reference_dir)
figures = render_comparison(case_result, report, reference_dir)
for index, figure in enumerate(figures):
    save_checked_figure(figure, report, options.output_dir / f"plot_torbpc_{index}.png")

# %%
# Quantitative summary
# --------------------
# This table is generated from the same report as the figures and sidecars.

ReportTable(report)

# %%
# Takeaway
# --------

print(summarize_report(report))

# %%
# All three event types are located and their multipliers agree with
# MatCont at the located events; the event-local error panel and the table
# show the remaining differences.
