r"""
adaptx: Hopf, phase response and period doubling
=================================================

The system
----------
MatCont's adaptive-control test system (``TestSystems/adaptx.m``) has state
(x,y,z), free alpha, and beta fixed to one:

.. math::

   \dot x=y,\quad \dot y=z,\quad
   \dot z=-\alpha z-\beta y-x+x^2.

The equilibrium computation starts at the origin. The PRC uses the
converged alpha and period recorded by the committed MatCont bundle. The
last section follows the cycle born at the Hopf point down to its first
period doubling and switches onto the doubled branch, against MatCont and
BifurcationKit.
"""

# %%
# What to look for
# ----------------
# The spectral panel reveals H on the otherwise flat equilibrium branch.
# The iPRC describes phase sensitivity
# along the cycle: its phase is aligned and radian units converted to cycle
# fractions before comparison. MatCont's time derivative of PRC is not
# JaxCont's parameter derivative and is never treated as a matching series.

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
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem
from examples.validation.reporting import make_report, save_checked_figure
from examples.validation.cases.python_cases.pd_switch import run_adaptx_pd_three_tool
from examples.validation.visualize import ReportTable, render_comparison, summarize_report
import examples.validation as validation_package
reference_dir = Path(validation_package.__file__).resolve().parent / "reference"

# %%
# The JaxCont computation
# -----------------------
# The following equations, seed, settings, and solve produce the fresh series.


def _adaptive_control_rhs(u, alpha, beta):
    """MatCont ``adaptx`` equilibrium system, with ``alpha`` free."""
    beta = 1.0 if beta is None else beta
    x, y, z = u
    return jnp.array([y, z, -alpha * z - beta * y - x + x**2])


beta = jnp.array(1.0)
problem = jc.bif_problem(
    _adaptive_control_rhs,
    u0=jnp.zeros(3),
    p0=-2.0,
    args=beta,
)
result = jc.continuation(
    problem,
    jc.PseudoArclength(),
    p_span=(-2.0, 2.0),
    settings=jc.ContinuationPar(
        ds=0.03,
        ds_max=0.08,
        max_steps=100,
        newton_tol=1e-7,
        compute_stability=True,
    ),
    events=[jc.Hopf()],
)
from examples.validation.cases.python_cases.equilibrium import run_adaptive_control_hopf

hopf_result = run_adaptive_control_hopf(computed=result)
# %%
# Phase response at the committed cycle parameter
# ------------------------------------------------
# The seed uses the committed parameter/period and a coarse Hopf-direction
# trajectory; JaxCont refines the orbit and computes the adjoint fresh.

from examples.validation.cases.python_cases.prc import (
    _load_converged_point,
    run_adaptx_prc_dprc,
)

def cycle_rhs(u, p, args):
    x, y, z = u
    alpha, beta = p
    return jnp.array([y, z, -alpha*z - beta*y - x + x*x])

alpha, period = _load_converged_point(reference_dir)
n_guess = 60
t_traj = jnp.linspace(0.0, period, n_guess, endpoint=False)
# adaptx's Hopf eigenvector direction (see run_prc_dprc.m's init_H_LC
# call): near the Hopf point x=y=z=0, the linearization's imaginary
# eigenvector for x'=y, y'=z, z'=-alpha*z-beta*y-x+x^2 (beta=1,
# omega=1 at alpha=1) is approximately (1, -i, -1) in (x, y, z) --
# i.e. y leads x by a quarter period and z is antiphase to x. This
# coarse guess only needs the right shape/phase relationship; the
# collocation Newton solve (periodic_orbit_problem's internal
# differentiable_root) converges it to the true nonlinear cycle at
# (alpha, period).
omega = 2 * jnp.pi / period
amplitude = 0.5
u_traj = jnp.stack(
    [
        amplitude * jnp.cos(omega * t_traj),
        -amplitude * jnp.sin(omega * t_traj),
        -amplitude * jnp.cos(omega * t_traj),
    ],
    axis=1,
)
mesh = Collocation(ntst=20, ncol=4)
p0 = jnp.array([alpha, 1.0])
problem = periodic_orbit_problem(cycle_rhs, u_traj, t_traj, period, p0, mesh)
Z = jc.prc_curve(cycle_rhs, mesh, problem.u0, problem.p0)
prc_result = run_adaptx_prc_dprc(reference_dir, computed=(problem, mesh))
# %%
# Comparison figures
# ------------------
# JaxCont is fresh; MatCont 7.6 is a committed series. Each family has an
# overlay and an error panel, with status from its authoritative report.

reports = []
for label, case_result in [("hopf", hopf_result), ("prc", prc_result)]:
    report = make_report(case_result, reference_dir)
    reports.append((label, report))
    for index, figure in enumerate(render_comparison(case_result, report, reference_dir)):
        save_checked_figure(figure, report, options.output_dir / f"plot_adaptx_{label}_{index}.png")

# %%
# Period doubling and branch switching
# ------------------------------------
# What to look for: the limit cycle born at the Hopf point (alpha = 1) grows
# as alpha decreases. Near alpha = 0.63 one real Floquet multiplier passes
# through -1 (PD): the cycle loses stability and a cycle of twice the period
# appears. JaxCont locates the PD point, switches onto the doubled branch
# (period about 12.7, twice the base period) and continues it to its own PD
# near alpha = 0.58, the next step of the cascade. The overlay shows
# amplitude, period and the nontrivial multipliers (drawn as points, since
# modes are not tracked across samples in the bundles); the error panels
# compare each pair of tools sheet by sheet on the sparser tool's samples.
# The base and doubled branches are computed by ``run_adaptx_pd_switch`` in
# ``examples/validation/cases/python_cases/pd_switch.py``: collocation
# continuation of the cycle, the ``PeriodDoubling`` event, and
# ``jc.branch_switch_pd``.

pd_case = run_adaptx_pd_three_tool()  # reads the reviewed MatCont PD bundle
pd_report = make_report(pd_case, reference_dir)
for index, figure in enumerate(render_comparison(pd_case, pd_report, reference_dir)):
    save_checked_figure(figure, pd_report, options.output_dir / f"plot_adaptx_pd_{index}.png")

# %%
# Quantitative summary
# --------------------
# Both tables read measured values and unchanged limits from the reports.

hopf_report, prc_report = (report for _, report in reports)
ReportTable(hopf_report)

# %%
# The phase-response report:

ReportTable(prc_report)

# %%
# The period-doubling report:

ReportTable(pd_report)

# %%
# Takeaway
# --------

print(summarize_report(hopf_report))
print(summarize_report(prc_report))
print(summarize_report(pd_report))

# %%
# The Hopf and iPRC reports quantify agreement separately, preserving phase
# alignment and unit conversion. JaxCont's dPRC is a parameter sensitivity;
# this page makes no MatCont cross-validation claim for that quantity. The
# period-doubling report compares both branches and both PD points across all
# three tools, so the switch is checked to land on the same doubled cycle.
