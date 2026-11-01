r"""
Independent shooting check of PRC and dPRC
==========================================

The system
----------
The sheared radial oscillator obeys :math:`\dot r=r(\rho-r^2)` and
:math:`\dot\theta=1+0.4(\rho-r^2)`, at rho=1. Van der Pol obeys
:math:`\dot x=y,\ \dot y=\mu(1-x^2)y-x`, at mu=1.
The shooting method integrates the variational and backward adjoint equations
with SciPy DOP853. It checks phase response, using JaxCont's converged orbit
as its seed; orbit finding itself is outside this independent check.
"""

# %%
# What to look for
# ----------------
# Coincident curves mean agreement between collocation and IVP shooting.
# Parameter differentiation of the whole shooting calculation supplies an
# independent dPRC estimate; the difference panel reveals its finite-step error.

import argparse
import sys
import tempfile
from pathlib import Path

if "__file__" in globals():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "jaxcont" / "generated")
options = parser.parse_args() if __name__ == "__main__" and "__file__" in globals() else parser.parse_args([])

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem
from examples.validation.reporting import make_report, save_checked_figure
from examples.validation.visualize import ReportTable, render_comparison, summarize_report
import examples.validation as validation_package
reference_dir = Path(validation_package.__file__).resolve().parent / "reference"

# %%
# The JaxCont computation
# -----------------------
# The circle seed and public adjoint calls are visible below. The producer
# retains the original DOP853 variational/backward-adjoint method and the
# Van der Pol simulation seed, with unchanged integration tolerances.

from examples.validation.cases.python_cases.shooting import run_shooting
from examples.validation.reporting import ValidationReport

C_SHEAR, OMEGA = 0.4, 1.0
from jaxcont.stability.prc import prc_curve, dprc_curve


def sheared_circle_rhs(u, p, args):
    x, y = (u[0], u[1])
    rho = p
    rad = rho - (x * x + y * y)
    theta_dot = OMEGA + C_SHEAR * rad
    return jnp.array([rad * x - theta_dot * y, rad * y + theta_dot * x])


rng = np.random.default_rng(0)
t_traj = np.sort(rng.uniform(0, 2 * np.pi / OMEGA, size=40))
t_traj[0] = 0.0
t_traj[-1] = 2 * np.pi / OMEGA
theta_guess = 2 * np.pi * t_traj / (2 * np.pi / OMEGA) + 0.3
u_traj = jnp.asarray(np.stack([0.8 * np.cos(theta_guess), 0.8 * np.sin(theta_guess)], axis=1))
mesh_circle = Collocation(ntst=10, ncol=4)
rho0 = 1.0
prob_circle = periodic_orbit_problem(
    sheared_circle_rhs, u_traj, jnp.asarray(t_traj), 2 * np.pi / OMEGA, rho0, mesh_circle
)
with jax.default_matmul_precision("float32"):
    Z_jaxcont_circle = np.asarray(
        prc_curve(sheared_circle_rhs, mesh_circle, prob_circle.u0, prob_circle.p0)
    )
    dZ_jaxcont_circle = np.asarray(dprc_curve(prob_circle))

case_result = run_shooting(circle=(prob_circle, mesh_circle, Z_jaxcont_circle, dZ_jaxcont_circle))
pairs = case_result.artifacts["pairs"]
metrics = [
    dict(
        metric=k, measured=case_result.checks[k], limit=limit, passed=case_result.checks[k] < limit
    )
    for k, (actual, reference, limit) in pairs.items()
]
report = ValidationReport(
    "shooting.prc-check",
    "PASS" if all(r["passed"] for r in metrics) else "FAIL",
    metrics,
    case_result.checks,
    None,
    None,
    "float64" if jax.config.x64_enabled else "float32",
    {},
    compared_domain={"phase_fraction": [0.0, 1.0]},
)

# %%
# Comparison figures
# ------------------
# Both JaxCont and independent shooting are fresh computations. This page
# does not use an external continuation reference series.

figure, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, (name, (actual, reference, limit)) in zip(axes.flat, pairs.items()):
    phase = np.arange(len(actual)) / len(actual)
    ax.plot(phase, actual[:, 0], "o-", label="JaxCont fresh")
    ax.plot(phase, reference[:, 0], "x--", label="shooting fresh")
    ax.set(title=name, xlabel="phase fraction", ylabel="x response")
    ax.legend()
figure.tight_layout()
save_checked_figure(figure, report, options.output_dir / "plot_prc_shooting.png")
error_figure, ax = plt.subplots()
for name, (actual, reference, limit) in pairs.items():
    ax.plot(
        np.arange(len(actual)) / len(actual), np.max(np.abs(actual - reference), axis=1), label=name
    )
ax.set(xlabel="phase fraction", ylabel="max component absolute error")
ax.legend()
save_checked_figure(error_figure, report, options.output_dir / "plot_prc_shooting_error.png")

# %%
# Quantitative summary
# --------------------
# The measured differences and predicates are taken from the report.

ReportTable(report)

# %%
# Takeaway
# --------

print(summarize_report(report))

# %%
# The report distinguishes PRC agreement from the finite-difference dPRC
# check. This validates adjoint response on a shared orbit, and does not
# establish an independent orbit-finding result.
