r"""
Van der Pol limit cycle
=======================

Continue the actual Van der Pol limit cycle -- not just the equilibrium's
degenerate Hopf crossing at :math:`\mu=0` shown in
``plot_04_vanderpol_equilibrium.py`` -- via periodic-orbit collocation.

.. math::

    \dot{x} &= y \\
    \dot{y} &= \mu (1 - x^2) y - x

As explained in ``lesson 04``, the cycle does not emerge as a
vanishing-amplitude Hopf cycle at :math:`\mu=0` (the crossing is degenerate:
zero first Lyapunov coefficient). Starting instead from a short simulation
at :math:`\mu=1` -- comfortably past that degenerate point -- and refining
it into an exact periodic orbit gives a real starting point to continue
into the classic large-:math:`\mu` relaxation-oscillator regime.

The periodic lesson uses the 90-line allowance to keep trajectory seeding,
collocation settings, and the plotted observables visible.
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
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

NTST, NCOL, N_STATE = (40, 4, 2)


def van_der_pol_rhs(u, p, args):
    x, y = u
    mu = p
    return jnp.array([y, mu * (1.0 - x**2) * y - x])


def _van_der_pol_rhs_np(t, u, mu):
    x, y = u
    return [y, mu * (1.0 - x**2) * y - x]


mu_start = 1.0
raw_solution = solve_ivp(
    _van_der_pol_rhs_np, [0.0, 60.0], [0.5, 0.0], args=(mu_start,), max_step=0.01, dense_output=True
)
tail_times = np.linspace(40.0, 60.0, 4000)
tail_x = raw_solution.sol(tail_times)[0]
peak_indices, _ = find_peaks(tail_x)
peak_times = tail_times[peak_indices]
period_guess = float(np.mean(np.diff(peak_times)))
cycle_start_time = peak_times[-2]
t_trajectory = np.linspace(0.0, period_guess, 60, endpoint=False)
u_trajectory = raw_solution.sol(t_trajectory + cycle_start_time).T
# %%
# Refine the trajectory on a collocation mesh.

mesh = Collocation(ntst=NTST, ncol=NCOL)
problem = periodic_orbit_problem(
    van_der_pol_rhs,
    jnp.asarray(u_trajectory),
    jnp.asarray(t_trajectory),
    period_guess,
    mu_start,
    mesh,
)
# %%
# Continue from the visible seed with the settings below.

result = jc.continuation(
    problem,
    jc.PseudoArclength(),
    p_span=(mu_start, 4.0),
    settings=jc.ContinuationPar(ds=0.05, max_steps=250, compute_stability=True),
)
if not bool(jnp.all(result.branch.stable)):
    raise RuntimeError("Expected the Van der Pol limit cycle to remain stable throughout")
mus = np.asarray(result.branch.params)
n_valid = result.branch.n_valid
# %%
# Unpack mesh-point coordinates and period for the figure.

states = np.asarray(result.branch.states)
mesh_states = states[:, : NTST * N_STATE].reshape(n_valid, NTST, N_STATE)
periods = states[:, -1]
amplitude = np.max(np.abs(mesh_states[:, :, 0]), axis=1)
# %%
# Plot the computed observables.

fig, axes = plt.subplots(1, 2, figsize=(10, 4))


def _closed_loop(mesh_pts):
    return np.concatenate([mesh_pts, mesh_pts[:1]], axis=0)


first_loop = _closed_loop(mesh_states[0])
last_loop = _closed_loop(mesh_states[-1])
axes[0].plot(first_loop[:, 0], first_loop[:, 1], "o-", label=f"mu={mus[0]:.2f}")
axes[0].plot(last_loop[:, 0], last_loop[:, 1], "o-", label=f"mu={mus[-1]:.2f}")
axes[0].set(xlabel="$x$", ylabel="$y$", title="Limit cycle shape", aspect="equal")
axes[0].grid(alpha=0.25)
axes[0].legend()
axes[1].plot(mus, periods, "o-")
axes[1].set(xlabel="$\\mu$", ylabel="period", title="Period growth")
axes[1].grid(alpha=0.25)
fig.tight_layout()
# %%
# Save this lesson to the explicit output directory.

fig.savefig(options.output_dir / "plot_09_vanderpol_cycle.png", dpi=150, bbox_inches="tight")
