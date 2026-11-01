r"""
Brusselator limit cycle
=======================

Continue the Brusselator's limit cycle via periodic-orbit collocation.

.. math::

    \dot{x} &= a - (b+1)x + x^2 y \\
    \dot{y} &= bx - x^2 y

With :math:`a=1` fixed, the origin-shifted fixed point
:math:`(x,y)=(a, b/a)` undergoes a Hopf bifurcation at :math:`b=1+a^2=2`.
Starting a short simulation at :math:`b=2.5` (past the Hopf point) and
refining it into an exact periodic orbit gives a starting point to continue
the cycle as :math:`b` grows.

Inspect the computed domain: a stalled corrector can stop before b=3.

The periodic lesson uses the 90-line allowance to keep trajectory seeding,
collocation settings, and the plotted observables visible.
"""

# %%
# Set up the output directory and imports. Figures stay open for the
# Sphinx-Gallery matplotlib scraper. JaxCont runs in float64 (import jaxcont enables it).

import argparse
import tempfile
from pathlib import Path
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
from jaxcont.utils.precision import default_tol
from jaxcont.problems.periodic import periodic_orbit_problem

# The refined mesh resolves the Floquet flow mode.
NTST, NCOL, N_STATE = 40, 6, 2
A = 1.0


def brusselator_rhs(u, p, args):
    x, y = u
    b = p
    return jnp.array([A - (b + 1.0) * x + x**2 * y, b * x - x**2 * y])


def _brusselator_rhs_np(t, u, b):
    x, y = u
    return [A - (b + 1.0) * x + x**2 * y, b * x - x**2 * y]


b_start = 2.5
raw_solution = solve_ivp(
    _brusselator_rhs_np, [0.0, 80.0], [1.1, 1.1], args=(b_start,), max_step=0.01, dense_output=True
)
tail_times = np.linspace(60.0, 80.0, 6000)
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
    brusselator_rhs,
    jnp.asarray(u_trajectory),
    jnp.asarray(t_trajectory),
    period_guess,
    b_start,
    mesh,
)
# %%
# Continue from the visible seed with the settings below.

result = jc.continuation(
    problem,
    jc.PseudoArclength(),
    p_span=(b_start, 3.0),
    settings=jc.ContinuationPar(
        ds=0.05, max_steps=350, newton_tol=default_tol(problem.u0.dtype), compute_stability=True
    ),
)
if not bool(jnp.all(result.branch.stable)):
    raise RuntimeError("Expected the Brusselator limit cycle to remain stable throughout")
bs = np.asarray(result.branch.params)
n_valid = result.branch.n_valid
# %%
# Unpack mesh-point coordinates and period for the figure.

states = np.asarray(result.branch.states)
mesh_states = states[:, : NTST * N_STATE].reshape(n_valid, NTST, N_STATE)
periods = states[:, -1]
amplitude = np.max(mesh_states[:, :, 0], axis=1) - np.min(mesh_states[:, :, 0], axis=1)
# %%
# Plot the computed observables.

fig, axes = plt.subplots(1, 2, figsize=(10, 4))


def _closed_loop(mesh_pts):
    return np.concatenate([mesh_pts, mesh_pts[:1]], axis=0)


first_loop = _closed_loop(mesh_states[0])
last_loop = _closed_loop(mesh_states[-1])
axes[0].plot(first_loop[:, 0], first_loop[:, 1], "o-", label=f"b={bs[0]:.2f}")
axes[0].plot(last_loop[:, 0], last_loop[:, 1], "o-", label=f"b={bs[-1]:.2f}")
axes[0].set(xlabel="$x$", ylabel="$y$", title="Limit cycle shape")
axes[0].grid(alpha=0.25)
axes[0].legend()
axes[1].plot(bs, amplitude, "o-")
axes[1].set(xlabel="$b$", ylabel="amplitude (max$-$min of $x$)", title="Amplitude growth")
axes[1].grid(alpha=0.25)
fig.tight_layout()
# %%
# Save this lesson to the explicit output directory.

fig.savefig(options.output_dir / "plot_10_brusselator_cycle.png", dpi=150, bbox_inches="tight")
