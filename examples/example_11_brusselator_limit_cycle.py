r"""
Brusselator limit cycle
=========================

Continue the Brusselator's limit cycle via periodic-orbit collocation.

.. math::

    \dot{x} &= a - (b+1)x + x^2 y \\
    \dot{y} &= bx - x^2 y

With :math:`a=1` fixed, the origin-shifted fixed point
:math:`(x,y)=(a, b/a)` undergoes a Hopf bifurcation at :math:`b=1+a^2=2`.
Starting a short simulation at :math:`b=2.5` (past the Hopf point) and
refining it into an exact periodic orbit gives a starting point to continue
the cycle as :math:`b` grows.
"""

# %%
# Setup

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

import jaxcont as jc
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

NTST, NCOL, N_STATE = 20, 4, 2
A = 1.0


# %%
# Define the system


def brusselator_rhs(u, p, args):
    x, y = u
    b = p
    return jnp.array([A - (b + 1.0) * x + x**2 * y, b * x - x**2 * y])


def _brusselator_rhs_np(t, u, b):
    x, y = u
    return [A - (b + 1.0) * x + x**2 * y, b * x - x**2 * y]


# %%
# Simulate to find an initial trajectory guess
# ---------------------------------------------
# Same rationale as the Van der Pol example: ``t_trajectory`` must start at
# 0 before ``periodic_orbit_problem`` resamples it (see that example's
# comment, or the design spec, for what silently goes wrong otherwise).

b_start = 2.5
raw_solution = solve_ivp(
    _brusselator_rhs_np, [0.0, 80.0], [1.1, 1.1], args=(b_start,),
    max_step=0.01, dense_output=True,
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
# Refine into an exact periodic orbit

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
# Continue the limit cycle
# --------------------------
# ``newton_tol=1e-4``, not Van der Pol's ``1e-5``: verified during design
# that this system's achievable float32 residual floor at this mesh size is
# looser than Van der Pol's -- ``1e-5`` here stalls continuation (adaptive
# step size collapses toward ``ds_min``, the branch barely advances) even
# though a fresh, independent refinement at a nearby parameter value
# converges cleanly. Do not assume one system's verified tolerance transfers
# to another.

result = jc.continuation(
    problem,
    jc.PseudoArclength(),
    p_span=(b_start, 4.0),
    settings=jc.ContinuationPar(
        ds=0.05,
        max_steps=350,
        newton_tol=1e-4,
        compute_stability=True,
    ),
)

if not bool(jnp.all(result.branch.stable)):
    raise RuntimeError("Expected the Brusselator limit cycle to remain stable throughout")

bs = np.asarray(result.branch.params)
n_valid = result.branch.n_valid
print(f"Continued b in [{bs[0]:.3f}, {bs[-1]:.3f}] over {n_valid} points, stable throughout.")

# %%
# Extract mesh-point states for plotting

states = np.asarray(result.branch.states)
mesh_states = states[:, : NTST * N_STATE].reshape(n_valid, NTST, N_STATE)
periods = states[:, -1]
amplitude = np.max(mesh_states[:, :, 0], axis=1) - np.min(mesh_states[:, :, 0], axis=1)
print(f"Amplitude range: [{amplitude.min():.4f}, {amplitude.max():.4f}] (expected to grow with b)")
print(f"Period range: [{periods.min():.4f}, {periods.max():.4f}]")

# %%
# Plot the cycle's growth
# --------------------------
# Unlike Van der Pol (near-constant amplitude, growing period), the
# Brusselator's amplitude grows substantially with ``b`` -- the salient
# feature to plot here.

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
plt.savefig("images/example_11_brusselator_limit_cycle.png")
plt.show()
