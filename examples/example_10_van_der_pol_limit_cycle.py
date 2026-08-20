r"""
Van der Pol limit cycle
========================

Continue the actual Van der Pol limit cycle -- not just the equilibrium's
degenerate Hopf crossing at :math:`\mu=0` shown in
``example_03_van_der_pol.py`` -- via periodic-orbit collocation.

.. math::

    \dot{x} &= y \\
    \dot{y} &= \mu (1 - x^2) y - x

As explained in ``example_03``, the cycle does not emerge as a
vanishing-amplitude Hopf cycle at :math:`\mu=0` (the crossing is degenerate:
zero first Lyapunov coefficient). Starting instead from a short simulation
at :math:`\mu=1` -- comfortably past that degenerate point -- and refining
it into an exact periodic orbit gives a real starting point to continue
into the classic large-:math:`\mu` relaxation-oscillator regime.
"""

# %%
# Setup
# -----
# JaxCont does not integrate ODEs itself: the initial trajectory guess below
# comes from the user's own simulation (``scipy.integrate.solve_ivp``), and
# ``periodic_orbit_problem`` only resamples and refines it.

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

import jaxcont as jc
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

NTST, NCOL, N_STATE = 20, 4, 2


# %%
# Define the system
# ------------------
# Same right-hand side as ``example_03_van_der_pol.py``.


def van_der_pol_rhs(u, p, args):
    x, y = u
    mu = p
    return jnp.array([y, mu * (1.0 - x**2) * y - x])


def _van_der_pol_rhs_np(t, u, mu):
    x, y = u
    return [y, mu * (1.0 - x**2) * y - x]


# %%
# Simulate to find an initial trajectory guess
# ---------------------------------------------
# Simulate at :math:`\mu=1` until the transient dies out, then locate one
# full period in the tail via successive peaks in :math:`x(t)`.
# ``t_trajectory`` must start at 0 -- ``periodic_orbit_problem`` resamples
# using ``t = tau * period0`` for ``tau`` in ``[0, 1]``, so an un-rebased
# time array (still carrying the simulation's raw offset) falls entirely
# outside ``jnp.interp``'s domain and silently clamps to a constant,
# producing a degenerate near-zero-period "solution" with deceptively small
# residual rather than an error -- verified during design.

mu_start = 1.0
raw_solution = solve_ivp(
    _van_der_pol_rhs_np, [0.0, 60.0], [0.5, 0.0], args=(mu_start,),
    max_step=0.01, dense_output=True,
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
# Refine into an exact periodic orbit
# -------------------------------------

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
# Continue the limit cycle
# --------------------------
# ``compute_stability=True`` stores Floquet multipliers so ``Branch.stable``
# is available to confirm the cycle remains stable throughout.

result = jc.continuation(
    problem,
    jc.PseudoArclength(),
    p_span=(mu_start, 4.0),
    settings=jc.ContinuationPar(
        ds=0.05,
        max_steps=250,
        newton_tol=1e-5,
        compute_stability=True,
    ),
)

if not bool(jnp.all(result.branch.stable)):
    raise RuntimeError("Expected the Van der Pol limit cycle to remain stable throughout")

mus = np.asarray(result.branch.params)
n_valid = result.branch.n_valid
print(f"Continued mu in [{mus[0]:.3f}, {mus[-1]:.3f}] over {n_valid} points, stable throughout.")

# %%
# Extract mesh-point states for plotting
# -----------------------------------------
# ``Branch.states`` packs each point's full collocation unknown vector:
# ``ntst`` mesh-point states first, then collocation-point states, then the
# period. The first ``ntst*N_STATE`` entries reshape into the mesh-point
# states -- dense enough (``ntst=20``) for a clean phase-portrait polygon.

states = np.asarray(result.branch.states)
mesh_states = states[:, : NTST * N_STATE].reshape(n_valid, NTST, N_STATE)
periods = states[:, -1]
amplitude = np.max(np.abs(mesh_states[:, :, 0]), axis=1)
print(f"Amplitude range: [{amplitude.min():.4f}, {amplitude.max():.4f}] (expected ~flat, near 2)")
print(f"Period range: [{periods.min():.4f}, {periods.max():.4f}] (expected to grow with mu)")

# %%
# Plot the cycle's shape change and period growth
# ---------------------------------------------------
# Amplitude barely changes for Van der Pol at this normalization -- period
# growth and waveform sharpening are the salient relaxation-oscillator
# signatures, not amplitude growth (contrast with the Brusselator example).

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
axes[1].set(xlabel=r"$\mu$", ylabel="period", title="Period growth")
axes[1].grid(alpha=0.25)

fig.tight_layout()
plt.savefig("images/example_10_van_der_pol_limit_cycle.png")
plt.show()
