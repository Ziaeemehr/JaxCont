r"""
Switch onto a period-doubled cycle
====================================

Detect a period-doubling (PD) bifurcation on a limit-cycle branch, seed the
period-doubled cycle with :func:`jaxcont.branch_switch_pd`, and continue it.
"""

# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxcont as jc
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

T_EXACT = 2 * np.pi
BETA = np.pi / T_EXACT
KAPPA = 0.2


# %%
# Define the system
# --------------------
#
# A circle oscillator drives a transverse block that saturates cubically:
#
# .. math::
#
#     \dot{x} &= (1-x^2-y^2)x - y, &
#     \dot{y} &= (1-x^2-y^2)y + x, \\
#     \dot{w} &= (\alpha-\kappa+i\beta)w + \kappa(x+iy)\bar{w} - |w|^2 w,
#     \qquad \beta = \pi/T, \quad \kappa = 0.2.
#
# The linear part alone (``example_08_period_doubling.py``'s system)
# already rotates :math:`w` by exactly :math:`\pi` per period, giving
# :math:`w(T)=-w(0)`: a period-doubling at :math:`\alpha=0`. That fixture is
# enough to *detect* a PD crossing, but :math:`w=0` solves it for every
# :math:`\alpha` -- there is no actual doubled branch to switch onto. The
# cubic term breaks that: for :math:`\alpha>0` a genuine period-doubled
# cycle exists in closed form, :math:`|w|=\sqrt{\alpha}` with period
# :math:`2T`, which is what makes this system useful for testing
# ``branch_switch_pd`` rather than just ``PeriodDoubling``.
#
# The :math:`\kappa` coupling breaks the rotation symmetry
# :math:`w\mapsto e^{i\phi}w`. Without it every rotated copy of the doubled
# cycle is also a solution, so the cycle is not isolated and Newton cannot
# converge to it tightly. With it the :math:`-1` multiplier at
# :math:`\alpha=0` is simple (the other transverse multiplier reaches
# :math:`-1` only at :math:`\alpha=2\kappa`), and the doubled cycle is still
# :math:`w=\sqrt{\alpha}\,e^{i\beta t}`.


def rhs(u, p, args):
    x, y, w1, w2 = u[0], u[1], u[2], u[3]
    r2 = x * x + y * y
    m2 = w1 * w1 + w2 * w2
    return jnp.array([
        (1.0 - r2) * x - y,
        (1.0 - r2) * y + x,
        (p - KAPPA) * w1 - BETA * w2 - w1 * m2 + KAPPA * (x * w1 + y * w2),
        BETA * w1 + (p - KAPPA) * w2 - w2 * m2 + KAPPA * (y * w1 - x * w2),
    ])


# %%
# Continue the base cycle and detect the flip
# ----------------------------------------------
#
# ``jc.PeriodDoubling()`` reads the raw right-hand side, mesh, and user
# arguments from the periodic problem's ``PeriodicArgs`` when it recomputes
# Floquet multipliers at corrected bisection midpoints.

t = np.linspace(0.0, T_EXACT, 120, endpoint=False)
u_guess = np.stack(
    [np.cos(t), np.sin(t), np.zeros_like(t), np.zeros_like(t)], axis=1
)
mesh = Collocation(ntst=10, ncol=4)
base = periodic_orbit_problem(
    rhs, jnp.asarray(u_guess), jnp.asarray(t), T_EXACT, -0.1, mesh
)

result = jc.continuation(
    base,
    p_span=(-0.1, 0.1),
    settings=jc.ContinuationPar(
        ds=0.02, max_steps=40, compute_stability=True
    ),
    events=[jc.PeriodDoubling()],
)

pd_events = [e for e in result.events if e.kind == "period_doubling"]
if not pd_events:
    raise RuntimeError("no period-doubling event detected")
event = pd_events[0]
print(f"PD detected at alpha = {event.p:.6f} (exact: 0)")

# %%
# Seed and continue the doubled-period cycle
# ----------------------------------------------
#
# Correcting exactly *at* the bifurcation parameter collapses to the
# trivial branch -- the doubled cycle has zero amplitude there, so any seed
# just Newton-converges back to :math:`w=0`. ``branch_switch_pd`` instead
# steps the parameter to :math:`\alpha=p_{PD}+dp` first (``dp=0.05`` here),
# then verifies the result is genuinely period-doubled and not the original
# orbit traversed twice -- a failure mode that converges just as cleanly
# and would otherwise be silently wrong. If the first seed amplitude
# collapses, it retries a geometric ladder of amplitudes and both possible
# signs of ``dp`` (the doubled branch can exist on either side of the
# bifurcation, and that is not knowable in advance) before raising.
#
# This whole search runs eager: it needs concrete residual values to
# branch on and to raise, so it is not ``vmap``/``jit``-traceable, even
# though ``periodic_orbit_problem`` construction itself now is (see
# ``example_23_periodic_vmap_grad.py``).

switched = jc.branch_switch_pd(rhs, event, mesh, dp=0.05)
doubled_mesh = Collocation(ntst=2 * mesh.ntst, ncol=mesh.ncol)
print(
    f"switched seed: alpha = {float(switched.p0):.6f}, "
    f"period = {float(switched.u0[-1]):.6f} (exact: {2 * T_EXACT:.6f})"
)

doubled = jc.continuation(
    switched,
    p_span=(float(switched.p0), 0.3),
    settings=jc.ContinuationPar(
        ds=0.02, max_steps=40, compute_stability=False
    ),
)

# %%
# Plot the combined diagram
# -----------------------------
#
# The doubled branch starts a little to the right of the PD marker rather
# than touching it -- that gap is the ``dp`` step described above, not a
# failure to connect.

n_state = 4
ntst_d = doubled_mesh.ntst
amps = [
    float(
        np.max(
            np.sqrt(
                np.asarray(U[: ntst_d * n_state].reshape(ntst_d, n_state))[:, 2] ** 2
                + np.asarray(U[: ntst_d * n_state].reshape(ntst_d, n_state))[:, 3] ** 2
            )
        )
    )
    for U in np.asarray(doubled.branch.states)
]
alphas = np.asarray(doubled.branch.params)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(np.asarray(result.branch.params), np.zeros(result.branch.n_valid),
             "o-", color="C0", label="base cycle (|w| = 0)")
axes[0].plot(alphas, amps, "o-", color="C1", label="period-doubled cycle")
axes[0].axvline(event.p, color="black", ls=":", lw=1, label="PD")
axes[0].set_xlabel(r"$\alpha$")
axes[0].set_ylabel(r"$\max |w|$")
axes[0].set_title("Period-doubling branch switching")
axes[0].legend()

axes[1].plot(alphas, amps, "o", color="C1", label="JaxCont")
axes[1].plot(alphas, np.sqrt(np.clip(alphas, 0, None)), "--", color="black",
             label=r"exact $\sqrt{\alpha}$")
axes[1].set_xlabel(r"$\alpha$")
axes[1].set_ylabel(r"$\max |w|$")
axes[1].set_title("Doubled-cycle amplitude vs closed form")
axes[1].legend()

plt.tight_layout()
plt.savefig("images/example_22_period_doubling_switch.png", dpi=150,
            bbox_inches="tight")
plt.show()
