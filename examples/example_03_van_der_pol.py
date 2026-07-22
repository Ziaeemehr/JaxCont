r"""
Van der Pol equilibrium stability crossing
===========================================

Continue the equilibrium at the origin of the Van der Pol oscillator through
the stability crossing at :math:`\mu=0`.

.. math::

    \dot{x} &= y \\
    \dot{y} &= \mu (1 - x^2) y - x

The origin is stable for :math:`\mu < 0` and unstable for
:math:`\mu > 0`. Its eigenvalues cross the imaginary axis at
:math:`\mu=0`, which equilibrium continuation detects as a Hopf event.
This particular crossing is degenerate: at :math:`\mu=0` the vector field
is the linear harmonic oscillator and MatCont reports a zero first Lyapunov
coefficient. The familiar Van der Pol cycle therefore does not emerge as a
generic, vanishing-amplitude Hopf cycle. Periodic-orbit continuation is
outside JaxCont's current scope.
"""

# %%
# Setup

import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxcont as jc
from jaxcont.viz import EigenvalueReference, plot_eigenvalues

# %%
# Define the system


def van_der_pol_rhs(u, p, args):
    x, y = u
    mu = p
    return jnp.array([y, mu * (1.0 - x**2) * y - x])


# %%
# Set up the problem and run the continuation
# -----------------------------------------------
# The origin is an exact equilibrium for every :math:`\mu`. Starting at
# :math:`\mu=-2` and continuing through :math:`\mu=2` includes both the
# stable and unstable sides of the crossing; starting at zero would hide that
# change.

prob = jc.bif_problem(
    van_der_pol_rhs,
    u0=jnp.array([0.0, 0.0]),
    p0=-2.0,
    state_names=["x", "y"],
    param_name="mu",
)

result = jc.continuation(
    prob,
    jc.PseudoArclength(),
    p_span=(-2.0, 2.0),
    settings=jc.ContinuationPar(
        ds=0.02,
        ds_max=0.05,
        max_steps=160,
        newton_tol=1e-7,
        compute_stability=True,
    ),
    events=[jc.Hopf()],
)
solution = result._solution

print(
    f"Continuation completed: {solution.n_points} points, "
    f"mu in [{float(solution.parameters.min()):.3f}, "
    f"{float(solution.parameters.max()):.3f}]"
)

# %%
# Cross-validation against MatCont 7.6
# ----------------------------------------
# `validation/matcont/run_vanderpol_hopf.m` continues these same equations
# from :math:`\mu=-2` and independently reports one `H` event at
# `(x, y, mu) = (0, 0, 0)`. It also reports first Lyapunov coefficient zero,
# exposing the degeneracy that a linear eigenvalue-crossing detector alone
# cannot classify. The analytic Jacobian at the origin is
# `[[0, 1], [-1, mu]]`, so its critical eigenvalues are exactly `+/- i`.

MATCONT_HOPF_MU = 0.0
MATCONT_FIRST_LYAPUNOV = 0.0

hopf_events = [event for event in result.events if event.kind == "hopf"]
if len(hopf_events) != 1:
    raise RuntimeError(f"Expected one Hopf event, found {len(hopf_events)}")

hopf = hopf_events[0]
max_residual = max(
    float(jnp.linalg.norm(van_der_pol_rhs(u, mu, None), ord=jnp.inf))
    for u, mu in zip(solution.states, solution.parameters)
)

print("\nCross-validation")
print("-" * 58)
print(f"{'Quantity':<27} {'JaxCont':>14} {'MatCont 7.6':>14}")
print("-" * 58)
print(f"{'Hopf parameter mu':<27} {hopf.p:>14.9f} {MATCONT_HOPF_MU:>14.9f}")
print(f"{'|parameter error|':<27} {abs(hopf.p - MATCONT_HOPF_MU):>14.3e}")
print(f"{'max equilibrium residual':<27} {max_residual:>14.3e}")
print(
    f"{'MatCont first Lyapunov':<27} "
    f"{'not computed':>14} {MATCONT_FIRST_LYAPUNOV:>14.1f}"
)
print("\nThe zero Lyapunov coefficient means this is a degenerate Hopf/center,")
print("not a generic local birth of a small-amplitude periodic orbit.")

# %%
# Visualize the stability crossing
# ------------------------------------
# A phase portrait of the *equilibrium branch* would contain only repeated
# copies of `(0, 0)` and is therefore not a time-domain phase portrait. The
# eigenvalue trajectories are the meaningful visualization here: their real
# parts change sign at the event, while their imaginary parts equal `+/- 1`
# at the crossing.

fig = plot_eigenvalues(
    result,
    shade_stability=True,
    references=[
        EigenvalueReference(MATCONT_HOPF_MU, r"MatCont H: $\mu=0$"),
    ],
    param_name=r"$\mu$",
)

fig.suptitle(
    "Van der Pol equilibrium: degenerate Hopf crossing",
    fontsize=15,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
plt.savefig("van_der_pol.png", dpi=150, bbox_inches="tight")
plt.show()
