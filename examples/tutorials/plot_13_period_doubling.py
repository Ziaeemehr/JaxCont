r"""
Period-doubling of a periodic orbit
===================================

Detect a period-doubling (flip) bifurcation by following the Floquet
multipliers of a periodic orbit.  We use a four-dimensional system with an
exact circular orbit,

.. math::

    \dot{x} &= (1-x^2-y^2)x-y, &
    \dot{y} &= (1-x^2-y^2)y+x, \\
    \dot{w}_1 &= \alpha w_1-\beta w_2, &
    \dot{w}_2 &= \beta w_1+\alpha w_2 .

The orbit :math:`(x,y,w_1,w_2)=(\cos t,\sin t,0,0)` has period
:math:`T=2\pi` for every continuation parameter :math:`\alpha`.  Choosing
:math:`\beta=\pi/T` gives the transverse Floquet multipliers
:math:`-\exp(\alpha T)`.  They therefore cross :math:`-1` exactly at
:math:`\alpha=0`, providing a closed-form check of
:class:`jaxcont.PeriodDoubling`.

This is an analytic control with a linear transverse block; it tests
detection, not a nonlinear branch or torus emerging from the crossing.

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
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

T_EXACT = 2.0 * np.pi
BETA = np.pi / T_EXACT


def period_doubling_rhs(u, alpha, args):
    x, y, w1, w2 = u
    radius_squared = x**2 + y**2
    return jnp.array(
        [
            (1.0 - radius_squared) * x - y,
            (1.0 - radius_squared) * y + x,
            alpha * w1 - BETA * w2,
            BETA * w1 + alpha * w2,
        ]
    )


alpha_start = -0.05
t_guess = np.linspace(0.0, T_EXACT, 60, endpoint=False)
trajectory_guess = np.column_stack(
    [np.cos(t_guess), np.sin(t_guess), np.zeros_like(t_guess), np.zeros_like(t_guess)]
)
# %%
# Refine the trajectory on a collocation mesh.

mesh = Collocation(ntst=10, ncol=4)
problem = periodic_orbit_problem(
    period_doubling_rhs,
    jnp.asarray(trajectory_guess),
    jnp.asarray(t_guess),
    T_EXACT,
    alpha_start,
    mesh,
)
# %%
# Continue from the visible seed with the settings below.

result = jc.continuation(
    problem,
    jc.PseudoArclength(),
    p_span=(alpha_start, 0.05),
    settings=jc.ContinuationPar(ds=0.02, max_steps=50, compute_stability=True),
    events=[jc.PeriodDoubling()],
)
events = [event for event in result.events if event.kind == "period_doubling"]
if len(events) != 1:
    raise RuntimeError(f"Expected one period-doubling event, found {len(events)}")
event = events[0]
alphas = np.asarray(result.branch.params)
all_multipliers = np.asarray(result.branch.eigenvalues)
transverse = []
for multipliers in all_multipliers:
    trivial = np.argmin(np.abs(multipliers - 1.0))
    candidates = np.delete(multipliers, trivial)
    real_candidates = candidates[np.abs(candidates.imag) < 1e-05]
    negative_candidates = real_candidates[real_candidates.real < 0.0]
    transverse.append(negative_candidates[0])
transverse = np.asarray(transverse)
exact = -np.exp(alphas * T_EXACT)
max_error = np.max(np.abs(transverse.real - exact))
# %%
# Plot the computed observables.

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(trajectory_guess[:, 0], trajectory_guess[:, 1], color="C0")
axes[0].set(xlabel="$x$", ylabel="$y$", title="Periodic orbit", aspect="equal")
axes[0].grid(alpha=0.25)
axes[1].plot(alphas, transverse.real, "o-", label="JaxCont")
axes[1].plot(alphas, exact, "--", color="black", label="$-\\exp(\\alpha T)$")
axes[1].axhline(-1.0, color="0.5", linewidth=1)
axes[1].axvline(event.p, color="C3", linestyle=":", label="detected PD")
axes[1].set(xlabel="$\\alpha$", ylabel="real Floquet multiplier", title="Crossing through $-1$")
axes[1].grid(alpha=0.25)
axes[1].legend()
fig.tight_layout()
# %%
# Save this lesson to the explicit output directory.

fig.savefig(options.output_dir / "plot_13_period_doubling.png", dpi=150, bbox_inches="tight")
