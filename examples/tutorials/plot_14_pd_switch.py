r"""
Switch onto a period-doubled cycle
==================================

Detect a period-doubling (PD) bifurcation on a limit-cycle branch, seed the
period-doubled cycle with :func:`jaxcont.branch_switch_pd`, and continue it.

The periodic lesson uses the 90-line allowance to keep trajectory seeding,
collocation settings, and the plotted observables visible.

With w=w1+i w2, the equations are
:math:`\dot x=(1-r^2)x-y,\ \dot y=(1-r^2)y+x` and
:math:`\dot w=(\alpha-\kappa+i\beta)w+\kappa(x+iy)\bar w-|w|^2w`,
where beta=1/2 and kappa=0.2. The coupling makes the flip simple;
the doubled branch has magnitude sqrt(alpha). Switching is eager and
cannot be traced with jit/vmap; the parameter offset prevents collapse
onto the base cycle.
"""

# %%
# Set up the output directory and imports. Figures stay open for the
# Sphinx-Gallery matplotlib scraper. JaxCont runs in float64 (import jaxcont enables it).

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

T_EXACT = 2 * np.pi
BETA = np.pi / T_EXACT
KAPPA = 0.2


def rhs(u, p, args):
    x, y, w1, w2 = (u[0], u[1], u[2], u[3])
    r2 = x * x + y * y
    m2 = w1 * w1 + w2 * w2
    return jnp.array(
        [
            (1.0 - r2) * x - y,
            (1.0 - r2) * y + x,
            (p - KAPPA) * w1 - BETA * w2 - w1 * m2 + KAPPA * (x * w1 + y * w2),
            BETA * w1 + (p - KAPPA) * w2 - w2 * m2 + KAPPA * (y * w1 - x * w2),
        ]
    )


t = np.linspace(0.0, T_EXACT, 120, endpoint=False)
u_guess = np.stack([np.cos(t), np.sin(t), np.zeros_like(t), np.zeros_like(t)], axis=1)
# %%
# Refine the trajectory on a collocation mesh.

mesh = Collocation(ntst=10, ncol=4)
base = periodic_orbit_problem(rhs, jnp.asarray(u_guess), jnp.asarray(t), T_EXACT, -0.1, mesh)
# %%
# Continue from the visible seed with the settings below.

result = jc.continuation(
    base,
    p_span=(-0.1, 0.1),
    settings=jc.ContinuationPar(ds=0.02, max_steps=40, compute_stability=True),
    events=[jc.PeriodDoubling()],
)
pd_events = [e for e in result.events if e.kind == "period_doubling"]
if not pd_events:
    raise RuntimeError("no period-doubling event detected")
event = pd_events[0]
switched = jc.branch_switch_pd(rhs, event, mesh, dp=0.05)
doubled_mesh = Collocation(ntst=2 * mesh.ntst, ncol=mesh.ncol)
doubled = jc.continuation(
    switched,
    p_span=(float(switched.p0), 0.3),
    settings=jc.ContinuationPar(ds=0.02, max_steps=40, compute_stability=False),
)
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
# %%
# Plot the computed observables.

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(
    np.asarray(result.branch.params),
    np.zeros(result.branch.n_valid),
    "o-",
    color="C0",
    label="base cycle (|w| = 0)",
)
axes[0].plot(alphas, amps, "o-", color="C1", label="period-doubled cycle")
axes[0].axvline(event.p, color="black", ls=":", lw=1, label="PD")
axes[0].set_xlabel("$\\alpha$")
axes[0].set_ylabel("$\\max |w|$")
axes[0].set_title("Period-doubling branch switching")
axes[0].legend()
axes[1].plot(alphas, amps, "o", color="C1", label="JaxCont")
axes[1].plot(
    alphas, np.sqrt(np.clip(alphas, 0, None)), "--", color="black", label="exact $\\sqrt{\\alpha}$"
)
axes[1].set_xlabel("$\\alpha$")
axes[1].set_ylabel("$\\max |w|$")
axes[1].set_title("Doubled-cycle amplitude vs closed form")
axes[1].legend()
plt.tight_layout()
# %%
# Save this lesson to the explicit output directory.

fig.savefig(options.output_dir / "plot_14_pd_switch.png", dpi=150, bbox_inches="tight")
