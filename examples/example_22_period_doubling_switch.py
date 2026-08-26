"""
Switch onto a period-doubled cycle
==================================

Detect a period-doubling (PD) bifurcation on a limit-cycle branch, seed the
period-doubled cycle with :func:`jaxcont.branch_switch_pd`, and continue it.

The system is a circle oscillator with a cubically-saturated transverse block::

    x' = (1 - r^2) x - y            r^2 = x^2 + y^2
    y' = (1 - r^2) y + x            -> circle, period T = 2*pi
    w' = (alpha + i*beta) w - |w|^2 w    w = w1 + i*w2, beta = pi/T

The linear part rotates ``w`` by exactly ``pi`` per period, so ``w(T) = -w(0)``:
a period-doubling at ``alpha = 0``. For ``alpha > 0`` the doubled cycle has the
closed form ``|w| = sqrt(alpha)`` with period ``2T``, which is what the right
panel checks against.

The doubled branch in the left panel starts a little to the right of the PD
marker rather than touching it. That gap is expected, not a failure to connect:
the doubled cycle has zero amplitude exactly at the bifurcation, so
``branch_switch_pd`` seeds it at ``alpha = p_PD + dp`` (here ``dp = 0.05``) and
continues from there. Correcting at the bifurcation parameter itself would just
collapse back onto the base cycle.

Note: period-doubling switching is eager-only -- it builds a new
periodic-orbit problem, and that construction is not safe under an outer
``jax.vmap``/``jax.grad``.
"""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxcont as jc
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

T_EXACT = 2 * np.pi
BETA = np.pi / T_EXACT


def rhs(u, p, args):
    x, y, w1, w2 = u[0], u[1], u[2], u[3]
    r2 = x * x + y * y
    m2 = w1 * w1 + w2 * w2
    return jnp.array([
        (1.0 - r2) * x - y,
        (1.0 - r2) * y + x,
        p * w1 - BETA * w2 - w1 * m2,
        BETA * w1 + p * w2 - w2 * m2,
    ])


def main():
    os.makedirs("images", exist_ok=True)

    t = np.linspace(0.0, T_EXACT, 120, endpoint=False)
    u_guess = np.stack(
        [np.cos(t), np.sin(t), np.zeros_like(t), np.zeros_like(t)], axis=1
    )
    mesh = Collocation(ntst=10, ncol=4)
    base = periodic_orbit_problem(
        rhs, jnp.asarray(u_guess), jnp.asarray(t), T_EXACT, -0.1, mesh
    )

    # newton_tol=1e-5, not the 1e-6 default: the collocation residual's float32
    # floor is ~3.4e-6, and a tighter tolerance stalls continuation.
    result = jc.continuation(
        base,
        p_span=(-0.1, 0.1),
        settings=jc.ContinuationPar(
            ds=0.02, max_steps=40, newton_tol=1e-5, compute_stability=True
        ),
        events=[jc.PeriodDoubling(raw_f=rhs, mesh=mesh)],
    )

    pd_events = [e for e in result.events if e.kind == "period_doubling"]
    if not pd_events:
        raise RuntimeError("no period-doubling event detected")
    event = pd_events[0]
    print(f"PD detected at alpha = {event.p:.6f} (exact: 0)")

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
            ds=0.02, max_steps=40, newton_tol=1e-5, compute_stability=False
        ),
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
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
