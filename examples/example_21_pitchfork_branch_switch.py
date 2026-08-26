"""
Switch branches at an equilibrium pitchfork
================================================

Detect the branch point (BP) on the trivial branch of the pitchfork normal
form, seed the crossing branch with :func:`jaxcont.branch_switch`, and
continue both branches for a combined bifurcation diagram.
"""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxcont as jc
from jaxcont.viz import plot_continuation


def pitchfork(u, p, args):
    return jnp.array([p * u[0] - u[0] ** 3])


def main():
    os.makedirs("images", exist_ok=True)

    trivial_problem = jc.bif_problem(
        pitchfork,
        u0=jnp.array([0.0]),
        p0=-0.5,
        state_names=["u"],
        param_name="p",
    )
    trivial_result = jc.continuation(
        trivial_problem,
        jc.PseudoArclength(),
        p_span=(-0.5, 0.5),
        settings=jc.ContinuationPar(
            ds=0.02,
            max_steps=60,
            newton_tol=1e-6,
            compute_stability=True,
        ),
        events=[jc.BP()],
    )

    bp_event = [
        event for event in trivial_result.events
        if event.kind == "branch_point"
    ][0]
    switched_problem = jc.branch_switch(pitchfork, bp_event, ds=0.05)
    switched_result = jc.continuation(
        switched_problem,
        jc.PseudoArclength(),
        p_span=(float(switched_problem.p0), 0.5),
        settings=jc.ContinuationPar(
            ds=0.02,
            max_steps=40,
            newton_tol=1e-6,
            compute_stability=True,
        ),
    )

    print(
        f"BP at p={bp_event.p:.6f}, u={float(bp_event.u[0]):.6f}; "
        f"switched seed p={float(switched_problem.p0):.6f}, "
        f"u={float(switched_problem.u0[0]):.6f}"
    )

    fig = plot_continuation(
        trivial_result._solution,
        title="Pitchfork branch switching",
        legend=False,
    )
    plot_continuation(
        switched_result._solution,
        ax=fig.axes[0],
        show_bifurcations=False,
        stable_color="#009E73",
        unstable_color="#009E73",
        title="Pitchfork branch switching",
        legend=False,
    )
    fig.axes[0].legend(["trivial branch", "switched branch"])
    plt.savefig(
        "images/example_21_pitchfork_branch_switch.png",
        dpi=150,
        bbox_inches="tight",
    )
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
