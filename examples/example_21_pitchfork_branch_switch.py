r"""
Switch branches at an equilibrium pitchfork
============================================

Detect the branch point (BP) on the trivial branch of the pitchfork normal
form, seed the crossing branch with :func:`jaxcont.branch_switch`, and
continue both branches for a combined bifurcation diagram.

The pitchfork :math:`f(u,p)=pu-u^3` has a trivial branch :math:`u=0` for
every :math:`p`, and two nontrivial branches :math:`u=\pm\sqrt{p}` that
cross it at :math:`p=0`. Continuing along the trivial branch, :math:`p`
increases straight through zero -- the branch itself never turns around, so
an ordinary fold detector would never fire here. What makes :math:`p=0`
special is a stronger condition: the augmented Jacobian
:math:`[f_u\,|\,f_p]` drops rank there too, not just :math:`f_u` alone. That
combined condition is what :class:`jaxcont.BP` tests for.
"""

# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxcont as jc
from jaxcont.viz import plot_continuation


def pitchfork(u, p, args):
    return jnp.array([p * u[0] - u[0] ** 3])


# %%
# Continue the trivial branch and detect the branch point
# ---------------------------------------------------------
#
# ``events=[jc.BP()]`` runs a cheap per-step test -- the same
# real-eigenvalue-crossing signal an ordinary fold gives -- and only pays for
# the expensive augmented-Jacobian rank check at a candidate crossing. So
# this costs no more than fold detection along a branch that never actually
# folds.

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

# %%
# Switch onto the crossing branch
# ----------------------------------
#
# ``branch_switch`` computes the crossing branch's tangent via SVD of the
# same augmented Jacobian ``BP`` used to classify the point, takes one
# predictor step of size ``ds`` along it, and Newton-corrects back onto
# ``f=0`` -- returning an ordinary seeded :class:`jaxcont.BifProblem` ready
# for its own ``continuation()`` call.

switched_problem = jc.branch_switch(pitchfork, bp_event, ds=0.05)
switched_result = jc.continuation(
    switched_problem,
    jc.PseudoArclength(),
    p_span=(float(switched_problem.p0), 0.5),
    settings=jc.ContinuationPar(
        ds=0.02,
        max_steps=40,
        newton_tol=1e-5,
        compute_stability=True,
    ),
)

print(
    f"BP at p={bp_event.p:.6f}, u={float(bp_event.u[0]):.6f}; "
    f"switched seed p={float(switched_problem.p0):.6f}, "
    f"u={float(switched_problem.u0[0]):.6f}"
)

# %%
# Plot both branches
# --------------------

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
plt.show()
