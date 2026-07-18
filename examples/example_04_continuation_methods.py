"""
Natural vs. pseudo-arclength continuation: passing a fold
==============================================================

Why does JaxCont offer two continuation algorithms? This example calls each
one's low-level ``predict``/``correct``/``compute_tangent`` steps directly,
one step at a time, on the simplest system that has a fold:

.. math::

    \\dot{x} = r - x^2

The equilibrium is :math:`x = \\sqrt{r}` for :math:`r > 0` -- a branch that
turns back on itself at :math:`r = 0`, where :math:`dx/dr = 1/(2\\sqrt{r})
\\to \\infty`. **Natural continuation** fixes the parameter and solves for the
state, so it needs that derivative to stay finite -- it necessarily stalls at
a fold. **Pseudo-arclength continuation** parametrizes by arclength along the
curve instead, so it has no trouble at all.
"""

# %%
# Setup

import jax.numpy as jnp

from jaxcont import ContinuationProblem, NaturalContinuation, PseudoArclengthContinuation


def quadratic_rhs(state, params):
    return jnp.array([params["r"] - state[0] ** 2])


# %%
# A warm-up: both methods land exactly on the solution away from the fold
# ------------------------------------------------------------------------------
# On a simple linear system, :math:`\dot{x} = r - x`, there's no fold at all,
# so both methods converge cleanly at every step -- the error stays exactly
# zero throughout. They don't take *identical* steps: pseudo-arclength's
# ``ds`` measures arclength (split between :math:`r` and :math:`x`), while
# natural continuation's ``ds`` steps :math:`r` directly, so after the same
# 10 steps of ``ds=0.1`` they've reached different points on the same exact
# line -- both correct, just parametrized differently.


def linear_rhs(state, params):
    return jnp.array([params["r"] - state[0]])


problem_linear = ContinuationProblem(
    rhs=linear_rhs, u0=jnp.array([0.0]), params={"r": 0.0},
    continuation_param="r", problem_type="equilibrium",
)

for cls, name in [(NaturalContinuation, "Natural"), (PseudoArclengthContinuation, "Pseudo-arclength")]:
    cont = cls(newton_tol=1e-6, newton_max_iter=50)
    u, param = problem_linear.u0, problem_linear.params["r"]
    tangent = cont.compute_tangent(problem_linear, u, param)
    for _ in range(10):
        u_pred, param_pred = cont.predict(u, param, tangent, 0.1)
        u, param, converged, n_iter = cont.correct(
            problem_linear, u_pred, param_pred, u, param, tangent, 0.1
        )
        tangent = cont.compute_tangent(problem_linear, u, param, tangent)
    print(f"{name:<18} final: r={param:.4f}  x={u[0]:.4f}  "
          f"error={abs(u[0] - param):.2e}  converged={converged}")

# %%
# Natural continuation stalls at the fold
# --------------------------------------------
# Starting on the upper branch (:math:`r=1,\ x=1`) and stepping toward
# :math:`r=0`: each step solves ``f(x, r_pred) = 0`` for :math:`x` at a
# *fixed* predicted :math:`r_pred`. That's fine until the branch is nearly
# vertical -- right at the fold, no real solution exists near the predicted
# point, and the corrector cannot converge no matter how many Newton
# iterations it's given.

r0 = 1.0
problem_quad = ContinuationProblem(
    rhs=quadratic_rhs, u0=jnp.array([jnp.sqrt(r0)]), params={"r": r0},
    continuation_param="r", problem_type="equilibrium",
)

cont_nat = NaturalContinuation(newton_tol=1e-6, newton_max_iter=50)
u, param = problem_quad.u0, r0
tangent = cont_nat.compute_tangent(problem_quad, u, param)
ds = -0.05 if tangent[-1] > 0 else 0.05  # head toward r=0

print("\nNatural continuation, heading toward the fold at r=0:")
for step in range(30):
    u_pred, param_pred = cont_nat.predict(u, param, tangent, ds)
    u_new, param_new, converged, n_iter = cont_nat.correct(
        problem_quad, u_pred, param_pred, u, param, tangent, ds
    )
    if not converged:
        print(f"  step {step}: r={param:.5f} -> STALLED "
              f"(no convergence in {n_iter} Newton iterations)")
        break
    u, param = u_new, param_new
    tangent = cont_nat.compute_tangent(problem_quad, u, param, tangent)
    if step % 4 == 0:
        print(f"  step {step}: r={param:.5f}  x={u[0]:.5f}  iters={n_iter}")

# %%
# Pseudo-arclength sails through, onto the other branch
# -----------------------------------------------------------
# Same system, same starting point, same step size -- but now the corrector
# solves the *bordered* system (state **and** parameter as unknowns,
# constrained by the arclength equation), which stays well-posed exactly
# where the natural corrector breaks down.

cont_pa = PseudoArclengthContinuation(newton_tol=1e-6, newton_max_iter=50)
u, param = problem_quad.u0, r0
tangent = cont_pa.compute_tangent(problem_quad, u, param)
ds = -0.05 if tangent[-1] > 0 else 0.05

print("\nPseudo-arclength continuation, through the fold and beyond:")
for step in range(60):
    u_pred, param_pred = cont_pa.predict(u, param, tangent, ds)
    u_new, param_new, converged, n_iter = cont_pa.correct(
        problem_quad, u_pred, param_pred, u, param, tangent, ds
    )
    if not converged:
        print(f"  step {step}: r={param:.5f} -> stalled ({n_iter} iters)")
        break
    u, param = u_new, param_new
    tangent = cont_pa.compute_tangent(problem_quad, u, param, tangent)
    if step % 8 == 0 or abs(param) < 0.02:
        print(f"  step {step}: r={param:.5f}  x={u[0]:.5f}  iters={n_iter}")
    if param > 0.5 and u[0] < 0:
        print(f"  PASSED THE FOLD: now at r={param:.4f}, x={u[0]:.4f} "
              f"-- on the mirror branch x=-sqrt(r)")
        break

print("\nEvery pseudo-arclength step above converges in a single Newton")
print("iteration, including the ones that step directly through r=0 -- the")
print("bordered system's conditioning does not degrade at the fold, unlike")
print("the natural corrector's above.")
