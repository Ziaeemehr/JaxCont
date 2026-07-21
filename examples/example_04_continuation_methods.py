"""
Natural vs. pseudo-arclength continuation: passing a fold
==============================================================

Why does JaxCont offer two continuation algorithms? Run each one, via
``jc.continuation()``, on the simplest system that has a fold:

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

import jaxcont as jc


def quadratic_rhs(u, p, args):
    return jnp.array([p - u[0] ** 2])


# %%
# A warm-up: both methods land exactly on the solution away from the fold
# ------------------------------------------------------------------------------
# On a simple linear system, :math:`\\dot{x} = r - x`, there's no fold at all,
# so both methods converge cleanly.


def linear_rhs(u, p, args):
    return jnp.array([p - u[0]])


prob_linear = jc.bif_problem(linear_rhs, u0=jnp.array([0.0]), p0=0.0)

for alg, name in [(jc.Natural(), "Natural"), (jc.PseudoArclength(), "Pseudo-arclength")]:
    result = jc.continuation(
        prob_linear, alg, p_span=(0.0, 1.0),
        settings=jc.ContinuationPar(ds=0.1, max_steps=10, newton_tol=1e-6, newton_max_iter=50),
    )
    n = result.branch.n_valid
    final_u = float(result.branch.states[n - 1, 0])
    final_p = float(result.branch.params[n - 1])
    print(f"{name:<18} final: r={final_p:.4f}  x={final_u:.4f}  "
          f"error={abs(final_u - final_p):.2e}  n_points={n}")

# %%
# Natural continuation stalls at the fold
# --------------------------------------------
# Starting on the upper branch (:math:`r=1,\\ x=1`) and continuing toward
# :math:`r=0`: each step solves ``f(x, r_pred) = 0`` for :math:`x` at a
# *fixed* predicted :math:`r_pred`. That's fine until the branch is nearly
# vertical -- right at the fold, no real solution exists near the predicted
# point, and the corrector cannot converge no matter how many Newton
# iterations it's given, so the branch simply stops short of r=0.

prob_quad = jc.bif_problem(quadratic_rhs, u0=jnp.array([1.0]), p0=1.0)

# NOTE: p_span[0] must equal the problem's actual p0 (1.0) -- jc.continuation()
# starts the scan AT p_span[0] using u0 directly (not at problem.p0), a
# pre-existing api.py behavior discovered while migrating example_02 (Task 10).
# u0=[1.0] is only a valid equilibrium at p=1.0, so p_span must start there.
result_nat = jc.continuation(
    prob_quad, jc.Natural(), p_span=(1.0, -1.0),
    settings=jc.ContinuationPar(ds=0.05, max_steps=30, newton_tol=1e-6, newton_max_iter=50),
)
n_nat = result_nat.branch.n_valid
print(f"\nNatural continuation, heading toward the fold at r=0:")
print(f"  reached r = {float(result_nat.branch.params[n_nat - 1]):.5f} "
      f"in {n_nat} points (started at r=1, target r=-1)")
print("  (stalled before reaching r=0 -- no real solution exists past the fold "
      "at a fixed predicted r)")

# %%
# Pseudo-arclength sails through, onto the other branch
# -----------------------------------------------------------
# Same system, same starting point, same step size -- but now the corrector
# solves the *bordered* system (state **and** parameter as unknowns,
# constrained by the arclength equation), which stays well-posed exactly
# where the natural corrector breaks down.

result_pa = jc.continuation(
    prob_quad, jc.PseudoArclength(), p_span=(1.0, -1.0),
    settings=jc.ContinuationPar(ds=0.05, max_steps=60, newton_tol=1e-6, newton_max_iter=50),
)
n_pa = result_pa.branch.n_valid
final_p = float(result_pa.branch.params[n_pa - 1])
final_u = float(result_pa.branch.states[n_pa - 1, 0])
print(f"\nPseudo-arclength continuation, through the fold and beyond:")
print(f"  reached r = {final_p:.5f}, x = {final_u:.5f} in {n_pa} points")
if final_p > 0.5 and final_u < 0:
    print(f"  PASSED THE FOLD: now on the mirror branch x=-sqrt(r)")

print("\nPseudo-arclength's bordered system stays well-conditioned through the")
print("fold, unlike the natural corrector's fixed-parameter Newton solve above.")
