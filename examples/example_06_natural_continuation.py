"""
Natural-parameter continuation
==============================

Natural-parameter continuation is the simplest predictor-corrector scheme:
predict the next point by stepping the parameter directly, then correct the
state with Newton's method. It is easy to reason about but cannot pass fold
points, where :math:`dx/dp \\to \\infty` (see
:doc:`example_04_pseudo_arclength`, which handles that case).

This example validates the method against three systems with known
closed-form equilibria, checking the computed branch against the exact
solution at every step.
"""

# %%
# Setup

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcont import ContinuationProblem, NaturalContinuation

os.makedirs("images", exist_ok=True)


def run_branch(problem, cont, ds, param_end, max_steps, exact):
    """Step natural continuation manually and track the error against ``exact(r)``."""
    u, param = problem.u0, problem.params[problem.continuation_param]
    params_out, states_out, errors = [param], [u[0]], []

    for step in range(max_steps):
        tangent = cont.compute_tangent(problem, u, param)
        u_pred, param_pred = cont.predict(u, param, tangent, ds)
        u, param, converged, n_iter = cont.correct(problem, u_pred, param_pred, u, param, tangent, ds)
        if not converged:
            print(f"  step {step}: did not converge, stopping")
            break

        params_out.append(param)
        states_out.append(u[0])
        errors.append(abs(u[0] - exact(param)))

        if param >= param_end:
            break

    max_error = max(errors) if errors else 0.0
    print(f"  {len(params_out)} points, max error vs. exact solution: {max_error:.2e}")
    return params_out, states_out


# %%
# Test 1: a linear system, :math:`\dot{x} = r - x`
# -----------------------------------------------------
# Exact solution: :math:`x = r`. This is the easiest possible case and should
# match to machine precision at every step.

def linear_rhs(state, params):
    return jnp.array([params["r"] - state[0]])


problem1 = ContinuationProblem(
    rhs=linear_rhs, u0=jnp.array([0.0]), params={"r": 0.0},
    continuation_param="r", problem_type="equilibrium",
)
cont1 = NaturalContinuation(newton_tol=1e-10, newton_max_iter=100)
print("Test 1: linear system (exact: x = r)")
result1 = run_branch(problem1, cont1, ds=0.1, param_end=1.0, max_steps=15, exact=lambda r: r)

# %%
# Test 2: a fold, :math:`\dot{x} = r - x^2`
# -----------------------------------------------
# Exact solution: :math:`x = \sqrt{r}` for :math:`r > 0`. We start away from
# the fold at :math:`r=0`, so natural continuation has no trouble here.

def quadratic_rhs(state, params):
    return jnp.array([params["r"] - state[0] ** 2])


r0 = 0.1
problem2 = ContinuationProblem(
    rhs=quadratic_rhs, u0=jnp.array([jnp.sqrt(r0)]), params={"r": r0},
    continuation_param="r", problem_type="equilibrium",
)
cont2 = NaturalContinuation(newton_tol=1e-10, newton_max_iter=50)
print("\nTest 2: quadratic system (exact: x = sqrt(r))")
result2 = run_branch(problem2, cont2, ds=0.05, param_end=1.0, max_steps=30, exact=jnp.sqrt)

# %%
# Test 3: a pitchfork, :math:`\dot{x} = r x - x^3`
# -------------------------------------------------------
# Exact solution on the upper branch: :math:`x = \sqrt{r}` for :math:`r > 0`.
# We start on that branch and stay there.

def cubic_rhs(state, params):
    r = params["r"]
    return jnp.array([r * state[0] - state[0] ** 3])


r0 = 0.5
problem3 = ContinuationProblem(
    rhs=cubic_rhs, u0=jnp.array([jnp.sqrt(r0)]), params={"r": r0},
    continuation_param="r", problem_type="equilibrium",
)
cont3 = NaturalContinuation(newton_tol=1e-8, newton_max_iter=100)
print("\nTest 3: pitchfork, upper branch (exact: x = sqrt(r))")
result3 = run_branch(problem3, cont3, ds=0.1, param_end=2.0, max_steps=30, exact=jnp.sqrt)

# %%
# Plot all three branches against their exact solutions
# -----------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
results = [result1, result2, result3]
titles = ["Test 1: Linear (x = r)", "Test 2: Quadratic (x = sqrt(r))", "Test 3: Cubic (x = sqrt(r))"]
exacts = [lambda r: r, lambda r: jnp.sqrt(jnp.maximum(r, 0.0)), lambda r: jnp.sqrt(jnp.maximum(r, 0.0))]

for ax, (rs, xs), title, exact in zip(axes, results, titles, exacts):
    rs = jnp.array(rs)
    ax.plot(rs, xs, "bo-", markersize=6, linewidth=2, label="Computed")
    ax.plot(rs, exact(rs), "r--", linewidth=2, label="Exact")
    ax.set_xlabel("Parameter r")
    ax.set_ylabel("State x")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("images/natural_continuation_tests.png", dpi=150, bbox_inches="tight")
plt.show()
