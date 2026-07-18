"""
Pseudo-arclength continuation, step by step
=============================================

The higher-level ``equilibrium_continuation`` helper hides the
predict/correct loop inside ``PseudoArclengthContinuation.run``. This example
calls ``predict`` and ``correct`` directly, one step at a time, so you can see
exactly what each stage does: computing a tangent, stepping along it, and
correcting back onto the solution curve with a bordered Newton solve.

We use two toy systems where the equilibrium is known in closed form, so
every step can be checked against the exact answer.
"""

# %%
# Setup

import jax.numpy as jnp

from jaxcont import ContinuationProblem, PseudoArclengthContinuation

# %%
# System 1: a linear equation, :math:`\dot{x} = r - x`
# --------------------------------------------------------
# The equilibrium is simply :math:`x = r`, so the exact solution is the
# 45-degree line -- useful for checking that each corrected point lands
# exactly where it should.


def linear_rhs(state, params):
    return jnp.array([params["r"] - state[0]])


problem_linear = ContinuationProblem(
    rhs=linear_rhs, u0=jnp.array([0.0]), params={"r": 0.0},
    continuation_param="r", problem_type="equilibrium",
)
cont = PseudoArclengthContinuation(newton_tol=1e-8, newton_max_iter=100)

# %%
# Step manually along the branch
# ----------------------------------
# Each iteration: compute the tangent at the current point, predict the next
# point by stepping ``ds`` along it, then correct that prediction back onto
# the curve with the bordered Newton solver.

u, param = problem_linear.u0, problem_linear.params["r"]
tangent = cont.compute_tangent(problem_linear, u, param)
print(f"Initial tangent: {tangent}  (norm={jnp.linalg.norm(tangent):.6f})")

ds = 0.1
for step in range(10):
    u_pred, param_pred = cont.predict(u, param, tangent, ds)
    u, param, converged, n_iter = cont.correct(
        problem_linear, u_pred, param_pred, u, param, tangent, ds
    )
    tangent = cont.compute_tangent(problem_linear, u, param, tangent)

    error = abs(u[0] - param)
    print(f"step {step+1:2d}: r={param:.4f}  x={u[0]:.4f}  "
          f"error={error:.2e}  newton_iters={n_iter}  converged={converged}")

    if param > 1.0:
        break

# %%
# System 2: a fold, :math:`\dot{x} = r - x^2`
# -----------------------------------------------
# The equilibrium is :math:`x = \sqrt{r}` for :math:`r > 0` -- a branch that
# folds back at :math:`r = 0`. Pseudo-arclength continuation is exactly the
# method that can walk through such a fold without stalling (natural
# continuation cannot, since :math:`dx/dr \to \infty` there).


def quadratic_rhs(state, params):
    return jnp.array([params["r"] - state[0] ** 2])


r0 = 0.5
problem_quad = ContinuationProblem(
    rhs=quadratic_rhs, u0=jnp.array([jnp.sqrt(r0)]), params={"r": r0},
    continuation_param="r", problem_type="equilibrium",
)

u, param = problem_quad.u0, problem_quad.params["r"]
tangent = cont.compute_tangent(problem_quad, u, param)
print(f"\nInitial tangent: {tangent}")

ds = 0.05
for step in range(5):
    u_pred, param_pred = cont.predict(u, param, tangent, ds)
    u, param, converged, n_iter = cont.correct(
        problem_quad, u_pred, param_pred, u, param, tangent, ds
    )
    tangent = cont.compute_tangent(problem_quad, u, param, tangent)

    expected = jnp.sqrt(param) if param > 0 else 0.0
    error = abs(u[0] - expected)
    print(f"step {step+1:2d}: r={param:.4f}  x={u[0]:.4f}  "
          f"expected={expected:.4f}  error={error:.2e}  newton_iters={n_iter}")
