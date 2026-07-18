"""
Differentiable bifurcation analysis
=======================================

The capability that Julia/MATLAB continuation tools don't offer natively:
because everything is JAX, we can *differentiate* through the analysis and
use gradients for inverse design and sensitivity.

Two demonstrations of what works today:

- **Part A** -- reverse-mode ``jax.grad`` of a **fold location**
  (:func:`jaxcont.fold_parameter`). A fold is characterized by the extended
  system :math:`G(u,p,v;\\theta)=0` and solved with a Newton iteration wrapped
  in ``jax.custom_vjp`` (the implicit function theorem), so the fold
  *parameter* :math:`p^*(\\theta)` is an exact, reverse-mode-differentiable
  function of any design parameters :math:`\\theta`. We verify the gradient
  against the analytic value, then use it for inverse design: choosing
  :math:`\\theta` so the fold sits at a target parameter value.
- **Part B** -- forward-mode ``jax.jacfwd`` sensitivity **through the
  continuation engine**. The whole-loop engine uses ``lax.while_loop``, which
  supports forward-mode but not reverse-mode autodiff, so ``jacfwd`` gives
  the sensitivity of a computed branch to a parameter directly.
"""

# %%
# Part A: differentiating a fold location
# =============================================
# We use the normal form :math:`f(u, p; \theta) = u^2 - \theta u + p`, whose
# fold sits at :math:`u = \theta/2`, :math:`p^*(\theta) = \theta^2/4` -- so
# every result below can be checked against a closed-form answer.

import jax
import jax.numpy as jnp

import jaxcont as jc


def f_fold(u, p, theta):
    return jnp.array([u[0] ** 2 - theta * u[0] + p])


def fold_p(theta):
    """Fold parameter as a differentiable function of theta."""
    return jc.fold_parameter(f_fold, jnp.array([0.4]), jnp.array(0.2), theta)


# %%
# Locate the fold and check the gradient
# --------------------------------------------

theta0 = jnp.array(1.0)
u, p, v = jc.fold_point(f_fold, jnp.array([0.4]), jnp.array(0.2), theta0)
print(f"fold at theta={float(theta0):.2f}: u*={float(u[0]):.4f}  p*={float(p):.4f}")
print(f"  (analytic: u=theta/2={float(theta0)/2:.4f}, p=theta^2/4={float(theta0)**2/4:.4f})")

g = jax.grad(fold_p)(theta0)
print(f"\ndp*/dtheta (jax.grad) = {float(g):.5f}  (analytic theta/2 = {float(theta0)/2:.5f})")

# %%
# Inverse design: place the fold at a target parameter value
# -----------------------------------------------------------------
# Because the gradient is exact, plain gradient descent on
# :math:`(p^*(\theta) - \text{target})^2` converges quickly.

target_p = 1.00  # analytic solution: theta = sqrt(4 * target) = 2
theta = theta0
lr = 0.6
loss_grad = jax.grad(lambda th: (fold_p(th) - target_p) ** 2)

print(f"\ninverse design: place fold at p* -> {target_p}")
for it in range(20):
    theta = theta - lr * loss_grad(theta)
    if it % 4 == 0 or it == 19:
        print(f"  it {it:2d}: theta={float(theta):.4f}  p*={float(fold_p(theta)):.4f}")

print(f"\nfinal theta={float(theta):.4f}  p*={float(fold_p(theta)):.4f}  "
      f"(target {target_p}; analytic theta={(4*target_p)**0.5:.4f})")

# %%
# Part B: forward-mode sensitivity through the engine
# ==========================================================
# ``jacfwd`` differentiates a scalar read off a *computed branch* -- the
# imperfect pitchfork :math:`f(u,p;b) = pu - u^3 + b` -- with respect to the
# imperfection :math:`b`. This works despite the engine's internal
# ``lax.while_loop`` because forward-mode AD (unlike reverse-mode) is
# supported through ``while_loop``.

from jaxcont.core.scan_continuation import pseudo_arclength_scan


def branch_observable(b, slot=8):
    rhs = lambda u, p: jnp.array([p * u[0] - u[0] ** 3 + b])
    res = pseudo_arclength_scan(
        rhs, jnp.array([0.6]), jnp.array(0.5), jnp.array(2.0),
        jnp.array(0.05), jnp.array(1e-5), jnp.array(0.2),
        jnp.array(1e-6), 80, jnp.array(20),
    )
    return res.states[slot, 0]


b0 = jnp.array(0.1)
d_fwd = jax.jacfwd(branch_observable)(b0)

h = 1e-3
fd = (branch_observable(b0 + h) - branch_observable(b0 - h)) / (2 * h)

print(f"d(branch state)/db (jacfwd)      = {float(d_fwd):+.5f}")
print(f"d(branch state)/db (finite diff) = {float(fd):+.5f}  (cross-check)")

# %%
# Reverse-mode ``jax.grad`` does *not* work through the full continuation
# *sweep* (``lax.while_loop`` blocks it) -- that's exactly why Part A
# differentiates the isolated extended-system fold solve instead of a full
# ``continuation()`` run.
