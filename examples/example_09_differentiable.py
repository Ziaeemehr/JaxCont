"""
Example 09 — Differentiable bifurcation analysis (`jax.grad` / `jacfwd`).

The capability that Julia/MATLAB continuation tools don't offer natively
(notes/ARCHITECTURE.md §3.2): because everything is JAX, we can *differentiate*
through the analysis and use gradients for inverse design and sensitivity.

Two honest demonstrations of what works today:

  Part A — Reverse-mode `jax.grad` for INVERSE DESIGN.
    The proper mechanism for differentiating a bifurcation quantity is the
    implicit function theorem: define the equilibrium implicitly by f(u,θ)=0,
    solve it with a (differentiable) Newton iteration, then differentiate a
    property of that equilibrium w.r.t. the design parameter θ. Here we tune θ
    so the equilibrium sits a *target distance from a fold* — gradient descent
    on a stability margin.

  Part B — Forward-mode `jacfwd` sensitivity THROUGH the continuation engine.
    The whole-loop engine uses `lax.while_loop`, which supports forward-mode
    autodiff (but not reverse-mode). So `jacfwd` gives the sensitivity of a
    computed branch to a parameter directly. (Reverse-mode through the full
    sweep needs the implicit-diff / `custom_root` formulation — ROADMAP §3.2.)
"""

import jax
import jax.numpy as jnp

from jaxcont.core.scan_continuation import pseudo_arclength_scan


# =========================================================================
# Part A — reverse-mode grad + inverse design on a differentiable equilibrium
# =========================================================================
#
# Normal form with a saddle-node (fold): f(u, θ) = θ - u + u^3/3.
#   equilibria:  θ = u - u^3/3
#   Jacobian:    f_u = -1 + u^2   (stable, <0, for |u|<1; fold at u=±1, θ=±2/3)
#   stability margin:  m(θ) = -f_u(u*(θ)) = 1 - u*(θ)^2   (0 at the fold)

def f_scalar(u, theta):
    return theta - u + u**3 / 3.0


def newton_equilibrium(theta, u_init=0.0, n_steps=25):
    """Differentiable equilibrium solve on the stable inner branch (|u|<1)."""
    fu = jax.grad(f_scalar, argnums=0)

    def step(u, _):
        u = u - f_scalar(u, theta) / fu(u, theta)
        return u, None

    u_star, _ = jax.lax.scan(step, u_init, None, length=n_steps)
    return u_star


def stability_margin(theta):
    """m(θ) = 1 - u*(θ)^2 — distance of the equilibrium from the fold."""
    u_star = newton_equilibrium(theta)
    fu = jax.grad(f_scalar, argnums=0)(u_star, theta)
    return -fu  # = 1 - u*^2


def part_a():
    print("=" * 72)
    print("Part A — reverse-mode jax.grad + inverse design")
    print("=" * 72)

    theta0 = jnp.array(0.2)
    m = stability_margin(theta0)
    # reverse-mode gradient through the (unrolled Newton) equilibrium solve
    dm = jax.grad(stability_margin)(theta0)
    # finite-difference cross-check
    h = 1e-4
    fd = (stability_margin(theta0 + h) - stability_margin(theta0 - h)) / (2 * h)
    print(f"\n  at θ={float(theta0):.3f}:  margin m={float(m):.4f}")
    print(f"  dm/dθ  (jax.grad)      = {float(dm):+.4f}")
    print(f"  dm/dθ  (finite diff)   = {float(fd):+.4f}   [cross-check]")

    # Inverse design: gradient-descent θ so the margin hits a target (move the
    # stable equilibrium closer to the fold, but not onto it). We take a
    # *projected* step, clipping θ safely below the fold (θ_fold = 2/3) so the
    # inner equilibrium — and thus the differentiable solve — stays well defined.
    target = 0.60
    theta = theta0
    lr = 0.15
    theta_max = 0.62  # < θ_fold = 2/3
    loss_grad = jax.grad(lambda th: (stability_margin(th) - target) ** 2)
    print(f"\n  inverse design: drive margin -> {target}  (projected, θ<{theta_max})")
    for it in range(30):
        loss = (stability_margin(theta) - target) ** 2
        theta = jnp.clip(theta - lr * loss_grad(theta), 0.0, theta_max)
        if it % 6 == 0 or it == 29:
            print(f"    it {it:2d}:  θ={float(theta):+.4f}  "
                  f"margin={float(stability_margin(theta)):.4f}  loss={float(loss):.2e}")
    print(f"\n  final θ={float(theta):+.4f}  margin={float(stability_margin(theta)):.4f} "
          f"(target {target})")


# =========================================================================
# Part B — forward-mode jacfwd through the continuation engine
# =========================================================================

def branch_observable(b, slot=8):
    """A smooth scalar read off a computed branch of the imperfect pitchfork."""
    rhs = lambda u, p: jnp.array([p * u[0] - u[0] ** 3 + b])
    res = pseudo_arclength_scan(
        rhs, jnp.array([0.6]), jnp.array(0.5), jnp.array(2.0),
        jnp.array(0.05), jnp.array(1e-5), jnp.array(0.2),
        jnp.array(1e-6), 80, jnp.array(20),
    )
    return res.states[slot, 0]


def part_b():
    print("\n" + "=" * 72)
    print("Part B — forward-mode jacfwd through the whole-loop engine")
    print("=" * 72)

    b0 = jnp.array(0.1)
    d_fwd = jax.jacfwd(branch_observable)(b0)   # works with lax.while_loop
    h = 1e-3
    fd = (branch_observable(b0 + h) - branch_observable(b0 - h)) / (2 * h)
    print(f"\n  d(branch state)/db  (jacfwd)      = {float(d_fwd):+.5f}")
    print(f"  d(branch state)/db  (finite diff) = {float(fd):+.5f}   [cross-check]")
    print("\n  (reverse-mode grad through the full sweep needs implicit diff —")
    print("   see ARCHITECTURE.md §3.2; forward-mode is available today.)")


def main():
    part_a()
    part_b()


if __name__ == "__main__":
    main()
